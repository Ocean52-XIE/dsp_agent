# -*- coding: utf-8 -*-
"""
该模块实现评测流程，负责样本执行、指标统计与结果输出。
"""
from __future__ import annotations

"""Issue-analysis routing evaluation script.

Evaluates whether queries expected to trigger issue-analysis flow are routed to:
- route: issue_analysis
- kind: issue_analysis
- status: completed
- module: expected_module (optional)

Additionally evaluates issue-analysis answer structure and LLM usage quality gates:
- generation_mode coverage (llm/fallback/error)
- root_cause / risks / fix_plan / verification completeness
- evidence anchor mention hit rate
- optional LLM-required enforcement
"""

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from workflow.engine import WorkflowService  # noqa: E402


@dataclass
class IssueEvalCase:
    """
    定义`IssueEvalCase`，用于封装相关数据结构与处理行为。
    """
    case_id: str
    query: str
    expected_route: str
    expected_kind: str
    expected_status: str
    expected_module: str
    intent: str
    note: str


def _to_bool(value: Any, default: bool = False) -> bool:
    """
    将输入值转换为布尔值。

    说明：
    兼容 bool/int/字符串等常见配置输入，避免 JSON 配置中出现类型差异导致逻辑偏移。
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if not text:
        return default
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _to_int(value: Any, default: int) -> int:
    """
    将输入值转换为整数，失败时返回默认值。
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float) -> float:
    """
    将输入值转换为浮点数，失败时返回默认值。
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_text_list(value: Any) -> list[str]:
    """
    将输入归一化为非空字符串列表。
    """
    if not isinstance(value, list):
        return []
    rows: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text:
            rows.append(text)
    return rows


def _resolve_generation_mode(analysis: dict[str, Any]) -> str:
    """
    统一读取 generation_mode（兼容旧字段）。
    """
    mode = str(analysis.get("generation_mode", "") or "").strip().lower()
    if mode:
        return mode
    legacy_mode = str(analysis.get("issue_analysis_generation_mode", "") or "").strip().lower()
    return legacy_mode


def _resolve_llm_fallback_reason(analysis: dict[str, Any]) -> str:
    """
    统一读取 llm_fallback_reason（兼容旧字段）。
    """
    reason = str(analysis.get("llm_fallback_reason", "") or "").strip()
    if reason:
        return reason
    legacy_reason = str(analysis.get("issue_analysis_llm_fallback_reason", "") or "").strip()
    return legacy_reason


def _collect_anchor_tokens(citations: list[dict[str, Any]]) -> list[str]:
    """
    从引用证据中提取可追踪锚点 token。

    说明：
    用于评估“回答是否引用了具体证据锚点”，而不仅是泛化描述。
    """
    tokens: list[str] = []
    for item in citations:
        path = str(item.get("path", "")).strip().lower()
        title = str(item.get("title", "")).strip().lower()
        section = str(item.get("section", "")).strip().lower()
        symbol_name = str(item.get("symbol_name", "")).strip().lower()
        if path:
            tokens.append(path)
            tokens.append(path.split("/")[-1])
            tokens.append(path.split("\\")[-1])
        if title:
            tokens.append(title)
        if section:
            tokens.append(section)
        if symbol_name:
            tokens.append(symbol_name)

    deduped: list[str] = []
    seen: set[str] = set()
    for raw in tokens:
        normalized = str(raw or "").strip().lower()
        if len(normalized) < 3:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _answer_mentions_anchor(answer: str, anchor_tokens: list[str]) -> bool:
    """
    判断回答正文是否提及任一证据锚点。
    """
    normalized_answer = str(answer or "").strip().lower()
    if not normalized_answer:
        return False
    return any(token in normalized_answer for token in anchor_tokens if token)


def _count_numbered_lines(text: str) -> int:
    """
    统计“1. xxx / 2. xxx”样式的编号行数量。
    """
    pattern = re.compile(r"(?m)^\s*\d+\.\s+")
    return len(pattern.findall(str(text or "")))


def _compute_backoff_seconds(*, base_delay_ms: int, retry_index: int) -> float:
    """
    计算指数退避时长（秒）。

    说明：
    评测重试用于吸收短时网络抖动，避免把临时调用失败误判为能力退化。
    """
    delay_ms = max(base_delay_ms, 0) * (2 ** max(retry_index - 1, 0))
    return delay_ms / 1000.0


def _is_retryable_issue_fallback(*, generation_mode: str, fallback_reason: str) -> bool:
    """
    判断 issue_analysis 的 fallback 是否属于可重试类型。
    """
    if generation_mode != "fallback_rule":
        return False
    retryable_prefixes = ("timeout", "http_error", "unknown_error", "empty_answer")
    normalized_reason = str(fallback_reason or "").strip().lower()
    return any(normalized_reason.startswith(prefix) for prefix in retryable_prefixes)


def _should_retry_issue_response(
    *,
    response: dict[str, Any],
    llm_enabled: bool,
    llm_available: bool,
    require_llm_mode: bool,
) -> bool:
    """
    判断当前响应是否值得重试。

    规则：
    1. 若 LLM 不可用，重试通常无意义，直接返回 False；
    2. 若配置要求必须走 LLM，则所有非 llm 模式都允许重试；
    3. 否则仅在“可重试 fallback 原因”场景重试。
    """
    if not (llm_enabled and llm_available):
        return False

    analysis = dict(response.get("analysis") or {})
    generation_mode = _resolve_generation_mode(analysis)
    if require_llm_mode and generation_mode != "llm":
        return True

    fallback_reason = _resolve_llm_fallback_reason(analysis)
    return _is_retryable_issue_fallback(
        generation_mode=generation_mode,
        fallback_reason=fallback_reason,
    )


def _apply_eval_env_from_config(config: dict[str, Any]) -> None:
    """
    从评测配置注入必要环境变量。

    说明：
    仅注入“domain / eval retry”这类非敏感控制项；
    LLM 的 API Key 等敏感变量仍建议通过外部环境传入。
    """
    domain_profile = str(config.get("domain_profile", "") or "").strip()
    domain_dir = str(config.get("domain_dir", "") or "").strip()
    code_retriever_dirs = str(config.get("code_retriever_dirs", "") or "").strip()
    if domain_profile:
        os.environ["WORKFLOW_DOMAIN_PROFILE"] = domain_profile
    if domain_dir:
        os.environ["WORKFLOW_DOMAIN_DIR"] = domain_dir
    if code_retriever_dirs:
        os.environ["WORKFLOW_CODE_RETRIEVER_DIRS"] = code_retriever_dirs


def _load_json(path: Path) -> dict[str, Any]:
    """
    内部辅助函数，负责`load json` 相关处理。
    
    参数:
        path: 文件或目录路径。
    
    返回:
        返回类型为 `dict[str, Any]` 的处理结果。
    """
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _display_path(path: Path) -> str:
    """
    内部辅助函数，负责`display path` 相关处理。
    
    参数:
        path: 文件或目录路径。
    
    返回:
        返回类型为 `str` 的处理结果。
    """
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _load_dataset(path: Path) -> list[IssueEvalCase]:
    """
    内部辅助函数，负责`load dataset` 相关处理。
    
    参数:
        path: 文件或目录路径。
    
    返回:
        返回类型为 `list[IssueEvalCase]` 的处理结果。
    """
    items: list[IssueEvalCase] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            query = str(record.get("query", "")).strip()
            if not query:
                raise ValueError(f"dataset line {line_number} missing query")

            items.append(
                IssueEvalCase(
                    case_id=str(record.get("id", f"issue_case_{line_number:04d}")),
                    query=query,
                    expected_route=str(record.get("expected_route", "issue_analysis")).strip() or "issue_analysis",
                    expected_kind=str(record.get("expected_kind", "issue_analysis")).strip() or "issue_analysis",
                    expected_status=str(record.get("expected_status", "completed")).strip() or "completed",
                    expected_module=str(record.get("expected_module", "")).strip(),
                    intent=str(record.get("intent", "issue_analysis")).strip() or "issue_analysis",
                    note=str(record.get("note", "")).strip(),
                )
            )
    return items


def run_eval(config_path: Path) -> dict[str, Any]:
    """
    执行对应子流程并返回执行结果。
    
    参数:
        config_path: 路径参数，用于定位文件或目录。
    
    返回:
        返回类型为 `dict[str, Any]` 的处理结果。
    """
    config = _load_json(config_path)
    _apply_eval_env_from_config(config)

    dataset_path = (PROJECT_ROOT / config["dataset_path"]).resolve()
    output_path = (PROJECT_ROOT / config["output_path"]).resolve()
    save_per_case = bool(config.get("save_per_case", True))
    session_id_prefix = str(config.get("session_id_prefix", "issue_eval_sess"))
    trace_id_prefix = str(config.get("trace_id_prefix", "issue_eval_trace"))

    # LLM 评测策略：
    # 1. require_llm_available: 启动前要求 LLM 客户端可用，否则直接失败；
    # 2. require_llm_mode: 每条样本要求 generation_mode=llm；
    # 3. llm_mode_min_rate: 允许设置“至少多少比例命中 llm 模式”。
    require_llm_available = _to_bool(config.get("require_llm_available"), False)
    require_llm_mode = _to_bool(config.get("require_llm_mode"), False)
    fail_on_llm_mode_min_rate = _to_bool(config.get("fail_on_llm_mode_min_rate"), False)
    llm_mode_min_rate = max(0.0, min(1.0, _to_float(config.get("llm_mode_min_rate"), 1.0 if require_llm_mode else 0.0)))

    retry_count = max(
        0,
        _to_int(
            config.get("retry_count", os.getenv("WORKFLOW_EVAL_RETRY_COUNT", 1)),
            1,
        ),
    )
    retry_base_delay_ms = max(
        0,
        _to_int(
            config.get("retry_base_delay_ms", os.getenv("WORKFLOW_EVAL_RETRY_BASE_DELAY_MS", 300)),
            300,
        ),
    )

    cases = _load_dataset(dataset_path)
    if not cases:
        raise ValueError(f"issue-analysis dataset is empty: {dataset_path}")
    service = WorkflowService()
    llm_client = getattr(service, "_llm_client", None)
    issue_llm_enabled = bool(llm_client is not None and llm_client.config.enabled)
    issue_llm_available = bool(llm_client is not None and llm_client.is_available)
    issue_llm_model = llm_client.config.model if llm_client is not None else ""

    if require_llm_available and not issue_llm_available:
        raise RuntimeError(
            "Issue-analysis eval requires an available LLM client, "
            "but current WORKFLOW_QA_LLM config is unavailable. "
            "Please set WORKFLOW_QA_LLM_ENABLED=true and valid WORKFLOW_QA_LLM_API_KEY/MODEL."
        )

    route_hit_sum = 0.0
    kind_hit_sum = 0.0
    status_hit_sum = 0.0
    module_hit_sum = 0.0
    module_eval_count = 0
    strict_pass_sum = 0.0
    root_cause_presence_sum = 0.0
    risks_coverage_sum = 0.0
    fix_plan_coverage_sum = 0.0
    verification_coverage_sum = 0.0
    structure_pass_sum = 0.0
    anchor_hit_sum = 0.0
    llm_mode_sum = 0.0
    fallback_mode_sum = 0.0
    valid_generation_mode_sum = 0.0
    need_user_confirmation_sum = 0.0
    numbered_answer_sum = 0.0
    runtime_error_sum = 0.0
    llm_required_pass_sum = 0.0
    attempts_sum = 0.0
    per_case_results: list[dict[str, Any]] = []

    for case_index, case in enumerate(cases, start=1):
        response: dict[str, Any] = {}
        response_attempts = 0
        last_response_error: str | None = None

        # 评测重试策略：
        # 1. 捕获运行时异常（网络抖动等）时重试；
        # 2. 在“应走 LLM 但落到 fallback”场景下按配置重试。
        for retry_index in range(0, retry_count + 1):
            response_attempts = retry_index + 1
            try:
                response = service.run_user_message(
                    session_id=f"{session_id_prefix}_{case.case_id}",
                    trace_id=f"{trace_id_prefix}_{case_index:04d}_try{retry_index + 1}",
                    user_query=case.query,
                    history=[],
                )
            except Exception as exc:  # noqa: BLE001
                last_response_error = f"issue_eval_runtime_error:{type(exc).__name__}"
                if retry_index >= retry_count:
                    response = {
                        "status": "error",
                        "kind": case.intent,
                        "intent": case.intent,
                        "content": "当前证据不足。",
                        "analysis": {
                            "generation_mode": "error",
                            "llm_fallback_reason": last_response_error,
                            "error": str(exc),
                        },
                        "citations": [],
                    }
                    break

                time.sleep(
                    _compute_backoff_seconds(
                        base_delay_ms=retry_base_delay_ms,
                        retry_index=retry_index + 1,
                    )
                )
                continue

            if retry_index < retry_count and _should_retry_issue_response(
                response=response,
                llm_enabled=issue_llm_enabled,
                llm_available=issue_llm_available,
                require_llm_mode=require_llm_mode,
            ):
                time.sleep(
                    _compute_backoff_seconds(
                        base_delay_ms=retry_base_delay_ms,
                        retry_index=retry_index + 1,
                    )
                )
                continue
            break

        actual_route = str(response.get("intent", "")).strip()
        actual_kind = str(response.get("kind", "")).strip()
        actual_status = str(response.get("status", "")).strip()
        analysis = dict(response.get("analysis") or {})
        answer = str(response.get("content", "") or "")
        citations = list(response.get("citations", []) or [])
        actual_module = str(analysis.get("module", "")).strip()

        route_hit = 1.0 if actual_route == case.expected_route else 0.0
        kind_hit = 1.0 if actual_kind == case.expected_kind else 0.0
        status_hit = 1.0 if actual_status == case.expected_status else 0.0
        route_hit_sum += route_hit
        kind_hit_sum += kind_hit
        status_hit_sum += status_hit

        root_cause = str(analysis.get("root_cause", "") or "").strip()
        risks = _as_text_list(analysis.get("risks"))
        fix_plan = _as_text_list(analysis.get("fix_plan"))
        verification_steps = _as_text_list(analysis.get("verification_steps"))
        generation_mode = _resolve_generation_mode(analysis)
        llm_fallback_reason = _resolve_llm_fallback_reason(analysis)
        need_user_confirmation = bool(analysis.get("need_user_confirmation", False))

        root_cause_present = 1.0 if bool(root_cause) else 0.0
        risks_covered = 1.0 if len(risks) >= 2 else 0.0
        fix_plan_covered = 1.0 if len(fix_plan) >= 2 else 0.0
        verification_covered = 1.0 if len(verification_steps) >= 2 else 0.0
        structure_pass = 1.0 if (root_cause_present and risks_covered and fix_plan_covered and verification_covered) else 0.0

        anchor_tokens = _collect_anchor_tokens(citations)
        anchor_hit = 1.0 if (anchor_tokens and _answer_mentions_anchor(answer, anchor_tokens)) else 0.0
        is_llm_mode = 1.0 if generation_mode == "llm" else 0.0
        is_fallback_mode = 1.0 if generation_mode == "fallback_rule" else 0.0
        has_valid_generation_mode = 1.0 if generation_mode in {"llm", "fallback_rule"} else 0.0
        has_numbered_answer = 1.0 if _count_numbered_lines(answer) >= 2 else 0.0
        runtime_error_hit = 1.0 if generation_mode == "error" else 0.0
        llm_required_pass = 1.0 if (generation_mode == "llm" or not require_llm_mode) else 0.0

        root_cause_presence_sum += root_cause_present
        risks_coverage_sum += risks_covered
        fix_plan_coverage_sum += fix_plan_covered
        verification_coverage_sum += verification_covered
        structure_pass_sum += structure_pass
        anchor_hit_sum += anchor_hit
        llm_mode_sum += is_llm_mode
        fallback_mode_sum += is_fallback_mode
        valid_generation_mode_sum += has_valid_generation_mode
        need_user_confirmation_sum += 1.0 if need_user_confirmation else 0.0
        numbered_answer_sum += has_numbered_answer
        runtime_error_sum += runtime_error_hit
        llm_required_pass_sum += llm_required_pass
        attempts_sum += float(response_attempts)

        module_hit: float | None = None
        if case.expected_module:
            module_eval_count += 1
            module_hit = 1.0 if actual_module == case.expected_module else 0.0
            module_hit_sum += module_hit

        strict_pass = route_hit == 1.0 and kind_hit == 1.0 and status_hit == 1.0
        if module_hit is not None:
            strict_pass = strict_pass and module_hit == 1.0
        if require_llm_mode:
            strict_pass = strict_pass and generation_mode == "llm"
        strict_pass_sum += 1.0 if strict_pass else 0.0

        if save_per_case:
            per_case_results.append(
                {
                    "id": case.case_id,
                    "query": case.query,
                    "intent": case.intent,
                    "note": case.note,
                    "expected": {
                        "route": case.expected_route,
                        "kind": case.expected_kind,
                        "status": case.expected_status,
                        "module": case.expected_module,
                    },
                    "actual": {
                        "route": actual_route,
                        "kind": actual_kind,
                        "status": actual_status,
                        "module": actual_module,
                    },
                    "hits": {
                        "route_hit": route_hit,
                        "kind_hit": kind_hit,
                        "status_hit": status_hit,
                        "module_hit": module_hit,
                    },
                    "quality": {
                        "root_cause_present": bool(root_cause_present),
                        "risk_count": len(risks),
                        "fix_plan_count": len(fix_plan),
                        "verification_step_count": len(verification_steps),
                        "structure_pass": bool(structure_pass),
                        "anchor_hit": bool(anchor_hit),
                        "generation_mode": generation_mode or "missing",
                        "llm_fallback_reason": llm_fallback_reason,
                        "need_user_confirmation": need_user_confirmation,
                        "citation_count": len(citations),
                        "numbered_line_count": _count_numbered_lines(answer),
                        "attempts": response_attempts,
                        "llm_required_pass": bool(llm_required_pass),
                        "runtime_error": bool(runtime_error_hit),
                        "last_response_error": last_response_error,
                    },
                    "strict_pass": strict_pass,
                }
            )

    total_cases = len(cases)
    metrics: dict[str, Any] = {
        "route_accuracy": round(route_hit_sum / total_cases, 6),
        "kind_accuracy": round(kind_hit_sum / total_cases, 6),
        "status_accuracy": round(status_hit_sum / total_cases, 6),
        "strict_pass_rate": round(strict_pass_sum / total_cases, 6),
        "root_cause_presence_rate": round(root_cause_presence_sum / total_cases, 6),
        "risk_coverage_rate": round(risks_coverage_sum / total_cases, 6),
        "fix_plan_coverage_rate": round(fix_plan_coverage_sum / total_cases, 6),
        "verification_coverage_rate": round(verification_coverage_sum / total_cases, 6),
        "analysis_structure_pass_rate": round(structure_pass_sum / total_cases, 6),
        "evidence_anchor_hit_rate": round(anchor_hit_sum / total_cases, 6),
        "llm_mode_rate": round(llm_mode_sum / total_cases, 6),
        "fallback_mode_rate": round(fallback_mode_sum / total_cases, 6),
        "valid_generation_mode_rate": round(valid_generation_mode_sum / total_cases, 6),
        "need_user_confirmation_rate": round(need_user_confirmation_sum / total_cases, 6),
        "numbered_answer_rate": round(numbered_answer_sum / total_cases, 6),
        "runtime_error_rate": round(runtime_error_sum / total_cases, 6),
        "llm_required_pass_rate": round(llm_required_pass_sum / total_cases, 6),
        "avg_attempts": round(attempts_sum / total_cases, 6),
        "llm_mode_min_rate": round(llm_mode_min_rate, 6),
    }
    if module_eval_count > 0:
        metrics["module_accuracy"] = round(module_hit_sum / module_eval_count, 6)

    llm_mode_threshold_passed = float(metrics["llm_mode_rate"]) >= float(metrics["llm_mode_min_rate"])
    metrics["llm_mode_threshold_passed"] = bool(llm_mode_threshold_passed)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": _display_path(config_path),
        "dataset_path": _display_path(dataset_path),
        "output_path": _display_path(output_path),
        "total_cases": total_cases,
        "module_eval_count": module_eval_count,
        "metrics": metrics,
        "llm_eval_context": {
            "enabled": issue_llm_enabled,
            "available": issue_llm_available,
            "model": issue_llm_model,
            "require_llm_available": require_llm_available,
            "require_llm_mode": require_llm_mode,
            "retry_count": retry_count,
            "retry_base_delay_ms": retry_base_delay_ms,
            "fail_on_llm_mode_min_rate": fail_on_llm_mode_min_rate,
        },
        "quality_gate": {
            "llm_mode_min_rate": llm_mode_min_rate,
            "llm_mode_rate": float(metrics["llm_mode_rate"]),
            "passed": bool(llm_mode_threshold_passed),
            "enforced": bool(fail_on_llm_mode_min_rate),
            "message": (
                ""
                if llm_mode_threshold_passed
                else (
                    f"llm_mode_rate({metrics['llm_mode_rate']}) "
                    f"< llm_mode_min_rate({metrics['llm_mode_min_rate']})"
                )
            ),
        },
        "per_case": per_case_results if save_per_case else [],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    if fail_on_llm_mode_min_rate and not llm_mode_threshold_passed:
        raise RuntimeError(report["quality_gate"]["message"])

    return report


def main() -> int:
    """
    执行`main` 相关处理逻辑。
    
    返回:
        返回类型为 `int` 的处理结果。
    """
    parser = argparse.ArgumentParser(description="Run issue-analysis routing evaluation.")
    parser.add_argument(
        "--config",
        default="src/workflow/eval/config.issue_analysis.template.json",
        help="Path to issue-analysis eval config JSON (relative to project root or absolute).",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")

    report = run_eval(config_path)
    metrics = report["metrics"]
    llm_context = report.get("llm_eval_context", {})
    quality_gate = report.get("quality_gate", {})
    print("==== Issue Analysis Eval Summary ====")
    print(f"total_cases: {report['total_cases']}")
    print(f"route_accuracy: {metrics['route_accuracy']}")
    print(f"kind_accuracy: {metrics['kind_accuracy']}")
    print(f"status_accuracy: {metrics['status_accuracy']}")
    if "module_accuracy" in metrics:
        print(f"module_accuracy: {metrics['module_accuracy']}")
    print(f"root_cause_presence_rate: {metrics['root_cause_presence_rate']}")
    print(f"risk_coverage_rate: {metrics['risk_coverage_rate']}")
    print(f"fix_plan_coverage_rate: {metrics['fix_plan_coverage_rate']}")
    print(f"verification_coverage_rate: {metrics['verification_coverage_rate']}")
    print(f"analysis_structure_pass_rate: {metrics['analysis_structure_pass_rate']}")
    print(f"evidence_anchor_hit_rate: {metrics['evidence_anchor_hit_rate']}")
    print(f"llm_mode_rate: {metrics['llm_mode_rate']}")
    print(f"fallback_mode_rate: {metrics['fallback_mode_rate']}")
    print(f"valid_generation_mode_rate: {metrics['valid_generation_mode_rate']}")
    print(f"llm_required_pass_rate: {metrics['llm_required_pass_rate']}")
    print(f"runtime_error_rate: {metrics['runtime_error_rate']}")
    print(f"avg_attempts: {metrics['avg_attempts']}")
    print(f"need_user_confirmation_rate: {metrics['need_user_confirmation_rate']}")
    print(f"numbered_answer_rate: {metrics['numbered_answer_rate']}")
    print(f"llm_mode_min_rate: {metrics['llm_mode_min_rate']}")
    print(f"llm_mode_threshold_passed: {metrics['llm_mode_threshold_passed']}")
    print(f"strict_pass_rate: {metrics['strict_pass_rate']}")
    print(
        "llm_context:",
        {
            "enabled": llm_context.get("enabled", False),
            "available": llm_context.get("available", False),
            "model": llm_context.get("model", ""),
            "require_llm_available": llm_context.get("require_llm_available", False),
            "require_llm_mode": llm_context.get("require_llm_mode", False),
            "retry_count": llm_context.get("retry_count", 0),
        },
    )
    if quality_gate and not quality_gate.get("passed", True):
        print("quality_gate_message:", quality_gate.get("message", "llm quality gate failed"))
    print("report_path:", report["output_path"])
    print("=====================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
