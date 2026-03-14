# -*- coding: utf-8 -*-
"""
问题分析节点（issue_analysis）。

设计目标：
1. 对齐 knowledge_answer 的“LLM 优先 + 规则兜底”执行模式；
2. 兜底输出必须尽量由证据驱动，避免固定模板导致结果空泛；
3. 统一输出观测字段，兼容前端与 observability 对 generation_mode 的读取；
4. 在证据不足场景显式给出 need_user_confirmation，便于后续追问与闭环。
"""
from __future__ import annotations

import os
import re
from typing import Any

from workflow.common.evidence import collect_evidence_hits
from workflow.common.func_utils import normalize_source_type
from workflow.llm.llm_client import CommonLLMRequest
from workflow.llm.llm_prompt_utils import build_evidence_block, looks_like_reasoning_dump
from workflow.common.node_trace import append_node_trace


ISSUE_SYSTEM_PROMPT_TEMPLATE = (
    "你是企业问题分析助手。"
    "必须严格基于提供的证据回答，不补充证据外事实。"
    "输出中文，优先给出结构化结果，并保留可追踪锚点。"
)

ISSUE_USER_PROMPT_TEMPLATE = """【待分析问题】
{user_query}

【当前模块】
- module_name: {module_name}
- module_hint: {module_hint}

【相关模块】
{related_modules_block}

【检索语句】
{retrieval_queries}

【症状结构化】
{issue_context_block}

【检索证据（按相关性排序）】
{evidence_block}

请严格按以下结构输出：
根因判断：一句话
风险提示：2-3条
修复建议：2-4条
验证步骤：2-3条
"""

_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*]\s+|\d+[.)、]\s+)(.+)$")
_ID_CAPTURE_RE = re.compile(
    r"\b(?:request_id|trace_id|campaign_id|job_id|task_id|order_id)\s*[:=]\s*([A-Za-z0-9_\-]+)\b",
    re.IGNORECASE,
)
_TIME_CAPTURE_RE = re.compile(
    r"(?:\b\d{1,2}:\d{2}\b|\b\d{4}-\d{1,2}-\d{1,2}\b|\b近\d+\s*(?:分钟|小时|天)\b)",
    re.IGNORECASE,
)
_ERROR_SIGNAL_RE = re.compile(
    r"(?:报错|错误|异常|失败|超时|timeout|exception|traceback|告警|报警|\b[45]\d{2}\b)",
    re.IGNORECASE,
)
_SECTION_HINTS = {
    "root_cause": ("根因判断", "根因", "结论"),
    "risks": ("风险提示", "风险"),
    "fix_plan": ("修复建议", "修复方案", "处理建议", "建议"),
    "verification_steps": ("验证步骤", "验证", "回归验证"),
}


def _strip_item(text: str) -> str:
    """
    清洗候选列表项文本。

    说明：
    1. 去除 markdown 列表前缀，提升“结构化解析”稳定性；
    2. 去除标题类残留符号，避免把“### 风险”误识别为正文条目。
    """
    value = str(text or "").strip()
    if not value:
        return ""
    match = _LIST_ITEM_RE.match(value)
    if match:
        value = match.group(1).strip()
    if value.startswith("#"):
        return ""
    return value.strip(" \t-*：:")


def _dedup(items: list[str], max_items: int) -> list[str]:
    """
    去重并限制条目数。

    说明：
    1. 去重可以避免 LLM 输出“同义重复句”挤占有效条目；
    2. max_items 保证最终响应长度可控，避免前端展示过长。
    """
    seen: set[str] = set()
    out: list[str] = []
    for raw in items:
        value = _strip_item(raw)
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
        if len(out) >= max_items:
            break
    return out


def _match_heading(line: str) -> tuple[str | None, str]:
    """
    判断一行文本是否命中结构化标题。

    返回：
    - section: 命中的分节名；
    - payload: 标题后内联的正文（若存在）。
    """
    normalized = _strip_item(line)
    if not normalized:
        return None, ""
    for section, labels in _SECTION_HINTS.items():
        for label in labels:
            if normalized.startswith(label):
                payload = normalized[len(label) :].lstrip("：:").strip()
                return section, payload
    return None, normalized


def _parse_llm_answer(answer: str) -> dict[str, Any]:
    """
    解析 LLM 文本为结构化分析字段。

    说明：
    1. 优先按标题分段解析；
    2. 若模型未严格遵守格式，则从 fallback 句子里做降级提取；
    3. 只返回已提取到的字段，缺失字段由规则兜底补齐。
    """
    root_cause = ""
    risks: list[str] = []
    fix_plan: list[str] = []
    verification_steps: list[str] = []
    fallback: list[str] = []

    current: str | None = None
    for raw in str(answer or "").splitlines():
        line = raw.strip()
        if not line:
            continue

        section, payload = _match_heading(line)
        if section is not None:
            current = section
            if payload:
                if section == "root_cause" and not root_cause:
                    root_cause = payload
                elif section == "risks":
                    risks.append(payload)
                elif section == "fix_plan":
                    fix_plan.append(payload)
                elif section == "verification_steps":
                    verification_steps.append(payload)
            continue

        item = _strip_item(line)
        if not item:
            continue
        if current == "root_cause":
            if not root_cause:
                root_cause = item
            else:
                fallback.append(item)
        elif current == "risks":
            risks.append(item)
        elif current == "fix_plan":
            fix_plan.append(item)
        elif current == "verification_steps":
            verification_steps.append(item)
        else:
            fallback.append(item)

    fallback = _dedup(fallback, 8)
    risks = _dedup(risks, 3)
    fix_plan = _dedup(fix_plan, 4)
    verification_steps = _dedup(verification_steps, 3)

    if not root_cause and fallback:
        root_cause = fallback[0]
    if not risks and len(fallback) > 1:
        risks = fallback[1:3]
    if not fix_plan and len(fallback) > 1:
        fix_plan = fallback[1:4]
    if not verification_steps and len(fallback) > 2:
        verification_steps = fallback[-2:]

    parsed: dict[str, Any] = {}
    if root_cause:
        parsed["root_cause"] = root_cause
    if risks:
        parsed["risks"] = risks
    if fix_plan:
        parsed["fix_plan"] = fix_plan
    if verification_steps:
        parsed["verification_steps"] = verification_steps
    return parsed


def _validate_issue_answer(answer: str) -> tuple[bool, str | None]:
    """
    统一校验 LLM 输出是否可用。

    规则：
    1. 空文本直接视为失败；
    2. 若疑似推理链泄漏文本，也视为失败并触发规则兜底。
    """
    if not answer.strip():
        return False, "empty_answer"
    if looks_like_reasoning_dump(answer):
        return False, "empty_answer:reasoning_dump"
    return True, None


def _default_llm_call_status() -> dict[str, Any]:
    """
    构造 llm_call_status 默认结构。

    说明：
    与 knowledge_answer 结构保持一致，便于前端和观测统一读取。
    """
    return {
        "status": "not_configured",
        "invoked": False,
        "request_sent": False,
        "attempts": 0,
        "latency_ms": 0,
        "reason": None,
        "model": None,
    }


def _resolve_issue_system_prompt(service: Any) -> str:
    """
    解析 issue_analysis 使用的系统提示词。

    优先级：
    1. 环境变量 `WORKFLOW_ISSUE_LLM_SYSTEM_PROMPT`；
    2. 领域配置 `prompts.issue_system`；
    3. 内置默认 ISSUE_SYSTEM_PROMPT_TEMPLATE。
    """
    env_prompt = os.getenv("WORKFLOW_ISSUE_LLM_SYSTEM_PROMPT", "").strip()
    if env_prompt:
        return env_prompt

    domain_profile = getattr(service, "domain_profile", None)
    if domain_profile is not None and hasattr(domain_profile, "issue_system_prompt"):
        profile_prompt = str(domain_profile.issue_system_prompt() or "").strip()
        if profile_prompt:
            return profile_prompt
    return ISSUE_SYSTEM_PROMPT_TEMPLATE


def _build_related_modules_block(related_modules: list[dict[str, Any]]) -> str:
    """
    构建相关模块提示词块。

    说明：
    保持“主模块 + 相关模块”上下文，不强制模型只看单模块，
    便于处理跨模块联动故障。
    """
    if not related_modules:
        return "- 无"
    rows: list[str] = []
    for item in related_modules[:3]:
        module_name = str(item.get("module_name", "")).strip()
        module_hint = str(item.get("module_hint", "")).strip() or "--"
        if not module_name:
            continue
        rows.append(f"- module_name: {module_name}")
        rows.append(f"  module_hint: {module_hint}")
    return "\n".join(rows) if rows else "- 无"


def _extract_issue_context(
    *,
    user_query: str,
    retrieval_queries: list[str],
    evidence_hits: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    从问题文本与检索上下文提取“症状结构化”信息。

    提取维度：
    1. 关键信号：error/exception/告警/超时/4xx/5xx 等；
    2. 标识符：request_id/trace_id/campaign_id 等；
    3. 时间窗口：时间点、日期、近 N 分钟；
    4. 检索与证据规模：用于后续置信度与追问判断。
    """
    normalized_query = str(user_query or "").strip()
    retrieval_query_count = len(retrieval_queries)
    ids = _dedup(_ID_CAPTURE_RE.findall(normalized_query), 4)
    time_hints = _dedup(_TIME_CAPTURE_RE.findall(normalized_query), 4)
    error_signals = _dedup([match.group(0) for match in _ERROR_SIGNAL_RE.finditer(normalized_query)], 5)
    symptom_excerpt = normalized_query[:140] if normalized_query else ""
    return {
        "symptom_excerpt": symptom_excerpt,
        "ids": ids,
        "time_hints": time_hints,
        "error_signals": error_signals,
        "retrieval_query_count": retrieval_query_count,
        "evidence_count": len(evidence_hits),
    }


def _build_issue_context_block(issue_context: dict[str, Any]) -> str:
    """
    将 issue_context 转换为提示词文本。

    说明：
    文本块保持简洁，避免把模型注意力拉走；
    同时保留“可定位信息”以提升建议可执行性。
    """
    rows = [
        f"- symptom_excerpt: {issue_context.get('symptom_excerpt', '') or '--'}",
        f"- ids: {', '.join(issue_context.get('ids', []) or []) or '--'}",
        f"- time_hints: {', '.join(issue_context.get('time_hints', []) or []) or '--'}",
        f"- error_signals: {', '.join(issue_context.get('error_signals', []) or []) or '--'}",
        f"- retrieval_query_count: {int(issue_context.get('retrieval_query_count', 0) or 0)}",
        f"- evidence_count: {int(issue_context.get('evidence_count', 0) or 0)}",
    ]
    return "\n".join(rows)


def _collect_anchor_tokens(evidence_hits: list[dict[str, Any]], *, max_items: int = 6) -> list[str]:
    """
    从证据中提取用于校验的锚点 token。

    锚点来源：
    - code: path 文件名、symbol_name、section；
    - wiki/case: title、section、path 文件名。
    """
    raw_tokens: list[str] = []
    for item in evidence_hits[:max_items]:
        path = str(item.get("path", "")).strip().lower()
        title = str(item.get("title", "")).strip().lower()
        section = str(item.get("section", "")).strip().lower()
        symbol_name = str(item.get("symbol_name", "")).strip().lower()

        if path:
            raw_tokens.append(path)
            raw_tokens.append(path.split("/")[-1])
            raw_tokens.append(path.split("\\")[-1])
        if title:
            raw_tokens.append(title)
        if section:
            raw_tokens.append(section)
        if symbol_name:
            raw_tokens.append(symbol_name)

    deduped: list[str] = []
    seen: set[str] = set()
    for token in raw_tokens:
        normalized = token.strip().lower()
        if len(normalized) < 3:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _answer_mentions_anchor(answer_text: str, anchor_tokens: list[str]) -> bool:
    """
    判断回答是否至少提到了一个证据锚点。

    说明：
    该校验用于拦截“看似完整但完全不可追踪”的模型输出，
    在命中证据较多时尤其重要。
    """
    normalized_answer = str(answer_text or "").lower().strip()
    if not normalized_answer:
        return False
    return any(token in normalized_answer for token in anchor_tokens if token)


def _format_anchor(item: dict[str, Any]) -> str:
    """
    格式化单条证据锚点为可读文本。
    """
    path = str(item.get("path", "")).strip() or "unknown_path"
    symbol_name = str(item.get("symbol_name", "")).strip() or str(item.get("section", "")).strip() or "unknown_section"
    start_line = item.get("start_line")
    end_line = item.get("end_line")
    if start_line is not None and end_line is not None:
        line_text = f"{start_line}-{end_line}"
    elif start_line is not None:
        line_text = str(start_line)
    else:
        line_text = "unknown"
    return f"`{path}` -> `{symbol_name}` (line: {line_text})"


def _split_evidence_by_source(evidence_hits: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    将融合后的证据按来源拆分，便于兜底策略按源组合。
    """
    wiki_hits: list[dict[str, Any]] = []
    code_hits: list[dict[str, Any]] = []
    case_hits: list[dict[str, Any]] = []
    for item in evidence_hits:
        source = normalize_source_type(item.get("source_type"))
        if source == "code":
            code_hits.append(item)
        elif source == "case":
            case_hits.append(item)
        else:
            wiki_hits.append(item)
    return wiki_hits, code_hits, case_hits


def _build_root_cause_from_evidence(
    *,
    module_name: str,
    evidence_hits: list[dict[str, Any]],
    issue_context: dict[str, Any],
) -> str:
    """
    基于头部证据构建“根因判断”兜底句。
    """
    if not evidence_hits:
        return (
            f"当前更可能与 `{module_name}` 相关，但缺少可直接佐证根因的证据。"
            "建议先补充关键日志片段、错误堆栈或具体函数名。"
        )

    top = evidence_hits[0]
    source = normalize_source_type(top.get("source_type"))
    error_signals = list(issue_context.get("error_signals", []) or [])
    error_hint = f" 已观测到异常信号：{', '.join(error_signals[:2])}。" if error_signals else ""
    if source == "code":
        return f"高概率问题集中在 {_format_anchor(top)} 所在逻辑分支，需重点核查输入边界、异常路径与状态更新链路。{error_hint}"
    title = str(top.get("title", "")).strip() or str(top.get("path", "")).strip() or "证据文档"
    return f"高概率与 `{title}` 描述的业务流程或规则执行有关，需结合模块 `{module_name}` 的实现路径做交叉核对。{error_hint}"


def _build_risks_from_evidence(
    *,
    issue_context: dict[str, Any],
    code_hits: list[dict[str, Any]],
    wiki_hits: list[dict[str, Any]],
    case_hits: list[dict[str, Any]],
) -> list[str]:
    """
    生成证据驱动的风险提示。
    """
    risks: list[str] = []
    ids = list(issue_context.get("ids", []) or [])
    time_hints = list(issue_context.get("time_hints", []) or [])
    if ids:
        risks.append(f"需沿 `{', '.join(ids[:3])}` 串联日志与链路节点，否则可能误判故障传播路径。")
    if time_hints:
        risks.append(f"需在 `{', '.join(time_hints[:2])}` 对齐监控窗口与发布/配置变更时间，避免把历史噪声当成本次根因。")
    if code_hits and wiki_hits:
        risks.append("代码实现与文档口径可能存在版本差异，若不比对当前线上版本，修复动作可能偏离真实故障点。")
    if case_hits:
        risks.append("历史案例仅能提供相似场景参考，若直接套用处理方案，可能忽略本次上下游差异。")
    if not risks:
        risks.append("当前证据覆盖面较窄，若直接实施修复可能带来误改风险。")
    if len(risks) < 2:
        risks.append("若未先做灰度与回滚预案，修复动作可能在高峰流量下放大影响面。")
    return _dedup(risks, 3)


def _build_fix_plan_from_evidence(
    *,
    module_name: str,
    code_hits: list[dict[str, Any]],
    wiki_hits: list[dict[str, Any]],
) -> list[str]:
    """
    生成可落地的修复建议。

    说明：
    优先给“定位点 + 动作”描述，避免输出泛化口号。
    """
    plans: list[str] = []
    for item in code_hits[:2]:
        plans.append(f"先检查 {_format_anchor(item)}，补齐输入校验、异常分支保护与关键状态写入条件。")

    if wiki_hits:
        wiki_title = str(wiki_hits[0].get("title", "")).strip() or str(wiki_hits[0].get("path", "")).strip() or "业务文档"
        plans.append(f"对照 `{wiki_title}` 的业务规则，核对 `{module_name}` 当前实现与规则是否一致。")

    plans.append("在修复分支中增加结构化日志字段（trace_id/request_id/关键参数），确保后续可快速复盘。")
    plans.append("补充针对本次症状的回归用例，并覆盖异常重试与边界输入场景。")
    return _dedup(plans, 4)


def _build_verification_steps(
    *,
    module_name: str,
    issue_context: dict[str, Any],
    evidence_hits: list[dict[str, Any]],
) -> list[str]:
    """
    生成验证步骤，确保“修复 -> 验证 -> 观测”闭环。
    """
    steps: list[str] = []
    ids = list(issue_context.get("ids", []) or [])
    time_hints = list(issue_context.get("time_hints", []) or [])
    if ids:
        steps.append(f"使用 `{', '.join(ids[:2])}` 复现并回放问题请求，确认核心异常不再出现。")
    else:
        steps.append("按原故障触发路径复现一次，并记录修复前后关键日志差异。")

    if time_hints:
        steps.append(f"对比 `{', '.join(time_hints[:2])}` 前后监控指标，确认故障指标恢复且未引入新波动。")
    else:
        steps.append("连续观察 1-2 个监控窗口，确认错误率、时延和队列积压恢复到正常区间。")

    if evidence_hits:
        steps.append(f"抽样核对证据锚点（如 {_format_anchor(evidence_hits[0])}）相关分支在生产流量下按预期执行。")
    else:
        steps.append(f"补充 `{module_name}` 链路的关键日志后再做二次验证，避免“看起来恢复”但缺少证据。")
    return _dedup(steps, 3)


def _build_fallback_from_evidence(
    *,
    module_name: str,
    issue_context: dict[str, Any],
    evidence_hits: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    构建证据驱动兜底结果。

    返回字段会直接写入 analysis，保持与 LLM 结构输出一致。
    """
    wiki_hits, code_hits, case_hits = _split_evidence_by_source(evidence_hits)
    root_cause = _build_root_cause_from_evidence(
        module_name=module_name,
        evidence_hits=evidence_hits,
        issue_context=issue_context,
    )
    risks = _build_risks_from_evidence(
        issue_context=issue_context,
        code_hits=code_hits,
        wiki_hits=wiki_hits,
        case_hits=case_hits,
    )
    fix_plan = _build_fix_plan_from_evidence(
        module_name=module_name,
        code_hits=code_hits,
        wiki_hits=wiki_hits,
    )
    verification_steps = _build_verification_steps(
        module_name=module_name,
        issue_context=issue_context,
        evidence_hits=evidence_hits,
    )
    need_user_confirmation = len(evidence_hits) < 2
    return {
        "root_cause": root_cause,
        "risks": risks,
        "fix_plan": fix_plan,
        "verification_steps": verification_steps,
        "need_user_confirmation": need_user_confirmation,
    }


def _infer_confidence(*, domain_relevance: float, evidence_hits: list[dict[str, Any]]) -> str:
    """
    推断分析置信度（high/medium/low）。
    """
    if not evidence_hits:
        return "low"
    source_set = {normalize_source_type(item.get("source_type")) for item in evidence_hits}
    if domain_relevance >= 0.75 and len(source_set) >= 2:
        return "high"
    if domain_relevance >= 0.6 and len(evidence_hits) >= 2:
        return "medium"
    return "low"


def _run_llm_for_issue(
    service: Any,
    *,
    user_query: str,
    module_name: str,
    module_hint: str,
    related_modules: list[dict[str, Any]],
    retrieval_queries: list[str],
    issue_context: dict[str, Any],
    evidence_hits: list[dict[str, Any]],
) -> tuple[str | None, str | None, dict[str, Any]]:
    """
    执行 issue_analysis 的 LLM 调用（若可用）。

    返回：
    - answer_text：成功文本；
    - fallback_reason：失败原因；
    - call_status：调用状态快照。
    """
    llm_client = getattr(service, "_llm_client", None)
    if llm_client is None:
        return None, None, _default_llm_call_status()

    retrieval_text = "\n".join(f"- {item}" for item in retrieval_queries) if retrieval_queries else "- 无"
    request = CommonLLMRequest(
        node_name="issue_analysis",
        system_prompt=_resolve_issue_system_prompt(service),
        user_prompt=ISSUE_USER_PROMPT_TEMPLATE.format(
            user_query=user_query,
            module_name=module_name,
            module_hint=module_hint,
            related_modules_block=_build_related_modules_block(related_modules),
            retrieval_queries=retrieval_text,
            issue_context_block=_build_issue_context_block(issue_context),
            evidence_block=build_evidence_block(evidence_hits),
        ),
        evidence_count=len(evidence_hits),
        require_evidence=True,
        log_namespace="workflow.llm_issue_analysis",
        metadata={
            "module_name": module_name,
            "retrieval_query_count": len(retrieval_queries),
            "related_module_count": len(related_modules),
            "user_query_preview": user_query[:120],
        },
        normalize_answer=lambda text: text.strip(),
        validate_answer=_validate_issue_answer,
    )

    if hasattr(llm_client, "generate_with_status"):
        return llm_client.generate_with_status(request)

    if hasattr(llm_client, "generate_issue_analysis_with_status"):
        return llm_client.generate_issue_analysis_with_status(
            user_query=user_query,
            module_name=module_name,
            module_hint=module_hint,
            retrieval_queries=retrieval_queries,
            evidence_hits=evidence_hits,
        )

    # 兼容旧接口：仅返回 text 与 reason，从 llm_client.last_call_status 补状态。
    llm_text, llm_fallback_reason = llm_client.generate_issue_analysis(
        user_query=user_query,
        module_name=module_name,
        module_hint=module_hint,
        retrieval_queries=retrieval_queries,
        evidence_hits=evidence_hits,
    )
    status = getattr(llm_client, "last_call_status", {}) or {}
    call_status = dict(status) if isinstance(status, dict) else _default_llm_call_status()
    return llm_text, llm_fallback_reason, call_status


def _format_numbered(items: list[str], *, max_items: int, default_line: str = "--") -> str:
    """
    将字符串列表格式化为编号文本。
    """
    rows = [str(item).strip() for item in items if str(item).strip()]
    if not rows:
        return f"1. {default_line}"
    return "\n".join(f"{index}. {text}" for index, text in enumerate(rows[:max_items], start=1))


def run(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """
    issue_analysis 节点执行入口。

    执行流程：
    1. 聚合问题上下文与证据；
    2. 优先尝试 LLM 结构化分析；
    3. 对 LLM 结果执行锚点校验，不合格则回退；
    4. 用证据驱动兜底补齐字段，返回统一分析结果。
    """
    module_name = str(state.get("module_name", "") or "")
    module_hint = str(state.get("module_hint", "") or "")
    related_modules = list(state.get("related_modules", []) or [])
    user_query = str(state.get("user_query", "") or "")
    retrieval_queries = list(state.get("retrieval_queries", []) or [])
    domain_relevance = float(state.get("domain_relevance", 0.0) or 0.0)
    evidence_hits = collect_evidence_hits(state)
    wiki_hits, code_hits, case_hits = _split_evidence_by_source(evidence_hits)
    issue_context = _extract_issue_context(
        user_query=user_query,
        retrieval_queries=retrieval_queries,
        evidence_hits=evidence_hits,
    )

    analysis = dict(state.get("analysis") or {})
    analysis.update(
        {
            "module": module_name,
            "related_modules": related_modules,
            "confidence": _infer_confidence(domain_relevance=domain_relevance, evidence_hits=evidence_hits),
        }
    )

    llm_mode = "fallback_rule"
    llm_fallback_reason: str | None = None
    llm_call_status = _default_llm_call_status()
    llm_answer_text: str | None = None
    llm_client = getattr(service, "_llm_client", None)

    llm_answer_text, llm_fallback_reason, llm_call_status = _run_llm_for_issue(
        service,
        user_query=user_query,
        module_name=module_name,
        module_hint=module_hint,
        related_modules=related_modules,
        retrieval_queries=retrieval_queries,
        issue_context=issue_context,
        evidence_hits=evidence_hits,
    )

    llm_call_status.setdefault("model", getattr(getattr(llm_client, "config", None), "model", None))
    llm_call_status.setdefault("invoked", bool(llm_client is not None))

    fallback_payload = _build_fallback_from_evidence(
        module_name=module_name,
        issue_context=issue_context,
        evidence_hits=evidence_hits,
    )
    anchor_tokens = _collect_anchor_tokens(evidence_hits, max_items=8)

    if llm_answer_text:
        parsed = _parse_llm_answer(llm_answer_text)
        if anchor_tokens and not _answer_mentions_anchor(llm_answer_text, anchor_tokens):
            llm_fallback_reason = "llm_missing_evidence_anchor"
            llm_call_status.update(
                {
                    "status": "success_discarded",
                    "reason": "llm_missing_evidence_anchor",
                }
            )
            analysis.update(fallback_payload)
        else:
            analysis.update(parsed)
            # LLM 成功后仍用证据兜底补齐缺失字段，避免输出结构不完整。
            for key, value in fallback_payload.items():
                analysis.setdefault(key, value)
            analysis["issue_analysis_llm_summary"] = llm_answer_text
            llm_mode = "llm"
    else:
        analysis.update(fallback_payload)

    root_cause = str(analysis.get("root_cause", "") or "").strip() or "--"
    fix_plan = [str(item) for item in list(analysis.get("fix_plan", []) or [])]
    risks = [str(item) for item in list(analysis.get("risks", []) or [])]
    verification_steps = [str(item) for item in list(analysis.get("verification_steps", []) or [])]
    need_user_confirmation = bool(analysis.get("need_user_confirmation", False))
    risk_text = _format_numbered(risks, max_items=3)
    fix_plan_text = _format_numbered(fix_plan, max_items=4)
    verification_text = _format_numbered(verification_steps, max_items=3)
    mode_line = "本轮结合 LLM 与证据完成分析。" if llm_mode == "llm" else "本轮使用证据驱动的规则策略完成分析。"

    analysis["summary"] = "问题分析已完成"
    analysis["generation_mode"] = llm_mode
    analysis["llm_fallback_reason"] = llm_fallback_reason
    analysis["llm_call_status"] = llm_call_status
    analysis["issue_analysis_generation_mode"] = llm_mode
    analysis["issue_analysis_llm_fallback_reason"] = llm_fallback_reason
    analysis["llm_enabled"] = bool(llm_client is not None and llm_client.config.enabled)
    analysis["llm_available"] = bool(llm_client is not None and llm_client.is_available)
    analysis["llm_model"] = llm_client.config.model if llm_client is not None else None
    analysis["issue_context"] = issue_context
    analysis["evidence_count"] = len(evidence_hits)
    analysis["wiki_evidence_count"] = len(wiki_hits)
    analysis["code_evidence_count"] = len(code_hits)
    analysis["case_evidence_count"] = len(case_hits)
    analysis["highlights"] = [
        "问题分析优先尝试 LLM 结构化输出，失败或不合格时回退证据驱动兜底",
        "LLM 输出会做证据锚点校验，避免不可追踪结论进入最终答复",
        "对外统一输出 generation_mode / llm_fallback_reason / llm_call_status",
    ]

    return {
        "response_kind": "issue_analysis",
        "status": "completed",
        "answer": (
            f"本轮输入已进入问题分析链路。当前更可能定位在 `{module_name}`。\n"
            f"{mode_line}\n"
            f"\n**根因判断**\n{root_cause}\n\n"
            f"**风险提示**\n{risk_text}\n\n"
            f"**修复建议**\n{fix_plan_text}\n\n"
            f"**验证步骤**\n{verification_text}"
            + (
                "\n\n**补充信息建议**\n1. 请补充更精确的错误堆栈或关键日志片段。\n"
                "2. 请补充影响范围（流量/模块/时间窗）以便收敛根因。"
                if need_user_confirmation
                else ""
            )
        ),
        "analysis": analysis,
        "node_trace": append_node_trace(
            state,
            "issue_analysis",
            f"module={module_name}, generation={llm_mode}, evidence={len(evidence_hits)}",
        ),
    }
