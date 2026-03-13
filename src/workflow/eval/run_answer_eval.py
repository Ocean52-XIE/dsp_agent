from __future__ import annotations

"""答案质量评测脚本。

该脚本用于补齐“只看检索指标不够”的评测空白，目标是评估：

1. 回答是否覆盖了预期要点（required facts）；
2. 回答是否出现禁忌结论（forbidden claims）；
3. 回答是否符合预期响应模式（正常回答 / 证据不足）；
4. 回答引用是否命中期望文档（citation hit）；
5. （可选）使用 LLM 作为评审器补充“语义正确性/幻觉风险”打分。

说明：
- 默认只启用规则评测，保证开箱可运行且可复现。
- 若配置了 LLM 评审环境变量，可在规则指标之外增加语义评审分。
"""

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any
from urllib import error, request


# 将项目根目录加入 sys.path，确保脚本在任意目录执行时都能 import workflow。
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from workflow.engine import WorkflowService  # noqa: E402


def _to_bool(raw_value: str | None, default: bool) -> bool:
    """将字符串环境变量解析为布尔值。"""
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _to_float(raw_value: str | None, default: float) -> float:
    """将字符串环境变量解析为浮点数。"""
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


def _to_int(raw_value: str | None, default: int) -> int:
    """将字符串环境变量解析为整数。"""
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


@dataclass
class AnswerEvalCase:
    """单条答案评测样本。"""

    case_id: str
    query: str
    required_facts: list[list[str]]
    forbidden_claims: list[str]
    expected_mode: str
    gold_paths: list[str]
    gold_wiki_paths: list[str]
    gold_code_paths: list[str]
    expected_sources: list[str]
    expected_code_symbols: list[str]
    hybrid_type: str
    intent: str
    note: str


@dataclass
class EvalJudgeLLMConfig:
    """答案评测中 LLM 评审器的配置。"""

    enabled: bool
    base_url: str
    api_key: str
    model: str
    timeout_seconds: int
    temperature: float
    max_tokens: int

    @classmethod
    def from_env(cls) -> "EvalJudgeLLMConfig":
        """从环境变量读取评审器配置。

        环境变量前缀统一使用 `WORKFLOW_EVAL_JUDGE_LLM_`，避免和业务问答 LLM 配置混淆。
        """
        return cls(
            enabled=_to_bool(os.getenv("WORKFLOW_EVAL_JUDGE_LLM_ENABLED"), False),
            base_url=os.getenv("WORKFLOW_EVAL_JUDGE_LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            api_key=os.getenv("WORKFLOW_EVAL_JUDGE_LLM_API_KEY", "").strip(),
            model=os.getenv("WORKFLOW_EVAL_JUDGE_LLM_MODEL", "gpt-4.1-mini").strip(),
            timeout_seconds=_to_int(os.getenv("WORKFLOW_EVAL_JUDGE_LLM_TIMEOUT_SECONDS"), 20),
            temperature=_to_float(os.getenv("WORKFLOW_EVAL_JUDGE_LLM_TEMPERATURE"), 0.0),
            max_tokens=_to_int(os.getenv("WORKFLOW_EVAL_JUDGE_LLM_MAX_TOKENS"), 400),
        )


class AnswerJudgeLLMClient:
    """用于答案质量评测的可选 LLM 评审器。"""

    SYSTEM_PROMPT = """你是一个严谨的问答评测器。
你的任务是根据输入的“问题、预期要点、禁忌结论、模型回答、引用路径”做质量评估。
你必须只输出 JSON，不要输出其他文本。
JSON 字段要求：
{
  "score": 0~1 浮点数,
  "pass": true/false,
  "hallucination_risk": 0~1 浮点数,
  "reason": "一句话说明"
}
评分标准：
1) 要点覆盖越全，score 越高；
2) 出现与证据无关或明显错误结论，score 明显降低；
3) 若回答应为“证据不足”却强行下结论，判定为高风险。"""

    USER_PROMPT_TEMPLATE = """【问题】
{query}

【预期响应模式】
{expected_mode}

【required_facts】
{required_facts}

【forbidden_claims】
{forbidden_claims}

【gold_paths】
{gold_paths}

【模型回答】
{answer}

【引用路径】
{citation_paths}
"""

    def __init__(self, config: EvalJudgeLLMConfig) -> None:
        self.config = config

    @property
    def is_available(self) -> bool:
        """判断当前是否可调用 LLM 评审器。"""
        if not self.config.enabled:
            return False
        if not self.config.api_key:
            return False
        if not self.config.model:
            return False
        return True

    @classmethod
    def from_env(cls) -> "AnswerJudgeLLMClient":
        """基于环境变量构建评审器实例。"""
        return cls(EvalJudgeLLMConfig.from_env())

    def evaluate(
        self,
        *,
        query: str,
        expected_mode: str,
        required_facts: list[list[str]],
        forbidden_claims: list[str],
        gold_paths: list[str],
        answer: str,
        citation_paths: list[str],
    ) -> tuple[dict[str, Any] | None, str | None]:
        """执行 LLM 评审。

        返回：
        - 评审结果 dict；失败时为 None
        - 失败原因；成功时为 None
        """
        if not self.is_available:
            return None, "llm_judge_unavailable"

        required_text = json.dumps(required_facts, ensure_ascii=False)
        forbidden_text = json.dumps(forbidden_claims, ensure_ascii=False)
        gold_paths_text = json.dumps(gold_paths, ensure_ascii=False)
        citation_paths_text = json.dumps(citation_paths, ensure_ascii=False)
        user_prompt = self.USER_PROMPT_TEMPLATE.format(
            query=query,
            expected_mode=expected_mode,
            required_facts=required_text,
            forbidden_claims=forbidden_text,
            gold_paths=gold_paths_text,
            answer=answer,
            citation_paths=citation_paths_text,
        )

        try:
            raw_text = self._chat_completion(system_prompt=self.SYSTEM_PROMPT, user_prompt=user_prompt)
            result = self._parse_json_result(raw_text)
            return result, None
        except TimeoutError:
            return None, "timeout"
        except (error.URLError, error.HTTPError):
            return None, "http_error"
        except Exception:
            return None, "unknown_error"

    def _chat_completion(self, *, system_prompt: str, user_prompt: str) -> str:
        """调用 OpenAI 兼容接口。"""
        payload = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        payload_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        endpoint = f"{self.config.base_url}/chat/completions"
        req = request.Request(
            endpoint,
            data=payload_bytes,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
        )
        with request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
            raw = resp.read().decode("utf-8")

        response_json = json.loads(raw)
        choices = response_json.get("choices", [])
        if not choices:
            return ""
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(part for part in parts if part).strip()
        return str(content)

    def _parse_json_result(self, raw_text: str) -> dict[str, Any]:
        """解析评审器返回 JSON。

        兼容场景：
        - 模型返回纯 JSON
        - 模型在 JSON 外包裹了少量文本（会尝试提取首个 `{...}`）
        """
        text = raw_text.strip()
        if not text:
            return {"score": 0.0, "pass": False, "hallucination_risk": 1.0, "reason": "empty_llm_output"}

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # 从文本中提取第一个 JSON 对象片段。
            matched = re.search(r"\{[\s\S]*\}", text)
            if not matched:
                return {"score": 0.0, "pass": False, "hallucination_risk": 1.0, "reason": "invalid_llm_json"}
            parsed = json.loads(matched.group(0))

        score = float(parsed.get("score", 0.0))
        hallucination_risk = float(parsed.get("hallucination_risk", 0.0))
        return {
            "score": max(0.0, min(1.0, score)),
            "pass": bool(parsed.get("pass", False)),
            "hallucination_risk": max(0.0, min(1.0, hallucination_risk)),
            "reason": str(parsed.get("reason", "")).strip(),
        }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _display_path(path: Path) -> str:
    """优先返回相对项目根目录路径，失败时回退绝对路径。"""
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _normalize_text(text: str) -> str:
    """文本归一化：统一小写并移除空白和常见标点。"""
    lowered = text.lower()
    no_spaces = re.sub(r"\s+", "", lowered)
    return re.sub(r"[`~!@#$%^&*()\-_=+\[\]{}\\|;:'\",.<>/?，。！？；：、“”‘’（）【】《》]", "", no_spaces)


def _contains_text(haystack: str, needle: str) -> bool:
    """判断归一化后的 haystack 是否包含 needle。"""
    normalized_haystack = _normalize_text(haystack)
    normalized_needle = _normalize_text(needle)
    if not normalized_needle:
        return False
    return normalized_needle in normalized_haystack


def _parse_required_facts(raw_required_facts: list[Any]) -> list[list[str]]:
    """将 required_facts 解析为“同义词组”结构。

    支持两种输入：
    - 字符串：表示单个要点
    - 字符串列表：表示同义词/等价表达组（命中任意一个即算命中该要点）
    """
    groups: list[list[str]] = []
    for item in raw_required_facts:
        if isinstance(item, str):
            normalized = item.strip()
            if normalized:
                groups.append([normalized])
            continue
        if isinstance(item, list):
            synonyms = [str(elem).strip() for elem in item if str(elem).strip()]
            if synonyms:
                groups.append(synonyms)
    return groups


def _dedupe_non_empty(items: list[str]) -> list[str]:
    """对字符串列表做去重并保持原始顺序。"""
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = str(item).strip()
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _normalize_source_type(raw_source: str) -> str:
    """将引用 source_type 归一化为稳定枚举值。"""
    normalized = str(raw_source).strip().lower()
    if normalized.startswith("wiki"):
        return "wiki"
    if normalized.startswith("code"):
        return "code"
    if normalized.startswith("case"):
        return "case"
    return normalized or "unknown"


def _normalize_hybrid_type(raw_type: str) -> str:
    """归一化 hybrid 题型标签。"""
    normalized = str(raw_type).strip().lower()
    mapping = {
        "formula": "formula",
        "reason": "reason",
        "function_location": "function_location",
        "location": "function_location",
        "mixed_followup": "mixed_followup",
        "mixed": "mixed_followup",
    }
    return mapping.get(normalized, normalized or "unknown")


def _load_dataset(path: Path) -> list[AnswerEvalCase]:
    """读取答案评测数据集（JSONL）。"""
    items: list[AnswerEvalCase] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)

            case_id = str(record.get("id", f"case_{line_number:04d}"))
            query = str(record.get("query", "")).strip()
            if not query:
                raise ValueError(f"dataset line {line_number} missing query")

            raw_required_facts = record.get("required_facts", [])
            if not isinstance(raw_required_facts, list):
                raise ValueError(f"dataset line {line_number} required_facts should be list")
            required_facts = _parse_required_facts(raw_required_facts)

            forbidden_claims = [str(item).strip() for item in record.get("forbidden_claims", []) if str(item).strip()]
            expected_mode = str(record.get("expected_mode", "answer")).strip() or "answer"
            # 兼容旧格式：
            # - 旧字段 gold_paths（默认按 wiki 处理）
            # 新增字段：
            # - gold_wiki_paths
            # - gold_code_paths
            # - expected_sources（期望至少出现哪些来源，如 ["wiki"] / ["code"] / ["wiki","code"]）
            legacy_gold_paths = [str(item).strip() for item in record.get("gold_paths", []) if str(item).strip()]
            gold_wiki_paths = _dedupe_non_empty(
                [str(item).strip() for item in record.get("gold_wiki_paths", legacy_gold_paths) if str(item).strip()]
            )
            gold_code_paths = _dedupe_non_empty(
                [str(item).strip() for item in record.get("gold_code_paths", []) if str(item).strip()]
            )
            gold_paths = _dedupe_non_empty([*gold_wiki_paths, *gold_code_paths])

            raw_expected_sources = record.get("expected_sources", [])
            if isinstance(raw_expected_sources, list):
                expected_sources = _dedupe_non_empty([_normalize_source_type(str(item)) for item in raw_expected_sources])
            else:
                expected_sources = []
            if not expected_sources:
                # 未显式配置时自动推断默认期望来源，便于旧数据平滑升级。
                if gold_wiki_paths:
                    expected_sources.append("wiki")
                if gold_code_paths:
                    expected_sources.append("code")

            expected_code_symbols = _dedupe_non_empty(
                [str(item).strip() for item in record.get("expected_code_symbols", []) if str(item).strip()]
            )
            hybrid_type = _normalize_hybrid_type(str(record.get("hybrid_type", "unknown")))

            items.append(
                AnswerEvalCase(
                    case_id=case_id,
                    query=query,
                    required_facts=required_facts,
                    forbidden_claims=forbidden_claims,
                    expected_mode=expected_mode,
                    gold_paths=gold_paths,
                    gold_wiki_paths=gold_wiki_paths,
                    gold_code_paths=gold_code_paths,
                    expected_sources=expected_sources,
                    expected_code_symbols=expected_code_symbols,
                    hybrid_type=hybrid_type,
                    intent=str(record.get("intent", "knowledge_qa")).strip() or "knowledge_qa",
                    note=str(record.get("note", "")).strip(),
                )
            )
    return items


def _weighted_average(scores: dict[str, float], weights: dict[str, float]) -> float:
    """按权重计算加权平均分。

    仅对同时出现在 `scores` 与 `weights` 且权重 > 0 的项参与计算。
    """
    numerator = 0.0
    denominator = 0.0
    for key, value in scores.items():
        weight = float(weights.get(key, 0.0))
        if weight <= 0:
            continue
        numerator += weight * value
        denominator += weight
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _compute_backoff_seconds(*, base_delay_ms: int, retry_index: int) -> float:
    """计算指数退避时间（秒）。

    参数说明：
    - base_delay_ms: 基础退避时长（毫秒）
    - retry_index: 从 1 开始，表示第几次重试
    """
    delay_ms = max(base_delay_ms, 0) * (2 ** max(retry_index - 1, 0))
    return delay_ms / 1000.0


def _should_retry_qa_response(
    *,
    response: dict[str, Any],
    qa_llm_enabled: bool,
    qa_llm_available: bool,
) -> bool:
    """判断 QA 结果是否需要重试。

    仅在以下条件触发：
    1. QA LLM 处于启用且可用状态；
    2. 当前回答走了 fallback_rule；
    3. fallback 原因为网络/超时类波动。
    """
    if not (qa_llm_enabled and qa_llm_available):
        return False
    analysis = response.get("analysis") or {}
    generation_mode = str(analysis.get("generation_mode", "")).strip().lower()
    fallback_reason = str(analysis.get("llm_fallback_reason", "")).strip().lower()
    if generation_mode != "fallback_rule":
        return False
    retryable_prefixes = ("timeout", "http_error", "unknown_error", "empty_answer")
    return any(fallback_reason.startswith(prefix) for prefix in retryable_prefixes)


def run_eval(config_path: Path) -> dict[str, Any]:
    """执行答案质量评测并输出结构化报告。"""
    config = _load_json(config_path)
    dataset_path = (PROJECT_ROOT / config["dataset_path"]).resolve()
    output_path = (PROJECT_ROOT / config["output_path"]).resolve()

    session_id_prefix = str(config.get("session_id_prefix", "ans_eval_sess"))
    trace_id_prefix = str(config.get("trace_id_prefix", "ans_eval_trace"))
    save_per_case = bool(config.get("save_per_case", True))

    # exact_correct 的阈值与约束：
    # - 覆盖率至少达到阈值；
    # - 不能触发 forbidden claims；
    # - expected_mode 必须正确；
    # - 如要求引用命中，则 citation_hit 也必须为真。
    min_required_coverage_for_pass = float(config.get("min_required_coverage_for_pass", 0.8))
    require_citation_for_exact = bool(config.get("require_citation_for_exact", True))

    # 可选 LLM 评审：用于补充语义正确性打分，不替代规则指标。
    enable_llm_judge = bool(config.get("enable_llm_judge", False))
    llm_judge_required_for_exact = bool(config.get("llm_judge_required_for_exact", False))
    llm_judge_pass_threshold = float(config.get("llm_judge_pass_threshold", 0.6))

    # 可配置指标权重。
    weights: dict[str, float] = {
        "required_fact_coverage": float(config.get("weights", {}).get("required_fact_coverage", 0.45)),
        "forbidden_claim_pass": float(config.get("weights", {}).get("forbidden_claim_pass", 0.2)),
        "expected_mode_accuracy": float(config.get("weights", {}).get("expected_mode_accuracy", 0.15)),
        "citation_hit": float(config.get("weights", {}).get("citation_hit", 0.2)),
        # 代码引用命中（仅在样本配置 gold_code_paths 时参与）。
        "code_citation_hit": float(config.get("weights", {}).get("code_citation_hit", 0.0)),
        # 期望来源覆盖（如期望同时命中 wiki + code，则需两者都出现）。
        "expected_source_hit": float(config.get("weights", {}).get("expected_source_hit", 0.0)),
        "llm_judge_score": float(config.get("weights", {}).get("llm_judge_score", 0.0)),
    }

    cases = _load_dataset(dataset_path)
    service = WorkflowService()
    llm_judge_client = AnswerJudgeLLMClient.from_env()
    qa_llm_available = bool(
        hasattr(service, "_knowledge_qa_llm")
        and service._knowledge_qa_llm is not None
        and service._knowledge_qa_llm.is_available
    )
    qa_llm_enabled = bool(
        hasattr(service, "_knowledge_qa_llm")
        and service._knowledge_qa_llm is not None
        and service._knowledge_qa_llm.config.enabled
    )
    qa_llm_model = (
        service._knowledge_qa_llm.config.model
        if hasattr(service, "_knowledge_qa_llm") and service._knowledge_qa_llm is not None
        else ""
    )
    # 评测稳定性参数（P0）：
    # - 默认重试 1 次（即总尝试 2 次）；
    # - 使用指数退避，减轻瞬时网络波动导致的误判。
    eval_retry_count = _to_int(os.getenv("WORKFLOW_EVAL_RETRY_COUNT"), 1)
    eval_retry_base_delay_ms = _to_int(os.getenv("WORKFLOW_EVAL_RETRY_BASE_DELAY_MS"), 300)
    if not enable_llm_judge:
        # 显式关闭时，不调用评审器。
        llm_judge_client.config.enabled = False

    metrics_sum = {
        "required_fact_coverage": 0.0,
        "forbidden_claim_pass_rate": 0.0,
        "expected_mode_accuracy": 0.0,
        "citation_hit_rate": 0.0,
        "overall_score": 0.0,
        "exact_correct_rate": 0.0,
    }
    # 多源引用指标采用“按适用样本计分”的口径：
    # - wiki_citation_hit_rate 仅统计配置了 gold_wiki_paths 的样本
    # - code_citation_hit_rate 仅统计配置了 gold_code_paths 的样本
    # - expected_source_hit_rate 仅统计配置了 expected_sources 的样本
    wiki_citation_hit_sum = 0.0
    wiki_citation_case_count = 0
    code_citation_hit_sum = 0.0
    code_citation_case_count = 0
    expected_source_hit_sum = 0.0
    expected_source_case_count = 0
    llm_judge_score_sum = 0.0
    llm_judge_score_count = 0
    per_case_results: list[dict[str, Any]] = []
    # hybrid 专项统计：
    # 1) 按题型（formula/reason/function_location/mixed_followup）统计通过情况；
    # 2) 统计结构化命中能力（公式结构、原因结构、定位锚点、混合双源覆盖）。
    hybrid_type_stats: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "cases": 0.0,
            "required_fact_coverage_sum": 0.0,
            "exact_correct_sum": 0.0,
            "expected_mode_accuracy_sum": 0.0,
        }
    )
    formula_structure_hit_sum = 0.0
    formula_structure_case_count = 0
    reason_structure_hit_sum = 0.0
    reason_structure_case_count = 0
    location_anchor_hit_sum = 0.0
    location_anchor_case_count = 0
    mixed_dual_source_hit_sum = 0.0
    mixed_dual_source_case_count = 0

    insufficient_markers = ("当前证据不足", "证据不足", "信息不足", "缺少证据")

    for case_index, case in enumerate(cases, start=1):
        response: dict[str, Any] = {}
        last_response_error: str | None = None
        # QA 调用重试：
        # 1. 捕获运行时异常（例如临时网络错误）时重试；
        # 2. 即便调用成功，如果落到 fallback_rule 且原因可重试，也再尝试一次。
        for retry_index in range(0, max(eval_retry_count, 0) + 1):
            try:
                response = service.run_user_message(
                    session_id=f"{session_id_prefix}_{case.case_id}",
                    trace_id=f"{trace_id_prefix}_{case_index:04d}_try{retry_index + 1}",
                    user_query=case.query,
                    history=[],
                )
                if not _should_retry_qa_response(
                    response=response,
                    qa_llm_enabled=qa_llm_enabled,
                    qa_llm_available=qa_llm_available,
                ):
                    break
                # 到这里代表“可重试的 fallback_rule”。
                if retry_index >= max(eval_retry_count, 0):
                    break
                time.sleep(_compute_backoff_seconds(base_delay_ms=eval_retry_base_delay_ms, retry_index=retry_index + 1))
            except Exception as exc:  # noqa: BLE001
                last_response_error = str(exc)
                if retry_index >= max(eval_retry_count, 0):
                    response = {
                        "status": "error",
                        "kind": case.intent,
                        "content": "当前证据不足。",
                        "analysis": {
                            "generation_mode": "error",
                            "llm_fallback_reason": "qa_runtime_error",
                            "error": last_response_error,
                        },
                        "citations": [],
                    }
                    break
                time.sleep(_compute_backoff_seconds(base_delay_ms=eval_retry_base_delay_ms, retry_index=retry_index + 1))

        answer = str(response.get("content", "")).strip()
        response_analysis = response.get("analysis") or {}
        generation_mode = str(response_analysis.get("generation_mode", ""))
        qa_llm_fallback_reason = str(response_analysis.get("llm_fallback_reason", ""))
        citations = response.get("citations", [])
        citation_paths_all: list[str] = []
        citation_paths_by_source: dict[str, list[str]] = {"wiki": [], "code": [], "case": []}
        citation_source_set: set[str] = set()
        for item in citations:
            path = str(item.get("path", "")).strip()
            if not path:
                continue
            source = _normalize_source_type(str(item.get("source_type", "")))
            citation_paths_all.append(path)
            citation_source_set.add(source)
            if source in citation_paths_by_source:
                citation_paths_by_source[source].append(path)

        # 保留旧字段语义：citation_paths 默认写入 wiki 引用，避免破坏历史消费逻辑。
        citation_paths = citation_paths_by_source["wiki"]

        # 1) required facts 覆盖率。
        matched_fact_groups: list[str] = []
        missing_fact_groups: list[str] = []
        for synonyms in case.required_facts:
            hit = any(_contains_text(answer, candidate) for candidate in synonyms)
            display_name = " / ".join(synonyms)
            if hit:
                matched_fact_groups.append(display_name)
            else:
                missing_fact_groups.append(display_name)
        if case.required_facts:
            required_fact_coverage = len(matched_fact_groups) / len(case.required_facts)
        else:
            required_fact_coverage = 1.0

        # 2) forbidden claims 命中检查。
        hit_forbidden_claims = [claim for claim in case.forbidden_claims if _contains_text(answer, claim)]
        forbidden_claim_pass = 1.0 if not hit_forbidden_claims else 0.0

        # 3) expected mode 是否符合预期。
        has_insufficient_marker = any(marker in answer for marker in insufficient_markers)
        if case.expected_mode in {"either", "answer_or_insufficient", "flexible"}:
            # 兼容“只关心检索/引用质量，不强约束回答形态”的评测场景。
            expected_mode_accuracy = 1.0
        elif case.expected_mode == "insufficient_evidence":
            expected_mode_accuracy = 1.0 if has_insufficient_marker else 0.0
        else:
            expected_mode_accuracy = 0.0 if has_insufficient_marker else 1.0

        # 4) 引用命中率（支持多源：wiki/code/case）。
        if case.gold_paths:
            citation_hit = 1.0 if any(path in set(case.gold_paths) for path in citation_paths_all) else 0.0
        else:
            citation_hit = 1.0

        wiki_citation_hit: float | None = None
        if case.gold_wiki_paths:
            wiki_citation_hit = (
                1.0 if any(path in set(case.gold_wiki_paths) for path in citation_paths_by_source["wiki"]) else 0.0
            )
            wiki_citation_hit_sum += wiki_citation_hit
            wiki_citation_case_count += 1

        code_citation_hit: float | None = None
        if case.gold_code_paths:
            code_citation_hit = (
                1.0 if any(path in set(case.gold_code_paths) for path in citation_paths_by_source["code"]) else 0.0
            )
            code_citation_hit_sum += code_citation_hit
            code_citation_case_count += 1

        expected_source_hit: float | None = None
        if case.expected_sources:
            expected_source_hit = 1.0 if all(source in citation_source_set for source in case.expected_sources) else 0.0
            expected_source_hit_sum += expected_source_hit
            expected_source_case_count += 1

        # hybrid 专项：代码锚点命中（答案中是否明确出现函数/路径锚点）。
        # 命中规则：
        # - 命中任一 expected_code_symbols，或
        # - 命中任一 gold_code_paths 的文件名/完整路径。
        code_anchor_hit: float | None = None
        if case.expected_code_symbols or case.gold_code_paths:
            expected_symbol_hit = any(_contains_text(answer, symbol) for symbol in case.expected_code_symbols)
            gold_path_hit = False
            for raw_path in case.gold_code_paths:
                normalized_path = str(raw_path).strip()
                if not normalized_path:
                    continue
                if _contains_text(answer, normalized_path):
                    gold_path_hit = True
                    break
                filename = normalized_path.replace("\\", "/").split("/")[-1]
                if filename and _contains_text(answer, filename):
                    gold_path_hit = True
                    break
            code_anchor_hit = 1.0 if (expected_symbol_hit or gold_path_hit) else 0.0

        # 5) 可选 LLM 评审结果。
        llm_judge_result: dict[str, Any] | None = None
        llm_judge_error: str | None = None
        if llm_judge_client.is_available:
            for retry_index in range(0, max(eval_retry_count, 0) + 1):
                llm_judge_result, llm_judge_error = llm_judge_client.evaluate(
                    query=case.query,
                    expected_mode=case.expected_mode,
                    required_facts=case.required_facts,
                    forbidden_claims=case.forbidden_claims,
                    gold_paths=case.gold_paths,
                    answer=answer,
                    citation_paths=citation_paths_all,
                )
                # 成功或非可重试错误，直接退出。
                if llm_judge_result is not None:
                    break
                if llm_judge_error not in {"timeout", "http_error"}:
                    break
                if retry_index >= max(eval_retry_count, 0):
                    break
                time.sleep(_compute_backoff_seconds(base_delay_ms=eval_retry_base_delay_ms, retry_index=retry_index + 1))

        score_components = {
            "required_fact_coverage": required_fact_coverage,
            "forbidden_claim_pass": forbidden_claim_pass,
            "expected_mode_accuracy": expected_mode_accuracy,
            "citation_hit": citation_hit,
        }
        # 代码样本可额外计入“代码引用命中”；多源样本可计入“期望来源覆盖”。
        if code_citation_hit is not None:
            score_components["code_citation_hit"] = code_citation_hit
        if expected_source_hit is not None:
            score_components["expected_source_hit"] = expected_source_hit
        if llm_judge_result is not None:
            score_components["llm_judge_score"] = float(llm_judge_result.get("score", 0.0))
            llm_judge_score_sum += score_components["llm_judge_score"]
            llm_judge_score_count += 1

        overall_score = _weighted_average(score_components, weights)

        # exact_correct：作为严格通过率指标，便于做版本门禁。
        exact_correct = (
            required_fact_coverage >= min_required_coverage_for_pass
            and forbidden_claim_pass == 1.0
            and expected_mode_accuracy == 1.0
            and (not require_citation_for_exact or citation_hit == 1.0)
        )
        if llm_judge_required_for_exact:
            llm_score = float(llm_judge_result.get("score", 0.0)) if llm_judge_result else 0.0
            exact_correct = exact_correct and llm_score >= llm_judge_pass_threshold

        # hybrid 题型分桶统计。
        hybrid_bucket = case.hybrid_type
        if hybrid_bucket and hybrid_bucket != "unknown":
            bucket = hybrid_type_stats[hybrid_bucket]
            bucket["cases"] += 1.0
            bucket["required_fact_coverage_sum"] += required_fact_coverage
            bucket["exact_correct_sum"] += 1.0 if exact_correct else 0.0
            bucket["expected_mode_accuracy_sum"] += expected_mode_accuracy

        # hybrid 专项结构指标：
        # 1) 公式问答：是否具备“公式结构”痕迹。
        if case.hybrid_type == "formula":
            formula_structure_case_count += 1
            has_formula_structure = bool(
                re.search(r"=", answer)
                or _contains_text(answer, "公式")
                or _contains_text(answer, "bid =")
            )
            formula_structure_hit_sum += 1.0 if has_formula_structure else 0.0

        # 2) 原因解释：是否具备“原因结构”（编号或明确原因词）。
        if case.hybrid_type == "reason":
            reason_structure_case_count += 1
            has_reason_structure = bool(
                re.search(r"(?m)^\s*(?:\d+\.|[-*])\s+", answer)
                or _contains_text(answer, "原因")
            )
            reason_structure_hit_sum += 1.0 if has_reason_structure else 0.0

        # 3) 函数定位：答案中是否命中代码锚点。
        if case.hybrid_type == "function_location":
            location_anchor_case_count += 1
            location_anchor_hit_sum += float(code_anchor_hit or 0.0)

        # 4) 混合追问：是否同时命中 wiki + code 来源。
        if case.hybrid_type == "mixed_followup":
            mixed_dual_source_case_count += 1
            mixed_dual_source_hit_sum += float(expected_source_hit or 0.0)

        metrics_sum["required_fact_coverage"] += required_fact_coverage
        metrics_sum["forbidden_claim_pass_rate"] += forbidden_claim_pass
        metrics_sum["expected_mode_accuracy"] += expected_mode_accuracy
        metrics_sum["citation_hit_rate"] += citation_hit
        metrics_sum["overall_score"] += overall_score
        metrics_sum["exact_correct_rate"] += 1.0 if exact_correct else 0.0

        if save_per_case:
            per_case_results.append(
                {
                    "id": case.case_id,
                    "query": case.query,
                    "intent": case.intent,
                    "note": case.note,
                    "expected_mode": case.expected_mode,
                    "gold_paths": case.gold_paths,
                    "gold_wiki_paths": case.gold_wiki_paths,
                    "gold_code_paths": case.gold_code_paths,
                    "expected_sources": case.expected_sources,
                    "expected_code_symbols": case.expected_code_symbols,
                    "hybrid_type": case.hybrid_type,
                    "required_facts": case.required_facts,
                    "forbidden_claims": case.forbidden_claims,
                    "answer": answer,
                    "citation_paths": citation_paths,
                    "citation_paths_all": citation_paths_all,
                    "citation_paths_by_source": citation_paths_by_source,
                    "matched_fact_groups": matched_fact_groups,
                    "missing_fact_groups": missing_fact_groups,
                    "hit_forbidden_claims": hit_forbidden_claims,
                    "required_fact_coverage": round(required_fact_coverage, 6),
                    "forbidden_claim_pass": forbidden_claim_pass,
                    "expected_mode_accuracy": expected_mode_accuracy,
                    "citation_hit": citation_hit,
                    "wiki_citation_hit": wiki_citation_hit,
                    "code_citation_hit": code_citation_hit,
                    "expected_source_hit": expected_source_hit,
                    "code_anchor_hit": code_anchor_hit,
                    "score_components": {k: round(v, 6) for k, v in score_components.items()},
                    "overall_score": round(overall_score, 6),
                    "exact_correct": exact_correct,
                    "response_status": response.get("status", ""),
                    "response_kind": response.get("kind", ""),
                    "generation_mode": generation_mode,
                    "qa_llm_fallback_reason": qa_llm_fallback_reason,
                    "llm_judge": llm_judge_result,
                    "llm_judge_error": llm_judge_error,
                }
            )

    total_cases = len(cases)
    metrics = {key: round(value / total_cases, 6) for key, value in metrics_sum.items()}
    if wiki_citation_case_count > 0:
        metrics["wiki_citation_hit_rate"] = round(wiki_citation_hit_sum / wiki_citation_case_count, 6)
    if code_citation_case_count > 0:
        metrics["code_citation_hit_rate"] = round(code_citation_hit_sum / code_citation_case_count, 6)
    if expected_source_case_count > 0:
        metrics["expected_source_hit_rate"] = round(expected_source_hit_sum / expected_source_case_count, 6)
    if llm_judge_score_count > 0:
        metrics["llm_judge_score_avg"] = round(llm_judge_score_sum / llm_judge_score_count, 6)
    if formula_structure_case_count > 0:
        metrics["formula_structure_hit_rate"] = round(formula_structure_hit_sum / formula_structure_case_count, 6)
    if reason_structure_case_count > 0:
        metrics["reason_structure_hit_rate"] = round(reason_structure_hit_sum / reason_structure_case_count, 6)
    if location_anchor_case_count > 0:
        metrics["location_anchor_hit_rate"] = round(location_anchor_hit_sum / location_anchor_case_count, 6)
    if mixed_dual_source_case_count > 0:
        metrics["mixed_dual_source_hit_rate"] = round(mixed_dual_source_hit_sum / mixed_dual_source_case_count, 6)

    # 按 hybrid 题型输出分桶指标，便于分别观察四类能力质量。
    hybrid_type_metrics: dict[str, dict[str, float]] = {}
    for hybrid_type, stat in sorted(hybrid_type_stats.items()):
        case_count = int(stat["cases"])
        if case_count <= 0:
            continue
        hybrid_type_metrics[hybrid_type] = {
            "cases": case_count,
            "required_fact_coverage": round(stat["required_fact_coverage_sum"] / case_count, 6),
            "expected_mode_accuracy": round(stat["expected_mode_accuracy_sum"] / case_count, 6),
            "exact_correct_rate": round(stat["exact_correct_sum"] / case_count, 6),
        }

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": _display_path(config_path),
        "dataset_path": _display_path(dataset_path),
        "output_path": _display_path(output_path),
        "total_cases": total_cases,
        "min_required_coverage_for_pass": min_required_coverage_for_pass,
        "require_citation_for_exact": require_citation_for_exact,
        "enable_llm_judge": enable_llm_judge,
        "llm_judge_available": llm_judge_client.is_available,
        "qa_llm_enabled": qa_llm_enabled,
        "qa_llm_available": qa_llm_available,
        "qa_llm_model": qa_llm_model,
        "llm_judge_required_for_exact": llm_judge_required_for_exact,
        "llm_judge_pass_threshold": llm_judge_pass_threshold,
        "weights": weights,
        "metric_denominators": {
            "total_cases": total_cases,
            "wiki_citation_cases": wiki_citation_case_count,
            "code_citation_cases": code_citation_case_count,
            "expected_source_cases": expected_source_case_count,
            "formula_cases": formula_structure_case_count,
            "reason_cases": reason_structure_case_count,
            "function_location_cases": location_anchor_case_count,
            "mixed_followup_cases": mixed_dual_source_case_count,
        },
        "metrics": metrics,
        "hybrid_type_metrics": hybrid_type_metrics,
        "per_case": per_case_results if save_per_case else [],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run answer-quality evaluation for workflow QA responses.")
    parser.add_argument(
        "--config",
        default="src/workflow/eval/config.answer.template.json",
        help="Path to answer-eval config JSON (relative to project root or absolute).",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")

    report = run_eval(config_path)
    metrics = report["metrics"]
    print("==== Answer Evaluation Summary ====")
    print(f"total_cases: {report['total_cases']}")
    print(f"required_fact_coverage: {metrics['required_fact_coverage']}")
    print(f"forbidden_claim_pass_rate: {metrics['forbidden_claim_pass_rate']}")
    print(f"expected_mode_accuracy: {metrics['expected_mode_accuracy']}")
    print(f"citation_hit_rate: {metrics['citation_hit_rate']}")
    if "wiki_citation_hit_rate" in metrics:
        print(f"wiki_citation_hit_rate: {metrics['wiki_citation_hit_rate']}")
    if "code_citation_hit_rate" in metrics:
        print(f"code_citation_hit_rate: {metrics['code_citation_hit_rate']}")
    if "expected_source_hit_rate" in metrics:
        print(f"expected_source_hit_rate: {metrics['expected_source_hit_rate']}")
    if "formula_structure_hit_rate" in metrics:
        print(f"formula_structure_hit_rate: {metrics['formula_structure_hit_rate']}")
    if "reason_structure_hit_rate" in metrics:
        print(f"reason_structure_hit_rate: {metrics['reason_structure_hit_rate']}")
    if "location_anchor_hit_rate" in metrics:
        print(f"location_anchor_hit_rate: {metrics['location_anchor_hit_rate']}")
    if "mixed_dual_source_hit_rate" in metrics:
        print(f"mixed_dual_source_hit_rate: {metrics['mixed_dual_source_hit_rate']}")
    print(f"overall_score: {metrics['overall_score']}")
    print(f"exact_correct_rate: {metrics['exact_correct_rate']}")
    if "llm_judge_score_avg" in metrics:
        print(f"llm_judge_score_avg: {metrics['llm_judge_score_avg']}")
    print("report_path:", report["output_path"])
    print("===================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
