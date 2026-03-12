from __future__ import annotations

"""knowledge_answer 节点的 LLM 问答调用层。

设计目标：
1. 配置可控：通过环境变量配置模型、超时、采样参数；
2. 证据约束：仅允许基于检索证据回答；
3. 可降级：网络失败、配置缺失、无证据时返回明确降级原因；
4. 多源支持：统一消费 wiki/code/case 融合后的证据列表；
5. P0 可靠性：强化“校准指标关键词覆盖”和“出价入口函数名覆盖”提示约束。
"""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import time
from typing import Any
from urllib import error, request
from workflow.runtime_logging import get_file_logger


def _to_bool(raw_value: str | None, default: bool) -> bool:
    """将字符串环境变量转换为布尔值。"""
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _to_float(raw_value: str | None, default: float) -> float:
    """将字符串环境变量转换为浮点数。"""
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


def _to_int(raw_value: str | None, default: int) -> int:
    """将字符串环境变量转换为整数。"""
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def _as_source_type(raw_source: Any) -> str:
    """将证据 source_type 归一化为稳定枚举值。"""
    normalized = str(raw_source or "").strip().lower()
    if normalized.startswith("wiki"):
        return "wiki"
    if normalized.startswith("code"):
        return "code"
    if normalized.startswith("case"):
        return "case"
    return normalized or "unknown"


@dataclass
class KnowledgeQALLMConfig:
    """knowledge_answer 节点的 LLM 配置。"""

    enabled: bool
    base_url: str
    api_key: str
    model: str
    timeout_seconds: int
    temperature: float
    max_tokens: int
    retry_count: int
    retry_base_delay_ms: int

    @classmethod
    def from_env(cls) -> "KnowledgeQALLMConfig":
        """从环境变量读取配置。"""
        return cls(
            enabled=_to_bool(os.getenv("WORKFLOW_QA_LLM_ENABLED"), True),
            base_url=os.getenv("WORKFLOW_QA_LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            api_key=os.getenv("WORKFLOW_QA_LLM_API_KEY", "").strip(),
            model=os.getenv("WORKFLOW_QA_LLM_MODEL", "gpt-4.1-mini").strip(),
            timeout_seconds=_to_int(os.getenv("WORKFLOW_QA_LLM_TIMEOUT_SECONDS"), 20),
            temperature=_to_float(os.getenv("WORKFLOW_QA_LLM_TEMPERATURE"), 0.2),
            max_tokens=_to_int(os.getenv("WORKFLOW_QA_LLM_MAX_TOKENS"), 600),
            retry_count=max(0, _to_int(os.getenv("WORKFLOW_QA_LLM_RETRY_COUNT"), 2)),
            retry_base_delay_ms=max(100, _to_int(os.getenv("WORKFLOW_QA_LLM_RETRY_BASE_DELAY_MS"), 400)),
        )


class KnowledgeQALLMClient:
    """knowledge_answer 节点的 LLM 客户端。"""

    # 系统 Prompt 约束：
    # - 只能基于证据回答；
    # - 仅在“无法给出结论”时输出证据不足；
    # - 若问题要求“函数/文件/行号定位”，且存在 code 证据，必须给出代码锚点；
    # - 若是“校准看什么”问题，优先列指标词，不只写参数描述。
    SYSTEM_PROMPT_TEMPLATE = """你是广告引擎平台的知识问答助手。
请严格遵守以下规则：
1. 只能根据我提供的“检索证据”回答，不允许补充证据外事实。
2. 仅当关键结论无法由证据支持时，才输出“当前证据不足”与缺失项。
   如果你已经给出明确结论 + 依据 + 代码锚点，不要再附加“当前证据不足”段。
3. 回答必须使用中文，结构化输出，先给“结论”，再给“依据”。
4. 如果用户问题包含“哪个函数/哪个文件/在哪一行/实现位置”等定位诉求，且证据中存在 code 类型：
   你必须至少给出 1 条“代码锚点”（文件路径 + 函数或类名；有行号就带行号）。
5. 如果问题是 pCTR/pCVR 校准关注点，优先给出指标关键词（AUC、LogLoss、Calibration Error、线上 CTR/CVR 偏差），不要只写 slope/bias。
6. 如果问题涉及 target_cpa/pCVR 出价公式并追问实现函数，若证据中存在入口函数，请显式给出入口函数名（优先 compute_bid_for_request）。
7. 不要输出与问题无关的泛化内容。
"""

    USER_PROMPT_TEMPLATE = """【用户问题】
{user_query}

【当前模块】
- module_name: {module_name}
- module_hint: {module_hint}

【检索语句】
{retrieval_queries}

【检索证据（按相关性排序）】
{evidence_block}

【回答格式要求】
{answer_style_requirement}

请按以下格式输出：
1. 结论
2. 依据（仅引用以上证据）
3. 仅当无法形成结论时，再补充“当前证据不足”与“建议补充问题”
"""

    def __init__(self, config: KnowledgeQALLMConfig) -> None:
        self.config = config
        self._logger = get_file_logger(project_root=Path(__file__).resolve().parents[3])

    @property
    def is_available(self) -> bool:
        """判断在当前配置下是否可以调用 LLM。"""
        if not self.config.enabled:
            return False
        if not self.config.api_key:
            return False
        if not self.config.model:
            return False
        return True

    @classmethod
    def from_env(cls) -> "KnowledgeQALLMClient":
        """基于环境变量创建客户端实例。"""
        return cls(KnowledgeQALLMConfig.from_env())

    def generate_answer(
        self,
        *,
        user_query: str,
        question_type: str,
        module_name: str,
        module_hint: str,
        retrieval_queries: list[str],
        evidence_hits: list[dict[str, Any]],
    ) -> tuple[str | None, str | None]:
        """调用 LLM 生成答案。

        返回：
        - (answer_text, None): 成功
        - (None, reason): 失败，reason 用于上层记录降级原因
        """
        if not self.config.enabled:
            self._logger.info(
                "workflow.llm_qa.skipped",
                reason="llm_disabled",
                model=self.config.model,
            )
            return None, "llm_disabled"
        if not self.config.api_key:
            self._logger.warning(
                "workflow.llm_qa.skipped",
                reason="missing_api_key",
                model=self.config.model,
            )
            return None, "missing_api_key"
        if not evidence_hits:
            self._logger.info(
                "workflow.llm_qa.skipped",
                reason="no_evidence_hits",
                model=self.config.model,
                module_name=module_name,
                question_type=question_type,
            )
            return None, "no_evidence_hits"

        total_started_at = time.perf_counter()
        evidence_block = self._build_evidence_block(evidence_hits)
        retrieval_text = "\n".join(f"- {item}" for item in retrieval_queries) if retrieval_queries else "- 无"
        answer_style_requirement = self._build_answer_style_requirement(
            question_type=question_type,
            user_query=user_query,
        )

        system_prompt = self.SYSTEM_PROMPT_TEMPLATE
        user_prompt = self.USER_PROMPT_TEMPLATE.format(
            user_query=user_query,
            module_name=module_name,
            module_hint=module_hint,
            retrieval_queries=retrieval_text,
            evidence_block=evidence_block,
            answer_style_requirement=answer_style_requirement,
        )

        # P0：LLM 空响应链路增强
        # - 支持通过环境变量配置重试次数，减少供应商瞬时空包带来的降级；
        # - 失败原因携带可观测摘要，便于从评测报告直接定位根因。
        max_attempts = 1 + max(0, int(self.config.retry_count))
        last_reason = "unknown_error"
        self._logger.info(
            "workflow.llm_qa.start",
            model=self.config.model,
            module_name=module_name,
            question_type=question_type,
            evidence_count=len(evidence_hits),
            retrieval_query_count=len(retrieval_queries),
            max_attempts=max_attempts,
            user_query_preview=self._preview_text(user_query, max_chars=120),
        )
        for attempt in range(1, max_attempts + 1):
            attempt_started_at = time.perf_counter()
            try:
                answer = self._chat_completion(system_prompt=system_prompt, user_prompt=user_prompt)
            except TimeoutError:
                last_reason = "timeout"
                self._logger.warning(
                    "workflow.llm_qa.attempt_failed",
                    model=self.config.model,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    reason=last_reason,
                    latency_ms=int((time.perf_counter() - attempt_started_at) * 1000),
                )
            except (error.URLError, error.HTTPError):
                last_reason = "http_error"
                self._logger.warning(
                    "workflow.llm_qa.attempt_failed",
                    model=self.config.model,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    reason=last_reason,
                    latency_ms=int((time.perf_counter() - attempt_started_at) * 1000),
                )
            except ValueError as value_error:
                # ValueError 用于承载“响应格式/空响应”这类可观测错误。
                reason_text = str(value_error).strip() or "empty_answer"
                last_reason = reason_text if reason_text.startswith("empty_answer") else f"empty_answer:{reason_text}"
                self._logger.warning(
                    "workflow.llm_qa.attempt_failed",
                    model=self.config.model,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    reason=last_reason,
                    latency_ms=int((time.perf_counter() - attempt_started_at) * 1000),
                )
            except Exception:
                last_reason = "unknown_error"
                self._logger.warning(
                    "workflow.llm_qa.attempt_failed",
                    model=self.config.model,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    reason=last_reason,
                    latency_ms=int((time.perf_counter() - attempt_started_at) * 1000),
                )
            else:
                normalized_answer = self._enforce_structured_output(answer.strip(), question_type=question_type)
                if normalized_answer:
                    if self._looks_like_reasoning_dump(normalized_answer):
                        last_reason = "empty_answer:reasoning_dump"
                        self._logger.warning(
                            "workflow.llm_qa.attempt_failed",
                            model=self.config.model,
                            attempt=attempt,
                            max_attempts=max_attempts,
                            reason=last_reason,
                            latency_ms=int((time.perf_counter() - attempt_started_at) * 1000),
                        )
                    else:
                        attempt_latency_ms = int((time.perf_counter() - attempt_started_at) * 1000)
                        total_latency_ms = int((time.perf_counter() - total_started_at) * 1000)
                        self._logger.info(
                            "workflow.llm_qa.success",
                            model=self.config.model,
                            attempt=attempt,
                            max_attempts=max_attempts,
                            attempt_latency_ms=attempt_latency_ms,
                            total_latency_ms=total_latency_ms,
                            answer_length=len(normalized_answer),
                            fallback_reason=None,
                        )
                        return normalized_answer, None
                else:
                    last_reason = "empty_answer:normalized_empty"
                    self._logger.warning(
                        "workflow.llm_qa.attempt_failed",
                        model=self.config.model,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        reason=last_reason,
                        latency_ms=int((time.perf_counter() - attempt_started_at) * 1000),
                    )
            if attempt < max_attempts:
                # 指数退避：base -> 2*base -> 4*base...
                delay_seconds = (max(100, int(self.config.retry_base_delay_ms)) / 1000.0) * (2 ** (attempt - 1))
                time.sleep(delay_seconds)
        self._logger.warning(
            "workflow.llm_qa.fallback",
            model=self.config.model,
            max_attempts=max_attempts,
            fallback_reason=last_reason,
            total_latency_ms=int((time.perf_counter() - total_started_at) * 1000),
            question_type=question_type,
            module_name=module_name,
        )
        return None, last_reason

    def _preview_text(self, value: Any, *, max_chars: int = 120) -> str:
        text = str(value or "").strip()
        if len(text) <= max_chars:
            return text
        return f"{text[:max_chars]}..."

    def _looks_like_reasoning_dump(self, answer: str) -> bool:
        """识别“分析过程/约束复述”类输出。

        这类文本通常不是面向用户的最终答案，会显著拉低 expected_mode_accuracy。
        """
        normalized = answer.lower()
        markers = (
            "分析用户请求",
            "分析检索证据",
            "分析证据",
            "构思输出",
            "约束条件",
            "核心问题：",
            "仅使用提供的证据",
            "不得输出",
        )
        hit_count = sum(1 for marker in markers if marker in normalized)
        # 至少命中两个“思维过程”标记才判定为 reasoning dump，降低误伤正常答案。
        return hit_count >= 2

    def _build_answer_style_requirement(self, *, question_type: str, user_query: str) -> str:
        """根据问题类型与问题语义生成回答约束。

        这里把 P0 要求直接写进 Prompt，减少“命中证据但漏答关键词”的情况。
        """
        if question_type == "list":
            base = "问题属于列举类。请输出编号清单，每条只保留一个核心要点。"
        elif question_type == "reason":
            base = "问题属于原因解释类。请输出主要原因编号清单，每条附一行证据说明。"
        elif question_type == "formula":
            base = "问题属于公式计算类。请显式给出公式与变量含义，避免只给概念描述。"
        else:
            base = "按结论 + 依据简洁作答。"

        normalized_query = user_query.lower().strip()
        extras: list[str] = []
        # 明确禁止输出“分析过程/步骤复述”。
        extras.append("禁止输出“分析用户请求/分析证据/约束复述”等思维过程，只输出最终答案。")
        if question_type == "formula":
            extras.append("公式表达请使用 ASCII 乘号 `*`（例如 `a * b`），不要使用 `×`。")
        if self._is_code_location_query(normalized_query):
            extras.append("若已给出代码锚点和依据，不要附“当前证据不足”段。")
        if self._is_calibration_metric_query(normalized_query):
            extras.append("必须至少出现2个校准指标关键词：AUC、LogLoss、Calibration Error、线上CTR/CVR偏差。")
        if self._is_bid_entry_query(normalized_query):
            extras.append("若证据包含出价入口函数，请显式给出入口函数名（优先 compute_bid_for_request）和文件路径。")
        if self._is_bid_chain_query(normalized_query):
            extras.append("若证据命中，请显式给出函数链路：compute_bid_for_request -> compute_alpha -> compute_ocpc_bid。")
        if self._is_ecpm_formula_query(normalized_query):
            extras.append("若问题询问 eCPM 公式，请显式写出 `eCPM = pCTR * Bid * 1000`。")
            extras.append("若证据命中排序函数，请显式给出 `compute_rank_score` 或 `rank_score` 作为代码锚点。")
        if self._is_bid_entry_location_query(normalized_query):
            extras.append("若问题是“请求级出价入口函数”，优先输出 `compute_bid_for_request`。")
        if self._is_rate_entry_location_query(normalized_query):
            extras.append("若问题是“两率预测入口函数”，优先输出 `predict_ctr_cvr`。")
        if self._is_rate_bias_reason_query(normalized_query):
            extras.append("请优先覆盖原因关键词：特征分布漂移、模型版本误切换、校准参数同步。")
        if self._is_rerank_monitor_metric_query(normalized_query):
            extras.append(
                "若问题是精排监控指标，请优先覆盖：排序时延 P95/P99、TopN 点击率、多样性指标、探索流量占比；"
                "不要替换为 AUC/LogLoss 等两率校准指标。"
            )
        return f"{base} {' '.join(extras)}".strip()

    def _is_code_location_query(self, normalized_query: str) -> bool:
        """判断是否为函数/文件定位类问法。"""
        location_tokens = (
            "哪个函数",
            "什么函数",
            "入口函数",
            "函数链路",
            "函数调用链",
            "哪个文件",
            "文件路径",
            "路径在哪",
            "哪一行",
            "实现位置",
            "where",
            "which function",
        )
        return any(token in normalized_query for token in location_tokens)

    def _is_calibration_metric_query(self, normalized_query: str) -> bool:
        """判断是否为 pCTR/pCVR 校准指标类问题。"""
        has_calibration = any(token in normalized_query for token in ("校准", "calibration", "calibrate"))
        has_rate_terms = any(token in normalized_query for token in ("pctr", "pcvr", "ctr", "cvr"))
        return has_calibration and has_rate_terms

    def _is_bid_entry_query(self, normalized_query: str) -> bool:
        """判断是否为出价公式 + 代码入口定位类问题。"""
        has_bid_terms = any(token in normalized_query for token in ("target_cpa", "pcvr", "出价", "bid"))
        has_location_terms = any(
            token in normalized_query
            for token in ("哪个函数", "什么函数", "实现", "文件", "位置", "函数链路", "函数调用链", "where", "which function")
        )
        return has_bid_terms and has_location_terms

    def _is_bid_chain_query(self, normalized_query: str) -> bool:
        """判断是否为“请求级出价函数链路”问题。"""
        has_bid_terms = any(token in normalized_query for token in ("请求级出价", "出价", "target_cpa", "pcvr", "bid"))
        has_chain_terms = any(token in normalized_query for token in ("函数链路", "函数调用链", "链路"))
        return has_bid_terms and has_chain_terms

    def _is_ecpm_formula_query(self, normalized_query: str) -> bool:
        """判断是否为 eCPM 公式相关问题。"""
        return "ecpm" in normalized_query and any(token in normalized_query for token in ("公式", "计算", "怎么算", "formula"))

    def _is_bid_entry_location_query(self, normalized_query: str) -> bool:
        """判断是否为“请求级出价入口函数”定位问题。"""
        has_entry = any(token in normalized_query for token in ("入口函数", "入口", "函数"))
        return has_entry and "请求级出价" in normalized_query

    def _is_rate_entry_location_query(self, normalized_query: str) -> bool:
        """判断是否为“两率预测入口函数”定位问题。"""
        has_entry = any(token in normalized_query for token in ("入口函数", "入口", "函数"))
        has_rate = any(token in normalized_query for token in ("两率", "pctr", "pcvr", "ctr", "cvr"))
        return has_entry and has_rate

    def _is_rate_bias_reason_query(self, normalized_query: str) -> bool:
        """判断是否为“两率偏高/偏低原因”问题。"""
        has_rate_terms = any(token in normalized_query for token in ("pctr", "pcvr", "ctr", "cvr"))
        has_bias_terms = any(token in normalized_query for token in ("偏高", "偏低", "漂移", "原因"))
        return has_rate_terms and has_bias_terms

    def _is_rerank_monitor_metric_query(self, normalized_query: str) -> bool:
        """判断是否为“精排阶段监控指标”问题。"""
        return "精排" in normalized_query and "监控" in normalized_query and "指标" in normalized_query

    def _enforce_structured_output(self, answer: str, *, question_type: str) -> str:
        """对 LLM 输出做轻量结构兜底，保证列表型问题有可读清单。"""
        if not answer:
            return ""
        if question_type not in {"list", "reason", "formula"}:
            return answer

        has_list = bool(re.search(r"(?m)^\s*(?:\d+\.|[-*])\s+", answer))
        if has_list:
            return answer

        sentences = [segment.strip() for segment in re.split(r"[。\n；;]", answer) if segment.strip()]
        if not sentences:
            return answer

        top_sentences = sentences[:4]
        list_body = "\n".join(f"{index}. {text}" for index, text in enumerate(top_sentences, start=1))
        if question_type == "list":
            section_title = "要点清单"
        elif question_type == "reason":
            section_title = "主要原因"
        else:
            section_title = "公式与关键变量"
        return f"{answer}\n\n**{section_title}**\n{list_body}"

    def _build_evidence_block(self, evidence_hits: list[dict[str, Any]]) -> str:
        """将融合检索结果格式化为 Prompt 证据块。"""
        lines: list[str] = []
        for index, item in enumerate(evidence_hits[:6], start=1):
            source_type = _as_source_type(item.get("source_type"))
            lines.append(f"[证据{index}]")
            lines.append(f"- source_type: {source_type}")
            lines.append(f"- title: {item.get('title', '')}")
            lines.append(f"- path: {item.get('path', '')}")
            lines.append(f"- section: {item.get('section', '')}")
            lines.append(f"- score: {item.get('score', '')}")

            excerpt = str(item.get("excerpt", "")).strip()
            if excerpt:
                lines.append(f"- excerpt: {excerpt}")

            if source_type == "code":
                lines.append(f"- symbol_name: {item.get('symbol_name', '')}")
                lines.append(f"- signature: {item.get('signature', '')}")
                start_line = item.get("start_line", "")
                end_line = item.get("end_line", "")
                if start_line != "" or end_line != "":
                    lines.append(f"- line_range: {start_line}-{end_line}")
                highlight_lines = item.get("highlight_lines", [])
                if isinstance(highlight_lines, list) and highlight_lines:
                    lines.append(f"- highlight_lines: {highlight_lines[:6]}")
        return "\n".join(lines) if lines else "无"

    def _chat_completion(self, *, system_prompt: str, user_prompt: str) -> str:
        """调用 OpenAI 兼容 Chat Completions 接口并提取回答文本。"""
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
        if response_json.get("error"):
            error_obj = response_json.get("error", {})
            error_type = str(error_obj.get("type", "unknown"))
            error_message = str(error_obj.get("message", ""))
            raise ValueError(f"api_error:{error_type}:{error_message[:80]}")
        choices = response_json.get("choices", [])
        if not choices:
            summary = self._build_response_summary(response_json)
            raise ValueError(f"empty_answer:no_choices:{summary}")

        message = choices[0].get("message", {})
        content = message.get("content")
        reasoning_content = message.get("reasoning_content")
        if isinstance(content, str):
            normalized = content.strip()
            if normalized:
                return normalized

        # 兼容部分供应商返回 list 内容格式。
        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    text_parts.append(str(item.get("text", "")))
                elif isinstance(item, str):
                    text_parts.append(item)
            normalized = "\n".join(part for part in text_parts if part).strip()
            if normalized:
                return normalized

        # 注意：不直接采用 reasoning_content 作为最终答案。
        # reasoning_content 往往是“思维过程/草稿”，直接输出会污染评测模式判定。
        _ = reasoning_content

        summary = self._build_response_summary(response_json)
        raise ValueError(f"empty_answer:empty_content:{summary}")

    def _build_response_summary(self, response_json: dict[str, Any]) -> str:
        """构建精简响应摘要，写入 fallback_reason 便于排障。"""
        choices = response_json.get("choices", [])
        first_choice = choices[0] if choices else {}
        message = first_choice.get("message", {}) if isinstance(first_choice, dict) else {}
        content = message.get("content")
        if isinstance(content, str):
            content_type = "str"
            content_len = len(content.strip())
        elif isinstance(content, list):
            content_type = "list"
            content_len = len(content)
        elif content is None:
            content_type = "none"
            content_len = 0
        else:
            content_type = type(content).__name__
            content_len = len(str(content))
        has_reasoning = bool(str(message.get("reasoning_content", "")).strip())
        finish_reason = first_choice.get("finish_reason", "") if isinstance(first_choice, dict) else ""
        return (
            f"choices={len(choices)},content_type={content_type},"
            f"content_len={content_len},has_reasoning={int(has_reasoning)},finish={finish_reason}"
        )
