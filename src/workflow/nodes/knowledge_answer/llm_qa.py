from __future__ import annotations

"""LLM QA client for knowledge_answer node.

This module uses LangChain wrappers instead of hand-written HTTP calls:
- `ChatPromptTemplate` for prompt assembly.
- `ChatOpenAI` for OpenAI-compatible chat completion calls.

The external contract remains unchanged:
- `KnowledgeQALLMClient.generate_answer(...) -> tuple[str | None, str | None]`
- Returns `(answer, None)` on success, `(None, reason)` on fallback.
"""

from dataclasses import dataclass
import os
from pathlib import Path
import re
import time
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from workflow.domain_profile import DomainProfile
from workflow.runtime_logging import get_file_logger
from workflow.utils import normalize_source_type, to_bool, to_float, to_int


@dataclass
class KnowledgeQALLMConfig:
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
        return cls(
            enabled=to_bool(os.getenv("WORKFLOW_QA_LLM_ENABLED"), True),
            base_url=os.getenv("WORKFLOW_QA_LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            api_key=os.getenv("WORKFLOW_QA_LLM_API_KEY", "").strip(),
            model=os.getenv("WORKFLOW_QA_LLM_MODEL", "gpt-4.1-mini").strip(),
            timeout_seconds=to_int(os.getenv("WORKFLOW_QA_LLM_TIMEOUT_SECONDS"), 20),
            temperature=to_float(os.getenv("WORKFLOW_QA_LLM_TEMPERATURE"), 0.2),
            max_tokens=to_int(os.getenv("WORKFLOW_QA_LLM_MAX_TOKENS"), 600),
            retry_count=max(0, to_int(os.getenv("WORKFLOW_QA_LLM_RETRY_COUNT"), 2)),
            retry_base_delay_ms=max(100, to_int(os.getenv("WORKFLOW_QA_LLM_RETRY_BASE_DELAY_MS"), 400)),
        )


class KnowledgeQALLMClient:
    """LLM client used by `knowledge_answer` node."""

    SYSTEM_PROMPT_TEMPLATE = """你是企业知识问答助手，请严格遵守：
1. 只能基于提供的检索证据回答，不补充证据外事实。
2. 只有在无法形成结论时，才输出“当前证据不足”。
3. 输出中文，结构化为：结论 -> 依据。
4. 若问题要求函数/文件定位且有 code 证据，至少给 1 个代码锚点（路径+符号，尽量带行号）。
5. 不输出与问题无关的泛化内容。"""

    USER_PROMPT_TEMPLATE = """【用户问题】
{user_query}

【当前模块】
- module_name: {module_name}
- module_hint: {module_hint}

【检索语句】
{retrieval_queries}

【检索证据（按相关性排序）】
{evidence_block}

【回答要求】
{answer_style_requirement}

请按以下格式输出：
1. 结论
2. 依据（仅引用上述证据）
3. 仅当无法形成结论时，再补充“当前证据不足”与“建议补充问题”。"""

    def __init__(self, config: KnowledgeQALLMConfig, *, domain_profile: DomainProfile | None = None) -> None:
        self.config = config
        self.domain_profile = domain_profile
        self._answering_profile = domain_profile.answering if domain_profile is not None else None
        self._system_prompt = self._resolve_system_prompt(domain_profile)
        self._logger = get_file_logger(project_root=Path(__file__).resolve().parents[4])
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "{system_prompt}"),
                ("human", "{user_prompt}"),
            ]
        )
        self._chat_model: ChatOpenAI | None = None
        self._init_error: str | None = None

        if self.config.enabled and self.config.api_key and self.config.model:
            try:
                self._chat_model = ChatOpenAI(
                    model=self.config.model,
                    api_key=self.config.api_key,
                    base_url=self.config.base_url,
                    timeout=self.config.timeout_seconds,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
            except Exception as exc:
                self._init_error = f"chat_model_init_failed:{type(exc).__name__}"
                self._logger.warning(
                    "workflow.llm_qa.init_failed",
                    model=self.config.model,
                    error_type=type(exc).__name__,
                )

    @property
    def is_available(self) -> bool:
        if not self.config.enabled:
            return False
        if not self.config.api_key:
            return False
        if not self.config.model:
            return False
        return self._chat_model is not None

    @classmethod
    def from_env(cls, *, domain_profile: DomainProfile | None = None) -> "KnowledgeQALLMClient":
        return cls(KnowledgeQALLMConfig.from_env(), domain_profile=domain_profile)

    def _resolve_system_prompt(self, domain_profile: DomainProfile | None) -> str:
        env_prompt = os.getenv("WORKFLOW_QA_LLM_SYSTEM_PROMPT", "").strip()
        if env_prompt:
            return env_prompt
        if domain_profile is not None:
            profile_prompt = domain_profile.system_prompt()
            if profile_prompt:
                return profile_prompt
        return self.SYSTEM_PROMPT_TEMPLATE

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
        if not self.config.enabled:
            self._logger.info("workflow.llm_qa.skipped", reason="llm_disabled", model=self.config.model)
            return None, "llm_disabled"
        if not self.config.api_key:
            self._logger.warning("workflow.llm_qa.skipped", reason="missing_api_key", model=self.config.model)
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
        if self._chat_model is None:
            reason = self._init_error or "chat_model_unavailable"
            self._logger.warning("workflow.llm_qa.skipped", reason=reason, model=self.config.model)
            return None, reason

        total_started_at = time.perf_counter()
        evidence_block = self._build_evidence_block(evidence_hits)
        retrieval_text = "\n".join(f"- {item}" for item in retrieval_queries) if retrieval_queries else "- 无"
        answer_style_requirement = self._build_answer_style_requirement(
            question_type=question_type,
            user_query=user_query,
        )

        system_prompt = self._system_prompt
        user_prompt = self.USER_PROMPT_TEMPLATE.format(
            user_query=user_query,
            module_name=module_name,
            module_hint=module_hint,
            retrieval_queries=retrieval_text,
            evidence_block=evidence_block,
            answer_style_requirement=answer_style_requirement,
        )

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
            except ValueError as value_error:
                reason_text = str(value_error).strip() or "empty_answer"
                last_reason = reason_text if reason_text.startswith("empty_answer") else f"empty_answer:{reason_text}"
            except Exception as exc:
                last_reason = self._map_exception_to_reason(exc)
            else:
                normalized_answer = self._enforce_structured_output(answer.strip(), question_type=question_type)
                if normalized_answer:
                    if self._looks_like_reasoning_dump(normalized_answer):
                        last_reason = "empty_answer:reasoning_dump"
                    else:
                        self._logger.info(
                            "workflow.llm_qa.success",
                            model=self.config.model,
                            module_name=module_name,
                            question_type=question_type,
                            attempts=attempt,
                            latency_ms=int((time.perf_counter() - total_started_at) * 1000),
                            answer_length=len(normalized_answer),
                        )
                        return normalized_answer, None
                else:
                    last_reason = "empty_answer"

            self._logger.warning(
                "workflow.llm_qa.attempt_failed",
                model=self.config.model,
                attempt=attempt,
                max_attempts=max_attempts,
                reason=last_reason,
                latency_ms=int((time.perf_counter() - attempt_started_at) * 1000),
            )
            if attempt < max_attempts:
                delay_ms = self.config.retry_base_delay_ms * (2 ** (attempt - 1))
                time.sleep(delay_ms / 1000.0)

        self._logger.warning(
            "workflow.llm_qa.fallback",
            model=self.config.model,
            module_name=module_name,
            question_type=question_type,
            fallback_reason=last_reason,
            latency_ms=int((time.perf_counter() - total_started_at) * 1000),
        )
        return None, last_reason

    def _preview_text(self, value: Any, *, max_chars: int = 80) -> str:
        text = str(value or "").strip()
        if len(text) <= max_chars:
            return text
        return f"{text[:max_chars]}..."

    def _build_answer_style_requirement(self, *, question_type: str, user_query: str) -> str:
        base = "按结论 + 依据简洁作答。"
        extras: list[str] = []
        normalized_query = user_query.lower().strip()

        if question_type == "list":
            extras.append("优先给 3-6 条要点，使用编号列表。")
        elif question_type == "reason":
            extras.append("优先列主要原因，再给对应证据。")
        elif question_type == "formula":
            extras.append("给出公式与变量含义，确保符号和字段名一致。")

        if self._is_code_location_query(normalized_query):
            extras.append("必须至少给 1 个代码锚点（路径 + 函数/类名；若有行号请带行号）。")
            extras.append("若已给出代码锚点和依据，不要附“当前证据不足”段。")

        if self._is_calibration_query(normalized_query):
            extras.append("必须至少出现2个校准指标关键词：AUC、LogLoss、Calibration Error、线上CTR/CVR偏差。")

        if self._is_bid_entry_query(normalized_query):
            entry_symbol = self._default_entry_symbol()
            if entry_symbol:
                extras.append(f"若证据包含入口函数，显式给出入口函数名（优先 {entry_symbol}）。")

        if self._is_reason_query(normalized_query):
            extras.append("优先覆盖原因关键词：特征分布漂移、模型版本误切换、校准参数同步。")

        return " ".join([base, *extras]).strip()

    def _is_code_location_query(self, normalized_query: str) -> bool:
        keywords = (
            "哪个函数",
            "什么函数",
            "入口函数",
            "函数链路",
            "函数调用链",
            "哪个文件",
            "文件路径",
            "哪一行",
            "实现位置",
            "代码位置",
            "where",
            "which function",
            "locate",
        )
        return any(token in normalized_query for token in keywords)

    def _is_calibration_query(self, normalized_query: str) -> bool:
        if self._answering_profile is not None and self._answering_profile.calibration_terms:
            return any(token in normalized_query for token in self._answering_profile.calibration_terms)
        return any(token in normalized_query for token in ("校准", "pctr", "pcvr", "ctr", "cvr"))

    def _is_bid_entry_query(self, normalized_query: str) -> bool:
        if self._answering_profile is not None:
            bid_terms = self._answering_profile.bid_terms or ("出价", "计费", "报价", "pricing", "bid")
            entry_terms = self._answering_profile.bid_entry_terms or ("入口", "函数", "实现")
        else:
            bid_terms = ("出价", "计费", "报价", "pricing", "bid")
            entry_terms = ("入口", "函数", "实现")
        has_bid = any(token in normalized_query for token in bid_terms)
        has_entry = any(token in normalized_query for token in entry_terms)
        return has_bid and has_entry

    def _default_entry_symbol(self) -> str:
        if self._answering_profile is None:
            return "main_entry"
        return self._answering_profile.default_entry_symbol or "main_entry"

    def _is_reason_query(self, normalized_query: str) -> bool:
        return any(token in normalized_query for token in ("原因", "为什么", "为何", "导致", "怎么会", "为啥"))

    def _looks_like_reasoning_dump(self, answer: str) -> bool:
        lowered = answer.lower()
        markers = (
            "chain of thought",
            "reasoning",
            "思考过程",
            "推理过程",
            "analysis:",
        )
        return any(marker in lowered for marker in markers)

    def _enforce_structured_output(self, answer: str, *, question_type: str) -> str:
        if not answer:
            return ""
        if question_type not in {"list", "reason", "formula"}:
            return answer

        has_list = bool(re.search(r"(?m)^\s*(?:\d+\.|[-*])\s+", answer))
        if has_list:
            return answer

        sentences = [segment.strip() for segment in re.split(r"[。\n；;!?！？]", answer) if segment.strip()]
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
        lines: list[str] = []
        for index, item in enumerate(evidence_hits[:6], start=1):
            source_type = normalize_source_type(item.get("source_type"))
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
        if self._chat_model is None:
            raise ValueError(self._init_error or "chat_model_unavailable")

        chain = self._prompt | self._chat_model
        try:
            message = chain.invoke(
                {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                }
            )
        except Exception as exc:
            if self._is_timeout_exception(exc):
                raise TimeoutError("llm_timeout") from exc
            raise

        content = self._extract_text_from_message(message)
        if content:
            return content

        summary = self._build_message_summary(message)
        raise ValueError(f"empty_answer:empty_content:{summary}")

    def _extract_text_from_message(self, message: Any) -> str:
        content = getattr(message, "content", None)
        if isinstance(content, str):
            normalized = content.strip()
            if normalized:
                return normalized
        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    value = str(item.get("text", "") or "").strip()
                    if value:
                        text_parts.append(value)
                elif isinstance(item, str):
                    value = item.strip()
                    if value:
                        text_parts.append(value)
            normalized = "\n".join(text_parts).strip()
            if normalized:
                return normalized
        return ""

    def _build_message_summary(self, message: Any) -> str:
        if not isinstance(message, BaseMessage):
            return f"message_type={type(message).__name__}"

        content = message.content
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

        finish_reason = ""
        response_meta = getattr(message, "response_metadata", {}) or {}
        if isinstance(response_meta, dict):
            finish_reason = str(response_meta.get("finish_reason", "") or "")
        return f"content_type={content_type},content_len={content_len},finish={finish_reason}"

    def _is_timeout_exception(self, exc: Exception) -> bool:
        timeout_markers = ("timeout", "timed out", "read timed out", "request timed out")
        text = f"{type(exc).__name__}:{exc}".lower()
        return any(marker in text for marker in timeout_markers)

    def _map_exception_to_reason(self, exc: Exception) -> str:
        if self._is_timeout_exception(exc):
            return "timeout"
        name = type(exc).__name__.lower()
        if "rate" in name and "limit" in name:
            return "rate_limit"
        if "auth" in name or "permission" in name:
            return "auth_error"
        if "api" in name or "http" in name or "connection" in name:
            return "http_error"
        return "unknown_error"
