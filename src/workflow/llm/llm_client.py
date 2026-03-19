# -*- coding: utf-8 -*-
"""Shared LLM client and invocation capability for workflow nodes."""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import re
import time
from typing import Any, Callable

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from workflow.common.runtime_logging import get_file_logger
from workflow.common.func_utils import to_bool, to_float, to_int

# LLM 日志相关常量：用于控制 prompt 截断长度
LLM_LOG_PROMPT_MAX_LENGTH = 500  # 单个 prompt 最大日志长度
LLM_LOG_RESPONSE_MAX_LENGTH = 1000  # 响应文本最大日志长度

# 用于匹配 LLM thinking 内容的正则表达式（支持 <thinking> 和 <thinker> 等变体）
_THINKING_PATTERN = re.compile(
    r"<(?:thinking|thinker|think)>[\s\S]*?</(?:thinking|thinker|think)>",
    re.IGNORECASE
)


def strip_thinking_content(text: str) -> str:
    """
    移除 LLM 响应中的 thinking 标签及其内容。

    部分 LLM（如 DeepSeek-R1、Claude 等）会返回 <thinking>...</thinking> 格式的思考过程，
    该函数用于在返回给用户界面之前过滤掉这部分内容。

    Args:
        text: 原始响应文本，可能包含 thinking 标签

    Returns:
        移除 thinking 内容后的文本
    """
    if not text:
        return text
    # 使用正则表达式移除所有 thinking 标签及其内容
    cleaned = _THINKING_PATTERN.sub("", text)
    # 清理可能产生的多余空行，保持格式整洁
    return cleaned.strip()


AnswerNormalizer = Callable[[str], str]
AnswerValidator = Callable[[str], tuple[bool, str | None]]


@dataclass(frozen=True)
class CommonLLMConfig:
    """Runtime config for shared LLM invocations."""

    enabled: bool
    base_url: str
    api_key: str
    model: str
    timeout_seconds: int
    temperature: float
    max_tokens: int
    retry_count: int
    retry_base_delay_ms: int
    debug_verbose: bool = False

    @classmethod
    def from_env(cls, *, prefix: str = "WORKFLOW_QA_LLM") -> "CommonLLMConfig":
        env = lambda key: f"{prefix}_{key}"
        return cls(
            enabled=to_bool(os.getenv(env("ENABLED")), True),
            base_url=os.getenv(env("BASE_URL"), "https://api.openai.com/v1").rstrip("/"),
            api_key=os.getenv(env("API_KEY"), "").strip(),
            model=os.getenv(env("MODEL"), "gpt-4.1-mini").strip(),
            timeout_seconds=to_int(os.getenv(env("TIMEOUT_SECONDS")), 20),
            temperature=to_float(os.getenv(env("TEMPERATURE")), 0.2),
            max_tokens=to_int(os.getenv(env("MAX_TOKENS")), 600),
            retry_count=max(0, to_int(os.getenv(env("RETRY_COUNT")), 2)),
            retry_base_delay_ms=max(100, to_int(os.getenv(env("RETRY_BASE_DELAY_MS")), 400)),
            debug_verbose=to_bool(os.getenv("WORKFLOW_DEBUG_VERBOSE"), False),
        )


@dataclass(frozen=True)
class CommonLLMRequest:
    """LLM request payload passed from a node-level wrapper.

    Attributes:
        node_name: 调用节点名称
        system_prompt: 系统提示词
        user_prompt: 用户提示词
        evidence_count: 证据数量
        require_evidence: 是否需要证据
        log_namespace: 日志命名空间
        metadata: 元数据
        normalize_answer: 答案标准化函数
        validate_answer: 答案验证函数
        tools: OpenAI 格式的工具 schema 列表（可选）
        tool_choice: 工具选择策略（"auto" | "none" | "required" | dict）
    """

    node_name: str
    system_prompt: str
    user_prompt: str
    evidence_count: int = 0
    require_evidence: bool = True
    log_namespace: str = "workflow.llm"
    metadata: dict[str, Any] = field(default_factory=dict)
    normalize_answer: AnswerNormalizer | None = None
    validate_answer: AnswerValidator | None = None
    # Tool use 支持
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None  # "auto" | "none" | "required" | {"type": "function", "function": {"name": "xxx"}}


@dataclass(frozen=True)
class CommonLLMResult:
    """LLM invocation result.

    Attributes:
        answer: LLM 生成的答案文本（无 tool_call 时）
        fallback_reason: 降级原因（如果调用失败或跳过）
        call_status: 调用状态信息
        tool_calls: LLM 返回的工具调用列表（如果有）
            格式: [{"name": "xxx", "args": {...}, "id": "..."}]
    """

    answer: str | None
    fallback_reason: str | None
    call_status: dict[str, Any]
    tool_calls: list[dict[str, Any]] | None = None


class WorkflowLLMClient:
    """Workflow-level LLM client with retry, timeout mapping and status tracing."""

    def __init__(self, config: CommonLLMConfig) -> None:
        self.config = config
        self._logger = get_file_logger(project_root=Path(__file__).resolve().parents[3])
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "{system_prompt}"),
                ("human", "{user_prompt}"),
            ]
        )
        self._chat_model: ChatOpenAI | None = None
        self._init_error: str | None = None
        self.last_call_status: dict[str, Any] = self._build_status(
            status="not_called",
            invoked=False,
            request_sent=False,
            attempts=0,
            latency_ms=0,
            reason=None,
        )
        self._init_model()

    @property
    def is_available(self) -> bool:
        if not self.config.enabled:
            return False
        if not self.config.api_key:
            return False
        if not self.config.model:
            return False
        return self._chat_model is not None

    def _generate_result(self, request: CommonLLMRequest) -> CommonLLMResult:
        started_status = self._build_status(
            status="started",
            invoked=True,
            request_sent=False,
            attempts=0,
            latency_ms=0,
            reason=None,
        )
        self.last_call_status = dict(started_status)
        if not self.config.enabled:
            return self._skip(request, "llm_disabled")
        if not self.config.api_key:
            return self._skip(request, "missing_api_key")
        if request.require_evidence and request.evidence_count <= 0:
            return self._skip(request, "no_evidence_hits")
        if self._chat_model is None:
            return self._skip(request, self._init_error or "chat_model_unavailable")

        started_at = time.perf_counter()
        max_attempts = 1 + max(0, int(self.config.retry_count))
        self._logger.info(
            f"{request.log_namespace}.start",
            **self._event_payload(
                request,
                {
                    "model": self.config.model,
                    "evidence_count": request.evidence_count,
                    "max_attempts": max_attempts,
                },
            ),
        )

        last_reason = "unknown_error"
        for attempt in range(1, max_attempts + 1):
            attempt_started = time.perf_counter()
            # 记录详细的 LLM 请求日志
            self._log_llm_request(
                request,
                call_type="generate",
                attempt=attempt,
                max_attempts=max_attempts,
            )
            # 保留原有的 debug 日志（仅在 debug_verbose 时生效）
            self._log_debug_request(
                request,
                attempt=attempt,
                max_attempts=max_attempts,
            )
            try:
                answer = self._chat_completion(
                    system_prompt=request.system_prompt,
                    user_prompt=request.user_prompt,
                )
                attempt_latency_ms = int((time.perf_counter() - attempt_started) * 1000)
                # 记录成功的 LLM 响应日志
                self._log_llm_response(
                    request,
                    call_type="generate",
                    attempt=attempt,
                    latency_ms=attempt_latency_ms,
                    response_text=answer,
                )
            except TimeoutError as exc:
                attempt_latency_ms = int((time.perf_counter() - attempt_started) * 1000)
                last_reason = "timeout"
                # 记录失败的 LLM 响应日志
                self._log_llm_response(
                    request,
                    call_type="generate",
                    attempt=attempt,
                    latency_ms=attempt_latency_ms,
                    reason=last_reason,
                    error=exc,
                )
                self._log_debug_response(
                    request,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    latency_ms=attempt_latency_ms,
                    response_text=None,
                    reason=last_reason,
                    error=exc,
                )
            except ValueError as value_error:
                attempt_latency_ms = int((time.perf_counter() - attempt_started) * 1000)
                reason_text = str(value_error).strip() or "empty_answer"
                last_reason = reason_text if reason_text.startswith("empty_answer") else f"empty_answer:{reason_text}"
                # 记录失败的 LLM 响应日志
                self._log_llm_response(
                    request,
                    call_type="generate",
                    attempt=attempt,
                    latency_ms=attempt_latency_ms,
                    reason=last_reason,
                    error=value_error,
                )
                self._log_debug_response(
                    request,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    latency_ms=attempt_latency_ms,
                    response_text=None,
                    reason=last_reason,
                    error=value_error,
                )
            except Exception as exc:  # pragma: no cover - defensive fallback
                attempt_latency_ms = int((time.perf_counter() - attempt_started) * 1000)
                last_reason = self._map_exception_to_reason(exc)
                # 记录失败的 LLM 响应日志
                self._log_llm_response(
                    request,
                    call_type="generate",
                    attempt=attempt,
                    latency_ms=attempt_latency_ms,
                    reason=last_reason,
                    error=exc,
                )
                self._log_debug_response(
                    request,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    latency_ms=attempt_latency_ms,
                    response_text=None,
                    reason=last_reason,
                    error=exc,
                )
            else:
                # 保留原有的 debug 日志（仅在 debug_verbose 时生效）
                self._log_debug_response(
                    request,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    latency_ms=int((time.perf_counter() - attempt_started) * 1000),
                    response_text=answer,
                    reason=None,
                    error=None,
                )
                normalized = answer.strip()
                if request.normalize_answer is not None:
                    normalized = request.normalize_answer(normalized)
                if normalized:
                    if request.validate_answer is not None:
                        passed, reason = request.validate_answer(normalized)
                        if not passed:
                            last_reason = reason or "invalid_answer"
                        else:
                            success_status = self._mark_success(
                                request,
                                attempt=attempt,
                                started_at=started_at,
                                answer=normalized,
                            )
                            return CommonLLMResult(
                                answer=normalized,
                                fallback_reason=None,
                                call_status=success_status,
                            )
                    else:
                        success_status = self._mark_success(
                            request,
                            attempt=attempt,
                            started_at=started_at,
                            answer=normalized,
                        )
                        return CommonLLMResult(
                            answer=normalized,
                            fallback_reason=None,
                            call_status=success_status,
                        )
                else:
                    last_reason = "empty_answer"

            self._logger.warning(
                f"{request.log_namespace}.attempt_failed",
                **self._event_payload(
                    request,
                    {
                        "model": self.config.model,
                        "attempt": attempt,
                        "max_attempts": max_attempts,
                        "reason": last_reason,
                        "latency_ms": int((time.perf_counter() - attempt_started) * 1000),
                    },
                ),
            )
            if attempt < max_attempts:
                delay_ms = self.config.retry_base_delay_ms * (2 ** (attempt - 1))
                time.sleep(delay_ms / 1000.0)

        latency_ms = int((time.perf_counter() - started_at) * 1000)
        self._logger.warning(
            f"{request.log_namespace}.fallback",
            **self._event_payload(
                request,
                {
                    "model": self.config.model,
                    "fallback_reason": last_reason,
                    "latency_ms": latency_ms,
                },
            ),
        )
        fallback_status = self._build_status(
            status="fallback",
            invoked=True,
            request_sent=True,
            attempts=max_attempts,
            latency_ms=latency_ms,
            reason=last_reason,
        )
        self.last_call_status = dict(fallback_status)
        return CommonLLMResult(
            answer=None,
            fallback_reason=last_reason,
            call_status=fallback_status,
        )

    def _init_model(self) -> None:
        if not (self.config.enabled and self.config.api_key and self.config.model):
            return
        try:
            self._chat_model = ChatOpenAI(
                model=self.config.model,
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout_seconds,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            self._init_error = f"chat_model_init_failed:{type(exc).__name__}"
            self._logger.warning(
                "workflow.llm.init_failed",
                model=self.config.model,
                error_type=type(exc).__name__,
            )

    def _skip(self, request: CommonLLMRequest, reason: str) -> CommonLLMResult:
        self._logger.info(
            f"{request.log_namespace}.skipped",
            **self._event_payload(
                request,
                {
                    "model": self.config.model,
                    "reason": reason,
                },
            ),
        )
        skipped_status = self._build_status(
            status="skipped",
            invoked=True,
            request_sent=False,
            attempts=0,
            latency_ms=0,
            reason=reason,
        )
        self.last_call_status = dict(skipped_status)
        return CommonLLMResult(
            answer=None,
            fallback_reason=reason,
            call_status=skipped_status,
        )

    def _mark_success(
        self,
        request: CommonLLMRequest,
        *,
        attempt: int,
        started_at: float,
        answer: str,
    ) -> dict[str, Any]:
        latency_ms = int((time.perf_counter() - started_at) * 1000)
        self._logger.info(
            f"{request.log_namespace}.success",
            **self._event_payload(
                request,
                {
                    "model": self.config.model,
                    "attempts": attempt,
                    "latency_ms": latency_ms,
                    "answer_length": len(answer),
                },
            ),
        )
        success_status = self._build_status(
            status="success",
            invoked=True,
            request_sent=True,
            attempts=attempt,
            latency_ms=latency_ms,
            reason=None,
        )
        self.last_call_status = dict(success_status)
        return success_status

    def _log_debug_request(
        self,
        request: CommonLLMRequest,
        *,
        attempt: int,
        max_attempts: int,
    ) -> None:
        if not self.config.debug_verbose:
            return
        self._logger.info(
            f"{request.log_namespace}.debug_request",
            **self._event_payload(
                request,
                {
                    "model": self.config.model,
                    "base_url": self.config.base_url,
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                    "timeout_seconds": self.config.timeout_seconds,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                    "evidence_count": request.evidence_count,
                    "system_prompt": request.system_prompt,
                    "user_prompt": request.user_prompt,
                },
            ),
        )

    def _log_debug_response(
        self,
        request: CommonLLMRequest,
        *,
        attempt: int,
        max_attempts: int,
        latency_ms: int,
        response_text: str | None,
        reason: str | None,
        error: Exception | None,
    ) -> None:
        if not self.config.debug_verbose:
            return
        payload: dict[str, Any] = {
            "model": self.config.model,
            "attempt": attempt,
            "max_attempts": max_attempts,
            "latency_ms": int(max(latency_ms, 0)),
            "reason": reason,
            "response_text": response_text or "",
            "response_length": len((response_text or "").strip()),
        }
        if error is not None:
            payload["error_type"] = type(error).__name__
            payload["error_message"] = str(error)
        self._logger.info(
            f"{request.log_namespace}.debug_response",
            **self._event_payload(request, payload),
        )

    def _event_payload(self, request: CommonLLMRequest, payload: dict[str, Any]) -> dict[str, Any]:
        merged: dict[str, Any] = {"node_name": request.node_name}
        merged.update(payload)
        for key, value in request.metadata.items():
            if key not in merged:
                merged[key] = value
        return merged

    def _truncate_text(self, text: str | None, max_length: int = LLM_LOG_PROMPT_MAX_LENGTH) -> str:
        """截断长文本以适应日志输出

        Args:
            text: 待截断的文本
            max_length: 最大长度

        Returns:
            截断后的文本，末尾添加 "...[truncated, total=N]" 标记
        """
        if not text:
            return ""
        if len(text) <= max_length:
            return text
        return f"{text[:max_length]}...[truncated, total={len(text)}]"

    def _log_llm_request(
        self,
        request: CommonLLMRequest,
        *,
        call_type: str = "generate",
        attempt: int = 1,
        max_attempts: int = 1,
    ) -> None:
        """记录 LLM 请求详情到日志

        Args:
            request: LLM 请求对象
            call_type: 调用类型（generate, tools, agent）
            attempt: 当前尝试次数
            max_attempts: 最大尝试次数
        """
        payload = {
            "call_type": call_type,
            "model": self.config.model,
            "base_url": self.config.base_url,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "timeout_seconds": self.config.timeout_seconds,
            "attempt": attempt,
            "max_attempts": max_attempts,
            "evidence_count": request.evidence_count,
            "system_prompt_length": len(request.system_prompt) if request.system_prompt else 0,
            "user_prompt_length": len(request.user_prompt) if request.user_prompt else 0,
            # 截断后的 prompt 内容，便于调试
            "system_prompt_preview": self._truncate_text(request.system_prompt),
            "user_prompt_preview": self._truncate_text(request.user_prompt),
        }

        # 如果有 tools，记录工具信息
        if request.tools:
            payload["tools_count"] = len(request.tools)
            payload["tool_names"] = [
                t.get("function", {}).get("name", "unknown") if isinstance(t, dict) else str(t)
                for t in request.tools
            ]
            payload["tool_choice"] = request.tool_choice

        self._logger.info(
            f"{request.log_namespace}.llm_request",
            **self._event_payload(request, payload),
        )

    def _log_llm_response(
        self,
        request: CommonLLMRequest,
        *,
        call_type: str = "generate",
        attempt: int = 1,
        latency_ms: int = 0,
        response_text: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        error: Exception | None = None,
        reason: str | None = None,
    ) -> None:
        """记录 LLM 响应详情到日志

        Args:
            request: LLM 请求对象
            call_type: 调用类型（generate, tools, agent）
            attempt: 当前尝试次数
            latency_ms: 响应延迟（毫秒）
            response_text: 响应文本
            tool_calls: 工具调用列表
            error: 异常信息
            reason: 失败原因
        """
        payload: dict[str, Any] = {
            "call_type": call_type,
            "model": self.config.model,
            "attempt": attempt,
            "latency_ms": latency_ms,
            "success": error is None and reason is None,
        }

        # 响应文本信息
        if response_text:
            payload["response_length"] = len(response_text)
            payload["response_preview"] = self._truncate_text(response_text, LLM_LOG_RESPONSE_MAX_LENGTH)
        else:
            payload["response_length"] = 0
            payload["response_preview"] = ""

        # 工具调用信息
        if tool_calls:
            payload["tool_calls_count"] = len(tool_calls)
            payload["tool_calls"] = [
                {
                    "name": tc.get("name", ""),
                    "args_preview": self._truncate_text(str(tc.get("args", {})), 200),
                }
                for tc in tool_calls
            ]

        # 错误信息
        if error:
            payload["error_type"] = type(error).__name__
            payload["error_message"] = self._truncate_text(str(error), 500)

        if reason:
            payload["reason"] = reason

        log_level = "info" if error is None and reason is None else "warning"
        getattr(self._logger, log_level)(
            f"{request.log_namespace}.llm_response",
            **self._event_payload(request, payload),
        )

    def _build_status(
        self,
        *,
        status: str,
        invoked: bool,
        request_sent: bool,
        attempts: int,
        latency_ms: int,
        reason: str | None,
    ) -> dict[str, Any]:
        return {
            "status": status,
            "invoked": bool(invoked),
            "request_sent": bool(request_sent),
            "attempts": int(max(attempts, 0)),
            "latency_ms": int(max(latency_ms, 0)),
            "reason": reason,
            "model": self.config.model,
        }

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
        """从 LLM 响应消息中提取文本内容，并自动过滤 thinking 标签。"""
        content = getattr(message, "content", None)
        if isinstance(content, str):
            normalized = content.strip()
            if normalized:
                # 过滤 thinking 内容后再返回
                return strip_thinking_content(normalized)
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
                # 过滤 thinking 内容后再返回
                return strip_thinking_content(normalized)
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

    @classmethod
    def from_env(cls, *, prefix: str = "WORKFLOW_QA_LLM", **_: Any) -> "WorkflowLLMClient":
        return cls(WorkflowLLMConfig.from_env(prefix=prefix))

    def generate(self, request: CommonLLMRequest) -> tuple[str | None, str | None]:
        answer, fallback_reason, _ = self.generate_with_status(request)
        return answer, fallback_reason

    def generate_with_status(self, request: CommonLLMRequest) -> tuple[str | None, str | None, dict[str, Any]]:
        result: CommonLLMResult = self._generate_result(request)
        # Keep compatibility for legacy readers, but callers should consume returned status.
        self.last_call_status = dict(result.call_status)
        return result.answer, result.fallback_reason, dict(result.call_status)

    def generate_with_tools(
        self,
        request: CommonLLMRequest,
    ) -> CommonLLMResult:
        """支持 tool use 的 LLM 调用

        当 request.tools 不为空时，使用 bind_tools 调用 LLM。
        LLM 可以选择：
        1. 直接回答（answer 不为空）
        2. 调用工具（tool_calls 不为空）

        Args:
            request: LLM 请求对象，可能包含 tools 和 tool_choice

        Returns:
            CommonLLMResult，包含 answer 和/或 tool_calls
        """
        # 如果没有 tools，降级为普通调用
        if not request.tools:
            answer, fallback_reason, call_status = self.generate_with_status(request)
            return CommonLLMResult(
                answer=answer,
                fallback_reason=fallback_reason,
                call_status=call_status,
                tool_calls=None,
            )

        # 检查模型可用性
        if self._chat_model is None:
            return self._skip_with_tools(request, self._init_error or "chat_model_unavailable")

        if not self.config.enabled:
            return self._skip_with_tools(request, "llm_disabled")

        if not self.config.api_key:
            return self._skip_with_tools(request, "missing_api_key")

        # 执行带 tools 的调用
        started_at = time.perf_counter()
        self._logger.info(
            f"{request.log_namespace}.tools_start",
            **self._event_payload(
                request,
                {
                    "model": self.config.model,
                    "tools_count": len(request.tools),
                    "tool_choice": request.tool_choice,
                },
            ),
        )

        # 记录详细的 LLM 请求日志
        self._log_llm_request(
            request,
            call_type="tools",
            attempt=1,
            max_attempts=1,
        )

        try:
            # 绑定 tools
            chat_model_with_tools = self._chat_model.bind_tools(
                request.tools,
                tool_choice=request.tool_choice or "auto",
            )

            chain = self._prompt | chat_model_with_tools
            message = chain.invoke(
                {
                    "system_prompt": request.system_prompt,
                    "user_prompt": request.user_prompt,
                }
            )

            latency_ms = int((time.perf_counter() - started_at) * 1000)

            # 提取响应
            answer = self._extract_text_from_message(message)
            tool_calls = self._extract_tool_calls(message)

            # 记录详细的 LLM 响应日志
            self._log_llm_response(
                request,
                call_type="tools",
                attempt=1,
                latency_ms=latency_ms,
                response_text=answer,
                tool_calls=tool_calls,
            )

            # 记录成功状态
            success_status = self._build_status(
                status="success",
                invoked=True,
                request_sent=True,
                attempts=1,
                latency_ms=latency_ms,
                reason=None,
            )
            self.last_call_status = dict(success_status)

            self._logger.info(
                f"{request.log_namespace}.tools_success",
                **self._event_payload(
                    request,
                    {
                        "model": self.config.model,
                        "latency_ms": latency_ms,
                        "has_answer": bool(answer),
                        "tool_calls_count": len(tool_calls) if tool_calls else 0,
                    },
                ),
            )

            return CommonLLMResult(
                answer=answer if answer else None,
                fallback_reason=None,
                call_status=success_status,
                tool_calls=tool_calls,
            )

        except TimeoutError as exc:
            latency_ms = int((time.perf_counter() - started_at) * 1000)
            # 记录详细的 LLM 响应日志（超时）
            self._log_llm_response(
                request,
                call_type="tools",
                attempt=1,
                latency_ms=latency_ms,
                reason="timeout",
                error=exc,
            )
            self._logger.warning(
                f"{request.log_namespace}.tools_timeout",
                **self._event_payload(
                    request,
                    {
                        "model": self.config.model,
                        "latency_ms": latency_ms,
                        "error": str(exc),
                    },
                ),
            )
            return self._build_error_result(request, "timeout", latency_ms)

        except Exception as exc:  # pragma: no cover
            latency_ms = int((time.perf_counter() - started_at) * 1000)
            reason = self._map_exception_to_reason(exc)
            # 记录详细的 LLM 响应日志（错误）
            self._log_llm_response(
                request,
                call_type="tools",
                attempt=1,
                latency_ms=latency_ms,
                reason=reason,
                error=exc,
            )
            self._logger.error(
                f"{request.log_namespace}.tools_error",
                **self._event_payload(
                    request,
                    {
                        "model": self.config.model,
                        "latency_ms": latency_ms,
                        "reason": reason,
                        "error": str(exc),
                    },
                ),
            )
            return self._build_error_result(request, reason, latency_ms)

    def _extract_tool_calls(self, message: Any) -> list[dict[str, Any]] | None:
        """从 LLM 响应中提取 tool_calls

        Args:
            message: LangChain AIMessage 对象

        Returns:
            tool_calls 列表，格式：[{"name": "xxx", "args": {...}, "id": "..."}]
        """
        # LangChain AIMessage 的 tool_calls 属性
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            return [
                {
                    "name": tc.get("name", ""),
                    "args": tc.get("args", {}),
                    "id": tc.get("id", ""),
                }
                for tc in tool_calls
            ]

        # 兼容：检查 additional_kwargs
        additional_kwargs = getattr(message, "additional_kwargs", {}) or {}
        raw_tool_calls = additional_kwargs.get("tool_calls", [])
        if raw_tool_calls:
            parsed_calls: list[dict[str, Any]] = []
            for tc in raw_tool_calls:
                if isinstance(tc, dict):
                    func = tc.get("function", {})
                    parsed_calls.append({
                        "name": func.get("name", ""),
                        "args": func.get("arguments", {}),
                        "id": tc.get("id", ""),
                    })
            return parsed_calls if parsed_calls else None

        return None

    def _skip_with_tools(self, request: CommonLLMRequest, reason: str) -> CommonLLMResult:
        """跳过调用并返回空结果"""
        skipped_status = self._build_status(
            status="skipped",
            invoked=True,
            request_sent=False,
            attempts=0,
            latency_ms=0,
            reason=reason,
        )
        self.last_call_status = dict(skipped_status)
        return CommonLLMResult(
            answer=None,
            fallback_reason=reason,
            call_status=skipped_status,
            tool_calls=None,
        )

    def generate_with_agent(
        self,
        request: CommonLLMRequest,
        tools: list[Any] | None = None,
        max_iterations: int = 3,
    ) -> CommonLLMResult:
        """Agent 模式：自动处理 tool calling 循环

        这个方法模拟 LangChain Agent 的行为：
        1. 调用 LLM（绑定 tools）
        2. 如果 LLM 返回 tool_calls，执行对应的工具
        3. 将工具结果作为 ToolMessage 发回 LLM
        4. 重复直到 LLM 返回最终答案或达到最大迭代次数

        Args:
            request: LLM 请求对象
            tools: LangChain Tool 对象列表（如 SkillTool）
                   注意：这里传入的是 Tool 对象，不是 OpenAI schema
            max_iterations: 最大迭代次数（防止无限循环）

        Returns:
            CommonLLMResult，包含最终答案
        """
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

        # 如果没有 tools，降级为普通调用
        if not tools:
            return self.generate_with_tools(request)

        # 检查模型可用性
        if self._chat_model is None:
            return self._skip_with_tools(request, self._init_error or "chat_model_unavailable")

        if not self.config.enabled:
            return self._skip_with_tools(request, "llm_disabled")

        if not self.config.api_key:
            return self._skip_with_tools(request, "missing_api_key")

        # 构建工具映射（name -> tool）
        tool_map = {tool.name: tool for tool in tools}

        started_at = time.perf_counter()
        total_latency_ms = 0
        iterations = 0

        self._logger.info(
            f"{request.log_namespace}.agent_start",
            **self._event_payload(
                request,
                {
                    "model": self.config.model,
                    "tools_count": len(tools),
                    "max_iterations": max_iterations,
                },
            ),
        )

        # 记录 Agent 模式的初始 LLM 请求日志
        self._log_llm_request(
            request,
            call_type="agent",
            attempt=1,
            max_attempts=max_iterations,
        )

        try:
            # 绑定 tools 到 LLM
            chat_model_with_tools = self._chat_model.bind_tools(tools, tool_choice="auto")

            # 构建初始消息
            messages: list[BaseMessage] = [
                SystemMessage(content=request.system_prompt),
                HumanMessage(content=request.user_prompt),
            ]

            while iterations < max_iterations:
                iterations += 1
                iteration_start = time.perf_counter()

                self._logger.debug(
                    f"{request.log_namespace}.agent_iteration",
                    **self._event_payload(
                        request,
                        {"iteration": iterations, "messages_count": len(messages)},
                    ),
                )

                # 记录每次迭代的 LLM 请求日志
                iteration_request_payload = {
                    "call_type": "agent_iteration",
                    "model": self.config.model,
                    "iteration": iterations,
                    "messages_count": len(messages),
                    "system_prompt_preview": self._truncate_text(request.system_prompt) if iterations == 1 else "[context from previous iterations]",
                    "user_prompt_preview": self._truncate_text(request.user_prompt) if iterations == 1 else "[context from previous iterations]",
                }
                self._logger.info(
                    f"{request.log_namespace}.llm_request",
                    **self._event_payload(request, iteration_request_payload),
                )

                # 调用 LLM
                response: AIMessage = chat_model_with_tools.invoke(messages)
                iteration_latency = int((time.perf_counter() - iteration_start) * 1000)

                # 添加 AI 响应到消息历史
                messages.append(response)

                # 检查是否有 tool_calls
                tool_calls = getattr(response, "tool_calls", None)

                # 记录每次迭代的 LLM 响应日志
                iteration_response_payload: dict[str, Any] = {
                    "call_type": "agent_iteration",
                    "model": self.config.model,
                    "iteration": iterations,
                    "latency_ms": iteration_latency,
                    "has_tool_calls": bool(tool_calls),
                }

                if not tool_calls:
                    # 没有 tool_calls，LLM 返回了最终答案
                    answer = self._extract_text_from_message(response) or ""
                    total_latency_ms = int((time.perf_counter() - started_at) * 1000)

                    # 记录最终响应日志
                    iteration_response_payload["success"] = True
                    iteration_response_payload["response_length"] = len(answer)
                    iteration_response_payload["response_preview"] = self._truncate_text(answer, LLM_LOG_RESPONSE_MAX_LENGTH)
                    self._logger.info(
                        f"{request.log_namespace}.llm_response",
                        **self._event_payload(request, iteration_response_payload),
                    )

                    success_status = self._build_status(
                        status="success",
                        invoked=True,
                        request_sent=True,
                        attempts=iterations,
                        latency_ms=total_latency_ms,
                        reason=None,
                    )
                    self.last_call_status = dict(success_status)

                    self._logger.info(
                        f"{request.log_namespace}.agent_complete",
                        **self._event_payload(
                            request,
                            {
                                "model": self.config.model,
                                "latency_ms": total_latency_ms,
                                "iterations": iterations,
                                "answer_length": len(answer),
                            },
                        ),
                    )

                    # 标准化答案
                    normalized = answer.strip()
                    if request.normalize_answer and normalized:
                        normalized = request.normalize_answer(normalized)

                    return CommonLLMResult(
                        answer=normalized or None,
                        fallback_reason=None,
                        call_status=success_status,
                        tool_calls=None,
                    )

                # 有 tool_calls，记录响应日志
                iteration_response_payload["success"] = True
                iteration_response_payload["tool_calls_count"] = len(tool_calls)
                iteration_response_payload["tool_calls"] = [
                    {
                        "name": tc.get("name", ""),
                        "args_preview": self._truncate_text(str(tc.get("args", {})), 200),
                    }
                    for tc in tool_calls
                ]
                self._logger.info(
                    f"{request.log_namespace}.llm_response",
                    **self._event_payload(request, iteration_response_payload),
                )

                # 有 tool_calls，执行工具
                self._logger.info(
                    f"{request.log_namespace}.agent_tool_call",
                    **self._event_payload(
                        request,
                        {
                            "iteration": iterations,
                            "tool_calls_count": len(tool_calls),
                            "tools": [tc.get("name") for tc in tool_calls],
                        },
                    ),
                )

                # 执行每个 tool_call
                for tool_call in tool_calls:
                    tool_name = tool_call.get("name", "")
                    tool_args = tool_call.get("args", {})
                    tool_id = tool_call.get("id", "")

                    tool = tool_map.get(tool_name)
                    if tool is None:
                        # 工具不存在
                        tool_result = f"Error: Tool '{tool_name}' not found"
                        self._logger.warning(
                            f"{request.log_namespace}.agent_tool_not_found",
                            **self._event_payload(request, {"tool_name": tool_name}),
                        )
                    else:
                        # 执行工具
                        try:
                            tool_result = tool.invoke(tool_args)
                            self._logger.debug(
                                f"{request.log_namespace}.agent_tool_success",
                                **self._event_payload(
                                    request,
                                    {
                                        "tool_name": tool_name,
                                        "result_length": len(str(tool_result)),
                                    },
                                ),
                            )
                        except Exception as e:
                            tool_result = f"Error executing tool '{tool_name}': {e}"
                            self._logger.error(
                                f"{request.log_namespace}.agent_tool_error",
                                **self._event_payload(
                                    request,
                                    {"tool_name": tool_name, "error": str(e)},
                                ),
                            )

                    # 记录工具调用结果发送给 LLM 的日志
                    tool_result_str = str(tool_result)
                    result_preview = self._truncate_text(tool_result_str, LLM_LOG_RESPONSE_MAX_LENGTH)
                    self._logger.info(
                        f"{request.log_namespace}.tool_result_to_llm",
                        **self._event_payload(
                            request,
                            {
                                "tool_name": tool_name,
                                "tool_call_id": tool_id,
                                "result_length": len(tool_result_str),
                                "result_preview": result_preview,
                            },
                        ),
                    )

                    # 添加 ToolMessage 到消息历史
                    messages.append(ToolMessage(content=tool_result_str, tool_call_id=tool_id))

            # 达到最大迭代次数
            total_latency_ms = int((time.perf_counter() - started_at) * 1000)

            # 记录达到最大迭代次数的 LLM 响应日志
            self._log_llm_response(
                request,
                call_type="agent",
                attempt=iterations,
                latency_ms=total_latency_ms,
                reason="max_iterations_reached",
            )

            self._logger.warning(
                f"{request.log_namespace}.agent_max_iterations",
                **self._event_payload(
                    request,
                    {"iterations": iterations, "max_iterations": max_iterations},
                ),
            )

            # 尝试从最后一条消息提取答案
            last_message = messages[-1] if messages else None
            final_answer = self._extract_text_from_message(last_message) if last_message else None

            max_iter_status = self._build_status(
                status="max_iterations_reached",
                invoked=True,
                request_sent=True,
                attempts=iterations,
                latency_ms=total_latency_ms,
                reason="max_iterations_reached",
            )
            self.last_call_status = dict(max_iter_status)

            return CommonLLMResult(
                answer=final_answer,
                fallback_reason="max_iterations_reached",
                call_status=max_iter_status,
                tool_calls=None,
            )

        except TimeoutError as exc:
            total_latency_ms = int((time.perf_counter() - started_at) * 1000)
            # 记录超时的 LLM 响应日志
            self._log_llm_response(
                request,
                call_type="agent",
                attempt=iterations or 1,
                latency_ms=total_latency_ms,
                reason="timeout",
                error=exc,
            )
            self._logger.warning(
                f"{request.log_namespace}.agent_timeout",
                **self._event_payload(
                    request,
                    {"latency_ms": total_latency_ms, "error": str(exc)},
                ),
            )
            return self._build_error_result(request, "timeout", total_latency_ms)

        except Exception as exc:
            total_latency_ms = int((time.perf_counter() - started_at) * 1000)
            reason = self._map_exception_to_reason(exc)
            # 记录错误的 LLM 响应日志
            self._log_llm_response(
                request,
                call_type="agent",
                attempt=iterations or 1,
                latency_ms=total_latency_ms,
                reason=reason,
                error=exc,
            )
            self._logger.error(
                f"{request.log_namespace}.agent_error",
                **self._event_payload(
                    request,
                    {"latency_ms": total_latency_ms, "reason": reason, "error": str(exc)},
                ),
            )
            return self._build_error_result(request, reason, total_latency_ms)

    def _build_error_result(
        self,
        request: CommonLLMRequest,
        reason: str,
        latency_ms: int,
    ) -> CommonLLMResult:
        """构建错误结果"""
        error_status = self._build_status(
            status="error",
            invoked=True,
            request_sent=True,
            attempts=1,
            latency_ms=latency_ms,
            reason=reason,
        )
        self.last_call_status = dict(error_status)
        return CommonLLMResult(
            answer=None,
            fallback_reason=reason,
            call_status=error_status,
            tool_calls=None,
        )


WorkflowLLMConfig = CommonLLMConfig
