# -*- coding: utf-8 -*-
"""Shared LLM client and invocation capability for workflow nodes."""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import time
from typing import Any, Callable

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from workflow.common.runtime_logging import get_file_logger
from workflow.common.func_utils import to_bool, to_float, to_int


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
    """LLM request payload passed from a node-level wrapper."""

    node_name: str
    system_prompt: str
    user_prompt: str
    evidence_count: int = 0
    require_evidence: bool = True
    log_namespace: str = "workflow.llm"
    metadata: dict[str, Any] = field(default_factory=dict)
    normalize_answer: AnswerNormalizer | None = None
    validate_answer: AnswerValidator | None = None


@dataclass(frozen=True)
class CommonLLMResult:
    """LLM invocation result."""

    answer: str | None
    fallback_reason: str | None
    call_status: dict[str, Any]


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
            except TimeoutError as exc:
                last_reason = "timeout"
                self._log_debug_response(
                    request,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    latency_ms=int((time.perf_counter() - attempt_started) * 1000),
                    response_text=None,
                    reason=last_reason,
                    error=exc,
                )
            except ValueError as value_error:
                reason_text = str(value_error).strip() or "empty_answer"
                last_reason = reason_text if reason_text.startswith("empty_answer") else f"empty_answer:{reason_text}"
                self._log_debug_response(
                    request,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    latency_ms=int((time.perf_counter() - attempt_started) * 1000),
                    response_text=None,
                    reason=last_reason,
                    error=value_error,
                )
            except Exception as exc:  # pragma: no cover - defensive fallback
                last_reason = self._map_exception_to_reason(exc)
                self._log_debug_response(
                    request,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    latency_ms=int((time.perf_counter() - attempt_started) * 1000),
                    response_text=None,
                    reason=last_reason,
                    error=exc,
                )
            else:
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

    def _set_last_call_status(
        self,
        *,
        status: str,
        invoked: bool,
        request_sent: bool,
        attempts: int,
        latency_ms: int,
        reason: str | None,
    ) -> None:
        self.last_call_status = self._build_status(
            status=status,
            invoked=invoked,
            request_sent=request_sent,
            attempts=attempts,
            latency_ms=latency_ms,
            reason=reason,
        )

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


WorkflowLLMConfig = CommonLLMConfig
