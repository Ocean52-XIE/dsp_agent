# -*- coding: utf-8 -*-
"""Single deep-agent service entrypoint."""
from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from typing import Any

from agent.factory import create_agent
from agent.result_parser import DeepAgentTurnResult, parse_agent_result
from common.request_context import reset_request_context, set_request_context
from init import get_database_status, init_database_async
from log import get_file_logger
from session.conversation_memory import render_conversation_memory

logger = logging.getLogger(__name__)


class DeepAgentService:
    """Thin async-first service wrapper around a single Deep Agent instance."""

    backend_name: str = "deepagents"
    max_context_messages: int = 6

    def __init__(
        self,
        *,
        project_root: Path,
        database_resource: Any | None,
        checkpointer: Any | None,
    ) -> None:
        self.project_root = project_root.resolve()
        self._database_resource = database_resource
        self._checkpointer = checkpointer
        self._checkpointer_status = get_database_status(self._checkpointer)
        self._runtime_logger = get_file_logger(project_root=self.project_root)
        self._agent = create_agent(
            project_root=self.project_root,
            checkpointer=self._checkpointer,
        )
        runtime_config = getattr(self._agent, "_dsp_runtime_config", None)
        runtime_config = runtime_config if isinstance(runtime_config, dict) else {}
        self._startup_summary = {
            "llm_model": str(runtime_config.get("model", "") or ""),
            "skills": list(runtime_config.get("skills", []) or []),
            "agent_tools": list(runtime_config.get("tools", []) or []),
            "checkpointer_backend": self._checkpointer_status.get("backend", ""),
            "checkpointer_status": self._checkpointer_status.get("status", ""),
            "checkpointer_reason": self._checkpointer_status.get("reason"),
            "checkpointer_fallback": self._checkpointer_status.get("fallback", False),
            "checkpointer_fallback_from": self._checkpointer_status.get("fallback_from"),
            "checkpointer_fallback_to": self._checkpointer_status.get("fallback_to"),
        }

    @classmethod
    async def create_async(cls, *, project_root: Path) -> "DeepAgentService":
        """Create the deep-agent service with async database resources."""
        resolved_root = project_root.resolve()
        runtime_logger = get_file_logger(project_root=resolved_root)
        runtime_logger.info("init.checkpointer.begin")
        checkpointer_started_at = perf_counter()
        database_resource, checkpointer = await init_database_async()
        checkpointer_status = get_database_status(checkpointer)
        runtime_logger.info(
            "init.checkpointer.completed",
            backend=checkpointer_status.get("backend", ""),
            status=checkpointer_status.get("status", ""),
            checkpointer_type=checkpointer_status.get("type"),
            reason=checkpointer_status.get("reason"),
            fallback=checkpointer_status.get("fallback", False),
            fallback_from=checkpointer_status.get("fallback_from"),
            fallback_to=checkpointer_status.get("fallback_to"),
            latency_ms=int((perf_counter() - checkpointer_started_at) * 1000),
        )

        runtime_logger.info(
            "init.agent.begin",
            checkpointer_backend=checkpointer_status.get("backend", ""),
        )
        agent_started_at = perf_counter()
        service = cls(
            project_root=resolved_root,
            database_resource=database_resource,
            checkpointer=checkpointer,
        )
        runtime_logger.info(
            "init.agent.completed",
            latency_ms=int((perf_counter() - agent_started_at) * 1000),
            **service.startup_summary(),
        )
        return service

    async def run_user_message_async(
        self,
        *,
        session_id: str,
        trace_id: str,
        user_query: str,
        history: list[dict[str, Any]],
        conversation_summary: str = "",
        conversation_memory: dict[str, Any] | None = None,
    ) -> DeepAgentTurnResult:
        """Run the deep agent and return a normalized execution result."""
        started_at = perf_counter()
        messages = self._build_messages(
            user_query=user_query,
            history=history,
            conversation_summary=conversation_summary,
            conversation_memory=conversation_memory,
        )
        llm_model = self._agent_model_name()
        logger.info(
            "Running deep agent: session_id=%s trace_id=%s history_size=%s",
            session_id,
            trace_id,
            len(history),
        )
        self._runtime_logger.info(
            "agent.turn.started",
            session_id=session_id,
            trace_id=trace_id,
            llm_model=llm_model,
            history_size=len(history),
            request_message_count=len(messages),
            user_query_preview=self._preview_text(user_query),
        )
        try:
            request_context_token = set_request_context(
                session_id=session_id,
                trace_id=trace_id,
            )
            try:
                result = await self._agent.ainvoke(
                    {"messages": messages},
                    config={
                        "configurable": {"thread_id": session_id},
                        "metadata": {
                            "session_id": session_id,
                            "trace_id": trace_id,
                            "llm_model": llm_model,
                            "agent_backend": self.backend_name,
                        },
                        "tags": ["dsp_agent", "deep_agent"],
                    },
                )
            finally:
                reset_request_context(request_context_token)
        # 仅对上游 LLM 调用异常做降级，避免把业务代码错误静默吞掉。
        except Exception as exc:
            if self._should_degrade_llm_error(exc):
                logger.warning(
                    "Deep agent degraded after LLM failure: trace_id=%s error_type=%s",
                    trace_id,
                    type(exc).__name__,
                    exc_info=exc,
                )
                degraded = self._build_degraded_result(
                    trace_id=trace_id,
                    started_at=started_at,
                    exc=exc,
                    llm_model=llm_model,
                )
                self._runtime_logger.warning(
                    "agent.turn.degraded",
                    session_id=session_id,
                    trace_id=trace_id,
                    llm_model=llm_model,
                    latency_ms=degraded.debug.get("latency_ms", 0),
                    fallback_reason=degraded.debug.get("fallback_reason", ""),
                    error_type=degraded.debug.get("error_type", ""),
                )
                return degraded
            logger.exception(
                "Deep agent failed with non-recoverable error: trace_id=%s error_type=%s",
                trace_id,
                type(exc).__name__,
            )
            self._runtime_logger.exception(
                "agent.turn.failed",
                session_id=session_id,
                trace_id=trace_id,
                llm_model=llm_model,
                error_type=type(exc).__name__,
            )
            raise
        logger.info("Deep agent finished: trace_id=%s", trace_id)
        parsed = parse_agent_result(
            trace_id=trace_id,
            result=result,
            runtime_debug={
                "latency_ms": int((perf_counter() - started_at) * 1000),
                "checkpointer_backend": self._checkpointer_status.get("backend"),
                "agent_backend": self.backend_name,
                "llm_model": llm_model,
            },
        )
        message_trace = parsed.debug.get("message_trace", [])
        self._runtime_logger.info(
            "agent.turn.completed",
            session_id=session_id,
            trace_id=trace_id,
            llm_model=llm_model,
            latency_ms=parsed.debug.get("latency_ms", 0),
            route=parsed.debug.get("route", ""),
            tool_call_count=parsed.debug.get("tool_call_count", 0),
            citation_count=len(parsed.citations),
            citation_scope=parsed.debug.get("citation_scope", ""),
            tool_calls=parsed.debug.get("tool_calls", []),
            message_trace_count=len(message_trace),
            message_trace_tail=self._summarize_message_trace_for_logging(message_trace),
            answer_preview=self._preview_text(parsed.answer, max_chars=120),
        )
        return parsed

    async def aclose(self) -> None:
        """Release async database resources held by the service."""
        if self._database_resource is None:
            return
        if hasattr(self._database_resource, "aclose"):
            await self._database_resource.aclose()
            return
        if hasattr(self._database_resource, "close"):
            self._database_resource.close()

    def runtime_log_status(self) -> dict[str, Any]:
        return {"backend": self.backend_name, "llm_model": self._agent_model_name()}

    def checkpointer_status(self) -> dict[str, Any]:
        return dict(self._checkpointer_status)

    def startup_summary(self) -> dict[str, Any]:
        return dict(self._startup_summary)

    def _agent_model_name(self) -> str:
        runtime_config = getattr(self._agent, "_dsp_runtime_config", None)
        if isinstance(runtime_config, dict):
            model = str(runtime_config.get("model", "") or "").strip()
            if model:
                return model
        return ""

    @staticmethod
    def _preview_text(value: Any, *, max_chars: int = 160) -> str:
        text = str(value or "").strip()
        if len(text) <= max_chars:
            return text
        return f"{text[:max_chars]}..."

    def _summarize_message_trace_for_logging(self, trace: Any) -> list[dict[str, Any]]:
        if not isinstance(trace, list):
            return []
        summarized: list[dict[str, Any]] = []
        for item in trace[-4:]:
            if not isinstance(item, dict):
                continue
            summary: dict[str, Any] = {
                "index": item.get("index"),
                "role": item.get("role"),
            }
            for field_name in (
                "name",
                "tool_name",
                "citation_count",
                "evidence_count",
                "retrieval_strategy",
                "requested_top_k",
                "returned_citation_count",
                "wiki_hits",
                "code_hits",
                "latency_ms",
            ):
                value = item.get(field_name)
                if value not in (None, "", []):
                    summary[field_name] = value
            content_preview = str(item.get("content_preview", "") or "").strip()
            if content_preview:
                summary["content_preview"] = self._preview_text(content_preview, max_chars=80)
            summarized.append(summary)
        return summarized

    def _build_degraded_result(
        self,
        *,
        trace_id: str,
        started_at: float,
        exc: Exception,
        llm_model: str = "",
    ) -> DeepAgentTurnResult:
        error_type = type(exc).__name__
        status_code = getattr(exc, "status_code", None)
        latency_ms = int((perf_counter() - started_at) * 1000)
        fallback_reason = (
            "llm_authentication_failed"
            if self._is_authentication_error(exc)
            else "llm_request_failed"
        )
        if fallback_reason == "llm_authentication_failed":
            answer = (
                "当前 LLM 服务鉴权失败，系统已降级返回。"
                "请检查 `AGENT_LLM_API_KEY` 是否配置正确后重试。"
            )
        else:
            answer = (
                "当前 LLM 服务暂时不可用，系统已降级返回。"
                "请稍后重试；若问题持续，请检查模型服务配置与网络连通性。"
            )

        analysis: dict[str, Any] = {
            "driver": self.backend_name,
            "message_count": 0,
            "citation_count": 0,
            "tool_call_count": 0,
            "tools_used": [],
            "skills_used": [],
            "fallback_reason": fallback_reason,
            "llm_model": llm_model,
        }
        debug: dict[str, Any] = {
            "route": "error",
            "skills_used": [],
            "tools_used": [],
            "tool_calls": [],
            "tool_call_count": 0,
            "backend": self.backend_name,
            "latency_ms": latency_ms,
            "checkpointer_backend": self._checkpointer_status.get("backend"),
            "agent_backend": self.backend_name,
            "fallback_reason": fallback_reason,
            "error_type": error_type,
            "llm_model": llm_model,
        }
        raw_result: dict[str, Any] = {
            "error_type": error_type,
            "fallback_reason": fallback_reason,
        }
        if status_code is not None:
            debug["upstream_status_code"] = status_code
            raw_result["status_code"] = status_code
        return DeepAgentTurnResult(
            trace_id=trace_id,
            answer=answer,
            citations=[],
            message_count=0,
            status="error",
            driver=self.backend_name,
            raw_result=raw_result,
            intent="error",
            analysis=analysis,
            debug=debug,
        )

    @staticmethod
    def _is_authentication_error(exc: Exception) -> bool:
        class_name = type(exc).__name__.lower()
        status_code = getattr(exc, "status_code", None)
        message = str(exc or "").lower()
        return (
            class_name == "authenticationerror"
            or status_code == 401
            or ("api key" in message and "invalid" in message)
            or "authentication fails" in message
        )

    @classmethod
    def _should_degrade_llm_error(cls, exc: Exception) -> bool:
        if cls._is_authentication_error(exc):
            return True
        if isinstance(exc, (ConnectionError, TimeoutError)):
            return True
        class_name = type(exc).__name__.lower()
        module_name = type(exc).__module__.lower()
        status_code = getattr(exc, "status_code", None)
        if module_name.startswith("openai"):
            return True
        if class_name in {
            "openaierror",
            "apiconnectionerror",
            "apitimeouterror",
            "ratelimiterror",
            "internalservererror",
        }:
            return True
        return status_code in {408, 409, 429, 500, 502, 503, 504}

    def _build_messages(
        self,
        *,
        user_query: str,
        history: list[dict[str, Any]],
        conversation_summary: str = "",
        conversation_memory: dict[str, Any] | None = None,
    ) -> list[dict[str, str]]:
        normalized_history: list[dict[str, str]] = []
        latest_query = user_query.strip()
        summary = str(conversation_summary or "").strip()
        structured_memory = render_conversation_memory(conversation_memory)

        for item in history:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "") or "").strip().lower()
            if role not in {"user", "assistant"}:
                continue
            content = str(item.get("content", "") or "").strip()
            if not content:
                continue
            normalized_history.append({"role": role, "content": content})

        if normalized_history:
            recent_messages = normalized_history[-self.max_context_messages :]
            if latest_query:
                last_message = recent_messages[-1]
                if not (
                    last_message["role"] == "user"
                    and last_message["content"] == latest_query
                ):
                    recent_messages.append({"role": "user", "content": latest_query})
            messages = recent_messages[-self.max_context_messages :]
            context_messages: list[dict[str, str]] = []
            if summary:
                context_messages.append({"role": "system", "content": f"会话摘要:\n{summary}"})
            if structured_memory:
                context_messages.append({"role": "system", "content": f"结构化记忆:\n{structured_memory}"})
            if context_messages:
                return [*context_messages, *messages]
            return messages

        if latest_query:
            messages = [{"role": "user", "content": latest_query}]
            context_messages = []
            if summary:
                context_messages.append({"role": "system", "content": f"会话摘要:\n{summary}"})
            if structured_memory:
                context_messages.append({"role": "system", "content": f"结构化记忆:\n{structured_memory}"})
            if context_messages:
                return [*context_messages, *messages]
            return messages
        return []
