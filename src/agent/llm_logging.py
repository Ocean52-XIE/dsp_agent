# -*- coding: utf-8 -*-
"""LLM 调用级别结构化日志回调。"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import UUID

from langchain_core.callbacks.base import AsyncCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import LLMResult

from common.func_utils import env_bool
from log import get_file_logger


class LLMCallLoggingCallback(AsyncCallbackHandler):
    """记录 LangChain ChatModel 每次调用的结构化日志。"""

    def __init__(
        self,
        *,
        project_root: Path,
        preview_chars: int = 96,
        max_logged_messages: int = 4,
        include_message_previews: bool | None = None,
    ) -> None:
        self._logger = get_file_logger(project_root=project_root)
        self._preview_chars = max(32, int(preview_chars))
        self._max_logged_messages = max(1, int(max_logged_messages))
        if include_message_previews is None:
            include_message_previews = env_bool("AGENT_LLM_LOG_INCLUDE_MESSAGE_PREVIEWS", False)
        self._include_message_previews = bool(include_message_previews)
        self._active_runs: dict[str, dict[str, Any]] = {}
        self._trace_call_counters: dict[str, int] = defaultdict(int)

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        metadata_dict = dict(metadata or {})
        session_id = self._as_text(metadata_dict.get("session_id"))
        trace_id = self._as_text(metadata_dict.get("trace_id"))
        llm_model = self._as_text(metadata_dict.get("llm_model"))
        counter_key = trace_id or session_id or "global"
        self._trace_call_counters[counter_key] += 1
        call_index = self._trace_call_counters[counter_key]

        run_key = str(run_id)
        self._active_runs[run_key] = {
            "started_at": perf_counter(),
            "session_id": session_id,
            "trace_id": trace_id,
            "llm_model": llm_model,
            "call_index": call_index,
        }

        payload = {
            "run_id": run_key,
            "parent_run_id": str(parent_run_id) if parent_run_id else "",
            "session_id": session_id,
            "trace_id": trace_id,
            "llm_model": llm_model,
            "call_index": call_index,
            "batch_size": len(messages),
            "input_summary": self._summarize_batches(messages),
            "tags": list(tags or []),
            "invocation_params": self._extract_invocation_params(kwargs),
            "serialized": self._extract_serialized(serialized),
        }
        self._logger.info("llm.call.started", **payload)

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        state = self._active_runs.pop(str(run_id), {})
        generation = self._first_generation(response)
        message = getattr(generation, "message", None)
        llm_output = dict(response.llm_output or {})
        usage = self._extract_usage(message, llm_output)
        payload = {
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id) if parent_run_id else "",
            "session_id": self._as_text(state.get("session_id")),
            "trace_id": self._as_text(state.get("trace_id")),
            "llm_model": self._resolve_model_name(state=state, llm_output=llm_output, message=message),
            "call_index": int(state.get("call_index", 0) or 0),
            "latency_ms": self._latency_ms(state),
            "finish_reason": self._extract_finish_reason(generation, message),
            "response_id": self._as_text(llm_output.get("id") or self._response_metadata(message).get("id")),
            "tool_call_count": len(getattr(message, "tool_calls", []) or []),
            "usage": usage,
            "output_preview": self._preview_message_content(message),
            "tags": list(tags or []),
        }
        self._logger.info("llm.call.completed", **payload)

    async def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        state = self._active_runs.pop(str(run_id), {})
        partial_response = kwargs.get("response")
        llm_output = {}
        message = None
        if isinstance(partial_response, LLMResult):
            generation = self._first_generation(partial_response)
            message = getattr(generation, "message", None)
            llm_output = dict(partial_response.llm_output or {})
        payload = {
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id) if parent_run_id else "",
            "session_id": self._as_text(state.get("session_id")),
            "trace_id": self._as_text(state.get("trace_id")),
            "llm_model": self._resolve_model_name(state=state, llm_output=llm_output, message=message),
            "call_index": int(state.get("call_index", 0) or 0),
            "latency_ms": self._latency_ms(state),
            "error_type": type(error).__name__,
            "error_message": self._preview_text(str(error or ""), max_chars=240),
            "response_id": self._as_text(llm_output.get("id") or self._response_metadata(message).get("id")),
            "usage": self._extract_usage(message, llm_output),
            "tags": list(tags or []),
        }
        self._logger.warning("llm.call.failed", **payload)

    def _summarize_batches(self, batches: list[list[BaseMessage]]) -> dict[str, Any]:
        flat_messages = [message for batch in batches for message in batch]
        role_counts: dict[str, int] = defaultdict(int)
        for message in flat_messages:
            role_counts[self._message_role(message)] += 1
        tail_messages = flat_messages[-self._max_logged_messages :]
        return {
            "message_count": len(flat_messages),
            "role_counts": dict(role_counts),
            "logged_message_count": len(tail_messages),
            "messages": [self._summarize_message(message) for message in tail_messages],
        }

    def _summarize_message(self, message: BaseMessage) -> dict[str, Any]:
        role = self._message_role(message)
        item = {"role": role}
        name = self._as_text(getattr(message, "name", ""))
        if name:
            item["name"] = name
        tool_call_count = len(getattr(message, "tool_calls", []) or [])
        if tool_call_count > 0:
            item["tool_call_count"] = tool_call_count
        content_preview = self._preview_message_content(message)
        if content_preview:
            item["content_chars"] = len(content_preview)
            if self._include_message_previews and role in {"human", "ai", "assistant"}:
                item["content_preview"] = content_preview
        return item

    def _extract_invocation_params(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        invocation_params = kwargs.get("invocation_params")
        if not isinstance(invocation_params, dict):
            return {}
        result: dict[str, Any] = {}
        for field_name in ("model", "temperature", "max_tokens", "max_completion_tokens", "timeout", "stream"):
            value = invocation_params.get(field_name)
            if value not in (None, "", []):
                result[field_name] = value
        return result

    def _extract_serialized(self, serialized: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(serialized, dict):
            return {}
        result: dict[str, Any] = {}
        for field_name in ("name",):
            value = serialized.get(field_name)
            if value not in (None, "", []):
                result[field_name] = value
        serialized_id = serialized.get("id")
        if isinstance(serialized_id, list) and serialized_id:
            result["id"] = serialized_id[-1]
        elif serialized_id not in (None, "", []):
            result["id"] = serialized_id
        return result

    @staticmethod
    def _first_generation(response: LLMResult) -> Any | None:
        if not response.generations:
            return None
        first_group = response.generations[0]
        if not first_group:
            return None
        return first_group[0]

    @staticmethod
    def _response_metadata(message: Any) -> dict[str, Any]:
        if message is None:
            return {}
        metadata = getattr(message, "response_metadata", {})
        return dict(metadata or {}) if isinstance(metadata, dict) else {}

    def _extract_finish_reason(self, generation: Any | None, message: Any) -> str:
        if generation is not None:
            generation_info = getattr(generation, "generation_info", None)
            if isinstance(generation_info, dict):
                finish_reason = self._as_text(generation_info.get("finish_reason"))
                if finish_reason:
                    return finish_reason
        return self._as_text(self._response_metadata(message).get("finish_reason"))

    def _extract_usage(self, message: Any, llm_output: dict[str, Any]) -> dict[str, Any]:
        usage_metadata = getattr(message, "usage_metadata", None)
        if isinstance(usage_metadata, dict):
            return {
                "input_tokens": int(usage_metadata.get("input_tokens", 0) or 0),
                "output_tokens": int(usage_metadata.get("output_tokens", 0) or 0),
                "total_tokens": int(usage_metadata.get("total_tokens", 0) or 0),
            }
        token_usage = llm_output.get("token_usage")
        if isinstance(token_usage, dict):
            return {
                "input_tokens": int(token_usage.get("prompt_tokens", 0) or 0),
                "output_tokens": int(token_usage.get("completion_tokens", 0) or 0),
                "total_tokens": int(token_usage.get("total_tokens", 0) or 0),
            }
        return {}

    def _resolve_model_name(self, *, state: dict[str, Any], llm_output: dict[str, Any], message: Any) -> str:
        return (
            self._as_text(llm_output.get("model_name"))
            or self._as_text(self._response_metadata(message).get("model_name"))
            or self._as_text(state.get("llm_model"))
        )

    def _latency_ms(self, state: dict[str, Any]) -> int:
        started_at = state.get("started_at")
        if not isinstance(started_at, (int, float)):
            return 0
        return int((perf_counter() - started_at) * 1000)

    def _preview_message_content(self, message: Any) -> str:
        if message is None:
            return ""
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return self._preview_text(content)
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                if isinstance(item, dict):
                    text = self._as_text(item.get("text") or item.get("content") or item.get("input"))
                    if text:
                        parts.append(text)
            return self._preview_text("\n".join(parts))
        return self._preview_text(str(content or ""))

    @staticmethod
    def _message_role(message: BaseMessage) -> str:
        return str(getattr(message, "type", "") or getattr(message, "role", "") or "unknown").lower()

    def _preview_text(self, value: str, *, max_chars: int | None = None) -> str:
        text = self._as_text(value)
        limit = max_chars or self._preview_chars
        if len(text) <= limit:
            return text
        return f"{text[:limit]}..."

    @staticmethod
    def _as_text(value: Any) -> str:
        return str(value or "").strip()
