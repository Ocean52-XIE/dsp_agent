# -*- coding: utf-8 -*-
"""LLM-backed conversation summary updater for long-running sessions."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agent.config import DeepAgentConfig
from agent.llm_logging import LLMCallLoggingCallback
from common.func_utils import to_bool, to_int
from domain_profile import get_domain_profile
from log import get_file_logger
from session.conversation_memory import render_conversation_memory


def _as_float(value: str, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class ConversationSummaryConfig:
    """Runtime settings for the session conversation summary."""

    enabled: bool
    model: str
    api_key: str
    base_url: str
    temperature: float
    max_tokens: int
    timeout_seconds: int
    trigger_message_count: int
    max_context_messages: int
    max_summary_chars: int

    @classmethod
    def from_env(cls, *, project_root: Path) -> "ConversationSummaryConfig":
        domain_profile = get_domain_profile(project_root=project_root)
        agent_config = DeepAgentConfig.from_domain_profile(
            domain_profile=domain_profile,
            project_root=project_root,
        )
        return cls(
            enabled=to_bool(os.getenv("AGENT_SESSION_SUMMARY_ENABLED"), True),
            model=str(os.getenv("AGENT_SESSION_SUMMARY_MODEL", "") or "").strip() or agent_config.model,
            api_key=str(os.getenv("AGENT_SESSION_SUMMARY_API_KEY", "") or "").strip() or agent_config.api_key,
            base_url=str(os.getenv("AGENT_SESSION_SUMMARY_BASE_URL", "") or "").strip() or agent_config.base_url,
            temperature=_as_float(
                str(os.getenv("AGENT_SESSION_SUMMARY_TEMPERATURE", "") or "").strip()
                or "0.1",
                0.1,
            ),
            max_tokens=max(128, to_int(os.getenv("AGENT_SESSION_SUMMARY_MAX_TOKENS"), 400)),
            timeout_seconds=max(
                5,
                to_int(
                    os.getenv("AGENT_SESSION_SUMMARY_TIMEOUT_SECONDS") or str(agent_config.timeout_seconds),
                    agent_config.timeout_seconds,
                ),
            ),
            trigger_message_count=max(
                4,
                to_int(os.getenv("AGENT_SESSION_SUMMARY_TRIGGER_MESSAGE_COUNT"), 8),
            ),
            max_context_messages=max(
                4,
                to_int(os.getenv("AGENT_SESSION_SUMMARY_MAX_CONTEXT_MESSAGES"), 12),
            ),
            max_summary_chars=max(
                120,
                to_int(os.getenv("AGENT_SESSION_SUMMARY_MAX_CHARS"), 400),
            ),
        )


@dataclass(frozen=True)
class ConversationSummaryUpdate:
    """Summary text plus structured updates extracted from the LLM response."""

    summary: str
    memory_updates: dict[str, Any]
    raw_text: str = ""


class ConversationSummarizer:
    """Generate and update a compact conversation summary with an LLM."""

    def __init__(
        self,
        *,
        project_root: Path,
        config: ConversationSummaryConfig,
        model: Any | None = None,
    ) -> None:
        self.project_root = project_root.resolve()
        self.config = config
        self._logger = get_file_logger(project_root=self.project_root)
        self._domain_profile = get_domain_profile(project_root=self.project_root)
        self._model = model if model is not None else self._build_model()

    @classmethod
    def create(cls, *, project_root: Path) -> "ConversationSummarizer":
        config = ConversationSummaryConfig.from_env(project_root=project_root)
        return cls(project_root=project_root, config=config)

    @property
    def is_enabled(self) -> bool:
        return bool(self.config.enabled)

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.config.enabled,
            "model": self.config.model,
            "trigger_message_count": self.config.trigger_message_count,
            "max_context_messages": self.config.max_context_messages,
            "max_summary_chars": self.config.max_summary_chars,
            "response_format": "json",
        }

    def should_refresh(self, *, messages: list[dict[str, Any]]) -> bool:
        if not self.is_enabled:
            return False
        return len(self._normalize_messages(messages)) >= self.config.trigger_message_count

    async def refresh_summary(
        self,
        *,
        messages: list[dict[str, Any]],
        previous_summary: str = "",
        conversation_memory: dict[str, Any] | None = None,
    ) -> ConversationSummaryUpdate:
        previous = str(previous_summary or "").strip()
        if not self.should_refresh(messages=messages):
            return ConversationSummaryUpdate(summary=previous, memory_updates={})

        prompt_messages = self._build_prompt(
            messages=messages,
            previous_summary=previous,
            conversation_memory=conversation_memory,
        )
        try:
            response = await self._model.ainvoke(prompt_messages)
        except Exception as exc:
            self._logger.warning(
                "session.summary.refresh_failed",
                error_type=type(exc).__name__,
            )
            return ConversationSummaryUpdate(summary=previous, memory_updates={})

        raw_text = self._extract_response_text(getattr(response, "content", ""))
        parsed = self._parse_summary_payload(raw_text)
        if parsed is not None:
            summary_text = self._normalize_summary_text(parsed.get("summary", ""))
            if not summary_text:
                summary_text = previous
            return ConversationSummaryUpdate(
                summary=summary_text,
                memory_updates=self._normalize_memory_updates(parsed),
                raw_text=raw_text,
            )

        summary = self._normalize_summary_text(raw_text)
        if not summary:
            summary = previous
        return ConversationSummaryUpdate(
            summary=summary,
            memory_updates={},
            raw_text=raw_text,
        )

    def _build_model(self) -> ChatOpenAI:
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "timeout": self.config.timeout_seconds,
            "callbacks": [LLMCallLoggingCallback(project_root=self.project_root)],
        }
        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        return ChatOpenAI(**kwargs)

    def _build_prompt(
        self,
        *,
        messages: list[dict[str, Any]],
        previous_summary: str,
        conversation_memory: dict[str, Any] | None,
    ) -> list[Any]:
        rendered_messages = self._render_messages(messages)
        rendered_memory = render_conversation_memory(conversation_memory)
        system_prompt = (
            f"你正在为{self._domain_profile.display_name or self._domain_profile.profile_id}领域助手维护会话摘要。"
            "请基于旧摘要、已有结构化记忆和最近对话，输出一个 JSON 对象。"
            "目标是让后续轮次既能看到紧凑摘要，也能拿到少量高价值结构化信息。"
            "不要虚构模块名、文件路径、函数名、指标结论或引用。"
            "不确定的信息请明确写成“待确认”。"
            "JSON 字段固定为: "
            '{"summary":"",'
            '"current_topic":"",'
            '"active_issue":"",'
            '"confirmed_facts":[],'
            '"open_questions":[]}'
            f"其中 summary 为中文纯文本摘要，控制在{self.config.max_summary_chars}字以内。"
            "confirmed_facts 与 open_questions 各保留 1-4 条最重要内容。"
        )
        human_prompt = (
            f"旧摘要:\n{previous_summary or '无'}\n\n"
            f"已有结构化记忆:\n{rendered_memory or '无'}\n\n"
            f"最近对话:\n{rendered_messages or '无'}\n\n"
            "请输出更新后的 JSON。"
        )
        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]

    def _render_messages(self, messages: list[dict[str, Any]]) -> str:
        recent_messages = self._normalize_messages(messages)[-self.config.max_context_messages :]
        return "\n".join(f"- {message['role']}: {message['content']}" for message in recent_messages)

    @staticmethod
    def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        for item in messages:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "") or "").strip().lower()
            if role not in {"user", "assistant"}:
                continue
            content = str(item.get("content", "") or "").strip()
            if not content:
                continue
            normalized.append({"role": role, "content": content})
        return normalized

    def _normalize_summary_text(self, value: Any) -> str:
        text = self._extract_response_text(value)
        if not text:
            return ""
        if len(text) > self.config.max_summary_chars:
            return text[: self.config.max_summary_chars].rstrip()
        return text

    @staticmethod
    def _extract_response_text(value: Any) -> str:
        if isinstance(value, str):
            text = value.strip()
        elif isinstance(value, list):
            parts: list[str] = []
            for item in value:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                if isinstance(item, dict):
                    maybe_text = str(item.get("text", "") or "").strip()
                    if maybe_text:
                        parts.append(maybe_text)
            text = "\n".join(parts).strip()
        else:
            text = str(value or "").strip()
        return text.replace("\r\n", "\n").strip()

    @staticmethod
    def _parse_summary_payload(text: str) -> dict[str, Any] | None:
        normalized = str(text or "").strip()
        if not normalized:
            return None
        try:
            payload = json.loads(normalized)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _normalize_memory_updates(payload: dict[str, Any]) -> dict[str, Any]:
        updates: dict[str, Any] = {}
        for field_name in ("current_topic", "active_issue"):
            value = str(payload.get(field_name, "") or "").strip()
            if value:
                updates[field_name] = value
        for field_name in ("confirmed_facts", "open_questions"):
            value = payload.get(field_name)
            if isinstance(value, list):
                normalized_values = [
                    str(item or "").strip()
                    for item in value
                    if str(item or "").strip()
                ]
                if normalized_values:
                    updates[field_name] = normalized_values
        return updates
