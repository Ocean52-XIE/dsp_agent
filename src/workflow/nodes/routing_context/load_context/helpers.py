# -*- coding: utf-8 -*-
"""Helpers local to load_context node."""
from __future__ import annotations

from typing import Any, Callable

from workflow.common.domain_profile import DomainProfile


def build_history_summary(history: list[dict[str, Any]]) -> str:
    recent_questions = [
        message.get("content", "")[:24]
        for message in history
        if message.get("role") == "user"
    ][-4:]
    return " / ".join(recent_questions) if recent_questions else "暂无历史用户提问"


def latest_message_by(
    history: list[dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool],
) -> dict[str, Any] | None:
    for message in reversed(history):
        if predicate(message):
            return message
    return None


def extract_module_from_message(
    message: dict[str, Any] | None,
    *,
    domain_profile: DomainProfile,
) -> tuple[str, str]:
    if not message:
        return "", ""
    analysis = message.get("analysis") or {}
    module_name = analysis.get("module", "")
    # 这里直接按模块名读取 hint，避免把模块名再次当成查询文本去路由，
    # 从而错误回落到默认模块的说明。
    module_hint = domain_profile.module_hint(module_name) if module_name else ""
    return module_name, module_hint
