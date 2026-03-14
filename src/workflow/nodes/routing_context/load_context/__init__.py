# -*- coding: utf-8 -*-
"""Load context node."""
from __future__ import annotations

from typing import Any

from workflow.common.domain_profile import DomainProfile
from workflow.common.node_trace import append_node_trace
from workflow.nodes.routing_context.load_context.helpers import (
    extract_module_from_message,
    latest_message_by,
)


def _looks_like_context_dependent_followup(text: str, *, domain_profile: DomainProfile) -> bool:
    if domain_profile.is_pronoun_followup(text):
        return True

    normalized = " ".join((text or "").strip().lower().split())
    compact = normalized.replace(" ", "")
    if not compact:
        return False

    if domain_profile.looks_like_code_location_query(normalized):
        return True

    has_code_anchor = any(
        token in normalized
        for token in (
            "\u4ee3\u7801",  # code
            "\u51fd\u6570",  # function
            "\u6587\u4ef6",  # file
            "\u8def\u5f84",  # path
            "\u5b9e\u73b0",  # implementation
            "\u5165\u53e3",  # entrypoint
            "\u884c\u53f7",  # line number
        )
    )
    has_where_anchor = any(
        token in normalized
        for token in (
            "\u54ea\u91cc",  # where
            "\u5728\u54ea",  # at where
            "\u54ea\u4e2a",  # which
            "\u54ea\u4e00",  # which one
            "\u4f4d\u7f6e",  # location
        )
    )
    return has_code_anchor and has_where_anchor and len(compact) <= 16


def run(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    history = state.get("history", [])
    user_query = state["user_query"]

    latest_qa_message = latest_message_by(
        history,
        lambda message: message.get("role") == "assistant" and message.get("intent") == "knowledge_qa",
    )
    latest_issue_message = latest_message_by(
        history,
        lambda message: message.get("role") == "assistant"
        and message.get("kind") == "issue_analysis"
        and message.get("analysis") is not None,
    )

    active_module_name, history_module_hint = extract_module_from_message(
        latest_issue_message or latest_qa_message,
        domain_profile=service.domain_profile,
    )

    inferred_module_name, inferred_module_hint = service.domain_profile.infer_module(user_query)
    reuse_history_topic = False
    if active_module_name and _looks_like_context_dependent_followup(user_query, domain_profile=service.domain_profile):
        reuse_history_topic = True
    elif (
        active_module_name
        and service.domain_profile.looks_like_code_location_query(user_query)
        and inferred_module_name == service.domain_profile.default_module
    ):
        # Generic code-location query without module keywords should stay on active topic.
        reuse_history_topic = True

    if reuse_history_topic:
        module_name = active_module_name
        module_hint = history_module_hint
        topic_source = "history_memory"
    else:
        module_name = inferred_module_name
        module_hint = inferred_module_hint
        topic_source = "current_query"

    return {
        "module_name": module_name,
        "module_hint": module_hint,
        "active_topic_source": topic_source,
        "active_module_name": active_module_name or module_name,
        "last_analysis_result": (latest_issue_message or {}).get("analysis"),
        "last_analysis_citations": (latest_issue_message or {}).get("citations", []),
        "node_trace": append_node_trace(
            state,
            "load_context",
            f"module={module_name}, source={topic_source}",
        ),
    }
