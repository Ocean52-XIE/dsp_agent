from __future__ import annotations

from typing import Any


def run(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """恢复多轮会话上下文。"""
    history = state.get("history", [])
    user_query = state["user_query"]

    latest_assistant = service._latest_message_by(history, lambda message: message.get("role") == "assistant")
    latest_qa_message = service._latest_message_by(
        history,
        lambda message: message.get("role") == "assistant" and message.get("intent") == "knowledge_qa",
    )
    latest_issue_message = service._latest_message_by(
        history,
        lambda message: message.get("role") == "assistant"
        and message.get("kind") == "issue_analysis"
        and message.get("analysis") is not None,
    )
    latest_pending_confirmation = service._latest_message_by(
        history,
        lambda message: message.get("role") == "assistant" and message.get("status") == "confirm_code",
    )

    active_task_stage = service._derive_task_stage(latest_assistant)
    active_module_name, active_module_hint = service._extract_module_from_message(
        latest_issue_message or latest_qa_message
    )

    if service._is_pronoun_followup(user_query) and active_module_name:
        module_name = active_module_name
        module_hint = active_module_hint
        topic_source = "history_memory"
    else:
        module_name, module_hint = service._infer_module(user_query)
        topic_source = "current_query"

    return {
        "module_name": module_name,
        "module_hint": module_hint,
        "active_topic": active_module_name or module_name,
        "active_topic_source": topic_source,
        "active_task_stage": active_task_stage,
        "active_module_name": active_module_name or module_name,
        "active_module_hint": active_module_hint or module_hint,
        "active_qa_context": (latest_qa_message or {}).get("analysis"),
        "active_issue_context": (latest_issue_message or {}).get("analysis"),
        "last_analysis_result": (latest_issue_message or {}).get("analysis"),
        "last_analysis_citations": (latest_issue_message or {}).get("citations", []),
        "pending_action": "confirm_code" if latest_pending_confirmation else "completed",
        "history_summary": service._build_history_summary(history),
        "node_trace": service._trace(
            state,
            "load_context",
            f"active_stage={active_task_stage}, module={module_name}",
        ),
    }

