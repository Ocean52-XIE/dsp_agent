# -*- coding: utf-8 -*-
"""
该模块实现工作流节点`load_context` 的处理逻辑，负责读取状态并输出增量结果。
"""
from __future__ import annotations

from typing import Any


def run(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """
    执行`load_context` 节点主流程，基于输入状态计算并返回状态增量。
    
    参数:
        service: 工作流服务对象，提供检索、路由、日志与配置能力。
        state: 工作流状态字典，包含会话上下文与中间结果。
    
    返回:
        返回类型为 `dict[str, Any]` 的处理结果。
    """
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

    inferred_module_name, inferred_module_hint = service._infer_module(user_query)
    reuse_history_topic = False
    if active_module_name and service._looks_like_context_dependent_followup(user_query):
        reuse_history_topic = True
    elif (
        active_module_name
        and service._looks_like_code_location_query(user_query)
        and inferred_module_name == service.domain_profile.default_module
    ):
        # Generic code-location query without module keywords should stay on active topic.
        reuse_history_topic = True

    if reuse_history_topic:
        module_name = active_module_name
        module_hint = active_module_hint
        topic_source = "history_memory"
    else:
        module_name = inferred_module_name
        module_hint = inferred_module_hint
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
