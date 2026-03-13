# -*- coding: utf-8 -*-
"""
该模块实现工作流节点`conversation_transition` 的处理逻辑，负责读取状态并输出增量结果。
"""
from __future__ import annotations

from typing import Any


def _normalized_route(route: str) -> str:
    """
    内部辅助函数，负责`normalized route` 相关处理。
    
    参数:
        route: 输入参数，用于控制当前处理逻辑。
    
    返回:
        返回类型为 `str` 的处理结果。
    """
    if route == "issue_analysis":
        return "issue_analysis"
    return "knowledge_qa"


def _has_analysis_context(state: dict[str, Any]) -> bool:
    """
    内部辅助函数，负责`has analysis context` 相关处理。
    
    参数:
        state: 工作流状态字典，包含会话上下文与中间结果。
    
    返回:
        返回类型为 `bool` 的处理结果。
    """
    return bool(state.get("last_analysis_result") or state.get("active_issue_context"))


def run(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """
    执行`conversation_transition` 节点主流程，基于输入状态计算并返回状态增量。
    
    参数:
        service: 工作流服务对象，提供检索、路由、日志与配置能力。
        state: 工作流状态字典，包含会话上下文与中间结果。
    
    返回:
        返回类型为 `dict[str, Any]` 的处理结果。
    """
    user_query = str(state.get("user_query", "") or "")
    current_route = _normalized_route(str(state.get("route", "knowledge_qa")))
    active_stage = str(state.get("active_task_stage", "idle") or "idle")
    pending_action = str(state.get("pending_action", "completed") or "completed")
    current_module = str(state.get("module_name", "") or "")
    active_module = str(state.get("active_module_name", "") or "")
    same_topic = service._is_same_topic(current_module, active_module, user_query)

    wants_code = service._looks_like_code_generation_request(user_query)
    declines_code = service._looks_like_decline_code_request(user_query)
    has_analysis_context = _has_analysis_context(state)
    resume_code_from_confirmation = pending_action == "confirm_code" or active_stage == "confirm_code"

    # 用户在确认阶段明确拒绝代码实现，直接走控制响应。
    if declines_code and pending_action == "confirm_code":
        transition_type = "decline_code_generation"
        return {
            "route": "issue_analysis",
            "execution_path": "decline_code_flow",
            "transition_type": transition_type,
            "task_stage": "issue_analysis",
            "node_trace": service._trace(
                state,
                "conversation_transition",
                f"transition={transition_type}, path=decline_code_flow",
            ),
        }

    # 用户显式请求代码实现：有分析上下文则直接进入代码生成；否则先补齐问题分析。
    if wants_code:
        if has_analysis_context and (same_topic or not active_module or resume_code_from_confirmation):
            transition_type = "continue_code_generation" if active_stage == "code_generation" else "upgrade_to_code_generation"
            return {
                "route": "issue_analysis",
                "execution_path": "code_generation_flow",
                "transition_type": transition_type,
                "task_stage": "code_generation",
                "node_trace": service._trace(
                    state,
                    "conversation_transition",
                    f"transition={transition_type}, path=code_generation_flow",
                ),
            }

        transition_type = "code_request_without_analysis"
        return {
            "route": "issue_analysis",
            "execution_path": "retrieval_flow",
            "transition_type": transition_type,
            "task_stage": "issue_analysis",
            "node_trace": service._trace(
                state,
                "conversation_transition",
                f"transition={transition_type}, path=retrieval_flow",
            ),
        }

    if current_route == "issue_analysis":
        if active_stage == "knowledge_qa":
            transition_type = "upgrade_from_qa_to_issue_analysis"
        elif active_stage in {"issue_analysis", "confirm_code", "code_generation"}:
            transition_type = "continue_issue_analysis" if same_topic else "switch_topic"
        else:
            transition_type = "start_issue_analysis"
        next_route = "issue_analysis"
        next_stage = "issue_analysis"
    else:
        if active_stage == "knowledge_qa":
            transition_type = "continue_knowledge_qa" if same_topic else "switch_topic"
        elif active_stage in {"issue_analysis", "confirm_code", "code_generation"}:
            transition_type = "continue_knowledge_qa" if same_topic else "switch_topic"
        else:
            transition_type = "start_knowledge_qa"
        next_route = "knowledge_qa"
        next_stage = "knowledge_qa"

    return {
        "route": next_route,
        "execution_path": "retrieval_flow",
        "transition_type": transition_type,
        "task_stage": next_stage,
        "node_trace": service._trace(
            state,
            "conversation_transition",
            f"transition={transition_type}, path=retrieval_flow, route={next_route}, active_stage={active_stage}",
        ),
    }
