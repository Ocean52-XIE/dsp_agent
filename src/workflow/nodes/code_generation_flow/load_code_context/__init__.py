# -*- coding: utf-8 -*-
"""
该模块实现工作流节点`load_code_context` 的处理逻辑，负责读取状态并输出增量结果。
"""
from __future__ import annotations

from typing import Any


def run(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """
    执行`load_code_context` 节点主流程，基于输入状态计算并返回状态增量。
    
    参数:
        service: 工作流服务对象，提供检索、路由、日志与配置能力。
        state: 工作流状态字典，包含会话上下文与中间结果。
    
    返回:
        返回类型为 `dict[str, Any]` 的处理结果。
    """
    source_message = state.get("source_message") or {}
    analysis = source_message.get("analysis") or state.get("last_analysis_result") or {}
    citations = source_message.get("citations") or state.get("last_analysis_citations", [])

    module_name = analysis.get("module") or state.get("active_module_name") or state.get("module_name")
    if not module_name:
        module_name = service.domain_profile.default_module
    module_hint = service._infer_module(module_name)[1]
    return {
        "route": "issue_analysis",
        "execution_path": "code_generation_flow",
        "transition_type": state.get("transition_type", "resume_code_generation"),
        "task_stage": "code_generation",
        "module_name": module_name,
        "module_hint": module_hint,
        "analysis": analysis or None,
        "citations": citations,
        "node_trace": service._trace(state, "load_code_context", f"module={module_name}"),
    }
