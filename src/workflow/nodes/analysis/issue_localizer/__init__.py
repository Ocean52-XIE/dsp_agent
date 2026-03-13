# -*- coding: utf-8 -*-
"""
该模块实现工作流节点`issue_localizer` 的处理逻辑，负责读取状态并输出增量结果。
"""
from __future__ import annotations

from typing import Any


def run(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """
    执行`issue_localizer` 节点主流程，基于输入状态计算并返回状态增量。
    
    参数:
        service: 工作流服务对象，提供检索、路由、日志与配置能力。
        state: 工作流状态字典，包含会话上下文与中间结果。
    
    返回:
        返回类型为 `dict[str, Any]` 的处理结果。
    """
    module_name = state["module_name"]
    analysis = dict(state.get("analysis") or {})
    analysis.update(
        {
            "module": module_name,
            "confidence": "high" if state["domain_relevance"] >= 0.75 else "medium",
            "transition_type": state.get("transition_type"),
        }
    )
    return {
        "analysis": analysis,
        "node_trace": service._trace(state, "issue_localizer", f"module={module_name}"),
    }

