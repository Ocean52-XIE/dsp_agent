# -*- coding: utf-8 -*-
"""
该模块实现工作流节点`decline_code_generation_response` 的处理逻辑，负责读取状态并输出增量结果。
"""
from __future__ import annotations

from typing import Any


def run(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """
    执行`decline_code_generation_response` 节点主流程，基于输入状态计算并返回状态增量。
    
    参数:
        service: 工作流服务对象，提供检索、路由、日志与配置能力。
        state: 工作流状态字典，包含会话上下文与中间结果。
    
    返回:
        返回类型为 `dict[str, Any]` 的处理结果。
    """
    module_name = state.get("active_module_name") or state["module_name"]
    analysis = dict(state.get("last_analysis_result") or state.get("active_issue_context") or {})
    if analysis:
        analysis["task_stage"] = "issue_analysis"

    return {
        "route": "issue_analysis",
        "response_kind": "conversation_control",
        "task_stage": "issue_analysis",
        "status": "completed",
        "next_action": "completed",
        "citations": state.get("last_analysis_citations", []),
        "analysis": analysis or None,
        "answer": (
            f"已记录当前 `{module_name}` 只保留问题分析结论，不进入代码生成阶段。"
            "如果你后面改变主意，可以继续追问实现思路或再次明确提出需要代码实现。"
        ),
        "node_trace": service._trace(
            state,
            "decline_code_generation_response",
            "transition=decline_code_generation",
        ),
    }

