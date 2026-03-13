from __future__ import annotations

from typing import Any


def run(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """处理“暂不需要代码实现”的多轮控制响应。"""
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

