from __future__ import annotations

from typing import Any


def run(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """问题分析第一步：定位候选模块。"""
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

