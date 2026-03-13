from __future__ import annotations

from typing import Any


def run(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """恢复代码生成所需的前置分析上下文。"""
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
