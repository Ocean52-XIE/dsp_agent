from __future__ import annotations

from typing import Any


def run(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """临时把普通消息路径统一压到知识问答链路。"""
    current_route = "knowledge_qa"
    active_stage = state.get("active_task_stage", "idle")
    transition_type = "continue_knowledge_qa" if active_stage == "knowledge_qa" else "force_knowledge_qa_mode"
    return {
        "route": current_route,
        "execution_path": "retrieval_flow",
        "transition_type": transition_type,
        "task_stage": "knowledge_qa",
        "node_trace": service._trace(
            state,
            "conversation_transition",
            f"transition={transition_type}, path=retrieval_flow",
        ),
    }

