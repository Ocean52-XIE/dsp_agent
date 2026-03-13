from __future__ import annotations

from typing import Any


def run(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """临时把所有普通输入都归类为业务知识问答。"""
    route = "knowledge_qa"
    return {
        "route": route,
        "node_trace": service._trace(state, "intent_classifier", f"route={route}"),
    }

