from __future__ import annotations

"""案例检索节点（当前为可扩展占位实现）。"""

from typing import Any


def run(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """按检索计划决定是否执行 case 检索。

    当前阶段尚未接入真实 case 索引，因此仅保留统一接口。
    后续接入时可直接在此节点替换 `items` 的生成逻辑。
    """
    retrieval_plan = state.get("retrieval_plan", {})
    if not retrieval_plan.get("enable_cases", False):
        return {
            "case_hits": [],
            "case_retrieval_grade": "disabled",
            "case_retrieval_profile": {
                "latency_ms": 0.0,
                "hits": 0,
                "top_k": 0,
                "strategy": retrieval_plan.get("strategy", "unknown"),
            },
            "node_trace": service._trace(state, "retrieve_cases", "disabled_by_plan"),
        }

    items: list[dict[str, Any]] = []
    return {
        "case_hits": items,
        "case_retrieval_grade": "insufficient" if not items else "medium",
        "case_retrieval_profile": {
            "latency_ms": 0.0,
            "hits": len(items),
            "top_k": int(retrieval_plan.get("case_top_k", 0)),
            "strategy": retrieval_plan.get("strategy", "unknown"),
        },
        "node_trace": service._trace(state, "retrieve_cases", f"hits={len(items)}"),
    }
