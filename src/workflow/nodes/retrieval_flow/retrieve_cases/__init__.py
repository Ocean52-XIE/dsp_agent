# -*- coding: utf-8 -*-
"""
该模块实现工作流节点`retrieve_cases` 的处理逻辑，负责读取状态并输出增量结果。

提供两种接口：
- run_with_retriever: 解耦版本，直接接收 retriever 参数
- run: 兼容接口，从 service 获取 retriever
"""
from __future__ import annotations

"""案例检索节点（当前为可扩展占位实现）。"""

from typing import Any, Callable

# 导出 CaseRetriever 类和单例函数
from workflow.nodes.retrieval_flow.retrieve_cases.case_retriever import (
    CaseRetriever,
    get_case_retriever,
    set_case_retriever,
)

__all__ = [
    "run_with_retriever",
    "run",
    "CaseRetriever",
    "get_case_retriever",
    "set_case_retriever",
]


def run_with_retriever(
    retriever: Any | None,
    state: dict[str, Any],
    trace_fn: Callable[[dict[str, Any], str, str], list[dict[str, str]]] | None = None,
) -> dict[str, Any]:
    """执行案例检索（解耦版本）

    直接接收 retriever 参数，不依赖 service 对象。
    适用于子图内部调用场景。

    Args:
        retriever: 案例检索器实例（可为 None）
        state: 工作流状态字典
        trace_fn: 追踪函数（可选），签名为 (state, node_name, summary) -> node_trace

    Returns:
        状态增量字典
    """
    # 构建 node_trace 的辅助函数
    def build_trace(summary: str) -> list[dict[str, str]]:
        if trace_fn:
            return trace_fn(state, "retrieve_cases", summary)
        existing_trace = list(state.get("node_trace", []) or [])
        return existing_trace + [{"node": "retrieve_cases", "summary": summary}]

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
            "node_trace": build_trace("disabled_by_plan"),
        }

    items: list[dict[str, Any]] = []

    # 如果有 retriever，则调用其 search 方法
    if retriever is not None:
        user_query = state.get("user_query", "")
        top_k = int(retrieval_plan.get("case_top_k", 2))
        items = retriever.search(query=user_query, top_k=top_k)

    return {
        "case_hits": items,
        "case_retrieval_grade": "insufficient" if not items else "medium",
        "case_retrieval_profile": {
            "latency_ms": 0.0,
            "hits": len(items),
            "top_k": int(retrieval_plan.get("case_top_k", 0)),
            "strategy": retrieval_plan.get("strategy", "unknown"),
        },
        "node_trace": build_trace(f"hits={len(items)}"),
    }


def run(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """执行案例检索（兼容接口）

    从全局单例获取 retriever，不再依赖 service 传递。

    Args:
        service: 工作流服务对象，需要提供 _trace 方法
        state: 工作流状态字典

    Returns:
        状态增量字典
    """
    # 从全局单例获取 retriever
    from workflow.nodes.retrieval_flow.retrieve_cases.case_retriever import get_case_retriever

    return run_with_retriever(
        retriever=get_case_retriever(),
        state=state,
        trace_fn=lambda s, n, d: service._trace(s, n, d),
    )
