from __future__ import annotations

"""Wiki 检索节点（支持动态 TopK 与低置信重试）。"""

from typing import Any

from workflow.nodes.retrieval_retry import dedupe_normalized_queries, run_with_retry
from workflow.utils import env_float, env_int


def _grade_wiki_hits(items: list[dict[str, Any]]) -> str:
    """基于 Top1 分数与命中数量给出检索质量分级。"""
    if not items:
        return "insufficient"
    top1_score = float(items[0].get("score", 0.0))
    if top1_score >= 6.0 and len(items) >= 2:
        return "high"
    if top1_score >= 3.0:
        return "medium"
    return "low"


def _build_retry_queries(state: dict[str, Any]) -> list[str]:
    """构建重试查询：在原 query 基础上补充高召回表达。"""
    module_name = str(state.get("module_name", "")).strip()
    user_query = str(state.get("user_query", "")).strip()
    queries = [
        *state.get("retrieval_queries", []),
        f"{module_name} 核心链路 关键指标",
        f"{module_name} 业务口径 常见问题",
        f"{user_query} 指标 口径 流程",
    ]
    return dedupe_normalized_queries(queries)


def run(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """执行 Wiki 检索，并在低置信时自动扩召回重试。"""
    retrieval_plan = state.get("retrieval_plan", {})
    if not retrieval_plan.get("enable_wiki", True):
        return {
            "wiki_hits": [],
            "wiki_retrieval_grade": "disabled",
            "wiki_retrieval_profile": {
                "latency_ms": 0.0,
                "hits": 0,
                "top_k": 0,
                "strategy": retrieval_plan.get("strategy", "unknown"),
                "retried": False,
            },
            "node_trace": service._trace(state, "retrieve_wiki", "disabled_by_plan"),
        }

    top_k = int(retrieval_plan.get("wiki_top_k", 4))
    retry_multiplier = env_int("WORKFLOW_WIKI_RETRY_TOPK_MULTIPLIER", 2, minimum=1)
    retry_max_top_k = env_int("WORKFLOW_WIKI_RETRY_MAX_TOPK", 14, minimum=1)
    retry_min_top1 = env_float("WORKFLOW_WIKI_RETRY_MIN_TOP1", 3.0, minimum=0.0)

    result = run_with_retry(
        top_k=top_k,
        retry_multiplier=retry_multiplier,
        retry_max_top_k=retry_max_top_k,
        base_queries=list(state.get("retrieval_queries", [])),
        retry_queries=_build_retry_queries(state),
        search=lambda current_top_k, queries: service._wiki_retriever.search(
            user_query=state["user_query"],
            retrieval_queries=queries,
            module_name=state["module_name"],
            top_k=current_top_k,
        ),
        grade=_grade_wiki_hits,
        should_retry=lambda first_grade, first_top1: first_grade in {"insufficient", "low"} or first_top1 < retry_min_top1,
    )

    profile = {
        "latency_ms": result.latency_ms,
        "hits": len(result.final_items),
        "top_k": result.final_top_k,
        "initial_top_k": result.initial_top_k,
        "strategy": retrieval_plan.get("strategy", "unknown"),
        "retried": result.retried,
        "first_grade": result.first_grade,
        "final_grade": result.final_grade,
        "first_top1": round(result.first_top1, 4),
    }
    return {
        "wiki_hits": result.final_items,
        "wiki_retrieval_grade": result.final_grade,
        "wiki_retrieval_profile": profile,
        "node_trace": service._trace(
            state,
            "retrieve_wiki",
            (
                f"hits={len(result.final_items)},grade={result.final_grade},"
                f"retried={result.retried},latency_ms={result.latency_ms}"
            ),
        ),
    }
