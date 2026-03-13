from __future__ import annotations

"""Wiki 检索节点（支持动态 TopK 与低置信重试）。"""

import os
from time import perf_counter
from typing import Any


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


def _env_int(name: str, default: int, minimum: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(int(raw), minimum)
    except ValueError:
        return default


def _env_float(name: str, default: float, minimum: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(float(raw), minimum)
    except ValueError:
        return default


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

    deduped: list[str] = []
    seen: set[str] = set()
    for query in queries:
        normalized = " ".join(query.strip().split())
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


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
    retry_multiplier = _env_int("WORKFLOW_WIKI_RETRY_TOPK_MULTIPLIER", 2, 1)
    retry_max_top_k = _env_int("WORKFLOW_WIKI_RETRY_MAX_TOPK", 14, 1)
    retry_min_top1 = _env_float("WORKFLOW_WIKI_RETRY_MIN_TOP1", 3.0, 0.0)

    started = perf_counter()
    first_items = service._wiki_retriever.search(
        user_query=state["user_query"],
        retrieval_queries=state.get("retrieval_queries", []),
        module_name=state["module_name"],
        top_k=top_k,
    )
    first_grade = _grade_wiki_hits(first_items)
    first_top1 = float(first_items[0].get("score", 0.0)) if first_items else 0.0

    should_retry = first_grade in {"insufficient", "low"} or first_top1 < retry_min_top1
    used_retry = False
    final_items = first_items
    final_top_k = top_k

    if should_retry:
        used_retry = True
        final_top_k = min(max(top_k * retry_multiplier, top_k + 2), retry_max_top_k)
        retry_items = service._wiki_retriever.search(
            user_query=state["user_query"],
            retrieval_queries=_build_retry_queries(state),
            module_name=state["module_name"],
            top_k=final_top_k,
        )

        retry_top1 = float(retry_items[0].get("score", 0.0)) if retry_items else 0.0
        retry_grade = _grade_wiki_hits(retry_items)
        # 以质量优先选择重试结果；若质量相同，则保留首轮结果避免抖动。
        grade_rank = {"high": 4, "medium": 3, "low": 2, "insufficient": 1, "disabled": 0}
        if grade_rank.get(retry_grade, 0) > grade_rank.get(first_grade, 0):
            final_items = retry_items
        elif retry_grade == first_grade and retry_top1 > first_top1 + 1e-6:
            final_items = retry_items

    latency_ms = round((perf_counter() - started) * 1000, 3)
    final_grade = _grade_wiki_hits(final_items)

    profile = {
        "latency_ms": latency_ms,
        "hits": len(final_items),
        "top_k": final_top_k,
        "initial_top_k": top_k,
        "strategy": retrieval_plan.get("strategy", "unknown"),
        "retried": used_retry,
        "first_grade": first_grade,
        "final_grade": final_grade,
        "first_top1": round(first_top1, 4),
    }
    return {
        "wiki_hits": final_items,
        "wiki_retrieval_grade": final_grade,
        "wiki_retrieval_profile": profile,
        "node_trace": service._trace(
            state,
            "retrieve_wiki",
            f"hits={len(final_items)},grade={final_grade},retried={used_retry},latency_ms={latency_ms}",
        ),
    }
