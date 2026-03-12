from __future__ import annotations

"""代码检索节点（支持动态 TopK 与低置信重试）。"""

import os
from typing import Any


def _grade_code_hits(items: list[dict[str, Any]], *, high_threshold: float, medium_threshold: float) -> str:
    """基于分数阈值给出代码检索质量分级。"""
    if not items:
        return "insufficient"
    top1_score = float(items[0].get("score", 0.0))
    if top1_score >= high_threshold and len(items) >= 2:
        return "high"
    if top1_score >= medium_threshold:
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


def _build_retry_queries(state: dict[str, Any]) -> list[str]:
    """构建代码检索重试查询。"""
    module_name = str(state.get("module_name", "")).strip()
    user_query = str(state.get("user_query", "")).strip()
    queries = [
        *state.get("retrieval_queries", []),
        f"{module_name} 实现 入口 函数",
        f"{module_name} 关键参数 校验",
        f"{user_query} 代码位置 文件路径",
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
    """执行代码检索，并在低置信时自动扩召回重试。"""
    retrieval_plan = state.get("retrieval_plan", {})
    if not retrieval_plan.get("enable_code", True):
        return {
            "code_hits": [],
            "code_retrieval_grade": "disabled",
            "code_retrieval_profile": {
                "latency_ms": 0.0,
                "hits": 0,
                "top_k": 0,
                "strategy": retrieval_plan.get("strategy", "unknown"),
                "retried": False,
            },
            "node_trace": service._trace(state, "retrieve_code", "disabled_by_plan"),
        }

    config = service._code_retriever.runtime_config
    top_k = int(retrieval_plan.get("code_top_k", 4))
    retry_multiplier = _env_int("WORKFLOW_CODE_RETRY_TOPK_MULTIPLIER", 2, 1)
    retry_max_top_k = _env_int("WORKFLOW_CODE_RETRY_MAX_TOPK", 14, 1)

    first_items = service._code_retriever.search(
        user_query=state["user_query"],
        retrieval_queries=state.get("retrieval_queries", []),
        module_name=state["module_name"],
        top_k=top_k,
    )
    first_grade = _grade_code_hits(
        first_items,
        high_threshold=float(config.grade_high_top1_threshold),
        medium_threshold=float(config.grade_medium_top1_threshold),
    )
    first_top1 = float(first_items[0].get("score", 0.0)) if first_items else 0.0

    should_retry = first_grade in {"insufficient", "low"}
    used_retry = False
    final_items = first_items
    final_top_k = top_k

    if should_retry:
        used_retry = True
        final_top_k = min(max(top_k * retry_multiplier, top_k + 2), retry_max_top_k)
        retry_items = service._code_retriever.search(
            user_query=state["user_query"],
            retrieval_queries=_build_retry_queries(state),
            module_name=state["module_name"],
            top_k=final_top_k,
        )
        retry_grade = _grade_code_hits(
            retry_items,
            high_threshold=float(config.grade_high_top1_threshold),
            medium_threshold=float(config.grade_medium_top1_threshold),
        )
        retry_top1 = float(retry_items[0].get("score", 0.0)) if retry_items else 0.0
        grade_rank = {"high": 4, "medium": 3, "low": 2, "insufficient": 1, "disabled": 0}
        if grade_rank.get(retry_grade, 0) > grade_rank.get(first_grade, 0):
            final_items = retry_items
        elif retry_grade == first_grade and retry_top1 > first_top1 + 1e-6:
            final_items = retry_items

    final_grade = _grade_code_hits(
        final_items,
        high_threshold=float(config.grade_high_top1_threshold),
        medium_threshold=float(config.grade_medium_top1_threshold),
    )
    profile = dict(service._code_retriever.last_search_profile)
    profile.update(
        {
            "hits": len(final_items),
            "top_k": final_top_k,
            "initial_top_k": top_k,
            "strategy": retrieval_plan.get("strategy", "unknown"),
            "retried": used_retry,
            "first_grade": first_grade,
            "final_grade": final_grade,
            "first_top1": round(first_top1, 4),
        }
    )

    return {
        "code_hits": final_items,
        "code_retrieval_grade": final_grade,
        "code_retrieval_profile": profile,
        "node_trace": service._trace(
            state,
            "retrieve_code",
            f"hits={len(final_items)},grade={final_grade},retried={used_retry},latency_ms={profile.get('latency_ms', 0.0)}",
        ),
    }
