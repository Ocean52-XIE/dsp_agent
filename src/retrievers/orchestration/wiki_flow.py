# -*- coding: utf-8 -*-
"""Wiki retrieval orchestration for the deep-agent runtime."""
from __future__ import annotations

import json
import logging
from typing import Any, Callable

from common.func_utils import env_float, env_int
from retrievers.orchestration.retry import dedupe_normalized_queries, run_with_retry

logger = logging.getLogger(__name__)

_NORMALIZED_HIGH_TOP1_THRESHOLD = 0.85
_NORMALIZED_MEDIUM_TOP1_THRESHOLD = 0.55
_LEGACY_HIGH_TOP1_THRESHOLD = 6.0
_LEGACY_MEDIUM_TOP1_THRESHOLD = 3.0


def _emit_event(event: str, **payload: Any) -> None:
    logger.info(
        "%s | %s",
        event,
        json.dumps(payload, ensure_ascii=False, default=str, separators=(",", ":")),
    )


def _uses_normalized_scores(score: float) -> bool:
    return -0.001 <= float(score) <= 1.001


def _grade_wiki_hits(items: list[dict[str, Any]]) -> str:
    if not items:
        return "insufficient"
    top1_score = float(items[0].get("score", 0.0))
    if _uses_normalized_scores(top1_score):
        high_threshold = _NORMALIZED_HIGH_TOP1_THRESHOLD
        medium_threshold = _NORMALIZED_MEDIUM_TOP1_THRESHOLD
    else:
        high_threshold = _LEGACY_HIGH_TOP1_THRESHOLD
        medium_threshold = _LEGACY_MEDIUM_TOP1_THRESHOLD
    if top1_score >= high_threshold and len(items) >= 2:
        return "high"
    if top1_score >= medium_threshold:
        return "medium"
    return "low"


def _should_retry_wiki(first_grade: str, first_top1: float, configured_retry_min_top1: float) -> bool:
    if first_grade not in {"insufficient", "low"}:
        return False
    if configured_retry_min_top1 >= 0:
        retry_min_top1 = configured_retry_min_top1
    elif _uses_normalized_scores(first_top1):
        retry_min_top1 = _NORMALIZED_MEDIUM_TOP1_THRESHOLD
    else:
        retry_min_top1 = _LEGACY_MEDIUM_TOP1_THRESHOLD
    return float(first_top1) < float(retry_min_top1)


def _build_retry_queries(state: dict[str, Any]) -> list[str]:
    module_name = str(state.get("module_name", "")).strip()
    user_query = str(state.get("user_query", "")).strip()
    queries = [
        *state.get("retrieval_queries", []),
        f"{module_name} 核心链路 关键指标",
        f"{module_name} 业务口径 常见问题",
        f"{user_query} 指标 口径 流程",
    ]
    return dedupe_normalized_queries(queries)


def _log_retrieval_details(
    trace_id: str,
    phase: str,
    *,
    input_queries: list[str],
    top_k: int,
    hits: list[dict[str, Any]],
    grade: str,
) -> None:
    if phase == "input":
        _emit_event(
            "retrieval.wiki.started",
            trace_id=trace_id,
            query_count=len(input_queries),
            queries=input_queries,
            top_k=top_k,
        )
        return

    for i, hit in enumerate(hits[:5], 1):
        _emit_event(
            "retrieval.wiki.hit",
            trace_id=trace_id,
            rank=i,
            score=round(float(hit.get("score", 0.0)), 4),
            path=hit.get("path", ""),
            section=hit.get("section", ""),
            excerpt_preview=str(hit.get("excerpt", "") or hit.get("content", ""))[:120].replace("\n", " "),
        )

    _emit_event(
        "retrieval.wiki.completed",
        trace_id=trace_id,
        hits=len(hits),
        grade=grade,
        scores=[round(h.get("score", 0), 4) for h in hits[:5]],
    )


def execute_wiki_retrieval(
    retriever: Any,
    state: dict[str, Any],
    trace_fn: Callable[[dict[str, Any], str, str], list[dict[str, str]]] | None = None,
) -> dict[str, Any]:
    trace_id = state.get("trace_id", "")
    retrieval_plan = state.get("retrieval_plan", {})

    def build_trace(summary: str) -> list[dict[str, str]]:
        if trace_fn:
            return trace_fn(state, "retrieve_wiki", summary)
        return [{"node": "retrieve_wiki", "summary": summary}]

    if not retrieval_plan.get("enable_wiki", True):
        _emit_event(
            "retrieval.wiki.disabled",
            trace_id=trace_id,
            reason="disabled_by_plan",
        )
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
            "node_trace": build_trace("disabled_by_plan"),
        }

    top_k = int(retrieval_plan.get("wiki_top_k", 4))
    retry_multiplier = env_int("AGENT_WIKI_RETRY_TOPK_MULTIPLIER", 2, minimum=1)
    retry_max_top_k = env_int("AGENT_WIKI_RETRY_MAX_TOPK", 14, minimum=1)
    configured_retry_min_top1 = env_float("AGENT_WIKI_RETRY_MIN_TOP1", -1.0, minimum=-1.0)
    base_queries = list(state.get("retrieval_queries", []))

    _log_retrieval_details(
        trace_id=trace_id,
        phase="input",
        input_queries=base_queries,
        top_k=top_k,
        hits=[],
        grade="",
    )

    result = run_with_retry(
        top_k=top_k,
        retry_multiplier=retry_multiplier,
        retry_max_top_k=retry_max_top_k,
        base_queries=base_queries,
        retry_queries=_build_retry_queries(state),
        search=lambda current_top_k, queries: retriever.search(
            user_query=state["user_query"],
            retrieval_queries=queries,
            module_name=state["module_name"],
            top_k=current_top_k,
        ),
        grade=_grade_wiki_hits,
        should_retry=lambda first_grade, first_top1: _should_retry_wiki(
            first_grade,
            first_top1,
            configured_retry_min_top1,
        ),
    )

    _log_retrieval_details(
        trace_id=trace_id,
        phase="output",
        input_queries=base_queries,
        top_k=result.final_top_k,
        hits=result.final_items,
        grade=result.final_grade,
    )

    if result.retried:
        _emit_event(
            "retrieval.wiki.retry",
            trace_id=trace_id,
            initial_top_k=result.initial_top_k,
            final_top_k=result.final_top_k,
            first_grade=result.first_grade,
            final_grade=result.final_grade,
            first_top1=round(result.first_top1, 4),
        )

    profile = dict(retriever.last_search_profile)
    profile.update(
        {
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
    )

    return {
        "wiki_hits": result.final_items,
        "wiki_retrieval_grade": result.final_grade,
        "wiki_retrieval_profile": profile,
        "node_trace": build_trace(
            f"hits={len(result.final_items)},grade={result.final_grade},"
            f"retried={result.retried},latency_ms={result.latency_ms}"
        ),
    }
