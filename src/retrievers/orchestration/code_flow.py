# -*- coding: utf-8 -*-
"""Code retrieval orchestration for the deep-agent runtime."""
from __future__ import annotations

import json
import logging
from typing import Any, Callable

from common.func_utils import env_int
from retrievers.orchestration.retry import dedupe_normalized_queries, run_with_retry

logger = logging.getLogger(__name__)


def _emit_event(event: str, **payload: Any) -> None:
    logger.info(
        "%s | %s",
        event,
        json.dumps(payload, ensure_ascii=False, default=str, separators=(",", ":")),
    )


def _grade_code_hits(items: list[dict[str, Any]], *, high_threshold: float, medium_threshold: float) -> str:
    if not items:
        return "insufficient"
    top1_score = float(items[0].get("score", 0.0))
    if top1_score >= high_threshold and len(items) >= 2:
        return "high"
    if top1_score >= medium_threshold:
        return "medium"
    return "low"


def _build_retry_queries(state: dict[str, Any]) -> list[str]:
    module_name = str(state.get("module_name", "")).strip()
    user_query = str(state.get("user_query", "")).strip()
    queries = [
        *state.get("retrieval_queries", []),
        f"{module_name} 实现 入口 函数",
        f"{module_name} 关键参数 校验",
        f"{user_query} 代码位置 文件路径",
    ]
    return dedupe_normalized_queries(queries)


def _line_range(hit: dict[str, Any]) -> str:
    explicit = str(hit.get("line_range", "") or "").strip()
    if explicit:
        return explicit
    start_line = hit.get("start_line")
    end_line = hit.get("end_line")
    if isinstance(start_line, int) and isinstance(end_line, int):
        return f"{start_line}-{end_line}" if start_line != end_line else str(start_line)
    return ""


def _log_retrieval_input(trace_id: str, queries: list[str], top_k: int) -> None:
    _emit_event(
        "retrieval.code.started",
        trace_id=trace_id,
        query_count=len(queries),
        queries=queries,
        top_k=top_k,
    )


def _log_retrieval_output(trace_id: str, hits: list[dict[str, Any]], grade: str) -> None:
    _emit_event(
        "retrieval.code.completed",
        trace_id=trace_id,
        hits=len(hits),
        grade=grade,
        scores=[round(h.get("score", 0), 4) for h in hits[:5]],
    )

    for i, hit in enumerate(hits[:5], 1):
        _emit_event(
            "retrieval.code.hit",
            trace_id=trace_id,
            rank=i,
            score=round(float(hit.get("score", 0.0)), 4),
            path=hit.get("path", ""),
            symbol_name=hit.get("symbol_name", ""),
            line_range=_line_range(hit),
            code_preview=str(hit.get("content", ""))[:80].replace("\n", " "),
        )


def execute_code_retrieval(
    retriever: Any,
    state: dict[str, Any],
    trace_fn: Callable[[dict[str, Any], str, str], list[dict[str, str]]] | None = None,
) -> dict[str, Any]:
    trace_id = state.get("trace_id", "")
    retrieval_plan = state.get("retrieval_plan", {})

    def build_trace(summary: str) -> list[dict[str, str]]:
        if trace_fn:
            return trace_fn(state, "retrieve_code", summary)
        return [{"node": "retrieve_code", "summary": summary}]

    if not retrieval_plan.get("enable_code", True):
        _emit_event(
            "retrieval.code.disabled",
            trace_id=trace_id,
            reason="disabled_by_plan",
        )
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
            "node_trace": build_trace("disabled_by_plan"),
        }

    config = retriever.runtime_config
    top_k = int(retrieval_plan.get("code_top_k", 4))
    retry_multiplier = env_int("AGENT_CODE_RETRY_TOPK_MULTIPLIER", 2, minimum=1)
    retry_max_top_k = env_int("AGENT_CODE_RETRY_MAX_TOPK", 14, minimum=1)
    base_queries = list(state.get("retrieval_queries", []))

    _log_retrieval_input(trace_id, base_queries, top_k)

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
        grade=lambda items: _grade_code_hits(
            items,
            high_threshold=float(config.grade_high_top1_threshold),
            medium_threshold=float(config.grade_medium_top1_threshold),
        ),
        should_retry=lambda first_grade, _: first_grade in {"insufficient", "low"},
    )

    _log_retrieval_output(trace_id, result.final_items, result.final_grade)

    if result.retried:
        _emit_event(
            "retrieval.code.retry",
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
        "code_hits": result.final_items,
        "code_retrieval_grade": result.final_grade,
        "code_retrieval_profile": profile,
        "node_trace": build_trace(
            f"hits={len(result.final_items)},grade={result.final_grade},"
            f"retried={result.retried},latency_ms={profile.get('latency_ms', 0.0)}"
        ),
    }
