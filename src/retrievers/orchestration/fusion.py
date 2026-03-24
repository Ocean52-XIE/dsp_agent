# -*- coding: utf-8 -*-
"""Multi-source evidence fusion for wiki/code retrieval results."""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)


def _emit_event(event: str, **payload: Any) -> None:
    logger.info(
        "%s | %s",
        event,
        json.dumps(payload, ensure_ascii=False, default=str, separators=(",", ":")),
    )


GRADE_BIAS = {
    "high": 0.35,
    "medium": 0.15,
    "low": 0.0,
    "insufficient": -0.1,
    "disabled": -0.2,
}

INTENT_SOURCE_BIAS = {
    "code_location": {"code": 0.25, "wiki": -0.1},
    "wiki_first": {"wiki": 0.15, "code": -0.05},
}


def _as_source_key(source_type: str) -> str:
    normalized = source_type.strip().lower()
    if normalized.startswith("wiki"):
        return "wiki"
    if normalized.startswith("code"):
        return "code"
    if normalized.startswith("case"):
        return "case"
    return normalized or "unknown"


def _normalize_scores_within_source(hits: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    if not hits:
        return hits

    scores = [float(hit.get("score", 0.0)) for hit in hits]
    max_score = max(scores) if scores else 1.0
    min_score = min(scores) if scores else 0.0
    already_normalized = min_score >= -0.001 and max_score <= 1.001

    if already_normalized:
        for hit, score in zip(hits, scores):
            hit["normalized_score"] = max(0.0, min(1.0, score))
        logger.debug(
            "[merge_evidence] scores already normalized | source=%s | hits=%s | range=[%.4f, %.4f]",
            source,
            len(hits),
            min_score,
            max_score,
        )
        return hits

    score_range = max_score - min_score
    if score_range == 0:
        for hit in hits:
            hit["normalized_score"] = 0.5
        logger.debug(
            "[merge_evidence] fallback normalization all scores same | source=%s | hits=%s | score=%.4f",
            source,
            len(hits),
            max_score,
        )
    else:
        for hit, score in zip(hits, scores):
            hit["normalized_score"] = (score - min_score) / score_range
        logger.info(
            "[merge_evidence] fallback normalization | source=%s | hits=%s | raw_range=[%.4f, %.4f]",
            source,
            len(hits),
            min_score,
            max_score,
        )
    return hits


def _normalize_retrieval_plan(state: dict[str, Any]) -> dict[str, Any]:
    raw_plan = state.get("retrieval_plan", {})
    final_top_k = max(int(raw_plan.get("final_top_k", 5)), 1)
    raw_weights = raw_plan.get("source_weights", {})
    raw_max_per_source = raw_plan.get("max_per_source", {})
    normalized_plan = {
        "strategy": str(raw_plan.get("strategy", "hybrid")),
        "final_top_k": final_top_k,
        "source_weights": {
            "wiki": max(float(raw_weights.get("wiki", 1.0)), 0.0),
            "code": max(float(raw_weights.get("code", 1.0)), 0.0),
        },
        "max_per_source": {
            "wiki": max(int(raw_max_per_source.get("wiki", final_top_k)), 0),
            "code": max(int(raw_max_per_source.get("code", final_top_k)), 0),
        },
        "intent_profile": dict(raw_plan.get("intent_profile", {})),
    }
    return _apply_intent_overrides(state=state, plan=normalized_plan)


def _apply_intent_overrides(*, state: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    intent_profile = dict(plan.get("intent_profile", {}))
    is_code_location = bool(intent_profile.get("is_code_location"))
    is_wiki_intent = bool(intent_profile.get("is_wiki_intent"))

    if is_code_location:
        plan["strategy"] = "code_first"
        plan["source_weights"]["code"] = max(float(plan["source_weights"].get("code", 1.0)), 1.35)
        plan["source_weights"]["wiki"] = min(float(plan["source_weights"].get("wiki", 1.0)), 0.8)
        plan["max_per_source"]["code"] = max(int(plan["max_per_source"].get("code", 0)), 5)
        current_wiki_quota = max(int(plan["max_per_source"].get("wiki", 1)), 1)
        plan["max_per_source"]["wiki"] = min(current_wiki_quota, 2)
        return plan

    if plan.get("strategy") == "wiki_first" or (is_wiki_intent and not intent_profile.get("is_code_intent")):
        plan["source_weights"]["wiki"] = max(float(plan["source_weights"].get("wiki", 1.0)), 1.15)
        plan["source_weights"]["code"] = min(float(plan["source_weights"].get("code", 1.0)), 0.9)
        plan["max_per_source"]["wiki"] = max(int(plan["max_per_source"].get("wiki", 0)), 4)
    return plan


def _get_retrieval_grade(state: dict[str, Any], source: str) -> str:
    if source == "wiki":
        return str(state.get("wiki_retrieval_grade", "unknown"))
    if source == "code":
        return str(state.get("code_retrieval_grade", "unknown"))
    return "unknown"


def _build_candidate_items(state: dict[str, Any], plan: dict[str, Any]) -> list[dict[str, Any]]:
    trace_id = str(state.get("trace_id", "") or "")
    source_to_hits: dict[str, list[dict[str, Any]]] = {
        "wiki": list(state.get("wiki_hits", [])),
        "code": list(state.get("code_hits", [])),
    }

    for source, hits in source_to_hits.items():
        _normalize_scores_within_source(hits, source)

    candidates: list[dict[str, Any]] = []
    intent_profile = dict(plan.get("intent_profile", {}))
    is_code_location = bool(intent_profile.get("is_code_location"))
    is_wiki_first = plan.get("strategy") == "wiki_first"

    for source, hits in source_to_hits.items():
        source_weight = float(plan["source_weights"].get(source, 1.0))
        grade = _get_retrieval_grade(state, source)
        grade_bias = float(GRADE_BIAS.get(grade, 0.0))
        intent_bias = 0.0
        if is_code_location:
            intent_bias = float(INTENT_SOURCE_BIAS["code_location"].get(source, 0.0))
        elif is_wiki_first:
            intent_bias = float(INTENT_SOURCE_BIAS["wiki_first"].get(source, 0.0))

        for rank, hit in enumerate(hits, start=1):
            normalized_score = float(hit.get("normalized_score", 0.5))
            raw_score = float(hit.get("score", 0.0))
            rank_bonus = 0.12 / rank
            fused_score = normalized_score * source_weight + rank_bonus + grade_bias + intent_bias

            item = dict(hit)
            item_source = _as_source_key(str(item.get("source_type", source)))
            item["source_type"] = item_source
            item["fusion_score"] = round(fused_score, 6)
            item["fusion_debug"] = {
                "raw_score": round(raw_score, 4),
                "normalized_score": round(normalized_score, 4),
                "source_weight": round(source_weight, 4),
                "rank_bonus": round(rank_bonus, 4),
                "grade": grade,
                "grade_bias": round(grade_bias, 4),
                "intent_bias": round(intent_bias, 4),
                "fused_score": round(fused_score, 4),
            }
            candidates.append(item)

    candidates.sort(
        key=lambda item: (
            float(item.get("fusion_score", 0.0)),
            float(item.get("normalized_score", 0.0)),
        ),
        reverse=True,
    )

    if candidates:
        top_items = candidates[:6]
        fusion_summary = ", ".join(
            f"#{i + 1}[{item['source_type']}:{item.get('fusion_score', 0):.3f}]"
            for i, item in enumerate(top_items)
        )
        _emit_event(
            "retrieval.fusion.completed",
            trace_id=trace_id,
            candidate_count=len(candidates),
            top6=fusion_summary,
        )

    return candidates


def _dedup_key(item: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(item.get("source_type", "")),
        str(item.get("path", "")),
        str(item.get("section", "")),
    )


def _select_with_quota(candidates: list[dict[str, Any]], plan: dict[str, Any]) -> list[dict[str, Any]]:
    final_top_k = int(plan["final_top_k"])
    max_per_source = plan["max_per_source"]
    selected: list[dict[str, Any]] = []
    overflow: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str]] = set()
    source_counts: dict[str, int] = defaultdict(int)

    for item in candidates:
        key = _dedup_key(item)
        if key in seen_keys:
            continue
        seen_keys.add(key)

        source = _as_source_key(str(item.get("source_type", "")))
        quota = int(max_per_source.get(source, final_top_k))
        if source_counts[source] >= quota:
            overflow.append(item)
            continue

        source_counts[source] += 1
        selected.append(item)
        if len(selected) >= final_top_k:
            return selected

    for item in overflow:
        if len(selected) >= final_top_k:
            break
        selected.append(item)
    return selected


def run(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    plan = _normalize_retrieval_plan(state)
    candidates = _build_candidate_items(state, plan)
    citations = _select_with_quota(candidates, plan)

    selected_counts: dict[str, int] = defaultdict(int)
    for index, item in enumerate(citations, start=1):
        source = _as_source_key(str(item.get("source_type", "")))
        selected_counts[source] += 1
        item["fusion_rank"] = index

    profile = {
        "strategy": plan["strategy"],
        "final_top_k": plan["final_top_k"],
        "source_weights": plan["source_weights"],
        "max_per_source": plan["max_per_source"],
        "input_counts": {
            "wiki": len(state.get("wiki_hits", [])),
            "code": len(state.get("code_hits", [])),
        },
        "candidate_count": len(candidates),
        "selected_count": len(citations),
        "selected_counts_by_source": dict(selected_counts),
    }

    summary = (
        f"strategy={plan['strategy']},candidates={len(candidates)},"
        f"selected={len(citations)},wiki={selected_counts.get('wiki', 0)},"
        f"code={selected_counts.get('code', 0)}"
    )
    return {
        "citations": citations,
        "evidence_fusion_profile": profile,
        "node_trace": service._trace(state, "merge_evidence", summary),
    }
