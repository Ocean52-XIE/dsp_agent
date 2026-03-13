from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable


_GRADE_RANK = {"high": 4, "medium": 3, "low": 2, "insufficient": 1, "disabled": 0}


@dataclass
class RetrySearchResult:
    first_items: list[dict[str, Any]]
    final_items: list[dict[str, Any]]
    first_grade: str
    final_grade: str
    first_top1: float
    retried: bool
    initial_top_k: int
    final_top_k: int
    latency_ms: float


def dedupe_normalized_queries(queries: list[str], *, limit: int | None = None) -> list[str]:
    """Normalize whitespace, drop empty rows, and dedupe case-insensitively."""
    deduped: list[str] = []
    seen: set[str] = set()
    for query in queries:
        normalized = " ".join(str(query).strip().split())
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    if limit is None:
        return deduped
    return deduped[: max(0, limit)]


def run_with_retry(
    *,
    top_k: int,
    retry_multiplier: int,
    retry_max_top_k: int,
    base_queries: list[str],
    retry_queries: list[str],
    search: Callable[[int, list[str]], list[dict[str, Any]]],
    grade: Callable[[list[dict[str, Any]]], str],
    should_retry: Callable[[str, float], bool],
) -> RetrySearchResult:
    """Run retrieval once, optionally retry with expanded queries, and pick best result."""
    started = perf_counter()
    first_items = search(top_k, base_queries)
    first_grade = grade(first_items)
    first_top1 = float(first_items[0].get("score", 0.0)) if first_items else 0.0

    retried = False
    final_items = first_items
    final_top_k = top_k

    if should_retry(first_grade, first_top1):
        retried = True
        final_top_k = min(max(top_k * retry_multiplier, top_k + 2), retry_max_top_k)
        retry_items = search(final_top_k, retry_queries)
        retry_grade = grade(retry_items)
        retry_top1 = float(retry_items[0].get("score", 0.0)) if retry_items else 0.0

        if _GRADE_RANK.get(retry_grade, 0) > _GRADE_RANK.get(first_grade, 0):
            final_items = retry_items
        elif retry_grade == first_grade and retry_top1 > first_top1 + 1e-6:
            final_items = retry_items

    final_grade = grade(final_items)
    latency_ms = round((perf_counter() - started) * 1000, 3)
    return RetrySearchResult(
        first_items=first_items,
        final_items=final_items,
        first_grade=first_grade,
        final_grade=final_grade,
        first_top1=first_top1,
        retried=retried,
        initial_top_k=top_k,
        final_top_k=final_top_k,
        latency_ms=latency_ms,
    )

