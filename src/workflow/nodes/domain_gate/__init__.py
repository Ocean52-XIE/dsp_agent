from __future__ import annotations

import re
from functools import lru_cache
from typing import Any


@lru_cache(maxsize=16)
def _compile_pattern(pattern: str, fallback: str) -> re.Pattern[str]:
    raw = (pattern or "").strip() or fallback
    return re.compile(raw)


def _normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _count_hits(text: str, terms: tuple[str, ...]) -> int:
    return sum(1 for term in terms if term and term in text)


def _is_small_talk(text: str, *, exact: tuple[str, ...], substr: tuple[str, ...], laugh_like: re.Pattern[str]) -> bool:
    if not text:
        return True
    if text in exact:
        return True
    if any(token in text for token in substr):
        return True
    return laugh_like.fullmatch(text) is not None


def _compute_domain_relevance(service: Any, state: dict[str, Any]) -> tuple[float, bool, str]:
    profile = service.domain_profile.domain_gate

    user_query = str(state.get("user_query", "") or "")
    normalized = _normalize(user_query)

    laugh_like = _compile_pattern(profile.laugh_like_regex, r"^[哈呵嘿啊嗯哦嗨~\s!！?？,，。…]+$")
    code_hint_re = _compile_pattern(profile.code_hint_regex, r"[A-Za-z_][A-Za-z0-9_]{2,}\s*(\(|\.py\b|/)")

    if _is_small_talk(
        normalized,
        exact=profile.small_talk_exact,
        substr=profile.small_talk_substr,
        laugh_like=laugh_like,
    ):
        return 0.0, False, "small_talk"

    domain_hits = _count_hits(normalized, profile.domain_terms)
    off_hits = _count_hits(normalized, profile.offtopic_terms)
    code_hint = code_hint_re.search(user_query) is not None

    relevance = min(1.0, domain_hits * 0.22 + (0.22 if code_hint else 0.0))
    if domain_hits > 0:
        relevance = max(relevance, profile.weak_in_scope_min_score)
    if code_hint:
        relevance = max(relevance, profile.weak_code_hint_min_score)

    if str(state.get("active_topic_source", "")) == "history_memory":
        relevance = min(1.0, relevance + profile.history_memory_bonus)

    if off_hits > 0 and domain_hits == 0 and not code_hint:
        return max(0.0, relevance - profile.offtopic_penalty), False, f"offtopic_hits={off_hits}"

    compact = normalized.replace(" ", "")
    if len(compact) <= profile.short_query_max_len and domain_hits == 0 and not code_hint:
        return max(0.0, relevance - profile.short_query_penalty), False, "short_non_domain"

    is_domain_related = relevance >= profile.threshold
    reason = (
        f"domain_hits={domain_hits}, off_hits={off_hits}, code_hint={int(code_hint)}, "
        f"source={state.get('active_topic_source', 'current_query')}"
    )
    return relevance, is_domain_related, reason


def run(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    relevance, is_domain_related, reason = _compute_domain_relevance(service, state)
    return {
        "domain_relevance": relevance,
        "is_domain_related": is_domain_related,
        "node_trace": service._trace(
            state,
            "domain_gate",
            f"relevance={relevance:.2f}, in_scope={is_domain_related}, {reason}",
        ),
    }
