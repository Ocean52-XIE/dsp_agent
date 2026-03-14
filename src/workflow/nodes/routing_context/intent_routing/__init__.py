# -*- coding: utf-8 -*-
"""Unified intent routing node."""
from __future__ import annotations

from functools import lru_cache
import re
from typing import Any

from workflow.common.node_trace import append_node_trace

_ISSUE_SIGNAL_RE = re.compile(
    r"(报错|错误|异常|失败|超时|告警|报警|traceback|error|exception|\b[45]\d{2}\b|panic|fatal)",
    re.IGNORECASE,
)


@lru_cache(maxsize=16)
def _compile_pattern(pattern: str, fallback: str) -> re.Pattern[str]:
    raw = (pattern or "").strip() or fallback
    return re.compile(raw)


def _normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term and term in text for term in terms)


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

    laugh_like = _compile_pattern(profile.laugh_like_regex, r"^[哈哈呵呵嘻嘻啊哦嗯\s!？?.。~～]+$")
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
    code_request = _looks_like_code_generation_request(user_query)
    has_active_module = bool(str(state.get("active_module_name", "") or "").strip())

    relevance = min(1.0, domain_hits * 0.22 + (0.22 if code_hint else 0.0))
    if domain_hits > 0:
        relevance = max(relevance, profile.weak_in_scope_min_score)
    if code_hint:
        relevance = max(relevance, profile.weak_code_hint_min_score)
    if code_request and has_active_module:
        relevance = max(relevance, profile.weak_code_hint_min_score)

    if str(state.get("active_topic_source", "")) == "history_memory":
        relevance = min(1.0, relevance + profile.history_memory_bonus)

    if off_hits > 0 and domain_hits == 0 and not code_hint:
        return max(0.0, relevance - profile.offtopic_penalty), False, f"offtopic_hits={off_hits}"

    compact = normalized.replace(" ", "")
    if len(compact) <= profile.short_query_max_len and domain_hits == 0 and not code_hint:
        return max(0.0, relevance - profile.short_query_penalty), False, "short_non_domain"

    in_scope = relevance >= profile.threshold
    reason = (
        f"domain_hits={domain_hits}, off_hits={off_hits}, code_hint={int(code_hint)}, "
        f"code_request={int(code_request)}, has_active_module={int(has_active_module)}, "
        f"source={state.get('active_topic_source', 'current_query')}"
    )
    return relevance, in_scope, reason


def _has_incident_detail_signal(normalized_query: str) -> bool:
    detail_terms = (
        "日志",
        "log",
        "trace",
        "stack",
        "request_id",
        "trace_id",
        "campaign",
        "adset",
        "线上",
        "离线",
        "告警",
        "报警",
        "timeout",
        "5xx",
        "4xx",
    )
    has_detail_term = _contains_any(normalized_query, detail_terms)
    has_number = re.search(r"\d", normalized_query) is not None
    has_time_window = re.search(r"(\d{1,2}:\d{2}|\d{4}-\d{1,2}-\d{1,2})", normalized_query) is not None
    return has_detail_term or has_number or has_time_window


def _is_generic_troubleshoot_question(normalized_query: str, *, has_troubleshoot_terms: bool) -> bool:
    if not has_troubleshoot_terms:
        return False
    method_terms = (
        "怎么排查",
        "如何排查",
        "排查思路",
        "排查步骤",
        "排查手册",
        "排障手册",
        "排查方法",
        "排查流程",
        "排查建议",
        "怎么定位",
        "如何定位",
        "怎么处理",
        "如何处理",
        "how to troubleshoot",
        "troubleshooting steps",
        "troubleshoot guide",
        "how to locate",
        "how to fix",
    )
    asks_for_method = _contains_any(normalized_query, method_terms)
    if not asks_for_method:
        return False
    return not _has_incident_detail_signal(normalized_query)


def _classify_in_scope_intent(service: Any, user_query: str) -> str:
    normalized_query = _normalize(user_query)
    intent_terms = service.domain_profile.query_rewrite.intent_terms

    if _looks_like_code_generation_request(user_query):
        return "code_generation"

    has_troubleshoot_terms = _contains_any(normalized_query, intent_terms.get("troubleshoot", ()))
    has_issue_regex_hit = _ISSUE_SIGNAL_RE.search(normalized_query) is not None
    is_generic_troubleshoot = _is_generic_troubleshoot_question(
        normalized_query,
        has_troubleshoot_terms=has_troubleshoot_terms,
    )

    if has_issue_regex_hit:
        return "issue_analysis"
    if has_troubleshoot_terms and not is_generic_troubleshoot:
        return "issue_analysis"
    return "knowledge_qa"


def _has_analysis_context(state: dict[str, Any]) -> bool:
    return bool(state.get("last_analysis_result"))


def run(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    user_query = str(state.get("user_query", "") or "")
    relevance, in_scope, reason = _compute_domain_relevance(service, state)

    if not in_scope:
        return {
            "domain_relevance": relevance,
            "route": "out_of_scope",
            "node_trace": append_node_trace(
                state,
                "intent_routing",
                f"route=out_of_scope, relevance={relevance:.2f}, reason={reason}",
            ),
        }

    route = _classify_in_scope_intent(service, user_query)
    has_analysis_context = _has_analysis_context(state)
    return {
        "domain_relevance": relevance,
        "route": route,
        "node_trace": append_node_trace(
            state,
            "intent_routing",
            f"route={route}, has_analysis_context={int(has_analysis_context)}, relevance={relevance:.2f}",
        ),
    }


def _looks_like_code_generation_request(text: str) -> bool:
    terms = (
        "给我代码",
        "给出代码",
        "直接给代码",
        "代码实现",
        "实现一个",
        "写一下代码",
        "补丁",
        "patch",
        "改代码",
        "修改代码",
        "直接修",
        "生成代码",
        "code snippet",
        "show me code",
        "write code",
    )
    lowered = str(text or "").lower()
    original = str(text or "")
    return any(term in original or term in lowered for term in terms)
