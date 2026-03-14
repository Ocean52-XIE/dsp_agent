# -*- coding: utf-8 -*-
"""Follow-up and query intent heuristics shared by routing nodes."""
from __future__ import annotations

from workflow.domain_profile import DomainProfile


def is_pronoun_followup(text: str) -> bool:
    pronouns = ("它", "这个", "这个问题", "那个", "那这个", "这块", "这里", "上面这个")
    normalized = str(text or "")
    return any(token in normalized for token in pronouns)


def looks_like_code_location_query(text: str, *, domain_profile: DomainProfile) -> bool:
    normalized = " ".join((text or "").strip().lower().split())
    if not normalized:
        return False

    code_location_terms = domain_profile.query_rewrite.intent_terms.get("code_location", ())
    if any(token and token.lower() in normalized for token in code_location_terms):
        return True

    fallback_terms = (
        "代码在哪里",
        "代码在哪",
        "代码位置",
        "实现在哪",
        "入口在哪",
        "哪个文件",
        "在哪个文件",
        "路径在哪",
        "哪一行",
        "line",
        "where is the code",
        "where is code",
    )
    return any(token in normalized for token in fallback_terms)


def looks_like_context_dependent_followup(text: str, *, domain_profile: DomainProfile) -> bool:
    if is_pronoun_followup(text):
        return True

    normalized = " ".join((text or "").strip().lower().split())
    compact = normalized.replace(" ", "")
    if not compact:
        return False

    if looks_like_code_location_query(normalized, domain_profile=domain_profile):
        return True

    has_code_anchor = any(token in normalized for token in ("代码", "函数", "文件", "路径", "实现", "入口", "行号"))
    has_where_anchor = any(token in normalized for token in ("哪里", "在哪", "哪个", "哪一", "位置"))
    return has_code_anchor and has_where_anchor and len(compact) <= 16
