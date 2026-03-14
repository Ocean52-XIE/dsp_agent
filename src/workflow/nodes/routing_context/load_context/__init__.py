# -*- coding: utf-8 -*-
"""Load context node."""
from __future__ import annotations

import re
from typing import Any

from workflow.common.domain_profile import DomainProfile
from workflow.common.node_trace import append_node_trace
from workflow.nodes.routing_context.load_context.helpers import (
    extract_module_from_message,
    latest_message_by,
)


_FOLLOWUP_PRONOUN_PREFIX_RE = re.compile(
    r"^(?:对应(?:的)?|这个(?:问题)?|这块|这段|上面(?:这个)?|上述|该)\s*",
    re.IGNORECASE,
)


def _extract_previous_user_query(history: list[dict[str, Any]], *, current_user_query: str) -> str:
    """
    从历史消息中提取“上一轮用户问题”。
    说明：
        API 正常链路下，history 已包含“当前 user 消息”，因此上一轮问题通常是倒数第二条 user 消息。
        若调用方未把当前消息放入 history，则回退使用最后一条 user 消息。
    """
    user_queries = [
        str(message.get("content", "") or "").strip()
        for message in history
        if message.get("role") == "user" and str(message.get("content", "") or "").strip()
    ]
    if not user_queries:
        return ""
    if len(user_queries) >= 2:
        return user_queries[-2]

    only_query = user_queries[-1]
    if only_query != str(current_user_query or "").strip():
        return only_query
    return ""


def _strip_trailing_punctuation(text: str) -> str:
    """去除句尾标点，避免拼接后的补全文本出现重复问句标记。"""
    return str(text or "").strip().rstrip("，,。.!！?？；;")


def _normalize_followup_action(current_query: str, *, domain_profile: DomainProfile) -> str:
    """
    归一化追问动作描述。
    说明：
        语义不全追问里常出现“对应的/这个/上述”这类代词前缀，这里统一裁剪后再判断动作类型，
        使补全后的 query 更稳定可检索。
    """
    raw = str(current_query or "").strip()
    if not raw:
        return ""

    stripped = _FOLLOWUP_PRONOUN_PREFIX_RE.sub("", raw).strip() or raw
    normalized = domain_profile.normalize_query_text(stripped)

    if ("文件" in normalized or "路径" in normalized) and ("哪里" in normalized or "在哪" in normalized):
        return "对应的文件路径在哪里"
    if "函数" in normalized and ("哪里" in normalized or "在哪" in normalized or "哪个" in normalized):
        return "对应的函数在哪里"
    if "实现" in normalized and ("哪里" in normalized or "在哪" in normalized):
        return "对应的代码实现在哪里"
    if "代码" in normalized and ("哪里" in normalized or "在哪" in normalized):
        return "对应的代码在哪里"
    return stripped


def _query_has_explicit_target(text: str, *, domain_profile: DomainProfile) -> bool:
    """
    判断追问本身是否已经包含明确目标。
    说明：
        若当前 query 已显式点名模块、函数、文件或符号，则无需做“上一问目标 + 当前动作”的补全，
        避免过度改写。
    """
    query = str(text or "").strip()
    if not query:
        return False

    inferred_module_name, _ = domain_profile.infer_module(query)
    if inferred_module_name and inferred_module_name != domain_profile.default_module:
        return True

    related = domain_profile.infer_related_modules(
        query,
        primary_module_name=inferred_module_name,
        limit=1,
    )
    if related:
        return True

    return bool(re.search(r"[A-Za-z_][A-Za-z0-9_]{2,}\s*(\(|\.py\b|/)", query))


def _rewrite_followup_query_if_needed(
    *,
    user_query: str,
    history: list[dict[str, Any]],
    domain_profile: DomainProfile,
) -> tuple[str, str, str]:
    """
    对语义不全的追问做前置补全。
    返回：
        (effective_query, original_query, rewrite_mode)
    说明：
        只在“代码定位类 + 语义不全 + 可从上一轮恢复目标”的场景触发，避免无关 query 被改写。
    """
    original_query = str(user_query or "").strip()
    if not original_query:
        return "", "", "original"

    normalized = domain_profile.normalize_query_text(original_query)
    if not domain_profile.is_code_location_query(normalized):
        return original_query, original_query, "original"

    if _query_has_explicit_target(original_query, domain_profile=domain_profile):
        return original_query, original_query, "original"

    previous_user_query = _extract_previous_user_query(history, current_user_query=original_query)
    if not previous_user_query:
        return original_query, original_query, "original"

    compact = normalized.replace(" ", "")
    has_followup_marker = any(
        token in normalized
        for token in ("对应", "这个", "上述", "上面", "这块", "这段")
    )
    if len(compact) > 24 and not has_followup_marker and not domain_profile.is_pronoun_followup(original_query):
        return original_query, original_query, "original"

    target = _strip_trailing_punctuation(previous_user_query)
    action = _normalize_followup_action(original_query, domain_profile=domain_profile)
    if not target or not action:
        return original_query, original_query, "original"

    effective_query = f"{target}，{action}"
    return effective_query, original_query, "followup_expanded"


def _normalize_related_modules(value: Any) -> list[dict[str, str]]:
    """
    归一化历史消息中的 `related_modules` 结构。

    说明：
        历史消息可能来自不同节点或不同版本，字段结构未必完全一致。
        这里统一转换成 `module_name/module_hint` 的最小结构，便于追问场景复用。
    """
    if not isinstance(value, list):
        return []

    rows: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        module_name = str(item.get("module_name", "")).strip()
        module_hint = str(item.get("module_hint", "")).strip()
        if not module_name:
            continue
        rows.append(
            {
                "module_name": module_name,
                "module_hint": module_hint,
            }
        )
    return rows


def _merge_related_modules(
    *,
    history_related_modules: list[dict[str, str]],
    current_related_modules: list[dict[str, str]],
    primary_module_name: str,
    limit: int = 2,
) -> list[dict[str, str]]:
    """
    合并历史相关模块与当前显式命中的相关模块。

    说明：
        对“代码在哪里”这类追问，当前 query 往往不再显式包含上一轮的跨模块关键词，
        因此需要优先保留历史 related_modules，再把当前命中的模块补进去。
    """
    primary_normalized = str(primary_module_name or "").strip().lower()
    merged: list[dict[str, str]] = []
    seen: set[str] = set()

    for item in [*history_related_modules, *current_related_modules]:
        module_name = str(item.get("module_name", "")).strip()
        module_hint = str(item.get("module_hint", "")).strip()
        normalized_name = module_name.lower()
        if not module_name or normalized_name == primary_normalized or normalized_name in seen:
            continue
        seen.add(normalized_name)
        merged.append(
            {
                "module_name": module_name,
                "module_hint": module_hint,
            }
        )
        if len(merged) >= limit:
            break
    return merged


def _looks_like_context_dependent_followup(text: str, *, domain_profile: DomainProfile) -> bool:
    """
    判断当前输入是否像依赖上一轮主题的短追问。

    说明：
        这里保留 pronoun / “代码在哪里” / “哪个文件” 等轻量启发式，
        目的是让短追问可以稳定复用上一轮的主模块。
    """
    if domain_profile.is_pronoun_followup(text):
        return True

    normalized = domain_profile.normalize_query_text(text)
    compact = normalized.replace(" ", "")
    if not compact:
        return False

    if domain_profile.is_code_location_query(normalized):
        return True

    has_code_anchor = any(
        token in normalized
        for token in (
            "\u4ee3\u7801",  # code
            "\u51fd\u6570",  # function
            "\u6587\u4ef6",  # file
            "\u8def\u5f84",  # path
            "\u5b9e\u73b0",  # implementation
            "\u5165\u53e3",  # entrypoint
            "\u884c\u53f7",  # line number
        )
    )
    has_where_anchor = any(
        token in normalized
        for token in (
            "\u54ea\u91cc",  # where
            "\u5728\u54ea",  # at where
            "\u54ea\u4e2a",  # which
            "\u54ea\u4e00",  # which one
            "\u4f4d\u7f6e",  # location
        )
    )
    return has_code_anchor and has_where_anchor and len(compact) <= 16


def _resolve_user_query(state: dict[str, Any], *, history: list[dict[str, Any]]) -> tuple[str, str]:
    """
    解析本轮真实用户问题。

    说明：
        正常情况下优先使用 state 中的 `user_query`；
        若调用侧未显式带入，则从历史里回退到最近一条 user message。
    """
    user_query = str(state.get("user_query", "") or "").strip()
    if user_query:
        return user_query, "state"

    latest_user_message = latest_message_by(
        history,
        lambda message: message.get("role") == "user",
    )
    fallback_query = str((latest_user_message or {}).get("content", "") or "").strip()
    if fallback_query:
        return fallback_query, "history_fallback"
    return "", "empty_fallback"


def run(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """
    执行 `load_context` 节点主流程。

    说明：
        该节点负责从历史消息中恢复当前主模块、相关模块与最近分析上下文，
        为后续路由、检索改写与回答提供统一输入。
    """
    history_raw = state.get("history", [])
    history = history_raw if isinstance(history_raw, list) else []
    resolved_user_query, query_source = _resolve_user_query(state, history=history)
    user_query, original_user_query, query_rewrite_mode = _rewrite_followup_query_if_needed(
        user_query=resolved_user_query,
        history=history,
        domain_profile=service.domain_profile,
    )

    latest_qa_message = latest_message_by(
        history,
        lambda message: message.get("role") == "assistant" and message.get("intent") == "knowledge_qa",
    )
    latest_issue_message = latest_message_by(
        history,
        lambda message: message.get("role") == "assistant"
        and message.get("kind") == "issue_analysis"
        and message.get("analysis") is not None,
    )
    latest_context_message = latest_issue_message or latest_qa_message
    latest_related_modules_message = latest_message_by(
        history,
        lambda message: message.get("role") == "assistant"
        and isinstance((message.get("analysis") or {}).get("related_modules"), list)
        and len((message.get("analysis") or {}).get("related_modules", [])) > 0,
    )

    active_module_name, history_module_hint = extract_module_from_message(
        latest_context_message,
        domain_profile=service.domain_profile,
    )
    history_related_modules = _normalize_related_modules(
        ((latest_related_modules_message or {}).get("analysis") or {}).get("related_modules", []),
    )

    inferred_module_name, inferred_module_hint = service.domain_profile.infer_module(user_query)
    current_related_modules = service.domain_profile.infer_related_modules(
        user_query,
        primary_module_name=inferred_module_name,
        limit=2,
    )

    reuse_history_topic = False
    if active_module_name and _looks_like_context_dependent_followup(user_query, domain_profile=service.domain_profile):
        reuse_history_topic = True
    elif (
        active_module_name
        and service.domain_profile.is_code_location_query(user_query)
        and inferred_module_name == service.domain_profile.default_module
    ):
        # Generic code-location query without module keywords should stay on active topic.
        reuse_history_topic = True

    if reuse_history_topic:
        module_name = active_module_name
        module_hint = history_module_hint
        topic_source = "history_memory"
        related_modules = _merge_related_modules(
            history_related_modules=history_related_modules,
            current_related_modules=service.domain_profile.infer_related_modules(
                user_query,
                primary_module_name=module_name,
                limit=2,
            ),
            primary_module_name=module_name,
            limit=2,
        )
    else:
        module_name = inferred_module_name
        module_hint = inferred_module_hint
        topic_source = "current_query"
        related_modules = current_related_modules

    return {
        "user_query": user_query,
        "original_user_query": original_user_query,
        "query_rewrite_mode": query_rewrite_mode,
        "module_name": module_name,
        "module_hint": module_hint,
        "related_modules": related_modules,
        "active_topic_source": topic_source,
        "active_module_name": active_module_name or module_name,
        "last_analysis_result": (latest_issue_message or {}).get("analysis"),
        "last_analysis_citations": (latest_issue_message or {}).get("citations", []),
        "node_trace": append_node_trace(
            state,
            "load_context",
            (
                f"module={module_name}, related={len(related_modules)}, source={topic_source}, "
                f"query_source={query_source}, rewrite={query_rewrite_mode}"
            ),
        ),
    }
