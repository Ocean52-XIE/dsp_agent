# -*- coding: utf-8 -*-
"""Parse raw Deep Agent outputs into a runtime-neutral result model."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any


@dataclass(frozen=True)
class DeepAgentTurnResult:
    """Normalized result returned by the deep-agent service."""

    trace_id: str
    answer: str
    citations: list[dict[str, Any]]
    message_count: int
    status: str
    driver: str
    raw_result: Any
    intent: str | None = None
    analysis: dict[str, Any] = field(default_factory=dict)
    debug: dict[str, Any] = field(default_factory=dict)


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = str(item.get("text", "") or item.get("content", "")).strip()
                if text:
                    parts.append(text)
        return "\n".join(part for part in parts if part).strip()
    return str(content or "").strip()


def _message_role(message: Any) -> str:
    if isinstance(message, dict):
        role = message.get("type") or message.get("role")
        return str(role or "").strip().lower()
    role = getattr(message, "type", None) or getattr(message, "role", None)
    return str(role or "").strip().lower()


def _extract_message_content(message: Any) -> str:
    if isinstance(message, dict):
        return _extract_text(message.get("content", ""))
    return _extract_text(getattr(message, "content", ""))


def _extract_message_name(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("name", "") or "").strip()
    return str(getattr(message, "name", "") or "").strip()


def _preview_text(value: Any, *, max_chars: int = 160) -> str:
    text = _extract_text(value)
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}..."


def _parse_tool_payload(text: str) -> dict[str, Any] | None:
    normalized = str(text or "").strip()
    if not normalized:
        return None
    try:
        payload = json.loads(normalized)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _extract_tool_payloads(result: Any) -> list[dict[str, Any]]:
    if not isinstance(result, dict):
        return []
    messages = list(result.get("messages", []) or [])
    payloads: list[dict[str, Any]] = []
    for message in messages:
        if _message_role(message) != "tool":
            continue
        payload = _parse_tool_payload(_extract_message_content(message))
        if not payload:
            continue
        tool_name = _extract_message_name(message)
        if tool_name:
            payload = {**payload, "_tool_name": tool_name}
        payloads.append(payload)
    return payloads


def _normalize_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _normalize_float(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").strip()
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _normalize_citation_item(item: dict[str, Any]) -> dict[str, Any]:
    source_type = str(item.get("source_type", "") or item.get("source", "") or "source").strip() or "source"
    title = (
        str(item.get("title", "") or "").strip()
        or str(item.get("symbol_name", "") or "").strip()
        or str(item.get("path", "") or "").strip()
    )
    normalized = {
        "source": str(item.get("source", "") or source_type),
        "source_type": source_type,
        "path": str(item.get("path", "") or ""),
        "title": title,
        "section": str(item.get("section", "") or ""),
        "score": _normalize_float(item.get("score", 0.0)),
        "excerpt": str(item.get("excerpt", "") or ""),
        "symbol_name": str(item.get("symbol_name", "") or ""),
        "start_line": _normalize_int(item.get("start_line")),
        "end_line": _normalize_int(item.get("end_line")),
    }
    if "fusion_rank" in item:
        normalized["fusion_rank"] = item.get("fusion_rank")
    if isinstance(item.get("fusion_debug"), dict):
        normalized["fusion_debug"] = dict(item["fusion_debug"])
    if isinstance(item.get("retrieval_debug"), dict):
        normalized["retrieval_debug"] = dict(item["retrieval_debug"])
    return normalized


def _count_payload_items(payload: dict[str, Any], field_name: str) -> int:
    value = payload.get(field_name)
    if not isinstance(value, list):
        return 0
    return sum(1 for item in value if isinstance(item, dict))


def _merge_citation_payloads(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    citations_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for payload in payloads:
        raw_items: list[dict[str, Any]] = []
        for field_name in ("evidence", "citations"):
            field_value = payload.get(field_name)
            if isinstance(field_value, list):
                raw_items.extend(item for item in field_value if isinstance(item, dict))

        for item in raw_items:
            normalized = _normalize_citation_item(item)
            key = (
                str(normalized.get("source_type", "")),
                str(normalized.get("path", "")),
                str(normalized.get("section", "")),
            )
            existing = citations_by_key.get(key)
            if existing is None:
                citations_by_key[key] = normalized
                continue
            merged = dict(existing)
            for field_name, value in normalized.items():
                if field_name not in merged or merged[field_name] in {"", None, 0, 0.0}:
                    merged[field_name] = value
            if not merged.get("excerpt") and normalized.get("excerpt"):
                merged["excerpt"] = normalized["excerpt"]
            citations_by_key[key] = merged
    return list(citations_by_key.values())


def extract_citations(result: Any) -> list[dict[str, Any]]:
    """Recover citations from tool messages in the deep agent result."""
    return _merge_citation_payloads(_extract_tool_payloads(result))


def extract_answer(result: Any) -> str:
    """Get the last assistant message content from a deep agent result."""
    if isinstance(result, str):
        return result.strip()
    if not isinstance(result, dict):
        return str(result or "").strip()
    messages = list(result.get("messages", []) or [])
    for message in reversed(messages):
        if _message_role(message) in {"assistant", "ai"}:
            text = _extract_message_content(message)
            if text:
                return text
    return ""


def _infer_skills_used(intent: str | None) -> list[str]:
    skills = ["intent-router"]
    if intent == "knowledge_qa":
        skills.append("knowledge-qa")
    elif intent == "issue_analysis":
        skills.append("issue-analysis")
    return skills


def _build_tool_calls(payloads: list[dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
    tools_used: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for payload in payloads:
        tool_name = str(payload.get("_tool_name", "") or "").strip() or "tool"
        if tool_name not in tools_used:
            tools_used.append(tool_name)
        debug = dict(payload.get("debug") or {}) if isinstance(payload.get("debug"), dict) else {}
        call = {"name": tool_name}
        for field_name in (
            "intent",
            "module_name",
            "retrieval_bias",
            "latency_ms",
            "wiki_hits",
            "code_hits",
            "wiki_grade",
            "code_grade",
            "query_count",
            "domain_id",
            "requested_top_k",
            "returned_citation_count",
        ):
            value = payload.get(field_name)
            if value in (None, "") and field_name in debug:
                value = debug.get(field_name)
            if value not in (None, "", []):
                call[field_name] = value
        citation_count = _count_payload_items(payload, "citations")
        evidence_count = _count_payload_items(payload, "evidence")
        if citation_count > 0:
            call["citation_count"] = citation_count
        if evidence_count > 0:
            call["evidence_count"] = evidence_count
        if "retrieval_strategy" not in call:
            strategy = debug.get("retrieval_strategy")
            if strategy not in (None, ""):
                call["retrieval_strategy"] = strategy
        tool_calls.append(call)
    return tools_used, tool_calls


def _build_message_trace(result: Any) -> list[dict[str, Any]]:
    if not isinstance(result, dict):
        return []
    messages = list(result.get("messages", []) or [])
    trace: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        role = _message_role(message) or "unknown"
        item: dict[str, Any] = {"index": index, "role": role}
        name = _extract_message_name(message)
        if name:
            item["name"] = name
        content = _extract_message_content(message)
        if role == "tool":
            payload = _parse_tool_payload(content)
            if payload:
                item["tool_name"] = str(payload.get("_tool_name", "") or name or "tool")
                citation_count = _count_payload_items(payload, "citations")
                evidence_count = _count_payload_items(payload, "evidence")
                if citation_count > 0:
                    item["citation_count"] = citation_count
                if evidence_count > 0:
                    item["evidence_count"] = evidence_count
                debug = payload.get("debug")
                if isinstance(debug, dict):
                    for field_name in (
                        "retrieval_strategy",
                        "requested_top_k",
                        "returned_citation_count",
                        "wiki_hits",
                        "code_hits",
                        "latency_ms",
                    ):
                        value = debug.get(field_name)
                        if value not in (None, "", []):
                            item[field_name] = value
            elif content:
                item["content_preview"] = _preview_text(content)
        elif content:
            item["content_preview"] = _preview_text(content)
        trace.append(item)
    return trace


def _extract_turn_metadata(
    *,
    driver: str,
    answer: str,
    message_count: int,
    citations: list[dict[str, Any]],
    result: Any,
    runtime_debug: dict[str, Any] | None,
) -> tuple[str | None, dict[str, Any], dict[str, Any]]:
    payloads = _extract_tool_payloads(result)
    primary_payload = payloads[-1] if payloads else {}
    tool_debug = dict(primary_payload.get("debug") or {}) if isinstance(primary_payload, dict) else {}
    message_trace = _build_message_trace(result)

    intent = str(primary_payload.get("intent", "") or tool_debug.get("intent", "") or "").strip() or None
    module_name = str(primary_payload.get("module_name", "") or tool_debug.get("module_name", "") or "").strip()
    retrieval_bias = str(
        primary_payload.get("retrieval_bias", "")
        or tool_debug.get("retrieval_strategy", "")
        or ""
    ).strip()
    domain_id = str(primary_payload.get("domain_id", "") or tool_debug.get("domain_id", "") or "").strip()
    tools_used, tool_calls = _build_tool_calls(payloads)
    skills_used = _infer_skills_used(intent)

    analysis: dict[str, Any] = {
        "driver": driver,
        "message_count": message_count,
        "citation_count": len(citations),
        "tool_call_count": len(tool_calls),
        "tools_used": tools_used,
        "skills_used": skills_used,
        "citation_scope": "all_tool_calls_deduped",
    }
    if module_name:
        analysis["module"] = module_name
    if retrieval_bias:
        analysis["retrieval_bias"] = retrieval_bias
    if domain_id:
        analysis["domain_id"] = domain_id
    llm_model = str((runtime_debug or {}).get("llm_model", "") or "").strip()
    if llm_model:
        analysis["llm_model"] = llm_model

    debug: dict[str, Any] = {
        "route": intent or "",
        "skills_used": skills_used,
        "tools_used": tools_used,
        "tool_calls": tool_calls,
        "tool_call_count": len(tool_calls),
        "backend": driver,
        "citation_scope": "all_tool_calls_deduped",
        "message_trace": message_trace,
    }
    if module_name:
        debug["module_name"] = module_name
    if domain_id:
        debug["domain_id"] = domain_id
    for field_name in (
        "latency_ms",
        "wiki_hits",
        "code_hits",
        "wiki_grade",
        "code_grade",
        "query_count",
        "fusion_profile",
    ):
        if field_name in tool_debug:
            debug[field_name] = tool_debug[field_name]
    if runtime_debug:
        debug.update({key: value for key, value in runtime_debug.items() if value is not None})
    if tool_calls:
        last_tool_call = tool_calls[-1]
        if "citation_count" in last_tool_call:
            debug["last_tool_call_citation_count"] = last_tool_call["citation_count"]
        if "requested_top_k" in last_tool_call:
            debug["last_tool_call_requested_top_k"] = last_tool_call["requested_top_k"]
    return intent, analysis, debug


def parse_agent_result(
    *,
    trace_id: str,
    result: Any,
    runtime_debug: dict[str, Any] | None = None,
) -> DeepAgentTurnResult:
    """Normalize the raw Deep Agent output for upper layers."""
    answer = extract_answer(result)
    citations = extract_citations(result)
    message_count = len(result.get("messages", []) or []) if isinstance(result, dict) else 0
    intent, analysis, debug = _extract_turn_metadata(
        driver="deepagents",
        answer=answer,
        message_count=message_count,
        citations=citations,
        result=result,
        runtime_debug=runtime_debug,
    )
    return DeepAgentTurnResult(
        trace_id=trace_id,
        answer=answer,
        citations=citations,
        message_count=message_count,
        status="completed" if answer else "failed",
        driver="deepagents",
        raw_result=result,
        intent=intent,
        analysis=analysis,
        debug=debug,
    )
