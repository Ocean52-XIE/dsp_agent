# -*- coding: utf-8 -*-
"""Structured conversation memory extracted from completed turns."""
from __future__ import annotations

import re
from typing import Any


_REFERENCE_TOKENS = (
    "这个",
    "这个问题",
    "刚才",
    "上面",
    "继续",
    "它",
    "该问题",
    "那个",
    "前面",
)

_ENTITY_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_./-]{2,}")


def default_conversation_memory() -> dict[str, Any]:
    """Return the normalized empty conversation memory payload."""
    return {
        "current_topic": "",
        "module_name": "",
        "related_modules": [],
        "entities": [],
        "active_issue": "",
        "referenced_paths": [],
        "referenced_symbols": [],
        "confirmed_facts": [],
        "open_questions": [],
        "last_intent": "",
    }


def normalize_conversation_memory(memory: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize a potentially sparse conversation memory payload."""
    normalized = default_conversation_memory()
    if not isinstance(memory, dict):
        return normalized

    for field_name in ("current_topic", "module_name", "active_issue", "last_intent"):
        normalized[field_name] = str(memory.get(field_name, "") or "").strip()

    for field_name in (
        "related_modules",
        "entities",
        "referenced_paths",
        "referenced_symbols",
        "confirmed_facts",
        "open_questions",
    ):
        value = memory.get(field_name)
        if isinstance(value, list):
            normalized[field_name] = _dedupe_strings(value)
    return normalized


def build_conversation_memory(
    *,
    previous_memory: dict[str, Any] | None,
    user_query: str,
    assistant_message: dict[str, Any],
) -> dict[str, Any]:
    """Merge deterministic fields from the latest turn into conversation memory."""
    memory = normalize_conversation_memory(previous_memory)
    query = str(user_query or "").strip()
    debug = dict(assistant_message.get("debug") or {}) if isinstance(assistant_message, dict) else {}
    analysis = dict(assistant_message.get("analysis") or {}) if isinstance(assistant_message, dict) else {}
    citations = list(assistant_message.get("citations") or []) if isinstance(assistant_message, dict) else []
    tool_calls = list(debug.get("tool_calls") or []) if isinstance(debug.get("tool_calls"), list) else []

    last_intent = str(assistant_message.get("intent", "") or "").strip()
    if last_intent:
        memory["last_intent"] = last_intent

    module_name = (
        str(debug.get("module_name", "") or "").strip()
        or str(analysis.get("module", "") or "").strip()
        or memory["module_name"]
    )
    if module_name:
        memory["module_name"] = module_name

    related_modules: list[str] = list(memory["related_modules"])
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        values = call.get("related_modules")
        if isinstance(values, list):
            related_modules.extend(values)
    if module_name and module_name not in related_modules:
        related_modules.insert(0, module_name)
    memory["related_modules"] = _dedupe_strings(related_modules, limit=6)

    referenced_paths = list(memory["referenced_paths"])
    referenced_symbols = list(memory["referenced_symbols"])
    for citation in citations:
        if not isinstance(citation, dict):
            continue
        path = str(citation.get("path", "") or "").strip()
        symbol_name = str(citation.get("symbol_name", "") or "").strip()
        if path:
            referenced_paths.append(path)
        if symbol_name:
            referenced_symbols.append(symbol_name)
    memory["referenced_paths"] = _dedupe_strings(referenced_paths, limit=8)
    memory["referenced_symbols"] = _dedupe_strings(referenced_symbols, limit=8)

    entities = list(memory["entities"])
    entities.extend(_extract_query_entities(query))
    entities.extend(memory["related_modules"])
    entities.extend(memory["referenced_symbols"])
    memory["entities"] = _dedupe_strings(entities, limit=12)

    if query:
        if last_intent == "issue_analysis":
            if not _looks_like_reference_only(query) or not memory["active_issue"]:
                memory["active_issue"] = _truncate(query, max_chars=120)
        elif not memory["current_topic"] or not _looks_like_reference_only(query):
            memory["current_topic"] = _truncate(query, max_chars=120)

    if not memory["current_topic"]:
        memory["current_topic"] = memory["active_issue"] or memory["module_name"]
    if not memory["active_issue"] and memory["last_intent"] == "issue_analysis":
        memory["active_issue"] = memory["current_topic"]

    return memory


def merge_summary_memory_updates(
    memory: dict[str, Any] | None,
    updates: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge LLM summary updates into the structured conversation memory."""
    normalized = normalize_conversation_memory(memory)
    if not isinstance(updates, dict):
        return normalized

    for field_name in ("current_topic", "active_issue"):
        value = str(updates.get(field_name, "") or "").strip()
        if value:
            normalized[field_name] = value

    for field_name in ("confirmed_facts", "open_questions"):
        value = updates.get(field_name)
        if isinstance(value, list):
            existing = list(normalized.get(field_name, []) or [])
            existing.extend(value)
            normalized[field_name] = _dedupe_strings(existing, limit=8)

    return normalized


def render_conversation_memory(memory: dict[str, Any] | None) -> str:
    """Render structured memory for injection into the agent context."""
    normalized = normalize_conversation_memory(memory)
    lines: list[str] = []

    field_labels = (
        ("current_topic", "当前主题"),
        ("module_name", "当前模块"),
        ("active_issue", "当前问题"),
        ("last_intent", "最近意图"),
    )
    for field_name, label in field_labels:
        value = str(normalized.get(field_name, "") or "").strip()
        if value:
            lines.append(f"- {label}: {value}")

    list_labels = (
        ("related_modules", "相关模块"),
        ("entities", "关键实体"),
        ("referenced_symbols", "最近符号"),
        ("referenced_paths", "最近路径"),
        ("confirmed_facts", "已确认事实"),
        ("open_questions", "待确认问题"),
    )
    for field_name, label in list_labels:
        values = normalized.get(field_name) or []
        if isinstance(values, list) and values:
            lines.append(f"- {label}: {', '.join(str(item) for item in values)}")

    return "\n".join(lines).strip()


def _looks_like_reference_only(query: str) -> bool:
    text = str(query or "").strip()
    if not text:
        return False
    return any(token in text for token in _REFERENCE_TOKENS)


def _extract_query_entities(query: str) -> list[str]:
    text = str(query or "").strip()
    if not text:
        return []
    return _dedupe_strings(_ENTITY_PATTERN.findall(text), limit=8)


def _dedupe_strings(values: list[Any], *, limit: int | None = None) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
        if limit is not None and len(result) >= limit:
            break
    return result


def _truncate(value: str, *, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}..."
