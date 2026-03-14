# -*- coding: utf-8 -*-
"""Prompt helper utilities shared by LLM-enabled nodes."""
from __future__ import annotations

import os
from typing import Any

from workflow.domain_profile import DomainProfile
from workflow.utils import normalize_source_type


def resolve_system_prompt(
    *,
    env_key: str,
    default_prompt: str,
    domain_profile: DomainProfile | None,
) -> str:
    env_prompt = os.getenv(env_key, "").strip()
    if env_prompt:
        return env_prompt
    if domain_profile is not None:
        profile_prompt = domain_profile.system_prompt()
        if profile_prompt:
            return profile_prompt
    return default_prompt


def build_evidence_block(evidence_hits: list[dict[str, Any]], *, max_items: int = 6) -> str:
    lines: list[str] = []
    for index, item in enumerate(evidence_hits[:max_items], start=1):
        source_type = normalize_source_type(item.get("source_type"))
        lines.append(f"[evidence_{index}]")
        lines.append(f"- source_type: {source_type}")
        lines.append(f"- title: {item.get('title', '')}")
        lines.append(f"- path: {item.get('path', '')}")
        lines.append(f"- section: {item.get('section', '')}")
        lines.append(f"- score: {item.get('score', '')}")

        excerpt = str(item.get("excerpt", "")).strip()
        if excerpt:
            lines.append(f"- excerpt: {excerpt}")

        if source_type == "code":
            lines.append(f"- symbol_name: {item.get('symbol_name', '')}")
            lines.append(f"- signature: {item.get('signature', '')}")
            start_line = item.get("start_line", "")
            end_line = item.get("end_line", "")
            if start_line != "" or end_line != "":
                lines.append(f"- line_range: {start_line}-{end_line}")
            highlight_lines = item.get("highlight_lines", [])
            if isinstance(highlight_lines, list) and highlight_lines:
                lines.append(f"- highlight_lines: {highlight_lines[:6]}")
    return "\n".join(lines) if lines else "none"


def looks_like_reasoning_dump(answer: str) -> bool:
    lowered = str(answer or "").lower()
    markers = (
        "chain of thought",
        "reasoning",
        "思考过程",
        "推理过程",
        "analysis:",
    )
    return any(marker in lowered for marker in markers)
