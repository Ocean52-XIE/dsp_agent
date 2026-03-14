# -*- coding: utf-8 -*-
"""Evidence collection helpers shared by analysis nodes."""
from __future__ import annotations

from typing import Any

from workflow.utils import normalize_source_type


def collect_evidence_hits(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Build normalized evidence rows from state citations/hits."""
    citations = list(state.get("citations", []) or [])
    if citations:
        rows: list[dict[str, Any]] = []
        for item in citations:
            row = dict(item)
            row["source_type"] = normalize_source_type(row.get("source_type", row.get("source", "")))
            rows.append(row)
        return rows

    fallback_hits: list[dict[str, Any]] = []
    for key, default_type in (("wiki_hits", "wiki"), ("code_hits", "code"), ("case_hits", "case")):
        for item in list(state.get(key, []) or []):
            row = dict(item)
            row["source_type"] = normalize_source_type(row.get("source_type", default_type))
            fallback_hits.append(row)
    return fallback_hits
