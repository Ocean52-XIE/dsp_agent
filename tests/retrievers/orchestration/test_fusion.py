# -*- coding: utf-8 -*-
from __future__ import annotations

import json

from retrievers.orchestration import fusion


def test_fusion_completed_log_includes_trace_id(monkeypatch) -> None:
    records: list[tuple[str, str]] = []

    def _capture(message: str, *args) -> None:
        payload = args[-1] if args else ""
        records.append((message, payload))

    monkeypatch.setattr(fusion.logger, "info", _capture)

    state = {
        "trace_id": "trace-123",
        "retrieval_plan": {
            "strategy": "wiki_first",
            "final_top_k": 5,
            "source_weights": {"wiki": 1.0, "code": 1.0},
            "max_per_source": {"wiki": 5, "code": 5},
            "intent_profile": {"is_wiki_intent": True},
        },
        "wiki_retrieval_grade": "medium",
        "code_retrieval_grade": "low",
        "wiki_hits": [
            {"source_type": "wiki", "path": "a.md", "section": "intro", "score": 0.9},
        ],
        "code_hits": [
            {"source_type": "code", "path": "a.py", "section": "fn", "score": 0.3},
        ],
    }

    plan = fusion._normalize_retrieval_plan(state)
    fusion._build_candidate_items(state, plan)

    event = [item for item in records if item[0] == "%s | %s"][-1]
    payload = json.loads(event[1])
    assert payload["trace_id"] == "trace-123"
    assert payload["candidate_count"] == 2
