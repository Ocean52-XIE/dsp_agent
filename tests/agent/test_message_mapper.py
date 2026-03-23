# -*- coding: utf-8 -*-
"""Tests for parsing deep-agent results."""
from __future__ import annotations

import json
from types import SimpleNamespace

from agent.result_parser import extract_answer, extract_citations, parse_agent_result


def test_extract_answer_prefers_last_assistant_message() -> None:
    result = {
        "messages": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "draft"},
            SimpleNamespace(type="ai", content=[{"text": "final answer"}]),
        ]
    }

    assert extract_answer(result) == "final answer"


def test_extract_citations_merges_evidence_and_citations() -> None:
    payload = json.dumps(
        {
            "domain_id": "ad_engine",
            "intent": "knowledge_qa",
            "module_name": "rerank-engine",
            "citations": [
                {"source": "wiki", "path": "a.md", "section": "intro", "score": 0.9},
                {"source": "wiki", "path": "a.md", "section": "intro", "score": 0.9},
                {"source": "code", "path": "b.py", "section": "func", "score": 0.8},
            ],
            "evidence": [
                {
                    "source_type": "code",
                    "path": "b.py",
                    "section": "func",
                    "title": "compute_rank_score",
                    "score": 0.8,
                    "excerpt": "def compute_rank_score(...):",
                    "symbol_name": "compute_rank_score",
                    "start_line": 10,
                    "end_line": 12,
                }
            ],
            "debug": {
                "latency_ms": 32,
                "retrieval_strategy": "code_first",
                "wiki_hits": 1,
                "code_hits": 5,
            },
        }
    )
    result = {
        "messages": [
            {"role": "tool", "name": "domain_retrieve", "content": payload},
            {"role": "tool", "content": "not-json"},
        ]
    }

    citations = extract_citations(result)

    assert sorted(citations, key=lambda item: (item["source_type"], item["path"])) == [
        {
            "source": "code",
            "source_type": "code",
            "path": "b.py",
            "title": "compute_rank_score",
            "section": "func",
            "score": 0.8,
            "excerpt": "def compute_rank_score(...):",
            "symbol_name": "compute_rank_score",
            "start_line": 10,
            "end_line": 12,
        },
        {
            "source": "wiki",
            "source_type": "wiki",
            "path": "a.md",
            "title": "a.md",
            "section": "intro",
            "score": 0.9,
            "excerpt": "",
            "symbol_name": "",
            "start_line": None,
            "end_line": None,
        },
    ]


def test_parse_agent_result_handles_missing_answer() -> None:
    result = parse_agent_result(trace_id="trace-1", result={"messages": []})

    assert result.trace_id == "trace-1"
    assert result.status == "failed"
    assert result.answer == ""
    assert result.driver == "deepagents"


def test_parse_agent_result_extracts_debug_metadata() -> None:
    payload = json.dumps(
        {
            "domain_id": "ad_engine",
            "intent": "knowledge_qa",
            "module_name": "rerank-engine",
            "retrieval_bias": "code_first",
            "citations": [
                {
                    "source": "code",
                    "source_type": "code",
                    "path": "domain/ad_engine/codes/rerank/rerank_engine.py",
                    "title": "compute_rank_score",
                    "section": "function:compute_rank_score",
                    "score": 0.88,
                    "excerpt": "def compute_rank_score(...):",
                }
            ],
            "debug": {
                "latency_ms": 41,
                "retrieval_strategy": "code_first",
                "wiki_hits": 1,
                "code_hits": 5,
            },
        }
    )
    parsed = parse_agent_result(
        trace_id="trace-2",
        result={
            "messages": [
                {"role": "tool", "name": "domain_retrieve", "content": payload},
                {"role": "assistant", "content": "final answer"},
            ]
        },
        runtime_debug={"latency_ms": 55, "checkpointer_backend": "postgres"},
    )

    assert parsed.intent == "knowledge_qa"
    assert parsed.analysis["module"] == "rerank-engine"
    assert parsed.analysis["retrieval_bias"] == "code_first"
    assert parsed.analysis["skills_used"] == ["intent-router", "knowledge-qa"]
    assert parsed.debug["route"] == "knowledge_qa"
    assert parsed.debug["tools_used"] == ["domain_retrieve"]
    assert parsed.debug["tool_call_count"] == 1
    assert parsed.debug["checkpointer_backend"] == "postgres"
    assert parsed.debug["latency_ms"] == 55
