# -*- coding: utf-8 -*-
"""Tests for API message mapping."""
from __future__ import annotations

from agent.result_parser import DeepAgentTurnResult
from api.message_mapper import to_assistant_message


def test_to_assistant_message_maps_parsed_result() -> None:
    result = DeepAgentTurnResult(
        trace_id="trace-1",
        answer="final answer",
        citations=[
            {
                "source": "wiki",
                "source_type": "wiki",
                "path": "a.md",
                "section": "intro",
                "score": 0.9,
                "excerpt": "summary",
            }
        ],
        message_count=3,
        status="completed",
        driver="deepagents",
        raw_result={"messages": []},
        intent="knowledge_qa",
        analysis={
            "module": "rerank-engine",
            "retrieval_bias": "code_first",
            "skills_used": ["knowledge-qa"],
        },
        debug={
            "latency_ms": 42,
            "checkpointer_backend": "postgres",
            "tools_used": ["domain_retrieve"],
        },
    )

    message = to_assistant_message(result)

    assert message["trace_id"] == "trace-1"
    assert message["role"] == "assistant"
    assert message["intent"] == "knowledge_qa"
    assert message["status"] == "completed"
    assert message["content"] == "final answer"
    assert message["analysis"]["driver"] == "deepagents"
    assert message["analysis"]["citation_count"] == 1
    assert message["analysis"]["module"] == "rerank-engine"
    assert message["debug"]["latency_ms"] == 42
    assert message["debug"]["tools_used"] == ["domain_retrieve"]
