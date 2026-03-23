# -*- coding: utf-8 -*-
"""Map parsed deep-agent results into API assistant messages."""
from __future__ import annotations

from typing import Any

from agent.result_parser import DeepAgentTurnResult


def to_assistant_message(result: DeepAgentTurnResult) -> dict[str, Any]:
    """Convert a normalized deep-agent result into the API message shape."""
    analysis = dict(result.analysis or {})
    analysis.setdefault("driver", result.driver)
    analysis.setdefault("message_count", result.message_count)
    analysis.setdefault("citation_count", len(result.citations))

    debug = dict(result.debug or {})
    debug.setdefault("backend", result.driver)
    debug.setdefault("message_count", result.message_count)

    return {
        "trace_id": result.trace_id,
        "role": "assistant",
        "kind": "deep_agent_response",
        "intent": result.intent,
        "status": result.status,
        "content": result.answer or "Deep agent did not produce a final answer.",
        "citations": result.citations,
        "analysis": analysis,
        "actions": [],
        "debug": debug,
    }
