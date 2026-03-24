# -*- coding: utf-8 -*-
"""Tests for LLM structured logging callback."""
from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from agent.llm_logging import LLMCallLoggingCallback


class _StubRuntimeLogger:
    def __init__(self) -> None:
        self.records: list[tuple[str, str, dict]] = []

    def info(self, event: str, **payload) -> None:
        self.records.append(("info", event, payload))

    def warning(self, event: str, **payload) -> None:
        self.records.append(("warning", event, payload))


@pytest.mark.asyncio
async def test_llm_logging_callback_records_start_and_end(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_logger = _StubRuntimeLogger()
    monkeypatch.setattr("agent.llm_logging.get_file_logger", lambda project_root: runtime_logger)

    callback = LLMCallLoggingCallback(project_root=Path("."))
    run_id = uuid4()
    parent_run_id = uuid4()

    await callback.on_chat_model_start(
        {"name": "ChatOpenAI", "id": ["langchain", "chat_models", "openai"]},
        [[
            HumanMessage(content="请解释 eCPM"),
            ToolMessage(content='{"citations":[1,2,3]}', tool_call_id="tool-1"),
        ]],
        run_id=run_id,
        parent_run_id=parent_run_id,
        metadata={"session_id": "sess-1", "trace_id": "trace-1", "llm_model": "deepseek-chat"},
        tags=["dsp_agent", "deep_agent"],
        invocation_params={"model": "deepseek-chat", "temperature": 0.1, "max_tokens": 4096},
    )

    result = LLMResult(
        generations=[[
            ChatGeneration(
                message=AIMessage(
                    content="eCPM 常见公式是 pCTR × Bid × 1000",
                    usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
                    response_metadata={"finish_reason": "stop", "id": "resp_123", "model_name": "deepseek-chat"},
                ),
                generation_info={"finish_reason": "stop"},
            )
        ]],
        llm_output={"model_name": "deepseek-chat", "token_usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}, "id": "resp_123"},
    )

    await callback.on_llm_end(
        result,
        run_id=run_id,
        parent_run_id=parent_run_id,
        tags=["dsp_agent", "deep_agent"],
    )

    assert runtime_logger.records[0][0] == "info"
    assert runtime_logger.records[0][1] == "llm.call.started"
    assert runtime_logger.records[0][2]["trace_id"] == "trace-1"
    assert runtime_logger.records[0][2]["call_index"] == 1
    assert runtime_logger.records[0][2]["input_summary"]["message_count"] == 2
    assert runtime_logger.records[0][2]["input_summary"]["role_counts"] == {"human": 1, "tool": 1}
    assert runtime_logger.records[0][2]["input_summary"]["logged_message_count"] == 2
    assert runtime_logger.records[0][2]["input_summary"]["messages"][0]["content_chars"] > 0
    assert "content_preview" not in runtime_logger.records[0][2]["input_summary"]["messages"][1]

    assert runtime_logger.records[1][0] == "info"
    assert runtime_logger.records[1][1] == "llm.call.completed"
    assert runtime_logger.records[1][2]["trace_id"] == "trace-1"
    assert runtime_logger.records[1][2]["call_index"] == 1
    assert runtime_logger.records[1][2]["finish_reason"] == "stop"
    assert runtime_logger.records[1][2]["response_id"] == "resp_123"
    assert runtime_logger.records[1][2]["usage"] == {
        "input_tokens": 100,
        "output_tokens": 20,
        "total_tokens": 120,
    }
    assert "eCPM" in runtime_logger.records[1][2]["output_preview"]


@pytest.mark.asyncio
async def test_llm_logging_callback_records_error(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_logger = _StubRuntimeLogger()
    monkeypatch.setattr("agent.llm_logging.get_file_logger", lambda project_root: runtime_logger)

    callback = LLMCallLoggingCallback(project_root=Path("."))
    run_id = uuid4()

    await callback.on_chat_model_start(
        {"name": "ChatOpenAI"},
        [[HumanMessage(content="hello")]],
        run_id=run_id,
        metadata={"session_id": "sess-err", "trace_id": "trace-err", "llm_model": "deepseek-chat"},
    )

    await callback.on_llm_error(
        TimeoutError("upstream timeout"),
        run_id=run_id,
    )

    assert runtime_logger.records[-1][0] == "warning"
    assert runtime_logger.records[-1][1] == "llm.call.failed"
    assert runtime_logger.records[-1][2]["trace_id"] == "trace-err"
    assert runtime_logger.records[-1][2]["error_type"] == "TimeoutError"
