# -*- coding: utf-8 -*-
from __future__ import annotations

from types import SimpleNamespace

import pytest

from api import main


class _StubLogger:
    def __init__(self) -> None:
        self.records: list[tuple[str, str, dict]] = []

    def info(self, event: str, **payload) -> None:
        self.records.append(("info", event, payload))

    def warning(self, event: str, **payload) -> None:
        self.records.append(("warning", event, payload))

    def exception(self, event: str, **payload) -> None:
        self.records.append(("exception", event, payload))


class _StubService:
    async def run_user_message_async(self, **kwargs):
        return SimpleNamespace()


@pytest.mark.asyncio
async def test_create_message_completed_log_includes_latency(monkeypatch) -> None:
    logger = _StubLogger()
    session = {
        "id": "sess-1",
        "title": "session",
        "created_at": "2026-03-24T14:00:00",
        "updated_at": "2026-03-24T14:00:00",
        "status": "idle",
        "messages": [],
        "conversation_summary": "",
        "conversation_summary_updated_at": None,
        "conversation_memory": {},
        "conversation_memory_updated_at": None,
    }

    async def _ensure_session(session_id: str) -> dict:
        return session

    async def _noop_async(*args, **kwargs) -> None:
        return None

    monkeypatch.setattr(main, "APP_LOGGER", logger)
    monkeypatch.setattr(main, "AGENT_SERVICE", _StubService(), raising=False)
    monkeypatch.setattr(main, "ensure_session", _ensure_session)
    monkeypatch.setattr(
        main,
        "to_assistant_message",
        lambda result: {
            "role": "assistant",
            "kind": "deep_agent_response",
            "status": "completed",
            "content": "ok",
            "trace_id": "trace-fixed",
            "citations": [{"path": "a.md"}],
            "analysis": None,
            "actions": [],
            "debug": {},
        },
    )
    monkeypatch.setattr(main, "refresh_conversation_memory", lambda **kwargs: None)
    monkeypatch.setattr(main, "attach_memory_debug_snapshot", lambda **kwargs: None)
    monkeypatch.setattr(main, "refresh_conversation_summary", _noop_async)
    monkeypatch.setattr(main, "persist_session_record", _noop_async)
    monkeypatch.setattr(main, "persist_observability_turn", _noop_async)

    result = await main.create_message(
        main.MessageCreateRequest(session_id="sess-1", content="广告引擎里 eCPM 的常见计算公式是什么"),
    )

    completed = [record for record in logger.records if record[1] == "api.message.create.completed"]
    assert result["assistant_message_id"].startswith("msg_")
    assert len(completed) == 1
    assert completed[0][2]["trace_id"].startswith("trace_")
    assert isinstance(completed[0][2]["latency_ms"], int)
    assert completed[0][2]["latency_ms"] >= 0
