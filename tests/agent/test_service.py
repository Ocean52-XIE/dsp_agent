# -*- coding: utf-8 -*-
"""Tests for the deep-agent service."""
from __future__ import annotations

from pathlib import Path

from agent.service import DeepAgentService


class _StubAgent:
    def __init__(self) -> None:
        self.calls: list[tuple[dict, dict]] = []

    def invoke(self, payload: dict, config: dict) -> dict:
        self.calls.append((payload, config))
        return {"messages": [{"role": "assistant", "content": "done"}]}


def test_run_user_message_uses_session_thread_and_current_message_only(monkeypatch) -> None:
    stub_agent = _StubAgent()
    monkeypatch.setattr("agent.service.init_database", lambda: (None, "checkpoint"))
    monkeypatch.setattr(
        "agent.service.get_database_status",
        lambda checkpointer: {"backend": "memory", "status": "active"},
    )
    monkeypatch.setattr(
        "agent.service.create_agent",
        lambda project_root, checkpointer: stub_agent,
    )

    service = DeepAgentService(project_root=Path("."))
    result = service.run_user_message(
        session_id="session-1",
        trace_id="trace-1",
        user_query="latest question",
        history=[
            {"role": "system", "content": "ignore"},
            {"role": "user", "content": "history question"},
            {"role": "assistant", "content": "history answer"},
            {"role": "user", "content": "latest question"},
        ],
    )

    assert stub_agent.calls == [
        (
            {"messages": [{"role": "user", "content": "latest question"}]},
            {"configurable": {"thread_id": "session-1"}},
        )
    ]
    assert result.trace_id == "trace-1"
    assert result.status == "completed"
    assert result.answer == "done"
    assert result.driver == "deepagents"
    assert service.checkpointer_status() == {"backend": "memory", "status": "active"}


def test_close_releases_database_resource(monkeypatch) -> None:
    stub_agent = _StubAgent()

    class _Resource:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    resource = _Resource()
    monkeypatch.setattr("agent.service.init_database", lambda: (resource, "checkpoint"))
    monkeypatch.setattr(
        "agent.service.get_database_status",
        lambda checkpointer: {"backend": "postgres", "status": "active"},
    )
    monkeypatch.setattr(
        "agent.service.create_agent",
        lambda project_root, checkpointer: stub_agent,
    )

    service = DeepAgentService(project_root=Path("."))
    service.close()

    assert resource.closed is True
