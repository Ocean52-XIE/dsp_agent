# -*- coding: utf-8 -*-
"""Tests for the deep-agent service."""
from __future__ import annotations

from pathlib import Path

import pytest

from agent.service import DeepAgentService


class _StubAgent:
    def __init__(self) -> None:
        self.calls: list[tuple[dict, dict]] = []
        self._dsp_runtime_config = {"model": "gpt-test"}

    async def ainvoke(self, payload: dict, config: dict) -> dict:
        self.calls.append((payload, config))
        return {"messages": [{"role": "assistant", "content": "done"}]}


class _FailingAgent:
    def __init__(self, exc: Exception) -> None:
        self.calls: list[tuple[dict, dict]] = []
        self._exc = exc

    async def ainvoke(self, payload: dict, config: dict) -> dict:
        self.calls.append((payload, config))
        raise self._exc


class AuthenticationError(Exception):
    def __init__(self, message: str = "Authentication Fails") -> None:
        super().__init__(message)
        self.status_code = 401


async def _create_service(monkeypatch: pytest.MonkeyPatch, agent: object) -> DeepAgentService:
    async def _init_database_async() -> tuple[None, str]:
        return None, "checkpoint"

    monkeypatch.setattr("agent.service.init_database_async", _init_database_async)
    monkeypatch.setattr(
        "agent.service.get_database_status",
        lambda checkpointer: {"backend": "memory", "status": "active"},
    )
    monkeypatch.setattr(
        "agent.service.create_agent",
        lambda project_root, checkpointer: agent,
    )
    return await DeepAgentService.create_async(project_root=Path("."))


@pytest.mark.asyncio
async def test_run_user_message_uses_session_thread_and_current_message_only(monkeypatch) -> None:
    stub_agent = _StubAgent()
    service = await _create_service(monkeypatch, stub_agent)
    result = await service.run_user_message_async(
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
            {
                "configurable": {"thread_id": "session-1"},
                "metadata": {
                    "session_id": "session-1",
                    "trace_id": "trace-1",
                    "llm_model": "gpt-test",
                    "agent_backend": "deepagents",
                },
                "tags": ["dsp_agent", "deep_agent"],
            },
        )
    ]
    assert result.trace_id == "trace-1"
    assert result.status == "completed"
    assert result.answer == "done"
    assert result.driver == "deepagents"
    assert result.analysis["llm_model"] == "gpt-test"
    assert result.debug["llm_model"] == "gpt-test"
    assert service.checkpointer_status() == {"backend": "memory", "status": "active"}
    assert service.runtime_log_status() == {"backend": "deepagents", "llm_model": "gpt-test"}


@pytest.mark.asyncio
async def test_run_user_message_degrades_on_authentication_error(monkeypatch) -> None:
    failing_agent = _FailingAgent(
        AuthenticationError("Authentication Fails, Your api key is invalid"),
    )
    service = await _create_service(monkeypatch, failing_agent)

    result = await service.run_user_message_async(
        session_id="session-auth",
        trace_id="trace-auth",
        user_query="help me",
        history=[],
    )

    assert result.status == "error"
    assert result.intent == "error"
    assert "鉴权失败" in result.answer
    assert result.debug["fallback_reason"] == "llm_authentication_failed"
    assert result.debug["upstream_status_code"] == 401
    assert failing_agent.calls == [
        (
            {"messages": [{"role": "user", "content": "help me"}]},
            {
                "configurable": {"thread_id": "session-auth"},
                "metadata": {
                    "session_id": "session-auth",
                    "trace_id": "trace-auth",
                    "llm_model": "",
                    "agent_backend": "deepagents",
                },
                "tags": ["dsp_agent", "deep_agent"],
            },
        )
    ]


@pytest.mark.asyncio
async def test_run_user_message_degrades_on_timeout_error(monkeypatch) -> None:
    failing_agent = _FailingAgent(TimeoutError("upstream llm timeout"))
    service = await _create_service(monkeypatch, failing_agent)

    result = await service.run_user_message_async(
        session_id="session-timeout",
        trace_id="trace-timeout",
        user_query="help me later",
        history=[],
    )

    assert result.status == "error"
    assert result.intent == "error"
    assert "暂时不可用" in result.answer
    assert result.debug["fallback_reason"] == "llm_request_failed"
    assert result.debug["error_type"] == "TimeoutError"


@pytest.mark.asyncio
async def test_run_user_message_reraises_non_llm_error(monkeypatch) -> None:
    service = await _create_service(monkeypatch, _FailingAgent(ValueError("bad state")))

    with pytest.raises(ValueError, match="bad state"):
        await service.run_user_message_async(
            session_id="session-fail",
            trace_id="trace-fail",
            user_query="boom",
            history=[],
        )


@pytest.mark.asyncio
async def test_aclose_releases_database_resource(monkeypatch) -> None:
    stub_agent = _StubAgent()

    class _Resource:
        def __init__(self) -> None:
            self.closed = False

        async def aclose(self) -> None:
            self.closed = True

    resource = _Resource()

    async def _init_database_async() -> tuple[_Resource, str]:
        return resource, "checkpoint"

    monkeypatch.setattr("agent.service.init_database_async", _init_database_async)
    monkeypatch.setattr(
        "agent.service.get_database_status",
        lambda checkpointer: {"backend": "postgres", "status": "active"},
    )
    monkeypatch.setattr(
        "agent.service.create_agent",
        lambda project_root, checkpointer: stub_agent,
    )

    service = await DeepAgentService.create_async(project_root=Path("."))
    await service.aclose()

    assert resource.closed is True
