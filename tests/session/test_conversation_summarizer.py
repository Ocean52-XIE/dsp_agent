# -*- coding: utf-8 -*-
"""Tests for the LLM-backed conversation summarizer."""
from __future__ import annotations

from pathlib import Path

import pytest

from session.conversation_summarizer import ConversationSummarizer, ConversationSummaryConfig


class _StubResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _StubModel:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[list[object]] = []

    async def ainvoke(self, messages: list[object]) -> _StubResponse:
        self.calls.append(messages)
        return _StubResponse(self.content)


@pytest.fixture
def summary_config() -> ConversationSummaryConfig:
    return ConversationSummaryConfig(
        enabled=True,
        model="gpt-test",
        api_key="",
        base_url="",
        temperature=0.1,
        max_tokens=256,
        timeout_seconds=30,
        trigger_message_count=4,
        max_context_messages=6,
        max_summary_chars=120,
    )


@pytest.mark.asyncio
async def test_refresh_summary_returns_previous_when_below_threshold(
    monkeypatch: pytest.MonkeyPatch,
    summary_config: ConversationSummaryConfig,
) -> None:
    monkeypatch.setattr(
        "session.conversation_summarizer.get_domain_profile",
        lambda project_root: type("Profile", (), {"display_name": "广告引擎", "profile_id": "ad_engine"})(),
    )
    model = _StubModel("new summary")
    summarizer = ConversationSummarizer(
        project_root=Path("."),
        config=summary_config,
        model=model,
    )

    result = await summarizer.refresh_summary(
        messages=[
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
        ],
        previous_summary="old summary",
    )

    assert result.summary == "old summary"
    assert result.memory_updates == {}
    assert model.calls == []


@pytest.mark.asyncio
async def test_refresh_summary_uses_llm_and_returns_new_summary(
    monkeypatch: pytest.MonkeyPatch,
    summary_config: ConversationSummaryConfig,
) -> None:
    monkeypatch.setattr(
        "session.conversation_summarizer.get_domain_profile",
        lambda project_root: type("Profile", (), {"display_name": "广告引擎", "profile_id": "ad_engine"})(),
    )
    model = _StubModel(
        (
            '{"summary":"用户正在排查出价胜率下降，当前重点关注 bid-optimizer 模块。",'
            '"current_topic":"排查出价胜率下降",'
            '"active_issue":"出价胜率下降",'
            '"confirmed_facts":["当前重点关注 bid-optimizer 模块"],'
            '"open_questions":["待确认波动发生在召回前还是排序后"]}'
        )
    )
    summarizer = ConversationSummarizer(
        project_root=Path("."),
        config=summary_config,
        model=model,
    )

    result = await summarizer.refresh_summary(
        messages=[
            {"role": "system", "content": "ignored"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
        ],
        previous_summary="old summary",
        conversation_memory={
            "module_name": "bid-optimizer",
            "related_modules": ["bid-optimizer", "rerank-engine"],
        },
    )

    assert result.summary == "用户正在排查出价胜率下降，当前重点关注 bid-optimizer 模块。"
    assert result.memory_updates == {
        "current_topic": "排查出价胜率下降",
        "active_issue": "出价胜率下降",
        "confirmed_facts": ["当前重点关注 bid-optimizer 模块"],
        "open_questions": ["待确认波动发生在召回前还是排序后"],
    }
    assert len(model.calls) == 1
