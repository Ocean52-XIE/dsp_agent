# -*- coding: utf-8 -*-
"""Tests for structured conversation memory extraction."""
from __future__ import annotations

from session.conversation_memory import (
    build_conversation_memory,
    default_conversation_memory,
    merge_summary_memory_updates,
    render_conversation_memory,
)


def test_build_conversation_memory_extracts_module_entities_and_references() -> None:
    memory = build_conversation_memory(
        previous_memory=None,
        user_query="继续排查 compute_bid_for_request 这个问题",
        assistant_message={
            "intent": "issue_analysis",
            "analysis": {"module": "bid-optimizer"},
            "debug": {
                "module_name": "bid-optimizer",
                "tool_calls": [
                    {
                        "name": "domain_retrieve",
                        "related_modules": ["bid-optimizer", "rerank-engine"],
                    }
                ],
            },
            "citations": [
                {
                    "path": "domain/ad_engine/codes/bid/bid_optimizer.py",
                    "symbol_name": "compute_bid_for_request",
                },
                {
                    "path": "domain/ad_engine/wiki/03-出价.md",
                    "symbol_name": "",
                },
            ],
        },
    )

    assert memory == {
        "current_topic": "继续排查 compute_bid_for_request 这个问题",
        "module_name": "bid-optimizer",
        "related_modules": ["bid-optimizer", "rerank-engine"],
        "entities": ["compute_bid_for_request", "bid-optimizer", "rerank-engine"],
        "active_issue": "继续排查 compute_bid_for_request 这个问题",
        "referenced_paths": [
            "domain/ad_engine/codes/bid/bid_optimizer.py",
            "domain/ad_engine/wiki/03-出价.md",
        ],
        "referenced_symbols": ["compute_bid_for_request"],
        "confirmed_facts": [],
        "open_questions": [],
        "last_intent": "issue_analysis",
    }


def test_build_conversation_memory_keeps_previous_issue_for_reference_only_query() -> None:
    memory = build_conversation_memory(
        previous_memory={
            **default_conversation_memory(),
            "current_topic": "排查出价胜率下降",
            "active_issue": "出价胜率下降",
            "module_name": "bid-optimizer",
            "last_intent": "issue_analysis",
        },
        user_query="继续看这个问题",
        assistant_message={
            "intent": "issue_analysis",
            "analysis": {},
            "debug": {},
            "citations": [],
        },
    )

    assert memory["active_issue"] == "出价胜率下降"
    assert memory["current_topic"] == "排查出价胜率下降"


def test_render_conversation_memory_formats_non_empty_fields() -> None:
    rendered = render_conversation_memory(
        {
            "current_topic": "排查出价胜率下降",
            "module_name": "bid-optimizer",
            "related_modules": ["bid-optimizer", "rerank-engine"],
            "entities": ["compute_bid_for_request"],
            "active_issue": "出价胜率下降",
            "referenced_paths": ["domain/ad_engine/codes/bid/bid_optimizer.py"],
            "referenced_symbols": ["compute_bid_for_request"],
            "confirmed_facts": ["当前重点关注 bid-optimizer 模块"],
            "open_questions": ["待确认波动发生在召回前还是排序后"],
            "last_intent": "issue_analysis",
        }
    )

    assert rendered == (
        "- 当前主题: 排查出价胜率下降\n"
        "- 当前模块: bid-optimizer\n"
        "- 当前问题: 出价胜率下降\n"
        "- 最近意图: issue_analysis\n"
        "- 相关模块: bid-optimizer, rerank-engine\n"
        "- 关键实体: compute_bid_for_request\n"
        "- 最近符号: compute_bid_for_request\n"
        "- 最近路径: domain/ad_engine/codes/bid/bid_optimizer.py\n"
        "- 已确认事实: 当前重点关注 bid-optimizer 模块\n"
        "- 待确认问题: 待确认波动发生在召回前还是排序后"
    )


def test_merge_summary_memory_updates_merges_lists_and_topic_fields() -> None:
    merged = merge_summary_memory_updates(
        {
            **default_conversation_memory(),
            "current_topic": "排查出价胜率下降",
            "confirmed_facts": ["当前重点关注 bid-optimizer 模块"],
        },
        {
            "current_topic": "继续排查出价胜率下降",
            "active_issue": "出价胜率下降",
            "confirmed_facts": ["当前重点关注 bid-optimizer 模块", "问题发生在 tcpa 场景"],
            "open_questions": ["待确认波动发生在召回前还是排序后"],
        },
    )

    assert merged["current_topic"] == "继续排查出价胜率下降"
    assert merged["active_issue"] == "出价胜率下降"
    assert merged["confirmed_facts"] == ["当前重点关注 bid-optimizer 模块", "问题发生在 tcpa 场景"]
    assert merged["open_questions"] == ["待确认波动发生在召回前还是排序后"]
