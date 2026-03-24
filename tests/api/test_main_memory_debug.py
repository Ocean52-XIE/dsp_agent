# -*- coding: utf-8 -*-
"""Tests for conversation memory debug snapshots in the API layer."""
from __future__ import annotations

from api.main import attach_memory_debug_snapshot


def test_attach_memory_debug_snapshot_copies_confirmed_facts_and_open_questions() -> None:
    assistant_message = {
        "debug": {
            "route": "issue_analysis",
        }
    }
    session = {
        "conversation_memory": {
            "current_topic": "排查出价胜率下降",
            "active_issue": "出价胜率下降",
            "confirmed_facts": ["当前重点关注 bid-optimizer 模块"],
            "open_questions": ["待确认波动发生在召回前还是排序后"],
        }
    }

    attach_memory_debug_snapshot(
        assistant_message=assistant_message,
        session=session,
    )

    assert assistant_message["debug"] == {
        "route": "issue_analysis",
        "current_topic": "排查出价胜率下降",
        "active_issue": "出价胜率下降",
        "confirmed_facts": ["当前重点关注 bid-optimizer 模块"],
        "open_questions": ["待确认波动发生在召回前还是排序后"],
    }
