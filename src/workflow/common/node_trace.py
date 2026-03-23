# -*- coding: utf-8 -*-
"""Node trace helpers."""
from __future__ import annotations

from typing import Any


def append_node_trace(state: dict[str, Any], node: str, summary: str) -> list[dict[str, str]]:
    """Return only the new trace entry (not the full list).

    由于 node_trace 字段使用了 merge_lists reducer，节点只需返回新增的条目，
    LangGraph 会自动合并。返回完整列表会导致重复累积。

    Args:
        state: 工作流状态（用于兼容旧接口，实际不再使用）
        node: 节点名称
        summary: 节点摘要

    Returns:
        只包含新增条目的列表
    """
    return [{"node": node, "summary": summary}]
