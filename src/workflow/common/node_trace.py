# -*- coding: utf-8 -*-
"""Node trace helpers."""
from __future__ import annotations

from typing import Any


def append_node_trace(state: dict[str, Any], node: str, summary: str) -> list[dict[str, str]]:
    """Append a trace row and return a new trace list."""
    return [*state.get("node_trace", []), {"node": node, "summary": summary}]
