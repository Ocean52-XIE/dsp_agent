# -*- coding: utf-8 -*-
"""Helpers for loading MCP tools into the deep agent."""
from __future__ import annotations

from typing import Any

from init.mcp_initializer import load_mcp_tools


def get_mcp_tools() -> list[Any]:
    """Return initialized MCP tools for the deep agent."""
    return list(load_mcp_tools())
