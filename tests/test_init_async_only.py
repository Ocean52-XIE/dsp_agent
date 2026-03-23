# -*- coding: utf-8 -*-
"""初始化模块只保留异步入口的回归测试。"""
from __future__ import annotations

import init
from init import mcp_initializer


def test_init_exports_async_only() -> None:
    """`init` 包应只导出异步初始化入口。"""
    assert hasattr(init, "initialize_async")
    assert hasattr(init, "init_database_async")
    assert not hasattr(init, "initialize")
    assert not hasattr(init, "init_database")


def test_mcp_initializer_exports_async_only() -> None:
    """MCP 初始化器应只保留异步入口。"""
    assert hasattr(mcp_initializer, "init_mcp_client_async")
    assert hasattr(mcp_initializer, "initialize_mcp_system_async")
    assert not hasattr(mcp_initializer, "init_mcp_client")
    assert not hasattr(mcp_initializer, "initialize_mcp_system")
