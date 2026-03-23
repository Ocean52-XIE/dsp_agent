# -*- coding: utf-8 -*-
"""MCP 系统初始化器。"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def init_mcp_client_async(domain_profile: Any) -> None:
    """异步初始化 MCP 客户端并设置全局单例。"""
    import os

    mcp_enabled = os.getenv("AGENT_MCP_ENABLED", "false").lower().strip()
    if mcp_enabled not in {"true", "1", "yes"}:
        logger.info("[MCPInit] MCP disabled by AGENT_MCP_ENABLED")
        return

    try:
        from agent.mcp import MCPClient, set_mcp_client
        from agent.mcp.config_loader import MCPServerConfigLoader

        loader = MCPServerConfigLoader()
        configs = loader.load_from_domain(domain_profile.domain_dir)
        if not configs:
            logger.info("[MCPInit] no MCP server configs found")
            return

        enabled_configs = {name: config for name, config in configs.items() if config.enabled}
        if not enabled_configs:
            logger.info("[MCPInit] no enabled MCP servers found")
            return

        client = MCPClient(enabled_configs)
        await client.initialize()
        set_mcp_client(client)

        logger.info(
            "[MCPInit] MCP client initialized: servers=%s, tools=%s",
            client.server_count,
            client.tool_count,
        )
    except ImportError as exc:
        logger.warning("[MCPInit] dependency missing: %s", exc)
    except Exception as exc:
        logger.warning("[MCPInit] client initialization failed: %s", exc)


def load_mcp_tools() -> list[Any]:
    """从已初始化的 MCP 客户端加载工具列表。"""
    from agent.mcp import get_mcp_client

    mcp_client = get_mcp_client()
    if mcp_client is None:
        return []

    try:
        if hasattr(mcp_client, "get_tool_adapters"):
            tools = list(mcp_client.get_tool_adapters())
            logger.info("[MCPInit] loaded MCP tools: count=%s", len(tools))
            return tools
        if hasattr(mcp_client, "tools"):
            tools = list(mcp_client.tools)
            logger.info("[MCPInit] loaded MCP tools: count=%s", len(tools))
            return tools
        logger.info("[MCPInit] MCP client has no tool adapters")
        return []
    except Exception as exc:
        logger.warning("[MCPInit] failed to load MCP tools: %s", exc)
        return []


async def initialize_mcp_system_async(domain_profile: Any) -> None:
    """异步批量初始化 MCP 系统。"""
    logger.info("[MCPInit] initializing MCP system asynchronously: domain=%s", domain_profile.profile_id)

    await init_mcp_client_async(domain_profile)

    from agent.mcp import get_mcp_client

    client = get_mcp_client()
    if client is None:
        logger.info("[MCPInit] MCP system initialization completed without active client")
        return

    logger.info(
        "[MCPInit] MCP system initialized: servers=%s, tools=%s",
        client.server_count,
        client.tool_count,
    )
