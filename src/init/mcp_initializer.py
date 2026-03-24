# -*- coding: utf-8 -*-
"""MCP system initializer."""
from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter
from typing import Any

from log import get_file_logger


def _project_root_from_domain(domain_profile: Any) -> Path:
    domain_dir = getattr(domain_profile, "domain_dir", None)
    if isinstance(domain_dir, Path):
        return domain_dir.parent.parent.resolve()
    return Path(__file__).resolve().parents[2]


def _build_mcp_summary(
    *,
    enabled: bool,
    servers: list[str],
    tools: list[str],
    server_tool_map: dict[str, list[str]],
    latency_ms: int = 0,
    reason: str = "",
    error_type: str = "",
) -> dict[str, Any]:
    return {
        "enabled": enabled,
        "servers": servers,
        "tools": tools,
        "server_tool_map": server_tool_map,
        "server_count": len(servers),
        "tool_count": len(tools),
        "latency_ms": latency_ms,
        "reason": reason or None,
        "error_type": error_type or None,
    }


async def init_mcp_client_async(domain_profile: Any) -> dict[str, Any]:
    """Initialize the MCP client and return a startup summary."""
    runtime_logger = get_file_logger(project_root=_project_root_from_domain(domain_profile))
    mcp_enabled = os.getenv("AGENT_MCP_ENABLED", "false").lower().strip()
    if mcp_enabled not in {"true", "1", "yes"}:
        return _build_mcp_summary(
            enabled=False,
            servers=[],
            tools=[],
            server_tool_map={},
            reason="config_disabled",
        )

    try:
        from agent.mcp import MCPClient, set_mcp_client
        from agent.mcp.config_loader import MCPServerConfigLoader

        loader = MCPServerConfigLoader()
        configs = loader.load_from_domain(domain_profile.domain_dir)
        servers = sorted(configs.keys())
        enabled_configs = {name: config for name, config in configs.items() if config.enabled}
        enabled_servers = sorted(enabled_configs.keys())
        runtime_logger.info(
            "init.mcp.config.loaded",
            domain=domain_profile.profile_id,
            servers=servers,
            enabled_servers=enabled_servers,
        )
        if not configs:
            return _build_mcp_summary(
                enabled=False,
                servers=[],
                tools=[],
                server_tool_map={},
                reason="no_server_configs",
            )
        if not enabled_configs:
            return _build_mcp_summary(
                enabled=False,
                servers=[],
                tools=[],
                server_tool_map={},
                reason="no_enabled_servers",
            )

        client = MCPClient(enabled_configs)
        await client.initialize()
        set_mcp_client(client)
        return _build_mcp_summary(
            enabled=True,
            servers=client.get_server_names(),
            tools=client.get_tool_names(),
            server_tool_map=client.get_server_tool_map(),
        )
    except ImportError as exc:
        return _build_mcp_summary(
            enabled=False,
            servers=[],
            tools=[],
            server_tool_map={},
            reason="import_dependency_failed",
            error_type=type(exc).__name__,
        )
    except Exception as exc:
        return _build_mcp_summary(
            enabled=False,
            servers=[],
            tools=[],
            server_tool_map={},
            reason="client_initialization_failed",
            error_type=type(exc).__name__,
        )


def load_mcp_tools() -> list[Any]:
    """Return initialized MCP tools for agent assembly."""
    from agent.mcp import get_mcp_client

    mcp_client = get_mcp_client()
    if mcp_client is None:
        return []

    try:
        if hasattr(mcp_client, "get_tool_adapters"):
            return list(mcp_client.get_tool_adapters())
        if hasattr(mcp_client, "tools"):
            return list(mcp_client.tools)
    except Exception:
        return []
    return []


async def initialize_mcp_system_async(domain_profile: Any) -> dict[str, Any]:
    """Initialize the MCP subsystem and return a structured summary."""
    runtime_logger = get_file_logger(project_root=_project_root_from_domain(domain_profile))
    runtime_logger.info("init.mcp.begin", domain=domain_profile.profile_id)
    started_at = perf_counter()
    summary = await init_mcp_client_async(domain_profile)
    summary["latency_ms"] = int((perf_counter() - started_at) * 1000)
    runtime_logger.info("init.mcp.completed", domain=domain_profile.profile_id, **summary)
    return summary
