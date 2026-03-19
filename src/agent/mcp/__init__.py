# -*- coding: utf-8 -*-
"""Agent MCP 模块

MCP (Model Context Protocol) 集成模块，使用 langchain-mcp-adapters 实现。

提供：
- MCPClient: MCP 客户端，管理多个 Server 连接
- MCPToolAdapter: 将 MCP Tool 适配为 LangChain Tool
- MCPServerConfigLoader: 从配置文件加载 Server 配置

使用示例：
    from agent.mcp import MCPClient, MCPServerConfigLoader
    from pathlib import Path

    # 加载配置
    loader = MCPServerConfigLoader()
    configs = loader.load_from_domain(Path("domain/ad_engine"))

    # 创建客户端
    client = MCPClient(configs)
    await client.initialize()

    # 获取工具适配器并注册到 ToolRegistry
    adapters = client.get_tool_adapters()
    for adapter in adapters:
        tool_registry.register_mcp_tool(adapter)

    # 关闭
    await client.shutdown()
"""
from agent.mcp.config_loader import (
    MCPServerConfig,
    MCPServerConfigLoader,
)
from agent.mcp.client import (
    MCPClient,
    MCPToolCallResult,
    MCPToolInfo,
)
from agent.mcp.tool_adapter import (
    MCPToolAdapter,
    create_mcp_tool_adapters,
    get_all_openai_schemas,
)

__all__ = [
    # Config
    "MCPServerConfig",
    "MCPServerConfigLoader",
    # Client
    "MCPClient",
    "MCPToolCallResult",
    "MCPToolInfo",
    # Tool Adapter
    "MCPToolAdapter",
    "create_mcp_tool_adapters",
    "get_all_openai_schemas",
]
