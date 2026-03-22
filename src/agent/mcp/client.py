# -*- coding: utf-8 -*-
"""MCP Client - MCP 客户端核心

使用 langchain-mcp-adapters 的 MultiServerMCPClient 管理多个 MCP Server 连接。

重要说明：
    langchain-mcp-adapters 使用 anyio 库，需要正确的异步上下文。
    本模块只提供异步方法，不支持同步调用。

使用示例（异步）：
    from agent.mcp import MCPClient, MCPServerConfigLoader

    # 加载配置
    loader = MCPServerConfigLoader()
    configs = loader.load_from_domain(Path("domain/ad_engine"))

    # 创建客户端
    client = MCPClient(configs)

    # 初始化连接（异步）
    await client.initialize()

    # 调用工具（异步）
    result = await client.call_tool("query_metrics", {"start_date": "2024-03-01"})

    # 关闭连接（异步）
    await client.shutdown()

在 FastAPI 中使用：
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        client = MCPClient(configs)
        await client.initialize()
        set_mcp_client(client)
        yield
        await client.shutdown()
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient

from agent.mcp.config_loader import MCPServerConfig, MCPServerConfigLoader

logger = logging.getLogger(__name__)


@dataclass
class MCPToolInfo:
    """MCP 工具信息

    Attributes:
        name: 工具名称
        description: 工具描述
        input_schema: 输入 Schema
        server_name: 所属 Server 名称
    """
    name: str
    description: str
    input_schema: dict[str, Any]
    server_name: str

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "server_name": self.server_name,
        }


@dataclass
class MCPToolCallResult:
    """MCP 工具调用结果

    Attributes:
        success: 是否成功
        content: 响应内容
        error: 错误信息
        latency_ms: 耗时 (毫秒)
        server_name: Server 名称
        tool_name: 工具名称
    """
    success: bool
    content: str = ""
    error: str | None = None
    latency_ms: int = 0
    server_name: str = ""
    tool_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "success": self.success,
            "content": self.content,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "server_name": self.server_name,
            "tool_name": self.tool_name,
        }


class MCPClient:
    """MCP 客户端

    使用 langchain-mcp-adapters 的 MultiServerMCPClient 管理多个 MCP Server 连接。

    核心功能：
    1. 管理多个 Server 连接
    2. 发现并缓存所有工具
    3. 路由工具调用到正确的 Server
    4. 提供 Tool Adapter 供 AgentLoop 使用

    """

    def __init__(
        self,
        server_configs: dict[str, MCPServerConfig] | None = None,
    ):
        """初始化 MCP Client

        Args:
            server_configs: Server 配置映射，key 为 server name
        """
        self._server_configs = server_configs or {}
        self._client: MultiServerMCPClient | None = None
        self._tools: dict[str, MCPToolInfo] = {}  # tool_name -> tool_info
        self._tool_to_server: dict[str, str] = {}  # tool_name -> server_name
        self._initialized = False

        logger.info(
            f"[MCPClient] Creating instance, "
            f"servers={len(self._server_configs)}"
        )

    @classmethod
    def from_config_file(cls, config_path: Path) -> "MCPClient":
        """从配置文件创建客户端

        Args:
            config_path: 配置文件路径

        Returns:
            MCPClient 实例
        """
        loader = MCPServerConfigLoader()
        configs = loader.load_from_file(config_path)
        return cls(configs)

    @classmethod
    def from_domain(cls, domain_root: Path) -> "MCPClient":
        """从领域目录创建客户端

        Args:
            domain_root: 领域目录路径

        Returns:
            MCPClient 实例
        """
        loader = MCPServerConfigLoader()
        configs = loader.load_from_domain(domain_root)
        return cls(configs)

    async def initialize(self) -> None:
        """初始化所有 Server 连接

        遍历所有配置的 Server，尝试连接并发现工具。
        单个 Server 连接失败不会影响其他 Server。
        """
        if self._initialized:
            logger.warning("[MCPClient] Already initialized, skipping")
            return

        logger.info("[MCPClient] Starting initialization...")

        # 构建 connections 字典
        connections = {}
        for name, config in self._server_configs.items():
            if not config.enabled:
                logger.debug(f"[MCPClient] Server '{name}' is disabled, skipping")
                continue

            if config.transport == "stdio":
                # StdioConnection is a TypedDict, use dict literal
                connections[name] = {
                    "transport": "stdio",
                    "command": config.command,
                    "args": config.args,
                    **({"env": config.env} if config.env else {}),
                }
            elif config.transport == "sse":
                # SSEConnection is a TypedDict
                connections[name] = {
                    "transport": "sse",
                    "url": config.url,
                }
            elif config.transport == "websocket":
                # WebsocketConnection is a TypedDict
                connections[name] = {
                    "transport": "websocket",
                    "url": config.url,
                }
            else:
                logger.warning(
                    f"[MCPClient] Unsupported transport '{config.transport}' for server '{name}'"
                )
                continue

        if not connections:
            logger.warning("[MCPClient] No valid server connections found")
            return

        try:
            # 创建 MultiServerMCPClient (不再使用上下文管理器)
            # As of langchain-mcp-adapters 0.1.0, context manager is not supported
            self._client = MultiServerMCPClient(connections)

            # 获取所有工具 (直接调用 get_tools，不需要上下文管理器)
            for server_name in connections.keys():
                try:
                    tools = await self._client.get_tools(server_name=server_name)

                    for tool in tools:
                        # 从 LangChain Tool 提取信息
                        tool_info = MCPToolInfo(
                            name=tool.name,
                            description=tool.description or "",
                            input_schema=getattr(tool, "args_schema", {})
                            or getattr(tool, "tool_schema", {})
                            or {},
                            server_name=server_name,
                        )
                        self._tools[tool.name] = tool_info
                        self._tool_to_server[tool.name] = server_name

                    logger.info(
                        f"[MCPClient] Server '{server_name}' connected, "
                        f"tools={len(tools)}"
                    )
                except Exception as e:
                    import traceback
                    logger.error(
                        f"[MCPClient] Failed to get tools from server '{server_name}': {e}"
                    )
                    logger.debug(
                        f"[MCPClient] Full traceback for server '{server_name}':\n"
                        f"{traceback.format_exc()}"
                    )

            self._initialized = True
            logger.info(
                f"[MCPClient] Initialization complete, "
                f"servers={len(connections)}, tools={len(self._tools)}"
            )

        except Exception as e:
            logger.error(f"[MCPClient] Initialization failed: {e}")
            raise

    async def shutdown(self) -> None:
        """关闭所有 Server 连接"""
        logger.info("[MCPClient] Closing all connections...")

        # MultiServerMCPClient 不再需要显式关闭
        # 清理内部状态即可
        self._client = None

        self._tools.clear()
        self._tool_to_server.clear()
        self._initialized = False

        logger.info("[MCPClient] All connections closed")

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        timeout: float | None = None,
    ) -> MCPToolCallResult:
        """调用工具

        Args:
            tool_name: 工具名称
            arguments: 工具参数
            timeout: 超时时间（秒）

        Returns:
            MCPToolCallResult 调用结果
        """
        import time

        start_time = time.time()

        # 查找工具所属的 Server
        server_name = self._tool_to_server.get(tool_name)
        if not server_name:
            logger.warning(f"[MCPClient] Tool not found: {tool_name}")
            return MCPToolCallResult(
                success=False,
                error=f"Tool not found: {tool_name}",
                tool_name=tool_name,
            )

        if not self._client:
            logger.warning(f"[MCPClient] Client not initialized")
            return MCPToolCallResult(
                success=False,
                error="Client not initialized",
                tool_name=tool_name,
                server_name=server_name,
            )

        try:
            # 获取 LangChain 工具并调用
            tools = await self._client.get_tools(server_name=server_name)
            tool = next((t for t in tools if t.name == tool_name), None)

            if not tool:
                logger.warning(f"[MCPClient] Tool '{tool_name}' not found in server '{server_name}'")
                return MCPToolCallResult(
                    success=False,
                    error=f"Tool not found in server: {tool_name}",
                    tool_name=tool_name,
                    server_name=server_name,
                )

            # 使用 LangChain Tool 的 ainvoke 方法
            if timeout:
                result = await asyncio.wait_for(
                    tool.ainvoke(arguments),
                    timeout=timeout,
                )
            else:
                result = await tool.ainvoke(arguments)

            latency_ms = int((time.time() - start_time) * 1000)

            logger.debug(
                f"[MCPClient] Tool call successful: {tool_name}, "
                f"server={server_name}, latency={latency_ms}ms"
            )

            return MCPToolCallResult(
                success=True,
                content=str(result) if result else "",
                latency_ms=latency_ms,
                tool_name=tool_name,
                server_name=server_name,
            )

        except asyncio.TimeoutError:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.error(
                f"[MCPClient] Tool call timeout: {tool_name}, "
                f"server={server_name}, timeout={timeout}s"
            )
            return MCPToolCallResult(
                success=False,
                error=f"Tool call timeout: {timeout}s",
                latency_ms=latency_ms,
                tool_name=tool_name,
                server_name=server_name,
            )
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.error(
                f"[MCPClient] Tool call failed: {tool_name}, "
                f"server={server_name}, error={e}"
            )
            return MCPToolCallResult(
                success=False,
                error=str(e),
                latency_ms=latency_ms,
                tool_name=tool_name,
                server_name=server_name,
            )

    def get_tool_info(self, tool_name: str) -> MCPToolInfo | None:
        """获取工具信息

        Args:
            tool_name: 工具名称

        Returns:
            工具信息，不存在返回 None
        """
        return self._tools.get(tool_name)

    def get_all_tools(self) -> list[MCPToolInfo]:
        """获取所有已发现的工具

        Returns:
            工具信息列表
        """
        return list(self._tools.values())

    def get_tool_names(self) -> list[str]:
        """获取所有工具名称

        Returns:
            工具名称列表
        """
        return list(self._tools.keys())

    def get_tool_adapters(self) -> list["MCPToolAdapter"]:
        """获取所有工具的适配器

        Returns:
            MCPToolAdapter 列表
        """
        from agent.mcp.tool_adapter import MCPToolAdapter

        return [
            MCPToolAdapter(self, tool_info)
            for tool_info in self._tools.values()
        ]

    @property
    def is_initialized(self) -> bool:
        """是否已初始化"""
        return self._initialized

    @property
    def server_count(self) -> int:
        """已连接的 Server 数量"""
        return len([c for c in self._server_configs.values() if c.enabled])

    @property
    def tool_count(self) -> int:
        """已发现的工具数量"""
        return len(self._tools)

    def get_stats(self) -> dict[str, Any]:
        """获取客户端统计信息

        Returns:
            统计信息字典
        """
        return {
            "initialized": self._initialized,
            "server_count": len([c for c in self._server_configs.values() if c.enabled]),
            "tool_count": len(self._tools),
            "servers": list(self._server_configs.keys()),
            "tools": list(self._tools.keys()),
        }


# ============================================================================
# 全局单例模式
# ============================================================================

# 全局 MCP 客户端实例（单例）
_mcp_client_instance: MCPClient | None = None


def get_mcp_client() -> MCPClient | None:
    """获取全局 MCP 客户端实例

    Returns:
        MCPClient 实例，如果未初始化则返回 None
    """
    return _mcp_client_instance


def set_mcp_client(client: MCPClient | None) -> None:
    """设置全局 MCP 客户端实例

    Args:
        client: MCPClient 实例
    """
    global _mcp_client_instance
    _mcp_client_instance = client
    if client:
        logger.info("[MCPClient] 全局单例已设置")
    else:
        logger.info("[MCPClient] 全局单例已清除")
