# -*- coding: utf-8 -*-
"""MCP 系统初始化器

负责初始化 MCP（Model Context Protocol）系统：
- MCP Client：连接到 MCP Server
- MCP Tools：从 MCP Server 获取可用工具

设计原则：
1. 在程序启动时完成所有初始化
2. MCP 配置从私域配置（domain_profile）加载
3. 支持多个 MCP Server 连接
4. 初始化后设置全局单例，其他模块通过单例访问
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# ============================================================================
# MCP Client 初始化
# ============================================================================

async def init_mcp_client_async(
    domain_profile: Any,
) -> None:
    """异步初始化 MCP 客户端并设置全局单例

    在异步上下文中初始化 MCP 客户端，确保 anyio 能正确检测到异步后端。
    从领域配置加载 MCP Server 配置，建立连接，
    并设置全局单例供其他模块访问。

    Args:
        domain_profile: 私域配置实例（DomainProfile）
    """
    import os

    # 检查 MCP 功能是否启用
    mcp_enabled = os.getenv("WORKFLOW_MCP_ENABLED", "false").lower().strip()
    if mcp_enabled not in ("true", "1", "yes"):
        logger.info("[MCPInit] MCP 功能未启用（WORKFLOW_MCP_ENABLED != true）")
        return

    try:
        # 导入实际 MCP 客户端和配置加载器
        from agent.mcp import MCPClient, set_mcp_client
        from agent.mcp.config_loader import MCPServerConfigLoader

        # 从领域目录加载配置
        loader = MCPServerConfigLoader()
        domain_dir = domain_profile.domain_dir

        configs = loader.load_from_domain(domain_dir)

        if not configs:
            logger.info("[MCPInit] 无 MCP Server 配置")
            return

        # 获取启用的配置
        enabled_configs = {k: v for k, v in configs.items() if v.enabled}

        if not enabled_configs:
            logger.info("[MCPInit] 无启用的 MCP Server")
            return

        # 创建 MCP 客户端
        client = MCPClient(enabled_configs)

        # 在异步上下文中直接调用异步初始化方法
        # 这样 anyio 能正确检测到 asyncio 后端
        await client.initialize()

        # 设置全局单例
        set_mcp_client(client)

        logger.info(
            f"[MCPInit] MCP 客户端初始化完成: "
            f"servers={client.server_count}, "
            f"tools={client.tool_count}, "
            f"已设置全局单例"
        )

    except ImportError as e:
        logger.warning(f"[MCPInit] MCP 依赖未安装，跳过初始化: {e}")

    except Exception as e:
        logger.warning(f"[MCPInit] MCP 客户端初始化失败: {e}")


def init_mcp_client(
    domain_profile: Any,
) -> None:
    """同步初始化 MCP 客户端（已废弃，保留向后兼容）

    注意：此方法在异步上下文中可能无法正常工作。
    建议在 FastAPI 等异步框架中使用 init_mcp_client_async()。

    Args:
        domain_profile: 私域配置实例（DomainProfile）
    """
    import os

    # 检查 MCP 功能是否启用
    mcp_enabled = os.getenv("WORKFLOW_MCP_ENABLED", "false").lower().strip()
    if mcp_enabled not in ("true", "1", "yes"):
        logger.info("[MCPInit] MCP 功能未启用（WORKFLOW_MCP_ENABLED != true）")
        return

    try:
        # 导入实际 MCP 客户端和配置加载器
        from agent.mcp import MCPClient, set_mcp_client
        from agent.mcp.config_loader import MCPServerConfigLoader

        # 从领域目录加载配置
        loader = MCPServerConfigLoader()
        domain_dir = domain_profile.domain_dir

        configs = loader.load_from_domain(domain_dir)

        if not configs:
            logger.info("[MCPInit] 无 MCP Server 配置")
            return

        # 获取启用的配置
        enabled_configs = {k: v for k, v in configs.items() if v.enabled}

        if not enabled_configs:
            logger.info("[MCPInit] 无启用的 MCP Server")
            return

        # 创建 MCP 客户端
        client = MCPClient(enabled_configs)

        # 使用 MCPClient 提供的同步初始化方法
        # 警告：在异步上下文中可能失败
        client.initialize_sync()

        # 设置全局单例
        set_mcp_client(client)

        logger.info(
            f"[MCPInit] MCP 客户端初始化完成: "
            f"servers={client.server_count}, "
            f"tools={client.tool_count}, "
            f"已设置全局单例"
        )

    except ImportError as e:
        logger.warning(f"[MCPInit] MCP 依赖未安装，跳过初始化: {e}")

    except Exception as e:
        logger.warning(f"[MCPInit] MCP 客户端初始化失败: {e}")


# ============================================================================
# MCP Tools 加载
# ============================================================================

def load_mcp_tools() -> list[Any]:
    """从 MCP Server 加载工具列表

    从全局单例获取 MCP 客户端并加载工具。

    Returns:
        工具列表（LangChain Tool 适配器）
    """
    from agent.mcp import get_mcp_client

    mcp_client = get_mcp_client()
    if mcp_client is None:
        return []

    tools: list[Any] = []

    try:
        # 从实际 MCP 客户端获取工具适配器
        if hasattr(mcp_client, "get_tool_adapters"):
            tool_adapters = mcp_client.get_tool_adapters()
            tools = list(tool_adapters)
            logger.info(f"[MCPInit] 加载 MCP 工具完成: count={len(tools)}")
        elif hasattr(mcp_client, "tools"):
            tools = list(mcp_client.tools)
            logger.info(f"[MCPInit] 加载 MCP 工具完成: count={len(tools)}")
        else:
            logger.info("[MCPInit] MCP 客户端无工具")

        return tools

    except Exception as e:
        logger.warning(f"[MCPInit] 加载 MCP 工具失败: {e}")
        return []


# ============================================================================
# 批量初始化
# ============================================================================

def initialize_mcp_system(
    domain_profile: Any,
) -> None:
    """批量初始化 MCP 系统（同步版本）

    警告：此同步版本在异步上下文中可能失败，推荐使用 initialize_mcp_system_async。

    一次性完成 Client 的初始化并设置单例。
    Tools 通过 load_mcp_tools() 单独加载。

    Args:
        domain_profile: 私域配置实例（DomainProfile）
    """
    logger.info(f"[MCPInit] 开始初始化 MCP 系统: domain={domain_profile.profile_id}")

    # 初始化 Client（内部会设置单例）
    init_mcp_client(domain_profile)

    # 获取单例检查结果
    from agent.mcp import get_mcp_client
    client = get_mcp_client()

    if client is None:
        logger.info("[MCPInit] MCP 系统初始化完成（未启用或无配置）")
    else:
        logger.info(
            f"[MCPInit] MCP 系统初始化完成: "
            f"servers={client.server_count}, "
            f"tools={client.tool_count}"
        )


async def initialize_mcp_system_async(
    domain_profile: Any,
) -> None:
    """异步批量初始化 MCP 系统（推荐）

    在异步上下文中初始化 MCP 客户端，确保 anyio 能正确检测到异步后端。
    一次性完成 Client 的初始化并设置单例。
    Tools 通过 load_mcp_tools() 单独加载。

    Args:
        domain_profile: 私域配置实例（DomainProfile）
    """
    logger.info(f"[MCPInit] 开始异步初始化 MCP 系统: domain={domain_profile.profile_id}")

    # 初始化 Client（内部会设置单例）
    await init_mcp_client_async(domain_profile)

    # 获取单例检查结果
    from agent.mcp import get_mcp_client
    client = get_mcp_client()

    if client is None:
        logger.info("[MCPInit] MCP 系统异步初始化完成（未启用或无配置）")
    else:
        logger.info(
            f"[MCPInit] MCP 系统异步初始化完成: "
            f"servers={client.server_count}, "
            f"tools={client.tool_count}"
        )
