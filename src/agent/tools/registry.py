# -*- coding: utf-8 -*-
"""工具注册中心

统一管理两类工具：
1. Local Tools: 本地 Python 工具 (包括 SkillManager)
2. MCP Tools: MCP 协议工具

设计说明：
- SkillManager 作为一个本地工具，内部管理所有 Skill 的调用
- LLM 通过调用 skill_manager 来执行 Skill，参数为 skill_name 和 skill_args
- 这种设计的优势：
  * 工具列表稳定，不因 Skill 数量变化而膨胀
  * 动态扩展：可以热加载新 Skill 而不需要重新注册工具
  * 统一入口：便于添加日志、监控、缓存等横切关注点

所有工具统一使用 LangChain BaseTool 格式，需要实现 to_openai_schema() 方法。
"""
import logging
from typing import Any

from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """工具注册中心

    统一管理所有工具，支持：
    - 按名称注册和获取工具
    - 按白名单过滤工具
    - 获取 OpenAI 兼容的 Schema

    Attributes:
        _local_tools: 本地工具字典 (包括 SkillManager)
        _mcp_tools: MCP 工具字典
        _tool_groups: 工具分组 (预设工具集)
    """

    def __init__(self):
        """初始化工具注册中心"""
        self._local_tools: dict[str, BaseTool] = {}
        self._mcp_tools: dict[str, BaseTool] = {}
        self._tool_groups: dict[str, list[str]] = {}

        logger.info("[ToolRegistry] 初始化完成")

    def _log_tool_summary(self) -> None:
        """打印工具加载摘要日志"""
        local_names = sorted(self._local_tools.keys())
        mcp_names = sorted(self._mcp_tools.keys())

        logger.info(
            f"[ToolRegistry] 工具加载摘要: "
            f"本地工具({len(local_names)})={local_names}, "
            f"MCP工具({len(mcp_names)})={mcp_names}, "
            f"总计={len(local_names) + len(mcp_names)}"
        )

    # =========================================================================
    # 注册方法
    # =========================================================================

    def register_local_tool(self, tool: BaseTool) -> None:
        """注册本地工具

        本地工具包括：
        - 业务工具 (如 web_search, code_retriever 等)
        - SkillManager (作为统一入口管理所有 Skill)

        Args:
            tool: 工具实例
        """
        if tool.name in self._local_tools:
            logger.warning(f"[ToolRegistry] 覆盖本地工具: {tool.name}")

        self._local_tools[tool.name] = tool
        logger.debug(f"[ToolRegistry] 注册本地工具: {tool.name}")

    def register_mcp_tool(self, tool: BaseTool) -> None:
        """注册 MCP 工具

        MCP 工具通过 MCP 协议从外部服务器获取，
        由 MCPToolAdapter 适配为 LangChain Tool 格式。

        Args:
            tool: MCP 工具适配器实例
        """
        if tool.name in self._mcp_tools:
            logger.warning(f"[ToolRegistry] 覆盖 MCP 工具: {tool.name}")

        self._mcp_tools[tool.name] = tool
        logger.debug(f"[ToolRegistry] 注册 MCP 工具: {tool.name}")

    def register_tool_group(self, group_name: str, tool_names: list[str]) -> None:
        """注册工具分组

        工具分组允许预设一组工具，便于按场景切换。

        Args:
            group_name: 分组名称
            tool_names: 工具名称列表
        """
        self._tool_groups[group_name] = tool_names
        logger.debug(f"[ToolRegistry] 注册工具分组: {group_name}, 工具数={len(tool_names)}")

    # =========================================================================
    # 获取方法
    # =========================================================================

    def get_tool(self, name: str) -> BaseTool | None:
        """按名称获取工具

        搜索顺序：本地 -> MCP

        Args:
            name: 工具名称

        Returns:
            工具实例，未找到返回 None
        """
        if name in self._local_tools:
            return self._local_tools[name]
        if name in self._mcp_tools:
            return self._mcp_tools[name]
        return None

    def get_tools(
        self,
        whitelist: list[str] | None = None,
        groups: list[str] | None = None,
        include_local: bool = True,
        include_mcp: bool = True,
    ) -> list[BaseTool]:
        """获取工具列表

        支持多种过滤方式，优先级：白名单 > 分组 > 默认

        Args:
            whitelist: 工具名称白名单
            groups: 工具分组列表
            include_local: 是否包含本地工具 (默认 True)
            include_mcp: 是否包含 MCP 工具 (默认 True)

        Returns:
            工具实例列表
        """
        result: list[BaseTool] = []

        # 优先级 1: 按白名单获取
        if whitelist:
            for name in whitelist:
                tool = self.get_tool(name)
                if tool:
                    result.append(tool)
                else:
                    logger.warning(f"[ToolRegistry] 白名单工具不存在: {name}")
            logger.info(
                f"[ToolRegistry] 按白名单获取工具: {len(result)} 个, "
                f"工具列表={sorted([t.name for t in result])}"
            )
            return result

        # 优先级 2: 按分组获取
        if groups:
            for group in groups:
                if group in self._tool_groups:
                    for name in self._tool_groups[group]:
                        tool = self.get_tool(name)
                        if tool:
                            result.append(tool)
            logger.info(
                f"[ToolRegistry] 按分组获取工具: {len(result)} 个, "
                f"分组={groups}, 工具列表={sorted([t.name for t in result])}"
            )
            return result

        # 优先级 3: 默认获取
        if include_local:
            result.extend(self._local_tools.values())
        if include_mcp:
            result.extend(self._mcp_tools.values())

        logger.info(
            f"[ToolRegistry] 获取默认工具: {len(result)} 个, "
            f"工具列表={sorted([t.name for t in result])}"
        )
        return result

    def get_default_tools(self) -> list[BaseTool]:
        """获取默认工具集

        默认只包含本地工具 (包括 SkillManager)，不包含 MCP 工具。
        MCP 工具需要显式通过 include_mcp=True 或白名单指定。

        Returns:
            工具实例列表
        """
        return self.get_tools(include_local=True, include_mcp=False)

    # =========================================================================
    # Schema 方法
    # =========================================================================

    def get_all_tool_schemas(
        self,
        whitelist: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """获取所有工具的 OpenAI Schema

        Args:
            whitelist: 工具名称白名单 (可选)

        Returns:
            OpenAI 兼容的工具 Schema 列表
        """
        tools = self.get_tools(whitelist) if whitelist else self.get_all_tools()
        return [tool.to_openai_schema() for tool in tools]

    def get_all_tools(self) -> list[BaseTool]:
        """获取所有已注册的工具

        Returns:
            所有工具实例列表
        """
        all_tools: list[BaseTool] = []
        all_tools.extend(self._local_tools.values())
        all_tools.extend(self._mcp_tools.values())
        return all_tools

    # =========================================================================
    # 辅助方法
    # =========================================================================

    def clear(self) -> None:
        """清空所有工具"""
        self._local_tools.clear()
        self._mcp_tools.clear()
        self._tool_groups.clear()
        logger.info("[ToolRegistry] 已清空")

    def get_stats(self) -> dict[str, Any]:
        """获取统计信息

        Returns:
            统计信息字典
        """
        return {
            "local_tools": len(self._local_tools),
            "mcp_tools": len(self._mcp_tools),
            "tool_groups": len(self._tool_groups),
            "total_tools": len(self._local_tools) + len(self._mcp_tools),
            "local_tool_names": sorted(self._local_tools.keys()),
            "mcp_tool_names": sorted(self._mcp_tools.keys()),
        }

    def log_summary(self) -> None:
        """打印工具加载摘要日志（公共方法）

        在所有工具注册完成后调用，打印当前加载的工具摘要。
        """
        self._log_tool_summary()

    def list_tool_names(self) -> list[str]:
        """列出所有工具名称

        Returns:
            工具名称列表
        """
        names: set[str] = set()
        names.update(self._local_tools.keys())
        names.update(self._mcp_tools.keys())
        return sorted(names)

    @property
    def total_count(self) -> int:
        """获取工具总数

        Returns:
            本地工具 + MCP 工具的总数
        """
        return len(self._local_tools) + len(self._mcp_tools)

    def list_local_tool_names(self) -> list[str]:
        """列出所有本地工具名称

        Returns:
            本地工具名称列表
        """
        return sorted(self._local_tools.keys())

    def list_mcp_tool_names(self) -> list[str]:
        """列出所有 MCP 工具名称

        Returns:
            MCP 工具名称列表
        """
        return sorted(self._mcp_tools.keys())

    def __repr__(self) -> str:
        stats = self.get_stats()
        return f"ToolRegistry(total={stats['total_tools']}, local={stats['local_tools']}, mcp={stats['mcp_tools']})"


# ============================================================================
# 全局单例
# ============================================================================

_tool_registry: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    """获取全局工具注册中心单例"""
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = ToolRegistry()
    return _tool_registry


def set_tool_registry(registry: ToolRegistry) -> None:
    """设置全局工具注册中心（启动时初始化使用）

    Args:
        registry: 工具注册中心实例
    """
    global _tool_registry
    _tool_registry = registry


def reset_tool_registry() -> None:
    """重置全局工具注册中心（用于测试）"""
    global _tool_registry
    if _tool_registry is not None:
        _tool_registry.clear()
    _tool_registry = None
