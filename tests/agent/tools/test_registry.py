# -*- coding: utf-8 -*-
"""测试 ToolRegistry 工具注册中心"""
import pytest

from agent.tools.registry import ToolRegistry
from agent.tools.base import FunctionTool


class TestToolRegistry:
    """测试 ToolRegistry"""

    def test_create_registry(self):
        """测试创建注册中心"""
        registry = ToolRegistry()

        assert registry is not None
        assert registry.get_stats()["total_tools"] == 0

    def test_register_local_tool(self, sample_tool):
        """测试注册本地工具"""
        registry = ToolRegistry()
        registry.register_local_tool(sample_tool)

        stats = registry.get_stats()
        assert stats["local_tools"] == 1
        assert stats["total_tools"] == 1

    def test_register_mcp_tool(self, sample_tool):
        """测试注册 MCP 工具"""
        registry = ToolRegistry()
        registry.register_mcp_tool(sample_tool)

        stats = registry.get_stats()
        assert stats["mcp_tools"] == 1
        assert stats["total_tools"] == 1

    def test_get_tool_by_name(self, sample_tool):
        """测试按名称获取工具"""
        registry = ToolRegistry()
        registry.register_local_tool(sample_tool)

        tool = registry.get_tool("test_tool")

        assert tool is not None
        assert tool.name == "test_tool"

    def test_get_nonexistent_tool(self):
        """测试获取不存在的工具"""
        registry = ToolRegistry()

        tool = registry.get_tool("nonexistent")

        assert tool is None

    def test_get_tools_no_filter(self, sample_tool, sample_tool_with_params):
        """测试获取所有工具（无过滤）"""
        registry = ToolRegistry()
        registry.register_local_tool(sample_tool)
        registry.register_local_tool(sample_tool_with_params)

        tools = registry.get_tools()

        assert len(tools) == 2

    def test_get_tools_with_whitelist(self, sample_tool, sample_tool_with_params):
        """测试使用白名单获取工具"""
        registry = ToolRegistry()
        registry.register_local_tool(sample_tool)
        registry.register_local_tool(sample_tool_with_params)

        tools = registry.get_tools(whitelist=["test_tool"])

        assert len(tools) == 1
        assert tools[0].name == "test_tool"

    def test_get_tools_whitelist_nonexistent(self, sample_tool):
        """测试白名单包含不存在的工具"""
        registry = ToolRegistry()
        registry.register_local_tool(sample_tool)

        tools = registry.get_tools(whitelist=["test_tool", "nonexistent"])

        assert len(tools) == 1

    def test_get_default_tools(self, sample_tool, sample_tool_with_params):
        """测试获取默认工具集"""
        registry = ToolRegistry()
        registry.register_local_tool(sample_tool)
        registry.register_local_tool(sample_tool_with_params)
        registry.register_mcp_tool(sample_tool)  # MCP 工具

        tools = registry.get_default_tools()

        # 默认只包含本地工具，不包含 MCP
        assert len(tools) == 2

    def test_get_tools_exclude_mcp(self, sample_tool):
        """测试排除 MCP 工具"""
        registry = ToolRegistry()
        registry.register_local_tool(sample_tool)
        registry.register_mcp_tool(sample_tool)

        tools = registry.get_tools(include_local=True, include_mcp=False)

        assert len(tools) == 1
        assert tools[0].name == "test_tool"

    def test_register_tool_group(self, sample_tool, sample_tool_with_params):
        """测试注册工具分组"""
        registry = ToolRegistry()
        registry.register_local_tool(sample_tool)
        registry.register_local_tool(sample_tool_with_params)

        registry.register_tool_group("search_tools", ["test_tool", "search"])

        tools = registry.get_tools(groups=["search_tools"])

        assert len(tools) == 2

    def test_get_tools_by_group(self, sample_tool, sample_tool_with_params):
        """测试按分组获取工具"""
        registry = ToolRegistry()
        registry.register_local_tool(sample_tool)
        registry.register_local_tool(sample_tool_with_params)
        registry.register_tool_group("search", ["search"])

        tools = registry.get_tools(groups=["search"])

        assert len(tools) == 1
        assert tools[0].name == "search"

    def test_get_all_tool_schemas(self, sample_tool):
        """测试获取所有工具 Schema"""
        registry = ToolRegistry()
        registry.register_local_tool(sample_tool)

        schemas = registry.get_all_tool_schemas()

        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "test_tool"

    def test_get_schemas_with_whitelist(self, sample_tool, sample_tool_with_params):
        """测试使用白名单获取 Schema"""
        registry = ToolRegistry()
        registry.register_local_tool(sample_tool)
        registry.register_local_tool(sample_tool_with_params)

        schemas = registry.get_all_tool_schemas(whitelist=["test_tool"])

        assert len(schemas) == 1

    def test_override_tool(self, sample_tool, sample_tool_with_params):
        """测试覆盖同名工具"""
        registry = ToolRegistry()

        # 先注册一个工具
        registry.register_local_tool(sample_tool)

        # 创建同名工具
        def new_func(query: str) -> str:
            return f"新工具: {query}"

        new_tool = FunctionTool.from_function(
            func=new_func,
            name="test_tool",  # 同名
            description="新工具",
        )
        registry.register_local_tool(new_tool)

        # 应该只有一个工具（被覆盖）
        stats = registry.get_stats()
        assert stats["local_tools"] == 1

        # 获取的应该是新工具
        tool = registry.get_tool("test_tool")
        assert "新工具" in tool.run(query="测试")

    def test_clear_registry(self, sample_tool):
        """测试清空注册中心"""
        registry = ToolRegistry()
        registry.register_local_tool(sample_tool)

        registry.clear()

        stats = registry.get_stats()
        assert stats["total_tools"] == 0

    def test_list_tool_names(self, sample_tool, sample_tool_with_params):
        """测试列出所有工具名称"""
        registry = ToolRegistry()
        registry.register_local_tool(sample_tool)
        registry.register_mcp_tool(sample_tool_with_params)

        names = registry.list_tool_names()

        assert len(names) == 2
        assert "test_tool" in names
        assert "search" in names

    def test_list_local_tool_names(self, sample_tool):
        """测试列出本地工具名称"""
        registry = ToolRegistry()
        registry.register_local_tool(sample_tool)

        names = registry.list_local_tool_names()

        assert "test_tool" in names

    def test_list_mcp_tool_names(self, sample_tool):
        """测试列出 MCP 工具名称"""
        registry = ToolRegistry()
        registry.register_mcp_tool(sample_tool)

        names = registry.list_mcp_tool_names()

        assert "test_tool" in names

    def test_get_stats(self, sample_tool, sample_tool_with_params):
        """测试获取统计信息"""
        registry = ToolRegistry()
        registry.register_local_tool(sample_tool)
        registry.register_mcp_tool(sample_tool_with_params)
        registry.register_tool_group("test_group", ["test_tool"])

        stats = registry.get_stats()

        assert stats["local_tools"] == 1
        assert stats["mcp_tools"] == 1
        assert stats["tool_groups"] == 1
        assert stats["total_tools"] == 2

    def test_search_priority_local_over_mcp(self, sample_tool, sample_tool_with_params):
        """测试搜索优先级（本地 > MCP）"""
        registry = ToolRegistry()

        # 创建同名工具
        def local_func() -> str:
            return "local"

        def mcp_func() -> str:
            return "mcp"

        local_tool = FunctionTool.from_function(
            func=local_func,
            name="same_name",
            description="本地工具",
        )
        mcp_tool = FunctionTool.from_function(
            func=mcp_func,
            name="same_name",
            description="MCP 工具",
        )

        registry.register_mcp_tool(mcp_tool)
        registry.register_local_tool(local_tool)

        # 应该优先返回本地工具
        tool = registry.get_tool("same_name")
        assert tool.run() == "local"


class TestToolRegistryIntegration:
    """测试 ToolRegistry 集成场景"""

    def test_realistic_tool_setup(self):
        """测试真实工具配置场景"""
        registry = ToolRegistry()

        # 注册本地工具 (包括 SkillManager 作为统一入口)
        def wiki_search(query: str) -> str:
            return f"Wiki: {query}"

        def code_search(query: str) -> str:
            return f"Code: {query}"

        def skill_manager(skill_name: str, arguments: dict) -> str:
            """SkillManager: 统一入口管理所有 Skill"""
            return f"Executed skill: {skill_name} with {arguments}"

        wiki_tool = FunctionTool.from_function(
            func=wiki_search,
            name="wiki_search",
            description="搜索 Wiki 文档",
        )
        code_tool = FunctionTool.from_function(
            func=code_search,
            name="code_search",
            description="搜索代码库",
        )
        skill_mgr_tool = FunctionTool.from_function(
            func=skill_manager,
            name="skill_manager",
            description="调用预定义的技能",
        )

        registry.register_local_tool(wiki_tool)
        registry.register_local_tool(code_tool)
        registry.register_local_tool(skill_mgr_tool)  # SkillManager 作为本地工具

        # 注册工具分组
        registry.register_tool_group("search", ["wiki_search", "code_search"])
        registry.register_tool_group("all", ["wiki_search", "code_search", "skill_manager"])

        # 验证
        tools = registry.get_tools(groups=["search"])
        assert len(tools) == 2

        schemas = registry.get_all_tool_schemas()
        assert len(schemas) == 3  # 包括 skill_manager

    def test_skill_manager_as_local_tool(self):
        """测试 SkillManager 作为本地工具的场景"""
        registry = ToolRegistry()

        # 注册 SkillManager
        def skill_manager(skill_name: str, arguments: dict = None) -> str:
            """统一入口管理所有 Skill"""
            return f"Skill {skill_name} executed"

        skill_mgr = FunctionTool.from_function(
            func=skill_manager,
            name="skill_manager",
            description="调用预定义的技能来处理特定任务",
        )
        registry.register_local_tool(skill_mgr)

        # 验证
        stats = registry.get_stats()
        assert stats["local_tools"] == 1

        tool = registry.get_tool("skill_manager")
        assert tool is not None
        assert "skill" in tool.run(skill_name="test", arguments={}).lower()
