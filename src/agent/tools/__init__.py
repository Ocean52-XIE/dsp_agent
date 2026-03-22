# -*- coding: utf-8 -*-
"""工具模块

提供统一的工具注册和管理。

所有工具统一使用 LangChain BaseTool，需要实现 to_openai_schema() 方法。

导出：
- ToolRegistry: 工具注册中心
- BaseTool: LangChain 工具基类（从 langchain_core.tools 导入）
- get_tool_registry: 获取全局工具注册中心单例
- set_tool_registry: 设置全局工具注册中心
- reset_tool_registry: 重置全局工具注册中心
"""
from agent.tools.registry import (
    ToolRegistry,
    get_tool_registry,
    set_tool_registry,
    reset_tool_registry,
)
from langchain_core.tools import BaseTool

__all__ = [
    "ToolRegistry",
    "BaseTool",
    "get_tool_registry",
    "set_tool_registry",
    "reset_tool_registry",
]
