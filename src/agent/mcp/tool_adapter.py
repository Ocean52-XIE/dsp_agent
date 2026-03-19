# -*- coding: utf-8 -*-
"""MCP Tool Adapter - 将 MCP Tool 适配为 LangChain Tool

将 MCP Server 提供的工具转换为 LangChain Tool 格式，
使其可以被 AgentLoop 统一调用。

使用示例：
    client = MCPClient(configs)
    await client.initialize()

    # 获取适配器
    adapters = client.get_tool_adapters()

    # 获取 OpenAI Schema
    schema = adapters[0].get_openai_schema()

    # 调用工具
    result = await adapters[0].ainvoke({"query": "test"})
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field, create_model

from agent.mcp.client import MCPClient, MCPToolInfo

logger = logging.getLogger(__name__)


def _schema_to_pydantic_field(
    name: str,
    schema: dict[str, Any],
    required: bool = True,
) -> tuple[Any, Any]:
    """将 JSON Schema 属性转换为 Pydantic Field

    Args:
        name: 字段名称
        schema: JSON Schema 定义
        required: 是否必需

    Returns:
        (field_type, field_default) 元组
    """
    # 提取类型
    json_type = schema.get("type", "string")
    description = schema.get("description", "")

    # 类型映射
    type_map = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    field_type = type_map.get(json_type, str)

    # 处理嵌套对象
    if json_type == "object" and "properties" in schema:
        # 创建嵌套模型
        nested_fields = {}
        nested_required = set(schema.get("required", []))

        for prop_name, prop_schema in schema.get("properties", {}).items():
            nested_fields[prop_name] = _schema_to_pydantic_field(
                prop_name,
                prop_schema,
                prop_name in nested_required,
            )

        field_type = create_model(f"{name.capitalize()}Model", **nested_fields)

    # 设置默认值
    if required:
        default = ...
    else:
        default = None

    return (field_type, Field(default=default, description=description))


def _create_args_model(tool_info: MCPToolInfo) -> type[BaseModel]:
    """从 MCP Tool Schema 创建 Pydantic 模型

    Args:
        tool_info: MCP 工具信息

    Returns:
        Pydantic 模型类
    """
    input_schema = tool_info.input_schema or {}
    properties = input_schema.get("properties", {})
    required = set(input_schema.get("required", []))

    if not properties:
        # 无参数的工具
        return create_model(
            f"{tool_info.name}Args",
            __base__=BaseModel,
        )

    fields = {}
    for prop_name, prop_schema in properties.items():
        fields[prop_name] = _schema_to_pydantic_field(
            prop_name,
            prop_schema,
            prop_name in required,
        )

    return create_model(
        f"{tool_info.name}Args",
        __base__=BaseModel,
        **fields,
    )


class MCPToolAdapter(BaseTool):
    """MCP Tool 适配器

    将 MCP Tool 适配为 LangChain Tool，支持：
    1. 转换为 OpenAI Function Calling Schema
    2. 异步调用
    3. 参数验证

    Attributes:
        client: MCP 客户端
        tool_info: MCP 工具信息
    """

    client: MCPClient
    tool_info: MCPToolInfo

    def __init__(
        self,
        client: MCPClient,
        tool_info: MCPToolInfo,
        **kwargs: Any,
    ):
        """初始化适配器

        Args:
            client: MCP 客户端
            tool_info: MCP 工具信息
            **kwargs: 其他参数
        """
        # 创建参数模型
        args_schema = _create_args_model(tool_info)

        super().__init__(
            name=tool_info.name,
            description=tool_info.description or f"MCP Tool: {tool_info.name}",
            args_schema=args_schema,
            client=client,
            tool_info=tool_info,
            **kwargs,
        )

        logger.debug(
            f"[MCPToolAdapter] 创建适配器: {tool_info.name}, "
            f"server={tool_info.server_name}"
        )

    def get_openai_schema(self) -> dict[str, Any]:
        """获取 OpenAI Function Calling Schema

        Returns:
            OpenAI 格式的工具 Schema
        """
        input_schema = self.tool_info.input_schema or {}

        # 确保有 type 字段
        if "type" not in input_schema:
            input_schema["type"] = "object"

        # 确保有 properties 字段
        if "properties" not in input_schema:
            input_schema["properties"] = {}

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": input_schema,
            },
        }

    def to_openai_schema(self) -> dict[str, Any]:
        """获取 OpenAI Function Calling Schema (别名方法)

        兼容 ToolRegistry 接口。

        Returns:
            OpenAI 格式的工具 Schema
        """
        return self.get_openai_schema()

    def _run(self, *args: Any, **kwargs: Any) -> str:
        """同步调用（不支持，抛出异常）

        MCP 工具只支持异步调用。
        """
        raise NotImplementedError(
            "MCPToolAdapter only supports async invocation. "
            "Use ainvoke() instead."
        )

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        """异步调用工具

        Args:
            *args: 位置参数（忽略）
            **kwargs: 工具参数

        Returns:
            工具调用结果
        """
        logger.debug(
            f"[MCPToolAdapter] 调用工具: {self.name}, args={kwargs}"
        )

        result = await self.client.call_tool(self.name, kwargs)

        if not result.success:
            raise ToolException(
                f"MCP Tool call failed: {self.name}, error={result.error}"
            )

        return result.content

    @property
    def server_name(self) -> str:
        """所属 Server 名称"""
        return self.tool_info.server_name

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "description": self.description,
            "server_name": self.server_name,
            "input_schema": self.tool_info.input_schema,
        }

    def __repr__(self) -> str:
        return f"MCPToolAdapter(name={self.name}, server={self.server_name})"


def create_mcp_tool_adapters(
    client: MCPClient,
) -> list[MCPToolAdapter]:
    """为 MCP Client 的所有工具创建适配器

    Args:
        client: MCP 客户端

    Returns:
        MCPToolAdapter 列表
    """
    return client.get_tool_adapters()


def get_all_openai_schemas(
    adapters: list[MCPToolAdapter],
) -> list[dict[str, Any]]:
    """获取所有适配器的 OpenAI Schema

    Args:
        adapters: 适配器列表

    Returns:
        OpenAI Schema 列表
    """
    return [adapter.get_openai_schema() for adapter in adapters]
