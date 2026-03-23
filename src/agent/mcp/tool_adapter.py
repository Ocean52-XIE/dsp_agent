# -*- coding: utf-8 -*-
"""Async MCP tool adapters for LangChain tools."""
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
    """Convert a JSON schema field definition into a Pydantic field."""
    json_type = schema.get("type", "string")
    description = schema.get("description", "")

    type_map = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    field_type = type_map.get(json_type, str)

    if json_type == "object" and "properties" in schema:
        nested_fields: dict[str, tuple[Any, Any]] = {}
        nested_required = set(schema.get("required", []))
        for prop_name, prop_schema in schema.get("properties", {}).items():
            nested_fields[prop_name] = _schema_to_pydantic_field(
                prop_name,
                prop_schema,
                prop_name in nested_required,
            )
        field_type = create_model(f"{name.capitalize()}Model", **nested_fields)

    default = ... if required else None
    return field_type, Field(default=default, description=description)


def _create_args_model(tool_info: MCPToolInfo) -> type[BaseModel]:
    """Build the Pydantic args model for an MCP tool schema."""
    input_schema = tool_info.input_schema or {}
    properties = input_schema.get("properties", {})
    required = set(input_schema.get("required", []))

    if not properties:
        return create_model(f"{tool_info.name}Args", __base__=BaseModel)

    fields: dict[str, tuple[Any, Any]] = {}
    for prop_name, prop_schema in properties.items():
        fields[prop_name] = _schema_to_pydantic_field(
            prop_name,
            prop_schema,
            prop_name in required,
        )
    return create_model(f"{tool_info.name}Args", __base__=BaseModel, **fields)


class MCPToolAdapter(BaseTool):
    """Adapt an MCP tool into an async LangChain tool."""

    client: MCPClient
    tool_info: MCPToolInfo

    def __init__(
        self,
        client: MCPClient,
        tool_info: MCPToolInfo,
        **kwargs: Any,
    ) -> None:
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
            "[MCPToolAdapter] created adapter: name=%s server=%s",
            tool_info.name,
            tool_info.server_name,
        )

    def get_openai_schema(self) -> dict[str, Any]:
        """Return the OpenAI function-calling schema for this tool."""
        input_schema = dict(self.tool_info.input_schema or {})
        input_schema.setdefault("type", "object")
        input_schema.setdefault("properties", {})
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": input_schema,
            },
        }

    def to_openai_schema(self) -> dict[str, Any]:
        """Alias kept for existing registry integration."""
        return self.get_openai_schema()

    def _run(self, *args: Any, **kwargs: Any) -> str:
        raise ToolException(
            f"MCP tool '{self.name}' only supports async invocation. Use ainvoke() instead."
        )

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        """Invoke the MCP tool asynchronously."""
        logger.debug("[MCPToolAdapter] calling tool: name=%s args=%s", self.name, kwargs)
        result = await self.client.call_tool(self.name, kwargs)
        if not result.success:
            raise ToolException(
                f"MCP Tool call failed: {self.name}, error={result.error}"
            )
        return result.content

    @property
    def server_name(self) -> str:
        return self.tool_info.server_name

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "server_name": self.server_name,
            "input_schema": self.tool_info.input_schema,
        }

    def __repr__(self) -> str:
        return f"MCPToolAdapter(name={self.name}, server={self.server_name})"


def create_mcp_tool_adapters(client: MCPClient) -> list[MCPToolAdapter]:
    """Return all adapters exposed by the MCP client."""
    return client.get_tool_adapters()


def get_all_openai_schemas(adapters: list[MCPToolAdapter]) -> list[dict[str, Any]]:
    """Return OpenAI function schemas for all adapters."""
    return [adapter.get_openai_schema() for adapter in adapters]
