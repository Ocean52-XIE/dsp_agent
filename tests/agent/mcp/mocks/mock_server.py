# -*- coding: utf-8 -*-
"""Mock MCP Server - 用于测试的模拟 MCP Server

提供轻量级的 MCP Server 模拟，支持：
- 工具发现 (list_tools)
- 工具调用 (call_tool)
- Stdio 传输模拟
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class MockTool:
    """模拟工具定义

    Attributes:
        name: 工具名称
        description: 工具描述
        input_schema: 输入 Schema
        handler: 工具处理函数
    """
    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    handler: Callable[[dict[str, Any]], str] | None = None

    def to_mcp_format(self) -> dict[str, Any]:
        """转换为 MCP 工具格式"""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


class MockMCPServer:
    """Mock MCP Server

    用于测试的模拟 MCP Server，可以：
    1. 作为独立进程运行，通过 stdio 通信
    2. 直接在测试中调用

    使用示例：
        # 创建并运行 Mock Server
        server = MockMCPServer()
        server.add_tool(MockTool(
            name="echo",
            description="Echo input",
            input_schema={
                "type": "object",
                "properties": {
                    "message": {"type": "string"}
                },
                "required": ["message"]
            },
            handler=lambda args: f"Echo: {args['message']}"
        ))

        # 作为独立进程运行
        await server.run_stdio()
    """

    def __init__(self, name: str = "mock_server"):
        """初始化 Mock Server

        Args:
            name: Server 名称
        """
        self.name = name
        self._tools: dict[str, MockTool] = {}

    def add_tool(self, tool: MockTool) -> None:
        """添加工具

        Args:
            tool: 工具定义
        """
        self._tools[tool.name] = tool
        logger.debug(f"[MockMCPServer] 添加工具: {tool.name}")

    def remove_tool(self, name: str) -> None:
        """移除工具

        Args:
            name: 工具名称
        """
        if name in self._tools:
            del self._tools[name]
            logger.debug(f"[MockMCPServer] 移除工具: {name}")

    def list_tools(self) -> list[dict[str, Any]]:
        """列出所有工具

        Returns:
            工具列表
        """
        return [tool.to_mcp_format() for tool in self._tools.values()]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """调用工具

        Args:
            name: 工具名称
            arguments: 工具参数

        Returns:
            工具执行结果

        Raises:
            ValueError: 工具不存在
        """
        if name not in self._tools:
            raise ValueError(f"Tool not found: {name}")

        tool = self._tools[name]

        if tool.handler:
            return tool.handler(arguments)

        return f"Mock result from {name}: {arguments}"

    async def run_stdio(self) -> None:
        """以 stdio 模式运行 Server

        从 stdin 读取 JSON-RPC 请求，将响应写入 stdout。
        """
        logger.info(f"[MockMCPServer] 启动 stdio 模式: {self.name}")

        reader = asyncio.StreamReader()
        reader_protocol = asyncio.StreamReaderProtocol(reader)
        await asyncio.get_event_loop().connect_read_pipe(
            lambda: reader_protocol,
            sys.stdin
        )

        while True:
            try:
                line = await reader.readline()
                if not line:
                    break

                request = json.loads(line.decode("utf-8").strip())
                response = await self._handle_request(request)

                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()

            except json.JSONDecodeError as e:
                logger.error(f"[MockMCPServer] JSON 解析错误: {e}")
            except Exception as e:
                logger.error(f"[MockMCPServer] 处理请求错误: {e}")
                break

        logger.info(f"[MockMCPServer] 关闭: {self.name}")

    async def _handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """处理 JSON-RPC 请求

        Args:
            request: JSON-RPC 请求

        Returns:
            JSON-RPC 响应
        """
        method = request.get("method", "")
        params = request.get("params", {})
        request_id = request.get("id")

        result = None
        error = None

        try:
            if method == "initialize":
                result = {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {
                        "name": self.name,
                        "version": "1.0.0"
                    },
                    "capabilities": {
                        "tools": {}
                    }
                }
            elif method == "tools/list":
                result = {
                    "tools": self.list_tools()
                }
            elif method == "tools/call":
                tool_name = params.get("name")
                arguments = params.get("arguments", {})
                result = {
                    "content": [
                        {"type": "text", "text": self.call_tool(tool_name, arguments)}
                    ]
                }
            else:
                error = {"code": -32601, "message": f"Method not found: {method}"}

        except Exception as e:
            error = {"code": -32603, "message": str(e)}

        response = {"jsonrpc": "2.0", "id": request_id}

        if error:
            response["error"] = error
        else:
            response["result"] = result

        return response


def create_echo_server() -> MockMCPServer:
    """创建 Echo 测试 Server

    Returns:
        配置好 echo 工具的 Mock Server
    """
    server = MockMCPServer("echo_server")

    server.add_tool(MockTool(
        name="echo",
        description="Echo the input message",
        input_schema={
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Message to echo"
                }
            },
            "required": ["message"]
        },
        handler=lambda args: f"Echo: {args['message']}"
    ))

    return server


def create_calculator_server() -> MockMCPServer:
    """创建 Calculator 测试 Server

    Returns:
        配置好计算工具的 Mock Server
    """
    server = MockMCPServer("calculator_server")

    def add_handler(args: dict[str, Any]) -> str:
        a = args.get("a", 0)
        b = args.get("b", 0)
        return str(a + b)

    def multiply_handler(args: dict[str, Any]) -> str:
        a = args.get("a", 0)
        b = args.get("b", 0)
        return str(a * b)

    server.add_tool(MockTool(
        name="add",
        description="Add two numbers",
        input_schema={
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "First number"},
                "b": {"type": "integer", "description": "Second number"}
            },
            "required": ["a", "b"]
        },
        handler=add_handler
    ))

    server.add_tool(MockTool(
        name="multiply",
        description="Multiply two numbers",
        input_schema={
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "First number"},
                "b": {"type": "integer", "description": "Second number"}
            },
            "required": ["a", "b"]
        },
        handler=multiply_handler
    ))

    return server


if __name__ == "__main__":
    # 作为独立进程运行时的入口
    import argparse

    parser = argparse.ArgumentParser(description="Mock MCP Server")
    parser.add_argument(
        "--type",
        choices=["echo", "calculator"],
        default="echo",
        help="Server type"
    )
    args = parser.parse_args()

    if args.type == "calculator":
        server = create_calculator_server()
    else:
        server = create_echo_server()

    asyncio.run(server.run_stdio())
