# -*- coding: utf-8 -*-
"""测试 MCP Session 模块"""
import sys
from pathlib import Path

# 确保可以导入 agent 模块
_project_root = Path(__file__).resolve().parents[3]
_src_path = _project_root / "src"
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agent.mcp.config_loader import MCPServerConfig
from agent.mcp.session import MCPSession, MCPToolInfo


class TestMCPToolInfo:
    """测试 MCPToolInfo 数据结构"""

    def test_create_tool_info(self):
        """测试创建工具信息"""
        tool_info = MCPToolInfo(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object"},
            server_name="test_server",
        )

        assert tool_info.name == "test_tool"
        assert tool_info.description == "A test tool"
        assert tool_info.input_schema == {"type": "object"}
        assert tool_info.server_name == "test_server"

    def test_to_dict(self):
        """测试转换为字典"""
        tool_info = MCPToolInfo(
            name="echo",
            description="Echo tool",
            input_schema={
                "type": "object",
                "properties": {"message": {"type": "string"}},
            },
            server_name="echo_server",
        )
        result = tool_info.to_dict()

        assert result["name"] == "echo"
        assert result["description"] == "Echo tool"
        assert "properties" in result["input_schema"]
        assert result["server_name"] == "echo_server"


class TestMCPSession:
    """测试 MCPSession"""

    def test_init_stdio_config(self):
        """测试初始化 stdio 配置"""
        config = MCPServerConfig(
            name="test",
            transport="stdio",
            command="python",
            args=["server.py"],
        )
        session = MCPSession(config)

        assert session.config == config
        assert session.server_name == "test"
        assert not session.is_connected

    def test_init_sse_config(self):
        """测试初始化 sse 配置"""
        config = MCPServerConfig(
            name="test",
            transport="sse",
            url="http://localhost:8080/sse",
        )
        session = MCPSession(config)

        assert session.config == config
        assert session.server_name == "test"

    @pytest.mark.asyncio
    async def test_connect_stdio(self):
        """测试 stdio 连接"""
        config = MCPServerConfig(
            name="test",
            transport="stdio",
            command="python",
            args=["-c", "print('mock')"],
        )
        session = MCPSession(config)

        # Mock stdio_client 和 ClientSession
        with patch("agent.mcp.session.stdio_client") as mock_stdio, \
             patch("agent.mcp.session.ClientSession") as mock_client_session:

            # 设置 mock
            mock_read_stream = AsyncMock()
            mock_write_stream = AsyncMock()
            mock_stdio.return_value.__aenter__ = AsyncMock(
                return_value=(mock_read_stream, mock_write_stream)
            )

            mock_session_instance = AsyncMock()
            mock_session_instance.initialize = AsyncMock()
            mock_client_session.return_value = mock_session_instance

            await session.connect()

            assert session.is_connected
            mock_stdio.assert_called_once()
            mock_session_instance.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_sse(self):
        """测试 sse 连接"""
        config = MCPServerConfig(
            name="test",
            transport="sse",
            url="http://localhost:8080/sse",
        )
        session = MCPSession(config)

        # sse_client 在 _connect_http 方法内部导入，所以需要 patch mcp.client.sse
        with patch("mcp.client.sse.sse_client") as mock_sse, \
             patch("agent.mcp.session.ClientSession") as mock_client_session:

            mock_read_stream = AsyncMock()
            mock_write_stream = AsyncMock()
            mock_sse.return_value.__aenter__ = AsyncMock(
                return_value=(mock_read_stream, mock_write_stream)
            )

            mock_session_instance = AsyncMock()
            mock_session_instance.initialize = AsyncMock()
            mock_client_session.return_value = mock_session_instance

            await session.connect()

            assert session.is_connected
            mock_sse.assert_called_once_with("http://localhost:8080/sse")

    @pytest.mark.asyncio
    async def test_connect_unsupported_transport(self):
        """测试不支持的传输模式"""
        config = MCPServerConfig(
            name="test",
            transport="invalid",
        )
        session = MCPSession(config)

        with pytest.raises(ConnectionError) as exc_info:
            await session.connect()

        assert "Unsupported transport" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_disconnect(self):
        """测试断开连接"""
        config = MCPServerConfig(
            name="test",
            transport="stdio",
            command="python",
        )
        session = MCPSession(config)

        # 设置为已连接状态
        session._connected = True
        session._session = AsyncMock()
        session._session.__aexit__ = AsyncMock()

        await session.disconnect()

        assert not session.is_connected
        assert session._session is None

    @pytest.mark.asyncio
    async def test_list_tools(self):
        """测试列出工具"""
        config = MCPServerConfig(
            name="test",
            transport="stdio",
            command="python",
        )
        session = MCPSession(config)
        session._connected = True

        # Mock session.list_tools 返回值
        mock_tool = MagicMock()
        mock_tool.name = "echo"
        mock_tool.description = "Echo tool"
        mock_tool.inputSchema = {"type": "object"}

        mock_result = MagicMock()
        mock_result.tools = [mock_tool]

        session._session = AsyncMock()
        session._session.list_tools = AsyncMock(return_value=mock_result)

        tools = await session.list_tools()

        assert len(tools) == 1
        assert tools[0].name == "echo"
        assert tools[0].server_name == "test"

    @pytest.mark.asyncio
    async def test_list_tools_not_connected(self):
        """测试未连接时列出工具"""
        config = MCPServerConfig(name="test", transport="stdio", command="python")
        session = MCPSession(config)

        with pytest.raises(RuntimeError) as exc_info:
            await session.list_tools()

        assert "not connected" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_call_tool(self):
        """测试调用工具"""
        config = MCPServerConfig(
            name="test",
            transport="stdio",
            command="python",
            timeout=60,
        )
        session = MCPSession(config)
        session._connected = True

        # 预注册工具
        session._tools["echo"] = MCPToolInfo(
            name="echo",
            description="Echo tool",
            input_schema={},
            server_name="test",
        )

        # Mock call_tool 返回值
        mock_result = MagicMock()
        mock_result.content = "Echo: hello"

        session._session = AsyncMock()
        session._session.call_tool = AsyncMock(return_value=mock_result)

        result = await session.call_tool("echo", {"message": "hello"})

        assert result["content"] == "Echo: hello"

    @pytest.mark.asyncio
    async def test_call_tool_not_found(self):
        """测试调用不存在的工具"""
        config = MCPServerConfig(
            name="test",
            transport="stdio",
            command="python",
        )
        session = MCPSession(config)
        session._connected = True

        # Mock list_tools 返回空列表
        mock_result = MagicMock()
        mock_result.tools = []

        session._session = AsyncMock()
        session._session.list_tools = AsyncMock(return_value=mock_result)

        with pytest.raises(ValueError) as exc_info:
            await session.call_tool("nonexistent", {})

        assert "Tool not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_call_tool_timeout(self):
        """测试工具调用超时"""
        config = MCPServerConfig(
            name="test",
            transport="stdio",
            command="python",
            timeout=0.1,  # 100ms
        )
        session = MCPSession(config)
        session._connected = True

        session._tools["slow_tool"] = MCPToolInfo(
            name="slow_tool",
            description="Slow tool",
            input_schema={},
            server_name="test",
        )

        # Mock 慢调用
        async def slow_call(*args, **kwargs):
            import asyncio
            await asyncio.sleep(1)
            return MagicMock(content="done")

        session._session = AsyncMock()
        session._session.call_tool = slow_call

        with pytest.raises(TimeoutError) as exc_info:
            await session.call_tool("slow_tool", {})

        assert "timeout" in str(exc_info.value).lower()

    def test_get_tool_info(self):
        """测试获取工具信息"""
        config = MCPServerConfig(name="test", transport="stdio", command="python")
        session = MCPSession(config)

        tool_info = MCPToolInfo(
            name="echo",
            description="Echo tool",
            input_schema={},
            server_name="test",
        )
        session._tools["echo"] = tool_info

        result = session.get_tool_info("echo")
        assert result == tool_info

        result = session.get_tool_info("nonexistent")
        assert result is None

    def test_get_all_tools(self):
        """测试获取所有工具"""
        config = MCPServerConfig(name="test", transport="stdio", command="python")
        session = MCPSession(config)

        tool1 = MCPToolInfo(
            name="echo",
            description="Echo tool",
            input_schema={},
            server_name="test",
        )
        tool2 = MCPToolInfo(
            name="add",
            description="Add tool",
            input_schema={},
            server_name="test",
        )
        session._tools["echo"] = tool1
        session._tools["add"] = tool2

        tools = session.get_all_tools()
        assert len(tools) == 2
        assert tools[0] in [tool1, tool2]
        assert tools[1] in [tool1, tool2]

    def test_get_tool_names(self):
        """测试获取工具名称列表"""
        config = MCPServerConfig(name="test", transport="stdio", command="python")
        session = MCPSession(config)

        session._tools["echo"] = MCPToolInfo(
            name="echo", description="", input_schema={}, server_name="test"
        )
        session._tools["add"] = MCPToolInfo(
            name="add", description="", input_schema={}, server_name="test"
        )

        names = session.get_tool_names()
        assert set(names) == {"echo", "add"}
