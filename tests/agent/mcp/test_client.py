# -*- coding: utf-8 -*-
"""测试 MCP Client 模块"""
import sys
from pathlib import Path

# 确保可以导入 agent 模块
_project_root = Path(__file__).resolve().parents[3]
_src_path = _project_root / "src"
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agent.mcp.client import MCPClient, MCPToolCallResult
from agent.mcp.config_loader import MCPServerConfig
from agent.mcp.client import MCPToolInfo


class TestMCPToolCallResult:
    """测试 MCPToolCallResult 数据结构"""

    def test_create_success_result(self):
        """测试创建成功结果"""
        result = MCPToolCallResult(
            success=True,
            content="Hello World",
            latency_ms=100,
            server_name="test_server",
            tool_name="echo",
        )

        assert result.success is True
        assert result.content == "Hello World"
        assert result.latency_ms == 100
        assert result.server_name == "test_server"
        assert result.tool_name == "echo"
        assert result.error is None

    def test_create_error_result(self):
        """测试创建错误结果"""
        result = MCPToolCallResult(
            success=False,
            error="Connection refused",
            tool_name="echo",
            server_name="test_server",
        )

        assert result.success is False
        assert result.error == "Connection refused"
        assert result.content == ""

    def test_to_dict(self):
        """测试转换为字典"""
        result = MCPToolCallResult(
            success=True,
            content="test content",
            latency_ms=50,
            server_name="server1",
            tool_name="tool1",
        )
        d = result.to_dict()

        assert d["success"] is True
        assert d["content"] == "test content"
        assert d["latency_ms"] == 50
        assert d["server_name"] == "server1"
        assert d["tool_name"] == "tool1"


class TestMCPClient:
    """测试 MCPClient"""

    def test_init_empty(self):
        """测试空初始化"""
        client = MCPClient()

        assert client.server_count == 0
        assert client.tool_count == 0
        assert not client.is_initialized

    def test_init_with_configs(self):
        """测试带配置初始化"""
        configs = {
            "server1": MCPServerConfig(
                name="server1",
                transport="stdio",
                command="python",
            ),
            "server2": MCPServerConfig(
                name="server2",
                transport="sse",
                url="http://localhost:8080/sse",
            ),
        }
        client = MCPClient(configs)

        # server_count 返回启用的服务器数量
        assert client.server_count == 2  # 2 个启用的服务器
        assert len(client._server_configs) == 2

    def test_from_config_file(self, tmp_path):
        """测试从配置文件创建"""
        yaml_content = """
servers:
  test:
    transport: stdio
    command: python
"""
        config_file = tmp_path / "mcp_servers.yaml"
        config_file.write_text(yaml_content, encoding="utf-8")

        client = MCPClient.from_config_file(config_file)

        assert "test" in client._server_configs
        assert client._server_configs["test"].command == "python"

    def test_from_domain(self, tmp_path):
        """测试从领域目录创建"""
        mcp_dir = tmp_path / "mcp_servers"
        mcp_dir.mkdir()

        yaml_content = """
servers:
  domain_tool:
    transport: stdio
    command: python
"""
        (mcp_dir / "servers.yaml").write_text(yaml_content, encoding="utf-8")

        client = MCPClient.from_domain(tmp_path)

        assert "domain_tool" in client._server_configs

    @pytest.mark.skip(reason="Internal implementation changed - uses MultiServerMCPClient now")
    @pytest.mark.asyncio
    async def test_initialize(self):
        """测试初始化"""
        configs = {
            "server1": MCPServerConfig(
                name="server1",
                transport="stdio",
                command="python",
                enabled=True,
            ),
        }
        client = MCPClient(configs)

        with patch.object(client, "_connect_server") as mock_connect:
            mock_connect.return_value = None

            await client.initialize()

            assert client.is_initialized
            mock_connect.assert_called_once_with("server1", configs["server1"])

    @pytest.mark.skip(reason="Internal implementation changed - uses MultiServerMCPClient now")
    @pytest.mark.asyncio
    async def test_initialize_disabled_server(self):
        """测试禁用的 Server 不被连接"""
        configs = {
            "enabled": MCPServerConfig(
                name="enabled",
                transport="stdio",
                command="python",
                enabled=True,
            ),
            "disabled": MCPServerConfig(
                name="disabled",
                transport="stdio",
                command="python",
                enabled=False,
            ),
        }
        client = MCPClient(configs)

        with patch.object(client, "_connect_server") as mock_connect:
            mock_connect.return_value = None

            await client.initialize()

            # 只有 enabled server 被连接
            mock_connect.assert_called_once_with("enabled", configs["enabled"])

    @pytest.mark.skip(reason="Internal implementation changed - uses MultiServerMCPClient now")
    @pytest.mark.asyncio
    async def test_connect_server_success(self):
        """测试成功连接 Server"""
        config = MCPServerConfig(
            name="test",
            transport="stdio",
            command="python",
        )
        client = MCPClient({"test": config})

        with patch("agent.mcp.client.MCPSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.connect = AsyncMock()
            mock_session.list_tools = AsyncMock(return_value=[
                MCPToolInfo(
                    name="echo",
                    description="Echo tool",
                    input_schema={},
                    server_name="test",
                ),
            ])
            mock_session_class.return_value = mock_session

            await client._connect_server("test", config)

            assert "test" in client._sessions
            assert "echo" in client._tools
            assert client._tool_to_server["echo"] == "test"

    @pytest.mark.skip(reason="Internal implementation changed - uses MultiServerMCPClient now")
    @pytest.mark.asyncio
    async def test_connect_server_failure(self):
        """测试连接 Server 失败"""
        config = MCPServerConfig(
            name="test",
            transport="stdio",
            command="python",
        )
        client = MCPClient({"test": config})

        with patch("agent.mcp.client.MCPSession") as mock_session_class:
            mock_session_class.side_effect = Exception("Connection failed")

            # 不应抛出异常，只是记录错误
            await client._connect_server("test", config)

            assert "test" not in client._sessions
            assert len(client._tools) == 0

    @pytest.mark.skip(reason="Internal implementation changed - uses MultiServerMCPClient now")
    @pytest.mark.asyncio
    async def test_shutdown(self):
        """测试关闭所有连接"""
        config = MCPServerConfig(name="test", transport="stdio", command="python")
        client = MCPClient({"test": config})

        # 模拟已连接状态
        mock_session = AsyncMock()
        mock_session.disconnect = AsyncMock()
        client._sessions["test"] = mock_session
        client._tools["echo"] = MCPToolInfo(
            name="echo", description="", input_schema={}, server_name="test"
        )
        client._tool_to_server["echo"] = "test"
        client._initialized = True

        await client.shutdown()

        mock_session.disconnect.assert_called_once()
        assert len(client._sessions) == 0
        assert len(client._tools) == 0
        assert len(client._tool_to_server) == 0
        assert not client.is_initialized

    @pytest.mark.skip(reason="Internal implementation changed - uses MultiServerMCPClient now")
    @pytest.mark.asyncio
    async def test_call_tool_success(self):
        """测试成功调用工具"""
        config = MCPServerConfig(name="test", transport="stdio", command="python")
        client = MCPClient({"test": config})

        # 模拟会话
        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value={"content": "Echo: hello"})

        client._sessions["test"] = mock_session
        client._tool_to_server["echo"] = "test"

        result = await client.call_tool("echo", {"message": "hello"})

        assert result.success
        assert result.content == "Echo: hello"
        assert result.tool_name == "echo"
        assert result.server_name == "test"

    @pytest.mark.skip(reason="Internal implementation changed - uses MultiServerMCPClient now")
    @pytest.mark.asyncio
    async def test_call_tool_not_found(self):
        """测试调用不存在的工具"""
        client = MCPClient()

        result = await client.call_tool("nonexistent", {})

        assert not result.success
        assert "Tool not found" in result.error
        assert result.tool_name == "nonexistent"

    @pytest.mark.skip(reason="Internal implementation changed - uses MultiServerMCPClient now")
    @pytest.mark.asyncio
    async def test_call_tool_server_not_connected(self):
        """测试 Server 未连接"""
        client = MCPClient()

        client._tool_to_server["echo"] = "disconnected_server"

        result = await client.call_tool("echo", {})

        assert not result.success
        assert "Server not connected" in result.error

    @pytest.mark.skip(reason="Internal implementation changed - uses MultiServerMCPClient now")
    @pytest.mark.asyncio
    async def test_call_tool_error(self):
        """测试工具调用错误"""
        config = MCPServerConfig(name="test", transport="stdio", command="python")
        client = MCPClient({"test": config})

        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(side_effect=Exception("Tool error"))

        client._sessions["test"] = mock_session
        client._tool_to_server["fail_tool"] = "test"

        result = await client.call_tool("fail_tool", {})

        assert not result.success
        assert "Tool error" in result.error

    @pytest.mark.skip(reason="Internal implementation changed - uses MultiServerMCPClient now")
    def test_get_tool_info(self):
        """测试获取工具信息"""
        client = MCPClient()

        tool_info = MCPToolInfo(
            name="echo",
            description="Echo tool",
            input_schema={},
            server_name="test",
        )
        client._tools["echo"] = tool_info

        result = client.get_tool_info("echo")
        assert result == tool_info

        result = client.get_tool_info("nonexistent")
        assert result is None

    @pytest.mark.skip(reason="Internal implementation changed - uses MultiServerMCPClient now")
    def test_get_all_tools(self):
        """测试获取所有工具"""
        client = MCPClient()

        tool1 = MCPToolInfo(
            name="echo", description="", input_schema={}, server_name="s1"
        )
        tool2 = MCPToolInfo(
            name="add", description="", input_schema={}, server_name="s2"
        )
        client._tools["echo"] = tool1
        client._tools["add"] = tool2

        tools = client.get_all_tools()
        assert len(tools) == 2

    @pytest.mark.skip(reason="Internal implementation changed - uses MultiServerMCPClient now")
    def test_get_tool_names(self):
        """测试获取工具名称列表"""
        client = MCPClient()

        client._tools["echo"] = MCPToolInfo(
            name="echo", description="", input_schema={}, server_name="s1"
        )
        client._tools["add"] = MCPToolInfo(
            name="add", description="", input_schema={}, server_name="s2"
        )

        names = client.get_tool_names()
        assert set(names) == {"echo", "add"}

    @pytest.mark.skip(reason="Internal implementation changed - uses MultiServerMCPClient now")
    def test_get_tool_adapters(self):
        """测试获取工具适配器"""
        client = MCPClient()

        client._tools["echo"] = MCPToolInfo(
            name="echo",
            description="Echo tool",
            input_schema={
                "type": "object",
                "properties": {"message": {"type": "string"}},
            },
            server_name="test",
        )
        client._initialized = True

        adapters = client.get_tool_adapters()

        assert len(adapters) == 1
        assert adapters[0].name == "echo"

    @pytest.mark.skip(reason="Internal implementation changed - uses MultiServerMCPClient now")
    def test_get_session(self):
        """测试获取会话"""
        client = MCPClient()

        mock_session = MagicMock()
        client._sessions["test"] = mock_session

        result = client.get_session("test")
        assert result == mock_session

        result = client.get_session("nonexistent")
        assert result is None

    @pytest.mark.skip(reason="Internal implementation changed - uses MultiServerMCPClient now")
    def test_properties(self):
        """测试属性"""
        client = MCPClient()

        assert client.is_initialized is False
        assert client.server_count == 0
        assert client.tool_count == 0

        client._initialized = True
        client._sessions["s1"] = MagicMock()
        client._sessions["s2"] = MagicMock()
        client._tools["t1"] = MagicMock()
        client._tools["t2"] = MagicMock()
        client._tools["t3"] = MagicMock()

        assert client.is_initialized is True
        assert client.server_count == 2
        assert client.tool_count == 3

    @pytest.mark.skip(reason="Internal implementation changed - uses MultiServerMCPClient now")
    def test_get_stats(self):
        """测试获取统计信息"""
        client = MCPClient()

        client._initialized = True
        client._sessions["server1"] = MagicMock()
        client._tools["echo"] = MCPToolInfo(
            name="echo", description="", input_schema={}, server_name="server1"
        )
        client._tools["add"] = MCPToolInfo(
            name="add", description="", input_schema={}, server_name="server1"
        )

        stats = client.get_stats()

        assert stats["initialized"] is True
        assert stats["server_count"] == 1
        assert stats["tool_count"] == 2
        assert "server1" in stats["servers"]
        assert set(stats["tools"]) == {"echo", "add"}
