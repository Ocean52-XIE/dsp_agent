# -*- coding: utf-8 -*-
"""测试 MCP Tool Adapter 模块"""
import sys
from pathlib import Path

# 确保可以导入 agent 模块
_project_root = Path(__file__).resolve().parents[3]
_src_path = _project_root / "src"
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agent.mcp.tool_adapter import (
    MCPToolAdapter,
    _schema_to_pydantic_field,
    _create_args_model,
    create_mcp_tool_adapters,
    get_all_openai_schemas,
)
from agent.mcp.client import MCPToolInfo
from agent.mcp.client import MCPClient
from agent.mcp.config_loader import MCPServerConfig


class TestSchemaToPydanticField:
    """测试 Schema 转换为 Pydantic Field"""

    def test_string_type(self):
        """测试字符串类型"""
        field_type, field_info = _schema_to_pydantic_field(
            "message",
            {"type": "string", "description": "A message"},
            required=True,
        )

        assert field_type == str
        assert field_info.description == "A message"

    def test_integer_type(self):
        """测试整数类型"""
        field_type, field_info = _schema_to_pydantic_field(
            "count",
            {"type": "integer", "description": "A count"},
            required=True,
        )

        assert field_type == int
        assert field_info.description == "A count"

    def test_boolean_type(self):
        """测试布尔类型"""
        field_type, field_info = _schema_to_pydantic_field(
            "enabled",
            {"type": "boolean"},
            required=False,
        )

        assert field_type == bool
        assert field_info.default is None

    def test_array_type(self):
        """测试数组类型"""
        field_type, field_info = _schema_to_pydantic_field(
            "items",
            {"type": "array"},
            required=True,
        )

        assert field_type == list

    def test_object_type(self):
        """测试对象类型"""
        field_type, field_info = _schema_to_pydantic_field(
            "config",
            {"type": "object"},
            required=True,
        )

        assert field_type == dict

    def test_nested_object(self):
        """测试嵌套对象"""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "value": {"type": "integer"},
            },
            "required": ["name"],
        }

        field_type, field_info = _schema_to_pydantic_field(
            "config",
            schema,
            required=True,
        )

        # 应该创建一个嵌套模型
        assert hasattr(field_type, "model_fields")
        assert "name" in field_type.model_fields
        assert "value" in field_type.model_fields


class TestCreateArgsModel:
    """测试创建参数模型"""

    def test_no_parameters(self):
        """测试无参数工具"""
        tool_info = MCPToolInfo(
            name="ping",
            description="Ping tool",
            input_schema={},
            server_name="test",
        )

        model = _create_args_model(tool_info)

        assert model.__name__ == "pingArgs"
        # 无参数模型应该只有默认字段
        instance = model()
        assert instance is not None

    def test_single_parameter(self):
        """测试单参数工具"""
        tool_info = MCPToolInfo(
            name="echo",
            description="Echo tool",
            input_schema={
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Message to echo"}
                },
                "required": ["message"],
            },
            server_name="test",
        )

        model = _create_args_model(tool_info)

        assert model.__name__ == "echoArgs"
        assert "message" in model.model_fields

        # 验证必填参数
        instance = model(message="hello")
        assert instance.message == "hello"

    def test_multiple_parameters(self):
        """测试多参数工具"""
        tool_info = MCPToolInfo(
            name="add",
            description="Add two numbers",
            input_schema={
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "First number"},
                    "b": {"type": "integer", "description": "Second number"},
                },
                "required": ["a", "b"],
            },
            server_name="test",
        )

        model = _create_args_model(tool_info)

        assert "a" in model.model_fields
        assert "b" in model.model_fields

        instance = model(a=1, b=2)
        assert instance.a == 1
        assert instance.b == 2

    def test_optional_parameter(self):
        """测试可选参数"""
        tool_info = MCPToolInfo(
            name="search",
            description="Search tool",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["query"],  # limit 是可选的
            },
            server_name="test",
        )

        model = _create_args_model(tool_info)

        # 只提供必填参数
        instance = model(query="test")
        assert instance.query == "test"
        assert instance.limit is None


class TestMCPToolAdapter:
    """测试 MCPToolAdapter"""

    @pytest.fixture
    def mock_client(self):
        """创建 mock 客户端"""
        client = MagicMock(spec=MCPClient)
        client.call_tool = AsyncMock(return_value=MagicMock(
            success=True,
            content="Echo: hello",
        ))
        return client

    @pytest.fixture
    def echo_tool_info(self):
        """创建 echo 工具信息"""
        return MCPToolInfo(
            name="echo",
            description="Echo the input message",
            input_schema={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Message to echo",
                    }
                },
                "required": ["message"],
            },
            server_name="test_server",
        )

    def test_init(self, mock_client, echo_tool_info):
        """测试初始化"""
        adapter = MCPToolAdapter(mock_client, echo_tool_info)

        assert adapter.name == "echo"
        assert adapter.description == "Echo the input message"
        assert adapter.server_name == "test_server"
        assert adapter.client == mock_client
        assert adapter.tool_info == echo_tool_info

    def test_get_openai_schema(self, mock_client, echo_tool_info):
        """测试获取 OpenAI Schema"""
        adapter = MCPToolAdapter(mock_client, echo_tool_info)

        schema = adapter.get_openai_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "echo"
        assert schema["function"]["description"] == "Echo the input message"
        assert "properties" in schema["function"]["parameters"]
        assert "message" in schema["function"]["parameters"]["properties"]

    def test_get_openai_schema_empty(self, mock_client):
        """测试获取空 Schema"""
        tool_info = MCPToolInfo(
            name="ping",
            description="Ping tool",
            input_schema=None,
            server_name="test",
        )
        adapter = MCPToolAdapter(mock_client, tool_info)

        schema = adapter.get_openai_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "ping"
        assert schema["function"]["parameters"]["type"] == "object"
        assert schema["function"]["parameters"]["properties"] == {}

    def test_run_sync_success(self, mock_client, echo_tool_info):
        """测试同步调用成功（使用线程池包装异步调用）"""
        # 配置 mock 返回值
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.content = "Echo: hello sync"
        mock_client.call_tool = AsyncMock(return_value=mock_result)

        adapter = MCPToolAdapter(mock_client, echo_tool_info)

        # 同步调用应该成功（内部使用线程池包装异步调用）
        result = adapter._run(message="hello sync")

        assert result == "Echo: hello sync"
        mock_client.call_tool.assert_called_once_with("echo", {"message": "hello sync"})

    def test_run_sync_failure(self, mock_client, echo_tool_info):
        """测试同步调用失败"""
        from langchain_core.tools import ToolException

        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error = "Sync tool execution failed"
        mock_client.call_tool = AsyncMock(return_value=mock_result)

        adapter = MCPToolAdapter(mock_client, echo_tool_info)

        # 同步调用失败时应该抛出 ToolException
        with pytest.raises(ToolException) as exc_info:
            adapter._run(message="hello")

        assert "Tool call failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_arun_success(self, mock_client, echo_tool_info):
        """测试异步调用成功"""
        # 配置 mock 返回值
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.content = "Echo: hello"
        mock_client.call_tool = AsyncMock(return_value=mock_result)

        adapter = MCPToolAdapter(mock_client, echo_tool_info)

        result = await adapter._arun(message="hello")

        assert result == "Echo: hello"
        mock_client.call_tool.assert_called_once_with("echo", {"message": "hello"})

    @pytest.mark.asyncio
    async def test_arun_failure(self, mock_client, echo_tool_info):
        """测试异步调用失败"""
        from langchain_core.tools import ToolException

        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error = "Tool execution failed"
        mock_client.call_tool = AsyncMock(return_value=mock_result)

        adapter = MCPToolAdapter(mock_client, echo_tool_info)

        with pytest.raises(ToolException) as exc_info:
            await adapter._arun(message="hello")

        assert "Tool call failed" in str(exc_info.value)

    def test_server_name_property(self, mock_client, echo_tool_info):
        """测试 server_name 属性"""
        adapter = MCPToolAdapter(mock_client, echo_tool_info)

        assert adapter.server_name == "test_server"

    def test_to_dict(self, mock_client, echo_tool_info):
        """测试转换为字典"""
        adapter = MCPToolAdapter(mock_client, echo_tool_info)

        d = adapter.to_dict()

        assert d["name"] == "echo"
        assert d["description"] == "Echo the input message"
        assert d["server_name"] == "test_server"
        assert "input_schema" in d

    def test_repr(self, mock_client, echo_tool_info):
        """测试字符串表示"""
        adapter = MCPToolAdapter(mock_client, echo_tool_info)

        repr_str = repr(adapter)

        assert "MCPToolAdapter" in repr_str
        assert "echo" in repr_str
        assert "test_server" in repr_str


class TestCreateMCPToolAdapters:
    """测试创建适配器工厂函数"""

    def test_create_adapters(self):
        """测试创建适配器列表"""
        client = MagicMock(spec=MCPClient)

        tool_info = MCPToolInfo(
            name="echo",
            description="Echo tool",
            input_schema={},
            server_name="test",
        )

        # 配置 mock 返回值
        adapter = MagicMock(spec=MCPToolAdapter)
        adapter.name = "echo"

        with patch.object(client, "get_tool_adapters", return_value=[adapter]):
            adapters = create_mcp_tool_adapters(client)

            assert len(adapters) == 1
            assert adapters[0].name == "echo"


class TestGetAllOpenAISchemas:
    """测试获取所有 OpenAI Schema"""

    def test_get_schemas(self):
        """测试获取所有 Schema"""
        # 创建 mock adapters
        adapter1 = MagicMock(spec=MCPToolAdapter)
        adapter1.get_openai_schema = MagicMock(return_value={
            "type": "function",
            "function": {"name": "echo"},
        })

        adapter2 = MagicMock(spec=MCPToolAdapter)
        adapter2.get_openai_schema = MagicMock(return_value={
            "type": "function",
            "function": {"name": "add"},
        })

        schemas = get_all_openai_schemas([adapter1, adapter2])

        assert len(schemas) == 2
        assert schemas[0]["function"]["name"] == "echo"
        assert schemas[1]["function"]["name"] == "add"

    def test_get_schemas_empty(self):
        """测试空适配器列表"""
        schemas = get_all_openai_schemas([])

        assert len(schemas) == 0


class TestMCPToolAdapterIntegration:
    """MCPToolAdapter 集成测试"""

    @pytest.fixture
    def full_client(self):
        """创建完整配置的客户端"""
        config = MCPServerConfig(
            name="test_server",
            transport="stdio",
            command="python",
        )
        client = MCPClient({"test_server": config})
        return client

    def test_full_adapter_creation(self, full_client):
        """测试完整适配器创建"""
        tool_info = MCPToolInfo(
            name="search",
            description="Search for code",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                    "file_pattern": {
                        "type": "string",
                        "description": "File pattern to search",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results",
                    },
                },
                "required": ["query"],
            },
            server_name="test_server",
        )

        adapter = MCPToolAdapter(full_client, tool_info)

        # 验证基本信息
        assert adapter.name == "search"
        assert adapter.description == "Search for code"

        # 验证 OpenAI Schema
        schema = adapter.get_openai_schema()
        assert schema["type"] == "function"
        assert len(schema["function"]["parameters"]["properties"]) == 3

        # 验证必填参数
        assert "query" in schema["function"]["parameters"]["required"]
