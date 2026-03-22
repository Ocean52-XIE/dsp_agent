# -*- coding: utf-8 -*-
"""测试 LLMClient LLM 客户端"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agent.llm.client import LLMClient, LLMResponse, ToolCall
from agent.llm.config import LLMConfig


# 全局 Mock ChatOpenAI，避免所有测试都需要真实 API key
@pytest.fixture(autouse=True)
def mock_chat_openai():
    """自动 Mock ChatOpenAI 类"""
    with patch("agent.llm.client.ChatOpenAI") as mock:
        mock_llm = MagicMock()
        mock.return_value = mock_llm
        yield mock


class TestToolCall:
    """测试 ToolCall"""

    def test_create_tool_call(self):
        """测试创建工具调用"""
        tc = ToolCall(
            id="call-123",
            name="test_tool",
            args={"query": "test"},
        )

        assert tc.id == "call-123"
        assert tc.name == "test_tool"
        assert tc.args == {"query": "test"}

    def test_tool_call_to_dict(self):
        """测试工具调用转字典"""
        tc = ToolCall(
            id="call-123",
            name="test_tool",
            args={"query": "test"},
        )

        result = tc.to_dict()

        assert result["id"] == "call-123"
        assert result["name"] == "test_tool"
        assert result["args"] == {"query": "test"}


class TestLLMResponse:
    """测试 LLMResponse"""

    def test_create_response(self):
        """测试创建响应"""
        response = LLMResponse(
            content="测试响应",
            tool_calls=[],
            latency_ms=100,
        )

        assert response.content == "测试响应"
        assert response.has_tool_calls is False
        assert response.latency_ms == 100

    def test_response_with_tool_calls(self):
        """测试带工具调用的响应"""
        tc = ToolCall(
            id="call-123",
            name="test_tool",
            args={},
        )
        response = LLMResponse(
            content="",
            tool_calls=[tc],
            latency_ms=150,
        )

        assert response.has_tool_calls is True
        assert len(response.tool_calls) == 1

    def test_response_to_dict(self):
        """测试响应转字典"""
        response = LLMResponse(
            content="测试",
            tool_calls=[],
            latency_ms=100,
        )

        result = response.to_dict()

        assert result["content"] == "测试"
        assert result["tool_calls"] == []
        assert result["latency_ms"] == 100

    def test_strip_thinking_content(self):
        """测试去除 thinking 内容"""
        from agent.llm.client import strip_thinking_content

        text = "这是思考过程<thinking>这部分应该被移除</thinking>这是正文"
        result = strip_thinking_content(text)

        assert "这部分应该被移除" not in result
        assert "这是思考过程" in result
        assert "这是正文" in result


class TestLLMClient:
    """测试 LLMClient"""

    def test_create_client(self):
        """测试创建客户端"""
        config = LLMConfig(
            model="gpt-4o",
            temperature=0.7,
        )
        client = LLMClient(config)

        assert client.config.model == "gpt-4o"
        assert client.config.temperature == 0.7

    def test_create_client_from_env(self):
        """测试从环境变量创建客户端"""
        # 设置环境变量进行测试
        import os
        old_model = os.environ.get("AGENT_LLM_MODEL")
        old_key = os.environ.get("AGENT_LLM_API_KEY")
        os.environ["AGENT_LLM_MODEL"] = "gpt-4o"
        os.environ["AGENT_LLM_API_KEY"] = "test-api-key"

        try:
            client = LLMClient.from_env("AGENT_LLM")

            assert client is not None
            assert client.config.model == "gpt-4o"
        finally:
            # 恢复环境变量
            if old_model is None:
                os.environ.pop("AGENT_LLM_MODEL", None)
            else:
                os.environ["AGENT_LLM_MODEL"] = old_model
            if old_key is None:
                os.environ.pop("AGENT_LLM_API_KEY", None)
            else:
                os.environ["AGENT_LLM_API_KEY"] = old_key

    @pytest.mark.asyncio
    async def test_ainvoke(self, mock_llm_client):
        """测试异步调用"""
        # 这个测试验证接口设计，实际调用需要 mock
        response = await mock_llm_client.ainvoke("测试消息")

        assert response is not None

    @pytest.mark.asyncio
    async def test_ainvoke_with_tools(self, mock_llm_client):
        """测试带工具的异步调用"""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "description": "测试工具",
                    "parameters": {},
                },
            }
        ]

        response = await mock_llm_client.ainvoke_with_tools(
            messages=[{"role": "user", "content": "测试"}],
            tools=tools,
        )

        assert response is not None

    def test_build_messages(self):
        """测试构建消息列表"""
        config = LLMConfig(model="gpt-4o")
        client = LLMClient(config)

        messages = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！"},
            {"role": "user", "content": "再见"},
        ]

        lc_messages = client._build_messages(messages)

        assert len(lc_messages) == 3

    def test_build_messages_with_system(self):
        """测试带系统提示的消息构建"""
        config = LLMConfig(model="gpt-4o")
        client = LLMClient(config)

        messages = [{"role": "user", "content": "你好"}]
        system_prompt = "你是一个助手"

        lc_messages = client._build_messages(messages, system_prompt)

        assert len(lc_messages) == 2
        # 第一个应该是系统消息
        assert lc_messages[0].content == "你是一个助手"

    def test_parse_tool_calls(self):
        """测试解析工具调用"""
        config = LLMConfig(model="gpt-4o")
        client = LLMClient(config)

        # 模拟 LLM 响应
        mock_response = MagicMock()
        mock_response.tool_calls = [
            {"id": "call-1", "name": "tool1", "args": {"x": 1}},
            {"id": "call-2", "name": "tool2", "args": {"y": 2}},
        ]

        tool_calls = client._parse_tool_calls(mock_response)

        assert len(tool_calls) == 2
        assert tool_calls[0].name == "tool1"
        assert tool_calls[1].name == "tool2"

    def test_parse_empty_tool_calls(self):
        """测试解析空工具调用"""
        config = LLMConfig(model="gpt-4o")
        client = LLMClient(config)

        mock_response = MagicMock()
        mock_response.tool_calls = []

        tool_calls = client._parse_tool_calls(mock_response)

        assert len(tool_calls) == 0

    def test_parse_tool_calls_no_attribute(self):
        """测试响应无 tool_calls 属性"""
        config = LLMConfig(model="gpt-4o")
        client = LLMClient(config)

        mock_response = MagicMock(spec=[])  # 没有 tool_calls 属性

        tool_calls = client._parse_tool_calls(mock_response)

        assert len(tool_calls) == 0


class TestLLMClientIntegration:
    """测试 LLMClient 集成场景"""

    @pytest.mark.asyncio
    async def test_full_conversation_flow(self):
        """测试完整对话流程"""
        # 这个测试验证接口设计
        config = LLMConfig(
            model="gpt-4o",
            temperature=0.7,
        )

        # 使用 mock 避免实际调用
        with patch.object(LLMClient, '_create_llm') as mock_create:
            mock_llm = MagicMock()
            mock_llm.ainvoke = AsyncMock(return_value=MagicMock(
                content="这是回复",
                tool_calls=[],
            ))
            mock_create.return_value = mock_llm

            client = LLMClient(config)
            client._llm = mock_llm

            response = await client.ainvoke("测试消息")

            assert response.content == "这是回复"

    def test_client_with_custom_base_url(self):
        """测试自定义 Base URL"""
        config = LLMConfig(
            model="gpt-4o",
            base_url="https://custom.api.com/v1",
        )
        client = LLMClient(config)

        # 验证配置正确
        assert client.config.base_url == "https://custom.api.com/v1"

    def test_client_with_extra_params(self):
        """测试额外参数"""
        config = LLMConfig(
            model="gpt-4o",
            extra_params={"top_p": 0.9, "frequency_penalty": 0.5},
        )
        client = LLMClient(config)

        assert client.config.extra_params["top_p"] == 0.9
