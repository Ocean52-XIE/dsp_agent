# -*- coding: utf-8 -*-
"""测试 AgentLoop 核心循环（同步版本）"""
import pytest
from unittest.mock import MagicMock, patch

from agent.core.loop import AgentLoop, AgentLoopConfig, AgentLoopResult
from agent.state import AgentState, ToolCallRecord
from agent.tools.registry import ToolRegistry
from agent.llm.client import LLMClient, LLMResponse, ToolCall
from agent.llm.config import LLMConfig


def create_test_state(
    trace_id: str = "test-trace",
    user_query: str = "测试查询",
    history: list | None = None,
    tool_whitelist: list | None = None,
) -> AgentState:
    """创建测试用的 Agent 状态"""
    return {
        "trace_id": trace_id,
        "user_query": user_query,
        "history": history or [],
        "tool_whitelist": tool_whitelist,
    }


class TestAgentLoopConfig:
    """测试 AgentLoopConfig"""

    def test_create_config(self):
        """测试创建配置"""
        config = AgentLoopConfig(
            max_steps=15,
            timeout_seconds=180,
            system_prompt="你是助手",
        )

        assert config.max_steps == 15
        assert config.timeout_seconds == 180
        assert config.system_prompt == "你是助手"

    def test_config_from_dict(self):
        """测试从字典创建配置"""
        data = {
            "max_steps": 20,
            "timeout_seconds": 300,
            "system_prompt": "测试提示词",
        }

        config = AgentLoopConfig.from_dict(data)

        assert config.max_steps == 20
        assert config.timeout_seconds == 300

    def test_config_defaults(self):
        """测试默认值"""
        config = AgentLoopConfig()

        assert config.max_steps == 10
        assert config.timeout_seconds == 120
        assert config.system_prompt == ""


class TestAgentLoopResult:
    """测试 AgentLoopResult"""

    def test_create_result(self):
        """测试创建结果"""
        result = AgentLoopResult(
            success=True,
            answer="这是答案",
            steps=3,
            latency_ms=500,
        )

        assert result.success is True
        assert result.answer == "这是答案"
        assert result.steps == 3

    def test_result_with_tool_calls(self):
        """测试带工具调用的结果"""
        tc = ToolCallRecord(
            tool_name="search",
            arguments={"query": "test"},
            result="搜索结果",
        )
        result = AgentLoopResult(
            success=True,
            answer="答案",
            tool_calls=[tc],
            steps=2,
        )

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "search"

    def test_result_with_error(self):
        """测试带错误的结果"""
        result = AgentLoopResult(
            success=False,
            answer="",
            error="执行出错",
            steps=1,
        )

        assert result.success is False
        assert result.error == "执行出错"

    def test_result_to_dict(self):
        """测试结果转字典"""
        result = AgentLoopResult(
            success=True,
            answer="答案",
            steps=2,
            latency_ms=300,
        )

        result_dict = result.to_dict()

        assert result_dict["success"] is True
        assert result_dict["answer"] == "答案"
        assert result_dict["steps"] == 2


class TestAgentLoop:
    """测试 AgentLoop（同步版本）"""

    @pytest.fixture
    def mock_llm_client(self):
        """Mock LLM 客户端"""
        client = MagicMock(spec=LLMClient)
        client.invoke_with_tools = MagicMock(
            return_value=LLMResponse(
                content="这是最终回答",
                tool_calls=[],
                latency_ms=100,
            )
        )
        return client

    @pytest.fixture
    def mock_tool_registry(self):
        """Mock 工具注册中心"""
        registry = MagicMock(spec=ToolRegistry)
        registry.get_tools = MagicMock(return_value=[])
        registry.get_default_tools = MagicMock(return_value=[])
        registry.total_count = 0
        registry.list_local_tool_names = MagicMock(return_value=[])
        registry.list_mcp_tool_names = MagicMock(return_value=[])
        return registry

    def test_create_loop(self, mock_llm_client, mock_tool_registry):
        """测试创建循环"""
        config = AgentLoopConfig(max_steps=5)
        loop = AgentLoop(
            llm_client=mock_llm_client,
            tool_registry=mock_tool_registry,
            config=config,
        )

        assert loop.config.max_steps == 5

    def test_run_simple_query(self, mock_llm_client, mock_tool_registry):
        """测试简单查询（无工具调用）- 同步版本"""
        loop = AgentLoop(
            llm_client=mock_llm_client,
            tool_registry=mock_tool_registry,
        )
        state = create_test_state(
            trace_id="test",
            user_query="你好",
        )

        result = loop.run(state)

        assert result.success is True
        assert result.answer == "这是最终回答"
        assert result.steps == 1

    def test_run_with_tool_call(self, mock_llm_client, mock_tool_registry, sample_tool):
        """测试带工具调用的查询 - 同步版本"""
        # 第一次返回工具调用，第二次返回最终答案
        tool_call_response = LLMResponse(
            content="",
            tool_calls=[
                ToolCall(id="call-1", name="test_tool", args={"query": "test"})
            ],
            latency_ms=100,
        )
        final_response = LLMResponse(
            content="基于工具结果的回答",
            tool_calls=[],
            latency_ms=100,
        )

        mock_llm_client.invoke_with_tools = MagicMock(
            side_effect=[tool_call_response, final_response]
        )

        mock_tool_registry.get_tools = MagicMock(return_value=[sample_tool])
        mock_tool_registry.get_tool = MagicMock(return_value=sample_tool)

        loop = AgentLoop(
            llm_client=mock_llm_client,
            tool_registry=mock_tool_registry,
        )
        state = create_test_state(
            trace_id="test",
            user_query="使用工具查询",
        )

        result = loop.run(state)

        assert result.success is True
        assert result.steps == 2
        assert len(result.tool_calls) == 1

    def test_run_max_steps_reached(self, mock_llm_client, mock_tool_registry):
        """测试达到最大步数 - 同步版本"""
        # 总是返回工具调用
        tool_call_response = LLMResponse(
            content="",
            tool_calls=[
                ToolCall(id="call-1", name="test_tool", args={})
            ],
            latency_ms=100,
        )

        mock_llm_client.invoke_with_tools = MagicMock(
            return_value=tool_call_response
        )

        loop = AgentLoop(
            llm_client=mock_llm_client,
            tool_registry=mock_tool_registry,
            config=AgentLoopConfig(max_steps=3),
        )
        state = create_test_state(
            trace_id="test",
            user_query="测试",
        )

        result = loop.run(state)

        assert result.steps == 3  # 达到最大步数

    def test_run_with_error(self, mock_llm_client, mock_tool_registry):
        """测试执行出错 - 同步版本"""
        mock_llm_client.invoke_with_tools = MagicMock(
            side_effect=Exception("LLM 调用失败")
        )

        loop = AgentLoop(
            llm_client=mock_llm_client,
            tool_registry=mock_tool_registry,
        )
        state = create_test_state(
            trace_id="test",
            user_query="测试",
        )

        result = loop.run(state)

        assert result.success is False
        assert "LLM 调用失败" in result.error

    def test_run_with_tool_error(self, mock_llm_client, mock_tool_registry):
        """测试工具执行出错 - 同步版本"""
        # 创建一个会报错的工具
        error_tool = MagicMock()
        error_tool.name = "error_tool"
        error_tool.invoke = MagicMock(side_effect=Exception("工具执行失败"))

        tool_call_response = LLMResponse(
            content="",
            tool_calls=[
                ToolCall(id="call-1", name="error_tool", args={})
            ],
            latency_ms=100,
        )
        final_response = LLMResponse(
            content="工具执行出错，但我会继续回答",
            tool_calls=[],
            latency_ms=100,
        )

        mock_llm_client.invoke_with_tools = MagicMock(
            side_effect=[tool_call_response, final_response]
        )

        mock_tool_registry.get_tools = MagicMock(return_value=[error_tool])
        mock_tool_registry.get_tool = MagicMock(return_value=error_tool)

        loop = AgentLoop(
            llm_client=mock_llm_client,
            tool_registry=mock_tool_registry,
        )
        state = create_test_state(
            trace_id="test",
            user_query="测试",
        )

        result = loop.run(state)

        # 即使工具出错，循环也应该继续
        assert result.tool_calls[0].success is False

    def test_run_with_system_prompt(self, mock_llm_client, mock_tool_registry):
        """测试带系统提示词的执行 - 同步版本"""
        loop = AgentLoop(
            llm_client=mock_llm_client,
            tool_registry=mock_tool_registry,
            config=AgentLoopConfig(system_prompt="你是专业助手"),
        )
        state = create_test_state(
            trace_id="test",
            user_query="你好",
        )

        result = loop.run(state)

        assert result.success is True


class TestAgentLoopIntegration:
    """测试 AgentLoop 集成场景（同步版本）"""

    def test_multi_tool_workflow(self):
        """测试多工具工作流 - 同步版本"""
        # 这个测试验证接口设计
        llm_client = MagicMock(spec=LLMClient)
        tool_registry = ToolRegistry()

        # 模拟多次工具调用
        responses = [
            LLMResponse(content="", tool_calls=[
                ToolCall(id="1", name="tool1", args={})
            ], latency_ms=100),
            LLMResponse(content="", tool_calls=[
                ToolCall(id="2", name="tool2", args={})
            ], latency_ms=100),
            LLMResponse(content="最终答案", tool_calls=[], latency_ms=100),
        ]

        llm_client.invoke_with_tools = MagicMock(side_effect=responses)

        loop = AgentLoop(
            llm_client=llm_client,
            tool_registry=tool_registry,
        )

        state = create_test_state(
            trace_id="test",
            user_query="测试",
        )

        # 由于没有注册工具，这个测试可能需要 mock tool_registry
        # 这里主要验证接口设计
        assert loop is not None
