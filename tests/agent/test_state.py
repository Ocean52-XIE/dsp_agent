# -*- coding: utf-8 -*-
"""测试 AgentState 状态定义"""
import pytest

from agent.state import (
    AgentState,
    AgentStatus,
    AgentLoopResult,
    ToolCallRecord,
    create_initial_state,
    state_add_message,
    state_add_tool_call,
)


class TestAgentStatus:
    """测试 AgentStatus 枚举"""

    def test_status_values(self):
        """测试状态值"""
        assert AgentStatus.INIT.value == "init"
        assert AgentStatus.ROUTING.value == "routing"
        assert AgentStatus.RUNNING.value == "running"
        assert AgentStatus.TOOL_CALLING.value == "tool_calling"
        assert AgentStatus.COMPLETED.value == "completed"
        assert AgentStatus.FAILED.value == "failed"
        assert AgentStatus.TIMEOUT.value == "timeout"

    def test_status_is_string(self):
        """测试状态是字符串类型"""
        assert isinstance(AgentStatus.INIT.value, str)
        assert AgentStatus.INIT.value == "init"


class TestToolCallRecord:
    """测试 ToolCallRecord"""

    def test_create_record(self):
        """测试创建工具调用记录"""
        record = ToolCallRecord(
            tool_name="test_tool",
            arguments={"query": "test"},
            result="测试结果",
            success=True,
            latency_ms=100,
        )

        assert record.tool_name == "test_tool"
        assert record.arguments == {"query": "test"}
        assert record.result == "测试结果"
        assert record.success is True
        assert record.latency_ms == 100

    def test_record_to_dict(self):
        """测试记录转换为字典"""
        record = ToolCallRecord(
            tool_name="test_tool",
            arguments={"query": "test"},
            result="测试结果",
            success=True,
            latency_ms=100,
        )

        result = record.to_dict()

        assert result["tool_name"] == "test_tool"
        assert result["arguments"] == {"query": "test"}
        assert result["success"] is True
        assert result["latency_ms"] == 100

    def test_record_with_error(self):
        """测试带错误的记录"""
        record = ToolCallRecord(
            tool_name="test_tool",
            arguments={},
            result="",
            success=False,
            error="工具执行失败",
            latency_ms=50,
        )

        assert record.success is False
        assert record.error == "工具执行失败"

    def test_record_truncates_long_result(self):
        """测试长结果被截断"""
        long_result = "x" * 1000
        record = ToolCallRecord(
            tool_name="test_tool",
            arguments={},
            result=long_result,
            success=True,
        )

        result_dict = record.to_dict()

        assert len(result_dict["result"]) == 500


class TestAgentLoopResult:
    """测试 AgentLoopResult"""

    def test_create_result(self):
        """测试创建循环结果"""
        result = AgentLoopResult(
            success=True,
            answer="这是答案",
            tool_calls=[],
            steps=3,
            latency_ms=500,
        )

        assert result.success is True
        assert result.answer == "这是答案"
        assert result.steps == 3

    def test_result_with_tool_calls(self):
        """测试带工具调用的结果"""
        tool_call = ToolCallRecord(
            tool_name="search",
            arguments={"query": "test"},
            result="搜索结果",
        )
        result = AgentLoopResult(
            success=True,
            answer="基于搜索结果的回答",
            tool_calls=[tool_call],
            steps=2,
        )

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "search"

    def test_result_to_dict(self):
        """测试结果转换为字典"""
        result = AgentLoopResult(
            success=True,
            answer="答案",
            steps=2,
        )

        result_dict = result.to_dict()

        assert result_dict["success"] is True
        assert result_dict["answer"] == "答案"
        assert result_dict["steps"] == 2


class TestCreateInitialState:
    """测试 create_initial_state"""

    def test_create_basic_state(self):
        """测试创建基本状态"""
        state = create_initial_state(
            trace_id="trace-123",
            session_id="session-456",
            user_query="测试查询",
        )

        assert state["trace_id"] == "trace-123"
        assert state["session_id"] == "session-456"
        assert state["user_query"] == "测试查询"
        assert state["status"] == AgentStatus.INIT.value
        assert state["current_step"] == 0
        assert state["is_complete"] is False

    def test_create_state_with_history(self):
        """测试带历史的状态"""
        history = [
            {"role": "user", "content": "问题1"},
            {"role": "assistant", "content": "回答1"},
        ]
        state = create_initial_state(
            trace_id="trace-123",
            session_id="session-456",
            user_query="继续",
            history=history,
        )

        assert len(state["history"]) == 2
        assert state["history"][0]["role"] == "user"

    def test_state_has_empty_defaults(self):
        """测试状态有空的默认值"""
        state = create_initial_state(
            trace_id="trace-123",
            session_id="session-456",
            user_query="查询",
        )

        assert state["tool_whitelist"] == []
        assert state["messages"] == []
        assert state["tool_calls"] == []
        assert state["answer"] == ""
        assert state["citations"] == []

    def test_state_strips_query(self):
        """测试查询被去除空白"""
        state = create_initial_state(
            trace_id="trace-123",
            session_id="session-456",
            user_query="  带空格的查询  ",
        )

        assert state["user_query"] == "带空格的查询"


class TestStateAddMessage:
    """测试 state_add_message"""

    def test_add_user_message(self, sample_state):
        """测试添加用户消息"""
        state_add_message(sample_state, "user", "用户消息")

        assert len(sample_state["messages"]) == 1
        assert sample_state["messages"][0]["role"] == "user"
        assert sample_state["messages"][0]["content"] == "用户消息"

    def test_add_assistant_message(self, sample_state):
        """测试添加助手消息"""
        state_add_message(sample_state, "assistant", "助手消息")

        assert len(sample_state["messages"]) == 1
        assert sample_state["messages"][0]["role"] == "assistant"

    def test_add_tool_message(self, sample_state):
        """测试添加工具消息"""
        state_add_message(
            sample_state,
            "tool",
            "工具结果",
            tool_call_id="call-123",
        )

        assert len(sample_state["messages"]) == 1
        assert sample_state["messages"][0]["role"] == "tool"
        assert sample_state["messages"][0]["tool_call_id"] == "call-123"

    def test_add_message_with_tool_calls(self, sample_state):
        """测试添加带工具调用的消息"""
        state_add_message(
            sample_state,
            "assistant",
            "",
            tool_calls=[{"id": "call-123", "name": "test"}],
        )

        assert len(sample_state["messages"]) == 1
        assert "tool_calls" in sample_state["messages"][0]


class TestStateAddToolCall:
    """测试 state_add_tool_call"""

    def test_add_tool_call(self, sample_state):
        """测试添加工具调用记录"""
        record = ToolCallRecord(
            tool_name="test_tool",
            arguments={"query": "test"},
            result="结果",
        )

        state_add_tool_call(sample_state, record)

        assert len(sample_state["tool_calls"]) == 1
        assert sample_state["tool_calls"][0]["tool_name"] == "test_tool"

    def test_add_multiple_tool_calls(self, sample_state):
        """测试添加多个工具调用"""
        for i in range(3):
            record = ToolCallRecord(
                tool_name=f"tool_{i}",
                arguments={},
                result=f"result_{i}",
            )
            state_add_tool_call(sample_state, record)

        assert len(sample_state["tool_calls"]) == 3
