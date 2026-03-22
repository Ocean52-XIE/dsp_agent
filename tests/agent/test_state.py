# -*- coding: utf-8 -*-
"""测试 AgentState 状态定义（简化版）"""
import pytest

from agent.state import (
    AgentState,
    AgentStatus,
    ToolCallRecord,
)


class TestAgentStatus:
    """测试 AgentStatus 枚举"""

    def test_status_values(self):
        """测试状态值"""
        assert AgentStatus.INIT.value == "init"
        assert AgentStatus.RUNNING.value == "running"
        assert AgentStatus.TOOL_CALLING.value == "tool_calling"
        assert AgentStatus.COMPLETED.value == "completed"
        assert AgentStatus.FAILED.value == "failed"
        assert AgentStatus.TIMEOUT.value == "timeout"

    def test_status_is_string(self):
        """测试状态是字符串类型"""
        assert isinstance(AgentStatus.INIT.value, str)
        assert AgentStatus.INIT.value == "init"

    def test_status_string_conversion(self):
        """测试状态枚举可以转换为字符串"""
        assert str(AgentStatus.RUNNING) == "AgentStatus.RUNNING"
        assert AgentStatus.RUNNING.value == "running"


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

    def test_record_default_values(self):
        """测试默认值"""
        record = ToolCallRecord(
            tool_name="test_tool",
            arguments={},
        )

        assert record.result == ""
        assert record.success is True
        assert record.error == ""
        assert record.latency_ms == 0


class TestAgentState:
    """测试 AgentState TypedDict"""

    def test_create_minimal_state(self):
        """测试创建最小状态"""
        state: AgentState = {
            "trace_id": "test-trace",
            "user_query": "测试查询",
        }

        assert state["trace_id"] == "test-trace"
        assert state["user_query"] == "测试查询"

    def test_create_state_with_history(self):
        """测试带历史的状态"""
        state: AgentState = {
            "trace_id": "test-trace",
            "user_query": "继续",
            "history": [
                {"role": "user", "content": "问题1"},
                {"role": "assistant", "content": "回答1"},
            ],
        }

        assert len(state["history"]) == 2
        assert state["history"][0]["role"] == "user"

    def test_create_state_with_tool_whitelist(self):
        """测试带工具白名单的状态"""
        state: AgentState = {
            "trace_id": "test-trace",
            "user_query": "使用工具",
            "tool_whitelist": ["search", "read_file"],
        }

        assert state["tool_whitelist"] == ["search", "read_file"]

    def test_state_all_fields_optional(self):
        """测试所有字段都是可选的"""
        # AgentState 使用 total=False，所有字段都是可选的
        state: AgentState = {}

        assert state.get("trace_id") is None
        assert state.get("user_query") is None
        assert state.get("history") is None
        assert state.get("tool_whitelist") is None
