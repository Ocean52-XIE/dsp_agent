# -*- coding: utf-8 -*-
"""测试 AgentFinalize 结果整理"""
import pytest

from agent.core.finalize import (
    AgentFinalize,
    FinalizeConfig,
    AgentResponse,
)
from agent.state import create_initial_state, ToolCallRecord, AgentStatus


class TestFinalizeConfig:
    """测试 FinalizeConfig"""

    def test_create_config(self):
        """测试创建配置"""
        config = FinalizeConfig(
            max_citations=5,
            include_tool_calls=True,
            include_debug_info=True,
        )

        assert config.max_citations == 5
        assert config.include_tool_calls is True
        assert config.include_debug_info is True

    def test_config_from_dict(self):
        """测试从字典创建配置"""
        data = {
            "max_citations": 10,
            "include_tool_calls": False,
            "include_debug_info": True,
        }

        config = FinalizeConfig.from_dict(data)

        assert config.max_citations == 10
        assert config.include_tool_calls is False

    def test_config_defaults(self):
        """测试默认值"""
        config = FinalizeConfig()

        assert config.max_citations == 4
        assert config.include_tool_calls is True
        assert config.include_debug_info is False


class TestAgentResponse:
    """测试 AgentResponse"""

    def test_create_response(self):
        """测试创建响应"""
        response = AgentResponse(
            role="assistant",
            content="这是回答",
            kind="knowledge_qa",
            intent="knowledge_qa",
            status="completed",
            trace_id="trace-123",
        )

        assert response.role == "assistant"
        assert response.content == "这是回答"
        assert response.kind == "knowledge_qa"

    def test_response_with_citations(self):
        """测试带引用的响应"""
        citations = [
            {"source": "wiki", "title": "文档1", "url": "http://example.com"},
            {"source": "code", "path": "/src/main.py", "line": 10},
        ]
        response = AgentResponse(
            role="assistant",
            content="基于引用的回答",
            citations=citations,
        )

        assert len(response.citations) == 2

    def test_response_with_tool_calls(self):
        """测试带工具调用的响应"""
        tool_calls = [
            {"tool_name": "search", "arguments": {"query": "test"}},
        ]
        response = AgentResponse(
            role="assistant",
            content="答案",
            tool_calls=tool_calls,
        )

        assert len(response.tool_calls) == 1

    def test_response_with_error(self):
        """测试带错误的响应"""
        response = AgentResponse(
            role="assistant",
            content="",
            status="failed",
            success=False,
            error="执行失败",
        )

        assert response.success is False
        assert response.error == "执行失败"

    def test_response_to_dict(self):
        """测试响应转字典"""
        response = AgentResponse(
            role="assistant",
            content="答案",
            kind="knowledge_qa",
            trace_id="trace-123",
            latency_ms=500,
        )

        result = response.to_dict()

        assert result["role"] == "assistant"
        assert result["content"] == "答案"
        assert result["latency_ms"] == 500

    def test_response_excludes_empty_fields(self):
        """测试空字段被排除"""
        response = AgentResponse(
            role="assistant",
            content="答案",
        )

        result = response.to_dict()

        assert "citations" not in result
        assert "tool_calls" not in result
        assert "error" not in result


class TestAgentFinalize:
    """测试 AgentFinalize"""

    def test_create_finalize(self):
        """测试创建 Finalize"""
        finalize = AgentFinalize()

        assert finalize is not None
        assert finalize.config is not None

    def test_create_finalize_with_config(self):
        """测试带配置创建"""
        config = FinalizeConfig(max_citations=10)
        finalize = AgentFinalize(config=config)

        assert finalize.config.max_citations == 10

    def test_run_basic(self, sample_state):
        """测试基本执行"""
        finalize = AgentFinalize()
        response = finalize.run(
            state=sample_state,
            answer="这是答案",
            latency_ms=100,
        )

        assert response.content == "这是答案"
        assert response.success is True
        assert response.status == "completed"

    def test_run_with_tool_calls(self, sample_state):
        """测试带工具调用"""
        tool_calls = [
            ToolCallRecord(
                tool_name="search",
                arguments={"query": "test"},
                result="结果",
                success=True,
                latency_ms=50,
            )
        ]

        finalize = AgentFinalize()
        response = finalize.run(
            state=sample_state,
            answer="基于搜索的答案",
            tool_calls=tool_calls,
        )

        assert len(response.tool_calls) == 1

    def test_run_with_error(self, sample_state):
        """测试带错误执行"""
        finalize = AgentFinalize()
        response = finalize.run(
            state=sample_state,
            answer="",
            error="执行失败",
        )

        assert response.success is False
        assert response.error == "执行失败"
        assert response.status == "failed"

    def test_run_with_citations(self, sample_state):
        """测试带引用"""
        sample_state["citations"] = [
            {"source": "wiki", "title": "文档1"},
            {"source": "wiki", "title": "文档2"},
        ]

        finalize = AgentFinalize()
        response = finalize.run(
            state=sample_state,
            answer="基于引用的答案",
        )

        assert len(response.citations) == 2

    def test_run_citation_limit(self, sample_state):
        """测试引用数量限制"""
        sample_state["citations"] = [
            {"source": f"wiki_{i}", "title": f"文档{i}"}
            for i in range(10)
        ]

        config = FinalizeConfig(max_citations=3)
        finalize = AgentFinalize(config=config)
        response = finalize.run(
            state=sample_state,
            answer="答案",
        )

        assert len(response.citations) == 3

    def test_run_with_debug_info(self, sample_state):
        """测试带调试信息"""
        config = FinalizeConfig(include_debug_info=True)
        finalize = AgentFinalize(config=config)
        response = finalize.run(
            state=sample_state,
            answer="答案",
        )

        assert response.debug != {}
        assert response.analysis != {}

    def test_run_without_debug_info(self, sample_state):
        """测试不带调试信息"""
        config = FinalizeConfig(include_debug_info=False)
        finalize = AgentFinalize(config=config)
        response = finalize.run(
            state=sample_state,
            answer="答案",
        )

        assert response.debug == {}
        assert response.analysis == {}

    def test_run_truncates_long_answer(self, sample_state):
        """测试截断长答案"""
        long_answer = "x" * 10000

        config = FinalizeConfig(
            truncate_answer=True,
            max_answer_length=1000,
        )
        finalize = AgentFinalize(config=config)
        response = finalize.run(
            state=sample_state,
            answer=long_answer,
        )

        assert len(response.content) == 1003  # 1000 + "..."

    def test_update_state(self, sample_state):
        """测试更新状态"""
        finalize = AgentFinalize()
        response = finalize.run(
            state=sample_state,
            answer="答案",
        )

        finalize.update_state(sample_state, response)

        assert sample_state["answer"] == "答案"
        assert "response" in sample_state["debug_info"]

    def test_determine_kind_knowledge_qa(self, sample_state):
        """测试确定类型 - 知识问答"""
        sample_state["route"] = "knowledge_qa"

        finalize = AgentFinalize()
        kind = finalize._determine_kind(sample_state)

        assert kind == "knowledge_qa"

    def test_determine_kind_issue_analysis(self, sample_state):
        """测试确定类型 - 问题分析"""
        sample_state["route"] = "issue_analysis"

        finalize = AgentFinalize()
        kind = finalize._determine_kind(sample_state)

        assert kind == "issue_analysis"

    def test_determine_status_completed(self, sample_state):
        """测试确定状态 - 完成"""
        sample_state["status"] = AgentStatus.COMPLETED.value

        finalize = AgentFinalize()
        status = finalize._determine_status(sample_state, "")

        assert status == "completed"

    def test_determine_status_failed(self, sample_state):
        """测试确定状态 - 失败"""
        sample_state["status"] = AgentStatus.FAILED.value

        finalize = AgentFinalize()
        status = finalize._determine_status(sample_state, "")

        assert status == "failed"

    def test_determine_status_with_error(self, sample_state):
        """测试确定状态 - 有错误"""
        finalize = AgentFinalize()
        status = finalize._determine_status(sample_state, "出错了")

        assert status == "failed"

    def test_deduplicate_citations(self):
        """测试去重引用"""
        citations = [
            {"source_type": "wiki", "path": "doc1.md", "section": "intro"},
            {"source_type": "wiki", "path": "doc1.md", "section": "intro"},  # 重复
            {"source_type": "wiki", "path": "doc2.md", "section": "main"},
        ]

        finalize = AgentFinalize()
        result = finalize._deduplicate_citations(citations)

        assert len(result) == 2


class TestAgentFinalizeEdgeCases:
    """测试 AgentFinalize 边界情况"""

    def test_run_with_empty_state(self):
        """测试空状态"""
        finalize = AgentFinalize()

        # 最小状态
        state = {"trace_id": "test"}
        response = finalize.run(state=state, answer="答案")

        assert response.content == "答案"

    def test_run_with_empty_answer(self, sample_state):
        """测试空答案"""
        finalize = AgentFinalize()
        response = finalize.run(
            state=sample_state,
            answer="",
        )

        assert response.content == ""

    def test_run_with_none_tool_calls(self, sample_state):
        """测试 None 工具调用"""
        finalize = AgentFinalize()
        response = finalize.run(
            state=sample_state,
            answer="答案",
            tool_calls=None,
        )

        assert response.tool_calls == []

    def test_run_with_empty_citations(self, sample_state):
        """测试空引用"""
        sample_state["citations"] = []

        finalize = AgentFinalize()
        response = finalize.run(
            state=sample_state,
            answer="答案",
        )

        assert response.citations == []

    def test_run_with_malformed_citations(self, sample_state):
        """测试格式错误的引用"""
        sample_state["citations"] = [
            {"source_type": "wiki"},  # 缺少 path 和 section
            None,  # None 值
            "invalid",  # 非字典
        ]

        finalize = AgentFinalize()
        response = finalize.run(
            state=sample_state,
            answer="答案",
        )

        # 应该处理异常情况
        assert response is not None
