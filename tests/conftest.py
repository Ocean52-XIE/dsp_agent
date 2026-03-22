# -*- coding: utf-8 -*-
"""Agent 模块测试配置

提供共享的 fixtures 和测试工具。
"""
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


def pytest_configure(config):
    """Pytest 配置钩子 - 在测试收集前设置 Python 路径"""
    # 确保可以导入 agent 模块
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


# 确保可以导入 agent 模块 (备用)
project_root = Path(__file__).resolve().parents[1]
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


# =============================================================================
# 环境变量 Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    """设置测试环境变量"""
    # LLM 配置
    monkeypatch.setenv("AGENT_LLM_MODEL", "gpt-4o")
    monkeypatch.setenv("AGENT_LLM_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("AGENT_LLM_API_KEY", "test-api-key")
    monkeypatch.setenv("AGENT_LLM_TEMPERATURE", "0.7")
    monkeypatch.setenv("AGENT_LLM_MAX_TOKENS", "4096")
    monkeypatch.setenv("AGENT_LLM_TIMEOUT", "120")

    # Agent 配置
    monkeypatch.setenv("AGENT_MAX_STEPS", "5")
    monkeypatch.setenv("AGENT_TIMEOUT_SECONDS", "60")
    monkeypatch.setenv("AGENT_DEBUG_VERBOSE", "false")


# =============================================================================
# Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_llm_response():
    """Mock LLM 响应"""
    from agent.llm.client import LLMResponse, ToolCall

    return LLMResponse(
        content="这是一个测试响应",
        tool_calls=[],
        latency_ms=100,
    )


@pytest.fixture
def mock_llm_response_with_tool():
    """Mock LLM 响应 (带工具调用)"""
    from agent.llm.client import LLMResponse, ToolCall

    return LLMResponse(
        content="",
        tool_calls=[
            ToolCall(
                id="call-123",
                name="test_tool",
                args={"query": "test"},
            )
        ],
        latency_ms=150,
    )


@pytest.fixture
def mock_llm_client():
    """Mock LLM 客户端（支持同步和异步方法）"""
    client = MagicMock()
    # 异步方法（已弃用，保留向后兼容）
    client.ainvoke = AsyncMock(return_value=MagicMock(content="测试响应", latency_ms=100))
    client.ainvoke_with_tools = AsyncMock(
        return_value=MagicMock(
            content="测试响应",
            tool_calls=[],
            latency_ms=100,
        )
    )
    # 同步方法（当前使用）
    client.invoke = MagicMock(return_value=MagicMock(content="测试响应", latency_ms=100))
    client.invoke_with_tools = MagicMock(
        return_value=MagicMock(
            content="测试响应",
            tool_calls=[],
            latency_ms=100,
        )
    )
    return client


@pytest.fixture
def mock_tool_registry():
    """Mock 工具注册中心"""
    registry = MagicMock()
    registry.get_tools = MagicMock(return_value=[])
    registry.get_tool = MagicMock(return_value=None)
    registry.get_default_tools = MagicMock(return_value=[])
    return registry


# =============================================================================
# State Fixtures
# =============================================================================

@pytest.fixture
def sample_state():
    """示例 Agent 状态（简化版）"""
    from agent.state import AgentState

    return {
        "trace_id": "test-trace-123",
        "user_query": "这是一个测试查询",
        "history": [],
        "tool_whitelist": None,
    }


@pytest.fixture
def sample_state_with_history():
    """带历史的 Agent 状态"""
    return {
        "trace_id": "test-trace-123",
        "user_query": "继续上一个问题",
        "history": [
            {"role": "user", "content": "之前的问题"},
            {"role": "assistant", "content": "之前的回答"},
        ],
        "tool_whitelist": None,
    }


# =============================================================================
# Tool Fixtures
# =============================================================================

@pytest.fixture
def sample_tool():
    """示例工具"""
    from langchain_core.tools import tool

    @tool
    def test_tool(query: str) -> str:
        """这是一个测试工具

        Args:
            query: 查询字符串

        Returns:
            结果字符串
        """
        return f"处理结果: {query}"

    return test_tool


@pytest.fixture
def sample_tool_with_params():
    """带参数的示例工具"""
    from langchain_core.tools import tool

    @tool
    def search(query: str, top_k: int = 5) -> str:
        """搜索相关内容

        Args:
            query: 搜索查询
            top_k: 返回数量

        Returns:
            搜索结果
        """
        return f"找到 {top_k} 条结果: {query}"

    return search


# =============================================================================
# Config Fixtures
# =============================================================================

@pytest.fixture
def sample_router_config():
    """示例路由配置"""
    from agent.core.router import RouterConfig

    return RouterConfig(
        domain_terms=("ctr", "cvr", "预估", "广告"),
        offtopic_terms=("天气", "股票", "新闻"),
        small_talk_exact=("你好", "在吗"),
        small_talk_substr=("怎么样", "好不好"),
        threshold=0.3,
        use_skill_matching=True,
    )


@pytest.fixture
def sample_loop_config():
    """示例循环配置"""
    from agent.core.loop import AgentLoopConfig

    return AgentLoopConfig(
        max_steps=5,
        timeout_seconds=60,
        system_prompt="你是一个测试助手",
    )


@pytest.fixture
def sample_finalize_config():
    """示例整理配置"""
    from agent.core.finalize import FinalizeConfig

    return FinalizeConfig(
        max_citations=3,
        include_tool_calls=True,
        include_debug_info=True,
    )
