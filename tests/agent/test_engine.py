# -*- coding: utf-8 -*-
"""测试 AgentEngine 引擎"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agent.engine import AgentEngine, EngineConfig
from agent.state import create_initial_state, AgentStatus
from agent.core.router import RouteType, RouterResult
from agent.core.loop import AgentLoopResult
from agent.core.finalize import AgentResponse
from agent.tools.registry import ToolRegistry
from agent.llm.client import LLMClient
from agent.llm.config import LLMConfig


class TestEngineConfig:
    """测试 EngineConfig"""

    def test_create_config(self):
        """测试创建配置"""
        config = EngineConfig(
            max_steps=15,
            timeout_seconds=300,
            debug_verbose=True,
        )

        assert config.max_steps == 15
        assert config.timeout_seconds == 300
        assert config.debug_verbose is True

    def test_config_from_dict(self):
        """测试从字典创建配置"""
        data = {
            "max_steps": 20,
            "timeout_seconds": 600,
            "debug_verbose": True,
        }

        config = EngineConfig.from_dict(data)

        assert config.max_steps == 20
        assert config.timeout_seconds == 600

    def test_config_with_llm_config(self):
        """测试带 LLM 配置"""
        llm_config = LLMConfig(model="gpt-4o", temperature=0.5)
        config = EngineConfig(llm_config=llm_config)

        assert config.llm_config.model == "gpt-4o"
        assert config.llm_config.temperature == 0.5


class TestAgentEngine:
    """测试 AgentEngine"""

    @pytest.fixture
    def mock_components(self):
        """Mock 组件"""
        llm_client = MagicMock(spec=LLMClient)
        tool_registry = MagicMock(spec=ToolRegistry)

        return llm_client, tool_registry

    def test_create_engine(self, mock_components):
        """测试创建引擎"""
        llm_client, tool_registry = mock_components
        config = EngineConfig(max_steps=5)

        engine = AgentEngine(
            llm_client=llm_client,
            tool_registry=tool_registry,
            config=config,
        )

        assert engine.config.max_steps == 5

    def test_create_engine_default_config(self, mock_components):
        """测试默认配置创建"""
        llm_client, tool_registry = mock_components

        engine = AgentEngine(
            llm_client=llm_client,
            tool_registry=tool_registry,
        )

        assert engine.config.max_steps == 10  # 默认值

    def test_backend_name(self, mock_components):
        """测试后端名称"""
        llm_client, tool_registry = mock_components
        engine = AgentEngine(
            llm_client=llm_client,
            tool_registry=tool_registry,
        )

        assert engine.backend_name == "agent_engine"

    @pytest.mark.asyncio
    async def test_arun_basic(self, mock_components):
        """测试基本执行"""
        llm_client, tool_registry = mock_components

        # Mock 路由返回
        router_result = RouterResult(
            route=RouteType.KNOWLEDGE_QA,
            domain_relevance=0.8,
        )

        # Mock 循环结果
        loop_result = AgentLoopResult(
            success=True,
            answer="这是答案",
            steps=2,
        )

        with patch.object(AgentEngine, '_run_router') as mock_router, \
             patch.object(AgentEngine, '_run_loop') as mock_loop, \
             patch.object(AgentEngine, '_run_finalize') as mock_finalize:

            mock_router.return_value = router_result
            mock_loop.return_value = loop_result
            mock_finalize.return_value = AgentResponse(
                role="assistant",
                content="最终答案",
            )

            engine = AgentEngine(
                llm_client=llm_client,
                tool_registry=tool_registry,
            )

            response = await engine.arun(
                user_query="测试查询",
                session_id="session-123",
                trace_id="trace-456",
            )

            assert response.content == "最终答案"

    @pytest.mark.asyncio
    async def test_arun_out_of_scope(self, mock_components):
        """测试领域外查询"""
        llm_client, tool_registry = mock_components

        engine = AgentEngine(
            llm_client=llm_client,
            tool_registry=tool_registry,
        )

        with patch.object(engine._router, 'route') as mock_route:
            mock_route.return_value = RouterResult(
                route=RouteType.OUT_OF_SCOPE,
                domain_relevance=0.1,
            )

            response = await engine.arun(
                user_query="今天天气怎么样",
            )

            assert response.kind == "out_of_scope"

    @pytest.mark.asyncio
    async def test_arun_small_talk(self, mock_components):
        """测试闲聊"""
        llm_client, tool_registry = mock_components

        engine = AgentEngine(
            llm_client=llm_client,
            tool_registry=tool_registry,
        )

        with patch.object(engine._router, 'route') as mock_route:
            mock_route.return_value = RouterResult(
                route=RouteType.SMALL_TALK,
                domain_relevance=0.0,
            )

            response = await engine.arun(
                user_query="你好",
            )

            assert response.kind == "small_talk"

    @pytest.mark.asyncio
    async def test_arun_with_error(self, mock_components):
        """测试执行出错"""
        llm_client, tool_registry = mock_components

        engine = AgentEngine(
            llm_client=llm_client,
            tool_registry=tool_registry,
        )

        with patch.object(engine._router, 'route') as mock_route:
            mock_route.side_effect = Exception("路由失败")

            response = await engine.arun(
                user_query="测试",
            )

            assert response.success is False
            assert "路由失败" in response.error

    def test_register_local_tool(self, mock_components, sample_tool):
        """测试注册本地工具"""
        llm_client, tool_registry = mock_components
        tool_registry.register_local_tool = MagicMock()

        engine = AgentEngine(
            llm_client=llm_client,
            tool_registry=tool_registry,
        )

        engine.register_local_tool(sample_tool)

        tool_registry.register_local_tool.assert_called_once_with(sample_tool)

    def test_register_mcp_tool(self, mock_components, sample_tool):
        """测试注册 MCP 工具"""
        llm_client, tool_registry = mock_components
        tool_registry.register_mcp_tool = MagicMock()

        engine = AgentEngine(
            llm_client=llm_client,
            tool_registry=tool_registry,
        )

        engine.register_mcp_tool(sample_tool)

        tool_registry.register_mcp_tool.assert_called_once_with(sample_tool)

    def test_get_stats(self, mock_components):
        """测试获取统计信息"""
        llm_client, tool_registry = mock_components
        llm_client.config = LLMConfig(model="gpt-4o")
        tool_registry.get_stats = MagicMock(return_value={"total_tools": 5})

        engine = AgentEngine(
            llm_client=llm_client,
            tool_registry=tool_registry,
        )

        stats = engine.get_stats()

        assert stats["backend"] == "agent_engine"
        assert stats["llm_model"] == "gpt-4o"
        assert stats["tools"]["total_tools"] == 5


class TestAgentEngineIntegration:
    """测试 AgentEngine 集成场景"""

    @pytest.mark.asyncio
    async def test_full_flow(self):
        """测试完整流程"""
        # 这个测试验证接口设计
        # 实际集成测试需要更多 mock

        config = LLMConfig(model="gpt-4o")
        llm_client = MagicMock(spec=LLMClient)
        tool_registry = ToolRegistry()

        engine = AgentEngine(
            llm_client=llm_client,
            tool_registry=tool_registry,
        )

        # 验证引擎结构
        assert engine._router is not None
        assert engine._loop is not None
        assert engine._finalize is not None

    @pytest.mark.asyncio
    async def test_skill_execution_route(self, mock_components):
        """测试 Skill 执行路由"""
        llm_client, tool_registry = mock_components

        engine = AgentEngine(
            llm_client=llm_client,
            tool_registry=tool_registry,
        )

        with patch.object(engine._router, 'route') as mock_route, \
             patch.object(engine._loop, 'run') as mock_loop:

            mock_route.return_value = RouterResult(
                route=RouteType.SKILL_EXECUTION,
                domain_relevance=0.9,
                active_skill_id="skill-123",
                tool_whitelist=["tool1", "tool2"],
            )
            mock_loop.return_value = AgentLoopResult(
                success=True,
                answer="Skill 执行结果",
                steps=1,
            )

            response = await engine.arun(
                user_query="执行 Skill",
            )

            # 验证 tool_whitelist 被传递
            mock_loop.assert_called_once()
            call_args = mock_loop.call_args
            state = call_args[0][0]  # 第一个位置参数
            assert "tool_whitelist" in state

    @pytest.mark.asyncio
    async def test_timeout_handling(self, mock_components):
        """测试超时处理"""
        llm_client, tool_registry = mock_components

        config = EngineConfig(timeout_seconds=1)
        engine = AgentEngine(
            llm_client=llm_client,
            tool_registry=tool_registry,
            config=config,
        )

        with patch.object(engine._router, 'route') as mock_route:
            mock_route.side_effect = TimeoutError("执行超时")

            response = await engine.arun(
                user_query="测试",
            )

            assert response.success is False


class TestAgentEngineEdgeCases:
    """测试 AgentEngine 边界情况"""

    @pytest.mark.asyncio
    async def test_empty_query(self, mock_components):
        """测试空查询"""
        llm_client, tool_registry = mock_components

        engine = AgentEngine(
            llm_client=llm_client,
            tool_registry=tool_registry,
        )

        with patch.object(engine._router, 'route') as mock_route:
            mock_route.return_value = RouterResult(
                route=RouteType.GENERAL,
                domain_relevance=0.0,
            )

            response = await engine.arun(user_query="")

            assert response is not None

    @pytest.mark.asyncio
    async def test_very_long_query(self, mock_components):
        """测试超长查询"""
        llm_client, tool_registry = mock_components

        engine = AgentEngine(
            llm_client=llm_client,
            tool_registry=tool_registry,
        )

        long_query = "x" * 10000

        with patch.object(engine._router, 'route') as mock_route:
            mock_route.return_value = RouterResult(
                route=RouteType.KNOWLEDGE_QA,
                domain_relevance=0.5,
            )
            with patch.object(engine._loop, 'run') as mock_loop:
                mock_loop.return_value = AgentLoopResult(
                    success=True,
                    answer="答案",
                    steps=1,
                )

                response = await engine.arun(user_query=long_query)

                assert response is not None

    @pytest.mark.asyncio
    async def test_with_history(self, mock_components):
        """测试带历史对话"""
        llm_client, tool_registry = mock_components

        engine = AgentEngine(
            llm_client=llm_client,
            tool_registry=tool_registry,
        )

        history = [
            {"role": "user", "content": "问题1"},
            {"role": "assistant", "content": "回答1"},
        ]

        with patch.object(engine._router, 'route') as mock_route:
            mock_route.return_value = RouterResult(
                route=RouteType.KNOWLEDGE_QA,
                domain_relevance=0.5,
            )
            with patch.object(engine._loop, 'run') as mock_loop:
                mock_loop.return_value = AgentLoopResult(
                    success=True,
                    answer="基于历史的答案",
                    steps=1,
                )

                response = await engine.arun(
                    user_query="继续",
                    history=history,
                )

                assert response is not None
