# -*- coding: utf-8 -*-
"""测试 AgentService 服务入口"""
import os
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from agent.service import AgentService, get_agent_service
from agent.engine import AgentEngine, EngineConfig
from agent.core.finalize import AgentResponse


class TestAgentService:
    """测试 AgentService"""

    def test_create_service(self):
        """测试创建服务"""
        service = AgentService()

        assert service is not None

    def test_create_service_with_engine(self):
        """测试带引擎创建服务"""
        engine = MagicMock(spec=AgentEngine)
        service = AgentService(engine=engine)

        assert service._engine == engine

    def test_create_service_with_config(self):
        """测试带配置创建服务"""
        config = EngineConfig(max_steps=5)
        service = AgentService(config=config)

        assert service._config.max_steps == 5

    @pytest.mark.asyncio
    async def test_arun_basic(self):
        """测试基本执行"""
        mock_engine = MagicMock(spec=AgentEngine)
        mock_engine.arun = AsyncMock(return_value=AgentResponse(
            role="assistant",
            content="测试回答",
        ))

        service = AgentService(engine=mock_engine)
        response = await service.arun(
            user_query="测试查询",
            session_id="session-123",
        )

        assert response.content == "测试回答"
        mock_engine.arun.assert_called_once()

    @pytest.mark.asyncio
    async def test_arun_with_trace_id(self):
        """测试带 trace_id 执行"""
        mock_engine = MagicMock(spec=AgentEngine)
        mock_engine.arun = AsyncMock(return_value=AgentResponse(
            role="assistant",
            content="回答",
        ))

        service = AgentService(engine=mock_engine)
        response = await service.arun(
            user_query="测试",
            trace_id="trace-456",
        )

        assert response is not None

    @pytest.mark.asyncio
    async def test_arun_with_history(self):
        """测试带历史对话执行"""
        mock_engine = MagicMock(spec=AgentEngine)
        mock_engine.arun = AsyncMock(return_value=AgentResponse(
            role="assistant",
            content="基于历史的回答",
        ))

        service = AgentService(engine=mock_engine)
        history = [
            {"role": "user", "content": "问题1"},
            {"role": "assistant", "content": "回答1"},
        ]
        response = await service.arun(
            user_query="继续",
            history=history,
        )

        assert response is not None

    def test_run_sync(self):
        """测试同步执行"""
        mock_engine = MagicMock(spec=AgentEngine)
        mock_engine.arun = AsyncMock(return_value=AgentResponse(
            role="assistant",
            content="同步回答",
        ))

        service = AgentService(engine=mock_engine)
        response = service.run(
            user_query="测试",
        )

        assert response.content == "同步回答"

    def test_register_local_tool(self, sample_tool):
        """测试注册本地工具"""
        mock_engine = MagicMock(spec=AgentEngine)
        mock_engine.register_local_tool = MagicMock()

        service = AgentService(engine=mock_engine)
        service.register_local_tool(sample_tool)

        mock_engine.register_local_tool.assert_called_once_with(sample_tool)

    def test_register_mcp_tool(self, sample_tool):
        """测试注册 MCP 工具"""
        mock_engine = MagicMock(spec=AgentEngine)
        mock_engine.register_mcp_tool = MagicMock()

        service = AgentService(engine=mock_engine)
        service.register_mcp_tool(sample_tool)

        mock_engine.register_mcp_tool.assert_called_once_with(sample_tool)

    def test_get_stats(self):
        """测试获取统计信息"""
        mock_engine = MagicMock(spec=AgentEngine)
        mock_engine.get_stats = MagicMock(return_value={"backend": "test"})

        service = AgentService(engine=mock_engine)
        stats = service.get_stats()

        assert "backend" in stats

    def test_backend_name(self):
        """测试后端名称"""
        service = AgentService()
        assert service.backend_name == "agent_service"


class TestAgentServiceFromEnv:
    """测试从环境变量创建服务"""

    def test_from_env_basic(self, monkeypatch):
        """测试基本创建"""
        monkeypatch.setenv("AGENT_LLM_MODEL", "gpt-4o")
        monkeypatch.setenv("AGENT_MAX_STEPS", "15")

        service = AgentService.from_env()

        assert service._config.max_steps == 15

    def test_from_env_with_prefix(self, monkeypatch):
        """测试带前缀创建"""
        monkeypatch.setenv("CUSTOM_AGENT_LLM_MODEL", "gpt-4")
        monkeypatch.setenv("CUSTOM_AGENT_MAX_STEPS", "20")

        service = AgentService.from_env(prefix="CUSTOM_AGENT")

        assert service._config.max_steps == 20

    def test_from_env_defaults(self):
        """测试默认值"""
        service = AgentService.from_env()

        assert service._config.max_steps == 10  # 默认值
        assert service._config.timeout_seconds == 120


class TestAgentServiceFromConfig:
    """测试从配置文件创建服务"""

    def test_from_config_file(self, tmp_path):
        """测试从配置文件创建"""
        config_data = {
            "max_steps": 25,
            "timeout_seconds": 300,
            "debug_verbose": True,
        }

        config_file = tmp_path / "config.json"
        import json
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config_data, f)

        service = AgentService.from_config(config_file)

        assert service._config.max_steps == 25
        assert service._config.timeout_seconds == 300
        assert service._config.debug_verbose is True

    def test_from_config_file_not_found(self):
        """测试配置文件不存在"""
        with pytest.raises(FileNotFoundError):
            AgentService.from_config("/nonexistent/config.json")


class TestGetAgentService:
    """测试 get_agent_service 全局单例"""

    def test_get_singleton(self):
        """测试获取单例"""
        # 清除之前可能存在的单例
        import agent.service as service_module
        service_module._agent_service = None

        service1 = get_agent_service()
        service2 = get_agent_service()

        assert service1 is service2

    def test_singleton_is_agent_service(self):
        """测试单例类型"""
        import agent.service as service_module
        service_module._agent_service = None

        service = get_agent_service()

        assert isinstance(service, AgentService)


class TestAgentServiceEdgeCases:
    """测试 AgentService 边界情况"""

    @pytest.mark.asyncio
    async def test_arun_with_empty_query(self):
        """测试空查询"""
        mock_engine = MagicMock(spec=AgentEngine)
        mock_engine.arun = AsyncMock(return_value=AgentResponse(
            role="assistant",
            content="",
        ))

        service = AgentService(engine=mock_engine)
        response = await service.arun(user_query="")

        assert response is not None

    @pytest.mark.asyncio
    async def test_arun_with_none_history(self):
        """测试 None 历史"""
        mock_engine = MagicMock(spec=AgentEngine)
        mock_engine.arun = AsyncMock(return_value=AgentResponse(
            role="assistant",
            content="回答",
        ))

        service = AgentService(engine=mock_engine)
        response = await service.arun(
            user_query="测试",
            history=None,
        )

        assert response is not None

    @pytest.mark.asyncio
    async def test_arun_with_extra_kwargs(self):
        """测试额外参数"""
        mock_engine = MagicMock(spec=AgentEngine)
        mock_engine.arun = AsyncMock(return_value=AgentResponse(
            role="assistant",
            content="回答",
        ))

        service = AgentService(engine=mock_engine)
        response = await service.arun(
            user_query="测试",
            custom_param="value",
        )

        assert response is not None

    def test_ensure_initialized(self):
        """测试延迟初始化"""
        service = AgentService()

        # 初始状态未初始化
        assert not service._initialized

        # 触发初始化
        service._ensure_initialized()

        assert service._initialized

    def test_lazy_engine_creation(self):
        """测试延迟创建引擎"""
        service = AgentService()

        # 引擎应该为 None（延迟创建）
        assert service._engine is None

        # 访问 engine 属性触发创建
        engine = service.engine

        assert engine is not None

    @pytest.mark.asyncio
    async def test_concurrent_access(self):
        """测试并发访问"""
        import asyncio

        mock_engine = MagicMock(spec=AgentEngine)
        mock_engine.arun = AsyncMock(return_value=AgentResponse(
            role="assistant",
            content="回答",
        ))

        service = AgentService(engine=mock_engine)

        async def run_query(i):
            return await service.arun(user_query=f"查询{i}")

        results = await asyncio.gather(*[run_query(i) for i in range(5)])

        assert len(results) == 5


class TestAgentServiceLogging:
    """测试 AgentService 日志"""

    def test_init_logs(self, caplog):
        """测试初始化日志"""
        with caplog.at_level("INFO"):
            AgentService()

        assert "初始化" in caplog.text or "创建实例" in caplog.text

    @pytest.mark.asyncio
    async def test_run_logs(self, caplog):
        """测试执行日志"""
        mock_engine = MagicMock(spec=AgentEngine)
        mock_engine.arun = AsyncMock(return_value=AgentResponse(
            role="assistant",
            content="回答",
        ))

        service = AgentService(engine=mock_engine)

        with caplog.at_level("INFO"):
            await service.arun(user_query="测试")

        # 应该有执行相关的日志
