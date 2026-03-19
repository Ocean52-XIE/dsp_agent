# -*- coding: utf-8 -*-
"""Agent 模块 - v2 通用 Agent 架构

提供通用 Agent 底座，支持：
- 动态工具调用循环
- Skill 按需加载
- MCP 工具集成
- 工具白名单控制

使用示例：
    from agent import AgentService, AgentEngine
    from agent.tools import ToolRegistry
    from agent.llm import LLMClient

    # 方式一：使用 AgentService (推荐)
    service = AgentService.from_env()
    response = await service.arun(
        user_query="查一下 CTR 预估模块",
        session_id="session-123",
    )

    # 方式二：使用 AgentEngine (高级)
    llm_client = LLMClient.from_env()
    tool_registry = ToolRegistry()

    engine = AgentEngine(
        llm_client=llm_client,
        tool_registry=tool_registry,
    )
    response = await engine.arun(
        user_query="查一下 CTR 预估模块",
    )
"""
# Engine
from agent.engine import AgentEngine, EngineConfig

# Service
from agent.service import AgentService, get_agent_service, ainit_agent_service

# State
from agent.state import (
    AgentState,
    AgentStatus,
    ToolCallRecord,
    create_initial_state,
    state_add_message,
    state_add_tool_call,
)

# Core - Router
from agent.core.router import (
    AgentRouter,
    RouterConfig,
    RouterResult,
    RouteType,
)

# Core - Loop
from agent.core.loop import (
    AgentLoop,
    AgentLoopConfig,
    AgentLoopResult,
)

# Core - Finalize
from agent.core.finalize import (
    AgentFinalize,
    FinalizeConfig,
    AgentResponse,
)

# Tools
from agent.tools.registry import ToolRegistry
from langchain_core.tools import BaseTool

# LLM
from agent.llm.client import LLMClient, LLMResponse
from agent.llm.config import LLMConfig

__all__ = [
    # Service
    "AgentService",
    "get_agent_service",
    "ainit_agent_service",
    # Engine
    "AgentEngine",
    "EngineConfig",
    # State
    "AgentState",
    "AgentStatus",
    "ToolCallRecord",
    "create_initial_state",
    "state_add_message",
    "state_add_tool_call",
    # Core - Router
    "AgentRouter",
    "RouterConfig",
    "RouterResult",
    "RouteType",
    # Core - Loop
    "AgentLoop",
    "AgentLoopConfig",
    "AgentLoopResult",
    # Core - Finalize
    "AgentFinalize",
    "FinalizeConfig",
    "AgentResponse",
    # Tools
    "ToolRegistry",
    "BaseTool",
    # LLM
    "LLMClient",
    "LLMConfig",
    "LLMResponse",
]
