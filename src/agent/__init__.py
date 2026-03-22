# -*- coding: utf-8 -*-
"""Agent 模块 - 通用 Agent 核心组件

提供 Agent 核心组件，支持：
- 动态工具调用循环 (AgentLoop)
- 结果整理 (AgentFinalize)
- MCP 工具集成
- 工具白名单控制

请求处理入口统一在 workflow/engine.py，Agent 模块仅提供核心能力。

使用示例：
    from agent import AgentLoop, AgentFinalize
    from agent.tools import ToolRegistry
    from agent.llm import LLMClient

    # 初始化组件
    llm_client = LLMClient.from_env()
    tool_registry = ToolRegistry()

    # 创建 AgentLoop
    loop = AgentLoop(
        llm_client=llm_client,
        tool_registry=tool_registry,
    )

    # 执行 Agent 循环
    result = await loop.run(state)

    # 整理结果
    finalize = AgentFinalize()
    response = finalize.run(state, result.answer, result.tool_calls)
"""
# State
from agent.state import (
    AgentState,
    AgentStatus,
    ToolCallRecord,
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
    # State
    "AgentState",
    "AgentStatus",
    "ToolCallRecord",
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
