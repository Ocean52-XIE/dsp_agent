# -*- coding: utf-8 -*-
"""Agent 核心模块

包含 Agent 循环的核心组件：
- AgentLoop: LLM + 工具调用循环 (LLM 自决策)
- AgentFinalize: 结果整理

路由功能已移至 workflow 层，Agent 仅提供 LLM + Finalize 能力。
"""
from agent.core.loop import (
    AgentLoop,
    AgentLoopConfig,
    AgentLoopResult,
)
from agent.core.finalize import (
    AgentFinalize,
    FinalizeConfig,
    AgentResponse,
)

__all__ = [
    # Loop
    "AgentLoop",
    "AgentLoopConfig",
    "AgentLoopResult",
    # Finalize
    "AgentFinalize",
    "FinalizeConfig",
    "AgentResponse",
]
