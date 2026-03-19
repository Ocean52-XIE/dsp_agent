# -*- coding: utf-8 -*-
"""Agent 核心模块

包含 Agent 循环的核心组件：
- DomainRouter: 规则层路由 (闲聊拦截 + 领域门控 + 模块分流)
- AgentLoop: LLM + 工具调用循环 (LLM 自决策)
- AgentFinalize: 结果整理
"""
# 新名称 (推荐使用)
from agent.core.router import (
    DomainRouter,
    DomainRouterConfig,
    DomainRouterResult,
    DomainRouteType,
)
# 向后兼容别名
from agent.core.router import (
    AgentRouter,
    RouterConfig,
    RouterResult,
    RouteType,
)
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
    # Domain Router (新名称)
    "DomainRouter",
    "DomainRouterConfig",
    "DomainRouterResult",
    "DomainRouteType",
    # 向后兼容
    "AgentRouter",
    "RouterConfig",
    "RouterResult",
    "RouteType",
    # Loop
    "AgentLoop",
    "AgentLoopConfig",
    "AgentLoopResult",
    # Finalize
    "AgentFinalize",
    "FinalizeConfig",
    "AgentResponse",
]
