# -*- coding: utf-8 -*-
"""Agent Loop 公共节点模块

提供 Agent Loop 节点的基类和配置，支持：
- LLM + Skill Tool 调用
- Agent 模式（自动处理 tool calling 循环）
- 规则 Fallback 机制
- 启动时初始化

使用示例：
    from workflow.nodes.agent_loop import (
        BaseAgentLoopNode,
        AgentLoopNodeConfig,
    )

    # 1. 定义配置
    config = AgentLoopNodeConfig(
        node_name="knowledge_answer",
        response_kind="knowledge_qa",
        system_prompt_template="你是知识问答助手...",
        user_prompt_template="问题：{user_query}",
        enable_skill_tool=True,
        max_iterations=3,
    )

    # 2. 创建节点（在 WorkflowEngine 初始化时）
    node = KnowledgeAnswerNode(
        config=config,
        llm_client=llm_client,
        skill_registry=skill_registry,
        skill_executor=skill_executor,
    )

    # 3. 执行节点
    result = node.run(service, state)
"""
from __future__ import annotations

from workflow.nodes.agent_loop.config import AgentLoopNodeConfig
from workflow.nodes.agent_loop.base import BaseAgentLoopNode

__all__ = [
    # 配置类
    "AgentLoopNodeConfig",
    # 基类
    "BaseAgentLoopNode",
]
