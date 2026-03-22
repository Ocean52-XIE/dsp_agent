# -*- coding: utf-8 -*-
"""Subgraph 模块

提供子图的基础类和场景实现。

子图从全局单例获取所有依赖组件（LLMClient, SkillRegistry, ToolRegistry, DomainProfile），
无需手动传递参数。

Usage:
    from workflow.subgraph import BaseSubgraph, SubgraphConfig
    from workflow.subgraph.knowledge_qa import KnowledgeQAConfig, KnowledgeQASubgraph
    from workflow.subgraph.issue_analysis import IssueAnalysisConfig, IssueAnalysisSubgraph

    # 直接创建子图实例（所有组件从全局单例获取）
    subgraph = KnowledgeQASubgraph(config=KnowledgeQAConfig())

    # 执行子图
    result = await subgraph.run(state)
"""
from __future__ import annotations

from workflow.subgraph.base import (
    BaseSubgraph,
    SubgraphConfig,
)

__all__ = [
    # 基础类
    "BaseSubgraph",
    "SubgraphConfig",
]
