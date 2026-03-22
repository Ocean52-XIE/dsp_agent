# -*- coding: utf-8 -*-
"""KnowledgeQA 子图模块

提供知识问答场景的子图实现。

Usage:
    from workflow.subgraph.knowledge_qa import (
        KnowledgeQAConfig,
        KnowledgeQASubgraph,
    )

    # 创建子图
    subgraph = KnowledgeQASubgraph(
        config=KnowledgeQAConfig(),
        llm_client=llm_client,
        skill_registry=skill_registry,
        skill_executor=skill_executor,
    )

    # 执行子图
    result = subgraph.run(state)
"""
from __future__ import annotations

from workflow.subgraph.knowledge_qa.config import KnowledgeQAConfig
from workflow.subgraph.knowledge_qa.engine import KnowledgeQASubgraph

__all__ = [
    "KnowledgeQAConfig",
    "KnowledgeQASubgraph",
]
