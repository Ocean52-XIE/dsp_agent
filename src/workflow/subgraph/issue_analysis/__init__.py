# -*- coding: utf-8 -*-
"""IssueAnalysis 子图模块

提供问题分析场景的子图实现。

Usage:
    from workflow.subgraph.issue_analysis import (
        IssueAnalysisConfig,
        IssueAnalysisSubgraph,
    )

    # 创建子图
    subgraph = IssueAnalysisSubgraph(
        config=IssueAnalysisConfig(),
        llm_client=llm_client,
        skill_registry=skill_registry,
        skill_executor=skill_executor,
    )

    # 执行子图
    result = subgraph.run(state)
"""
from __future__ import annotations

from workflow.subgraph.issue_analysis.config import IssueAnalysisConfig
from workflow.subgraph.issue_analysis.engine import IssueAnalysisSubgraph

__all__ = [
    "IssueAnalysisConfig",
    "IssueAnalysisSubgraph",
]
