# -*- coding: utf-8 -*-
"""IssueAnalysis 子图配置

定义问题分析子图的配置类。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from workflow.subgraph.base import SubgraphConfig


@dataclass
class IssueAnalysisConfig(SubgraphConfig):
    """问题分析子图配置

    继承 SubgraphConfig，添加问题分析特定配置。

    Attributes:
        subgraph_id: 子图 ID（固定为 issue_analysis）
        display_name: 显示名称
        system_prompt_template: 系统提示词模板
        user_prompt_template: 用户提示词模板
        system_prompt_env_key: 系统提示词环境变量 key
    """
    subgraph_id: str = "issue_analysis"
    display_name: str = "问题分析"
    description: str = "基于检索证据的问题分析场景，输出结构化分析结果"

    # 提示词配置
    system_prompt_template: str = (
        "你是企业问题分析助手。"
        "必须严格基于提供的证据回答，不补充证据外事实。"
        "如果需要查询实时数据或执行特定技能，请使用 skill_tool。"
        "输出中文，优先给出结构化结果，并保留可追踪锚点。"
    )
    user_prompt_template: str = """【待分析问题】
{user_query}

【当前模块】
- module_name: {module_name}
- module_hint: {module_hint}

【相关模块】
{related_modules_block}

【检索语句】
{retrieval_queries}

【症状结构化】
{issue_context_block}

【检索证据（按相关性排序）】
{evidence_block}

请严格按以下结构输出：
根因判断：一句话
风险提示：2-3条
修复建议：2-4条
验证步骤：2-3条
"""
    system_prompt_env_key: str = "WORKFLOW_ISSUE_LLM_SYSTEM_PROMPT"

    # 节点配置
    enable_skill_tool: bool = True
    max_iterations: int = 3
    require_evidence: bool = True
    enable_fallback: bool = True

    # 证据配置
    max_evidence_count: int = 10
    max_related_modules: int = 3

    # 输出结构配置
    output_sections: list[str] = field(default_factory=lambda: [
        "root_cause",
        "risks",
        "fix_plan",
        "verification_steps",
    ])

    # 元数据
    metadata: dict[str, Any] = field(default_factory=lambda: {
        "tags": ["analysis", "issue"],
        "version": "1.0.0",
    })
