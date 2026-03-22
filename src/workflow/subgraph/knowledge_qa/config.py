# -*- coding: utf-8 -*-
"""KnowledgeQA 子图配置

定义知识问答子图的配置类。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from workflow.subgraph.base import SubgraphConfig


@dataclass
class KnowledgeQAConfig(SubgraphConfig):
    """知识问答子图配置

    继承 SubgraphConfig，添加知识问答特定配置。

    Attributes:
        subgraph_id: 子图 ID（固定为 knowledge_qa）
        display_name: 显示名称
        system_prompt_template: 系统提示词模板
        user_prompt_template: 用户提示词模板
        system_prompt_env_key: 系统提示词环境变量 key
    """
    subgraph_id: str = "knowledge_qa"
    display_name: str = "知识问答"
    description: str = "基于检索证据的知识问答场景"

    # 提示词配置
    system_prompt_template: str = (
        "你是企业知识问答助手。"
        "必须严格基于提供的证据回答，不补充证据外事实。"
        "如果用户的问题需要查询实时数据或执行特定技能，请使用 skill_tool。"
        "输出中文，结构尽量为：结论 -> 依据。"
    )
    user_prompt_template: str = """【用户问题】
{user_query}

【当前主模块】
- module_name: {module_name}
- module_hint: {module_hint}

【相关模块】
{related_modules_block}

【检索证据（按相关性排序）】
{evidence_block}
"""
    system_prompt_env_key: str = "WORKFLOW_QA_LLM_SYSTEM_PROMPT"

    # 节点配置
    enable_skill_tool: bool = True
    max_iterations: int = 3
    require_evidence: bool = True
    enable_fallback: bool = True

    # 证据配置
    max_evidence_count: int = 10
    max_related_modules: int = 3

    # 元数据
    metadata: dict[str, Any] = field(default_factory=lambda: {
        "tags": ["qa", "knowledge"],
        "version": "1.0.0",
    })
