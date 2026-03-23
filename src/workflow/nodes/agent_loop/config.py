# -*- coding: utf-8 -*-
"""Agent Loop 节点配置模块

提供可配置的 Agent Loop 节点行为，包括：
- 提示词模板配置
- 工具/技能约束
- 循环控制参数
- 校验和后处理函数
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


# 类型别名
AnswerValidator = Callable[[str], tuple[bool, str | None]]
AnswerNormalizer = Callable[[str], str]


@dataclass
class AgentLoopNodeConfig:
    """Agent Loop 节点配置

    用于配置节点的行为，支持：
    - 提示词模板定制
    - 工具/技能白名单
    - 循环控制参数
    - 校验和后处理

    Attributes:
        node_name: 节点名称（用于日志和调试）
        response_kind: 响应类型（如 knowledge_qa / issue_analysis）
        system_prompt_template: 系统提示词模板
        user_prompt_template: 用户提示词模板
        system_prompt_env_key: 环境变量覆盖键（可选）
        tool_whitelist: 工具白名单（为空则使用默认）
        skill_whitelist: 技能白名单（为空则使用默认）
        enable_skill_tool: 是否启用 Skill Tool
        max_iterations: Agent 模式最大迭代次数
        timeout_seconds: 超时时间（秒）
        require_evidence: 是否需要证据
        validate_answer_func: 答案校验函数（可选）
        normalize_answer_func: 答案标准化函数（可选）
        enable_fallback: 是否启用规则 Fallback
        metadata: 额外元数据
    """

    # ========== 基础配置 ==========
    node_name: str
    """节点名称

    用于日志、调试和节点追踪。
    """

    response_kind: str
    """响应类型

    标识节点产生的响应类型，如：
    - knowledge_qa: 知识问答
    - issue_analysis: 问题分析
    """

    # ========== 提示词配置 ==========
    system_prompt_template: str = ""
    """系统提示词模板

    可以包含占位符，在运行时通过 format() 替换。
    如果为空，子类应通过 _get_system_prompt() 方法动态生成。
    """

    user_prompt_template: str = ""
    """用户提示词模板

    可以包含以下占位符：
    - {user_query}: 用户问题
    - {module_name}: 当前模块名
    - {module_hint}: 模块提示
    - {related_modules_block}: 相关模块块
    - {evidence_block}: 证据块

    如果为空，子类应通过 _get_user_prompt() 方法动态生成。
    """

    system_prompt_env_key: str | None = None
    """环境变量覆盖键

    如果设置，会优先从环境变量读取系统提示词。
    格式：resolve_system_prompt(env_key, default_prompt, domain_profile)
    """

    # ========== 工具约束 ==========
    tool_whitelist: list[str] = field(default_factory=list)
    """工具白名单

    如果为空，则使用全局默认工具列表。
    """

    skill_whitelist: list[str] = field(default_factory=list)
    """技能白名单

    当 LLM 调用 skill_manager 时，会检查 skill_name 是否在白名单中。
    """

    enable_skill_tool: bool = True
    """是否启用 Skill Tool

    如果为 True，会将 SkillManager 作为工具绑定到 LLM。
    """

    # ========== 循环控制 ==========
    max_iterations: int = 3
    """Agent 模式最大迭代次数

    LLM + 工具调用的最大循环次数。
    """

    timeout_seconds: int = 60
    """超时时间（秒）

    单次节点执行的超时时间。
    """

    # ========== 证据配置 ==========
    require_evidence: bool = True
    """是否需要证据

    如果为 True，当 evidence_count <= 0 时会跳过 LLM 调用。
    """

    # ========== 校验配置 ==========
    validate_answer_func: AnswerValidator | None = None
    """答案校验函数

    函数签名：(answer: str) -> tuple[bool, str | None]
    返回 (is_valid, error_reason) 元组。
    """

    normalize_answer_func: AnswerNormalizer | None = None
    """答案标准化函数

    函数签名：(answer: str) -> str
    在返回前对答案进行标准化处理。
    """

    # ========== Fallback 配置 ==========
    enable_fallback: bool = True
    """是否启用规则 Fallback

    当 LLM 调用失败或返回无效答案时，使用规则生成兜底响应。
    """

    # ========== 元数据 ==========
    metadata: dict[str, Any] = field(default_factory=dict)
    """额外元数据

    可用于存储节点特定的配置信息。
    """

    def to_dict(self) -> dict[str, Any]:
        """转换为字典

        Returns:
            配置的字典表示
        """
        return {
            "node_name": self.node_name,
            "response_kind": self.response_kind,
            "system_prompt_env_key": self.system_prompt_env_key,
            "tool_whitelist": self.tool_whitelist,
            "skill_whitelist": self.skill_whitelist,
            "enable_skill_tool": self.enable_skill_tool,
            "max_iterations": self.max_iterations,
            "timeout_seconds": self.timeout_seconds,
            "require_evidence": self.require_evidence,
            "enable_fallback": self.enable_fallback,
            "metadata": self.metadata,
        }
