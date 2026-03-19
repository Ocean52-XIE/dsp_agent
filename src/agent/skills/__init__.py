# -*- coding: utf-8 -*-
"""Agent Skill 系统

提供 Skill 的加载、注册、执行和管理功能。

支持两种技能类型：
- prompt: 提示词模板渲染
- execution: CLI 命令执行

使用示例：
    from pathlib import Path
    from agent.skills import SkillRegistry, SkillExecutor, SkillManager

    # 1. 初始化并加载技能
    registry = SkillRegistry()
    registry.load_from_directory(Path("domain/ad_engine"))

    # 2. 创建执行器
    executor = SkillExecutor()

    # 3. 创建 SkillManager 并注册为工具
    skill_manager = SkillManager(registry=registry, executor=executor)
    tool_registry.register(skill_manager)

    # 4. 在 Agent Loop 中，LLM 可以调用 skill_manager 工具
"""

# 数据结构
from agent.skills.base import (
    Skill,
    SkillTrigger,
    SkillReference,
    ExecutionConfig,
    SkillCatalogItem,
)

# 加载器
from agent.skills.loader import (
    SkillLoader,
    SkillLoadError,
)

# 注册中心
from agent.skills.registry import (
    SkillRegistry,
    CandidateMatch,
    get_skill_registry,
    reset_skill_registry,
)

# 执行器
from agent.skills.executor import (
    SkillExecutor,
    SkillExecutionResult,
    ToolCallRecord,
)

# 管理工具
from agent.skills.manager import (
    SkillManager,
    SkillManagerInput,
    create_skill_manager,
)

__all__ = [
    # 数据结构
    "Skill",
    "SkillTrigger",
    "SkillReference",
    "ExecutionConfig",
    "SkillCatalogItem",
    # 加载器
    "SkillLoader",
    "SkillLoadError",
    # 注册中心
    "SkillRegistry",
    "CandidateMatch",
    "get_skill_registry",
    "reset_skill_registry",
    # 执行器
    "SkillExecutor",
    "SkillExecutionResult",
    "ToolCallRecord",
    # 管理工具
    "SkillManager",
    "SkillManagerInput",
    "create_skill_manager",
]

__version__ = "1.0.0"
