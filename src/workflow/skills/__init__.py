"""技能运行时模块（One-Stage 简化版）

提供技能加载、注册、规划、候选筛选和执行的完整功能。

主要组件：
- SkillLoader: 技能加载器，加载 SKILL.md 格式的技能
- SkillRegistry: 技能注册中心，管理技能、筛选候选、生成 Catalog
- SkillPlanner: 技能规划器，封装 Registry，提供 tool schema
- SkillExecutor: 技能执行器，统一执行 skill
- ResponseComposer: 响应组装器，组装执行结果为 LLM 内容
- SkillTool: LangChain Tool 封装，支持 Agent 模式

使用示例（Agent 模式）：
    from pathlib import Path
    from src.workflow.skills import (
        SkillRegistry, SkillExecutor, SkillTool, create_skill_tool
    )

    # 初始化
    registry = SkillRegistry()
    registry.load_from_directory(Path("domain/ad_engine"))
    executor = SkillExecutor()

    # 创建 LangChain Tool
    skill_tool = create_skill_tool(registry, executor)

    # 使用 Agent 模式调用
    result = llm_client.generate_with_agent(request, tools=[skill_tool])

使用示例（手动模式）：
    from pathlib import Path
    from src.workflow.skills import SkillRegistry, SkillPlanner, SkillExecutor, ResponseComposer

    # 初始化
    registry = SkillRegistry()
    registry.load_from_directory(Path("domain/ad_engine"))

    # 创建规划器（持有 Registry）
    planner = SkillPlanner(registry)

    # 创建执行器和组装器
    executor = SkillExecutor(external_registry)
    composer = ResponseComposer()

    # 构建 tool schema（内部自动筛选候选技能）
    tools = planner.build_tool_schema(query)

    # 调用 LLM（由业务层控制）
    response = llm.bind_tools(tools).invoke(messages)

    # 解析结果
    plan = planner.parse_tool_call(response.tool_calls[0])

    # 执行技能
    if plan.is_skill_call():
        result = executor.execute_plan(plan, query)
        llm_content = composer.compose(query, result)
        if llm_content:
            return llm.invoke(llm_content)
"""
import logging
from pathlib import Path
from typing import Any

from src.workflow.skills.base import (
    ReferenceDocument,
    SkillCallPlan,
    SkillCatalogItem,
    SkillLoadError,
    StandardSkill,
    StandardTool,
    ValidationResult,
)
from src.workflow.skills.composer import ResponseComposer
from src.workflow.skills.executor import (
    SkillExecutionResult,
    SkillExecutor,
    ToolCallRecord,
)
from src.workflow.skills.loader import SkillLoader
from src.workflow.skills.planner import SkillPlanner
from src.workflow.skills.registry import (
    CandidateSkill,
    SkillRegistry,
    get_skill_registry,
)
from src.workflow.skills.skill_tool import (
    SkillTool,
    SkillToolInput,
    create_skill_tool,
)

logger = logging.getLogger(__name__)


def initialize_skills(
    domain_root: Path,
    external_registry: Any = None,
) -> tuple[SkillRegistry, SkillPlanner, SkillExecutor]:
    """初始化技能系统

    便捷函数，完成技能加载、规划器和执行器初始化。

    Args:
        domain_root: 领域根目录
        external_registry: 外部系统注册中心（可选）

    Returns:
        (SkillRegistry, SkillPlanner, SkillExecutor) 元组
    """
    # 加载技能
    registry = SkillRegistry()
    registry.load_from_directory(domain_root)

    # 创建规划器
    planner = SkillPlanner(registry)

    # 创建执行器（不依赖 LLM client）
    executor = SkillExecutor(external_registry)

    stats = registry.get_stats()
    logger.info(
        f"[Skills] 初始化完成: "
        f"技能数={stats['total_skills']}, "
        f"关键词={stats['total_keywords']}, "
        f"模式={stats['total_patterns']}"
    )

    return registry, planner, executor


# 导出公共接口
__all__ = [
    # 加载器
    "SkillLoader",
    # 注册中心
    "SkillRegistry",
    "CandidateSkill",
    "get_skill_registry",
    # 规划器
    "SkillPlanner",
    # 执行器
    "SkillExecutor",
    "SkillExecutionResult",
    "ToolCallRecord",
    # 响应组装器
    "ResponseComposer",
    # 数据结构
    "StandardSkill",
    "StandardTool",
    "ReferenceDocument",
    # One-Stage 数据结构
    "SkillCatalogItem",
    "SkillCallPlan",
    "ValidationResult",
    # 异常
    "SkillLoadError",
    # LangChain Tool (Agent 模式)
    "SkillTool",
    "SkillToolInput",
    "create_skill_tool",
    # 便捷函数
    "initialize_skills",
]
