# -*- coding: utf-8 -*-
"""Skill Tool - 将技能系统封装为单个 LangChain Tool

设计理念：
- 只创建一个 LangChain Tool: skill_tool
- LLM 调用 skill_tool 时指定 skill_name 和参数
- skill_tool 内部根据 skill_name 调度执行对应的 skill
- 返回执行结果给 LLM

使用示例：
    from src.workflow.skills import create_skill_tool

    # 创建 skill_tool
    skill_tool = create_skill_tool(registry, executor)

    # 绑定到 LLM
    llm_with_tools = llm.bind_tools([skill_tool])

    # LLM 调用流程（自动或手动）：
    # 1. LLM 返回 tool_calls: [{name: 'skill_tool', args: {skill_name: 'query_metrics', ...}}]
    # 2. 执行 skill_tool.invoke(args)
    # 3. 结果作为 ToolMessage 发回 LLM
    # 4. LLM 生成最终答案
"""
import json
import logging
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from src.workflow.skills.base import StandardSkill
from src.workflow.skills.executor import SkillExecutor, SkillExecutionResult
from src.workflow.skills.registry import SkillRegistry

logger = logging.getLogger(__name__)


class SkillToolInput(BaseModel):
    """skill_tool 的输入参数 Schema"""

    skill_name: str = Field(
        description="要使用的技能名称。根据用户问题选择最合适的技能。"
    )
    query: str = Field(
        description="用户的原始问题，保持原样传递。"
    )
    context: dict[str, Any] | None = Field(
        default=None,
        description="可选的上下文参数，从用户问题中提取的关键信息（如日期、指标名、实体ID等）。"
    )


class SkillTool(BaseTool):
    """技能 LangChain Tool

    将整个技能系统封装为单个 LangChain Tool。
    LLM 通过调用此工具来执行技能，获取结构化数据或渲染后的提示词。

    Attributes:
        name: 工具名称，固定为 "skill_tool"
        description: 工具描述，动态生成（包含可用技能列表）
        registry: 技能注册中心
        executor: 技能执行器
        candidate_skill_ids: 候选技能 ID 列表（用于筛选）
    """

    # LangChain Tool 固定属性
    name: str = "skill_tool"
    args_schema: type[BaseModel] = SkillToolInput

    # 自定义属性
    registry: SkillRegistry | None = None
    executor: SkillExecutor | None = None
    candidate_skill_ids: list[str] = Field(default_factory=list)

    # 类级别配置
    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        registry: SkillRegistry | None = None,
        executor: SkillExecutor | None = None,
        candidate_skill_ids: list[str] | None = None,
        **kwargs,
    ):
        """初始化 Skill Tool

        Args:
            registry: 技能注册中心
            executor: 技能执行器
            candidate_skill_ids: 候选技能 ID 列表（用于筛选可用的技能）
            **kwargs: 传递给 BaseTool 的其他参数
        """
        # 先设置自定义属性
        kwargs["registry"] = registry
        kwargs["executor"] = executor
        kwargs["candidate_skill_ids"] = candidate_skill_ids or []

        # 动态生成描述
        kwargs["description"] = "执行技能工具。根据用户需求选择合适的技能执行。"

        super().__init__(**kwargs)

        logger.info(
            f"[SkillTool] 初始化完成, "
            f"候选技能数: {len(self.candidate_skill_ids)}"
        )

    def _generate_description(self) -> str:
        """生成包含可用技能列表的描述

        Returns:
            完整的工具描述
        """
        base_desc = "执行技能工具。根据用户需求选择合适的技能执行。\n\n可用技能：\n"

        if not self.registry:
            return base_desc + "- 无可用技能"

        # 获取候选技能
        if self.candidate_skill_ids:
            skills = [
                self.registry.get_skill(skill_id)
                for skill_id in self.candidate_skill_ids
            ]
            skills = [s for s in skills if s is not None]
        else:
            skills = list(self.registry.skills.values())

        # 构建技能列表
        skill_lines = []
        for skill in skills[:10]:  # 最多显示 10 个
            desc_preview = skill.description[:80]
            if len(skill.description) > 80:
                desc_preview += "..."
            skill_lines.append(f"- {skill.skill_id}: {desc_preview}")

        return base_desc + "\n".join(skill_lines)

    def update_candidates(self, candidate_skill_ids: list[str]) -> None:
        """更新候选技能列表

        Args:
            candidate_skill_ids: 新的候选技能 ID 列表
        """
        self.candidate_skill_ids = candidate_skill_ids
        # 更新描述
        self.description = self._generate_description()
        logger.debug(
            f"[SkillTool] 更新候选技能: {candidate_skill_ids}"
        )

    def _run(
        self,
        skill_name: str,
        query: str,
        context: dict[str, Any] | None = None,
        **kwargs,
    ) -> str:
        """同步执行技能（LangChain Tool 入口）

        Args:
            skill_name: 技能名称
            query: 用户原始问题
            context: 上下文参数
            **kwargs: 其他参数

        Returns:
            执行结果（JSON 字符串格式，供 LLM 理解）
        """
        logger.info(
            f"[SkillTool] 执行技能: {skill_name}, "
            f"query={query[:50]}..."
        )

        # 1. 检查 registry 和 executor
        if self.registry is None or self.executor is None:
            error_msg = "Skill 组件未初始化（registry 或 executor 为空）"
            logger.error(f"[SkillTool] {error_msg}")
            return json.dumps({"success": False, "error": error_msg}, ensure_ascii=False)

        # 2. 获取技能
        skill = self.registry.get_skill(skill_name)
        if skill is None:
            error_msg = f"技能不存在: {skill_name}"
            logger.warning(f"[SkillTool] {error_msg}")
            return json.dumps(
                {"success": False, "error": error_msg, "available_skills": list(self.registry.skills.keys())},
                ensure_ascii=False,
            )

        # 3. 构建执行参数
        from src.workflow.skills.base import SkillCallPlan
        plan = SkillCallPlan(
            action="single_tool_call",
            skill_name=skill_name,
            input=context or {},
            skill=skill,
        )

        # 4. 执行技能
        try:
            result = self.executor.execute_plan(plan, query)
            return self._format_result(result, skill)
        except Exception as e:
            logger.error(f"[SkillTool] 执行失败: {e}")
            return json.dumps(
                {"success": False, "error": str(e), "skill_name": skill_name},
                ensure_ascii=False,
            )

    def _format_result(self, result: SkillExecutionResult, skill: StandardSkill) -> str:
        """格式化执行结果为 LLM 可理解的字符串

        Args:
            result: 技能执行结果
            skill: 技能对象

        Returns:
            JSON 字符串格式的结果
        """
        if not result.success:
            return json.dumps(
                {
                    "success": False,
                    "error": result.error,
                    "skill_name": result.skill_id,
                },
                ensure_ascii=False,
            )

        # 根据技能类型返回不同的结果格式
        if result.has_data():
            # tool_execution: 返回数据
            return json.dumps(
                {
                    "success": True,
                    "skill_name": result.skill_id,
                    "skill_type": "tool_execution",
                    "data": result.data,
                },
                ensure_ascii=False,
                indent=2,
            )
        elif result.has_prompt():
            # prompt_template: 返回渲染后的提示词
            # 注意：对于 prompt_template，返回的 prompt 应该作为 LLM 的输入
            # 但在 Tool 模式下，我们将其作为结果返回给 LLM
            return json.dumps(
                {
                    "success": True,
                    "skill_name": result.skill_id,
                    "skill_type": "prompt_template",
                    "prompt": result.prompt,
                },
                ensure_ascii=False,
            )
        else:
            return json.dumps(
                {
                    "success": True,
                    "skill_name": result.skill_id,
                    "message": "技能执行成功，但无返回数据",
                },
                ensure_ascii=False,
            )

    async def _arun(
        self,
        skill_name: str,
        query: str,
        context: dict[str, Any] | None = None,
        **kwargs,
    ) -> str:
        """异步执行技能（暂不支持，降级为同步）

        Args:
            skill_name: 技能名称
            query: 用户原始问题
            context: 上下文参数
            **kwargs: 其他参数

        Returns:
            执行结果
        """
        # 暂不支持异步，降级为同步
        return self._run(skill_name, query, context, **kwargs)

    def get_openai_tool_schema(self) -> dict[str, Any]:
        """获取 OpenAI 格式的 Tool Schema

        Returns:
            OpenAI 格式的工具定义
        """
        # 获取 Pydantic schema
        args_schema = self.args_schema.model_json_schema()

        # 构建 OpenAI 格式
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self._generate_description(),
                "parameters": args_schema,
            },
        }


def create_skill_tool(
    registry: SkillRegistry,
    executor: SkillExecutor,
    candidate_skill_ids: list[str] | None = None,
) -> SkillTool:
    """创建 Skill Tool 的便捷函数

    Args:
        registry: 技能注册中心
        executor: 技能执行器
        candidate_skill_ids: 候选技能 ID 列表

    Returns:
        SkillTool 实例
    """
    return SkillTool(
        registry=registry,
        executor=executor,
        candidate_skill_ids=candidate_skill_ids,
    )
