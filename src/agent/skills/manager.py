# -*- coding: utf-8 -*-
"""Skill 管理工具

将 Skill 系统封装为 LangChain Tool，供 Agent Loop 调用。

设计理念：
- 只创建一个 LangChain Tool: skill_manager
- LLM 调用 skill_manager 时指定 skill_name 和参数
- skill_manager 内部根据 skill_name 调度执行对应的技能
- 返回执行结果给 LLM

使用示例：
    from agent.skills import SkillRegistry, SkillExecutor, SkillManager

    # 创建组件
    registry = SkillRegistry()
    registry.load_from_directory(Path("domain/ad_engine"))
    executor = SkillExecutor()

    # 创建 SkillManager
    skill_manager = SkillManager(registry=registry, executor=executor)

    # 注册到 ToolRegistry
    tool_registry.register(skill_manager)

    # LLM 调用流程（自动或手动）：
    # 1. LLM 返回 tool_calls: [{name: 'skill_manager', args: {skill_name: 'query_metrics', ...}}]
    # 2. 执行 skill_manager.invoke(args)
    # 3. 结果作为 ToolMessage 发回 LLM
    # 4. LLM 生成最终答案
"""

import json
import logging
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from agent.skills.executor import SkillExecutor, SkillExecutionResult
from agent.skills.registry import SkillRegistry

logger = logging.getLogger(__name__)


class SkillManagerInput(BaseModel):
    """SkillManager 输入参数 Schema"""

    skill_name: str = Field(
        description=(
            "要使用的技能名称。根据用户问题选择最合适的技能。"
            "可用的技能名称见工具描述中的【可用技能】列表。"
        )
    )
    query: str = Field(
        description="用户的原始问题，保持原样传递。"
    )
    params: dict[str, Any] | None = Field(
        default=None,
        description=(
            "技能参数，从用户问题中提取的关键信息。"
            "例如：指标类型、时间范围、产品名称等。"
        )
    )


class SkillManager(BaseTool):
    """技能管理工具

    将整个技能系统封装为单个 LangChain Tool。
    LLM 通过调用此工具来执行技能，获取提示词或结构化数据。

    Attributes:
        name: 工具名称，固定为 "skill_manager"
        description: 工具描述，动态生成（包含可用技能列表）
        registry: 技能注册中心
        executor: 技能执行器
        candidate_skill_ids: 候选技能 ID 列表（用于筛选）
    """

    # LangChain Tool 固定属性
    name: str = "skill_manager"
    args_schema: type[BaseModel] = SkillManagerInput

    # 自定义属性
    registry: SkillRegistry | None = None
    executor: SkillExecutor | None = None
    candidate_skill_ids: list[str] = Field(default_factory=list)

    # Pydantic 配置
    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        registry: SkillRegistry | None = None,
        executor: SkillExecutor | None = None,
        candidate_skill_ids: list[str] | None = None,
        **kwargs,
    ):
        """初始化 SkillManager

        Args:
            registry: 技能注册中心
            executor: 技能执行器
            candidate_skill_ids: 候选技能 ID 列表（用于筛选可用的技能）
            **kwargs: 传递给 BaseTool 的其他参数
        """
        # 设置自定义属性
        kwargs["registry"] = registry
        kwargs["executor"] = executor
        kwargs["candidate_skill_ids"] = candidate_skill_ids or []

        # 动态生成描述（包含可用技能列表）
        kwargs["description"] = self._build_description_static(
            registry, candidate_skill_ids or []
        )

        super().__init__(**kwargs)

        logger.info(
            f"[SkillManager] 初始化完成, "
            f"候选技能数: {len(self.candidate_skill_ids)}, "
            f"description_length={len(self.description)}"
        )

    @staticmethod
    def _build_description_static(
        registry: SkillRegistry | None,
        candidate_skill_ids: list[str],
    ) -> str:
        """静态方法生成包含可用技能列表的描述

        设计原则：
        - 保持通用性，不硬编码领域特定的提示词
        - 从技能元数据（keywords）动态生成使用场景提示
        - 包含参数说明，让 LLM 知道如何传参
        - 使用强指令性语言确保 LLM 理解何时调用

        Args:
            registry: 技能注册中心
            candidate_skill_ids: 候选技能 ID 列表

        Returns:
            完整的工具描述
        """
        if not registry:
            return "执行技能工具。当前无可用技能。"

        # 获取候选技能
        if candidate_skill_ids:
            skills = [
                registry.get_skill(skill_id)
                for skill_id in candidate_skill_ids
            ]
            skills = [s for s in skills if s is not None]
        else:
            skills = list(registry.skills.values())

        if not skills:
            return "执行技能工具。当前无可用技能。"

        # 从所有技能中聚合关键词，构建"使用场景"提示
        all_keywords: set[str] = set()
        skill_details: list[str] = []

        for skill in skills[:10]:  # 最多显示 10 个
            # 聚合关键词
            keywords = skill.get_keywords()
            all_keywords.update(keywords[:5])

            # 技能详情
            type_icon = "📄" if skill.is_prompt() else "⚡"
            desc_preview = skill.description[:50]
            if len(skill.description) > 50:
                desc_preview += "..."

            # 构建参数说明
            params_hint = SkillManager._build_params_hint(skill)

            keyword_str = f"触发词: {', '.join(keywords[:3])}" if keywords else ""
            skill_block = f"""【{type_icon} {skill.skill_id}】
  描述: {desc_preview}
  类型: {skill.skill_type}{f" | {keyword_str}" if keyword_str else ""}
{params_hint}"""
            skill_details.append(skill_block)

        # 构建关键词提示（取前 15 个最常见的）
        keyword_list = sorted(all_keywords, key=len)[:15]
        keyword_hint = "、".join(keyword_list) if keyword_list else "见下方技能列表"

        # 构建完整描述（强指令性语言 + 参数说明）
        desc = f"""执行技能工具。当用户问题涉及以下场景时，**必须**使用此工具。

【触发关键词】
{keyword_hint}

【调用规则】
1. 根据用户问题选择合适的 skill_name
2. **必须**从用户问题中提取参数，通过 params 字段传递
3. 如果用户问题中包含具体数值（如"5条"、"最近3次"），必须提取为对应参数

【可用技能】
""" + "\n\n".join(skill_details)

        return desc

    @staticmethod
    def _build_params_hint(skill) -> str:
        """构建技能参数说明

        Args:
            skill: 技能对象

        Returns:
            参数说明字符串
        """
        if not skill.params:
            return "  参数: 无"

        params_lines = ["  参数:"]
        for param_name, param_def in skill.params.items():
            if isinstance(param_def, dict):
                param_type = param_def.get("type", "any")
                param_desc = param_def.get("description", "")
                # 截断描述
                if len(param_desc) > 40:
                    param_desc = param_desc[:40] + "..."
                # 枚举值
                enum_values = param_def.get("enum", [])
                enum_str = f" [可选值: {', '.join(map(str, enum_values))}]" if enum_values else ""
                # 默认值
                default_val = param_def.get("default")
                default_str = f" (默认: {default_val})" if default_val is not None else ""
                # 必填
                required = "必填" if param_def.get("required") else "可选"

                params_lines.append(
                    f"    - {param_name}({param_type}, {required}): {param_desc}{enum_str}{default_str}"
                )
            else:
                params_lines.append(f"    - {param_name}: {param_def}")

        return "\n".join(params_lines)

    def to_openai_schema(self) -> dict[str, Any]:
        """生成 OpenAI 兼容的工具 Schema

        由于 SkillManager 继承自 LangChain BaseTool，
        需要实现此方法以兼容 AgentLoop 的工具调用接口。

        Returns:
            OpenAI 格式的工具定义字典
        """
        # 使用 LangChain 的 tool_call_schema 获取参数 schema
        try:
            parameters = self.tool_call_schema.model_json_schema()
        except Exception:
            # 回退到手动构建
            parameters = {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "要使用的技能名称",
                    },
                    "query": {
                        "type": "string",
                        "description": "用户的原始问题",
                    },
                    "params": {
                        "type": "object",
                        "description": "技能参数",
                    },
                },
                "required": ["skill_name", "query"],
            }

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            }
        }

    def _run(
        self,
        skill_name: str,
        query: str,
        params: dict[str, Any] | None = None,
        **kwargs,
    ) -> str:
        """同步执行技能（LangChain Tool 入口）

        Args:
            skill_name: 技能名称
            query: 用户原始问题
            params: 技能参数
            **kwargs: 其他参数

        Returns:
            执行结果（JSON 字符串格式，供 LLM 理解）
        """
        logger.info(
            f"[SkillManager] 执行技能: {skill_name}, "
            f"query={query[:50]}..."
        )

        # 1. 检查 registry 和 executor
        if self.registry is None or self.executor is None:
            error_msg = "Skill 组件未初始化（registry 或 executor 为空）"
            logger.error(f"[SkillManager] {error_msg}")
            return json.dumps({"success": False, "error": error_msg}, ensure_ascii=False)

        # 2. 获取技能（支持大小写不敏感匹配）
        skill = self._resolve_skill(skill_name)
        if skill is None:
            error_msg = f"技能不存在: {skill_name}"
            logger.warning(f"[SkillManager] {error_msg}")
            return json.dumps(
                {
                    "success": False,
                    "error": error_msg,
                    "available_skills": list(self.registry.skills.keys()),
                },
                ensure_ascii=False,
            )

        # 记录实际使用的技能ID（用于调试）
        if skill.skill_id != skill_name:
            logger.info(f"[SkillManager] 技能名称解析: '{skill_name}' -> '{skill.skill_id}'")

        # 3. 执行技能
        try:
            result = self.executor.execute(skill, params or {}, query)
            formatted_result = self._format_result(result)

            # 记录返回给 LLM 的工具调用结果日志
            result_preview = formatted_result[:500] if len(formatted_result) > 500 else formatted_result
            logger.info(
                f"[SkillManager] 工具调用结果返回给LLM: skill={skill.skill_id}, "
                f"success={result.success}, result_length={len(formatted_result)}, "
                f"result_preview={result_preview}"
            )

            return formatted_result

        except Exception as e:
            logger.error(f"[SkillManager] 执行失败: {e}")
            return json.dumps(
                {"success": False, "error": str(e), "skill_name": skill.skill_id},
                ensure_ascii=False,
            )

    def _resolve_skill(self, skill_name: str):
        """解析技能名称，支持大小写不敏感匹配

        Args:
            skill_name: LLM 传入的技能名称

        Returns:
            匹配到的技能对象，未找到返回 None
        """
        if not self.registry:
            return None

        skill_name_lower = skill_name.lower().strip()

        # 1. 精确匹配技能ID（区分大小写）
        skill = self.registry.get_skill(skill_name)
        if skill:
            return skill

        # 2. 忽略大小写精确匹配技能ID
        for s in self.registry.skills.values():
            if s.skill_id.lower() == skill_name_lower:
                return s

        # 未找到匹配
        logger.warning(f"[SkillManager] 技能不存在: '{skill_name}'")
        return None

    def _format_result(self, result: SkillExecutionResult) -> str:
        """格式化执行结果为 LLM 可理解的字符串

        Args:
            result: 技能执行结果

        Returns:
            JSON 字符串格式的结果
        """
        if not result.success:
            return json.dumps(
                {
                    "success": False,
                    "error": result.error,
                    "skill_id": result.skill_id,
                },
                ensure_ascii=False,
            )

        # 根据技能类型返回不同的结果格式
        if result.has_prompt():
            # Prompt Skill: 返回渲染后的提示词
            return json.dumps(
                {
                    "success": True,
                    "skill_id": result.skill_id,
                    "skill_type": "prompt",
                    "prompt": result.prompt,
                },
                ensure_ascii=False,
                indent=2,
            )
        elif result.has_data():
            # Execution Skill: 返回数据
            return json.dumps(
                {
                    "success": True,
                    "skill_id": result.skill_id,
                    "skill_type": "execution",
                    "data": result.data,
                },
                ensure_ascii=False,
                indent=2,
            )
        else:
            return json.dumps(
                {
                    "success": True,
                    "skill_id": result.skill_id,
                    "message": "技能执行成功，但无返回数据",
                },
                ensure_ascii=False,
            )


def create_skill_manager(
    registry: SkillRegistry,
    executor: SkillExecutor | None = None,
    candidate_skill_ids: list[str] | None = None,
) -> SkillManager:
    """创建 SkillManager 的便捷函数

    Args:
        registry: 技能注册中心
        executor: 技能执行器（可选，不传则自动创建）
        candidate_skill_ids: 候选技能 ID 列表

    Returns:
        SkillManager 实例
    """
    if executor is None:
        executor = SkillExecutor()

    return SkillManager(
        registry=registry,
        executor=executor,
        candidate_skill_ids=candidate_skill_ids,
    )
