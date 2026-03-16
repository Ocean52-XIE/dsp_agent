"""Skill Planner - 技能规划器

封装 SkillRegistry，业务层只和 SkillPlanner 交互。

职责：
- 构建 OpenAI/LangChain 兼容的 tool schema
- 解析 LLM tool_call 为 SkillCallPlan
- 获取技能定义

不负责：
- 调用 LLM（由业务层控制）

使用示例：
    # 应用启动时初始化（全局唯一）
    registry = SkillRegistry()
    registry.load_from_directory(Path("domain/ad_engine"))
    planner = SkillPlanner(registry)

    # 业务层使用
    tools = planner.build_tool_schema(query)
    response = llm.bind_tools(tools).invoke(messages)
    plan = planner.parse_tool_call(response.tool_calls[0])
"""
import logging
from typing import Any

from src.workflow.skills.base import (
    SkillCatalogItem,
    SkillCallPlan,
    StandardSkill,
)
from src.workflow.skills.registry import SkillRegistry

logger = logging.getLogger(__name__)


class SkillPlanner:
    """技能规划器

    封装 SkillRegistry，业务层只和 SkillPlanner 交互。

    Attributes:
        TOOL_NAME: 工具名称常量
        _registry: SkillRegistry 实例（全局唯一）
    """

    TOOL_NAME = "invoke_skill"

    def __init__(self, registry: SkillRegistry):
        """初始化

        Args:
            registry: 全局唯一的 SkillRegistry 实例
        """
        self._registry = registry
        logger.info(f"[SkillPlanner] 初始化完成，已注册 {len(registry.skills)} 个技能")

    def build_tool_schema(
        self,
        query: str,
        max_candidates: int = 5,
    ) -> list[dict[str, Any]]:
        """构建 tool schema

        内部自动完成：
        1. 根据 query 筛选候选技能
        2. 获取候选技能的 catalog
        3. 构建 OpenAI/LangChain 兼容的 tool schema

        Args:
            query: 用户查询
            max_candidates: 最大候选数量，默认 5

        Returns:
            OpenAI 格式的 tools 列表
        """
        # 1. 筛选候选技能
        candidates = self._registry.get_candidates(query, max_candidates=max_candidates)

        if not candidates:
            logger.warning(f"[SkillPlanner] 无候选技能: query='{query[:50]}...'")
            return self._build_empty_tool_schema()

        # 2. 获取 catalog
        skill_names = [s.skill_id for s in candidates]
        catalog = self._registry.get_catalog(skill_names)

        logger.info(
            f"[SkillPlanner] 构建 tool schema: "
            f"query='{query[:30]}...', 候选={[s.skill_id for s in candidates]}"
        )

        # 3. 构建 tool schema
        return self._build_tool_schema(catalog)

    def build_tool_schema_all(self) -> list[dict[str, Any]]:
        """构建包含所有技能的 tool schema

        不进行候选筛选，包含所有已注册的技能。
        适用于技能数量较少或需要让 LLM 自行选择的场景。

        Returns:
            OpenAI 格式的 tools 列表
        """
        catalog = self._registry.get_catalog()

        if not catalog:
            logger.warning("[SkillPlanner] 无已注册技能")
            return self._build_empty_tool_schema()

        logger.info(f"[SkillPlanner] 构建全量 tool schema: {len(catalog)} 个技能")
        return self._build_tool_schema(catalog)

    def parse_tool_call(self, tool_call: dict[str, Any]) -> SkillCallPlan:
        """解析 LLM tool_call 为 SkillCallPlan

        解析时自动填充 skill 引用，供 Executor 使用。

        Args:
            tool_call: LangChain/OpenAI 的 tool_call 对象
                格式: {"name": "invoke_skill", "args": {...}, "id": "..."}

        Returns:
            SkillCallPlan 调用计划（包含完整 skill 引用）
        """
        try:
            # 检查工具名称
            if tool_call.get("name") != self.TOOL_NAME:
                logger.warning(f"[SkillPlanner] 未知 tool: {tool_call.get('name')}")
                return SkillCallPlan(
                    action="respond",
                    response="抱歉，无法处理该请求。",
                )

            # 获取参数
            args = tool_call.get("args", {})
            action = args.get("action", "single_tool_call")

            # 验证 action
            valid_actions = ["single_tool_call", "need_more_info", "respond"]
            if action not in valid_actions:
                logger.warning(f"[SkillPlanner] 无效 action: {action}, 使用 respond")
                action = "respond"

            # 构建基础 plan
            plan = SkillCallPlan(
                action=action,
                response=args.get("response"),
                missing_fields=args.get("missing_fields", []),
                skill_name=args.get("skill_name"),
                input=args.get("input", {}),
            )

            # 关键：如果是 single_tool_call，填充 skill 引用
            if action == "single_tool_call" and plan.skill_name:
                plan.skill = self._registry.get_skill(plan.skill_name)

                if plan.skill is None:
                    # skill 不存在，降级为 respond
                    logger.warning(f"[SkillPlanner] skill 不存在: {plan.skill_name}")
                    plan.action = "respond"
                    plan.response = f"抱歉，技能 {plan.skill_name} 不存在。"

            logger.info(
                f"[SkillPlanner] 解析 tool_call: action={plan.action}, "
                f"skill={plan.skill_name}, skill_type={plan.skill.skill_type if plan.skill else None}"
            )

            return plan

        except Exception as e:
            logger.error(f"[SkillPlanner] 解析失败: {e}")
            return SkillCallPlan(
                action="respond",
                response="抱歉，请求解析失败。",
            )

    def get_skill(self, skill_name: str) -> StandardSkill | None:
        """获取技能定义

        供执行器使用，根据技能名称获取完整定义。

        Args:
            skill_name: 技能名称

        Returns:
            StandardSkill 或 None
        """
        return self._registry.get_skill(skill_name)

    def get_registry_stats(self) -> dict[str, Any]:
        """获取 Registry 统计信息

        Returns:
            统计信息字典
        """
        return self._registry.get_stats()

    def _build_tool_schema(self, catalog: list[SkillCatalogItem]) -> list[dict[str, Any]]:
        """构建 OpenAI tool schema

        Args:
            catalog: 技能 catalog 列表

        Returns:
            OpenAI 格式的 tools 列表
        """
        skill_names = [item.name for item in catalog]

        # 构建技能描述
        skill_descriptions = []
        for item in catalog:
            desc = f"- **{item.name}**: {item.description}"
            if item.tags:
                desc += f" (标签: {', '.join(item.tags)})"
            skill_descriptions.append(desc)

        return [
            {
                "type": "function",
                "function": {
                    "name": self.TOOL_NAME,
                    "description": "调用技能执行特定任务。\n\n可用技能：\n" + "\n".join(skill_descriptions),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["single_tool_call", "need_more_info", "respond"],
                                "description": "操作类型：single_tool_call=调用技能，need_more_info=需要更多信息，respond=直接回答",
                            },
                            "skill_name": {
                                "type": "string",
                                "enum": skill_names,
                                "description": "技能名称（action=single_tool_call 时必填）",
                            },
                            "input": {
                                "type": "object",
                                "description": "技能参数（action=single_tool_call 时必填）",
                            },
                            "response": {
                                "type": "string",
                                "description": "直接回答内容（action=respond 时使用）",
                            },
                            "missing_fields": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "缺少的字段（action=need_more_info 时使用）",
                            },
                        },
                        "required": ["action"],
                    },
                },
            }
        ]

    def _build_empty_tool_schema(self) -> list[dict[str, Any]]:
        """构建空技能的 tool schema

        当没有候选技能时返回，限制 LLM 只能直接回答。

        Returns:
            OpenAI 格式的 tools 列表
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": self.TOOL_NAME,
                    "description": "当前没有可用的技能，请直接回答用户问题。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["respond"],
                                "description": "只能选择 respond",
                            },
                            "response": {
                                "type": "string",
                                "description": "直接回答内容",
                            },
                        },
                        "required": ["action", "response"],
                    },
                },
            }
        ]
