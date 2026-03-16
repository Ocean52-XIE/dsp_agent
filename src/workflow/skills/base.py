"""技能数据结构定义（One-Stage 简化版）

本模块定义了技能和工具的统一数据结构。

主要组件：
- StandardTool: 标准化工具定义
- StandardSkill: 标准化技能定义（唯一格式）
- ReferenceDocument: 参考文档
- SkillCatalogItem: One-Stage 技能目录项
- SkillCallPlan: One-Stage 调用计划
- ValidationResult: 参数校验结果
- SkillLoadError: 加载错误异常
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StandardTool:
    """标准化工具定义

    用于技能内部工具的定义。

    Attributes:
        name: 工具名称，用于 LLM 调用时的标识
        description: 工具描述，帮助 LLM 理解工具用途
        parameters: 参数定义，JSON Schema 格式
        required: 必需参数列表
        handler_config: 处理器配置
    """
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    required: list[str] = field(default_factory=list)
    handler_config: dict[str, Any] = field(default_factory=dict)

    def to_openai_schema(self) -> dict[str, Any]:
        """转换为 OpenAI 工具 Schema 格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required,
                }
            }
        }


@dataclass
class ReferenceDocument:
    """参考文档

    用于存储技能的参考文档信息，支持延迟加载。

    Attributes:
        name: 文档名称（相对路径）
        path: 文档绝对路径
        content: 文档内容（延迟加载）
    """
    name: str
    path: Path
    content: str | None = None

    def load_content(self) -> str:
        """加载文档内容"""
        if self.content is None:
            with open(self.path, "r", encoding="utf-8") as f:
                self.content = f.read()
        return self.content


@dataclass
class StandardSkill:
    """标准化技能定义（唯一格式）

    包含技能的所有元数据和配置。

    Attributes:
        skill_id: 技能唯一标识
        display_name: 显示名称
        description: 技能描述
        version: 版本号
        author: 作者
        tags: 标签列表
        enabled: 是否启用
        skill_type: 技能类型（tool_execution, prompt_template, llm_orchestrated）
        tools: 内部工具列表
        trigger: 触发配置（keywords, patterns, priority）
        param_schema: 参数 Schema
        prompt_template: 提示词模板
        execution: 执行配置
        examples: 调用示例列表
        scripts_path: scripts 目录路径
        references_path: references 目录路径
        references: 参考文档列表
        source_path: 源文件路径
    """
    skill_id: str
    display_name: str
    description: str = ""
    version: str = "1.0.0"
    author: str = ""
    tags: list[str] = field(default_factory=list)
    enabled: bool = True
    skill_type: str = "llm_orchestrated"
    tools: list[StandardTool] = field(default_factory=list)
    trigger: dict[str, Any] = field(default_factory=dict)
    param_schema: dict[str, Any] = field(default_factory=dict)
    prompt_template: str = ""
    execution: dict[str, Any] = field(default_factory=dict)
    examples: list[dict[str, Any]] = field(default_factory=list)
    # 工程化目录支持
    scripts_path: Path | None = None
    references_path: Path | None = None
    references: list[ReferenceDocument] = field(default_factory=list)
    source_path: Path | None = None

    def get_openai_tools_schema(self) -> list[dict[str, Any]]:
        """获取所有工具的 OpenAI Schema"""
        return [tool.to_openai_schema() for tool in self.tools]

    def get_keywords(self) -> list[str]:
        """获取触发关键词列表"""
        return self.trigger.get("keywords", [])

    def get_patterns(self) -> list[str]:
        """获取触发模式列表"""
        return self.trigger.get("patterns", [])

    def get_priority(self) -> int:
        """获取触发优先级"""
        return self.trigger.get("priority", 10)

    def get_tool_mapping(self) -> dict[str, str]:
        """获取参数到工具的映射配置"""
        return self.execution.get("tool_mapping", {})

    def get_default_tool(self) -> str | None:
        """获取默认工具名称"""
        return self.execution.get("default_tool")

    def get_handler_script_path(self) -> Path | None:
        """获取处理器脚本路径（handler.py）"""
        if self.scripts_path is None:
            return None
        handler_py = self.scripts_path / "handler.py"
        return handler_py if handler_py.exists() else None

    def load_reference(self, name: str) -> str | None:
        """加载指定参考文档"""
        for ref in self.references:
            if ref.name == name:
                return ref.load_content()
        return None

    def is_tool_execution(self) -> bool:
        """是否为工具调用类技能"""
        return self.skill_type == "tool_execution"

    def is_prompt_template(self) -> bool:
        """是否为提示词类技能"""
        return self.skill_type == "prompt_template"

    def is_llm_orchestrated(self) -> bool:
        """是否为 LLM 编排类技能"""
        return self.skill_type == "llm_orchestrated"


# ============================================================================
# One-Stage Skill 数据结构
# ============================================================================

@dataclass
class SkillCatalogItem:
    """技能目录项（One-Stage）

    给 LLM 做第一轮选择的精简对象。

    Attributes:
        name: 技能名称
        description: 技能描述
        tags: 标签列表
        input_schema: 参数 Schema
        examples: 调用示例列表
    """
    name: str
    description: str
    tags: list[str] = field(default_factory=list)
    input_schema: dict[str, Any] = field(default_factory=dict)
    examples: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式"""
        return {
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "input_schema": self.input_schema,
            "examples": self.examples,
        }


@dataclass
class SkillCallPlan:
    """技能调用计划（One-Stage）

    LLM 产出的调用意图，由 SkillPlanner 解析并填充 skill 引用。

    Attributes:
        action: 行动类型（single_tool_call | need_more_info | respond）
        response: 直接回答内容（action=respond 时使用）
        missing_fields: 缺少的必填字段（action=need_more_info 时使用）
        skill_name: 要调用的技能名称（action=single_tool_call 时使用）
        input: 技能调用参数（action=single_tool_call 时使用）
        skill: 技能对象引用（由 Planner 填充，供 Executor 使用）
    """
    action: str
    response: str | None = None
    missing_fields: list[str] = field(default_factory=list)
    skill_name: str | None = None
    input: dict[str, Any] = field(default_factory=dict)
    skill: "StandardSkill | None" = None  # 由 Planner 填充

    def is_skill_call(self) -> bool:
        """是否需要调用技能（进入 Executor）

        Returns:
            True 如果需要进入 Executor 执行
        """
        return self.action == "single_tool_call" and self.skill is not None

    def needs_more_info(self) -> bool:
        """是否需要更多信息"""
        return self.action == "need_more_info"

    def is_direct_response(self) -> bool:
        """是否直接回答"""
        return self.action == "respond"

    # 兼容旧方法
    def is_tool_call(self) -> bool:
        """是否为工具调用（兼容旧代码）"""
        return self.is_skill_call()

    def is_respond(self) -> bool:
        """是否为直接回答（兼容旧代码）"""
        return self.is_direct_response()

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式"""
        return {
            "action": self.action,
            "response": self.response,
            "missing_fields": self.missing_fields,
            "skill_name": self.skill_name,
            "input": self.input,
            "skill_type": self.skill.skill_type if self.skill else None,
        }


@dataclass
class ValidationResult:
    """参数校验结果

    Attributes:
        ok: 校验是否通过
        message: 错误信息
        missing_fields: 缺少的必填字段
        invalid_fields: 类型或格式错误的字段
    """
    ok: bool
    message: str = ""
    missing_fields: list[str] = field(default_factory=list)
    invalid_fields: list[str] = field(default_factory=list)


class SkillLoadError(Exception):
    """技能加载错误

    Attributes:
        path: 出错的文件路径
        message: 错误信息
    """

    def __init__(self, path: Path, message: str):
        self.path = path
        self.message = message
        super().__init__(f"加载 {path} 失败: {message}")
