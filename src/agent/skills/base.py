# -*- coding: utf-8 -*-
"""Skill 数据结构定义

定义 Skill 系统的核心数据结构，支持两种技能类型：
- prompt: 提示词模板渲染
- execution: 命令执行

使用示例：
    from agent.skills.base import Skill, SkillTrigger, ExecutionConfig

    # 创建触发配置
    trigger = SkillTrigger(
        keywords=["查询指标", "CTR"],
        patterns=[r"查.*指标"],
        priority=30
    )

    # 创建执行配置
    execution = ExecutionConfig(
        command=["curl", "-s", "https://api.example.com/metrics"],
        timeout=30,
        output="json"
    )

    # 创建技能
    skill = Skill(
        skill_id="query_metrics",
        display_name="指标查询",
        description="查询广告系统指标数据",
        skill_type="execution",
        trigger=trigger,
        execution=execution
    )
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Union


@dataclass
class SkillTrigger:
    """技能触发配置

    定义技能如何被触发，支持关键词匹配和正则模式匹配。

    Attributes:
        keywords: 触发关键词列表，用户查询包含任意关键词即触发
        patterns: 触发正则模式列表，用户查询匹配任意模式即触发
        priority: 优先级，数值越大优先级越高，默认 10
    """
    keywords: list[str] = field(default_factory=list)
    patterns: list[str] = field(default_factory=list)
    priority: int = 10


@dataclass
class SkillReference:
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
        """加载文档内容

        延迟加载文档内容，只在首次访问时从文件读取。

        Returns:
            文档内容字符串
        """
        if self.content is None:
            self.content = self.path.read_text(encoding="utf-8")
        return self.content


@dataclass
class ExecutionConfig:
    """执行配置（Execution Skill 专用）

    定义 Execution Skill 的命令执行配置。

    Attributes:
        command: 要执行的命令，支持字符串或数组形式
            - 字符串形式: "curl -s https://api.example.com"
            - 数组形式: ["curl", "-s", "https://api.example.com"]
            - 支持 Jinja2 模板变量: "python script.py --query '{{ query }}'"
        cwd: 工作目录，默认为 skill 目录
        env: 环境变量字典，支持 ${ENV_VAR} 和 {{ param }} 格式
        timeout: 命令执行超时时间（秒），默认 30
        output: 输出格式，支持 json/text/raw，默认 text
        extract: JSON 输出提取路径，如 "data.records"
    """
    command: Union[str, list[str]]
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    timeout: int = 30
    output: Literal["json", "text", "raw"] = "text"
    extract: str | None = None


@dataclass
class SkillCatalogItem:
    """技能目录项

    给 LLM 做技能选择的精简对象，包含技能的基本信息。

    Attributes:
        skill_id: 技能唯一标识
        display_name: 显示名称
        description: 技能描述
        skill_type: 技能类型（prompt 或 execution）
        tags: 标签列表
        keywords: 触发关键词列表
    """
    skill_id: str
    display_name: str
    description: str
    skill_type: Literal["prompt", "execution"]
    tags: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式"""
        return {
            "skill_id": self.skill_id,
            "display_name": self.display_name,
            "description": self.description,
            "skill_type": self.skill_type,
            "tags": self.tags,
            "keywords": self.keywords,
        }


@dataclass
class Skill:
    """技能定义

    支持两种类型的技能：
    - prompt: 使用 Jinja2 渲染提示词模板
    - execution: 执行 CLI 命令获取数据

    Attributes:
        skill_id: 技能唯一标识
        display_name: 显示名称
        description: 技能描述
        version: 版本号，默认 "1.0.0"
        tags: 标签列表
        enabled: 是否启用，默认 True
        skill_type: 技能类型（prompt 或 execution）
        trigger: 触发配置
        params: 参数定义，JSON Schema 格式
        prompt_template: 提示词模板（Prompt Skill 专用）
        execution: 执行配置（Execution Skill 专用）
        references: 参考文档列表
        examples: 使用示例列表
        source_path: SKILL.md 源文件路径
        references_path: references 目录路径
    """
    # === 基础信息 ===
    skill_id: str
    display_name: str
    description: str = ""
    version: str = "1.0.0"
    tags: list[str] = field(default_factory=list)
    enabled: bool = True

    # === 类型 ===
    skill_type: Literal["prompt", "execution"] = "prompt"

    # === 触发配置 ===
    trigger: SkillTrigger = field(default_factory=SkillTrigger)

    # === 参数定义 ===
    params: dict[str, Any] = field(default_factory=dict)

    # === Prompt Skill ===
    prompt_template: str = ""

    # === Execution Skill ===
    execution: ExecutionConfig | None = None

    # === 参考文档 ===
    references: list[SkillReference] = field(default_factory=list)

    # === 示例 ===
    examples: list[dict[str, Any]] = field(default_factory=list)

    # === 源文件信息 ===
    source_path: Path | None = None
    references_path: Path | None = None

    # === 便捷方法 ===

    def is_prompt(self) -> bool:
        """是否为 Prompt 类型"""
        return self.skill_type == "prompt"

    def is_execution(self) -> bool:
        """是否为 Execution 类型"""
        return self.skill_type == "execution"

    def get_keywords(self) -> list[str]:
        """获取触发关键词"""
        return self.trigger.keywords

    def get_patterns(self) -> list[str]:
        """获取触发正则模式"""
        return self.trigger.patterns

    def get_priority(self) -> int:
        """获取优先级"""
        return self.trigger.priority

    def to_catalog_item(self) -> SkillCatalogItem:
        """转换为目录项"""
        return SkillCatalogItem(
            skill_id=self.skill_id,
            display_name=self.display_name,
            description=self.description,
            skill_type=self.skill_type,
            tags=self.tags,
            keywords=self.trigger.keywords,
        )

    def get_required_params(self) -> list[str]:
        """获取必填参数列表"""
        required = []
        for param_name, param_def in self.params.items():
            if isinstance(param_def, dict) and param_def.get("required"):
                required.append(param_name)
        return required

    def validate_params(self, params: dict[str, Any]) -> tuple[bool, list[str]]:
        """校验参数

        Args:
            params: 用户提供的参数

        Returns:
            (是否通过, 缺少的必填参数列表)
        """
        required = self.get_required_params()
        missing = [p for p in required if p not in params or params[p] is None]
        return len(missing) == 0, missing
