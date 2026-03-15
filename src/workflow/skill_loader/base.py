"""技能加载器基类和标准数据结构

本模块定义了技能加载器的基类和统一的数据结构，
用于支持多种格式的技能定义（Markdown、OpenAI JSON 等）。

主要组件：
- StandardTool: 标准化工具定义
- StandardSkill: 标准化技能定义
- BaseSkillLoader: 加载器抽象基类
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StandardTool:
    """标准化工具定义

    统一不同格式（OpenAI、LangChain、Markdown）的工具表示。

    Attributes:
        name: 工具名称，用于 LLM 调用时的标识
        description: 工具描述，帮助 LLM 理解工具用途
        parameters: 参数定义，JSON Schema 格式
        required: 必需参数列表
        handler_config: 处理器配置，包含 external_system、connector_method 等
        source_format: 来源格式标识（markdown、openai、langchain 等）
    """
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    required: list[str] = field(default_factory=list)
    handler_config: dict[str, Any] = field(default_factory=dict)
    source_format: str = "unknown"

    def to_openai_schema(self) -> dict[str, Any]:
        """转换为 OpenAI 工具 Schema 格式

        Returns:
            OpenAI 格式的工具定义
        """
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

    用于存储技能的参考文档信息，支持延迟加载内容。

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

        Returns:
            文档内容字符串
        """
        if self.content is None:
            with open(self.path, "r", encoding="utf-8") as f:
                self.content = f.read()
        return self.content


@dataclass
class StandardSkill:
    """标准化技能定义

    统一不同格式的技能表示，包含技能的所有元数据和配置。

    Attributes:
        skill_id: 技能唯一标识
        display_name: 显示名称
        description: 技能描述
        version: 版本号
        author: 作者
        tags: 标签列表
        enabled: 是否启用
        tools: 工具列表
        trigger: 触发配置
        prompt_template: 提示词模板
        execution: 执行配置
        source_format: 来源格式
        source_path: 源文件路径
    """
    skill_id: str
    display_name: str
    description: str = ""
    version: str = "1.0.0"
    author: str = ""
    tags: list[str] = field(default_factory=list)
    enabled: bool = True
    tools: list[StandardTool] = field(default_factory=list)
    trigger: dict[str, Any] = field(default_factory=dict)
    prompt_template: str = ""
    execution: dict[str, Any] = field(default_factory=dict)
    source_format: str = "unknown"
    source_path: Path | None = None

    def get_openai_tools_schema(self) -> list[dict[str, Any]]:
        """获取所有工具的 OpenAI Schema

        Returns:
            OpenAI 格式的工具 Schema 列表
        """
        return [tool.to_openai_schema() for tool in self.tools]

    def get_keywords(self) -> list[str]:
        """获取触发关键词列表

        Returns:
            关键词列表
        """
        return self.trigger.get("keywords", [])

    def get_patterns(self) -> list[str]:
        """获取触发模式列表

        Returns:
            正则表达式模式列表
        """
        return self.trigger.get("patterns", [])


@dataclass
class MarkdownSkill(StandardSkill):
    """Markdown 格式技能（扩展版）

    继承 StandardSkill，增加对工程化目录结构的支持。

    Attributes:
        scripts_path: scripts 目录路径
        references_path: references 目录路径
        references: 参考文档列表
    """
    scripts_path: Path | None = None
    references_path: Path | None = None
    references: list[ReferenceDocument] = field(default_factory=list)

    def get_handler_script_path(self) -> Path | None:
        """获取处理器脚本路径

        Returns:
            handler.py 的路径，如不存在返回 None
        """
        if self.scripts_path is None:
            return None

        handler_py = self.scripts_path / "handler.py"
        if handler_py.exists():
            return handler_py
        return None

    def load_reference(self, name: str) -> str | None:
        """加载指定参考文档

        Args:
            name: 文档名称（相对路径）

        Returns:
            文档内容，如不存在返回 None
        """
        for ref in self.references:
            if ref.name == name:
                return ref.load_content()
        return None


class BaseSkillLoader(ABC):
    """技能加载器抽象基类

    定义了技能加载器的标准接口，所有格式加载器都需要实现此接口。

    Class Attributes:
        _format_name: 格式名称标识
        _supported_extensions: 支持的文件扩展名列表
    """

    @property
    @abstractmethod
    def format_name(self) -> str:
        """格式名称标识

        Returns:
            格式名称，如 "markdown"、"openai" 等
        """
        pass

    @property
    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """支持的文件扩展名

        Returns:
            扩展名列表，如 [".md", ".markdown"]
        """
        pass

    @abstractmethod
    def can_load(self, path: Path) -> bool:
        """判断是否可以加载指定文件

        Args:
            path: 文件路径

        Returns:
            是否可以加载
        """
        pass

    @abstractmethod
    def load(self, path: Path) -> StandardSkill | list[StandardSkill]:
        """加载技能文件

        Args:
            path: 文件路径

        Returns:
            加载的技能对象，可能是单个或列表

        Raises:
            SkillLoadError: 加载失败时抛出
        """
        pass

    def _read_file(self, path: Path, encoding: str = "utf-8") -> str:
        """读取文件内容

        Args:
            path: 文件路径
            encoding: 文件编码

        Returns:
            文件内容

        Raises:
            FileNotFoundError: 文件不存在
            UnicodeDecodeError: 编码错误
        """
        with open(path, "r", encoding=encoding) as f:
            return f.read()

    def _read_json(self, path: Path) -> dict[str, Any]:
        """读取 JSON 文件

        Args:
            path: 文件路径

        Returns:
            解析后的字典
        """
        import json
        content = self._read_file(path)
        return json.loads(content)


class SkillLoadError(Exception):
    """技能加载错误

    当技能加载过程中出现问题时抛出此异常。

    Attributes:
        path: 出错的文件路径
        format_name: 加载器格式名称
        message: 错误信息
    """

    def __init__(self, path: Path, format_name: str, message: str):
        self.path = path
        self.format_name = format_name
        self.message = message
        super().__init__(f"[{format_name}] 加载 {path} 失败: {message}")
