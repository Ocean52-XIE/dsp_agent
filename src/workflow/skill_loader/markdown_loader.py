"""Markdown Skill 格式加载器

支持主流的 Markdown + YAML front matter 格式技能定义。

格式规范:
- 文件名: SKILL.md (推荐) 或任意 .md 文件
- 内容结构: YAML front matter + Markdown 提示词模板
- 目录结构:
  skill-name/
    SKILL.md           # 技能定义（必需）
    scripts/           # 执行脚本（可选）
      handler.py
    references/        # 参考文档（可选）
      guide.md

YAML Front Matter 示例:
---
name: query_metrics
version: 1.0.0
description: 查询监控指标
trigger:
  type: keyword
  keywords: [指标, 监控]
tools:
  - name: query_prometheus
    description: 查询 Prometheus
    parameters:
      promql:
        type: string
        description: PromQL 查询语句
        required: true
    external_system: prometheus
---

Markdown 内容作为提示词模板...
"""
import logging
import re
from pathlib import Path
from typing import Any

import yaml

from src.workflow.skill_loader.base import (
    BaseSkillLoader,
    MarkdownSkill,
    ReferenceDocument,
    SkillLoadError,
    StandardTool,
)

logger = logging.getLogger(__name__)


class MarkdownSkillLoader(BaseSkillLoader):
    """Markdown Skill 格式加载器

    支持解析 SKILL.md 文件，提取 YAML front matter 中的元数据和工具定义，
    以及 Markdown 内容作为提示词模板。

    Features:
    - YAML front matter 解析
    - Markdown 提示词提取
    - scripts 目录扫描
    - references 目录索引
    """

    # YAML Front Matter 正则模式
    # 匹配 --- 开头和结尾的 YAML 块，以及后续的 Markdown 内容
    FRONT_MATTER_PATTERN = re.compile(
        r'^---\s*\n(.*?)\n---\s*\n(.*)$',
        re.DOTALL
    )

    # 支持的文件名（不区分大小写）
    SKILL_FILE_NAMES = ["SKILL.md", "skill.md", "SKILL", "skill"]

    @property
    def format_name(self) -> str:
        """格式名称"""
        return "markdown"

    @property
    def supported_extensions(self) -> list[str]:
        """支持的文件扩展名"""
        return [".md", ".markdown"]

    def can_load(self, path: Path) -> bool:
        """判断是否可以加载指定文件

        检查规则：
        1. 文件扩展名为 .md 或 .markdown
        2. 文件名匹配 SKILL.md（不区分大小写）或包含 YAML front matter

        Args:
            path: 文件路径

        Returns:
            是否可以加载
        """
        # 检查扩展名
        if path.suffix.lower() not in self.supported_extensions:
            return False

        # 检查是否是 SKILL.md 文件
        if path.name.upper() in ["SKILL.MD", "SKILL"]:
            return True

        # 检查是否包含 YAML front matter
        try:
            content = self._read_file(path)
            return content.strip().startswith("---")
        except Exception:
            return False

    def load(self, path: Path) -> MarkdownSkill:
        """加载 Markdown 格式的技能文件

        Args:
            path: SKILL.md 文件路径

        Returns:
            MarkdownSkill 对象

        Raises:
            SkillLoadError: 加载失败时抛出
        """
        try:
            # 读取文件内容
            content = self._read_file(path)

            # 解析 YAML front matter 和 Markdown 内容
            front_matter, markdown_content = self._parse_front_matter(content)

            # 获取技能目录（SKILL.md 所在目录）
            skill_dir = path.parent

            # 构建技能定义
            skill = MarkdownSkill(
                skill_id=self._extract_skill_id(front_matter, skill_dir),
                display_name=front_matter.get("name", skill_dir.name),
                description=front_matter.get("description", ""),
                version=front_matter.get("version", "1.0.0"),
                author=front_matter.get("author", ""),
                tags=front_matter.get("tags", []),
                enabled=front_matter.get("enabled", True),
                tools=self._parse_tools(front_matter.get("tools", [])),
                trigger=front_matter.get("trigger", {}),
                prompt_template=markdown_content,
                execution=front_matter.get("execution", {}),
                source_format="markdown",
                source_path=path,
                scripts_path=self._get_scripts_path(skill_dir),
                references_path=self._get_references_path(skill_dir),
            )

            # 加载参考文档索引
            if skill.references_path:
                skill.references = self._load_references_index(skill.references_path)

            logger.info(
                f"[MarkdownLoader] 成功加载技能: {skill.skill_id}, "
                f"工具数={len(skill.tools)}, 参考文档数={len(skill.references)}"
            )

            return skill

        except SkillLoadError:
            raise
        except Exception as e:
            raise SkillLoadError(path, self.format_name, str(e)) from e

    def _parse_front_matter(self, content: str) -> tuple[dict[str, Any], str]:
        """解析 YAML front matter 和 Markdown 内容

        将文件内容分割为 YAML 元数据部分和 Markdown 内容部分。

        Args:
            content: 文件完整内容

        Returns:
            元组 (front_matter_dict, markdown_content)
            如果没有 front matter，返回 ({}, content)
        """
        match = self.FRONT_MATTER_PATTERN.match(content)

        if match:
            yaml_content = match.group(1)
            markdown_content = match.group(2).strip()

            try:
                front_matter = yaml.safe_load(yaml_content) or {}
            except yaml.YAMLError as e:
                logger.warning(f"[MarkdownLoader] YAML 解析警告: {e}")
                front_matter = {}

            return front_matter, markdown_content

        # 没有 front matter，整个内容作为 Markdown
        logger.debug("[MarkdownLoader] 未找到 YAML front matter，使用默认配置")
        return {}, content.strip()

    def _extract_skill_id(self, front_matter: dict[str, Any], skill_dir: Path) -> str:
        """提取技能 ID

        优先级：
        1. front_matter 中的 name 字段
        2. front_matter 中的 id 字段
        3. 目录名称

        Args:
            front_matter: YAML front matter 字典
            skill_dir: 技能目录路径

        Returns:
            技能 ID
        """
        # 优先使用 name
        if "name" in front_matter:
            return str(front_matter["name"])

        # 其次使用 id
        if "id" in front_matter:
            return str(front_matter["id"])

        # 最后使用目录名
        return skill_dir.name

    def _parse_tools(self, tools_config: list[dict[str, Any]]) -> list[StandardTool]:
        """解析工具定义列表

        将 YAML 中的工具配置转换为 StandardTool 对象列表。

        Args:
            tools_config: 工具配置列表

        Returns:
            StandardTool 对象列表
        """
        tools = []

        for tool_def in tools_config:
            if not isinstance(tool_def, dict):
                logger.warning(f"[MarkdownLoader] 跳过无效的工具定义: {tool_def}")
                continue

            # 解析参数定义
            parameters, required = self._parse_parameters(tool_def.get("parameters", {}))

            # 构建处理器配置
            handler_config = {
                "external_system": tool_def.get("external_system"),
                "connector_method": tool_def.get("connector_method"),
                "script": tool_def.get("script"),
            }
            # 移除 None 值
            handler_config = {k: v for k, v in handler_config.items() if v is not None}

            tool = StandardTool(
                name=tool_def.get("name", "unknown"),
                description=tool_def.get("description", ""),
                parameters=parameters,
                required=required,
                handler_config=handler_config,
                source_format="markdown",
            )
            tools.append(tool)

        return tools

    def _parse_parameters(
        self,
        params_config: dict[str, Any]
    ) -> tuple[dict[str, Any], list[str]]:
        """解析参数定义

        支持两种格式：
        1. 完整格式：
           promql:
             type: string
             description: PromQL 查询语句
             required: true

        2. 简化格式：
           promql: string

        Args:
            params_config: 参数配置字典

        Returns:
            元组 (properties_dict, required_list)
        """
        properties = {}
        required = []

        for param_name, param_def in params_config.items():
            if isinstance(param_def, dict):
                # 完整格式
                properties[param_name] = {
                    "type": param_def.get("type", "string"),
                    "description": param_def.get("description", ""),
                }

                # 处理枚举值
                if "enum" in param_def:
                    properties[param_name]["enum"] = param_def["enum"]

                # 处理默认值
                if "default" in param_def:
                    properties[param_name]["default"] = param_def["default"]

                # 收集必需参数
                if param_def.get("required", False):
                    required.append(param_name)

            elif isinstance(param_def, str):
                # 简化格式：直接是类型字符串
                properties[param_name] = {
                    "type": param_def,
                    "description": "",
                }
            else:
                logger.warning(
                    f"[MarkdownLoader] 跳过无效的参数定义: {param_name}={param_def}"
                )

        return properties, required

    def _get_scripts_path(self, skill_dir: Path) -> Path | None:
        """获取 scripts 目录路径

        Args:
            skill_dir: 技能目录

        Returns:
            scripts 目录路径，如不存在返回 None
        """
        scripts_path = skill_dir / "scripts"
        return scripts_path if scripts_path.exists() and scripts_path.is_dir() else None

    def _get_references_path(self, skill_dir: Path) -> Path | None:
        """获取 references 目录路径

        Args:
            skill_dir: 技能目录

        Returns:
            references 目录路径，如不存在返回 None
        """
        refs_path = skill_dir / "references"
        return refs_path if refs_path.exists() and refs_path.is_dir() else None

    def _load_references_index(self, refs_path: Path) -> list[ReferenceDocument]:
        """加载参考文档索引

        扫描 references 目录，建立文档索引。
        实际内容延迟加载，只在需要时读取。

        Args:
            refs_path: references 目录路径

        Returns:
            ReferenceDocument 对象列表
        """
        references = []

        # 支持的文档扩展名
        supported_extensions = {".md", ".markdown", ".txt", ".json"}

        # 递归扫描目录
        for file_path in refs_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                # 计算相对路径作为名称
                relative_name = file_path.relative_to(refs_path).as_posix()

                references.append(ReferenceDocument(
                    name=relative_name,
                    path=file_path,
                    content=None,  # 延迟加载
                ))

        return references

    def load_reference_content(self, ref: ReferenceDocument) -> str:
        """加载参考文档内容

        延迟加载参考文档的实际内容。

        Args:
            ref: 参考文档对象

        Returns:
            文档内容
        """
        if ref.content is None:
            ref.content = self._read_file(ref.path)
        return ref.content

    def load_script_handler(self, skill: MarkdownSkill) -> dict[str, Any] | None:
        """加载 scripts 目录中的处理器模块

        从 scripts/handler.py 加载工具函数映射。

        Args:
            skill: MarkdownSkill 对象

        Returns:
            工具函数字典 {tool_name: callable}，如加载失败返回 None
        """
        if not skill.scripts_path:
            return None

        handler_file = skill.scripts_path / "handler.py"
        if not handler_file.exists():
            logger.debug(f"[MarkdownLoader] 未找到处理器文件: {handler_file}")
            return None

        try:
            import importlib.util

            # 动态加载模块
            module_name = f"skill_{skill.skill_id}_handler"
            spec = importlib.util.spec_from_file_location(module_name, handler_file)

            if spec is None or spec.loader is None:
                logger.warning(f"[MarkdownLoader] 无法创建模块规范: {handler_file}")
                return None

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # 获取 TOOLS 字典
            handlers = getattr(module, "TOOLS", {})

            if not handlers:
                logger.warning(f"[MarkdownLoader] 处理器模块未定义 TOOLS: {handler_file}")
                return None

            logger.info(f"[MarkdownLoader] 从处理器加载 {len(handlers)} 个工具函数")
            return handlers

        except Exception as e:
            logger.error(f"[MarkdownLoader] 加载处理器失败: {e}")
            return None
