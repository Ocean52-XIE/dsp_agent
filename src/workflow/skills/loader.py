"""技能加载器（One-Stage 简化版）

只支持 SKILL.md 格式的技能定义。

格式规范：
- 文件名: SKILL.md
- 内容结构: YAML front matter + Markdown 提示词模板
- 目录结构:
  skill-name/
    SKILL.md           # 技能定义（必需）
    scripts/           # 执行脚本（可选）
      handler.py
    references/        # 参考文档（可选）

使用示例：
    from pathlib import Path
    from src.workflow.skills.loader import SkillLoader

    loader = SkillLoader(Path("domain/ad_engine"))
    loader.load_all()
"""
import logging
import re
from pathlib import Path
from typing import Any

import yaml

from src.workflow.skills.base import (
    ReferenceDocument,
    StandardSkill,
    StandardTool,
)

logger = logging.getLogger(__name__)


class SkillLoader:
    """技能加载器（One-Stage 简化版）

    只支持 SKILL.md 格式的技能定义。

    Attributes:
        domain_root: 领域根目录
        skills: 已加载的技能字典 {skill_id: StandardSkill}
    """

    # YAML Front Matter 正则模式
    FRONT_MATTER_PATTERN = re.compile(
        r'^---\s*\n(.*?)\n---\s*\n(.*)$',
        re.DOTALL
    )

    # 支持的文件名
    SKILL_FILE_NAMES = ["SKILL.md", "skill.md"]

    def __init__(self, domain_root: Path):
        """初始化技能加载器

        Args:
            domain_root: 领域根目录，如 Path("domain/ad_engine")
        """
        self.domain_root = domain_root
        self.skills: dict[str, StandardSkill] = {}
        logger.info(f"[SkillLoader] 初始化完成，领域根目录: {domain_root}")

    def load_all(self) -> dict[str, StandardSkill]:
        """加载所有技能

        扫描技能目录，加载所有有效的 SKILL.md 文件。

        Returns:
            技能字典 {skill_id: StandardSkill}
        """
        skills_dir = self.domain_root / "skills"

        if not skills_dir.exists():
            logger.warning(f"[SkillLoader] 技能目录不存在: {skills_dir}")
            return {}

        self.skills = {}

        for item in skills_dir.iterdir():
            if item.name.startswith(".") or item.name.startswith("__"):
                continue

            try:
                if item.is_dir():
                    self._load_skill_dir(item)
            except Exception as e:
                logger.error(f"[SkillLoader] 加载技能失败 {item}: {e}")

        logger.info(f"[SkillLoader] 共加载 {len(self.skills)} 个技能")
        return self.skills

    def _load_skill_dir(self, skill_dir: Path) -> None:
        """加载技能目录"""
        for skill_file_name in self.SKILL_FILE_NAMES:
            skill_file = skill_dir / skill_file_name
            if skill_file.exists():
                skill = self._load_skill_file(skill_file)
                if skill:
                    self.skills[skill.skill_id] = skill
                return

        logger.debug(f"[SkillLoader] 未找到 SKILL.md: {skill_dir}")

    def _load_skill_file(self, path: Path) -> StandardSkill | None:
        """加载 SKILL.md 文件"""
        try:
            content = self._read_file(path)
            front_matter, markdown_content = self._parse_front_matter(content)
            skill_dir = path.parent

            skill = StandardSkill(
                skill_id=self._extract_skill_id(front_matter, skill_dir),
                display_name=front_matter.get("name", skill_dir.name),
                description=front_matter.get("description", ""),
                version=front_matter.get("version", "1.0.0"),
                author=front_matter.get("author", ""),
                tags=front_matter.get("tags", []),
                enabled=front_matter.get("enabled", True),
                skill_type=front_matter.get("type", "llm_orchestrated"),
                tools=self._parse_tools(front_matter.get("tools", [])),
                trigger=front_matter.get("trigger", {}),
                param_schema=front_matter.get("params", {}),
                prompt_template=markdown_content,
                execution=front_matter.get("execution", {}),
                examples=self._parse_examples(front_matter.get("examples", [])),
                scripts_path=self._get_path(skill_dir / "scripts"),
                references_path=self._get_path(skill_dir / "references"),
                references=[],
                source_path=path,
            )

            if skill.references_path:
                skill.references = self._load_references_index(skill.references_path)

            logger.info(
                f"[SkillLoader] 加载技能: {skill.skill_id}, "
                f"类型={skill.skill_type}, 工具={len(skill.tools)}, 示例={len(skill.examples)}"
            )

            return skill

        except Exception as e:
            logger.error(f"[SkillLoader] 加载失败 {path}: {e}")
            return None

    def _read_file(self, path: Path, encoding: str = "utf-8") -> str:
        """读取文件内容"""
        with open(path, "r", encoding=encoding) as f:
            return f.read()

    def _parse_front_matter(self, content: str) -> tuple[dict[str, Any], str]:
        """解析 YAML front matter"""
        match = self.FRONT_MATTER_PATTERN.match(content)
        if match:
            try:
                front_matter = yaml.safe_load(match.group(1)) or {}
            except yaml.YAMLError as e:
                logger.warning(f"[SkillLoader] YAML 解析警告: {e}")
                front_matter = {}
            return front_matter, match.group(2).strip()
        return {}, content.strip()

    def _extract_skill_id(self, front_matter: dict[str, Any], skill_dir: Path) -> str:
        """提取技能 ID"""
        return front_matter.get("name") or front_matter.get("id") or skill_dir.name

    def _parse_tools(self, tools_config: list[dict[str, Any]]) -> list[StandardTool]:
        """解析工具定义"""
        tools = []
        for tool_def in tools_config or []:
            if not isinstance(tool_def, dict):
                continue
            parameters, required = self._parse_parameters(tool_def.get("parameters", {}))
            handler_config = {
                k: v for k, v in {
                    "external_system": tool_def.get("external_system"),
                    "connector_method": tool_def.get("connector_method"),
                    "script": tool_def.get("script"),
                }.items() if v is not None
            }
            tools.append(StandardTool(
                name=tool_def.get("name", "unknown"),
                description=tool_def.get("description", ""),
                parameters=parameters,
                required=required,
                handler_config=handler_config,
            ))
        return tools

    def _parse_parameters(self, params_config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        """解析参数定义"""
        properties = {}
        required = []
        for param_name, param_def in (params_config or {}).items():
            if isinstance(param_def, dict):
                properties[param_name] = {
                    "type": param_def.get("type", "string"),
                    "description": param_def.get("description", ""),
                }
                if "enum" in param_def:
                    properties[param_name]["enum"] = param_def["enum"]
                if "default" in param_def:
                    properties[param_name]["default"] = param_def["default"]
                if param_def.get("required"):
                    required.append(param_name)
            elif isinstance(param_def, str):
                properties[param_name] = {"type": param_def, "description": ""}
        return properties, required

    def _parse_examples(self, examples_config: list[Any]) -> list[dict[str, Any]]:
        """解析调用示例"""
        examples = []
        for ex in examples_config or []:
            if isinstance(ex, dict):
                examples.append({"user": ex.get("user", ""), "arguments": ex.get("arguments", {})})
            elif isinstance(ex, str):
                examples.append({"user": ex, "arguments": {}})
        return examples

    def _get_path(self, path: Path) -> Path | None:
        """获取有效路径"""
        return path if path.exists() and path.is_dir() else None

    def _load_references_index(self, refs_path: Path) -> list[ReferenceDocument]:
        """加载参考文档索引"""
        refs = []
        for f in refs_path.rglob("*"):
            if f.is_file() and f.suffix.lower() in {".md", ".txt", ".json"}:
                refs.append(ReferenceDocument(
                    name=str(f.relative_to(refs_path)),
                    path=f,
                ))
        return refs

    def get_skill(self, skill_id: str) -> StandardSkill | None:
        """获取指定技能"""
        return self.skills.get(skill_id)

    def list_skills(self) -> list[StandardSkill]:
        """获取所有技能列表"""
        return list(self.skills.values())

    def reload(self) -> dict[str, StandardSkill]:
        """重新加载所有技能"""
        logger.info("[SkillLoader] 重新加载")
        return self.load_all()
