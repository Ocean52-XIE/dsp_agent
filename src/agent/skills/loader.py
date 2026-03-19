# -*- coding: utf-8 -*-
"""Skill 加载器

只支持 SKILL.md 格式的技能定义。

格式规范：
- 文件名: SKILL.md
- 内容结构: YAML front matter + Markdown 提示词模板
- 目录结构:
  skill-name/
    SKILL.md           # 技能定义（必需）
    references/        # 参考文档（可选）
      guide.md

使用示例：
    from pathlib import Path
    from agent.skills.loader import SkillLoader

    loader = SkillLoader(Path("domain/ad_engine"))
    loader.load_all()

    for skill in loader.skills.values():
        print(f"{skill.skill_id}: {skill.description}")
"""

import logging
import re
from pathlib import Path
from typing import Any

import yaml

from agent.skills.base import (
    ExecutionConfig,
    Skill,
    SkillReference,
    SkillTrigger,
)

logger = logging.getLogger(__name__)


class SkillLoadError(Exception):
    """技能加载错误"""

    def __init__(self, path: Path, message: str):
        self.path = path
        self.message = message
        super().__init__(f"加载 {path} 失败: {message}")


class SkillLoader:
    """技能加载器

    只支持 SKILL.md 格式的技能定义。

    Attributes:
        domain_root: 领域根目录
        skills: 已加载的技能字典 {skill_id: Skill}
    """

    # YAML Front Matter 正则模式
    # 允许文件开头有可选的空白或 Markdown 注释行（以 # 开头）
    # 格式：[可选的前置内容] ---\n[YAML]\n---\n[Markdown内容]
    FRONT_MATTER_PATTERN = re.compile(
        r'^(?:[ \t]*#[^\n]*\n|[ \t]*\n)*---\s*\n(.*?)\n---\s*\n(.*)$',
        re.DOTALL
    )

    # 支持的文件名
    SKILL_FILE_NAMES = ["SKILL.md", "skill.md"]

    def __init__(self, domain_root: Path):
        """初始化技能加载器

        Args:
            domain_root: 领域根目录，如 Path("domain/ad_engine")
        """
        self.domain_root = Path(domain_root)
        self.skills: dict[str, Skill] = {}
        logger.info(f"[SkillLoader] 初始化完成，领域根目录: {domain_root}")

    def load_all(self) -> dict[str, Skill]:
        """加载所有技能

        扫描技能目录，加载所有有效的 SKILL.md 文件。

        Returns:
            技能字典 {skill_id: Skill}
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
            except SkillLoadError as e:
                logger.error(f"[SkillLoader] {e}")
            except Exception as e:
                logger.error(f"[SkillLoader] 加载技能失败 {item}: {e}")

        logger.info(f"[SkillLoader] 共加载 {len(self.skills)} 个技能")
        return self.skills

    def _load_skill_dir(self, skill_dir: Path) -> Skill | None:
        """加载技能目录"""
        for skill_file_name in self.SKILL_FILE_NAMES:
            skill_file = skill_dir / skill_file_name
            if skill_file.exists():
                skill = self._load_skill_file(skill_file)
                if skill:
                    self.skills[skill.skill_id] = skill
                return skill

        logger.debug(f"[SkillLoader] 未找到 SKILL.md: {skill_dir}")
        return None

    def _load_skill_file(self, path: Path) -> Skill | None:
        """加载 SKILL.md 文件"""
        try:
            content = self._read_file(path)
            front_matter, markdown = self._parse_front_matter(content)
            skill_dir = path.parent

            # 解析类型
            skill_type = front_matter.get("skill_type", "prompt")
            if skill_type not in ("prompt", "execution"):
                logger.warning(
                    f"[SkillLoader] 无效的 skill_type '{skill_type}', "
                    f"默认使用 prompt: {path}"
                )
                skill_type = "prompt"

            # 解析触发配置
            trigger = self._parse_trigger(front_matter.get("trigger", {}))

            # 解析执行配置（仅 execution 类型）
            execution = None
            if skill_type == "execution":
                execution = self._parse_execution(front_matter.get("execution", {}))
                if execution is None:
                    logger.warning(f"[SkillLoader] execution 未配置: {path}")

            # 解析参数
            params = front_matter.get("params", {})

            # 解析示例
            examples = front_matter.get("examples", [])

            skill = Skill(
                # 基础信息
                skill_id=self._extract_skill_id(front_matter, skill_dir),
                display_name=front_matter.get("display_name", skill_dir.name),
                description=front_matter.get("description", ""),
                version=front_matter.get("version", "1.0.0"),
                tags=front_matter.get("tags", []),
                enabled=front_matter.get("enabled", True),
                # 类型
                skill_type=skill_type,
                # 触发配置
                trigger=trigger,
                # 参数
                params=params,
                # Prompt Skill
                prompt_template=markdown if skill_type == "prompt" else "",
                # Execution Skill
                execution=execution,
                # 示例
                examples=examples,
                # 源文件信息
                source_path=path,
                references_path=self._get_dir(skill_dir / "references"),
            )

            # 加载参考文档索引
            if skill.references_path:
                skill.references = self._load_references(skill.references_path)

            logger.info(
                f"[SkillLoader] 加载技能: {skill.skill_id}, "
                f"类型={skill.skill_type}, 关键词={len(trigger.keywords)}, "
                f"模式={len(trigger.patterns)}"
            )

            return skill

        except Exception as e:
            logger.error(f"[SkillLoader] 加载失败 {path}: {e}")
            return None

    def _read_file(self, path: Path, encoding: str = "utf-8") -> str:
        """读取文件内容"""
        return path.read_text(encoding=encoding)

    def _parse_front_matter(self, content: str) -> tuple[dict[str, Any], str]:
        """解析 YAML front matter

        Args:
            content: 文件内容

        Returns:
            (front_matter_dict, markdown_content)
        """
        match = self.FRONT_MATTER_PATTERN.match(content)
        if match:
            try:
                front_matter = yaml.safe_load(match.group(1)) or {}
            except yaml.YAMLError as e:
                logger.warning(f"[SkillLoader] YAML 解析警告: {e}")
                front_matter = {}
            return front_matter, match.group(2).strip()
        return {}, content.strip()

    def _extract_skill_id(
        self,
        front_matter: dict[str, Any],
        skill_dir: Path
    ) -> str:
        """提取技能 ID"""
        # 优先使用 skill_id，其次 name，最后目录名
        return (
            front_matter.get("skill_id") or
            front_matter.get("name") or
            skill_dir.name
        )

    def _parse_trigger(self, config: dict[str, Any]) -> SkillTrigger:
        """解析触发配置"""
        return SkillTrigger(
            keywords=config.get("keywords", []),
            patterns=config.get("patterns", []),
            priority=config.get("priority", 10),
        )

    def _parse_execution(
        self,
        config: dict[str, Any]
    ) -> ExecutionConfig | None:
        """解析执行配置"""
        if not config:
            return None

        command = config.get("command")
        if not command:
            return None

        return ExecutionConfig(
            command=command,
            cwd=config.get("cwd"),
            env=config.get("env", {}),
            timeout=config.get("timeout", 30),
            output=config.get("output", "text"),
            extract=config.get("extract"),
        )

    def _get_dir(self, path: Path) -> Path | None:
        """获取有效目录"""
        return path if path.exists() and path.is_dir() else None

    def _load_references(self, refs_path: Path) -> list[SkillReference]:
        """加载参考文档索引"""
        refs = []
        for f in refs_path.rglob("*"):
            if f.is_file() and f.suffix.lower() in {".md", ".txt", ".json"}:
                refs.append(SkillReference(
                    name=str(f.relative_to(refs_path)),
                    path=f,
                ))
        return refs

    def get_skill(self, skill_id: str) -> Skill | None:
        """获取指定技能"""
        return self.skills.get(skill_id)

    def list_skills(self) -> list[Skill]:
        """获取所有技能列表"""
        return list(self.skills.values())

    def reload(self) -> dict[str, Skill]:
        """重新加载所有技能"""
        logger.info("[SkillLoader] 重新加载")
        return self.load_all()
