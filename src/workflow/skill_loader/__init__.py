"""技能加载器模块

本模块提供统一的技能加载能力，支持多种格式的技能定义。

支持的格式：
- Markdown + YAML front matter (SKILL.md) - 推荐格式
- 可扩展支持其他格式（OpenAI JSON、LangChain 等）

主要组件：
- UnifiedSkillLoader: 统一技能加载器，自动识别格式并加载
- MarkdownSkillLoader: Markdown 格式加载器
- MarkdownSkill: Markdown 技能数据结构
- StandardSkill: 标准化技能数据结构
- StandardTool: 标准化工具数据结构

使用示例：
    from pathlib import Path
    from src.workflow.skill_loader import UnifiedSkillLoader

    # 创建加载器
    domain_root = Path("domain/ad_engine")
    loader = UnifiedSkillLoader(domain_root)

    # 加载所有技能
    skills = loader.load_all()

    # 获取特定技能
    skill = loader.get_skill("query_metrics")

    # 获取 OpenAI 工具 Schema（用于 LLM 调用）
    schema = loader.get_openai_tools_schema()
"""
import logging
from pathlib import Path
from typing import Any

from src.workflow.skill_loader.base import (
    BaseSkillLoader,
    MarkdownSkill,
    ReferenceDocument,
    SkillLoadError,
    StandardSkill,
    StandardTool,
)
from src.workflow.skill_loader.markdown_loader import MarkdownSkillLoader

logger = logging.getLogger(__name__)


class UnifiedSkillLoader:
    """统一技能加载器

    自动识别并加载多种格式的技能定义。
    按优先级依次尝试各加载器，直到成功加载。

    Features:
    - 自动识别技能格式
    - 支持目录形式和单文件形式的技能
    - 统一的技能数据结构
    - 生成 OpenAI 工具 Schema

    Attributes:
        domain_root: 领域根目录
        skills: 已加载的技能字典 {skill_id: StandardSkill}
        loaders: 加载器列表（按优先级排序）
    """

    def __init__(self, domain_root: Path):
        """初始化统一加载器

        Args:
            domain_root: 领域根目录，如 Path("domain/ad_engine")
        """
        self.domain_root = domain_root
        self.skills: dict[str, StandardSkill] = {}

        # 注册加载器（按优先级排序）
        # Markdown 格式优先级最高，是推荐的主流格式
        self.loaders: list[BaseSkillLoader] = [
            MarkdownSkillLoader(),
            # 后续可扩展其他加载器：
            # OpenAISkillLoader(),
            # LangChainSkillLoader(),
        ]

        logger.info(f"[UnifiedLoader] 初始化完成，领域根目录: {domain_root}")

    def load_all(self) -> dict[str, StandardSkill]:
        """加载所有技能

        扫描技能目录，加载所有有效的技能定义。

        Returns:
            技能字典 {skill_id: StandardSkill}
        """
        skills_dir = self.domain_root / "skills"

        if not skills_dir.exists():
            logger.warning(f"[UnifiedLoader] 技能目录不存在: {skills_dir}")
            return {}

        # 清空已有技能
        self.skills = {}

        # 遍历技能目录
        for item in skills_dir.iterdir():
            # 跳过隐藏目录和 __pycache__
            if item.name.startswith(".") or item.name.startswith("__"):
                continue

            try:
                if item.is_dir():
                    # 目录形式：尝试加载目录中的配置
                    self._load_skill_dir(item)
                elif item.is_file():
                    # 文件形式：单文件技能定义
                    self._load_skill_file(item)
            except Exception as e:
                logger.error(f"[UnifiedLoader] 加载技能失败 {item}: {e}")
                continue

        logger.info(f"[UnifiedLoader] 共加载 {len(self.skills)} 个技能")
        return self.skills

    def _load_skill_dir(self, skill_dir: Path) -> None:
        """加载技能目录

        按优先级查找配置文件：
        1. SKILL.md（Markdown 格式，推荐）
        2. skill.json（JSON 格式）
        3. 其他被加载器支持的文件

        Args:
            skill_dir: 技能目录路径
        """
        # 优先查找 SKILL.md
        for skill_file_name in MarkdownSkillLoader.SKILL_FILE_NAMES:
            skill_file = skill_dir / skill_file_name
            if skill_file.exists():
                skill = self._load_with_loader(skill_file)
                if skill:
                    self._register_skill(skill)
                return

        # 其次查找其他配置文件
        config_candidates = [
            skill_dir / "skill.json",
            skill_dir / "tool.json",
            skill_dir / "tools.json",
        ]

        for config_file in config_candidates:
            if config_file.exists():
                skill = self._load_with_loader(config_file)
                if skill:
                    self._register_skill(skill)
                return

        # 检查目录中是否有被加载器支持的文件
        for file_path in skill_dir.iterdir():
            if file_path.is_file():
                skill = self._load_with_loader(file_path)
                if skill:
                    self._register_skill(skill)
                    return

        logger.debug(f"[UnifiedLoader] 未找到有效的技能配置: {skill_dir}")

    def _load_skill_file(self, skill_file: Path) -> None:
        """加载单文件技能

        Args:
            skill_file: 技能文件路径
        """
        skill = self._load_with_loader(skill_file)
        if skill:
            self._register_skill(skill)

    def _load_with_loader(
        self,
        path: Path
    ) -> StandardSkill | list[StandardSkill] | None:
        """使用合适的加载器加载文件

        按加载器优先级依次尝试，直到成功加载。

        Args:
            path: 文件路径

        Returns:
            加载的技能对象，可能是单个或列表。加载失败返回 None
        """
        for loader in self.loaders:
            try:
                if loader.can_load(path):
                    logger.info(
                        f"[UnifiedLoader] 使用 {loader.format_name} 加载器: {path.name}"
                    )
                    return loader.load(path)
            except SkillLoadError as e:
                logger.warning(f"[UnifiedLoader] {e}")
                continue
            except Exception as e:
                logger.warning(
                    f"[UnifiedLoader] {loader.format_name} 加载失败 {path}: {e}"
                )
                continue

        logger.debug(f"[UnifiedLoader] 无法识别技能格式: {path}")
        return None

    def _register_skill(
        self,
        skill: StandardSkill | list[StandardSkill]
    ) -> None:
        """注册技能到技能字典

        Args:
            skill: 技能对象，可能是单个或列表
        """
        if isinstance(skill, list):
            for s in skill:
                self.skills[s.skill_id] = s
                logger.debug(f"[UnifiedLoader] 注册技能: {s.skill_id}")
        else:
            self.skills[skill.skill_id] = skill
            logger.debug(f"[UnifiedLoader] 注册技能: {skill.skill_id}")

    def get_skill(self, skill_id: str) -> StandardSkill | None:
        """获取指定技能

        Args:
            skill_id: 技能 ID

        Returns:
            技能对象，如不存在返回 None
        """
        return self.skills.get(skill_id)

    def list_skills(self) -> list[StandardSkill]:
        """获取所有技能列表

        Returns:
            技能列表
        """
        return list(self.skills.values())

    def get_enabled_skills(self) -> list[StandardSkill]:
        """获取所有启用的技能

        Returns:
            启用的技能列表
        """
        return [skill for skill in self.skills.values() if skill.enabled]

    def get_openai_tools_schema(self) -> list[dict[str, Any]]:
        """获取所有技能的 OpenAI 工具 Schema

        用于 LLM 工具调用时的工具定义。

        Returns:
            OpenAI 格式的工具 Schema 列表
        """
        schemas = []

        for skill in self.skills.values():
            if not skill.enabled:
                continue

            for tool in skill.tools:
                schema = tool.to_openai_schema()
                schemas.append(schema)

        return schemas

    def get_skill_by_keyword(self, text: str) -> StandardSkill | None:
        """通过关键词匹配技能

        检查文本是否包含技能的触发关键词。

        Args:
            text: 待匹配的文本

        Returns:
            匹配的技能，如无匹配返回 None
        """
        text_lower = text.lower()

        for skill in self.skills.values():
            if not skill.enabled:
                continue

            trigger = skill.trigger
            if not trigger:
                continue

            # 检查关键词列表（包含匹配）
            keywords = trigger.get("keywords", [])
            for kw in keywords:
                if kw.lower() in text_lower:
                    return skill

        return None

    def get_skill_by_pattern(self, text: str) -> StandardSkill | None:
        """通过正则模式匹配技能

        根据技能的触发模式进行匹配。

        Args:
            text: 待匹配的文本

        Returns:
            匹配的技能，如无匹配返回 None
        """
        import re

        for skill in self.skills.values():
            if not skill.enabled:
                continue

            trigger = skill.trigger
            if not trigger:
                continue

            # 检查正则模式列表
            patterns = trigger.get("patterns", [])
            for pattern in patterns:
                try:
                    if re.search(pattern, text, re.IGNORECASE):
                        return skill
                except re.error:
                    logger.warning(f"[UnifiedLoader] 无效的正则模式: {pattern}")
                    continue

        return None

    def reload(self) -> dict[str, StandardSkill]:
        """重新加载所有技能

        清空现有技能并重新加载。

        Returns:
            技能字典
        """
        logger.info("[UnifiedLoader] 重新加载所有技能")
        return self.load_all()


# 导出公共接口
__all__ = [
    # 加载器
    "UnifiedSkillLoader",
    "MarkdownSkillLoader",
    "BaseSkillLoader",
    # 数据结构
    "StandardSkill",
    "StandardTool",
    "MarkdownSkill",
    "ReferenceDocument",
    # 异常
    "SkillLoadError",
]
