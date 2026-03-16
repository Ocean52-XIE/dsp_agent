"""技能注册中心（One-Stage 简化版）

管理所有已加载的技能，提供：
- 技能加载和注册
- 基于规则的候选筛选
- Catalog 生成（One-Stage）

使用示例：
    from pathlib import Path
    from src.workflow.skills.registry import SkillRegistry

    registry = SkillRegistry()
    registry.load_from_directory(Path("domain/ad_engine"))

    # 获取候选技能
    candidates = registry.get_candidates("查一下 CTR")

    # 获取 Catalog（给 LLM）
    catalog = registry.get_catalog()
"""
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.workflow.skills.base import (
    SkillCatalogItem,
    StandardSkill,
)
from src.workflow.skills.loader import SkillLoader

logger = logging.getLogger(__name__)


@dataclass
class CandidateSkill:
    """候选技能（带匹配分数）"""
    skill: StandardSkill
    score: int = 0
    match_reasons: list[str] = field(default_factory=list)


class SkillRegistry:
    """技能注册中心（One-Stage 简化版）

    管理所有已加载的技能，提供加载、筛选和 Catalog 生成功能。

    Attributes:
        skills: 技能字典 {skill_id: StandardSkill}
        _keyword_index: 关键词索引
        _pattern_index: 模式索引
        _catalog_cache: Catalog 缓存（在注册时立即更新）
    """

    def __init__(self):
        """初始化技能注册中心"""
        self.skills: dict[str, StandardSkill] = {}
        self._keyword_index: dict[str, set[str]] = {}
        self._pattern_index: list[tuple[str, str]] = []
        # 缓存在初始化时创建，避免多线程竞争
        self._catalog_cache: list[SkillCatalogItem] = []
        logger.info("[SkillRegistry] 初始化完成")

    def load_from_directory(self, domain_root: Path) -> int:
        """从目录加载技能

        便捷方法，自动创建 SkillLoader 并加载技能。

        Args:
            domain_root: 领域根目录

        Returns:
            加载的技能数量
        """
        loader = SkillLoader(domain_root)
        loader.load_all()
        return self.register_from_loader(loader)

    def register(self, skill: StandardSkill) -> None:
        """注册技能

        将技能添加到注册中心，并建立索引和更新缓存。

        Args:
            skill: 要注册的技能对象
        """
        if not skill.enabled:
            logger.debug(f"[SkillRegistry] 跳过禁用的技能: {skill.skill_id}")
            return

        self.skills[skill.skill_id] = skill

        # 建立关键词索引
        for keyword in skill.get_keywords():
            kw = keyword.lower()
            if kw not in self._keyword_index:
                self._keyword_index[kw] = set()
            self._keyword_index[kw].add(skill.skill_id)

        # 建立模式索引
        for pattern in skill.get_patterns():
            self._pattern_index.append((pattern, skill.skill_id))

        # 立即更新缓存
        self._add_to_catalog_cache(skill)

        logger.info(f"[SkillRegistry] 注册: {skill.skill_id}, 类型={skill.skill_type}")

    def _add_to_catalog_cache(self, skill: StandardSkill) -> None:
        """添加技能到 Catalog 缓存"""
        self._catalog_cache.append(SkillCatalogItem(
            name=skill.skill_id,
            description=skill.description,
            tags=skill.tags,
            input_schema=skill.param_schema,
            examples=skill.examples[:3],
        ))

    def _remove_from_catalog_cache(self, skill_id: str) -> None:
        """从 Catalog 缓存中移除技能"""
        self._catalog_cache = [item for item in self._catalog_cache if item.name != skill_id]

    def register_from_loader(self, loader: SkillLoader) -> int:
        """从加载器批量注册技能

        Args:
            loader: 技能加载器

        Returns:
            注册的技能数量
        """
        count = 0
        for skill in loader.skills.values():
            self.register(skill)
            count += 1
        logger.info(f"[SkillRegistry] 从加载器注册 {count} 个技能")
        return count

    def unregister(self, skill_id: str) -> bool:
        """注销技能"""
        if skill_id not in self.skills:
            return False

        skill = self.skills[skill_id]

        # 移除关键词索引
        for keyword in skill.get_keywords():
            kw = keyword.lower()
            if kw in self._keyword_index:
                self._keyword_index[kw].discard(skill_id)
                if not self._keyword_index[kw]:
                    del self._keyword_index[kw]

        # 移除模式索引
        self._pattern_index = [(p, sid) for p, sid in self._pattern_index if sid != skill_id]

        # 移除技能
        del self.skills[skill_id]

        # 立即从缓存中移除
        self._remove_from_catalog_cache(skill_id)

        logger.info(f"[SkillRegistry] 注销: {skill_id}")
        return True

    def get_skill(self, skill_id: str) -> StandardSkill | None:
        """获取指定技能"""
        return self.skills.get(skill_id)

    def list_skills(self) -> list[StandardSkill]:
        """获取所有技能列表"""
        return list(self.skills.values())

    def get_catalog(self, skill_names: list[str] | None = None) -> list[SkillCatalogItem]:
        """获取技能 Catalog（One-Stage）

        返回给 LLM 做第一轮选择的精简对象列表。
        缓存在注册/注销时已经创建，此方法只负责读取。

        Args:
            skill_names: 指定技能名称列表，为空则返回所有缓存的技能

        Returns:
            SkillCatalogItem 列表
        """
        if skill_names is None:
            return self._catalog_cache.copy()

        # 指定技能名称时，从缓存中过滤
        name_set = set(skill_names)
        return [item for item in self._catalog_cache if item.name in name_set]

    def get_candidates(
        self,
        query: str,
        max_candidates: int = 5,
        min_score: int = 0,
    ) -> list[StandardSkill]:
        """获取候选技能（规则筛选）

        通过关键词和模式匹配筛选候选技能。

        Args:
            query: 用户查询文本
            max_candidates: 最大候选数量
            min_score: 最小匹配分数

        Returns:
            候选技能列表（按分数排序）
        """
        candidates: dict[str, CandidateSkill] = {}

        # 关键词匹配
        query_lower = query.lower()
        for keyword, skill_ids in self._keyword_index.items():
            if keyword in query_lower:
                for skill_id in skill_ids:
                    if skill_id not in candidates:
                        candidates[skill_id] = CandidateSkill(skill=self.skills[skill_id])
                    candidates[skill_id].score += 10
                    candidates[skill_id].match_reasons.append(f"keyword:{keyword}")

        # 模式匹配
        for pattern, skill_id in self._pattern_index:
            try:
                if re.search(pattern, query, re.IGNORECASE):
                    if skill_id not in candidates:
                        candidates[skill_id] = CandidateSkill(skill=self.skills[skill_id])
                    candidates[skill_id].score += 20
                    candidates[skill_id].match_reasons.append(f"pattern:{pattern}")
            except re.error:
                logger.warning(f"[SkillRegistry] 无效模式: {pattern}")

        # 过滤和排序
        filtered = [c for c in candidates.values() if c.score >= min_score]
        sorted_candidates = sorted(
            filtered,
            key=lambda c: (c.score, c.skill.get_priority()),
            reverse=True
        )

        result = [c.skill for c in sorted_candidates[:max_candidates]]
        logger.debug(f"[SkillRegistry] 候选: {[s.skill_id for s in result]}")
        return result

    def clear(self) -> None:
        """清空所有技能"""
        self.skills.clear()
        self._keyword_index.clear()
        self._pattern_index.clear()
        self._catalog_cache.clear()
        logger.info("[SkillRegistry] 已清空")

    def get_stats(self) -> dict[str, Any]:
        """获取统计信息"""
        type_counts = {}
        for skill in self.skills.values():
            t = skill.skill_type
            type_counts[t] = type_counts.get(t, 0) + 1

        return {
            "total_skills": len(self.skills),
            "total_keywords": len(self._keyword_index),
            "total_patterns": len(self._pattern_index),
            "catalog_size": len(self._catalog_cache),
            "type_distribution": type_counts,
        }


# 全局单例
_skill_registry: SkillRegistry | None = None


def get_skill_registry() -> SkillRegistry:
    """获取全局技能注册中心单例"""
    global _skill_registry
    if _skill_registry is None:
        _skill_registry = SkillRegistry()
    return _skill_registry
