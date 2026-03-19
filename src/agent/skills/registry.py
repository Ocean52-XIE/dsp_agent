# -*- coding: utf-8 -*-
"""Skill 注册中心

管理所有已加载的技能，提供：
- 技能加载和注册
- 基于规则的候选筛选（关键词 + 正则）
- Catalog 生成（给 LLM 选择）

使用示例：
    from pathlib import Path
    from agent.skills.registry import SkillRegistry

    # 初始化并加载
    registry = SkillRegistry()
    count = registry.load_from_directory(Path("domain/ad_engine"))

    # 获取候选技能
    candidates = registry.get_candidates("查一下昨天的 CTR")

    # 获取 Catalog
    catalog = registry.get_catalog()
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent.skills.base import Skill, SkillCatalogItem

logger = logging.getLogger(__name__)


@dataclass
class CandidateMatch:
    """候选匹配结果"""
    skill: Skill
    score: int = 0
    match_reasons: list[str] = field(default_factory=list)


class SkillRegistry:
    """Skill 注册中心

    管理所有已加载的技能，提供加载、索引、筛选功能。

    Attributes:
        skills: 技能字典 {skill_id: Skill}
        _keyword_index: 关键词索引 {keyword: set(skill_ids)}
        _pattern_index: 正则索引 [(compiled_pattern, skill_id)]
        _catalog_cache: Catalog 缓存
    """

    def __init__(self):
        """初始化技能注册中心"""
        self.skills: dict[str, Skill] = {}
        self._keyword_index: dict[str, set[str]] = {}
        self._pattern_index: list[tuple[re.Pattern, str]] = []
        self._catalog_cache: list[SkillCatalogItem] = []
        logger.info("[SkillRegistry] 初始化完成")

    def load_from_directory(self, domain_root: Path) -> int:
        """从领域目录加载所有技能

        便捷方法，自动创建 SkillLoader 并加载技能。

        Args:
            domain_root: 领域根目录，如 Path("domain/ad_engine")

        Returns:
            加载的技能数量
        """
        from agent.skills.loader import SkillLoader

        loader = SkillLoader(domain_root)
        loader.load_all()

        count = 0
        for skill in loader.skills.values():
            self.register(skill)
            count += 1

        logger.info(f"[SkillRegistry] 从目录加载 {count} 个技能: {domain_root}")
        return count

    def register(self, skill: Skill) -> None:
        """注册技能

        将技能添加到注册中心，并建立索引。

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

        # 建立正则索引
        for pattern in skill.get_patterns():
            try:
                compiled = re.compile(pattern, re.IGNORECASE)
                self._pattern_index.append((compiled, skill.skill_id))
            except re.error as e:
                logger.warning(f"[SkillRegistry] 无效正则模式 '{pattern}': {e}")

        # 更新 Catalog 缓存
        self._catalog_cache.append(skill.to_catalog_item())

        logger.info(
            f"[SkillRegistry] 注册技能: {skill.skill_id}, "
            f"类型={skill.skill_type}, 关键词={len(skill.get_keywords())}, "
            f"模式={len(skill.get_patterns())}"
        )

    def unregister(self, skill_id: str) -> bool:
        """注销技能

        Args:
            skill_id: 技能 ID

        Returns:
            是否成功注销
        """
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

        # 移除正则索引
        self._pattern_index = [
            (p, sid) for p, sid in self._pattern_index if sid != skill_id
        ]

        # 移除技能
        del self.skills[skill_id]

        # 移除 Catalog 缓存
        self._catalog_cache = [
            item for item in self._catalog_cache if item.skill_id != skill_id
        ]

        logger.info(f"[SkillRegistry] 注销技能: {skill_id}")
        return True

    def get_skill(self, skill_id: str) -> Skill | None:
        """获取指定技能

        Args:
            skill_id: 技能 ID

        Returns:
            技能对象，不存在则返回 None
        """
        return self.skills.get(skill_id)

    def list_skills(self) -> list[Skill]:
        """获取所有技能列表"""
        return list(self.skills.values())

    def get_catalog(
        self,
        skill_ids: list[str] | None = None
    ) -> list[SkillCatalogItem]:
        """获取技能目录

        返回给 LLM 做选择的精简对象列表。

        Args:
            skill_ids: 指定技能 ID 列表，为空则返回所有

        Returns:
            SkillCatalogItem 列表
        """
        if skill_ids is None:
            return self._catalog_cache.copy()

        id_set = set(skill_ids)
        return [item for item in self._catalog_cache if item.skill_id in id_set]

    def get_candidates(
        self,
        query: str,
        max_count: int = 5,
        min_score: int = 0,
    ) -> list[Skill]:
        """获取候选技能（规则筛选）

        通过关键词和正则模式匹配筛选候选技能。
        匹配规则：
        - 关键词匹配: +10 分
        - 正则模式匹配: +20 分
        - 按分数和优先级排序

        Args:
            query: 用户查询文本
            max_count: 最大候选数量，默认 5
            min_score: 最小匹配分数，默认 0

        Returns:
            候选技能列表（按分数排序）
        """
        candidates: dict[str, CandidateMatch] = {}
        query_lower = query.lower()

        # 关键词匹配 (+10 分)
        for keyword, skill_ids in self._keyword_index.items():
            if keyword in query_lower:
                for skill_id in skill_ids:
                    if skill_id not in candidates:
                        candidates[skill_id] = CandidateMatch(
                            skill=self.skills[skill_id]
                        )
                    candidates[skill_id].score += 10
                    candidates[skill_id].match_reasons.append(f"keyword:{keyword}")

        # 正则模式匹配 (+20 分)
        for pattern, skill_id in self._pattern_index:
            if pattern.search(query):
                if skill_id not in candidates:
                    candidates[skill_id] = CandidateMatch(
                        skill=self.skills[skill_id]
                    )
                candidates[skill_id].score += 20
                candidates[skill_id].match_reasons.append(f"pattern:{pattern.pattern}")

        # 过滤和排序
        filtered = [c for c in candidates.values() if c.score >= min_score]
        sorted_candidates = sorted(
            filtered,
            key=lambda c: (c.score, c.skill.get_priority()),
            reverse=True
        )

        result = [c.skill for c in sorted_candidates[:max_count]]

        if result:
            logger.debug(
                f"[SkillRegistry] 候选技能: {[s.skill_id for s in result]}, "
                f"查询: {query[:50]}"
            )

        return result

    def clear(self) -> None:
        """清空所有技能"""
        self.skills.clear()
        self._keyword_index.clear()
        self._pattern_index.clear()
        self._catalog_cache.clear()
        logger.info("[SkillRegistry] 已清空所有技能")

    def get_stats(self) -> dict[str, Any]:
        """获取统计信息"""
        type_counts: dict[str, int] = {}
        for skill in self.skills.values():
            t = skill.skill_type
            type_counts[t] = type_counts.get(t, 0) + 1

        return {
            "total_skills": len(self.skills),
            "total_keywords": len(self._keyword_index),
            "total_patterns": len(self._pattern_index),
            "catalog_size": len(self._catalog_cache),
            "type_distribution": type_counts,
            "skill_ids": list(self.skills.keys()),
        }


# 全局单例
_skill_registry: SkillRegistry | None = None


def get_skill_registry() -> SkillRegistry:
    """获取全局技能注册中心单例"""
    global _skill_registry
    if _skill_registry is None:
        _skill_registry = SkillRegistry()
    return _skill_registry


def reset_skill_registry() -> None:
    """重置全局技能注册中心（用于测试）"""
    global _skill_registry
    if _skill_registry is not None:
        _skill_registry.clear()
    _skill_registry = None
