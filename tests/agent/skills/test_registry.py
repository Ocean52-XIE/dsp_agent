# -*- coding: utf-8 -*-
"""SkillRegistry 测试"""

import pytest
from pathlib import Path

from agent.skills.base import Skill, SkillTrigger
from agent.skills.registry import SkillRegistry, reset_skill_registry


class TestSkillRegistry:
    """SkillRegistry 测试"""

    def setup_method(self):
        """每个测试前重置注册中心"""
        reset_skill_registry()

    def test_init(self):
        """测试初始化"""
        registry = SkillRegistry()
        assert registry.skills == {}
        assert registry._keyword_index == {}
        assert registry._pattern_index == []
        assert registry._catalog_cache == []

    def test_register_skill(self):
        """测试注册技能"""
        registry = SkillRegistry()
        skill = Skill(
            skill_id="test_skill",
            display_name="测试技能",
            description="这是一个测试技能",
            trigger=SkillTrigger(
                keywords=["测试", "test"],
                patterns=[r"测试.*"],
            ),
        )

        registry.register(skill)

        assert "test_skill" in registry.skills
        assert "测试" in registry._keyword_index
        assert "test" in registry._keyword_index
        assert len(registry._pattern_index) == 1

    def test_register_disabled_skill(self):
        """测试注册禁用的技能"""
        registry = SkillRegistry()
        skill = Skill(
            skill_id="disabled_skill",
            display_name="禁用技能",
            enabled=False,
        )

        registry.register(skill)

        assert "disabled_skill" not in registry.skills

    def test_unregister_skill(self):
        """测试注销技能"""
        registry = SkillRegistry()
        skill = Skill(
            skill_id="test_skill",
            display_name="测试技能",
            trigger=SkillTrigger(keywords=["测试"]),
        )

        registry.register(skill)
        assert "test_skill" in registry.skills

        result = registry.unregister("test_skill")
        assert result is True
        assert "test_skill" not in registry.skills

    def test_get_candidates_by_keyword(self):
        """测试通过关键词获取候选技能"""
        registry = SkillRegistry()

        skill1 = Skill(
            skill_id="skill1",
            display_name="技能1",
            trigger=SkillTrigger(keywords=["测试", "test"], priority=10),
        )
        skill2 = Skill(
            skill_id="skill2",
            display_name="技能2",
            trigger=SkillTrigger(keywords=["查询", "query"], priority=20),
        )

        registry.register(skill1)
        registry.register(skill2)

        # 测试关键词匹配
        candidates = registry.get_candidates("这是一个测试查询", max_count=5)
        assert len(candidates) == 2

        # 测试评分
        # "测试" 和 "查询" 都在查询中，各 +10 分
        # skill1 匹配 "测试": 10 分
        # skill2 匹配 "查询": 10 分
        # skill2 优先级更高 (20 vs 10)

    def test_get_candidates_by_pattern(self):
        """测试通过正则模式获取候选技能"""
        registry = SkillRegistry()

        skill = Skill(
            skill_id="pattern_skill",
            display_name="模式技能",
            trigger=SkillTrigger(
                patterns=[r"查.*指标"],
                priority=30,
            ),
        )

        registry.register(skill)

        candidates = registry.get_candidates("查一下昨天的指标", max_count=5)
        assert len(candidates) == 1
        assert candidates[0].skill_id == "pattern_skill"

    def test_get_candidates_max_count(self):
        """测试候选数量限制"""
        registry = SkillRegistry()

        for i in range(10):
            skill = Skill(
                skill_id=f"skill_{i}",
                display_name=f"技能{i}",
                trigger=SkillTrigger(keywords=["测试"], priority=i),
            )
            registry.register(skill)

        candidates = registry.get_candidates("测试", max_count=3)
        assert len(candidates) == 3

    def test_get_catalog(self):
        """测试获取目录"""
        registry = SkillRegistry()

        skill = Skill(
            skill_id="test_skill",
            display_name="测试技能",
            description="描述",
            skill_type="prompt",
            trigger=SkillTrigger(keywords=["测试"]),
        )
        registry.register(skill)

        catalog = registry.get_catalog()
        assert len(catalog) == 1
        assert catalog[0].skill_id == "test_skill"

    def test_get_stats(self):
        """测试获取统计信息"""
        registry = SkillRegistry()

        skill1 = Skill(
            skill_id="prompt_skill",
            display_name="Prompt技能",
            skill_type="prompt",
            trigger=SkillTrigger(keywords=["测试"]),
        )
        skill2 = Skill(
            skill_id="execution_skill",
            display_name="Execution技能",
            skill_type="execution",
            trigger=SkillTrigger(keywords=["执行"]),
        )

        registry.register(skill1)
        registry.register(skill2)

        stats = registry.get_stats()
        assert stats["total_skills"] == 2
        assert stats["type_distribution"]["prompt"] == 1
        assert stats["type_distribution"]["execution"] == 1

    def test_clear(self):
        """测试清空"""
        registry = SkillRegistry()
        skill = Skill(
            skill_id="test_skill",
            display_name="测试技能",
            trigger=SkillTrigger(keywords=["测试"]),
        )
        registry.register(skill)

        registry.clear()
        assert len(registry.skills) == 0
        assert len(registry._keyword_index) == 0
