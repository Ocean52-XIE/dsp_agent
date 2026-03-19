# -*- coding: utf-8 -*-
"""Skill 数据结构测试"""

import pytest
from pathlib import Path

from agent.skills.base import (
    Skill,
    SkillTrigger,
    SkillReference,
    ExecutionConfig,
    SkillCatalogItem,
)


class TestSkillTrigger:
    """SkillTrigger 测试"""

    def test_default_values(self):
        """测试默认值"""
        trigger = SkillTrigger()
        assert trigger.keywords == []
        assert trigger.patterns == []
        assert trigger.priority == 10

    def test_custom_values(self):
        """测试自定义值"""
        trigger = SkillTrigger(
            keywords=["测试", "test"],
            patterns=[r"测试.*"],
            priority=30,
        )
        assert trigger.keywords == ["测试", "test"]
        assert trigger.patterns == [r"测试.*"]
        assert trigger.priority == 30


class TestSkillReference:
    """SkillReference 测试"""

    def test_lazy_load(self, tmp_path):
        """测试延迟加载"""
        ref_file = tmp_path / "test.md"
        ref_file.write_text("测试内容", encoding="utf-8")

        ref = SkillReference(
            name="test.md",
            path=ref_file,
        )

        # 首次访问时加载
        assert ref.content is None
        content = ref.load_content()
        assert content == "测试内容"
        assert ref.content == "测试内容"

        # 再次访问使用缓存
        content2 = ref.load_content()
        assert content2 == "测试内容"


class TestExecutionConfig:
    """ExecutionConfig 测试"""

    def test_string_command(self):
        """测试字符串命令"""
        config = ExecutionConfig(command="echo hello")
        assert config.command == "echo hello"
        assert config.timeout == 30
        assert config.output == "text"

    def test_array_command(self):
        """测试数组命令"""
        config = ExecutionConfig(
            command=["curl", "-s", "https://example.com"],
            timeout=60,
            output="json",
            extract="data.records",
        )
        assert config.command == ["curl", "-s", "https://example.com"]
        assert config.timeout == 60
        assert config.output == "json"
        assert config.extract == "data.records"


class TestSkill:
    """Skill 测试"""

    def test_prompt_skill_creation(self):
        """测试 Prompt Skill 创建"""
        skill = Skill(
            skill_id="test_skill",
            display_name="测试技能",
            description="这是一个测试技能",
            skill_type="prompt",
            prompt_template="Hello {{ name }}",
        )

        assert skill.skill_id == "test_skill"
        assert skill.is_prompt() is True
        assert skill.is_execution() is False

    def test_execution_skill_creation(self):
        """测试 Execution Skill 创建"""
        skill = Skill(
            skill_id="test_exec",
            display_name="执行技能",
            skill_type="execution",
            execution=ExecutionConfig(command="echo test"),
        )

        assert skill.skill_id == "test_exec"
        assert skill.is_prompt() is False
        assert skill.is_execution() is True

    def test_get_keywords(self):
        """测试获取关键词"""
        skill = Skill(
            skill_id="test",
            display_name="Test",
            trigger=SkillTrigger(keywords=["测试", "test"]),
        )

        assert skill.get_keywords() == ["测试", "test"]

    def test_get_patterns(self):
        """测试获取正则模式"""
        skill = Skill(
            skill_id="test",
            display_name="Test",
            trigger=SkillTrigger(patterns=[r"测试.*"]),
        )

        assert skill.get_patterns() == [r"测试.*"]

    def test_get_priority(self):
        """测试获取优先级"""
        skill = Skill(
            skill_id="test",
            display_name="Test",
            trigger=SkillTrigger(priority=30),
        )

        assert skill.get_priority() == 30

    def test_to_catalog_item(self):
        """测试转换为目录项"""
        skill = Skill(
            skill_id="test_skill",
            display_name="测试技能",
            description="测试描述",
            skill_type="prompt",
            tags=["test"],
            trigger=SkillTrigger(keywords=["测试"]),
        )

        item = skill.to_catalog_item()

        assert isinstance(item, SkillCatalogItem)
        assert item.skill_id == "test_skill"
        assert item.display_name == "测试技能"
        assert item.description == "测试描述"
        assert item.skill_type == "prompt"
        assert item.tags == ["test"]
        assert item.keywords == ["测试"]

    def test_validate_params_success(self):
        """测试参数校验成功"""
        skill = Skill(
            skill_id="test",
            display_name="Test",
            params={
                "name": {"type": "string", "required": True},
                "age": {"type": "integer"},
            },
        )

        ok, missing = skill.validate_params({"name": "张三", "age": 25})
        assert ok is True
        assert missing == []

    def test_validate_params_missing_required(self):
        """测试参数校验缺少必填参数"""
        skill = Skill(
            skill_id="test",
            display_name="Test",
            params={
                "name": {"type": "string", "required": True},
            },
        )

        ok, missing = skill.validate_params({})
        assert ok is False
        assert "name" in missing

    def test_get_required_params(self):
        """测试获取必填参数列表"""
        skill = Skill(
            skill_id="test",
            display_name="Test",
            params={
                "name": {"type": "string", "required": True},
                "age": {"type": "integer"},
                "email": {"type": "string", "required": True},
            },
        )

        required = skill.get_required_params()
        assert set(required) == {"name", "email"}
