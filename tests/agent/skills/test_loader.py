# -*- coding: utf-8 -*-
"""SkillLoader 测试"""

import pytest
from pathlib import Path

from agent.skills.loader import SkillLoader, SkillLoadError
from agent.skills.base import Skill


class TestSkillLoader:
    """SkillLoader 测试"""

    def test_init(self, temp_skill_dir):
        """测试初始化"""
        tmpdir, _ = temp_skill_dir
        loader = SkillLoader(Path(tmpdir))
        assert loader.domain_root == Path(tmpdir)
        assert loader.skills == {}

    def test_load_all_empty(self, temp_skill_dir):
        """测试加载空目录"""
        tmpdir, _ = temp_skill_dir
        loader = SkillLoader(Path(tmpdir))
        skills = loader.load_all()
        assert skills == {}

    def test_load_prompt_skill(self, sample_prompt_skill_file):
        """测试加载 Prompt Skill"""
        tmpdir, skill_file = sample_prompt_skill_file
        loader = SkillLoader(Path(tmpdir))
        skills = loader.load_all()

        assert len(skills) == 1
        assert "test_prompt_skill" in skills

        skill = skills["test_prompt_skill"]
        assert skill.skill_id == "test_prompt_skill"
        assert skill.display_name == "测试提示词技能"
        assert skill.skill_type == "prompt"
        assert skill.prompt_template != ""
        assert "测试" in skill.trigger.keywords[0]

    def test_load_execution_skill(self, sample_execution_skill_file):
        """测试加载 Execution Skill"""
        tmpdir, skill_file = sample_execution_skill_file
        loader = SkillLoader(Path(tmpdir))
        skills = loader.load_all()

        assert len(skills) == 1
        assert "test_execution_skill" in skills

        skill = skills["test_execution_skill"]
        assert skill.skill_id == "test_execution_skill"
        assert skill.skill_type == "execution"
        assert skill.execution is not None
        assert skill.execution.command == "echo hello {{ input }}"
        assert skill.execution.timeout == 10
        assert skill.execution.output == "text"

    def test_load_multiple_skills(
        self,
        sample_prompt_skill_file,
        sample_execution_skill_file,
    ):
        """测试加载多个技能"""
        tmpdir, _ = sample_prompt_skill_file
        loader = SkillLoader(Path(tmpdir))
        skills = loader.load_all()

        assert len(skills) == 2
        assert "test_prompt_skill" in skills
        assert "test_execution_skill" in skills

    def test_parse_front_matter(self, temp_skill_dir):
        """测试解析 YAML front matter"""
        tmpdir, _ = temp_skill_dir
        loader = SkillLoader(Path(tmpdir))

        content = """---
skill_id: test
name: Test
---
Markdown content
"""
        front_matter, markdown = loader._parse_front_matter(content)

        assert front_matter["skill_id"] == "test"
        assert front_matter["name"] == "Test"
        assert markdown == "Markdown content"

    def test_parse_front_matter_no_yaml(self, temp_skill_dir):
        """测试没有 YAML front matter 的情况"""
        tmpdir, _ = temp_skill_dir
        loader = SkillLoader(Path(tmpdir))

        content = "Just markdown content"
        front_matter, markdown = loader._parse_front_matter(content)

        assert front_matter == {}
        assert markdown == "Just markdown content"

    def test_load_skill_with_references(self, temp_skill_dir):
        """测试加载带参考文档的技能"""
        tmpdir, skills_dir = temp_skill_dir

        # 创建技能目录
        skill_dir = skills_dir / "ref_skill"
        skill_dir.mkdir(parents=True)

        # 创建参考文档
        refs_dir = skill_dir / "references"
        refs_dir.mkdir(parents=True)
        ref_file = refs_dir / "guide.md"
        ref_file.write_text("参考指南内容", encoding="utf-8")

        # 创建技能文件
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text("""---
skill_id: ref_skill
display_name: 带参考的技能
skill_type: prompt
---
模板内容
""", encoding="utf-8")

        loader = SkillLoader(Path(tmpdir))
        skills = loader.load_all()

        assert "ref_skill" in skills
        skill = skills["ref_skill"]
        assert len(skill.references) == 1
        assert skill.references[0].name == "guide.md"

    def test_load_skill_file_invalid_yaml(self, temp_skill_dir):
        """测试加载无效 YAML 的技能文件"""
        tmpdir, skills_dir = temp_skill_dir

        skill_dir = skills_dir / "invalid_skill"
        skill_dir.mkdir(parents=True)
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text("""---
invalid yaml content [[[
---
Content
""", encoding="utf-8")

        loader = SkillLoader(Path(tmpdir))
        skills = loader.load_all()

        # 无效 YAML 的技能加载会失败，不应该出现在结果中
        # 因为 front_matter 解析后不是字典，后续处理会出错
        assert len(skills) == 0

    def test_get_skill(self, sample_prompt_skill_file):
        """测试获取指定技能"""
        tmpdir, _ = sample_prompt_skill_file
        loader = SkillLoader(Path(tmpdir))
        loader.load_all()

        skill = loader.get_skill("test_prompt_skill")
        assert skill is not None
        assert skill.skill_id == "test_prompt_skill"

        skill = loader.get_skill("nonexistent")
        assert skill is None

    def test_list_skills(self, sample_prompt_skill_file):
        """测试获取技能列表"""
        tmpdir, _ = sample_prompt_skill_file
        loader = SkillLoader(Path(tmpdir))
        loader.load_all()

        skills = loader.list_skills()
        assert len(skills) == 1

    def test_reload(self, sample_prompt_skill_file):
        """测试重新加载"""
        tmpdir, _ = sample_prompt_skill_file
        loader = SkillLoader(Path(tmpdir))
        loader.load_all()

        # 重新加载
        skills = loader.reload()
        assert len(skills) == 1


class TestSkillLoadError:
    """SkillLoadError 测试"""

    def test_error_message(self):
        """测试错误消息"""
        path = Path("/tmp/test.md")
        error = SkillLoadError(path, "解析失败")

        assert error.path == path
        assert error.message == "解析失败"
        # 跨平台检查：Windows 使用反斜杠
        error_str = str(error)
        assert "加载" in error_str and "失败" in error_str
        assert "test.md" in error_str
        assert "解析失败" in error_str
