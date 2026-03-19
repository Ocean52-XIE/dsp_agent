# -*- coding: utf-8 -*-
"""SkillManager 测试"""

import json
import pytest
from pathlib import Path

from agent.skills.base import Skill, SkillTrigger, ExecutionConfig
from agent.skills.registry import SkillRegistry
from agent.skills.executor import SkillExecutor
from agent.skills.manager import SkillManager, create_skill_manager


class TestSkillManager:
    """SkillManager 测试"""

    def test_init(self):
        """测试初始化"""
        registry = SkillRegistry()
        executor = SkillExecutor()
        manager = SkillManager(registry=registry, executor=executor)

        assert manager.registry is registry
        assert manager.executor is executor
        assert manager.name == "skill_manager"

    def test_init_with_candidate_skills(self):
        """测试带候选技能初始化"""
        registry = SkillRegistry()
        executor = SkillExecutor()

        # 注册技能
        skill = Skill(
            skill_id="test_skill",
            display_name="测试技能",
            description="这是一个测试技能",
            skill_type="prompt",
            trigger=SkillTrigger(keywords=["测试"]),
        )
        registry.register(skill)

        manager = SkillManager(
            registry=registry,
            executor=executor,
            candidate_skill_ids=["test_skill"],
        )

        assert "test_skill" in manager.candidate_skill_ids
        assert "test_skill" in manager.description

    def test_build_description_static(self):
        """测试静态描述生成"""
        registry = SkillRegistry()

        # 注册技能
        skill = Skill(
            skill_id="prompt_skill",
            display_name="Prompt 技能",
            description="这是一个 Prompt 技能",
            skill_type="prompt",
            trigger=SkillTrigger(keywords=["测试", "prompt"]),
        )
        registry.register(skill)

        desc = SkillManager._build_description_static(registry, None)

        assert "执行技能工具" in desc
        assert "prompt_skill" in desc
        assert "📄" in desc  # Prompt 类型图标

    def test_run_prompt_skill(self):
        """测试执行 Prompt Skill"""
        registry = SkillRegistry()
        executor = SkillExecutor()

        skill = Skill(
            skill_id="test_prompt",
            display_name="测试提示词",
            skill_type="prompt",
            prompt_template="你好，{{ name }}！",
        )
        registry.register(skill)

        manager = SkillManager(registry=registry, executor=executor)

        result = manager._run(
            skill_name="test_prompt",
            query="测试查询",
            params={"name": "张三"},
        )

        result_dict = json.loads(result)
        assert result_dict["success"] is True
        assert result_dict["skill_type"] == "prompt"
        assert "张三" in result_dict["prompt"]

    def test_run_execution_skill(self):
        """测试执行 Execution Skill"""
        registry = SkillRegistry()
        executor = SkillExecutor()

        skill = Skill(
            skill_id="test_execution",
            display_name="测试执行",
            skill_type="execution",
            execution=ExecutionConfig(
                command=["echo", "hello"],
                output="text",
            ),
        )
        registry.register(skill)

        manager = SkillManager(registry=registry, executor=executor)

        result = manager._run(
            skill_name="test_execution",
            query="执行命令",
            params={},
        )

        result_dict = json.loads(result)
        assert result_dict["success"] is True
        assert result_dict["skill_type"] == "execution"
        assert result_dict["data"] is not None

    def test_run_skill_not_found(self):
        """测试技能不存在"""
        registry = SkillRegistry()
        executor = SkillExecutor()
        manager = SkillManager(registry=registry, executor=executor)

        result = manager._run(
            skill_name="nonexistent_skill",
            query="测试",
            params={},
        )

        result_dict = json.loads(result)
        assert result_dict["success"] is False
        assert "不存在" in result_dict["error"]

    def test_run_skill_case_insensitive(self):
        """测试技能名称大小写不敏感"""
        registry = SkillRegistry()
        executor = SkillExecutor()

        skill = Skill(
            skill_id="Test_Skill",
            display_name="测试技能",
            skill_type="prompt",
            prompt_template="Hello",
        )
        registry.register(skill)

        manager = SkillManager(registry=registry, executor=executor)

        # 使用小写名称
        result = manager._run(
            skill_name="test_skill",
            query="测试",
            params={},
        )

        result_dict = json.loads(result)
        assert result_dict["success"] is True

    def test_resolve_skill(self):
        """测试技能名称解析"""
        registry = SkillRegistry()
        executor = SkillExecutor()

        skill = Skill(
            skill_id="MySkill",
            display_name="我的技能",
        )
        registry.register(skill)

        manager = SkillManager(registry=registry, executor=executor)

        # 精确匹配
        resolved = manager._resolve_skill("MySkill")
        assert resolved is not None
        assert resolved.skill_id == "MySkill"

        # 大小写不敏感
        resolved = manager._resolve_skill("myskill")
        assert resolved is not None

        # 不存在
        resolved = manager._resolve_skill("nonexistent")
        assert resolved is None

    def test_format_result_success_prompt(self):
        """测试格式化成功的 Prompt 结果"""
        from agent.skills.executor import SkillExecutionResult

        registry = SkillRegistry()
        executor = SkillExecutor()
        manager = SkillManager(registry=registry, executor=executor)

        result = SkillExecutionResult(
            skill_id="test",
            skill_type="prompt",
            success=True,
            prompt="生成的提示词",
        )

        formatted = manager._format_result(result)
        result_dict = json.loads(formatted)

        assert result_dict["success"] is True
        assert result_dict["prompt"] == "生成的提示词"

    def test_format_result_success_execution(self):
        """测试格式化成功的 Execution 结果"""
        from agent.skills.executor import SkillExecutionResult

        registry = SkillRegistry()
        executor = SkillExecutor()
        manager = SkillManager(registry=registry, executor=executor)

        result = SkillExecutionResult(
            skill_id="test",
            skill_type="execution",
            success=True,
            data={"key": "value"},
        )

        formatted = manager._format_result(result)
        result_dict = json.loads(formatted)

        assert result_dict["success"] is True
        assert result_dict["data"] == {"key": "value"}

    def test_format_result_failure(self):
        """测试格式化失败的结果"""
        from agent.skills.executor import SkillExecutionResult

        registry = SkillRegistry()
        executor = SkillExecutor()
        manager = SkillManager(registry=registry, executor=executor)

        result = SkillExecutionResult(
            skill_id="test",
            skill_type="execution",
            success=False,
            error="执行失败",
        )

        formatted = manager._format_result(result)
        result_dict = json.loads(formatted)

        assert result_dict["success"] is False
        assert result_dict["error"] == "执行失败"

    def test_create_skill_manager(self):
        """测试创建 SkillManager 的便捷函数"""
        registry = SkillRegistry()
        manager = create_skill_manager(registry)

        assert manager.registry is registry
        assert manager.executor is not None

    def test_create_skill_manager_with_executor(self):
        """测试带执行器创建 SkillManager"""
        registry = SkillRegistry()
        executor = SkillExecutor()
        manager = create_skill_manager(registry, executor=executor)

        assert manager.executor is executor

    def test_manager_without_registry(self):
        """测试没有 registry 的情况"""
        manager = SkillManager()

        result = manager._run(
            skill_name="test",
            query="测试",
            params={},
        )

        result_dict = json.loads(result)
        assert result_dict["success"] is False
        assert "未初始化" in result_dict["error"]


class TestSkillManagerInput:
    """SkillManagerInput 测试"""

    def test_valid_input(self):
        """测试有效输入"""
        from agent.skills.manager import SkillManagerInput

        input_data = SkillManagerInput(
            skill_name="test_skill",
            query="测试查询",
            params={"key": "value"},
        )

        assert input_data.skill_name == "test_skill"
        assert input_data.query == "测试查询"
        assert input_data.params == {"key": "value"}

    def test_optional_params(self):
        """测试可选参数"""
        from agent.skills.manager import SkillManagerInput

        input_data = SkillManagerInput(
            skill_name="test_skill",
            query="测试查询",
        )

        assert input_data.params is None
