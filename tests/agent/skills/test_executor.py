# -*- coding: utf-8 -*-
"""SkillExecutor 测试"""

import pytest
from pathlib import Path

from agent.skills.base import Skill, SkillTrigger, ExecutionConfig
from agent.skills.executor import SkillExecutor, SkillExecutionResult


class TestSkillExecutor:
    """SkillExecutor 测试"""

    def test_init(self):
        """测试初始化"""
        executor = SkillExecutor()
        assert executor._jinja_env is not None

    def test_execute_prompt_skill_success(self):
        """测试执行 Prompt Skill 成功"""
        executor = SkillExecutor()

        skill = Skill(
            skill_id="test_prompt",
            display_name="测试提示词",
            skill_type="prompt",
            prompt_template="你好，{{ name }}！今天是 {{ current_time }}。",
        )

        result = executor.execute(skill, {"name": "张三"}, "测试查询")

        assert result.success is True
        assert result.skill_id == "test_prompt"
        assert result.skill_type == "prompt"
        assert "张三" in result.prompt
        assert result.latency_ms >= 0

    def test_execute_prompt_skill_with_reference(self, tmp_path):
        """测试执行 Prompt Skill 带参考文档"""
        from agent.skills.base import SkillReference

        # 创建参考文档
        ref_file = tmp_path / "guide.md"
        ref_file.write_text("这是参考指南内容", encoding="utf-8")

        executor = SkillExecutor()

        skill = Skill(
            skill_id="test_prompt",
            display_name="测试提示词",
            skill_type="prompt",
            prompt_template="参考: {{ references.guide_md }}",
            references=[
                SkillReference(name="guide.md", path=ref_file)
            ],
        )

        result = executor.execute(skill, {}, "测试")

        assert result.success is True
        assert "这是参考指南内容" in result.prompt

    def test_execute_prompt_skill_missing_template(self):
        """测试执行 Prompt Skill 缺少模板"""
        executor = SkillExecutor()

        skill = Skill(
            skill_id="test_prompt",
            display_name="测试提示词",
            skill_type="prompt",
            prompt_template="",
        )

        result = executor.execute(skill, {}, "测试")

        assert result.success is False
        assert "未定义 prompt_template" in result.error

    def test_execute_execution_skill_echo_command(self):
        """测试执行 Execution Skill echo 命令"""
        executor = SkillExecutor()

        skill = Skill(
            skill_id="test_execution",
            display_name="测试执行",
            skill_type="execution",
            execution=ExecutionConfig(
                command=["echo", "hello world"],
                output="text",
            ),
        )

        result = executor.execute(skill, {}, "测试")

        assert result.success is True
        assert result.skill_id == "test_execution"
        assert result.skill_type == "execution"
        assert result.data is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].success is True

    def test_execute_execution_skill_with_params(self):
        """测试执行 Execution Skill 带参数"""
        executor = SkillExecutor()

        skill = Skill(
            skill_id="test_execution",
            display_name="测试执行",
            skill_type="execution",
            execution=ExecutionConfig(
                command="echo {{ message }}",
                output="text",
            ),
        )

        result = executor.execute(skill, {"message": "test123"}, "测试")

        assert result.success is True
        assert "test123" in result.data.get("text", "")

    def test_execute_execution_skill_json_output(self):
        """测试执行 Execution Skill JSON 输出"""
        executor = SkillExecutor()

        skill = Skill(
            skill_id="test_execution",
            display_name="测试执行",
            skill_type="execution",
            execution=ExecutionConfig(
                command=["echo", '{"data": {"value": 123}}'],
                output="json",
                extract="data.value",
            ),
        )

        result = executor.execute(skill, {}, "测试")

        assert result.success is True
        assert result.data == 123

    def test_execute_execution_skill_command_failed(self):
        """测试执行 Execution Skill 命令失败"""
        executor = SkillExecutor()

        skill = Skill(
            skill_id="test_execution",
            display_name="测试执行",
            skill_type="execution",
            execution=ExecutionConfig(
                # 使用一个不存在的命令
                command=["nonexistent_command_12345"],
                timeout=5,
            ),
        )

        result = executor.execute(skill, {}, "测试")

        assert result.success is False
        assert result.error is not None

    def test_execute_execution_skill_missing_config(self):
        """测试执行 Execution Skill 缺少配置"""
        executor = SkillExecutor()

        skill = Skill(
            skill_id="test_execution",
            display_name="测试执行",
            skill_type="execution",
            execution=None,
        )

        result = executor.execute(skill, {}, "测试")

        assert result.success is False
        assert "未配置 execution" in result.error

    def test_execute_execution_skill_timeout(self):
        """测试执行 Execution Skill 超时"""
        import platform

        executor = SkillExecutor()

        # Windows 和 Unix 使用不同的 sleep 命令
        if platform.system() == "Windows":
            cmd = ["ping", "-n", "10", "127.0.0.1"]
        else:
            cmd = ["sleep", "10"]

        skill = Skill(
            skill_id="test_timeout",
            display_name="测试超时",
            skill_type="execution",
            execution=ExecutionConfig(
                command=cmd,
                timeout=1,  # 1秒超时
            ),
        )

        result = executor.execute(skill, {}, "测试")

        assert result.success is False
        assert "超时" in result.error

    def test_render_template(self):
        """测试模板渲染"""
        executor = SkillExecutor()

        template = "Hello {{ name }}, you are {{ age }} years old."
        params = {"name": "Alice", "age": 25}

        result = executor._render_template(template, params)
        assert "Alice" in result
        assert "25" in result

    def test_resolve_env_value(self):
        """测试环境变量解析"""
        import os

        executor = SkillExecutor()
        os.environ["TEST_VAR"] = "test_value"

        # 测试 ${VAR} 格式
        result = executor._resolve_env_value("${TEST_VAR}", {})
        assert result == "test_value"

        # 测试混合格式
        result = executor._resolve_env_value("prefix_${TEST_VAR}_suffix", {})
        assert result == "prefix_test_value_suffix"

    def test_parse_output_json(self):
        """测试 JSON 输出解析"""
        executor = SkillExecutor()

        output = '{"data": {"records": [1, 2, 3]}}'

        # 不提取
        result = executor._parse_output(output, "json", None)
        assert result["data"]["records"] == [1, 2, 3]

        # 提取
        result = executor._parse_output(output, "json", "data.records")
        assert result == [1, 2, 3]

    def test_parse_output_text(self):
        """测试文本输出解析"""
        executor = SkillExecutor()

        output = "Hello World"
        result = executor._parse_output(output, "text", None)
        assert result == {"text": "Hello World"}

    def test_parse_output_raw(self):
        """测试原始输出解析"""
        executor = SkillExecutor()

        output = "Raw content"
        result = executor._parse_output(output, "raw", None)
        assert result == {"raw": "Raw content"}
