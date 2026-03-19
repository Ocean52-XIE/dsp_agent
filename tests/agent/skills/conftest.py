# -*- coding: utf-8 -*-
"""Skill 模块测试配置"""

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_skill_dir():
    """创建临时技能目录"""
    with tempfile.TemporaryDirectory() as tmpdir:
        skills_dir = Path(tmpdir) / "skills"
        skills_dir.mkdir(parents=True)
        yield tmpdir, skills_dir


@pytest.fixture
def sample_prompt_skill_content():
    """Prompt Skill 示例内容"""
    return '''---
skill_id: test_prompt_skill
display_name: 测试提示词技能
description: 这是一个测试用的 Prompt Skill
version: 1.0.0
tags:
  - test
  - prompt

skill_type: prompt

trigger:
  keywords:
    - 测试
    - prompt
  patterns:
    - "测试.*prompt"
  priority: 20

params:
  name:
    type: string
    description: 名称参数
    required: true
  style:
    type: string
    description: 风格
    default: "normal"

examples:
  - user: "测试一下"
    params:
      name: "测试"
---

你是一个测试助手。

用户名称: {{ name }}
{% if style %}风格: {{ style }}{% endif %}

请回复用户。
'''


@pytest.fixture
def sample_execution_skill_content():
    """Execution Skill 示例内容"""
    return '''---
skill_id: test_execution_skill
display_name: 测试执行技能
description: 这是一个测试用的 Execution Skill
version: 1.0.0
tags:
  - test
  - execution

skill_type: execution

trigger:
  keywords:
    - 执行
    - 命令
  priority: 30

params:
  input:
    type: string
    description: 输入参数
    required: true

execution:
  command: "echo hello {{ input }}"
  timeout: 10
  output: text

examples:
  - user: "执行测试命令"
    params:
      input: "world"
---

执行测试命令。
'''


@pytest.fixture
def sample_prompt_skill_file(temp_skill_dir, sample_prompt_skill_content):
    """创建示例 Prompt Skill 文件"""
    tmpdir, skills_dir = temp_skill_dir
    skill_dir = skills_dir / "test_prompt_skill"
    skill_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(sample_prompt_skill_content, encoding="utf-8")
    return Path(tmpdir), skill_file


@pytest.fixture
def sample_execution_skill_file(temp_skill_dir, sample_execution_skill_content):
    """创建示例 Execution Skill 文件"""
    tmpdir, skills_dir = temp_skill_dir
    skill_dir = skills_dir / "test_execution_skill"
    skill_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(sample_execution_skill_content, encoding="utf-8")
    return Path(tmpdir), skill_file


@pytest.fixture
def sample_skills_dir(
    sample_prompt_skill_file,
    sample_execution_skill_file,
):
    """创建包含多个示例技能的目录"""
    prompt_tmpdir, _ = sample_prompt_skill_file
    execution_tmpdir, _ = sample_execution_skill_file
    # 返回第一个临时目录（两个应该是同一个）
    return prompt_tmpdir
