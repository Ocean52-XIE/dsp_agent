# Phase 3: Skill 系统设计

## 1. 概述

### 1.1 目标

在 `agent/` 目录中独立实现 Skill 系统，支持两种技能类型：
- **Prompt Skill**：渲染提示词模板，返回 prompt 给 LLM
- **Execution Skill**：执行命令（CLI），返回结构化数据

### 1.2 核心设计原则

| 原则 | 说明 |
|------|------|
| **简洁优先** | 只支持两种类型，去除不必要的复杂度 |
| **统一执行** | Execution Skill 只支持 Command，handler.py 也是 Command 的一种 |
| **代码隔离** | 不复用 `workflow/` 目录代码，完全独立实现 |
| **Agent 集成** | Skill 通过 ToolRegistry 注册为 LangChain Tool |
| **领域隔离** | Skill 定义文件存放在 `domain/<domain_id>/skills/` |

### 1.3 与 v1 的差异

| 特性 | v1 (workflow/skills) | v2 (agent/skills) |
|------|---------------------|-------------------|
| Skill 类型 | 3 种 (tool_execution, prompt_template, llm_orchestrated) | **2 种 (prompt, execution)** |
| 执行方式 | handler.py + external_system | **仅 Command** |
| 工具白名单 | 支持 | **移除**（由 Agent Loop 统一管理） |
| MCP 集成 | 无 | **通过 MCP Tool 实现，不在此模块** |

---

## 2. Skill Spec 设计

### 2.1 两种 Skill 类型对比

| 特性 | Prompt Skill | Execution Skill |
|------|--------------|-----------------|
| **用途** | 生成高质量提示词 | 执行命令获取数据 |
| **返回** | 渲染后的 prompt 字符串 | 结构化数据 (dict) |
| **执行方式** | Jinja2 模板渲染 | CLI 命令执行 |
| **适用场景** | 文案生成、代码生成、复杂推理 | 数据查询、API 调用、脚本执行 |
| **配置复杂度** | 简单 | 中等 |

### 2.2 完整 Spec Schema

```yaml
---
# === 基础信息 ===
skill_id: string                    # 必填，技能唯一标识
display_name: string                # 显示名称
description: string                 # 技能描述
version: string                     # 版本号，默认 "1.0.0"
tags: [string]                      # 标签列表
enabled: boolean                    # 是否启用，默认 true

# === 类型定义 ===
skill_type: prompt | execution      # 必填，技能类型

# === 触发配置 ===
trigger:
  keywords: [string]                # 触发关键词
  patterns: [string]                # 触发正则模式
  priority: integer                 # 优先级（越大越优先），默认 10

# === 参数定义 ===
params:
  param_name:
    type: string | integer | number | boolean | array | object
    description: string
    required: boolean
    default: any
    enum: [any]                     # 枚举值

# === Prompt Skill 专用 ===
# （prompt_template 在 YAML 后的 Markdown 部分）

# === Execution Skill 专用 ===
execution:
  command: string | [string]        # 必填，命令（字符串或数组）
  cwd: string                       # 工作目录，默认为 skill 目录
  env:                              # 环境变量
    KEY: value                      # 支持 ${ENV_VAR} 引用系统环境变量
  timeout: integer                  # 超时秒数，默认 30
  output: json | text | raw         # 输出格式，默认 text
  extract: string                   # JSON 提取路径（如 "data.records"）

# 注意：Execution Skill 的参数通过 stdin 以 JSON 格式传递
# command 不再支持模板语法（如 {{ param }}），参数由脚本自行解析

# === 示例 ===
examples:
  - user: string                    # 用户问题示例
    params:                         # 对应参数
      key: value
---

# Markdown 内容
# - Prompt Skill: 作为 prompt_template（Jinja2 模板）
# - Execution Skill: 作为使用说明
```

---

## 3. SKILL.md 文件格式示例

### 3.1 目录结构

```
domain/ad_engine/skills/
├── ad_copy_generator/              # Prompt Skill 示例
│   ├── SKILL.md                    # 技能定义
│   └── references/                 # 参考文档（可选）
│       └── copywriting_guide.md
│
├── query_metrics/                  # Execution Skill 示例（curl）
│   └── SKILL.md
│
├── code_search/                    # Execution Skill 示例（Python 脚本）
│   └── SKILL.md
│
└── git_info/                       # Execution Skill 示例（系统命令）
    └── SKILL.md
```

### 3.2 Prompt Skill 示例

```yaml
# domain/ad_engine/skills/ad_copy_generator/SKILL.md
---
skill_id: ad_copy_generator
display_name: 广告文案生成器
description: 根据产品信息和目标受众生成吸引人的广告文案
version: 1.0.0
tags:
  - 文案
  - 创意
  - 广告

skill_type: prompt

trigger:
  keywords:
    - 生成文案
    - 写广告
    - 文案创作
    - 广告语
  patterns:
    - "帮我.*写.*文案"
    - "生成.*广告"
  priority: 20

params:
  product_name:
    type: string
    description: 产品名称
    required: true
  target_audience:
    type: string
    description: 目标受众
    required: true
  style:
    type: string
    description: 文案风格
    enum: [formal, casual, creative]
    default: creative
  key_points:
    type: array
    items:
      type: string
    description: 需要突出的卖点

examples:
  - user: "帮我给新款手机写个广告文案，目标受众是年轻人"
    params:
      product_name: "新款智能手机"
      target_audience: "年轻人"
      style: "creative"
  - user: "生成一个护肤品的正式广告"
    params:
      product_name: "护肤精华"
      style: "formal"
---

你是一位资深的广告文案创意总监，擅长创作引人注目的广告文案。

## 任务
为以下产品创作广告文案：

**产品名称**: {{ product_name }}
**目标受众**: {{ target_audience }}
{% if style %}
**文案风格**: {% if style == 'formal' %}正式专业{% elif style == 'casual' %}轻松活泼{% else %}创意新颖{% endif %}
{% endif %}
{% if key_points %}

**需要突出的卖点**:
{% for point in key_points %}
- {{ point }}
{% endfor %}
{% endif %}

## 要求
1. 文案长度控制在 50-100 字
2. 突出产品核心价值
3. 符合目标受众的语言习惯
4. 具有吸引力和记忆点

请直接输出广告文案，无需解释。
```

### 3.3 Execution Skill 示例 - Python 脚本（推荐方式）

> **重要**: Execution Skill 的参数通过 stdin 以 JSON 格式传递，脚本自行处理默认值和参数验证。

```yaml
# domain/ad_engine/skills/query_ad_info/SKILL.md
---
skill_id: query_ad_info
display_name: 广告信息查询
description: 查询广告系统的广告计划、广告组、创意等信息
version: 1.0.0
tags:
  - 广告
  - 查询
  - ad

skill_type: execution

trigger:
  keywords:
    - 查询广告
    - 广告计划
    - 广告组
    - 创意
  patterns:
    - "查.*广告"
    - "获取.*广告.*信息"
  priority: 30

params:
  ad_type:
    type: string
    description: 广告类型
    enum: [campaign, adgroup, creative]
  ad_id:
    type: string
    description: 广告 ID（可选）
  status:
    type: string
    description: 状态过滤
    enum: [active, paused, archived, all]
  limit:
    type: integer
    description: 返回数量限制

execution:
  # 调用 Python 脚本，参数通过 stdin 以 JSON 格式传递
  command:
    - "python"
    - "scripts/query_ad.py"
  timeout: 30
  output: json

examples:
  - user: "查询所有活跃的广告计划"
    params:
      ad_type: campaign
      status: active
  - user: "查看广告组 12345 的详情"
    params:
      ad_type: adgroup
      ad_id: "12345"
---

查询广告系统中的广告信息。

## 输入参数

参数通过 stdin 以 JSON 格式传递，脚本会自动处理默认值。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| ad_type | string | 是 | - | 广告类型 |
| ad_id | string | 否 | - | 广告 ID |
| status | string | 否 | active | 状态过滤 |
| limit | integer | 否 | 10 | 返回数量限制 |
```

**对应的 Python 脚本示例**:

```python
# scripts/query_ad.py
# -*- coding: utf-8 -*-
"""广告信息查询脚本

使用方式（JSON+stdin 模式）：
    echo '{"ad_type": "campaign", "status": "active"}' | python scripts/query_ad.py
"""

import json
import sys

# 默认参数值
DEFAULT_PARAMS = {
    "ad_type": None,      # 必填，无默认值
    "ad_id": None,        # 可选
    "status": "active",   # 默认值
    "limit": 10,          # 默认值
}


def main():
    # 1. 从 stdin 读取 JSON
    params = {}
    try:
        params_text = sys.stdin.read()
        if params_text.strip():
            params = json.loads(params_text)
    except json.JSONDecodeError as e:
        print(json.dumps({"success": False, "error": f"JSON 解析失败: {e}"}))
        sys.exit(1)

    # 2. 合并默认值（用户参数覆盖默认值）
    merged = {**DEFAULT_PARAMS, **params}

    # 3. 验证必填参数
    if not merged["ad_type"]:
        print(json.dumps({"success": False, "error": "ad_type 是必填参数"}))
        sys.exit(1)

    # 4. 执行业务逻辑...
    result = do_query(merged)

    # 5. 输出 JSON 结果
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

### 3.4 Execution Skill 示例 - Git 信息查询

```yaml
# domain/ad_engine/skills/git_info/SKILL.md
---
skill_id: git_info
display_name: Git 信息查询
description: 查询 Git 仓库的提交记录、分支状态等信息
version: 1.0.0
tags:
  - git
  - 版本控制
  - 代码仓库

skill_type: execution

trigger:
  keywords:
    - git
    - commit
    - 分支
    - 提交
  patterns:
    - "查看.*提交"
    - "git.*状态"
    - "最近.*commit"
  priority: 15

params:
  command:
    type: string
    description: Git 子命令
    enum: [log, status, branch, diff, show]
  limit:
    type: integer
    description: 返回数量限制（用于 log 命令）
  branch:
    type: string
    description: 指定分支名称（可选）

execution:
  # 调用 Python 脚本，参数通过 stdin 以 JSON 格式传递
  command:
    - "python"
    - "scripts/git_info.py"
  timeout: 30
  output: json

examples:
  - user: "查看最近的提交记录"
    params:
      command: log
      limit: 5
  - user: "当前分支状态是什么"
    params:
      command: status
  - user: "列出所有分支"
    params:
      command: branch
---

查询 Git 仓库信息。

## 输入参数

参数通过 stdin 以 JSON 格式传递，脚本会自动处理默认值。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| command | string | 否 | status | Git 子命令 |
| limit | integer | 否 | 10 | 返回数量限制 |
| branch | string | 否 | - | 指定分支名称 |

## 支持的命令

| 命令 | 说明 | 用途 |
|------|------|------|
| log | 查看提交历史 | 了解代码变更记录 |
| status | 查看工作区状态 | 检查未提交的变更 |
| branch | 列出分支 | 了解分支结构 |
| diff | 查看差异 | 比较代码变更 |
| show | 查看提交详情 | 了解具体提交内容 |
```

**对应的 Python 脚本示例**:

```python
# scripts/git_info.py
# -*- coding: utf-8 -*-
"""Git 信息查询脚本

使用方式（JSON+stdin 模式）：
    echo '{"command": "log", "limit": 5}' | python scripts/git_info.py
    echo '{"command": "status"}' | python scripts/git_info.py
"""

import json
import subprocess
import sys

# 默认参数值
DEFAULT_PARAMS = {
    "command": "status",
    "limit": 10,
    "branch": None,
}


def build_git_command(command: str, limit: int, branch: str | None) -> list[str]:
    """构建 Git 命令"""
    cmd = ["git"]

    if command == "log":
        cmd.extend(["log", "--oneline", "-n", str(limit)])
        if branch:
            cmd.append(branch)
    elif command == "status":
        cmd.append("status")
    elif command == "branch":
        cmd.extend(["branch", "-a"])
    elif command == "diff":
        cmd.append("diff")
    elif command == "show":
        cmd.append("show")

    return cmd


def main():
    # 1. 从 stdin 读取 JSON
    params = {}
    try:
        params_text = sys.stdin.read()
        if params_text.strip():
            params = json.loads(params_text)
    except json.JSONDecodeError as e:
        print(json.dumps({"success": False, "error": f"JSON 解析失败: {e}"}))
        sys.exit(1)

    # 2. 合并默认值
    merged = {**DEFAULT_PARAMS, **params}

    # 3. 构建并执行 Git 命令
    cmd = build_git_command(
        command=merged["command"],
        limit=merged["limit"],
        branch=merged["branch"],
    )

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )

        print(json.dumps({
            "success": result.returncode == 0,
            "command": merged["command"],
            "output": result.stdout.strip(),
            "error": result.stderr.strip() if result.returncode != 0 else None,
        }, ensure_ascii=False))

    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
```

---

## 4. Execution Skill 参数传递规范

### 4.1 统一使用 JSON+stdin 模式

所有 Execution Skill 的参数传递统一采用 **JSON+stdin** 模式：

```
┌─────────────────────────────────────────────────────────────────┐
│                        参数传递流程                              │
└─────────────────────────────────────────────────────────────────┘
    LLM 识别参数
         │
         ▼
    { "ad_type": "campaign", "limit": 20 }   // LLM 只返回识别到的参数
         │
         ▼
    SkillExecutor.execute(skill, params)
         │
         ▼
    json.dumps(params) → stdin
         │
         ▼
    ┌─────────────────────────────────────┐
    │  Python 脚本执行                     │
    │  1. 读取 stdin JSON                  │
    │  2. 合并默认值（脚本内部处理）        │
    │  3. 验证参数                         │
    │  4. 执行业务逻辑                     │
    │  5. 输出 JSON 结果                   │
    └─────────────────────────────────────┘
         │
         ▼
    SkillExecutor 解析输出 → 返回结果
```

### 4.2 设计优势

| 优势 | 说明 |
|------|------|
| **简化 LLM 负担** | LLM 只需返回识别到的参数，无需处理默认值 |
| **统一参数格式** | 所有 Execution Skill 使用相同的 JSON 输入格式 |
| **脚本自治** | 默认值和参数验证逻辑集中在脚本中，便于维护 |
| **简化 SKILL.md** | execution.command 不再需要模板语法 |
| **易于调试** | 可以直接使用 `echo '...' | python script.py` 测试脚本 |

### 4.3 SKILL.md 定义规范

```yaml
# params 定义：只描述参数 schema，供 LLM 参考
params:
  ad_type:
    type: string
    description: 广告类型
    enum: [campaign, adgroup, creative]
  # 不需要 default，默认值在脚本中定义

# execution 定义：简洁的命令调用
execution:
  command:
    - "python"
    - "scripts/query_ad.py"  # 不包含任何参数模板
  timeout: 30
  output: json
```

### 4.4 Python 脚本模板

```python
# -*- coding: utf-8 -*-
"""Skill 执行脚本模板

使用方式（JSON+stdin 模式）：
    echo '{"param1": "value1"}' | python scripts/xxx.py
"""

import json
import sys

# 默认参数值（在脚本中定义）
DEFAULT_PARAMS = {
    "param1": "default_value",
    "param2": 10,
    # ...
}


def main():
    # 1. 从 stdin 读取 JSON
    params = {}
    try:
        params_text = sys.stdin.read()
        if params_text.strip():
            params = json.loads(params_text)
    except json.JSONDecodeError as e:
        print(json.dumps({"success": False, "error": f"JSON 解析失败: {e}"}))
        sys.exit(1)

    # 2. 合并默认值（用户参数覆盖默认值）
    merged = {**DEFAULT_PARAMS, **params}

    # 3. 验证必填参数
    # if not merged["required_param"]:
    #     print(json.dumps({"success": False, "error": "required_param 是必填参数"}))
    #     sys.exit(1)

    # 4. 执行业务逻辑
    result = do_something(merged)

    # 5. 输出 JSON 结果
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

---

## 5. 数据结构定义

### 5.1 base.py

```python
# agent/skills/base.py
"""Skill 数据结构定义"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Union


@dataclass
class SkillTrigger:
    """技能触发配置"""
    keywords: list[str] = field(default_factory=list)
    patterns: list[str] = field(default_factory=list)
    priority: int = 10


@dataclass
class SkillReference:
    """参考文档"""
    name: str
    path: Path
    content: str | None = None

    def load_content(self) -> str:
        """延迟加载文档内容"""
        if self.content is None:
            self.content = self.path.read_text(encoding="utf-8")
        return self.content


@dataclass
class ExecutionConfig:
    """执行配置（Execution Skill 专用）"""

    # 命令：字符串或数组（必填）
    command: Union[str, list[str]]

    # 可选配置
    cwd: str | None = None                          # 工作目录
    env: dict[str, str] = field(default_factory=dict)  # 环境变量
    timeout: int = 30                               # 超时秒数

    # 输出解析
    output: Literal["json", "text", "raw"] = "text"  # 输出格式
    extract: str | None = None                      # JSON 提取路径


@dataclass
class SkillCatalogItem:
    """技能目录项（给 LLM 选择）"""
    skill_id: str
    display_name: str
    description: str
    skill_type: Literal["prompt", "execution"]
    tags: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)


@dataclass
class Skill:
    """技能定义"""

    # === 基础信息 ===
    skill_id: str
    display_name: str
    description: str = ""
    version: str = "1.0.0"
    tags: list[str] = field(default_factory=list)
    enabled: bool = True

    # === 类型 ===
    skill_type: Literal["prompt", "execution"] = "prompt"

    # === 触发配置 ===
    trigger: SkillTrigger = field(default_factory=SkillTrigger)

    # === 参数定义 ===
    params: dict[str, Any] = field(default_factory=dict)

    # === Prompt Skill ===
    prompt_template: str = ""

    # === Execution Skill ===
    execution: ExecutionConfig | None = None

    # === 参考文档 ===
    references: list[SkillReference] = field(default_factory=list)

    # === 示例 ===
    examples: list[dict[str, Any]] = field(default_factory=list)

    # === 源文件信息 ===
    source_path: Path | None = None
    references_path: Path | None = None

    # === 便捷方法 ===
    def is_prompt(self) -> bool:
        """是否为 Prompt 类型"""
        return self.skill_type == "prompt"

    def is_execution(self) -> bool:
        """是否为 Execution 类型"""
        return self.skill_type == "execution"

    def get_keywords(self) -> list[str]:
        """获取触发关键词"""
        return self.trigger.keywords

    def get_patterns(self) -> list[str]:
        """获取触发正则模式"""
        return self.trigger.patterns

    def get_priority(self) -> int:
        """获取优先级"""
        return self.trigger.priority

    def to_catalog_item(self) -> SkillCatalogItem:
        """转换为目录项"""
        return SkillCatalogItem(
            skill_id=self.skill_id,
            display_name=self.display_name,
            description=self.description,
            skill_type=self.skill_type,
            tags=self.tags,
            keywords=self.trigger.keywords,
        )
```

### 5.2 执行结果

```python
# agent/skills/executor.py (部分)

@dataclass
class ToolCallRecord:
    """工具调用记录"""
    command: list[str]              # 执行的命令
    arguments: dict[str, Any]       # 输入参数
    success: bool                   # 是否成功
    result: Any = None              # 返回结果
    error: str | None = None        # 错误信息
    latency_ms: int = 0             # 耗时（毫秒）


@dataclass
class SkillExecutionResult:
    """技能执行结果"""
    skill_id: str
    skill_type: str
    success: bool

    # Prompt Skill 返回
    prompt: str | None = None

    # Execution Skill 返回
    data: Any = None
    tool_calls: list[ToolCallRecord] = field(default_factory=list)

    # 错误信息
    error: str | None = None
    latency_ms: int = 0
```

---

## 6. 组件设计

### 6.1 组件职责

| 组件 | 职责 | 文件 |
|------|------|------|
| **SkillRegistry** | Skill 注册、索引、候选筛选 | `registry.py` |
| **SkillLoader** | SKILL.md 文件加载解析 | `loader.py` |
| **SkillExecutor** | Skill 执行（Prompt 渲染 / Command 执行） | `executor.py` |
| **SkillManager** | 作为 LangChain Tool 注册到 ToolRegistry | `manager.py` |

### 6.2 组件交互流程

```
┌─────────────────────────────────────────────────────────────────┐
│                        AgentService 初始化                       │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. SkillRegistry.load_from_directory(domain_root)              │
│     ├── SkillLoader.load_all()                                  │
│     │   └── 解析所有 SKILL.md 文件                              │
│     └── 建立关键词索引、正则索引、Catalog 缓存                   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. 创建 SkillExecutor                                          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. 创建 SkillManager 并注册到 ToolRegistry                     │
│     skill_manager = SkillManager(registry, executor)            │
│     tool_registry.register(skill_manager)                       │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        AgentLoop 执行                            │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ LLM 决策 → 调用 skill_manager 工具                        │  │
│  │     ↓                                                     │  │
│  │ SkillManager._run(skill_name, query, params)              │  │
│  │     ↓                                                     │  │
│  │ SkillExecutor.execute(skill, params, query)               │  │
│  │     ├── Prompt: Jinja2 渲染 → 返回 prompt                 │  │
│  │     └── Execution: Command 执行 → 返回 data               │  │
│  │     ↓                                                     │  │
│  │ 返回 JSON 结果给 LLM                                      │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.3 SkillRegistry

```python
class SkillRegistry:
    """Skill 注册中心"""

    def __init__(self):
        self.skills: dict[str, Skill] = {}
        self._keyword_index: dict[str, set[str]] = {}      # 关键词索引
        self._pattern_index: list[tuple[re.Pattern, str]] = []  # 正则索引
        self._catalog_cache: list[SkillCatalogItem] = []

    # === 加载 ===
    def load_from_directory(self, domain_root: Path) -> int
    def register(self, skill: Skill) -> None
    def unregister(self, skill_id: str) -> bool

    # === 查询 ===
    def get_skill(self, skill_id: str) -> Skill | None
    def list_skills() -> list[Skill]
    def get_catalog(skill_ids: list[str] | None = None) -> list[SkillCatalogItem]

    # === 候选筛选 ===
    def get_candidates(query: str, max_count: int = 5) -> list[Skill]
    # 关键词匹配: +10 分
    # 正则匹配: +20 分
    # 按分数和优先级排序

    # === 管理 ===
    def clear() -> None
    def get_stats() -> dict
```

### 6.4 SkillLoader

```python
class SkillLoader:
    """SKILL.md 文件加载器"""

    FRONT_MATTER_PATTERN = re.compile(r'^---\s*\n(.*?)\n---\s*\n(.*)$', re.DOTALL)

    def __init__(self, domain_root: Path):
        self.domain_root = domain_root
        self.skills: dict[str, Skill] = {}

    def load_all() -> dict[str, Skill]
    def _load_skill_dir(skill_dir: Path) -> Skill | None
    def _parse_skill_file(path: Path) -> Skill | None
    def _parse_front_matter(content: str) -> tuple[dict, str]
    def _parse_trigger(config: dict) -> SkillTrigger
    def _parse_execution(config: dict) -> ExecutionConfig | None
    def _load_references(refs_path: Path) -> list[SkillReference]
```

### 6.5 SkillExecutor

```python
class SkillExecutor:
    """Skill 执行器"""

    def __init__(self):
        self._jinja_env = Environment(...)

    def execute(skill: Skill, params: dict, query: str) -> SkillExecutionResult

    # === Prompt Skill ===
    def _execute_prompt(skill: Skill, params: dict, query: str) -> SkillExecutionResult
    # - 构建上下文 (query, params, references)
    # - Jinja2 渲染
    # - 返回 prompt

    # === Execution Skill ===
    def _execute_command(skill: Skill, params: dict) -> SkillExecutionResult
    # - 构建命令（不渲染模板）
    # - 序列化参数为 JSON
    # - 解析环境变量
    # - 执行命令，通过 stdin 传递 JSON 参数
    # - 解析输出 (json/text/raw)
    # - 返回 data

    # === 辅助方法 ===
    def _build_command(config: ExecutionConfig) -> list[str]
    def _resolve_env_value(value: str) -> str
    def _parse_output(output: str, format: str, extract: str | None) -> Any
```

### 6.6 SkillManager

```python
class SkillManagerInput(BaseModel):
    """SkillManager 输入参数"""
    skill_name: str = Field(description="要使用的技能名称")
    query: str = Field(description="用户原始问题")
    params: dict[str, Any] | None = Field(description="技能参数")


class SkillManager(BaseTool):
    """Skill 管理工具 - 作为 LangChain Tool"""

    name: str = "skill_manager"
    args_schema: type[BaseModel] = SkillManagerInput

    registry: SkillRegistry
    executor: SkillExecutor
    candidate_skill_ids: list[str] = []

    def __init__(registry, executor, candidate_skill_ids=None)
    def _build_description() -> str        # 动态生成工具描述
    def _run(skill_name, query, params) -> str  # 执行技能
    def _format_result(result) -> str      # 格式化返回结果
```

---

## 7. 文件清单

### 7.1 Phase 3 需要创建的文件

```
src/agent/skills/
├── __init__.py                 # 模块导出
├── base.py                     # 数据结构定义
├── registry.py                 # SkillRegistry
├── loader.py                   # SkillLoader
├── executor.py                 # SkillExecutor
└── manager.py                  # SkillManager (LangChain Tool)

tests/agent/skills/
├── __init__.py
├── conftest.py                 # 测试 fixtures
├── test_base.py                # 数据结构测试
├── test_registry.py            # 注册中心测试
├── test_loader.py              # 加载器测试
├── test_executor.py            # 执行器测试
└── test_manager.py             # 管理工具测试
```

### 7.2 示例 Skill 文件

```
domain/ad_engine/skills/
├── ad_copy_generator/
│   ├── SKILL.md
│   └── references/
│       └── copywriting_guide.md
├── query_metrics/
│   └── SKILL.md
├── code_search/
│   └── SKILL.md
└── git_info/
    └── SKILL.md
```

---

## 8. 实现计划

### 8.1 任务清单

| 序号 | 任务 | 文件 | 估算时间 | 依赖 |
|------|------|------|----------|------|
| 3.1 | 实现 Skill 数据结构 | `base.py` | 0.5 天 | - |
| 3.2 | 实现 SkillRegistry | `registry.py` | 0.5 天 | 3.1 |
| 3.3 | 实现 SkillLoader | `loader.py` | 1 天 | 3.1 |
| 3.4 | 实现 SkillExecutor | `executor.py` | 1 天 | 3.1 |
| 3.5 | 实现 SkillManager | `manager.py` | 0.5 天 | 3.2, 3.4 |
| 3.6 | 编写单元测试 | `tests/` | 1 天 | 3.1-3.5 |
| 3.7 | 创建示例 Skill | `domain/` | 0.5 天 | 3.3 |
| **总计** | | | **5 天** | |

### 8.2 详细任务

#### 3.1 实现 Skill 数据结构 (0.5 天)

**文件**: `agent/skills/base.py`

**内容**:
- `SkillTrigger` 触发配置
- `SkillReference` 参考文档
- `ExecutionConfig` 执行配置
- `SkillCatalogItem` 目录项
- `Skill` 技能定义

**验收**:
- [ ] 所有 dataclass 定义完整
- [ ] 类型标注正确
- [ ] 便捷方法实现

---

#### 3.2 实现 SkillRegistry (0.5 天)

**文件**: `agent/skills/registry.py`

**内容**:
- 关键词索引建立
- 正则索引建立
- 候选筛选算法
- Catalog 缓存

**验收**:
- [ ] `load_from_directory` 正常加载
- [ ] 关键词匹配正确
- [ ] 正则匹配正确
- [ ] 优先级排序正确

---

#### 3.3 实现 SkillLoader (1 天)

**文件**: `agent/skills/loader.py`

**内容**:
- YAML front matter 解析
- Markdown 内容提取
- 触发配置解析
- 执行配置解析
- 参考文档加载

**验收**:
- [ ] SKILL.md 格式正确解析
- [ ] Prompt Skill 模板提取
- [ ] Execution Skill 配置解析
- [ ] 参考文档索引建立

---

#### 3.4 实现 SkillExecutor (1 天)

**文件**: `agent/skills/executor.py`

**内容**:
- Prompt Skill: Jinja2 模板渲染
- Execution Skill: Command 执行
- 命令模板渲染
- 环境变量解析
- 输出解析 (json/text/raw)
- 错误处理

**验收**:
- [ ] Prompt 模板正确渲染
- [ ] Command 字符串形式执行
- [ ] Command 数组形式执行
- [ ] 环境变量 ${VAR} 解析
- [ ] JSON 输出解析和提取
- [ ] 超时处理
- [ ] 错误处理

---

#### 3.5 实现 SkillManager (0.5 天)

**文件**: `agent/skills/manager.py`

**内容**:
- LangChain Tool 定义
- 动态描述生成
- 执行入口
- 结果格式化

**验收**:
- [ ] 注册为 LangChain Tool
- [ ] 描述包含可用技能列表
- [ ] 执行返回 JSON 格式

---

#### 3.6 编写单元测试 (1 天)

**文件**: `tests/agent/skills/`

**内容**:
- `test_base.py`: 数据结构测试
- `test_registry.py`: 注册、索引、筛选测试
- `test_loader.py`: 文件加载测试
- `test_executor.py`: 执行逻辑测试
- `test_manager.py`: Tool 集成测试

**验收**:
- [ ] 测试覆盖率 > 80%
- [ ] 所有测试通过

---

#### 3.7 创建示例 Skill (0.5 天)

**文件**: `domain/ad_engine/skills/`

**内容**:
- `ad_copy_generator`: Prompt Skill 示例
- `query_metrics`: Execution Skill 示例 (curl)
- `code_search`: Execution Skill 示例 (Python)
- `git_info`: Execution Skill 示例 (系统命令)

**验收**:
- [ ] 示例 Skill 可正常加载
- [ ] 示例 Skill 可正常执行

---

## 9. 验收标准

| 序号 | 标准 | 说明 |
|------|------|------|
| 1 | SKILL.md 加载 | 支持 YAML front matter + Markdown |
| 2 | Prompt Skill | Jinja2 模板渲染，支持参考文档 |
| 3 | Execution Skill | Command 执行，参数通过 stdin 以 JSON 格式传递 |
| 4 | 环境变量 | 支持 `${ENV_VAR}` 引用系统环境变量 |
| 5 | 输出解析 | 支持 json/text/raw 三种格式 |
| 6 | 触发匹配 | 关键词 + 正则模式匹配，优先级排序 |
| 7 | Tool 集成 | SkillManager 注册为 LangChain Tool |
| 8 | 单元测试 | 覆盖率 > 80% |

---

## 10. 风险与缓解

| 风险 | 影响 | 缓解策略 |
|------|------|----------|
| 命令注入安全 | 高 | 使用数组形式参数，避免 shell 解析 |
| 超时控制 | 中 | 设置默认超时，支持配置 |
| 输出解析失败 | 低 | 提供 raw 格式兜底 |
| Jinja2 模板错误 | 低 | 捕获异常，返回错误信息 |

---

## 11. 与其他 Phase 的关系

```
Phase 1 (基础架构)
    ↓ 依赖
Phase 3 (Skill 系统)
    ↓ 集成到
AgentLoop (Phase 1)
    ↓ 可调用
SkillManager (Phase 3)
    ↓ 可调用
MCP Tools (Phase 2) - 作为 Command 的一种
```

**说明**:
- Phase 3 依赖 Phase 1 的 `ToolRegistry`
- Skill 执行的 Command 可以调用 MCP Server（如 `python -m mcp_client ...`）
- Skill 系统与 MCP 系统独立，通过 Command 方式松耦合
