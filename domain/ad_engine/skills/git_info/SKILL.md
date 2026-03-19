---
skill_id: git_info
display_name: Git 信息查询
description: 查询 Git 仓库的提交记录、分支状态等信息
version: 1.0.0
tags:
  - git
  - 版本控制
  - 代码仓库
  - commit

skill_type: execution

trigger:
  keywords:
    - git
    - commit
    - 分支
    - 提交
    - 版本
    - 变更
    - 代码记录
  patterns:
    - "查看.*提交"
    - "git.*状态"
    - "最近.*commit"
    - "分支.*信息"
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
  - user: "查看 main 分支的最近 3 次提交"
    params:
      command: log
      limit: 3
      branch: main
---

查询 Git 仓库信息。

## 输入参数

参数通过 stdin 以 JSON 格式传递，脚本会自动处理默认值。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| command | string | 否 | status | Git 子命令 |
| limit | integer | 否 | 10 | 返回数量限制（用于 log 命令） |
| branch | string | 否 | - | 指定分支名称 |

## 支持的命令

| 命令 | 说明 | 用途 |
|------|------|------|
| log | 查看提交历史 | 了解代码变更记录 |
| status | 查看工作区状态 | 检查未提交的变更 |
| branch | 列出分支 | 了解分支结构 |
| diff | 查看差异 | 比较代码变更 |
| show | 查看提交详情 | 了解具体提交内容 |

## 返回格式

返回 JSON 格式的结果：

```json
{
  "success": true,
  "command": "log",
  "output": "...",
  "items": [...],      // 结构化数据（视命令而定）
  "params": {...},     // 实际使用的参数
  "timestamp": "..."
}
```
