# 目录结构设计

## 1. 概述

本文档详细说明 v3 架构的目录结构设计。

**核心原则**：
1. `workflow/` 只专注于流程编排
2. 基础设施组件放在 `src/` 下共享
3. `agent_loop_node` 作为公共节点放在 `common_nodes/`
4. 移除冗余的 `skills/`、`llm/` 目录

---

## 2. 完整目录结构

```
src/
├── workflow/                           # Workflow 层（系统入口）
│   ├── __init__.py
│   ├── service.py                      # WorkflowService（入口服务）
│   ├── engine.py                       # WorkflowEngine（LangGraph 主图编排）
│   ├── state.py                        # WorkflowState（状态定义）
│   ├── router.py                       # DomainRouter（规则路由）
│   │
│   ├── subgraph/                       # 【新增】Subgraph 子图目录
│   │   ├── __init__.py
│   │   ├── base.py                     # BaseSubgraph（子图基类）
│   │   ├── registry.py                 # SubgraphRegistry（子图注册中心）
│   │   │
│   │   ├── knowledge_qa/               # 知识问答场景
│   │   │   ├── __init__.py
│   │   │   ├── config.py               # KnowledgeQAConfig
│   │   │   ├── engine.py               # KnowledgeQAEngine
│   │   │   └── nodes/                  # 独有节点
│   │   │       ├── __init__.py
│   │   │       ├── query_rewriter.py
│   │   │       └── retrieval_flow.py
│   │   │
│   │   └── issue_analysis/             # 问题分析场景
│   │       ├── __init__.py
│   │       ├── config.py
│   │       ├── engine.py
│   │       └── nodes/
│   │           ├── __init__.py
│   │           ├── context_loader.py
│   │           └── issue_classifier.py
│   │
│   ├── common_nodes/                   # 【新增】公共节点
│   │   ├── __init__.py
│   │   ├── finalize_response.py
│   │   ├── out_of_scope_response.py
│   │   ├── control_response.py
│   │   ├── evidence_merger.py
│   │   │
│   │   └── agent_loop_node/            # 【关键】Agent Loop 公共节点
│   │       ├── __init__.py
│   │       ├── node.py
│   │       └── context.py
│   │
│   └── common/                         # 通用工具（保持）
│       ├── __init__.py
│       ├── domain_profile.py
│       ├── evidence.py
│       ├── runtime_logging.py
│       └── func_utils.py
│
├── agent/                         # Agent Loop 核心引擎层
│   ├── __init__.py
│   ├── engine.py                       # AgentLoopEngine（核心引擎）
│   ├── state.py                        # AgentLoopState
│   ├── config.py                       # AgentLoopConfig
│   │
│   ├── core/                           # 核心模块
│   │   ├── __init__.py
│   │   ├── executor.py                 # LoopExecutor
│   │   ├── tool_caller.py              # ToolCaller
│   │   ├── prompt_builder.py           # PromptBuilder
│   │   └── router.py                   # Router
│   │
│   ├── llm/                            # LLM 客户端（已存在）
│   │   ├── __init__.py
│   │   └── client.py                   # LLMClient
│   │
│   ├── tools/                          # 工具管理
│   │   ├── __init__.py
│   │   ├── registry.py                 # ToolRegistry
│   │   ├── base.py                     # BaseTool
│   │   ├── whitelist.py                # ToolWhitelist
│   │   └── local_tools.py
│   │
│   ├── skills/                         # 技能管理（已存在）
│   │   ├── __init__.py
│   │   ├── base.py                     # Skill
│   │   ├── registry.py                 # SkillRegistry
│   │   ├── loader.py                   # SkillLoader
│   │   ├── executor.py                 # SkillExecutor
│   │   └── manager.py                  # SkillManager
│   │
│   ├── mcp/                            # MCP 集成
│   │   ├── __init__.py
│   │   ├── client.py
│   │   ├── tool_adapter.py
│   │   ├── session.py
│   │   └── config_loader.py
│   │
│   └── common/                         # 通用工具
│       ├── __init__.py
│       ├── logging.py
│       └── utils.py
│
├── retrievers/                         # 【移动】检索器（原 workflow/retrievers/）
│   ├── __init__.py
│   ├── wiki_retriever.py
│   └── code_retriever.py
│
├── session/                            # 【移动】会话存储（原 workflow/session/）
│   ├── __init__.py
│   └── session_store.py
│
├── observability/                      # 【移动】可观测性（原 workflow/observability/）
│   ├── __init__.py
│   ├── metrics.py
│   ├── trace.py
│   └── audit.py
│
├── eval/                               # 【移动】评测（原 workflow/eval/）
│   ├── __init__.py
│   ├── runner.py
│   └── results/
│
├── api/                                # API 层
│   ├── __init__.py
│   └── main.py
│
├── bootstrap/                          # 启动引导
│   └── ...
│
└── web/                                # 前端静态资源
    └── assets/
```

---

## 3. 目录职责详解

### 3.1 Workflow 层 (`src/workflow/`)

**只专注于流程编排**，不包含基础设施组件。

| 目录/文件 | 职责 | 变更说明 |
|-----------|------|----------|
| `service.py` | WorkflowService 入口 | 保持 |
| `engine.py` | LangGraph 主图编排 | 保持 |
| `state.py` | WorkflowState 状态定义 | 保持 |
| `router.py` | DomainRouter 规则路由 | 保持 |
| `subgraph/` | 场景子图 | **新增** |
| `common_nodes/` | 公共节点（含 agent_loop_node） | **新增** |
| `common/` | 通用工具 | 保持 |

**移除的目录**：

| 目录 | 原位置 | 处理方式 |
|------|--------|----------|
| `skills/` | `workflow/skills/` | 迁移到 `agent/skills/`（或保留 v1 兼容） |
| `llm/` | `workflow/llm/` | **废弃**（`agent/llm/client.py` 已存在） |
| `retrievers/` | `workflow/retrievers/` | 移动到 `src/retrievers/` |
| `session/` | `workflow/session/` | 移动到 `src/session/` |
| `observability/` | `workflow/observability/` | 移动到 `src/observability/` |
| `eval/` | `workflow/eval/` | 移动到 `src/eval/` |

### 3.2 Agent Loop 层 (`src/agent/`)

Agent Loop 核心引擎，包含 LLM、工具、技能管理。

| 目录/文件 | 职责 | 变更说明 |
|-----------|------|----------|
| `engine.py` | AgentLoopEngine 核心引擎 | **新增** |
| `core/` | 核心模块（执行器、路由等） | 已存在 |
| `llm/client.py` | LLM 客户端 | 已存在 |
| `tools/` | 工具管理 | 保持 |
| `skills/` | 技能管理 | 已存在 |
| `mcp/` | MCP 集成 | 保持 |

### 3.3 基础设施层 (`src/` 顶层)

共享的基础设施组件，可被多个模块使用。

| 目录 | 职责 | 来源 |
|------|------|------|
| `retrievers/` | 检索器（Wiki、代码） | 从 `workflow/retrievers/` 移动 |
| `session/` | 会话存储 | 从 `workflow/session/` 移动 |
| `observability/` | 可观测性（Trace、Metrics） | 从 `workflow/observability/` 移动 |
| `eval/` | 评测系统 | 从 `workflow/eval/` 移动 |

---

## 4. 调整对比

### 4.1 调整前

```
src/workflow/
├── skills/                 # v1 Skill 系统（冗余）
├── llm/                    # LLM 客户端（应在 agent_loop）
├── retrievers/             # 检索器（基础设施）
├── session/                # 会话存储（基础设施）
├── observability/          # 可观测性（基础设施）
├── eval/                   # 评测（基础设施）
├── common/
├── nodes/
└── ...
```

### 4.2 调整后

```
src/
├── workflow/               # 只做流程编排
│   ├── subgraph/          # 场景子图
│   ├── common_nodes/      # 公共节点
│   └── common/            # 通用工具
│
├── agent/                  # Agent Loop 核心
│   ├── core/              # 核心模块
│   ├── llm/               # LLM 客户端
│   ├── tools/
│   └── skills/            # 技能系统
│
├── retrievers/             # 共享基础设施
├── session/
├── observability/
└── eval/
```

---

## 5. 移动清单

### 5.1 废弃 workflow/llm/

```bash
# workflow/llm/ 目录已废弃
# agent/llm/client.py 已有完整实现，无需迁移

# 如果有提示词工具需要复用，可以迁移
# mv src/workflow/llm/llm_prompt_utils.py src/agent/core/prompt_utils.py

# 最终删除 workflow/llm/ 目录
rm -rf src/workflow/llm/
```

> **说明**：`src/agent/llm/client.py` 已实现完整的 LLM 客户端功能，包括：
> - async 原生调用
> - 工具调用 (Function Calling)
> - 流式响应
> - 详细的请求/响应日志
>
> 因此 `workflow/llm/llm_client.py` 不再需要，直接废弃即可。

### 5.2 Skill 系统处理

```bash
# Skill 系统保留在 workflow/skills/（v1 兼容）
# agent/skills/ 有独立实现

# 如果确定不再需要 v1 skills，可以删除
# rm -rf src/workflow/skills/
```

### 5.2 移动到 src/ 顶层

```bash
# 检索器
mv src/workflow/retrievers src/retrievers

# 会话存储
mv src/workflow/session src/session

# 可观测性
mv src/workflow/observability src/observability

# 评测
mv src/workflow/eval src/eval
```

### 5.3 删除冗余目录

```bash
# 删除空的 llm 目录
rmdir src/workflow/llm

# 如果 v1 skills 不再需要
rm -rf src/workflow/skills
```

---

## 6. 导入路径变更

### 6.1 导入变更对照表

| 原路径 | 新路径 |
|--------|--------|
| `workflow.llm.llm_client` | `agent.core.llm_client` |
| `workflow.llm.llm_prompt_utils` | `agent.core.prompt_utils` |
| `workflow.skills.*` | `agent.skills.*` |
| `workflow.retrievers.*` | `retrievers.*` |
| `workflow.session.*` | `session.*` |
| `workflow.observability.*` | `observability.*` |

### 6.2 兼容性导入（可选）

如果需要保持向后兼容，可以在原位置添加重导出：

```python
# src/workflow/llm/__init__.py（兼容层）

import warnings

warnings.warn(
    "从 workflow.llm 导入已废弃，请使用 agent.core.llm_client",
    DeprecationWarning,
    stacklevel=2
)

from agent.core.llm_client import WorkflowLLMClient
from agent.core.prompt_utils import resolve_system_prompt

__all__ = ["WorkflowLLMClient", "resolve_system_prompt"]
```

---

## 7. 模块依赖关系

### 7.1 依赖图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              模块依赖关系                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   api/main.py                                                                │
│        │                                                                     │
│        ▼                                                                     │
│   workflow/service.py                                                        │
│        │                                                                     │
│        ├── workflow/subgraph/                                                │
│        │        │                                                            │
│        │        ├── subgraph/nodes/ (独有节点)                               │
│        │        │                                                            │
│        │        └── workflow/common_nodes/                                   │
│        │                    │                                                │
│        │                    └── agent_loop_node ───────┐                    │
│        │                                               │                     │
│        └── agent/engine.py ◀──────────────────────┘                     │
│                │                                                             │
│                ├── agent/core/ (含 llm_client)                          │
│                ├── agent/tools/                                         │
│                └── agent/skills/                                        │
│                                                                              │
│   ──────────────────────────────────────────────────────────────────────    │
│   共享基础设施（src/ 顶层）                                                   │
│                                                                              │
│   retrievers/ ──────────┐                                                    │
│   session/ ─────────────┼──▶ 被 workflow/ 和 agent/ 共享               │
│   observability/ ───────┤                                                    │
│   eval/ ────────────────┘                                                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 导入规范

```python
# ✅ 正确的导入方式

# Workflow 层
from workflow.service import WorkflowService
from workflow.subgraph.knowledge_qa import KnowledgeQAEngine
from workflow.common_nodes.agent_loop_node import create_agent_loop_node

# Agent Loop 层
from agent.engine import AgentLoopEngine
from agent.core.llm_client import LLMClient
from agent.skills.registry import SkillRegistry

# 基础设施层（src/ 顶层）
from retrievers.wiki_retriever import WikiRetriever
from session.session_store import SessionStore
from observability.metrics import record_metric
```

---

## 8. 检查清单

### 8.1 目录调整检查

- [ ] `workflow/llm/` 迁移到 `agent/core/`
- [ ] `workflow/skills/` 迁移到 `agent/skills/`（或保留兼容）
- [ ] `workflow/retrievers/` 移动到 `src/retrievers/`
- [ ] `workflow/session/` 移动到 `src/session/`
- [ ] `workflow/observability/` 移动到 `src/observability/`
- [ ] `workflow/eval/` 移动到 `src/eval/`

### 8.2 导入更新检查

- [ ] 所有 `workflow.llm.*` 导入更新
- [ ] 所有 `workflow.skills.*` 导入更新
- [ ] 所有 `workflow.retrievers.*` 导入更新
- [ ] 所有 `workflow.session.*` 导入更新
- [ ] 所有 `workflow.observability.*` 导入更新

### 8.3 功能检查

- [ ] 现有测试通过
- [ ] 导入无循环依赖
- [ ] API 接口正常响应

---

## 9. 总结

### 9.1 调整优势

| 优势 | 说明 |
|------|------|
| **职责清晰** | `workflow/` 只做流程编排 |
| **减少冗余** | `skills/`、`llm/` 集中到 `agent/` |
| **共享基础设施** | `retrievers/` 等可被多个模块使用 |
| **易于维护** | 模块边界更清晰 |

### 9.2 最终结构

```
src/
├── workflow/           # 流程编排（subgraph、common_nodes、common）
├── agent/         # Agent Loop 核心（engine、core、tools、skills、mcp）
├── retrievers/         # 检索器（共享）
├── session/            # 会话存储（共享）
├── observability/      # 可观测性（共享）
├── eval/               # 评测（共享）
├── api/                # API 入口
├── bootstrap/          # 启动引导
└── web/                # 前端资源
```
