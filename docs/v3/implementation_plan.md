# V3 架构实施执行计划

## 1. 概述

### 1.1 目标

将 v3 架构从设计文档落地为可运行的代码，实现：
1. **目录结构重组**：清晰的模块边界 ✅ 已完成
2. **Agent Loop 节点基类**：`BaseAgentLoopNode` 作为公共基础
3. **节点重构**：`knowledge_answer` 和 `issue_analysis` 继承基类
4. **启动时初始化**：组件在 WorkflowEngine 启动时完成初始化

### 1.2 实施原则

| 原则 | 说明 |
|------|------|
| **渐进式** | 分阶段实施，每个阶段可独立验证 |
| **向后兼容** | 保留 v1 代码，通过配置开关切换 |
| **测试驱动** | 每个阶段完成后进行测试验证 |
| **文档同步** | 代码变更同步更新文档 |
| **启动时初始化** | 避免延迟初始化，在引擎构造时完成组件加载 |

### 1.3 总体时间线

```
Week 1                    Week 2                    Week 3
├─────────────────────────┼─────────────────────────┼─────────────────────────┤
│ Phase 1: 目录重组 ✅    │ Phase 3: 重构节点       │ Phase 5: 测试与文档      │
│ Phase 2: 基础节点       │ Phase 4: 集成测试       │                          │
└─────────────────────────┴─────────────────────────┴─────────────────────────┘
```

---

## 2. Phase 1: 目录结构重组 ✅ 已完成

### 2.1 已完成任务

| 序号 | 任务 | 状态 |
|------|------|------|
| 1.1 | 创建 common_nodes 目录 | ✅ 完成 |
| 1.2 | 创建 subgraph 目录结构 | ✅ 完成 |
| 1.3 | 移动 retrievers 到 src 顶层 | ✅ 完成 |
| 1.4 | 移动 session 到 src 顶层 | ✅ 完成 |
| 1.5 | 移动 observability 到 src 顶层 | ✅ 完成 |
| 1.6 | 移动 eval 到 src 顶层 | ✅ 完成 |
| 1.7 | 更新所有导入路径 | ✅ 完成 |
| 1.8 | 修复循环导入问题 | ✅ 完成 |

---

## 3. Phase 2: Agent Loop 基础节点（2 天）

### 3.1 目标

- 实现 `BaseAgentLoopNode` 基类
- 实现 `AgentLoopNodeConfig` 配置类
- 提供启动时初始化支持

详细设计见：[agent_loop_node_refactor.md](./agent_loop_node_refactor.md)

### 3.2 任务清单

#### Day 1: 基础节点框架

| 序号 | 任务 | 文件路径 | 验证标准 |
|------|------|----------|----------|
| 2.1 | 实现 AgentLoopNodeConfig | `common_nodes/agent_loop_node/config.py` | 配置类可实例化 |
| 2.2 | 实现 BaseAgentLoopNode | `common_nodes/agent_loop_node/base.py` | 基类方法完整 |
| 2.3 | 实现 `__init__.py` | `common_nodes/agent_loop_node/__init__.py` | 导出正确 |
| 2.4 | 单元测试 | `tests/workflow/common_nodes/test_agent_loop_node.py` | 测试通过 |

#### Day 2: 集成到 WorkflowEngine

| 序号 | 任务 | 文件路径 | 验证标准 |
|------|------|----------|----------|
| 2.5 | 添加 Skill 组件初始化 | `workflow/engine.py` | 启动时加载 |
| 2.6 | 添加节点初始化方法 | `workflow/engine.py` | 依赖注入正确 |
| 2.7 | 更新节点调用方式 | `workflow/engine.py` | 兼容现有逻辑 |
| 2.8 | 集成测试 | `tests/workflow/test_engine.py` | 测试通过 |

### 3.3 产出物

```
src/workflow/common_nodes/agent_loop_node/
├── __init__.py              # 导出 BaseAgentLoopNode, AgentLoopNodeConfig
├── config.py                # AgentLoopNodeConfig 配置类
└── base.py                  # BaseAgentLoopNode 基类
```

---

## 4. Phase 3: 节点重构（3 天）

### 4.1 目标

- 将 `knowledge_answer` 重构为继承 `BaseAgentLoopNode`
- 将 `issue_analysis` 重构为继承 `BaseAgentLoopNode`
- 复用现有的 Fallback 和提示词逻辑

### 4.2 任务清单

#### Day 1: knowledge_answer 重构

| 序号 | 任务 | 文件路径 | 验证标准 |
|------|------|----------|----------|
| 3.1 | 创建 KnowledgeAnswerConfig | `nodes/analysis/knowledge_answer/config.py` | 配置类完整 |
| 3.2 | 创建 KnowledgeAnswerNode | `nodes/analysis/knowledge_answer/node.py` | 继承 BaseAgentLoopNode |
| 3.3 | 迁移提示词和 Fallback | 从现有代码迁移 | 功能不变 |
| 3.4 | 更新 `__init__.py` | `nodes/analysis/knowledge_answer/__init__.py` | 导出正确 |

#### Day 2: issue_analysis 重构

| 序号 | 任务 | 文件路径 | 验证标准 |
|------|------|----------|----------|
| 3.5 | 创建 IssueAnalysisConfig | `nodes/analysis/issue_analysis/config.py` | 配置类完整 |
| 3.6 | 创建 IssueAnalysisNode | `nodes/analysis/issue_analysis/node.py` | 继承 BaseAgentLoopNode |
| 3.7 | 迁移提示词和 Fallback | 从现有代码迁移 | 功能不变 |
| 3.8 | 更新 `__init__.py` | `nodes/analysis/issue_analysis/__init__.py` | 导出正确 |

#### Day 3: 集成和测试

| 序号 | 任务 | 验证标准 |
|------|------|----------|
| 3.9 | 更新 WorkflowEngine 节点初始化 | 节点正确注入依赖 |
| 3.10 | 单元测试 | 测试通过 |
| 3.11 | 集成测试 | API 调用正常 |

### 4.3 产出物

```
src/workflow/nodes/analysis/
├── knowledge_answer/
│   ├── __init__.py          # 导出 KnowledgeAnswerNode, run()
│   ├── config.py            # KnowledgeAnswerConfig
│   └── node.py              # KnowledgeAnswerNode 类
│
└── issue_analysis/
    ├── __init__.py          # 导出 IssueAnalysisNode, run()
    ├── config.py            # IssueAnalysisConfig
    └── node.py              # IssueAnalysisNode 类
```

---

## 5. Phase 4: 集成测试（2 天）

### 5.1 目标

- 验证启动时初始化正确执行
- 验证 API 接口正常工作
- 验证节点继承关系正确

### 5.2 任务清单

| 序号 | 任务 | 验证标准 |
|------|------|----------|
| 4.1 | 启动时初始化测试 | 组件在引擎构造时加载 |
| 4.2 | 节点依赖注入测试 | LLM 和 Skill 组件正确注入 |
| 4.3 | knowledge_answer 端到端测试 | API 返回正确 |
| 4.4 | issue_analysis 端到端测试 | API 返回正确 |
| 4.5 | Fallback 机制测试 | LLM 失败时正确降级 |

---

## 6. Phase 5: 文档更新（1 天）

### 6.1 任务清单

| 序号 | 任务 | 文件 |
|------|------|------|
| 5.1 | 更新 README | `README.md` |
| 5.2 | 更新 API 文档 | `docs/api.md` |
| 5.3 | 清理过时文档 | `docs/v3/` |

---

## 7. 检查清单

### 7.1 Phase 完成检查

| Phase | 检查项 | 状态 |
|-------|--------|------|
| Phase 1 | 目录结构正确 | ✅ |
| Phase 1 | 导入路径更新 | ✅ |
| Phase 1 | 现有测试通过 | ✅ |
| Phase 2 | BaseAgentLoopNode 可运行 | [ ] |
| Phase 2 | AgentLoopNodeConfig 完整 | [ ] |
| Phase 2 | 启动时初始化正常 | [ ] |
| Phase 3 | KnowledgeAnswerNode 可运行 | [ ] |
| Phase 3 | IssueAnalysisNode 可运行 | [ ] |
| Phase 3 | 节点继承关系正确 | [ ] |
| Phase 4 | 集成测试通过 | [ ] |
| Phase 4 | API 接口正常响应 | [ ] |
| Phase 5 | 文档更新完成 | [ ] |

### 7.2 最终验收

- [ ] 所有单元测试通过
- [ ] 所有集成测试通过
- [ ] API 接口正常响应
- [ ] 启动时初始化正常
- [ ] 文档更新完整

---

## 8. 关键设计决策

### 8.1 启动时初始化 vs 延迟初始化

**决策**：采用启动时初始化

**理由**：
1. 避免首次请求时的延迟
2. 启动时即可发现配置问题
3. 便于监控和管理资源

**实现**：
```python
class WorkflowEngine:
    def __init__(self):
        # 启动时完成所有初始化
        self._llm_client = self._init_llm_client()
        self._skill_registry, self._skill_executor = self._init_skill_components()
        self._nodes = self._init_nodes()
```

### 8.2 入口点选择

**决策**：入口点保持为 `workflow/engine.py`

**理由**：
1. 与现有架构兼容
2. WorkflowEngine 负责编排和初始化
3. API 层只做协议转换

### 8.3 节点继承模式

**决策**：`knowledge_answer` 和 `issue_analysis` 继承 `BaseAgentLoopNode`

**理由**：
1. 共享 LLM + Skill 调用逻辑
2. 共享 Fallback 机制
3. 便于新增场景节点
| 1.9 | 更新 observability 导入 | 全局搜索 `from workflow.observability` → `from observability` | 无旧导入 |
| 1.10 | 运行测试 | `pytest tests/` | 测试通过 |
| 1.11 | 验证 API 启动 | `python -m uvicorn api.main:app` | 服务正常启动 |

### 2.3 产出物

```
src/
├── workflow/
│   ├── common_nodes/        # 新增（公共节点目录）
│   ├── subgraph/            # 新增（子图目录）
│   │   ├── knowledge_qa/    # 知识问答场景
│   │   └── issue_analysis/  # 问题分析场景
│   └── ...
├── agent/                   # 保持原有名称
│   └── ...
├── retrievers/              # 移动完成
├── session/                 # 移动完成
├── observability/           # 移动完成
└── eval/                    # 移动完成
```

### 2.4 回滚方案

```bash
# 如果需要回滚
mv src/retrievers src/workflow/retrievers
mv src/session src/workflow/session
mv src/observability src/workflow/observability
mv src/eval src/workflow/eval
rm -rf src/workflow/common_nodes
rm -rf src/workflow/subgraph
```

---

## 3. Phase 2: Agent Loop Engine（3 天）

### 3.1 目标

- 实现 `AgentLoopEngine` 核心引擎
- 实现 `agent_loop_node` 公共节点
- 实现 `NodeExecutionContext` 配置类

### 3.2 任务清单

#### Day 1: 核心引擎框架

| 序号 | 任务 | 文件路径 | 验证标准 |
|------|------|----------|----------|
| 2.1 | 实现 AgentLoopConfig | `agent/config.py` | 配置类可实例化 |
| 2.2 | 实现 AgentLoopState | `agent/state.py` | 状态类定义完整 |
| 2.3 | 实现 AgentLoopEngine 骨架 | `agent/engine.py` | 引擎可实例化 |
| 2.4 | 复用现有 LLM Client | `agent/llm/client.py`（已存在） | 无需迁移 |
| 2.5 | 单元测试 | `tests/agent/test_engine.py` | 测试通过 |

#### Day 2: 核心执行器

| 序号 | 任务 | 文件路径 | 验证标准 |
|------|------|----------|----------|
| 2.5 | 实现 LoopExecutor | `agent/core/executor.py` | 循环逻辑正确 |
| 2.6 | 实现 ToolCaller | `agent/core/tool_caller.py` | 工具调用正确 |
| 2.7 | 实现 PromptBuilder | `agent/core/prompt_builder.py` | 提示词构建正确 |
| 2.8 | 复用现有 LLM Client | `agent/llm/client.py`（已存在） | 无需迁移 |
| 2.9 | 单元测试 | `tests/agent/core/` | 测试通过 |

#### Day 3: 公共节点封装

| 序号 | 任务 | 文件路径 | 验证标准 |
|------|------|----------|----------|
| 2.10 | 实现 NodeExecutionContext | `common_nodes/agent_loop_node/context.py` | 配置类完整 |
| 2.11 | 实现 create_agent_loop_node | `common_nodes/agent_loop_node/node.py` | 节点可创建 |
| 2.12 | 实现 `__init__.py` | `common_nodes/agent_loop_node/__init__.py` | 导出正确 |
| 2.13 | 实现 `__init__.py` | `common_nodes/__init__.py` | 导出所有公共节点 |
| 2.14 | 集成测试 | `tests/workflow/common_nodes/test_agent_loop_node.py` | 测试通过 |

### 3.3 产出物

```
src/
├── agent/                           # 保持现有
│   └── llm/
│       └── client.py                # 已存在，复用
│
├── agent/                      # 重命名后
│   ├── config.py                    # 新增
│   ├── state.py                     # 新增
│   ├── engine.py                    # 新增
│   └── core/
│       ├── executor.py              # 新增
│       ├── tool_caller.py           # 新增
│       └── prompt_builder.py        # 新增
│
└── workflow/common_nodes/
    ├── __init__.py                  # 新增
    └── agent_loop_node/
        ├── __init__.py              # 新增
        ├── context.py               # 新增
        └── node.py                  # 新增
```

> **注意**：LLM Client 直接复用 `agent/llm/client.py`，无需从 `workflow/llm/` 迁移。

### 3.4 验收标准

```python
# 验收测试代码

from workflow.common_nodes.agent_loop_node import (
    create_agent_loop_node,
    NodeExecutionContext
)
from agent.engine import AgentLoopEngine

# 1. 创建引擎
engine = AgentLoopEngine(
    llm_client=mock_llm_client,
    tool_registry=mock_tool_registry,
    skill_registry=mock_skill_registry,
)

# 2. 创建节点
node = create_agent_loop_node(
    engine=engine,
    node_name="test_node",
    context=NodeExecutionContext(
        tool_whitelist=["skill_manager"],
        max_iterations=3,
    )
)

# 3. 执行节点
result = await node({"user_query": "测试问题"})
assert "answer" in result
assert result["agent_loop_completed"] is True
```

---

## 4. Phase 3: Subgraph 框架（2 天）

### 4.1 目标

- 实现 Subgraph 基类和配置
- 实现 SubgraphRegistry 注册中心
- 创建 knowledge_qa 和 issue_analysis 目录结构

### 4.2 任务清单

#### Day 1: 基础框架

| 序号 | 任务 | 文件路径 | 验证标准 |
|------|------|----------|----------|
| 3.1 | 实现 SubgraphConfig 基类 | `subgraph/base.py` | 配置类可实例化 |
| 3.2 | 实现 BaseSubgraph 基类 | `subgraph/base.py` | 基类方法完整 |
| 3.3 | 实现 SubgraphRegistry | `subgraph/registry.py` | 注册中心可工作 |
| 3.4 | 实现 `__init__.py` | `subgraph/__init__.py` | 导出正确 |
| 3.5 | 单元测试 | `tests/workflow/subgraph/test_base.py` | 测试通过 |

#### Day 2: 场景目录结构

| 序号 | 任务 | 文件路径 | 验证标准 |
|------|------|----------|----------|
| 3.6 | 创建 knowledge_qa 目录 | `subgraph/knowledge_qa/` | 目录存在 |
| 3.7 | 实现 KnowledgeQAConfig | `subgraph/knowledge_qa/config.py` | 配置完整 |
| 3.8 | 实现 KnowledgeQAEngine 骨架 | `subgraph/knowledge_qa/engine.py` | 引擎可实例化 |
| 3.9 | 创建 issue_analysis 目录 | `subgraph/issue_analysis/` | 目录存在 |
| 3.10 | 实现 IssueAnalysisConfig | `subgraph/issue_analysis/config.py` | 配置完整 |
| 3.11 | 实现 IssueAnalysisEngine 骨架 | `subgraph/issue_analysis/engine.py` | 引擎可实例化 |
| 3.12 | 单元测试 | `tests/workflow/subgraph/` | 测试通过 |

### 4.3 产出物

```
src/workflow/subgraph/
├── __init__.py
├── base.py                     # SubgraphConfig, BaseSubgraph
├── registry.py                 # SubgraphRegistry
│
├── knowledge_qa/
│   ├── __init__.py
│   ├── config.py               # KnowledgeQAConfig
│   ├── engine.py               # KnowledgeQAEngine
│   └── nodes/                  # 空目录，Phase 4 填充
│
└── issue_analysis/
    ├── __init__.py
    ├── config.py               # IssueAnalysisConfig
    ├── engine.py               # IssueAnalysisEngine
    └── nodes/                  # 空目录，Phase 4 填充
```

### 4.4 验收标准

```python
# 验收测试代码

from workflow.subgraph.registry import get_subgraph_registry
from workflow.subgraph.knowledge_qa import KnowledgeQAConfig, KnowledgeQAEngine

# 1. 注册子图
registry = get_subgraph_registry()
registry.register(
    subgraph_id="knowledge_qa",
    config=KnowledgeQAConfig(),
    factory=KnowledgeQAEngine,
)

# 2. 创建子图实例
subgraph = registry.create(
    subgraph_id="knowledge_qa",
    agent_loop_engine=mock_engine,
    common_nodes=mock_common_nodes,
)

assert subgraph is not None
assert subgraph.config.subgraph_id == "knowledge_qa"
```

---

## 5. Phase 4: 场景迁移（3 天）

### 5.1 目标

- 迁移 knowledge_qa 场景节点
- 迁移 issue_analysis 场景节点
- 实现完整的 Subgraph 流程

### 5.2 任务清单

#### Day 1: 公共节点迁移

| 序号 | 任务 | 源路径 | 目标路径 | 验证标准 |
|------|------|--------|----------|----------|
| 4.1 | 迁移 finalize_response | `nodes/control_response/finalize_response/` | `common_nodes/finalize_response.py` | 节点可运行 |
| 4.2 | 迁移 out_of_scope_response | `nodes/control_response/out_of_scope_response/` | `common_nodes/out_of_scope_response.py` | 节点可运行 |
| 4.3 | 迁移 evidence_merger | `nodes/retrieval_flow/merge_evidence/` | `common_nodes/evidence_merger.py` | 节点可运行 |
| 4.4 | 更新导入 | 所有引用文件 | - | 导入正确 |
| 4.5 | 单元测试 | `tests/workflow/common_nodes/` | - | 测试通过 |

#### Day 2: knowledge_qa 迁移

| 序号 | 任务 | 源路径 | 目标路径 | 验证标准 |
|------|------|--------|----------|----------|
| 4.6 | 迁移 query_rewriter | `nodes/analysis/knowledge_answer/` | `subgraph/knowledge_qa/nodes/query_rewriter.py` | 节点可运行 |
| 4.7 | 实现 retrieval_flow | - | `subgraph/knowledge_qa/nodes/retrieval_flow.py` | 节点可运行 |
| 4.8 | 完善 KnowledgeQAEngine | `subgraph/knowledge_qa/engine.py` | - | 图可编译 |
| 4.9 | 集成测试 | `tests/workflow/subgraph/knowledge_qa/` | - | 测试通过 |

#### Day 3: issue_analysis 迁移

| 序号 | 任务 | 源路径 | 目标路径 | 验证标准 |
|------|------|--------|----------|----------|
| 4.10 | 迁移 context_loader | `nodes/analysis/issue_analysis/` | `subgraph/issue_analysis/nodes/context_loader.py` | 节点可运行 |
| 4.11 | 实现 issue_classifier | - | `subgraph/issue_analysis/nodes/issue_classifier.py` | 节点可运行 |
| 4.12 | 完善 IssueAnalysisEngine | `subgraph/issue_analysis/engine.py` | - | 图可编译 |
| 4.13 | 集成测试 | `tests/workflow/subgraph/issue_analysis/` | - | 测试通过 |

### 5.3 产出物

```
src/workflow/
├── common_nodes/
│   ├── __init__.py
│   ├── finalize_response.py        # 迁移完成
│   ├── out_of_scope_response.py    # 迁移完成
│   ├── evidence_merger.py          # 迁移完成
│   └── agent_loop_node/            # Phase 2 完成
│
└── subgraph/
    ├── knowledge_qa/
    │   ├── config.py
    │   ├── engine.py               # 完整实现
    │   └── nodes/
    │       ├── query_rewriter.py   # 迁移完成
    │       └── retrieval_flow.py   # 新增
    │
    └── issue_analysis/
        ├── config.py
        ├── engine.py               # 完整实现
        └── nodes/
            ├── context_loader.py   # 迁移完成
            └── issue_classifier.py # 新增
```

### 5.4 验收标准

```python
# 验收测试代码 - knowledge_qa

from workflow.subgraph.knowledge_qa import KnowledgeQAEngine, KnowledgeQAConfig

# 1. 创建引擎
engine = KnowledgeQAEngine(
    config=KnowledgeQAConfig(),
    agent_loop_engine=mock_agent_loop_engine,
    common_nodes={
        "finalize_response": finalize_response,
    },
)

# 2. 执行完整流程
result = await engine.run({
    "user_query": "CTR 预估模块的入口函数在哪里？",
    "session_id": "test-session",
})

assert "answer" in result
assert result["agent_loop_completed"] is True
```

---

## 6. Phase 5: 测试与文档（2 天）

### 6.1 目标

- 完善测试覆盖率
- 更新 API 文档
- 验证端到端流程

### 6.2 任务清单

#### Day 1: 测试完善

| 序号 | 任务 | 范围 | 验证标准 |
|------|------|------|----------|
| 5.1 | Agent Loop 单元测试 | `tests/agent/` | 覆盖率 > 80% |
| 5.2 | Subgraph 单元测试 | `tests/workflow/subgraph/` | 覆盖率 > 80% |
| 5.3 | 公共节点单元测试 | `tests/workflow/common_nodes/` | 覆盖率 > 80% |
| 5.4 | 集成测试 | `tests/integration/` | 关键流程通过 |
| 5.5 | 端到端测试 | `tests/e2e/` | API 调用正常 |

#### Day 2: 文档更新

| 序号 | 任务 | 文件 | 内容 |
|------|------|------|------|
| 5.6 | 更新 README | `README.md` | v3 架构说明 |
| 5.7 | 更新 API 文档 | `docs/api.md` | v3 接口说明 |
| 5.8 | 更新开发指南 | `docs/development.md` | 开发规范 |
| 5.9 | 清理 v1 文档 | `docs/` | 标记废弃 |

### 6.3 产出物

```
tests/
├── agent/
│   ├── test_engine.py
│   ├── test_config.py
│   └── core/
│       ├── test_executor.py
│       ├── test_tool_caller.py
│       └── test_prompt_builder.py
│
├── workflow/
│   ├── common_nodes/
│   │   ├── test_agent_loop_node.py
│   │   ├── test_finalize_response.py
│   │   └── test_out_of_scope.py
│   │
│   └── subgraph/
│       ├── test_base.py
│       ├── test_registry.py
│       ├── knowledge_qa/
│       │   └── test_engine.py
│       └── issue_analysis/
│           └── test_engine.py
│
├── integration/
│   └── test_subgraph_integration.py
│
└── e2e/
    └── test_api_e2e.py
```

### 6.4 验收标准

```bash
# 1. 运行所有测试
pytest tests/ -v --cov=src --cov-report=html

# 2. 检查覆盖率
# 总覆盖率 > 80%

# 3. 端到端测试
curl -X POST http://localhost:8000/api/messages \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test", "message": "测试问题"}'

# 4. 检查响应
# HTTP 200
# 包含 answer 字段
```

---

## 7. 风险与缓解

### 7.1 风险列表

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| 导入路径遗漏 | 高 | 中 | 使用全局搜索，分批迁移 |
| LLM 调用失败 | 中 | 高 | Mock 测试，降级策略 |
| 性能下降 | 低 | 中 | 性能基准测试 |
| 向后兼容问题 | 中 | 高 | 保留 v1 代码，配置开关 |

### 7.2 回滚策略

```bash
# 每个 Phase 完成后打 Tag
git tag v3-phase1-done
git tag v3-phase2-done
git tag v3-phase3-done
git tag v3-phase4-done
git tag v3-phase5-done

# 如果出现问题，回滚到对应 Tag
git checkout v3-phase3-done
```

---

## 8. 检查清单

### 8.1 Phase 完成检查

| Phase | 检查项 | 状态 |
|-------|--------|------|
| Phase 1 | 目录结构正确 | [ ] |
| Phase 1 | 导入路径更新 | [ ] |
| Phase 1 | 现有测试通过 | [ ] |
| Phase 2 | AgentLoopEngine 可运行 | [ ] |
| Phase 2 | agent_loop_node 可创建 | [ ] |
| Phase 2 | 单元测试通过 | [ ] |
| Phase 3 | Subgraph 基类完整 | [ ] |
| Phase 3 | Registry 可工作 | [ ] |
| Phase 3 | 场景目录结构正确 | [ ] |
| Phase 4 | 公共节点迁移完成 | [ ] |
| Phase 4 | knowledge_qa 可运行 | [ ] |
| Phase 4 | issue_analysis 可运行 | [ ] |
| Phase 5 | 测试覆盖率 > 80% | [ ] |
| Phase 5 | 文档更新完成 | [ ] |
| Phase 5 | 端到端测试通过 | [ ] |

### 8.2 最终验收

- [ ] 所有单元测试通过
- [ ] 所有集成测试通过
- [ ] API 接口正常响应
- [ ] 文档更新完整
- [ ] 代码 Review 完成
- [ ] 性能无明显下降

---

## 9. 时间线甘特图

```
Phase           Day 1   Day 2   Day 3   Day 4   Day 5   Day 6   Day 7   Day 8   Day 9   Day 10  Day 11  Day 12
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Phase 1         ████████
目录重组        [====]

Phase 2                 ████████████████████
Agent Loop              [========]

Phase 3                                 ████████████████████
Subgraph                                [========]

Phase 4                                                 ████████████████████████████████
场景迁移                                                [================]

Phase 5                                                                                 ████████████████████
测试文档                                                                                [========]
```

---

## 10. 资源需求

### 10.1 人力

| 角色 | 职责 | 投入 |
|------|------|------|
| 后端开发 | 核心代码实现 | 100% |
| 测试工程师 | 测试用例编写 | 50% |
| 文档工程师 | 文档更新 | 30% |

### 10.2 环境

| 环境 | 用途 |
|------|------|
| 开发环境 | 日常开发 |
| 测试环境 | 集成测试 |
| 预发布环境 | 端到端验证 |

---

## 附录

### A. 相关文档

| 文档 | 路径 |
|------|------|
| 架构设计 | `docs/v3/design.md` |
| 目录结构 | `docs/v3/directory_structure.md` |
| Agent Loop 节点 | `docs/v3/agent_loop_node.md` |
| Subgraph 设计 | `docs/v3/subgraph_design.md` |

### B. 联系人

| 角色 | 负责人 |
|------|--------|
| 架构师 | TBD |
| 后端负责人 | TBD |
| 测试负责人 | TBD |
