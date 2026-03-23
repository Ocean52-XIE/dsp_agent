# Subgraph 设计文档

## 1. 概述

### 1.1 什么是 Subgraph

Subgraph（子图）是特定场景的 Workflow 封装，包含：
1. **固定节点**：检索、重写等确定性流程
2. **Agent Loop 节点**：使用公共节点 `agent_loop_node`
3. **配置**：工具白名单、提示词等

### 1.2 设计目标

1. **场景封装**：将特定场景的完整流程封装为子图
2. **权限隔离**：不同场景有不同的 tool/skill 白名单
3. **复用性**：公共节点（含 agent_loop_node）可被多个子图复用
4. **可扩展**：方便添加新的场景子图

---

## 2. 架构设计

### 2.1 Subgraph 在系统中的位置

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           WorkflowService                                    │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                         DomainRouter                                   │  │
│  │    OUT_OF_SCOPE / SMALL_TALK / MODULE_ROUTED / PASS_TO_AGENT          │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                      │                                       │
│         ┌────────────────────────────┼────────────────────────────┐         │
│         │                            │                            │         │
│         ▼                            ▼                            ▼         │
│  ┌─────────────────┐       ┌─────────────────────┐       ┌─────────────────┐│
│  │ control_response │       │  knowledge_qa       │       │ issue_analysis  ││
│  │   (控制响应)     │       │   (知识问答)         │       │   (问题分析)    ││
│  │                 │       │                     │       │                 ││
│  │ - out_of_scope  │       │ - query_rewriter    │       │ - load_context  ││
│  │ - finalize ◀────┼───────│ - retrieval_flow    │       │ - agent_loop ◀──┼──┐
│  │                 │       │ - agent_loop ◀──────┼───────│ - finalize ◀────┼──┤
│  │                 │       │ - finalize ◀────────┼───────│                 │  │
│  └─────────────────┘       └─────────────────────┘       └─────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    common_nodes/agent_loop_node/                       │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │ create_agent_loop_node()                                        │  │  │
│  │  │   └── 调用 agent/engine.py (AgentLoopEngine)              │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Subgraph 类型

| Subgraph | 场景 | 节点流程 | 公共节点 |
|----------|------|----------|----------|
| **knowledge_qa** | 知识问答 | query_rewriter → retrieval_flow → agent_loop → finalize | agent_loop_node, finalize |
| **issue_analysis** | 问题分析 | load_context → agent_loop → finalize | agent_loop_node, finalize |
| **code_generation** | 代码生成 | load_context → retrieve_code → agent_loop → finalize | agent_loop_node, finalize |

---

## 3. 目录结构

```
src/workflow/subgraph/
├── __init__.py
├── base.py                     # BaseSubgraph, SubgraphConfig
├── registry.py                 # SubgraphRegistry
│
├── knowledge_qa/               # 知识问答场景
│   ├── __init__.py
│   ├── config.py               # KnowledgeQAConfig
│   ├── engine.py               # KnowledgeQAEngine
│   └── nodes/                  # 独有节点
│       ├── __init__.py
│       ├── query_rewriter.py
│       └── retrieval_flow.py
│
└── issue_analysis/             # 问题分析场景
    ├── __init__.py
    ├── config.py               # IssueAnalysisConfig
    ├── engine.py               # IssueAnalysisEngine
    └── nodes/                  # 独有节点
        ├── __init__.py
        ├── context_loader.py
        └── issue_classifier.py
```

---

## 4. 基类设计

### 4.1 SubgraphConfig 基类

```python
# src/workflow/subgraph/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable
from langgraph.graph import StateGraph

@dataclass
class SubgraphConfig:
    """Subgraph 配置基类"""
    # 基础信息
    subgraph_id: str                          # 子图唯一标识
    display_name: str                         # 显示名称
    description: str = ""                     # 描述

    # 节点定义
    nodes: list[str] = field(default_factory=list)          # 节点名称列表
    agent_loop_nodes: list[str] = field(default_factory=list)  # Agent Loop 节点

    # 工具约束
    available_tools: list[str] = field(default_factory=list)   # 工具白名单
    available_skills: list[str] = field(default_factory=list)  # 技能白名单

    # 节点配置
    node_configs: dict[str, dict[str, Any]] = field(default_factory=dict)


class BaseSubgraph(ABC):
    """Subgraph 基类

    子类需要实现：
    1. _build_graph(): 构建子图
    2. get_nodes(): 获取所有节点函数
    """

    def __init__(
        self,
        config: SubgraphConfig,
        agent_loop_engine: "AgentLoopEngine",
        common_nodes: dict[str, Callable],
    ):
        self.config = config
        self.agent_loop_engine = agent_loop_engine
        self.common_nodes = common_nodes
        self._graph = None

    @property
    def graph(self) -> StateGraph:
        """获取编译后的图（延迟构建）"""
        if self._graph is None:
            self._graph = self._build_graph()
        return self._graph

    @abstractmethod
    def _build_graph(self) -> StateGraph:
        """构建子图（子类实现）"""
        pass

    @abstractmethod
    def get_nodes(self) -> dict[str, Callable]:
        """获取所有节点函数（子类实现）"""
        pass

    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """执行子图"""
        state["_subgraph_id"] = self.config.subgraph_id
        result = await self.graph.ainvoke(state)
        result.pop("_subgraph_id", None)
        return result

    def _create_agent_loop_node(self, node_name: str) -> Callable:
        """创建 Agent Loop 节点（使用公共节点）"""
        # 【关键】从 common_nodes 获取工厂函数
        from workflow.common_nodes.agent_loop_node import create_agent_loop_node, NodeExecutionContext

        node_config = self.config.node_configs.get(node_name, {})

        # 合并约束
        tool_whitelist = node_config.get("tool_whitelist", self.config.available_tools)
        skill_whitelist = node_config.get("skill_whitelist", self.config.available_skills)

        return create_agent_loop_node(
            engine=self.agent_loop_engine,
            node_name=node_name,
            context=NodeExecutionContext(
                tool_whitelist=tool_whitelist,
                skill_whitelist=skill_whitelist,
                system_prompt_append=node_config.get("system_prompt_append"),
                max_iterations=node_config.get("max_iterations", 5),
                inject_evidence=node_config.get("inject_evidence", True),
                output_format=node_config.get("output_format", "markdown"),
            )
        )
```

### 4.2 SubgraphRegistry

```python
# src/workflow/subgraph/registry.py

from typing import Type
from workflow.subgraph.base import BaseSubgraph, SubgraphConfig

class SubgraphRegistry:
    """Subgraph 注册中心"""

    def __init__(self):
        self._configs: dict[str, SubgraphConfig] = {}
        self._factories: dict[str, Type[BaseSubgraph]] = {}

    def register(
        self,
        subgraph_id: str,
        config: SubgraphConfig,
        factory: Type[BaseSubgraph],
    ) -> None:
        """注册子图"""
        self._configs[subgraph_id] = config
        self._factories[subgraph_id] = factory

    def create(
        self,
        subgraph_id: str,
        agent_loop_engine: "AgentLoopEngine",
        common_nodes: dict,
    ) -> BaseSubgraph | None:
        """创建子图实例"""
        config = self._configs.get(subgraph_id)
        factory = self._factories.get(subgraph_id)

        if not config or not factory:
            return None

        return factory(
            config=config,
            agent_loop_engine=agent_loop_engine,
            common_nodes=common_nodes,
        )

    def list_subgraphs(self) -> list[str]:
        """列出所有子图 ID"""
        return list(self._configs.keys())


# 全局注册中心
_registry: SubgraphRegistry | None = None

def get_subgraph_registry() -> SubgraphRegistry:
    """获取全局子图注册中心"""
    global _registry
    if _registry is None:
        _registry = SubgraphRegistry()
    return _registry
```

---

## 5. Knowledge QA Subgraph

### 5.1 配置定义

```python
# src/workflow/subgraph/knowledge_qa/config.py

from dataclasses import dataclass
from workflow.subgraph.base import SubgraphConfig

@dataclass
class KnowledgeQAConfig(SubgraphConfig):
    """知识问答场景配置"""

    subgraph_id: str = "knowledge_qa"
    display_name: str = "知识问答"
    description: str = "基于知识库和代码库的问答场景"

    nodes: list[str] = None
    agent_loop_nodes: list[str] = None
    available_tools: list[str] = None
    available_skills: list[str] = None
    node_configs: dict = None

    def __post_init__(self):
        if self.nodes is None:
            self.nodes = [
                "query_rewriter",
                "retrieval_flow",
                "knowledge_answer",    # Agent Loop 节点
                "finalize_response",
            ]

        if self.agent_loop_nodes is None:
            self.agent_loop_nodes = ["knowledge_answer"]

        if self.available_tools is None:
            self.available_tools = [
                "skill_manager",
                "code_retriever",
                "wiki_search",
            ]

        if self.available_skills is None:
            self.available_skills = [
                "query_code",
                "query_wiki",
            ]

        if self.node_configs is None:
            self.node_configs = {
                "knowledge_answer": {
                    "system_prompt_append": "你是知识问答助手...",
                    "max_iterations": 3,
                    "inject_evidence": True,
                    "output_format": "markdown",
                }
            }
```

### 5.2 引擎实现

```python
# src/workflow/subgraph/knowledge_qa/engine.py

from langgraph.graph import StateGraph
from workflow.subgraph.base import BaseSubgraph
from workflow.subgraph.knowledge_qa.config import KnowledgeQAConfig

class KnowledgeQAEngine(BaseSubgraph):
    """知识问答子图引擎

    流程：
    1. query_rewriter: 查询重写（独有节点）
    2. retrieval_flow: 检索（独有节点）
    3. knowledge_answer: Agent Loop（公共节点）
    4. finalize_response: 结果整理（公共节点）
    """

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(dict)

        nodes = self.get_nodes()

        for node_name in self.config.nodes:
            graph.add_node(node_name, nodes[node_name])

        # 顺序边
        for i in range(len(self.config.nodes) - 1):
            graph.add_edge(self.config.nodes[i], self.config.nodes[i + 1])

        graph.set_entry_point(self.config.nodes[0])
        graph.set_finish_point(self.config.nodes[-1])

        return graph.compile()

    def get_nodes(self) -> dict:
        """获取所有节点"""
        # 1. 独有节点
        from .nodes.query_rewriter import run as query_rewriter
        from .nodes.retrieval_flow import run as retrieval_flow

        # 2. 公共节点
        # - finalize_response 来自 common_nodes
        # - knowledge_answer 使用 agent_loop_node

        return {
            # 独有节点
            "query_rewriter": query_rewriter,
            "retrieval_flow": retrieval_flow,
            # 公共节点：Agent Loop
            "knowledge_answer": self._create_agent_loop_node("knowledge_answer"),
            # 公共节点：Finalize
            "finalize_response": self.common_nodes["finalize_response"],
        }
```

---

## 6. 与公共节点的集成

### 6.1 获取公共节点

```python
# src/workflow/service.py

from workflow.subgraph.registry import get_subgraph_registry
from agent.engine import AgentLoopEngine

class WorkflowService:
    """Workflow 服务入口"""

    def __init__(self, ...):
        # 初始化 Agent Loop 引擎
        self.agent_loop_engine = AgentLoopEngine(
            llm_client=self._llm_client,
            tool_registry=self._tool_registry,
            skill_registry=self._skill_registry,
        )

        # 获取公共节点
        self._common_nodes = self._get_common_nodes()

        # 初始化子图注册中心
        self.subgraph_registry = get_subgraph_registry()
        self._register_subgraphs()

    def _get_common_nodes(self) -> dict:
        """获取公共节点"""
        from workflow.common_nodes.finalize_response import run as finalize_response
        from workflow.common_nodes.out_of_scope_response import run as out_of_scope

        return {
            "finalize_response": finalize_response,
            "out_of_scope": out_of_scope,
        }

    def _register_subgraphs(self):
        """注册所有子图"""
        from workflow.subgraph.knowledge_qa import KnowledgeQAEngine, KnowledgeQAConfig
        from workflow.subgraph.issue_analysis import IssueAnalysisEngine, IssueAnalysisConfig

        self.subgraph_registry.register(
            subgraph_id="knowledge_qa",
            config=KnowledgeQAConfig(),
            factory=KnowledgeQAEngine,
        )

        self.subgraph_registry.register(
            subgraph_id="issue_analysis",
            config=IssueAnalysisConfig(),
            factory=IssueAnalysisEngine,
        )
```

### 6.2 子图获取 agent_loop_node 的方式

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Subgraph 获取公共节点                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   KnowledgeQAEngine                                                          │
│        │                                                                     │
│        │  get_nodes()                                                        │
│        │                                                                     │
│        ├─── 独有节点 ────────────────────────────────────────────────────    │
│        │    from .nodes.query_rewriter import run                           │
│        │    from .nodes.retrieval_flow import run                           │
│        │                                                                     │
│        └─── 公共节点 ────────────────────────────────────────────────────    │
│             │                                                                │
│             ├── finalize_response                                           │
│             │    └── self.common_nodes["finalize_response"]                 │
│             │                                                                │
│             └── knowledge_answer (Agent Loop)                               │
│                  │                                                           │
│                  └── self._create_agent_loop_node("knowledge_answer")       │
│                           │                                                  │
│                           ▼                                                  │
│                  ┌──────────────────────────────────────────────┐            │
│                  │  workflow/common_nodes/agent_loop_node/      │            │
│                  │  ┌────────────────────────────────────────┐  │            │
│                  │  │ create_agent_loop_node()               │  │            │
│                  │  │   └── 调用 agent/engine.py        │  │            │
│                  │  └────────────────────────────────────────┘  │            │
│                  └──────────────────────────────────────────────┘            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. 配置示例

### 7.1 profile.json 配置

```json
{
  "modules": [
    {
      "name": "knowledge-qa",
      "display_name": "知识问答",
      "keywords": ["怎么", "如何", "什么是"],
      "subgraph": "knowledge_qa"
    }
  ],

  "subgraphs": {
    "knowledge_qa": {
      "available_tools": ["skill_manager", "code_retriever", "wiki_search"],
      "available_skills": ["query_code", "query_wiki"],
      "node_configs": {
        "knowledge_answer": {
          "system_prompt_append": "你是企业知识问答助手...",
          "max_iterations": 3
        }
      }
    }
  }
}
```

---

## 8. 导入规范

### 8.1 正确的导入方式

```python
# ✅ 独有节点导入
from workflow.subgraph.knowledge_qa.nodes.query_rewriter import run as query_rewriter
from workflow.subgraph.issue_analysis.nodes.context_loader import run as context_loader

# ✅ 公共节点导入
from workflow.common_nodes.finalize_response import run as finalize_response
from workflow.common_nodes.agent_loop_node import create_agent_loop_node, NodeExecutionContext

# ✅ Agent Loop 引擎导入
from agent.engine import AgentLoopEngine
from agent.skills.registry import SkillRegistry

# ✅ 基础设施导入（src/ 顶层）
from retrievers.wiki_retriever import WikiRetriever
from session.session_store import SessionStore
from observability.metrics import record_metric
```

---

## 9. 总结

### 9.1 核心设计要点

| 要点 | 说明 |
|------|------|
| **Subgraph 职责** | 场景封装，编排独有节点和公共节点 |
| **公共节点位置** | `workflow/common_nodes/`（含 agent_loop_node） |
| **agent_loop_node** | 作为公共节点，可被多个 Subgraph 复用 |
| **约束机制** | Subgraph → Node 逐层收窄 |

### 9.2 节点获取方式

| 节点类型 | 获取方式 |
|----------|----------|
| 独有节点 | `from .nodes.xxx import run` |
| 公共节点 | `self.common_nodes["xxx"]` |
| Agent Loop 节点 | `self._create_agent_loop_node("xxx")` |
