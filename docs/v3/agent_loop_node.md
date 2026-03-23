# Agent Loop Node 设计文档

> **注意**：本文档描述的是最初的设计方案。重构后的设计请参考 [agent_loop_node_refactor.md](./agent_loop_node_refactor.md)。

## 1. 概述

### 1.1 定位

`agent_loop_node` 是一个**公共节点**，位于 `workflow/common_nodes/agent_loop_node/`，可被多个 Subgraph 复用。

### 1.2 核心职责

1. **封装 AgentLoopEngine**：将引擎封装为 LangGraph 节点
2. **提供配置接口**：通过 `NodeExecutionContext` 配置行为
3. **应用约束机制**：工具/技能白名单、迭代次数等
4. **上下文注入**：证据、历史等上下文自动注入

### 1.3 设计原则

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          设计原则                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   1. 作为公共节点：位于 common_nodes/，可被多个 Subgraph 复用               │
│                                                                              │
│   2. 配置驱动：通过 NodeExecutionContext 配置，无需修改代码                 │
│                                                                              │
│   3. 约束继承：继承 Subgraph 级别约束，可进一步收窄                         │
│                                                                              │
│   4. 引擎分离：核心逻辑在 AgentLoopEngine，节点只是封装层                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 目录结构

```
src/workflow/common_nodes/agent_loop_node/
├── __init__.py                 # 导出接口
├── node.py                     # 节点函数实现
└── context.py                  # 执行上下文定义
```

---

## 3. 核心组件

### 3.1 NodeExecutionContext（执行上下文）

```python
# src/workflow/common_nodes/agent_loop_node/context.py

from dataclasses import dataclass, field
from typing import Any

@dataclass
class NodeExecutionContext:
    """Agent Loop 节点执行上下文

    用于配置 agent_loop_node 的行为，包括：
    - 工具/技能白名单
    - 提示词定制
    - 循环控制
    - 输出格式
    """

    # ========== 工具约束 ==========

    tool_whitelist: list[str] = field(default_factory=list)
    """工具白名单

    如果为空，则使用 Subgraph 级别的约束；
    如果 Subgraph 级别也为空，则使用全局默认。

    Example:
        ["skill_manager", "code_retriever", "wiki_search"]
    """

    skill_whitelist: list[str] = field(default_factory=list)
    """技能白名单

    当 LLM 调用 skill_manager 时，会检查 skill_name 是否在白名单中。

    Example:
        ["query_code", "query_wiki", "search_similar_issues"]
    """

    # ========== 提示词配置 ==========

    system_prompt_append: str | None = None
    """追加到默认系统提示词后的内容

    用于场景特定的指令注入。

    Example:
        "你是企业知识问答助手，必须严格基于提供的证据回答。"
    """

    system_prompt_override: str | None = None
    """覆盖默认系统提示词

    如果设置，将完全替换默认提示词。
    """

    # ========== 循环控制 ==========

    max_iterations: int = 5
    """最大迭代次数

    LLM + 工具调用的最大循环次数。
    """

    timeout_seconds: int = 60
    """超时时间（秒）

    单次节点执行的超时时间。
    """

    # ========== 上下文注入 ==========

    inject_evidence: bool = True
    """是否注入检索证据

    如果为 True，会自动将 state["evidence_hits"] 注入到用户消息中。
    """

    inject_history: bool = False
    """是否注入对话历史

    如果为 True，会自动将 state["history"] 注入到用户消息中。
    """

    # ========== 输出配置 ==========

    output_format: str = "markdown"
    """输出格式

    - text: 纯文本
    - markdown: Markdown 格式
    - json: JSON 格式（需要配合 output_schema）
    """

    output_schema: dict[str, Any] | None = None
    """JSON 输出 Schema

    当 output_format="json" 时，用于定义输出结构。
    """

    # ========== 调试配置 ==========

    debug_mode: bool = False
    """调试模式

    如果为 True，会输出详细的执行日志。
    """

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "tool_whitelist": self.tool_whitelist,
            "skill_whitelist": self.skill_whitelist,
            "system_prompt_append": self.system_prompt_append,
            "system_prompt_override": self.system_prompt_override,
            "max_iterations": self.max_iterations,
            "timeout_seconds": self.timeout_seconds,
            "inject_evidence": self.inject_evidence,
            "inject_history": self.inject_history,
            "output_format": self.output_format,
            "output_schema": self.output_schema,
            "debug_mode": self.debug_mode,
        }
```

### 3.2 create_agent_loop_node（节点工厂函数）

```python
# src/workflow/common_nodes/agent_loop_node/node.py

from typing import Callable, Any
import logging

from agent.engine import AgentLoopEngine
from .context import NodeExecutionContext

logger = logging.getLogger(__name__)


def create_agent_loop_node(
    engine: AgentLoopEngine,
    node_name: str,
    context: NodeExecutionContext,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """创建 Agent Loop 节点

    将 AgentLoopEngine 封装为 LangGraph 节点函数。

    Args:
        engine: AgentLoopEngine 实例（核心引擎）
        node_name: 节点名称（用于标识和调试）
        context: 节点执行上下文（配置约束、提示词等）

    Returns:
        可作为 LangGraph 节点的异步函数

    Usage:
        from workflow.common_nodes.agent_loop_node import (
            create_agent_loop_node,
            NodeExecutionContext
        )

        # 创建知识问答节点
        knowledge_answer_node = create_agent_loop_node(
            engine=agent_loop_engine,
            node_name="knowledge_answer",
            context=NodeExecutionContext(
                tool_whitelist=["skill_manager", "wiki_search"],
                skill_whitelist=["query_code", "query_wiki"],
                system_prompt_append="你是知识问答助手...",
                max_iterations=3,
            )
        )

        # 添加到图
        graph.add_node("knowledge_answer", knowledge_answer_node)

    Raises:
        ValueError: 如果 engine 为 None
    """
    if engine is None:
        raise ValueError("engine 不能为 None")

    async def node_fn(state: dict[str, Any]) -> dict[str, Any]:
        """节点函数

        Args:
            state: Workflow 状态

        Returns:
            更新后的状态增量
        """
        # 记录开始
        logger.info(
            f"[AgentLoopNode:{node_name}] 开始执行, "
            f"query={state.get('user_query', '')[:50]}..."
        )

        # 添加节点标识（用于调试）
        state["_agent_loop_node"] = node_name

        try:
            # 调用引擎执行
            result = await engine.run(state, context)

            # 记录完成
            logger.info(
                f"[AgentLoopNode:{node_name}] 执行完成, "
                f"iterations={result.get('agent_loop_iterations', 0)}, "
                f"completed={result.get('agent_loop_completed', False)}"
            )

            return result

        except Exception as e:
            logger.error(f"[AgentLoopNode:{node_name}] 执行失败: {e}")
            return {
                "answer": f"抱歉，处理您的请求时出现错误：{str(e)}",
                "agent_loop_error": str(e),
                "agent_loop_completed": False,
            }

        finally:
            # 清理临时字段
            state.pop("_agent_loop_node", None)

    # 设置函数名便于调试
    node_fn.__name__ = f"agent_loop_node_{node_name}"
    node_fn.__qualname__ = f"agent_loop_node_{node_name}"

    # 添加元数据
    node_fn._agent_loop_context = context
    node_fn._agent_loop_node_name = node_name

    return node_fn
```

### 3.3 `__init__.py`

```python
# src/workflow/common_nodes/agent_loop_node/__init__.py

"""Agent Loop 公共节点

将 AgentLoopEngine 封装为 LangGraph 节点，可被多个 Subgraph 复用。

Usage:
    from workflow.common_nodes.agent_loop_node import (
        create_agent_loop_node,
        NodeExecutionContext
    )

    # 创建节点
    node = create_agent_loop_node(
        engine=agent_loop_engine,
        node_name="knowledge_answer",
        context=NodeExecutionContext(
            tool_whitelist=["skill_manager"],
            max_iterations=3,
        )
    )

    # 添加到图
    graph.add_node("knowledge_answer", node)
"""

from .node import create_agent_loop_node
from .context import NodeExecutionContext

__all__ = [
    "create_agent_loop_node",
    "NodeExecutionContext",
]
```

---

## 4. 使用示例

### 4.1 基础用法

```python
from workflow.common_nodes.agent_loop_node import (
    create_agent_loop_node,
    NodeExecutionContext
)
from agent.engine import AgentLoopEngine

# 初始化引擎
engine = AgentLoopEngine(
    llm_client=llm_client,
    tool_registry=tool_registry,
    skill_registry=skill_registry,
)

# 创建节点
knowledge_answer_node = create_agent_loop_node(
    engine=engine,
    node_name="knowledge_answer",
    context=NodeExecutionContext(
        tool_whitelist=["skill_manager", "wiki_search"],
        max_iterations=3,
        inject_evidence=True,
        output_format="markdown",
    )
)

# 添加到 LangGraph
graph.add_node("knowledge_answer", knowledge_answer_node)
```

### 4.2 在 Subgraph 中使用

```python
# src/workflow/subgraph/knowledge_qa/engine.py

from langgraph.graph import StateGraph
from workflow.subgraph.base import BaseSubgraph
from workflow.common_nodes.agent_loop_node import (
    create_agent_loop_node,
    NodeExecutionContext
)
from workflow.common_nodes.finalize_response import run as finalize_response


class KnowledgeQAEngine(BaseSubgraph):
    """知识问答子图"""

    def get_nodes(self) -> dict:
        """获取所有节点"""
        # 1. 独有节点
        from .nodes.query_rewriter import run as query_rewriter
        from .nodes.retrieval_flow import run as retrieval_flow

        # 2. 创建 Agent Loop 公共节点
        knowledge_answer_node = create_agent_loop_node(
            engine=self.agent_loop_engine,
            node_name="knowledge_answer",
            context=NodeExecutionContext(
                # 继承 Subgraph 级别的约束
                tool_whitelist=self.config.available_tools,
                skill_whitelist=self.config.available_skills,
                # 节点级别的配置
                system_prompt_append=self._get_system_prompt(),
                max_iterations=3,
                inject_evidence=True,
                output_format="markdown",
            )
        )

        return {
            # 独有节点
            "query_rewriter": query_rewriter,
            "retrieval_flow": retrieval_flow,
            # 公共节点
            "knowledge_answer": knowledge_answer_node,
            "finalize_response": finalize_response,
        }

    def _get_system_prompt(self) -> str:
        """获取系统提示词"""
        base = "你是企业知识问答助手。"
        append = self.config.node_configs.get("knowledge_answer", {}).get(
            "system_prompt_append", ""
        )
        return f"{base}\n\n{append}" if append else base
```

### 4.3 从配置创建节点

```python
def create_agent_loop_node_from_config(
    engine: AgentLoopEngine,
    node_name: str,
    config: dict,
    subgraph_tools: list[str] = None,
    subgraph_skills: list[str] = None,
) -> Callable:
    """从配置创建 Agent Loop 节点

    Args:
        engine: AgentLoopEngine 实例
        node_name: 节点名称
        config: 节点配置字典
        subgraph_tools: Subgraph 级别的工具白名单
        subgraph_skills: Subgraph 级别的技能白名单

    Returns:
        节点函数
    """
    # 合并约束：节点级别优先，否则继承 Subgraph 级别
    tool_whitelist = config.get("tool_whitelist", subgraph_tools or [])
    skill_whitelist = config.get("skill_whitelist", subgraph_skills or [])

    return create_agent_loop_node(
        engine=engine,
        node_name=node_name,
        context=NodeExecutionContext(
            tool_whitelist=tool_whitelist,
            skill_whitelist=skill_whitelist,
            system_prompt_append=config.get("system_prompt_append"),
            max_iterations=config.get("max_iterations", 5),
            inject_evidence=config.get("inject_evidence", True),
            output_format=config.get("output_format", "markdown"),
        )
    )


# 使用示例
node_config = profile["subgraphs"]["knowledge_qa"]["node_configs"]["knowledge_answer"]
node = create_agent_loop_node_from_config(
    engine=agent_loop_engine,
    node_name="knowledge_answer",
    config=node_config,
    subgraph_tools=profile["subgraphs"]["knowledge_qa"]["available_tools"],
    subgraph_skills=profile["subgraphs"]["knowledge_qa"]["available_skills"],
)
```

---

## 5. 与 AgentLoopEngine 的关系

### 5.1 职责分离

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           职责分离图                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   agent_loop_node/ (公共节点)                  agent/ (核心引擎)        │
│   ┌─────────────────────────────┐             ┌─────────────────────────────┐│
│   │  职责：                      │             │  职责：                      ││
│   │  - 封装为 LangGraph 节点     │             │  - LLM 循环执行             ││
│   │  - 配置接口 (Context)       │             │  - 工具调用                 ││
│   │  - 节点标识和日志           │             │  - 技能执行                 ││
│   │  - 错误处理                 │             │  - 约束检查                 ││
│   │                             │             │  - 消息构建                 ││
│   └─────────────────────────────┘             └─────────────────────────────┘│
│                │                                          ▲                 │
│                │          调用 engine.run()               │                 │
│                └──────────────────────────────────────────┘                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 调用流程

```python
# agent_loop_node 内部调用流程

async def node_fn(state):
    # 1. 节点层：添加标识
    state["_agent_loop_node"] = node_name

    # 2. 节点层：调用引擎
    result = await engine.run(state, context)
    #                      │       │
    #                      │       └── NodeExecutionContext
    #                      │
    #                      └── AgentLoopEngine

    # 3. 引擎层：
    #    - 获取工具 (应用 tool_whitelist)
    #    - 构建提示词 (应用 system_prompt_append)
    #    - 构建用户消息 (应用 inject_evidence)
    #    - 执行循环 (应用 max_iterations)
    #    - 执行工具调用 (应用 skill_whitelist)

    # 4. 节点层：返回结果
    return result
```

---

## 6. 配置示例

### 6.1 知识问答场景

```python
knowledge_answer_context = NodeExecutionContext(
    # 工具约束
    tool_whitelist=[
        "skill_manager",      # 技能管理（入口）
        "code_retriever",     # 代码检索
        "wiki_search",        # Wiki 搜索
    ],
    skill_whitelist=[
        "query_code",         # 查询代码
        "query_wiki",         # 查询 Wiki
    ],

    # 提示词
    system_prompt_append="""你是企业知识问答助手，必须严格基于提供的证据回答。

规则：
1. 如果证据不足，明确说明"根据现有证据无法回答"
2. 引用证据时标注来源（如：[代码] path/to/file.py）
3. 优先使用代码证据，其次使用文档证据""",

    # 循环控制
    max_iterations=3,

    # 上下文注入
    inject_evidence=True,
    inject_history=False,

    # 输出
    output_format="markdown",
)
```

### 6.2 问题分析场景

```python
issue_analysis_context = NodeExecutionContext(
    # 工具约束
    tool_whitelist=[
        "skill_manager",
        "code_retriever",
        "metrics_query",      # 指标查询
        "log_query",          # 日志查询
    ],
    skill_whitelist=[
        "analyze_error",      # 错误分析
        "query_metrics",      # 查询指标
        "search_logs",        # 搜索日志
    ],

    # 提示词
    system_prompt_append="""你是问题分析专家，请分析问题的根本原因并提供解决建议。

分析步骤：
1. 问题理解：明确问题的现象和影响范围
2. 信息收集：基于相关代码和日志分析
3. 根因分析：定位问题的根本原因
4. 解决方案：提供具体的修复建议

输出格式：
## 问题概述
## 根因分析
## 解决方案
## 预防措施""",

    # 循环控制
    max_iterations=5,

    # 上下文注入
    inject_evidence=True,
    inject_history=True,     # 需要历史上下文

    # 输出
    output_format="markdown",
)
```

---

## 7. 测试用例

### 7.1 单元测试

```python
# tests/workflow/common_nodes/test_agent_loop_node.py

import pytest
from unittest.mock import AsyncMock, MagicMock

from workflow.common_nodes.agent_loop_node import (
    create_agent_loop_node,
    NodeExecutionContext
)


class TestNodeExecutionContext:

    def test_default_values(self):
        """测试默认值"""
        ctx = NodeExecutionContext()

        assert ctx.tool_whitelist == []
        assert ctx.skill_whitelist == []
        assert ctx.system_prompt_append is None
        assert ctx.max_iterations == 5
        assert ctx.inject_evidence is True
        assert ctx.output_format == "markdown"

    def test_custom_values(self):
        """测试自定义值"""
        ctx = NodeExecutionContext(
            tool_whitelist=["skill_manager"],
            max_iterations=3,
            system_prompt_append="测试提示词",
        )

        assert ctx.tool_whitelist == ["skill_manager"]
        assert ctx.max_iterations == 3
        assert ctx.system_prompt_append == "测试提示词"

    def test_to_dict(self):
        """测试转换为字典"""
        ctx = NodeExecutionContext(max_iterations=3)
        d = ctx.to_dict()

        assert d["max_iterations"] == 3
        assert "tool_whitelist" in d


class TestCreateAgentLoopNode:

    @pytest.fixture
    def mock_engine(self):
        engine = MagicMock()
        engine.run = AsyncMock(return_value={
            "answer": "测试答案",
            "agent_loop_iterations": 1,
            "agent_loop_completed": True,
        })
        return engine

    async def test_create_node(self, mock_engine):
        """测试创建节点"""
        node = create_agent_loop_node(
            engine=mock_engine,
            node_name="test_node",
            context=NodeExecutionContext(),
        )

        assert callable(node)
        assert "test_node" in node.__name__

    async def test_node_execution(self, mock_engine):
        """测试节点执行"""
        node = create_agent_loop_node(
            engine=mock_engine,
            node_name="test_node",
            context=NodeExecutionContext(max_iterations=3),
        )

        state = {"user_query": "测试问题"}
        result = await node(state)

        assert "answer" in result
        assert result["agent_loop_completed"] is True
        mock_engine.run.assert_called_once()

    async def test_engine_none_raises(self):
        """测试 engine 为 None 时抛出异常"""
        with pytest.raises(ValueError):
            create_agent_loop_node(
                engine=None,
                node_name="test_node",
                context=NodeExecutionContext(),
            )
```

---

## 8. 总结

### 8.1 核心设计要点

| 要点 | 说明 |
|------|------|
| **位置** | `workflow/common_nodes/agent_loop_node/`（公共节点） |
| **职责** | 封装 AgentLoopEngine 为 LangGraph 节点 |
| **配置** | 通过 `NodeExecutionContext` 配置行为 |
| **约束** | 继承 Subgraph 级别，可进一步收窄 |
| **复用** | 可被多个 Subgraph 复用 |

### 8.2 关键 API

| API | 用途 |
|-----|------|
| `create_agent_loop_node()` | 创建 Agent Loop 节点 |
| `NodeExecutionContext` | 节点执行上下文配置 |
| `engine.run()` | 核心引擎执行（被节点调用） |

### 8.3 使用流程

```
1. 初始化 AgentLoopEngine
2. 创建 NodeExecutionContext（配置约束、提示词等）
3. 调用 create_agent_loop_node() 创建节点
4. 将节点添加到 LangGraph
```
