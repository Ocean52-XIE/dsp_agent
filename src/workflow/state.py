# -*- coding: utf-8 -*-
"""主工作流状态定义

主图状态（WorkflowState）是全局统一上下文，贯穿整个请求生命周期。

## 设计要点

### 1. 状态分层架构

```
┌─────────────────────────────────────────────────────────────┐
│                 WorkflowState (主图全局上下文)                │
│                                                             │
│  职责：                                                      │
│  - 会话标识和用户输入                                         │
│  - 路由决策结果                                              │
│  - 最终输出（answer, citations, analysis）                   │
│  - 流程追踪（node_trace）                                    │
│                                                             │
│  生命周期：整个请求期间                                        │
└─────────────────────────────────────────────────────────────┘
                              │
            ┌─────────────────┴─────────────────┐
            │                                   │
            ▼                                   ▼
┌─────────────────────────┐    ┌─────────────────────────┐
│  KnowledgeQAState       │    │  IssueAnalysisState     │
│  (knowledge_qa 子图)    │    │  (issue_analysis 子图)  │
├─────────────────────────┤    ├─────────────────────────┤
│  子图独立上下文：         │    │  子图独立上下文：         │
│  - retrieval_plan       │    │  - retrieval_plan       │
│  - retrieval_queries    │    │  - retrieval_queries    │
│  - wiki_hits            │    │  - wiki_hits            │
│  - code_hits            │    │  - code_hits            │
│  - *_grade, *_profile   │    │  - *_grade, *_profile   │
│                         │    │                         │
│  生命周期：子图运行期间   │    │  生命周期：子图运行期间   │
└─────────────────────────┘    └─────────────────────────┘
```

### 2. 主图与子图的状态关系

- **主图 state 是全局统一上下文**：
  - 在整个请求期间持续存在
  - 所有节点都可以读写
  - 包含最终输出所需的全部字段

- **子图 state 是子图独立上下文**：
  - 仅在子图运行期间存在
  - 子图运行完成后不再需要
  - 存储子图内部中间产物（检索结果、评级等）

- **子图与主图的交互**：
  - 子图运行时可以访问主图 state（只读）
  - 子图节点可以修改子图 state（内部流转）
  - 子图节点也可以直接修改主图 state（写回最终结果）
  - 子图运行完成后，所有必需字段已写回主图 state

### 3. 字段分类

| 分类 | 字段 | 说明 |
|------|------|------|
| 会话标识 | trace_id, session_id | 请求唯一标识 |
| 用户输入 | user_query, original_user_query, history | 原始输入 |
| 路由结果 | route, status, response_kind, domain_relevance | 路由决策 |
| 路由上下文 | module_name, module_hint, related_modules | 传递给子图 |
| 子图输出 | answer, analysis, citations | 由子图写回 |
| 流程追踪 | node_trace | 全流程追踪 |
| 调试信息 | debug_info | 可选，调试模式 |
| 最终输出 | assistant_message | API 响应 |
"""
from __future__ import annotations

from typing import Annotated, Any, TypedDict

from workflow.subgraph.base import merge_lists


class WorkflowState(TypedDict, total=False):
    """主工作流状态（全局统一上下文）

    设计原则：
    1. 主图只关心路由决策和最终结果
    2. 检索中间产物由子图内部管理，不暴露给主图
    3. 调试信息封装在 debug_info 字段中

    状态流转：
    - 入口：trace_id, session_id, user_query, history
    - load_context：module_name, module_hint, related_modules
    - intent_routing：route, domain_relevance
    - 子图：answer, analysis, citations, debug_info
    - finalize_response：assistant_message

    使用 Annotated 类型提示配置 reducer，确保多节点返回值正确合并。
    """

    # === 会话标识 ===
    trace_id: str
    session_id: str

    # === 用户输入 ===
    user_query: str
    original_user_query: str
    history: Annotated[list[dict[str, Any]], merge_lists]

    # === 路由结果 ===
    route: str                              # 路由目标: knowledge_qa | issue_analysis | code_generation | out_of_scope
    status: str                             # 执行状态
    response_kind: str                      # 响应类型
    domain_relevance: float                 # 领域相关性分数

    # === 路由上下文（传递给子图） ===
    module_name: str                        # 当前主模块名
    module_hint: str                        # 模块提示
    related_modules: Annotated[list[dict[str, str]], merge_lists]  # 相关模块列表

    # === 子图输出（由子图返回） ===
    answer: str                             # 最终答案
    analysis: dict[str, Any] | None         # 分析元信息
    citations: Annotated[list[dict[str, Any]], merge_lists]  # 证据列表（已融合）

    # === 流程追踪 ===
    node_trace: Annotated[list[dict[str, str]], merge_lists]  # 节点追踪

    # === 调试信息（可选，仅在 debug 模式填充） ===
    debug_info: dict[str, Any]              # 封装所有中间调试数据

    # === 最终输出 ===
    assistant_message: dict[str, Any]       # 最终响应消息


# ============================================================================
# 辅助函数
# ============================================================================

def create_initial_state(
    *,
    trace_id: str,
    session_id: str,
    user_query: str,
    history: list[dict[str, Any]] | None = None,
) -> WorkflowState:
    """创建初始工作流状态

    Args:
        trace_id: 请求追踪 ID
        session_id: 会话 ID
        user_query: 用户查询
        history: 对话历史（可选）

    Returns:
        初始化的工作流状态
    """
    return {
        "trace_id": trace_id,
        "session_id": session_id,
        "user_query": user_query.strip(),
        "history": history or [],
        "node_trace": [],
    }
