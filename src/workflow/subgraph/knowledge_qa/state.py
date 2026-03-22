# -*- coding: utf-8 -*-
"""KnowledgeQA 子图状态定义

知识问答子图的独立状态类型。

## 设计要点

### 主图与子图的状态关系（自动关联模式）

- **主图 state（WorkflowState）是全局统一上下文**：
  - 在整个请求期间持续存在
  - 包含会话标识、用户输入、路由结果、最终输出

- **子图 state（KnowledgeQAState）是子图独立上下文**：
  - 仅在子图运行期间存在
  - 包含从主图自动接收的字段 + 子图内部字段 + 输出字段

### 自动字段关联

LangGraph 自动关联父子图中相同名称的字段：
- **输入阶段**：主图调用子图时，同名字段自动传递给子图
- **输出阶段**：子图返回时，同名字段自动合并回主图

本子图从主图自动接收的字段：
- trace_id, user_query, module_name, module_hint, related_modules

### 子图节点的状态访问

子图节点直接从 state 读取字段（无需通过 _main_state）：

```python
def _run_query_rewriter(self, state: KnowledgeQAState) -> dict[str, Any]:
    # 直接从子图 state 读取（已从主图自动接收）
    user_query = state.get("user_query", "")
    module_name = state.get("module_name", "")

    return {
        "retrieval_queries": [user_query],
        "retrieval_plan": {...},
    }
```

### 子图运行完成后的状态处理

子图运行完成后：
1. 输出字段（answer, citations, analysis, node_trace）自动合并到主图 state
2. 子图内部字段（wiki_hits, code_hits 等）不会传回主图
3. 主图进入 finalize_response 节点，使用主图 state 生成最终响应
"""
from __future__ import annotations

from typing import Annotated, Any, TypedDict

from workflow.subgraph.base import merge_lists, merge_dicts


class KnowledgeQAState(TypedDict, total=False):
    """KnowledgeQA 子图完整状态

    子图状态包含三类字段：
    1. 从主图自动接收的字段：LangGraph 自动关联同名字段
    2. 子图内部字段：检索中间产物，仅在子图内部流转
    3. 输出字段：自动合并回主图的字段

    说明：
    - 使用 total=False 允许所有字段可选
    - LangGraph 会根据节点返回值自动合并状态
    - 节点返回的同名字段会自动合并回主图
    """

    # === 从主图自动接收的字段（只读） ===
    trace_id: str
    user_query: str  # 用户问题
    module_name: str  # 当前主模块名
    module_hint: str  # 模块提示
    related_modules: Annotated[list[dict[str, str]], merge_lists]  # 相关模块列表

    # === 子图内部字段（检索中间产物） ===
    retrieval_plan: Annotated[dict[str, Any], merge_dicts]
    retrieval_queries: Annotated[list[str], merge_lists]
    query_rewrite_mode: str

    wiki_hits: Annotated[list[dict[str, Any]], merge_lists]
    wiki_retrieval_grade: str
    wiki_retrieval_profile: dict[str, Any]

    code_hits: Annotated[list[dict[str, Any]], merge_lists]
    code_retrieval_grade: str
    code_retrieval_profile: dict[str, Any]

    case_hits: Annotated[list[dict[str, Any]], merge_lists]
    case_retrieval_grade: str
    case_retrieval_profile: dict[str, Any]

    evidence_fusion_profile: dict[str, Any]

    # === 输出字段（自动合并回主图） ===
    response_kind: str
    status: str
    answer: str
    analysis: dict[str, Any] | None
    citations: Annotated[list[dict[str, Any]], merge_lists]
    node_trace: Annotated[list[dict[str, str]], merge_lists]
    debug_info: dict[str, Any]


# ============================================================================
# 辅助函数
# ============================================================================

def build_debug_info(state: dict[str, Any]) -> dict[str, Any]:
    """构建调试信息

    从子图状态中提取调试相关字段，封装到 debug_info 中。
    使用自动关联模式，直接从子图 state 读取字段（已从主图自动接收）。

    Args:
        state: 子图状态

    Returns:
        调试信息字典
    """
    return {
        # 从主图自动接收的字段
        "module_name": str(state.get("module_name", "") or ""),
        "related_modules": list(state.get("related_modules", []) or []),
        # 子图内部的检索信息
        "retrieval_queries": list(state.get("retrieval_queries", []) or []),
        "retrieval_plan": dict(state.get("retrieval_plan", {}) or {}),
        "query_rewrite_mode": str(state.get("query_rewrite_mode", "") or ""),
        "wiki_retrieval_grade": str(state.get("wiki_retrieval_grade", "unknown") or "unknown"),
        "code_retrieval_grade": str(state.get("code_retrieval_grade", "unknown") or "unknown"),
        "case_retrieval_grade": str(state.get("case_retrieval_grade", "unknown") or "unknown"),
        "wiki_retrieval_profile": dict(state.get("wiki_retrieval_profile", {}) or {}),
        "code_retrieval_profile": dict(state.get("code_retrieval_profile", {}) or {}),
        "case_retrieval_profile": dict(state.get("case_retrieval_profile", {}) or {}),
        "evidence_fusion_profile": dict(state.get("evidence_fusion_profile", {}) or {}),
        "wiki_hit_count": len(state.get("wiki_hits", []) or []),
        "code_hit_count": len(state.get("code_hits", []) or []),
        "case_hit_count": len(state.get("case_hits", []) or []),
    }
