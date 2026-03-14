# -*- coding: utf-8 -*-
"""
该模块实现工作流节点`retrieve_code` 的处理逻辑，负责读取状态并输出增量结果。
"""
from __future__ import annotations

"""代码检索节点（支持动态 TopK 与低置信重试）。"""

from typing import Any

from workflow.nodes.retrieval_retry import dedupe_normalized_queries, run_with_retry
from workflow.common.func_utils import env_int


def _grade_code_hits(items: list[dict[str, Any]], *, high_threshold: float, medium_threshold: float) -> str:
    """
    内部辅助函数，负责`grade code hits` 相关处理。
    
    参数:
        items: 列表参数，用于承载批量输入数据。
    
    返回:
        返回类型为 `str` 的处理结果。
    """
    if not items:
        return "insufficient"
    top1_score = float(items[0].get("score", 0.0))
    if top1_score >= high_threshold and len(items) >= 2:
        return "high"
    if top1_score >= medium_threshold:
        return "medium"
    return "low"


def _build_retry_queries(state: dict[str, Any]) -> list[str]:
    """
    构建当前步骤所需的数据结构或文本内容。
    
    参数:
        state: 工作流状态字典，包含会话上下文与中间结果。
    
    返回:
        返回类型为 `list[str]` 的处理结果。
    """
    module_name = str(state.get("module_name", "")).strip()
    user_query = str(state.get("user_query", "")).strip()
    queries = [
        *state.get("retrieval_queries", []),
        f"{module_name} 实现 入口 函数",
        f"{module_name} 关键参数 校验",
        f"{user_query} 代码位置 文件路径",
    ]
    return dedupe_normalized_queries(queries)


def run(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """
    执行`retrieve_code` 节点主流程，基于输入状态计算并返回状态增量。
    
    参数:
        service: 工作流服务对象，提供检索、路由、日志与配置能力。
        state: 工作流状态字典，包含会话上下文与中间结果。
    
    返回:
        返回类型为 `dict[str, Any]` 的处理结果。
    """
    retrieval_plan = state.get("retrieval_plan", {})
    if not retrieval_plan.get("enable_code", True):
        return {
            "code_hits": [],
            "code_retrieval_grade": "disabled",
            "code_retrieval_profile": {
                "latency_ms": 0.0,
                "hits": 0,
                "top_k": 0,
                "strategy": retrieval_plan.get("strategy", "unknown"),
                "retried": False,
            },
            "node_trace": service._trace(state, "retrieve_code", "disabled_by_plan"),
        }

    config = service._code_retriever.runtime_config
    top_k = int(retrieval_plan.get("code_top_k", 4))
    retry_multiplier = env_int("WORKFLOW_CODE_RETRY_TOPK_MULTIPLIER", 2, minimum=1)
    retry_max_top_k = env_int("WORKFLOW_CODE_RETRY_MAX_TOPK", 14, minimum=1)

    result = run_with_retry(
        top_k=top_k,
        retry_multiplier=retry_multiplier,
        retry_max_top_k=retry_max_top_k,
        base_queries=list(state.get("retrieval_queries", [])),
        retry_queries=_build_retry_queries(state),
        search=lambda current_top_k, queries: service._code_retriever.search(
            user_query=state["user_query"],
            retrieval_queries=queries,
            module_name=state["module_name"],
            top_k=current_top_k,
        ),
        grade=lambda items: _grade_code_hits(
            items,
            high_threshold=float(config.grade_high_top1_threshold),
            medium_threshold=float(config.grade_medium_top1_threshold),
        ),
        should_retry=lambda first_grade, _: first_grade in {"insufficient", "low"},
    )
    profile = dict(service._code_retriever.last_search_profile)
    profile.update(
        {
            "hits": len(result.final_items),
            "top_k": result.final_top_k,
            "initial_top_k": result.initial_top_k,
            "strategy": retrieval_plan.get("strategy", "unknown"),
            "retried": result.retried,
            "first_grade": result.first_grade,
            "final_grade": result.final_grade,
            "first_top1": round(result.first_top1, 4),
        }
    )

    return {
        "code_hits": result.final_items,
        "code_retrieval_grade": result.final_grade,
        "code_retrieval_profile": profile,
        "node_trace": service._trace(
            state,
            "retrieve_code",
            (
                f"hits={len(result.final_items)},grade={result.final_grade},"
                f"retried={result.retried},latency_ms={profile.get('latency_ms', 0.0)}"
            ),
        ),
    }
