# -*- coding: utf-8 -*-
"""
该模块实现工作流节点`retrieve_code` 的处理逻辑，负责读取状态并输出增量结果。

提供两种接口：
- run_with_retriever: 解耦版本，直接接收 retriever 参数
- run: 兼容接口，从 service 获取 retriever
"""
from __future__ import annotations

"""代码检索节点（支持动态 TopK 与低置信重试）。"""

import logging
from typing import Any, Callable

from workflow.nodes.retrieval_retry import dedupe_normalized_queries, run_with_retry
from workflow.common.func_utils import env_int

logger = logging.getLogger(__name__)


def _grade_code_hits(items: list[dict[str, Any]], *, high_threshold: float, medium_threshold: float) -> str:
    """评估代码检索结果质量

    Args:
        items: 检索结果列表
        high_threshold: 高质量阈值
        medium_threshold: 中等质量阈值

    Returns:
        质量评级：insufficient/medium/high
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
    """构建重试查询

    Args:
        state: 工作流状态

    Returns:
        去重后的查询列表
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


def _log_retrieval_input(trace_id: str, queries: list[str], top_k: int) -> None:
    """记录检索输入的日志"""
    logger.info(
        f"[retrieve_code] INPUT | trace_id={trace_id} | "
        f"queries={queries} | top_k={top_k}"
    )


def _log_retrieval_output(trace_id: str, hits: list[dict[str, Any]], grade: str) -> None:
    """记录检索输出的日志"""
    # 输出阶段：记录分数汇总
    logger.info(
        f"[retrieve_code] OUTPUT | trace_id={trace_id} | "
        f"hits={len(hits)} | grade={grade} | "
        f"scores={[round(h.get('score', 0), 4) for h in hits[:5]]}"
    )

    # 每个 hit 单独一行详细日志
    for i, hit in enumerate(hits[:5], 1):  # 只记录前 5 个
        path = hit.get("path", "")
        symbol = hit.get("symbol_name", "")
        line_range = hit.get("line_range", "")
        score = hit.get("score", 0.0)
        content = str(hit.get("content", ""))[:80].replace("\n", " ")
        logger.info(
            f"[retrieve_code] HIT | trace_id={trace_id} | "
            f"[{i}] score={score:.4f} | path={path} | symbol={symbol} | lines={line_range} | code={content}..."
        )


def run_with_retriever(
    retriever: Any,
    state: dict[str, Any],
    trace_fn: Callable[[dict[str, Any], str, str], list[dict[str, str]]] | None = None,
) -> dict[str, Any]:
    """执行代码检索（解耦版本）

    直接接收 retriever 参数，不依赖 service 对象。
    适用于子图内部调用场景。

    Args:
        retriever: 代码检索器实例
        state: 工作流状态字典
        trace_fn: 追踪函数（可选），签名为 (state, node_name, summary) -> node_trace

    Returns:
        状态增量字典
    """
    trace_id = state.get("trace_id", "")
    retrieval_plan = state.get("retrieval_plan", {})

    # 构建 node_trace 的辅助函数（只返回新增条目，由 merge_lists reducer 合并）
    def build_trace(summary: str) -> list[dict[str, str]]:
        if trace_fn:
            return trace_fn(state, "retrieve_code", summary)
        return [{"node": "retrieve_code", "summary": summary}]

    if not retrieval_plan.get("enable_code", True):
        logger.info(f"[retrieve_code] DISABLED | trace_id={trace_id} | reason=disabled_by_plan")
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
            "node_trace": build_trace("disabled_by_plan"),
        }

    config = retriever.runtime_config
    top_k = int(retrieval_plan.get("code_top_k", 4))
    retry_multiplier = env_int("WORKFLOW_CODE_RETRY_TOPK_MULTIPLIER", 2, minimum=1)
    retry_max_top_k = env_int("WORKFLOW_CODE_RETRY_MAX_TOPK", 14, minimum=1)

    base_queries = list(state.get("retrieval_queries", []))

    # 记录输入日志
    _log_retrieval_input(trace_id, base_queries, top_k)

    result = run_with_retry(
        top_k=top_k,
        retry_multiplier=retry_multiplier,
        retry_max_top_k=retry_max_top_k,
        base_queries=base_queries,
        retry_queries=_build_retry_queries(state),
        search=lambda current_top_k, queries: retriever.search(
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

    # 记录输出日志
    _log_retrieval_output(trace_id, result.final_items, result.final_grade)

    # 记录重试信息
    if result.retried:
        logger.info(
            f"[retrieve_code] RETRY | trace_id={trace_id} | "
            f"initial_top_k={result.initial_top_k} -> final_top_k={result.final_top_k} | "
            f"first_grade={result.first_grade} -> final_grade={result.final_grade} | "
            f"first_top1={result.first_top1:.4f}"
        )

    profile = dict(retriever.last_search_profile)
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
        "node_trace": build_trace(
            f"hits={len(result.final_items)},grade={result.final_grade},"
            f"retried={result.retried},latency_ms={profile.get('latency_ms', 0.0)}"
        ),
    }


def run(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """执行代码检索（兼容接口）

    从全局单例获取 retriever，不再依赖 service 传递。

    Args:
        service: 工作流服务对象，需要提供 _trace 方法
        state: 工作流状态字典

    Returns:
        状态增量字典
    """
    # 从全局单例获取 retriever
    from workflow.nodes.retrieval_flow.retrieve_code.code_retriever import get_code_retriever

    return run_with_retriever(
        retriever=get_code_retriever(),
        state=state,
        trace_fn=lambda s, n, d: service._trace(s, n, d),
    )
