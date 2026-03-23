# -*- coding: utf-8 -*-
"""Wiki retrieval orchestration for the deep-agent runtime."""
from __future__ import annotations

"""Wiki 检索节点（支持动态 TopK 与低置信重试）。"""

import logging
from typing import Any, Callable

from retrievers.orchestration.retry import dedupe_normalized_queries, run_with_retry
from common.func_utils import env_float, env_int

logger = logging.getLogger(__name__)


def _grade_wiki_hits(items: list[dict[str, Any]]) -> str:
    """评估 Wiki 检索结果质量

    Args:
        items: 检索结果列表

    Returns:
        质量评级：insufficient/medium/high
    """
    if not items:
        return "insufficient"
    top1_score = float(items[0].get("score", 0.0))
    if top1_score >= 6.0 and len(items) >= 2:
        return "high"
    if top1_score >= 3.0:
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
        f"{module_name} 核心链路 关键指标",
        f"{module_name} 业务口径 常见问题",
        f"{user_query} 指标 口径 流程",
    ]
    return dedupe_normalized_queries(queries)


def _log_retrieval_details(
    trace_id: str,
    phase: str,
    *,
    input_queries: list[str],
    top_k: int,
    hits: list[dict[str, Any]],
    grade: str,
) -> None:
    """记录检索的详细日志

    Args:
        trace_id: 追踪 ID
        phase: 阶段（input/output）
        input_queries: 输入查询列表
        top_k: TopK 配置
        hits: 检索结果
        grade: 质量评级
    """
    if phase == "input":
        logger.info(
            f"[retrieve_wiki] INPUT | trace_id={trace_id} | "
            f"queries={input_queries} | top_k={top_k}"
        )
    else:
        # 输出阶段：记录每个 hit 的详细信息
        hit_summaries = []
        for i, hit in enumerate(hits[:5], 1):  # 只记录前 5 个
            path = hit.get("path", "")
            section = hit.get("section", "")
            score = hit.get("score", 0.0)
            content = str(hit.get("content", ""))[:100].replace("\n", " ")
            hit_summaries.append(
                f"[{i}] score={score:.4f} | path={path} | section={section} | content={content}..."
            )

        logger.info(
            f"[retrieve_wiki] OUTPUT | trace_id={trace_id} | "
            f"hits={len(hits)} | grade={grade} | "
            f"scores={[round(h.get('score', 0), 4) for h in hits[:5]]}"
        )

        # 每个 hit 单独一行详细日志
        for summary in hit_summaries:
            logger.info(f"[retrieve_wiki] HIT | trace_id={trace_id} | {summary}")


def execute_wiki_retrieval(
    retriever: Any,
    state: dict[str, Any],
    trace_fn: Callable[[dict[str, Any], str, str], list[dict[str, str]]] | None = None,
) -> dict[str, Any]:
    """执行 Wiki 检索（解耦版本）

    直接接收 retriever 参数，不依赖 service 对象。
    适用于子图内部调用场景。

    Args:
        retriever: Wiki 检索器实例
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
            return trace_fn(state, "retrieve_wiki", summary)
        return [{"node": "retrieve_wiki", "summary": summary}]

    if not retrieval_plan.get("enable_wiki", True):
        logger.info(f"[retrieve_wiki] DISABLED | trace_id={trace_id} | reason=disabled_by_plan")
        return {
            "wiki_hits": [],
            "wiki_retrieval_grade": "disabled",
            "wiki_retrieval_profile": {
                "latency_ms": 0.0,
                "hits": 0,
                "top_k": 0,
                "strategy": retrieval_plan.get("strategy", "unknown"),
                "retried": False,
            },
            "node_trace": build_trace("disabled_by_plan"),
        }

    top_k = int(retrieval_plan.get("wiki_top_k", 4))
    retry_multiplier = env_int("WORKFLOW_WIKI_RETRY_TOPK_MULTIPLIER", 2, minimum=1)
    retry_max_top_k = env_int("WORKFLOW_WIKI_RETRY_MAX_TOPK", 14, minimum=1)
    retry_min_top1 = env_float("WORKFLOW_WIKI_RETRY_MIN_TOP1", 3.0, minimum=0.0)

    base_queries = list(state.get("retrieval_queries", []))

    # 记录输入日志
    _log_retrieval_details(
        trace_id=trace_id,
        phase="input",
        input_queries=base_queries,
        top_k=top_k,
        hits=[],
        grade="",
    )

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
        grade=_grade_wiki_hits,
        should_retry=lambda first_grade, first_top1: first_grade in {"insufficient", "low"} or first_top1 < retry_min_top1,
    )

    # 记录输出日志
    _log_retrieval_details(
        trace_id=trace_id,
        phase="output",
        input_queries=base_queries,
        top_k=result.final_top_k,
        hits=result.final_items,
        grade=result.final_grade,
    )

    # 记录重试信息
    if result.retried:
        logger.info(
            f"[retrieve_wiki] RETRY | trace_id={trace_id} | "
            f"initial_top_k={result.initial_top_k} -> final_top_k={result.final_top_k} | "
            f"first_grade={result.first_grade} -> final_grade={result.final_grade} | "
            f"first_top1={result.first_top1:.4f}"
        )

    profile = dict(retriever.last_search_profile)
    profile.update(
        {
            "latency_ms": result.latency_ms,
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
        "wiki_hits": result.final_items,
        "wiki_retrieval_grade": result.final_grade,
        "wiki_retrieval_profile": profile,
        "node_trace": build_trace(
            f"hits={len(result.final_items)},grade={result.final_grade},"
            f"retried={result.retried},latency_ms={result.latency_ms}"
        ),
    }
