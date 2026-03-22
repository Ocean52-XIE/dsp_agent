# -*- coding: utf-8 -*-
"""
该模块实现工作流节点 `merge_evidence` 的处理逻辑，负责读取状态并输出增量结果。

核心功能：
1. 多源证据融合：将 wiki、code、case 三种来源的检索结果合并
2. 分数归一化：对每个来源的原始分数进行 min-max 归一化，确保跨来源可比性
3. 配额选择：按 source 配额和融合分数选出最终证据

修复记录：
- 2026-03-21: 添加分数归一化逻辑，解决 code（BM25）和 wiki（embedding）
  分数量级差异过大导致 wiki 证据被淹没的问题
"""
from __future__ import annotations

"""多源证据融合节点。"""

import logging
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)


GRADE_BIAS = {
    "high": 0.35,
    "medium": 0.15,
    "low": 0.0,
    "insufficient": -0.1,
    "disabled": -0.2,
}

# 按检索意图附加的 source 偏置（P0）。
# 说明：
# - code_location：函数/文件定位类问题，必须明显拉高 code 证据优先级；
# - wiki_first：业务说明类问题，适度提升 wiki；
# - 默认不加偏置。
INTENT_SOURCE_BIAS = {
    "code_location": {"code": 0.25, "wiki": -0.1, "case": 0.0},
    "wiki_first": {"wiki": 0.15, "code": -0.05, "case": 0.0},
}


def _as_source_key(source_type: str) -> str:
    """
    内部辅助函数，负责将 source_type 转换为标准化的来源键。

    参数:
        source_type: 原始来源类型字符串

    返回:
        标准化的来源键（wiki/code/case/unknown）
    """
    normalized = source_type.strip().lower()
    if normalized.startswith("wiki"):
        return "wiki"
    if normalized.startswith("code"):
        return "code"
    if normalized.startswith("case"):
        return "case"
    return normalized or "unknown"


def _normalize_scores_within_source(hits: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    """处理单个来源的分数，确保跨来源可比性

    优化说明（2026-03-21）：
        各检索器已统一使用 RRF 归一化策略，输出分数范围 [0, 1]。
        - Wiki 检索器：bm25_score/embedding_score/lexical_score 都使用 RRF(k=60)
        - Code 检索器：bm25_score/tfidf_score/embedding_score 都使用 RRF(k=60)

        因此融合层不再需要额外的归一化处理，直接使用检索器输出的分数。

    兼容性处理：
        如果分数已经在 [0, 1] 范围内，直接使用；
        如果分数超出范围（旧版检索器），仍执行 min-max 归一化作为兜底。

    Args:
        hits: 单个来源的命中列表（会被原地修改，添加 normalized_score 字段）
        source: 来源标识（用于日志）

    Returns:
        添加了 normalized_score 字段的命中列表
    """
    if not hits:
        return hits

    # 提取所有分数
    scores = [float(hit.get("score", 0.0)) for hit in hits]
    max_score = max(scores) if scores else 1.0
    min_score = min(scores) if scores else 0.0

    # 检查分数是否已经在 [0, 1] 范围内（RRF 归一化后的特征）
    # 允许小的浮点误差（0.001）
    already_normalized = min_score >= -0.001 and max_score <= 1.001

    if already_normalized:
        # 分数已经在 [0, 1] 范围内，直接使用
        for hit, score in zip(hits, scores):
            # 确保分数在 [0, 1] 范围内（处理浮点误差）
            hit["normalized_score"] = max(0.0, min(1.0, score))
        logger.debug(
            f"[merge_evidence] 分数已归一化（RRF） | source={source} | "
            f"hits={len(hits)} | range=[{min_score:.4f}, {max_score:.4f}] -> direct use"
        )
        return hits

    # 分数超出 [0, 1] 范围，执行 min-max 归一化（兜底逻辑）
    score_range = max_score - min_score
    if score_range == 0:
        # 所有分数相同，归一化为 0.5
        for hit in hits:
            hit["normalized_score"] = 0.5
        logger.debug(
            f"[merge_evidence] 分数归一化（兜底） | source={source} | "
            f"hits={len(hits)} | all_scores_same={max_score:.4f} -> normalized=0.5"
        )
    else:
        # 标准 min-max 归一化
        for hit, score in zip(hits, scores):
            hit["normalized_score"] = (score - min_score) / score_range
        logger.info(
            f"[merge_evidence] 分数归一化（兜底） | source={source} | "
            f"hits={len(hits)} | raw_range=[{min_score:.4f}, {max_score:.4f}] -> normalized=[0.0, 1.0]"
        )

    return hits


def _normalize_retrieval_plan(state: dict[str, Any]) -> dict[str, Any]:
    """
    归一化检索计划，确保所有参数都有默认值。

    参数:
        state: 工作流状态字典，包含 retrieval_plan

    返回:
        归一化后的检索计划字典
    """
    raw_plan = state.get("retrieval_plan", {})
    final_top_k = max(int(raw_plan.get("final_top_k", 5)), 1)

    raw_weights = raw_plan.get("source_weights", {})
    source_weights = {
        "wiki": max(float(raw_weights.get("wiki", 1.0)), 0.0),
        "code": max(float(raw_weights.get("code", 1.0)), 0.0),
        "case": max(float(raw_weights.get("case", 0.6)), 0.0),
    }

    raw_max_per_source = raw_plan.get("max_per_source", {})
    max_per_source = {
        "wiki": max(int(raw_max_per_source.get("wiki", final_top_k)), 0),
        "code": max(int(raw_max_per_source.get("code", final_top_k)), 0),
        "case": max(int(raw_max_per_source.get("case", 1)), 0),
    }

    normalized_plan = {
        "strategy": str(raw_plan.get("strategy", "hybrid")),
        "final_top_k": final_top_k,
        "source_weights": source_weights,
        "max_per_source": max_per_source,
        "intent_profile": dict(raw_plan.get("intent_profile", {})),
    }
    return _apply_intent_overrides(state=state, plan=normalized_plan)


def _apply_intent_overrides(*, state: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    """
    根据意图配置调整检索计划参数。

    返回:
        调整后的检索计划
    """
    intent_profile = dict(plan.get("intent_profile", {}))
    is_code_location = bool(intent_profile.get("is_code_location"))
    is_wiki_intent = bool(intent_profile.get("is_wiki_intent"))

    # 兜底策略 1：函数定位问题强制 code-first，避免"路径/函数名被 wiki 解释性段落淹没"。
    if is_code_location:
        plan["strategy"] = "code_first"
        plan["source_weights"]["code"] = max(float(plan["source_weights"].get("code", 1.0)), 1.35)
        plan["source_weights"]["wiki"] = min(float(plan["source_weights"].get("wiki", 1.0)), 0.8)
        plan["max_per_source"]["code"] = max(int(plan["max_per_source"].get("code", 0)), 5)
        # wiki 仍保留 1~2 条兜底证据，避免完全丢失业务解释上下文。
        current_wiki_quota = max(int(plan["max_per_source"].get("wiki", 1)), 1)
        plan["max_per_source"]["wiki"] = min(current_wiki_quota, 2)
        return plan

    # 兜底策略 2：显式 wiki-first 时，避免 code 过度侵入。
    if plan.get("strategy") == "wiki_first" or (is_wiki_intent and not intent_profile.get("is_code_intent")):
        plan["source_weights"]["wiki"] = max(float(plan["source_weights"].get("wiki", 1.0)), 1.15)
        plan["source_weights"]["code"] = min(float(plan["source_weights"].get("code", 1.0)), 0.9)
        plan["max_per_source"]["wiki"] = max(int(plan["max_per_source"].get("wiki", 0)), 4)
    return plan


def _get_retrieval_grade(state: dict[str, Any], source: str) -> str:
    """
    获取指定来源的检索质量等级。

    参数:
        state: 工作流状态字典
        source: 来源标识（wiki/code/case）

    返回:
        检索质量等级（high/medium/low/insufficient/disabled/unknown）
    """
    if source == "wiki":
        return str(state.get("wiki_retrieval_grade", "unknown"))
    if source == "code":
        return str(state.get("code_retrieval_grade", "unknown"))
    if source == "case":
        return str(state.get("case_retrieval_grade", "unknown"))
    return "unknown"


def _build_candidate_items(state: dict[str, Any], plan: dict[str, Any]) -> list[dict[str, Any]]:
    """
    构建候选证据列表，并进行分数归一化和融合打分。

    核心逻辑：
    1. 从 state 中提取各来源的命中结果（wiki_hits, code_hits, case_hits）
    2. 对每个来源的原始分数进行 min-max 归一化（解决分数量级差异问题）
    3. 计算融合分数：fused_score = normalized_score * source_weight + rank_bonus + grade_bias + intent_bias
    4. 按融合分数降序排列

    参数:
        state: 工作流状态字典，包含会话上下文与中间结果。
        plan: 归一化后的检索计划。

    返回:
        候选证据列表，每个元素包含 fusion_score 和 fusion_debug 字段。
    """
    source_to_hits: dict[str, list[dict[str, Any]]] = {
        "wiki": list(state.get("wiki_hits", [])),
        "case": list(state.get("case_hits", [])),
        "code": list(state.get("code_hits", [])),
    }

    # ============ 关键修复：对每个来源的分数进行归一化 ============
    # 解决问题：code（BM25）分数可达 50+，wiki（embedding）分数通常 < 5
    # 归一化后每个来源的最高分为 1.0，确保跨来源可比性
    for source, hits in source_to_hits.items():
        _normalize_scores_within_source(hits, source)
    # ===========================================================

    candidates: list[dict[str, Any]] = []

    intent_profile = dict(plan.get("intent_profile", {}))
    is_code_location = bool(intent_profile.get("is_code_location"))
    is_wiki_first = plan.get("strategy") == "wiki_first"

    for source, hits in source_to_hits.items():
        source_weight = float(plan["source_weights"].get(source, 1.0))
        grade = _get_retrieval_grade(state, source)
        grade_bias = float(GRADE_BIAS.get(grade, 0.0))
        # P0：在融合打分时引入"意图偏置"。
        # 即便 source_weight 接近，也能保证函数定位问题里 code 证据更稳定进入头部。
        intent_bias = 0.0
        if is_code_location:
            intent_bias = float(INTENT_SOURCE_BIAS["code_location"].get(source, 0.0))
        elif is_wiki_first:
            intent_bias = float(INTENT_SOURCE_BIAS["wiki_first"].get(source, 0.0))

        for rank, hit in enumerate(hits, start=1):
            # 使用归一化后的分数（而非原始分数）计算融合分
            normalized_score = float(hit.get("normalized_score", 0.5))
            raw_score = float(hit.get("score", 0.0))  # 保留原始分数用于调试

            rank_bonus = 0.12 / rank
            # 融合公式：使用归一化分数，乘以 source_weight，再加上各项偏置
            fused_score = normalized_score * source_weight + rank_bonus + grade_bias + intent_bias

            item = dict(hit)
            item_source = _as_source_key(str(item.get("source_type", source)))
            item["source_type"] = item_source
            item["fusion_score"] = round(fused_score, 6)
            item["fusion_debug"] = {
                "raw_score": round(raw_score, 4),
                "normalized_score": round(normalized_score, 4),
                "source_weight": round(source_weight, 4),
                "rank_bonus": round(rank_bonus, 4),
                "grade": grade,
                "grade_bias": round(grade_bias, 4),
                "intent_bias": round(intent_bias, 4),
                "fused_score": round(fused_score, 4),
            }
            candidates.append(item)

    candidates.sort(
        key=lambda item: (
            float(item.get("fusion_score", 0.0)),
            float(item.get("normalized_score", 0.0)),  # 使用归一化分数作为次级排序
        ),
        reverse=True,
    )

    # 输出融合结果日志，方便调试
    if candidates:
        top_items = candidates[:6]  # 显示前 6 个
        fusion_summary = ", ".join(
            f"#{i+1}[{item['source_type']}:{item.get('fusion_score', 0):.3f}]"
            for i, item in enumerate(top_items)
        )
        logger.info(
            f"[merge_evidence] 融合排序完成 | total={len(candidates)} | top6: {fusion_summary}"
        )

    return candidates


def _dedup_key(item: dict[str, Any]) -> tuple[str, str, str]:
    """
    生成去重键，用于识别重复证据。

    参数:
        item: 证据项

    返回:
        (source_type, path, section) 三元组
    """
    return (
        str(item.get("source_type", "")),
        str(item.get("path", "")),
        str(item.get("section", "")),
    )


def _select_with_quota(candidates: list[dict[str, Any]], plan: dict[str, Any]) -> list[dict[str, Any]]:
    """
    按配额选择最终证据。

    选择策略：
    1. 第一轮：严格执行 source 配额，按融合分数顺序选择，保证多源覆盖
    2. 第二轮：如果仍不足 final_top_k，放开配额从 overflow 补齐

    参数:
        candidates: 已排序的候选证据列表
        plan: 检索计划，包含 final_top_k 和 max_per_source

    返回:
        最终选中的证据列表
    """
    final_top_k = int(plan["final_top_k"])
    max_per_source = plan["max_per_source"]

    selected: list[dict[str, Any]] = []
    overflow: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str]] = set()
    source_counts: dict[str, int] = defaultdict(int)

    # 第一轮：严格执行 source 配额，保证多源覆盖。
    for item in candidates:
        key = _dedup_key(item)
        if key in seen_keys:
            continue
        seen_keys.add(key)

        source = _as_source_key(str(item.get("source_type", "")))
        quota = int(max_per_source.get(source, final_top_k))
        if source_counts[source] >= quota:
            overflow.append(item)
            continue

        source_counts[source] += 1
        selected.append(item)
        if len(selected) >= final_top_k:
            return selected

    # 第二轮：如果仍不足 final_top_k，放开配额从 overflow 补齐。
    for item in overflow:
        if len(selected) >= final_top_k:
            break
        selected.append(item)
    return selected


def run(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """
    执行 `merge_evidence` 节点主流程，基于输入状态计算并返回状态增量。

    处理流程：
    1. 归一化检索计划，确保加权融合参数完整且可用
    2. 构建候选证据列表（含分数归一化和融合打分）
    3. 按配额和分数选出最终引用证据

    参数:
        service: 工作流服务对象，提供 _trace 方法用于节点追踪。
        state: 工作流状态字典，包含 wiki_hits、code_hits、case_hits 等。

    返回:
        状态增量字典，包含：
        - citations: 最终选中的证据列表
        - evidence_fusion_profile: 融合过程的详细统计
        - node_trace: 节点追踪信息
    """
    plan = _normalize_retrieval_plan(state)

    candidates = _build_candidate_items(state, plan)

    citations = _select_with_quota(candidates, plan)

    selected_counts: dict[str, int] = defaultdict(int)
    for index, item in enumerate(citations, start=1):
        source = _as_source_key(str(item.get("source_type", "")))
        selected_counts[source] += 1
        item["fusion_rank"] = index

    profile = {
        "strategy": plan["strategy"],
        "final_top_k": plan["final_top_k"],
        "source_weights": plan["source_weights"],
        "max_per_source": plan["max_per_source"],
        "input_counts": {
            "wiki": len(state.get("wiki_hits", [])),
            "case": len(state.get("case_hits", [])),
            "code": len(state.get("code_hits", [])),
        },
        "candidate_count": len(candidates),
        "selected_count": len(citations),
        "selected_counts_by_source": dict(selected_counts),
    }

    summary = (
        f"strategy={plan['strategy']},candidates={len(candidates)},"
        f"selected={len(citations)},wiki={selected_counts.get('wiki', 0)},"
        f"code={selected_counts.get('code', 0)},case={selected_counts.get('case', 0)}"
    )
    return {
        "citations": citations,
        "evidence_fusion_profile": profile,
        "node_trace": service._trace(state, "merge_evidence", summary),
    }
