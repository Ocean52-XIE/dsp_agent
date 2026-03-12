from __future__ import annotations

"""多源证据融合节点。"""

from collections import defaultdict
from typing import Any


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
    """将证据 source_type 归一化为融合阶段使用的键。"""
    normalized = source_type.strip().lower()
    if normalized.startswith("wiki"):
        return "wiki"
    if normalized.startswith("code"):
        return "code"
    if normalized.startswith("case"):
        return "case"
    return normalized or "unknown"


def _normalize_retrieval_plan(state: dict[str, Any]) -> dict[str, Any]:
    """标准化 retrieval_plan，补齐默认值并保证参数合法。"""
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
    """根据意图标签做融合参数兜底。

    背景：
    上游 query_rewriter 已经给出 retrieval_plan，但线上会存在两类风险：
    1. retrieval_plan 缺字段或被旧逻辑覆盖；
    2. 实际问题是函数定位，但融合阶段又被 wiki 证据抢前。

    这里做“最后一道兜底”：
    - 若识别为 code-location，强制收紧 wiki 配额并提高 code 权重；
    - 若识别为 wiki-first，反向做轻量倾斜。
    """
    intent_profile = dict(plan.get("intent_profile", {}))
    is_code_location = bool(intent_profile.get("is_code_location"))
    is_wiki_intent = bool(intent_profile.get("is_wiki_intent"))

    # 兜底策略 1：函数定位问题强制 code-first，避免“路径/函数名被 wiki 解释性段落淹没”。
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
    """读取指定检索源的质量分级。"""
    if source == "wiki":
        return str(state.get("wiki_retrieval_grade", "unknown"))
    if source == "code":
        return str(state.get("code_retrieval_grade", "unknown"))
    if source == "case":
        return str(state.get("case_retrieval_grade", "unknown"))
    return "unknown"


def _build_candidate_items(state: dict[str, Any], plan: dict[str, Any]) -> list[dict[str, Any]]:
    """构建融合候选项并计算融合分数。"""
    source_to_hits: dict[str, list[dict[str, Any]]] = {
        "wiki": list(state.get("wiki_hits", [])),
        "case": list(state.get("case_hits", [])),
        "code": list(state.get("code_hits", [])),
    }
    candidates: list[dict[str, Any]] = []

    intent_profile = dict(plan.get("intent_profile", {}))
    is_code_location = bool(intent_profile.get("is_code_location"))
    is_wiki_first = plan.get("strategy") == "wiki_first"
    for source, hits in source_to_hits.items():
        source_weight = float(plan["source_weights"].get(source, 1.0))
        grade = _get_retrieval_grade(state, source)
        grade_bias = float(GRADE_BIAS.get(grade, 0.0))
        # P0：在融合打分时引入“意图偏置”。
        # 即便 source_weight 接近，也能保证函数定位问题里 code 证据更稳定进入头部。
        intent_bias = 0.0
        if is_code_location:
            intent_bias = float(INTENT_SOURCE_BIAS["code_location"].get(source, 0.0))
        elif is_wiki_first:
            intent_bias = float(INTENT_SOURCE_BIAS["wiki_first"].get(source, 0.0))
        for rank, hit in enumerate(hits, start=1):
            base_score = float(hit.get("score", 0.0))
            rank_bonus = 0.12 / rank
            fused_score = base_score * source_weight + rank_bonus + grade_bias + intent_bias
            item = dict(hit)
            item_source = _as_source_key(str(item.get("source_type", source)))
            item["source_type"] = item_source
            item["fusion_score"] = round(fused_score, 6)
            item["fusion_debug"] = {
                "base_score": round(base_score, 6),
                "source_weight": round(source_weight, 6),
                "rank_bonus": round(rank_bonus, 6),
                "grade": grade,
                "grade_bias": round(grade_bias, 6),
                "intent_bias": round(intent_bias, 6),
            }
            candidates.append(item)

    candidates.sort(
        key=lambda item: (
            float(item.get("fusion_score", 0.0)),
            float(item.get("score", 0.0)),
        ),
        reverse=True,
    )
    return candidates


def _dedup_key(item: dict[str, Any]) -> tuple[str, str, str]:
    """构造去重键，避免同一证据重复入选。"""
    return (
        str(item.get("source_type", "")),
        str(item.get("path", "")),
        str(item.get("section", "")),
    )


def _select_with_quota(candidates: list[dict[str, Any]], plan: dict[str, Any]) -> list[dict[str, Any]]:
    """按融合分排序后执行“先配额、再补齐”的证据选择策略。"""
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
    """将 wiki/case/code 结果做加权融合并输出最终引用列表。"""
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
