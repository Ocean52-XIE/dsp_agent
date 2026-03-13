# -*- coding: utf-8 -*-
"""
该模块实现工作流节点`query_rewriter` 的处理逻辑，负责读取状态并输出增量结果。
"""
from __future__ import annotations

from typing import Any


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    """
    内部辅助函数，负责`contains any` 相关处理。
    
    参数:
        text: 待处理的文本内容。
        keywords: 列表参数，用于承载批量输入数据。
    
    返回:
        返回类型为 `bool` 的处理结果。
    """
    return any(keyword in text for keyword in keywords if keyword)


def _infer_code_location_intent(normalized_query: str, *, code_location_terms: tuple[str, ...]) -> bool:
    """
    内部辅助函数，负责`infer code location intent` 相关处理。
    
    参数:
        normalized_query: 输入参数，用于控制当前处理逻辑。
    
    返回:
        返回类型为 `bool` 的处理结果。
    """
    return _contains_any(normalized_query, code_location_terms)


def _infer_query_flags(service: Any, user_query: str) -> dict[str, bool]:
    """
    内部辅助函数，负责`infer query flags` 相关处理。
    
    参数:
        service: 工作流服务对象，提供检索、路由、日志与配置能力。
        user_query: 用户输入问题文本。
    
    返回:
        返回类型为 `dict[str, bool]` 的处理结果。
    """
    profile = service.domain_profile.query_rewrite
    intent_terms = profile.intent_terms

    normalized = user_query.lower().strip()
    code_location_terms = intent_terms.get("code_location", ())
    return {
        "metric": _contains_any(normalized, intent_terms.get("metric", ())),
        "pipeline": _contains_any(normalized, intent_terms.get("pipeline", ())),
        "architecture": _contains_any(normalized, intent_terms.get("architecture", ())),
        "troubleshoot": _contains_any(normalized, intent_terms.get("troubleshoot", ())),
        "code": _contains_any(normalized, intent_terms.get("code", ())),
        "code_location": _infer_code_location_intent(normalized, code_location_terms=code_location_terms),
    }


def _module_alias_queries(service: Any, module_name: str) -> list[str]:
    """
    内部辅助函数，负责`module alias queries` 相关处理。
    
    参数:
        service: 工作流服务对象，提供检索、路由、日志与配置能力。
        module_name: 输入参数，用于控制当前处理逻辑。
    
    返回:
        返回类型为 `list[str]` 的处理结果。
    """
    return service.domain_profile.module_alias_queries(module_name)


def _expand_queries(service: Any, base_queries: list[str], *, original_user_query: str) -> list[str]:
    """
    内部辅助函数，负责`expand queries` 相关处理。
    
    参数:
        service: 工作流服务对象，提供检索、路由、日志与配置能力。
        base_queries: 列表参数，用于承载批量输入数据。
    
    返回:
        返回类型为 `list[str]` 的处理结果。
    """
    profile = service.domain_profile.query_rewrite
    merged = " ".join(base_queries).lower()
    original = original_user_query.lower().strip()
    expanded = list(base_queries)

    for anchor, aliases in profile.synonyms.items():
        trigger_terms = (anchor, *aliases)
        if any(term in original for term in trigger_terms):
            expanded.extend([anchor, *aliases])

    for abbr, canonical in profile.abbreviations.items():
        if abbr in merged:
            expanded.append(canonical)

    deduped: list[str] = []
    seen: set[str] = set()
    for query in expanded:
        normalized = " ".join(query.strip().split())
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped[:20]


def _expand_symbol_alias_queries(service: Any, user_query: str) -> list[str]:
    """
    内部辅助函数，负责`expand symbol alias queries` 相关处理。
    
    参数:
        service: 工作流服务对象，提供检索、路由、日志与配置能力。
        user_query: 用户输入问题文本。
    
    返回:
        返回类型为 `list[str]` 的处理结果。
    """
    lowered = user_query.lower().strip()
    expanded: list[str] = []
    for phrase, aliases in service.domain_profile.query_rewrite.symbol_aliases.items():
        if phrase in lowered:
            expanded.extend(list(aliases))
    return expanded


def _expand_template_queries(service: Any, user_query: str) -> list[str]:
    """
    内部辅助函数，负责`expand template queries` 相关处理。
    
    参数:
        service: 工作流服务对象，提供检索、路由、日志与配置能力。
        user_query: 用户输入问题文本。
    
    返回:
        返回类型为 `list[str]` 的处理结果。
    """
    lowered = user_query.lower().strip()
    expanded: list[str] = []
    for item in service.domain_profile.query_rewrite.query_templates:
        trigger_all = tuple(str(term).lower().strip() for term in item.get("trigger_all", []) if str(term).strip())
        trigger_any = tuple(str(term).lower().strip() for term in item.get("trigger_any", []) if str(term).strip())
        if trigger_all and not all(token in lowered for token in trigger_all):
            continue
        if trigger_any and not any(token in lowered for token in trigger_any):
            continue
        queries = [str(query).strip() for query in item.get("queries", []) if str(query).strip()]
        expanded.extend(queries)
    return expanded


def _build_retrieval_plan(
    service: Any,
    *,
    user_query: str,
    route: str,
    transition_type: str,
    query_flags: dict[str, bool],
) -> dict[str, Any]:
    """
    构建当前步骤所需的数据结构或文本内容。
    
    参数:
        service: 工作流服务对象，提供检索、路由、日志与配置能力。
    
    返回:
        返回类型为 `dict[str, Any]` 的处理结果。
    """
    retrieval_profile = service.domain_profile.retrieval
    normalized = user_query.lower().strip()

    is_code_intent = bool(query_flags.get("code"))
    is_code_location = bool(query_flags.get("code_location"))
    is_wiki_intent = bool(query_flags.get("architecture") or query_flags.get("pipeline") or query_flags.get("metric"))
    if not is_wiki_intent:
        is_wiki_intent = any(token in normalized for token in ("架构", "流程", "阶段", "指标", "口径", "规则", "原理", "背景", "说明"))
    is_issue_analysis = route == "issue_analysis"

    strategy = "hybrid"
    if is_code_location:
        strategy = "code_first"
    elif is_code_intent and not is_wiki_intent:
        strategy = "code_first"
    elif is_wiki_intent and not is_code_intent:
        strategy = "wiki_first"

    preset = retrieval_profile.preset(strategy)
    source_weights = dict(retrieval_profile.source_weights)
    max_per_source = dict(retrieval_profile.max_per_source)
    reasons: list[str] = []

    if strategy == "code_first":
        source_weights["code"] = max(source_weights.get("code", 1.0), 1.2)
        source_weights["wiki"] = min(source_weights.get("wiki", 1.0), 0.9)
        max_per_source["code"] = max(int(max_per_source.get("code", 4)), 4)
        max_per_source["wiki"] = min(int(max_per_source.get("wiki", 4)), 2)
        reasons.append("query_prefers_code")
    elif strategy == "wiki_first":
        source_weights["wiki"] = max(source_weights.get("wiki", 1.0), 1.15)
        source_weights["code"] = min(source_weights.get("code", 1.0), 0.9)
        reasons.append("query_prefers_wiki")
    else:
        reasons.append("query_is_hybrid")

    if is_issue_analysis:
        preset["code_top_k"] = max(int(preset.get("code_top_k", 4)), 6)
        preset["wiki_top_k"] = max(int(preset.get("wiki_top_k", 4)), 3)
        source_weights["code"] = max(float(source_weights.get("code", 1.0)), 1.15)
        reasons.append("issue_analysis_bias_code")

    if transition_type == "upgrade_from_qa_to_issue_analysis":
        source_weights["code"] = max(float(source_weights.get("code", 1.0)), 1.25)
        reasons.append("upgrade_to_issue_analysis")

    plan: dict[str, Any] = {
        "strategy": strategy,
        "enable_wiki": bool(retrieval_profile.enable_wiki),
        "enable_code": bool(retrieval_profile.enable_code),
        "enable_cases": bool(retrieval_profile.enable_cases),
        "wiki_top_k": int(preset.get("wiki_top_k", 4)),
        "code_top_k": int(preset.get("code_top_k", 4)),
        "case_top_k": int(preset.get("case_top_k", 2)),
        "final_top_k": int(preset.get("final_top_k", 6)),
        "max_per_source": {
            "wiki": max(int(max_per_source.get("wiki", 4)), 0),
            "code": max(int(max_per_source.get("code", 4)), 0),
            "case": max(int(max_per_source.get("case", 1)), 0),
        },
        "source_weights": {
            "wiki": max(float(source_weights.get("wiki", 1.0)), 0.0),
            "code": max(float(source_weights.get("code", 1.0)), 0.0),
            "case": max(float(source_weights.get("case", 0.6)), 0.0),
        },
        "intent_profile": {
            "is_code_intent": is_code_intent,
            "is_code_location": is_code_location,
            "is_wiki_intent": is_wiki_intent,
            "is_issue_analysis": is_issue_analysis,
        },
        "reasons": reasons,
    }
    return plan


def run(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """
    执行`query_rewriter` 节点主流程，基于输入状态计算并返回状态增量。
    
    参数:
        service: 工作流服务对象，提供检索、路由、日志与配置能力。
        state: 工作流状态字典，包含会话上下文与中间结果。
    
    返回:
        返回类型为 `dict[str, Any]` 的处理结果。
    """
    module_name = state["module_name"]
    user_query = state["user_query"]
    route = state.get("route", "knowledge_qa")
    transition_type = state.get("transition_type", "start_knowledge_qa")
    flags = _infer_query_flags(service, user_query)
    # 基于 route 区分检索改写策略：问题分析更偏日志/案例，知识问答更偏语义扩展。
    # 同时根据查询意图标记补充模块别名与模板查询，提升召回覆盖率。
    # 当用户问题较短且无明显意图时，追加通用业务说明查询做兜底。
    # 若会话处于 issue_analysis/confirm_code/code_generation 阶段，追加设计背景类查询。

    if route == "issue_analysis":
        base_queries = [
            user_query,
            f"{module_name} 异常日志",
            f"{module_name} 排查案例",
            f"{module_name} 联调排障",
        ]
        if flags["pipeline"] or flags["architecture"]:
            base_queries.extend(_module_alias_queries(service, module_name))
    else:
        base_queries = [user_query]
        base_queries.extend(_expand_symbol_alias_queries(service, user_query))
        base_queries.extend(_expand_template_queries(service, user_query))
        if (
            flags["pipeline"]
            or flags["architecture"]
            or flags["metric"]
            or flags["troubleshoot"]
            or flags["code"]
            or flags["code_location"]
        ):
            base_queries.extend(_module_alias_queries(service, module_name)[:2])
        if not any(flags.values()):
            base_queries.append(f"{module_name} 业务说明")
        if state.get("active_task_stage") in {"issue_analysis", "confirm_code", "code_generation"}:
            base_queries.append(f"{module_name} 设计背景 业务口径")

    queries = _expand_queries(service, base_queries, original_user_query=user_query)
    retrieval_plan = _build_retrieval_plan(
        # 根据 route、transition_type 与 query flags 动态确定检索策略和 top_k。

        service,
        user_query=user_query,
        route=route,
        transition_type=transition_type,
        query_flags=flags,
    )
    trace_summary = (
        f"queries={len(queries)},strategy={retrieval_plan['strategy']},"
        f"wiki_top_k={retrieval_plan['wiki_top_k']},code_top_k={retrieval_plan['code_top_k']}"
    )
    return {
        "retrieval_queries": queries,
        "retrieval_plan": retrieval_plan,
        "node_trace": service._trace(state, "query_rewriter", trace_summary),
    }
