# -*- coding: utf-8 -*-
"""Unified intent routing node.

路由决策流程：
1. 规则路由计算（领域相关性 + 意图分类）
2. 置信度评估
3. 低置信度时调用 LLM 兜底
4. 返回最终路由结果

路由结果类型：
- knowledge_qa: 知识问答
- issue_analysis: 问题分析
- code_generation: 代码生成
- other_query: 其他查询（领域相关但需要灵活处理）
- out_of_scope: 领域无关
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import re

from workflow.common.node_trace import append_node_trace
from workflow.nodes.routing_context.llm_router import LLMRouter, LLMRoutingResult

logger = logging.getLogger(__name__)

_ISSUE_SIGNAL_RE = re.compile(
    r"(报错|错误|异常|失败|超时|告警|报警|traceback|error|exception|\b[45]\d{2}\b|panic|fatal)",
    re.IGNORECASE,
)


@lru_cache(maxsize=16)
def _compile_pattern(pattern: str, fallback: str) -> re.Pattern[str]:
    raw = (pattern or "").strip() or fallback
    return re.compile(raw)


def _normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term and term in text for term in terms)


def _count_hits(text: str, terms: tuple[str, ...]) -> int:
    return sum(1 for term in terms if term and term in text)


def _is_small_talk(text: str, *, exact: tuple[str, ...], substr: tuple[str, ...], laugh_like: re.Pattern[str]) -> bool:
    if not text:
        return True
    if text in exact:
        return True
    if any(token in text for token in substr):
        return True
    return laugh_like.fullmatch(text) is not None


def _compute_domain_relevance(service: Any, state: dict[str, Any]) -> tuple[float, bool, str, int, bool]:
    """计算领域相关性

    Returns:
        tuple: (relevance, in_scope, reason, domain_hits, code_hint)
    """
    profile = service.domain_profile.domain_gate
    user_query = str(state.get("user_query", "") or "")
    normalized = _normalize(user_query)

    laugh_like = _compile_pattern(profile.laugh_like_regex, r"^[哈哈呵呵嘻嘻啊哦嗯\s!？?.。~～]+$")
    code_hint_re = _compile_pattern(profile.code_hint_regex, r"[A-Za-z_][A-Za-z0-9_]{2,}\s*(\(|\.py\b|/)")

    if _is_small_talk(
        normalized,
        exact=profile.small_talk_exact,
        substr=profile.small_talk_substr,
        laugh_like=laugh_like,
    ):
        return 0.0, False, "small_talk", 0, False

    domain_hits = _count_hits(normalized, profile.domain_terms)
    off_hits = _count_hits(normalized, profile.offtopic_terms)
    code_hint = code_hint_re.search(user_query) is not None
    code_request = _looks_like_code_generation_request(user_query)
    has_active_module = bool(str(state.get("active_module_name", "") or "").strip())

    relevance = min(1.0, domain_hits * 0.22 + (0.22 if code_hint else 0.0))
    if domain_hits > 0:
        relevance = max(relevance, profile.weak_in_scope_min_score)
    if code_hint:
        relevance = max(relevance, profile.weak_code_hint_min_score)
    if code_request and has_active_module:
        relevance = max(relevance, profile.weak_code_hint_min_score)

    if str(state.get("active_topic_source", "")) == "history_memory":
        relevance = min(1.0, relevance + profile.history_memory_bonus)

    if off_hits > 0 and domain_hits == 0 and not code_hint:
        return max(0.0, relevance - profile.offtopic_penalty), False, f"offtopic_hits={off_hits}", domain_hits, code_hint

    compact = normalized.replace(" ", "")
    if len(compact) <= profile.short_query_max_len and domain_hits == 0 and not code_hint:
        return max(0.0, relevance - profile.short_query_penalty), False, "short_non_domain", domain_hits, code_hint

    in_scope = relevance >= profile.threshold
    reason = (
        f"domain_hits={domain_hits}, off_hits={off_hits}, code_hint={int(code_hint)}, "
        f"code_request={int(code_request)}, has_active_module={int(has_active_module)}, "
        f"source={state.get('active_topic_source', 'current_query')}"
    )
    return relevance, in_scope, reason, domain_hits, code_hint


def _has_incident_detail_signal(normalized_query: str) -> bool:
    detail_terms = (
        "日志",
        "log",
        "trace",
        "stack",
        "request_id",
        "trace_id",
        "campaign",
        "adset",
        "线上",
        "离线",
        "告警",
        "报警",
        "timeout",
        "5xx",
        "4xx",
    )
    has_detail_term = _contains_any(normalized_query, detail_terms)
    has_number = re.search(r"\d", normalized_query) is not None
    has_time_window = re.search(r"(\d{1,2}:\d{2}|\d{4}-\d{1,2}-\d{1,2})", normalized_query) is not None
    return has_detail_term or has_number or has_time_window


def _is_generic_troubleshoot_question(normalized_query: str, *, has_troubleshoot_terms: bool) -> bool:
    if not has_troubleshoot_terms:
        return False
    method_terms = (
        "怎么排查",
        "如何排查",
        "排查思路",
        "排查步骤",
        "排查手册",
        "排障手册",
        "排查方法",
        "排查流程",
        "排查建议",
        "怎么定位",
        "如何定位",
        "怎么处理",
        "如何处理",
        "how to troubleshoot",
        "troubleshooting steps",
        "troubleshoot guide",
        "how to locate",
        "how to fix",
    )
    asks_for_method = _contains_any(normalized_query, method_terms)
    if not asks_for_method:
        return False
    return not _has_incident_detail_signal(normalized_query)


def _classify_in_scope_intent(service: Any, user_query: str) -> tuple[str, str]:
    """分类域内意图

    Returns:
        tuple: (intent, match_type)
        - intent: 意图类型
        - match_type: 匹配类型 "explicit" | "heuristic" | "fallback"
    """
    normalized_query = _normalize(user_query)
    intent_terms = service.domain_profile.query_rewrite.intent_terms

    if _looks_like_code_generation_request(user_query):
        return "code_generation", "explicit"

    has_troubleshoot_terms = _contains_any(normalized_query, intent_terms.get("troubleshoot", ()))
    has_issue_regex_hit = _ISSUE_SIGNAL_RE.search(normalized_query) is not None
    is_generic_troubleshoot = _is_generic_troubleshoot_question(
        normalized_query,
        has_troubleshoot_terms=has_troubleshoot_terms,
    )

    if has_issue_regex_hit:
        return "issue_analysis", "explicit"
    if has_troubleshoot_terms and not is_generic_troubleshoot:
        return "issue_analysis", "explicit"

    # 默认路由到 knowledge_qa，标记为 fallback
    return "knowledge_qa", "fallback"


def _has_analysis_context(state: dict[str, Any]) -> bool:
    return bool(state.get("last_analysis_result"))


def _compute_routing_confidence(
    relevance: float,
    domain_hits: int,
    code_hint: bool,
    intent_match_type: str,
    has_context: bool,
) -> float:
    """计算规则路由置信度

    Args:
        relevance: 领域相关性分数
        domain_hits: 领域词命中数量
        code_hint: 是否有代码线索
        intent_match_type: 意图匹配类型 ("explicit" | "heuristic" | "fallback")
        has_context: 是否有上下文

    Returns:
        置信度分数 (0.0 ~ 1.0)
    """
    base_confidence = 0.5

    # 领域词命中加成 (每个命中 +0.1，上限 0.3)
    domain_bonus = min(0.3, domain_hits * 0.1)

    # 代码线索加成
    code_bonus = 0.15 if code_hint else 0.0

    # 意图匹配强度加成
    intent_bonus = {
        "explicit": 0.25,
        "heuristic": 0.10,
        "fallback": 0.0,
    }.get(intent_match_type, 0.0)

    # 上下文加成
    context_bonus = 0.1 if has_context else 0.0

    confidence = min(1.0, base_confidence + domain_bonus + code_bonus + intent_bonus + context_bonus)
    return confidence


def _should_call_llm_routing(
    confidence: float,
    relevance: float,
    routing_profile: Any,
) -> bool:
    """判断是否需要 LLM 兜底

    触发条件：
    1. LLM 兜底已启用
    2. 置信度低于阈值
    3. 相关性在指定范围内
    """
    if not routing_profile.llm_fallback_enabled:
        return False

    if confidence >= routing_profile.llm_fallback_confidence_threshold:
        return False

    min_rel, max_rel = routing_profile.llm_fallback_relevance_range
    return min_rel <= relevance <= max_rel


def _get_llm_router(service: Any) -> LLMRouter | None:
    """获取 LLM 路由器实例

    优先从 service 获取，否则创建新实例。
    """
    # 尝试从 service 获取预初始化的路由器
    if hasattr(service, "_llm_router") and service._llm_router is not None:
        return service._llm_router

    # 动态创建
    try:
        from agent.llm.client import get_llm_client

        llm_client = get_llm_client()
        if llm_client is None:
            logger.warning("[intent_routing] LLM 客户端未初始化，跳过 LLM 路由")
            return None

        routing_profile = service.domain_profile.routing_profile
        router = LLMRouter.create_from_config(
            llm_client=llm_client,
            routing_profile=routing_profile,
        )
        return router
    except Exception as e:
        logger.warning(f"[intent_routing] 创建 LLM 路由器失败: {e}")
        return None


def _call_llm_routing(
    service: Any,
    user_query: str,
    relevance: float,
    module_name: str,
    module_hint: str,
) -> LLMRoutingResult | None:
    """调用 LLM 路由决策

    Args:
        service: 服务实例
        user_query: 用户查询
        relevance: 领域相关性
        module_name: 模块名
        module_hint: 模块提示

    Returns:
        LLMRoutingResult 或 None（调用失败时）
    """
    router = _get_llm_router(service)
    if router is None:
        return None

    # 构建领域上下文
    domain_context = ""
    if module_name and module_hint:
        domain_context = f"当前模块: {module_name} ({module_hint})"

    try:
        result = router.route(
            user_query=user_query,
            domain_context=domain_context,
            relevance=relevance,
        )
        return result
    except Exception as e:
        logger.error(f"[intent_routing] LLM 路由调用失败: {e}")
        return None


def run(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """执行意图路由

    流程：
    1. 规则路由计算（领域相关性 + 意图分类）
    2. 置信度评估
    3. 低置信度时调用 LLM 兜底
    4. 返回最终路由结果

    Returns:
        状态增量，包含：
        - domain_relevance: 领域相关性
        - route: 路由目标
        - routing_confidence: 置信度
        - routing_method: 路由方式
        - routing_intent_reason: 路由理由
        - llm_routing_result: LLM 路由结果（如有）
        - node_trace: 节点追踪
    """
    user_query = str(state.get("user_query", "") or "")
    relevance, in_scope, relevance_reason, domain_hits, code_hint = _compute_domain_relevance(service, state)

    # 领域无关，直接返回
    if not in_scope:
        return {
            "domain_relevance": relevance,
            "route": "out_of_scope",
            "routing_confidence": 0.9,  # 领域无关的规则判断通常比较可靠
            "routing_method": "rule",
            "routing_intent_reason": relevance_reason,
            "node_trace": append_node_trace(
                state,
                "intent_routing",
                f"route=out_of_scope, relevance={relevance:.2f}, reason={relevance_reason}",
            ),
        }

    # 规则路由分类
    rule_route, intent_match_type = _classify_in_scope_intent(service, user_query)

    # 计算置信度
    has_context = bool(state.get("module_name") or state.get("active_topic_source") == "history_memory")
    confidence = _compute_routing_confidence(
        relevance=relevance,
        domain_hits=domain_hits,
        code_hint=code_hint,
        intent_match_type=intent_match_type,
        has_context=has_context,
    )

    routing_profile = service.domain_profile.routing_profile
    routing_method = "rule"
    final_route = rule_route
    routing_reason = f"rule:{intent_match_type}"
    llm_routing_result = None

    # 判断是否需要 LLM 兜底
    if _should_call_llm_routing(confidence, relevance, routing_profile):
        logger.info(
            f"[intent_routing] 触发 LLM 兜底: confidence={confidence:.2f}, "
            f"relevance={relevance:.2f}, rule_route={rule_route}"
        )

        module_name = str(state.get("module_name", "") or "")
        module_hint = str(state.get("module_hint", "") or "")

        llm_result = _call_llm_routing(
            service=service,
            user_query=user_query,
            relevance=relevance,
            module_name=module_name,
            module_hint=module_hint,
        )

        if llm_result is not None and llm_result.success:
            routing_method = "llm"
            final_route = llm_result.intent
            routing_reason = f"llm:{llm_result.reason}"
            llm_routing_result = {
                "intent": llm_result.intent,
                "reason": llm_result.reason,
                "latency_ms": llm_result.latency_ms,
            }
            logger.info(f"[intent_routing] LLM 路由结果: {final_route}, reason={routing_reason}")

    # 构建返回结果
    has_analysis_context = _has_analysis_context(state)
    result = {
        "domain_relevance": relevance,
        "route": final_route,
        "routing_confidence": confidence,
        "routing_method": routing_method,
        "routing_intent_reason": routing_reason,
        "node_trace": append_node_trace(
            state,
            "intent_routing",
            f"route={final_route}, method={routing_method}, confidence={confidence:.2f}, "
            f"relevance={relevance:.2f}, has_analysis_context={int(has_analysis_context)}",
        ),
    }

    if llm_routing_result is not None:
        result["llm_routing_result"] = llm_routing_result

    return result


def _looks_like_code_generation_request(text: str) -> bool:
    terms = (
        "给我代码",
        "给出代码",
        "直接给代码",
        "代码实现",
        "实现一个",
        "写一下代码",
        "补丁",
        "patch",
        "改代码",
        "修改代码",
        "直接修",
        "生成代码",
        "code snippet",
        "show me code",
        "write code",
    )
    lowered = str(text or "").lower()
    original = str(text or "")
    return any(term in original or term in lowered for term in terms)
