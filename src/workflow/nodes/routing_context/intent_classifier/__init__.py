# -*- coding: utf-8 -*-
"""
该模块实现工作流节点`intent_classifier` 的处理逻辑，负责读取状态并输出增量结果。
"""
from __future__ import annotations

import re
from typing import Any


_ISSUE_SIGNAL_RE = re.compile(
    r"(报错|错误|异常|失败|超时|告警|报警|traceback|error|exception|\b[45]\d{2}\b|panic|fatal)"
)


def _normalize(text: str) -> str:
    """
    内部辅助函数，负责`normalize` 相关处理。
    
    参数:
        text: 待处理的文本内容。
    
    返回:
        返回类型为 `str` 的处理结果。
    """
    return " ".join((text or "").strip().lower().split())


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    """
    内部辅助函数，负责`contains any` 相关处理。
    
    参数:
        text: 待处理的文本内容。
        terms: 列表参数，用于承载批量输入数据。
    
    返回:
        返回类型为 `bool` 的处理结果。
    """
    return any(term and term in text for term in terms)


def _has_incident_detail_signal(normalized_query: str) -> bool:
    """
    内部辅助函数，负责`has incident detail signal` 相关处理。
    
    参数:
        normalized_query: 输入参数，用于控制当前处理逻辑。
    
    返回:
        返回类型为 `bool` 的处理结果。
    """
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
        "超时",
        "5xx",
        "4xx",
    )
    has_detail_term = _contains_any(normalized_query, detail_terms)
    has_number = re.search(r"\d", normalized_query) is not None
    has_time_window = (
        re.search(r"(昨天|今天|近\d+(分钟|小时|天)|\d{1,2}:\d{2}|\d{4}-\d{1,2}-\d{1,2})", normalized_query) is not None
    )
    return has_detail_term or has_number or has_time_window


def _is_generic_troubleshoot_question(normalized_query: str, *, has_troubleshoot_terms: bool) -> bool:
    """
    判断输入是否满足特定条件，并返回布尔结果。
    
    参数:
        normalized_query: 输入参数，用于控制当前处理逻辑。
    
    返回:
        返回类型为 `bool` 的处理结果。
    """
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
    )
    asks_for_method = _contains_any(normalized_query, method_terms)
    if not asks_for_method:
        return False
    return not _has_incident_detail_signal(normalized_query)


def _is_issue_followup(service: Any, state: dict[str, Any], normalized_query: str) -> bool:
    """
    判断输入是否满足特定条件，并返回布尔结果。
    
    参数:
        service: 工作流服务对象，提供检索、路由、日志与配置能力。
        state: 工作流状态字典，包含会话上下文与中间结果。
        normalized_query: 输入参数，用于控制当前处理逻辑。
    
    返回:
        返回类型为 `bool` 的处理结果。
    """
    active_stage = str(state.get("active_task_stage", "idle") or "idle")
    if active_stage not in {"issue_analysis", "confirm_code"}:
        return False

    if service._is_pronoun_followup(normalized_query):
        return True

    followup_terms = (
        "继续",
        "然后",
        "那这个",
        "这个问题",
        "为什么",
        "原因",
        "怎么修",
        "怎么处理",
        "怎么排查",
        "怎么解决",
        "还会",
        "会不会",
    )
    return _contains_any(normalized_query, followup_terms)


def run(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """
    执行`intent_classifier` 节点主流程，基于输入状态计算并返回状态增量。
    
    参数:
        service: 工作流服务对象，提供检索、路由、日志与配置能力。
        state: 工作流状态字典，包含会话上下文与中间结果。
    
    返回:
        返回类型为 `dict[str, Any]` 的处理结果。
    """
    user_query = str(state.get("user_query", "") or "")
    normalized_query = _normalize(user_query)
    intent_terms = service.domain_profile.query_rewrite.intent_terms

    wants_code = service._looks_like_code_generation_request(user_query)
    declines_code = service._looks_like_decline_code_request(user_query)
    has_troubleshoot_terms = _contains_any(normalized_query, intent_terms.get("troubleshoot", ()))
    has_issue_regex_hit = _ISSUE_SIGNAL_RE.search(normalized_query) is not None
    is_generic_troubleshoot = _is_generic_troubleshoot_question(
        normalized_query,
        has_troubleshoot_terms=has_troubleshoot_terms,
    )
    issue_followup = _is_issue_followup(service, state, normalized_query)

    if wants_code or declines_code or issue_followup or has_issue_regex_hit:
        route = "issue_analysis"
    elif has_troubleshoot_terms and not is_generic_troubleshoot:
        route = "issue_analysis"
    else:
        route = "knowledge_qa"

    reason_flags = {
        "wants_code": wants_code,
        "declines_code": declines_code,
        "has_troubleshoot_terms": has_troubleshoot_terms,
        "has_issue_regex_hit": has_issue_regex_hit,
        "is_generic_troubleshoot": is_generic_troubleshoot,
        "issue_followup": issue_followup,
    }
    enabled_reasons = [name for name, enabled in reason_flags.items() if enabled]
    summary = f"route={route}, reasons={','.join(enabled_reasons) if enabled_reasons else 'default_knowledge_qa'}"

    return {
        "route": route,
        "node_trace": service._trace(state, "intent_classifier", summary),
    }
