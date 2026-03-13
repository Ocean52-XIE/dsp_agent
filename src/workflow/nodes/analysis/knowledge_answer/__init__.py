# -*- coding: utf-8 -*-
"""
该模块实现工作流节点`knowledge_answer` 的处理逻辑，负责读取状态并输出增量结果。
"""
from __future__ import annotations

import re
from typing import Any

from workflow.utils import normalize_source_type


def _infer_question_type(service: Any, user_query: str) -> str:
    """
    内部辅助函数，负责`infer question type` 相关处理。
    
    参数:
        service: 工作流服务对象，提供检索、路由、日志与配置能力。
        user_query: 用户输入问题文本。
    
    返回:
        返回类型为 `str` 的处理结果。
    """
    normalized = user_query.lower().strip()
    formula_terms = {
        "公式",
        "怎么算",
        "如何计算",
        "计算方式",
        "calculate",
        "formula",
        "bid",
    }
    formula_terms.update(service.domain_profile.answering.bid_terms)
    if any(token in normalized for token in formula_terms if token):
        return "formula"
    if any(token in normalized for token in ("为什么", "为何", "原因", "怎么会", "为啥")):
        return "reason"
    if any(token in normalized for token in ("有哪些", "哪些", "包括什么", "包含什么", "分别是", "列出", "清单")):
        return "list"
    return "general"


def _is_code_location_query(service: Any, user_query: str) -> bool:
    """
    判断输入是否满足特定条件，并返回布尔结果。
    
    参数:
        service: 工作流服务对象，提供检索、路由、日志与配置能力。
        user_query: 用户输入问题文本。
    
    返回:
        返回类型为 `bool` 的处理结果。
    """
    normalized = user_query.lower().strip()
    code_location_terms = service.domain_profile.query_rewrite.intent_terms.get("code_location", ())
    if code_location_terms:
        return any(token in normalized for token in code_location_terms)
    fallback = (
        "哪个函数",
        "什么函数",
        "入口函数",
        "哪个文件",
        "文件路径",
        "哪一行",
        "where",
        "which function",
    )
    return any(token in normalized for token in fallback)


def _collect_evidence_hits(state: dict[str, Any]) -> list[dict[str, Any]]:
    """
    收集并标准化当前流程依赖的输入数据。
    
    参数:
        state: 工作流状态字典，包含会话上下文与中间结果。
    
    返回:
        返回类型为 `list[dict[str, Any]]` 的处理结果。
    """
    citations = state.get("citations", [])
    if isinstance(citations, list) and citations:
        return citations

    fallback_hits: list[dict[str, Any]] = []
    for key, default_type in (("wiki_hits", "wiki"), ("code_hits", "code"), ("case_hits", "case")):
        for item in list(state.get(key, []) or []):
            row = dict(item)
            row["source_type"] = normalize_source_type(row.get("source_type", default_type))
            fallback_hits.append(row)
    return fallback_hits


def _format_line_range(item: dict[str, Any]) -> str:
    """
    内部辅助函数，负责`format line range` 相关处理。
    
    参数:
        item: 输入参数，用于控制当前处理逻辑。
    
    返回:
        返回类型为 `str` 的处理结果。
    """
    start_line = item.get("start_line")
    end_line = item.get("end_line")
    if start_line is not None and end_line is not None:
        return f"{start_line}-{end_line}"
    if start_line is not None:
        return str(start_line)
    return "unknown"


def _answer_mentions_code_anchor(answer_text: str, code_hits: list[dict[str, Any]]) -> bool:
    """
    内部辅助函数，负责`answer mentions code anchor` 相关处理。
    
    参数:
        answer_text: 输入参数，用于控制当前处理逻辑。
        code_hits: 列表参数，用于承载批量输入数据。
    
    返回:
        返回类型为 `bool` 的处理结果。
    """
    normalized_answer = answer_text.lower()
    for item in code_hits[:6]:
        path = str(item.get("path", "")).strip().lower()
        symbol_name = str(item.get("symbol_name", "")).strip().lower()
        if path and (path in normalized_answer or path.split("/")[-1] in normalized_answer):
            return True
        if symbol_name and symbol_name in normalized_answer:
            return True
    return False


def _extract_points_from_hits(evidence_hits: list[dict[str, Any]], *, max_points: int = 6) -> list[str]:
    """
    内部辅助函数，负责`extract points from hits` 相关处理。
    
    参数:
        evidence_hits: 列表参数，用于承载批量输入数据。
    
    返回:
        返回类型为 `list[str]` 的处理结果。
    """
    points: list[str] = []
    for item in evidence_hits:
        source_type = normalize_source_type(item.get("source_type"))
        excerpt = re.sub(r"\s+", " ", str(item.get("excerpt", "")).strip())
        if source_type == "code":
            path = str(item.get("path", "")).strip() or "unknown_path"
            symbol_name = str(item.get("symbol_name", "")).strip() or str(item.get("section", "")).strip() or "unknown_symbol"
            points.append(f"代码位置：`{path}` -> `{symbol_name}`（line: {_format_line_range(item)}）")
            if excerpt:
                points.append(f"代码摘要：{excerpt[:140]}")
        else:
            title = str(item.get("title", "")).strip() or str(item.get("path", "")).strip() or "未命名证据"
            if excerpt:
                points.append(f"{title}：{excerpt[:140]}")
            else:
                points.append(title)
        if len(points) >= max_points:
            break
    return points[:max_points]


def _build_code_location_fallback(module_name: str, module_hint: str, code_hits: list[dict[str, Any]]) -> str:
    """
    构建当前步骤所需的数据结构或文本内容。
    
    参数:
        module_name: 输入参数，用于控制当前处理逻辑。
        module_hint: 输入参数，用于控制当前处理逻辑。
        code_hits: 列表参数，用于承载批量输入数据。
    
    返回:
        返回类型为 `str` 的处理结果。
    """
    lines: list[str] = []
    for item in code_hits[:3]:
        path = str(item.get("path", "")).strip() or "unknown_path"
        symbol_name = str(item.get("symbol_name", "")).strip() or str(item.get("section", "")).strip() or "unknown_symbol"
        lines.append(f"{len(lines) + 1}. `{path}` -> `{symbol_name}`（line: {_format_line_range(item)}）")
    if not lines:
        lines = [
            f"1. 当前问题主要落在 `{module_name}`。",
            f"2. 当前模块说明：{module_hint or '--'}。",
            "3. 目前没有命中可直接定位的代码证据，建议补充函数名、文件名或调用链关键字后重试。",
        ]
    return "\n".join(lines)


def _build_general_fallback(
    *,
    module_name: str,
    module_hint: str,
    question_type: str,
    evidence_hits: list[dict[str, Any]],
) -> str:
    """
    构建当前步骤所需的数据结构或文本内容。
    
    返回:
        返回类型为 `str` 的处理结果。
    """
    points = _extract_points_from_hits(evidence_hits, max_points=6)
    if not points:
        return (
            f"当前问题主要落在 `{module_name}`。"
            f" 模块说明：{module_hint or '--'}。"
            " 当前没有命中足够证据，建议补充更具体的业务词、字段名或代码符号后重试。"
        )

    if question_type == "list":
        return "\n".join(f"{index}. {point}" for index, point in enumerate(points[:5], start=1))
    if question_type == "reason":
        return "\n".join(f"{index}. {point}" for index, point in enumerate(points[:4], start=1))
    if question_type == "formula":
        return "\n".join(f"{index}. {point}" for index, point in enumerate(points[:4], start=1))
    return "；".join(points[:3])


def run(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """
    执行`knowledge_answer` 节点主流程，基于输入状态计算并返回状态增量。
    
    参数:
        service: 工作流服务对象，提供检索、路由、日志与配置能力。
        state: 工作流状态字典，包含会话上下文与中间结果。
    
    返回:
        返回类型为 `dict[str, Any]` 的处理结果。
    """
    module_name = state["module_name"]
    module_hint = state["module_hint"]
    user_query = str(state.get("user_query", ""))
    transition_type = state.get("transition_type", "start_knowledge_qa")
    question_type = _infer_question_type(service, user_query)

    wiki_hits = list(state.get("wiki_hits", []))
    code_hits = list(state.get("code_hits", []))
    evidence_hits = _collect_evidence_hits(state)

    llm_mode = "fallback_rule"
    llm_fallback_reason: str | None = None
    llm_call_status: dict[str, Any] = {
        "status": "not_configured",
        "invoked": False,
        "request_sent": False,
        "attempts": 0,
        "latency_ms": 0,
        "reason": None,
        "model": None,
    }
    final_answer: str

    llm_answer_text: str | None = None
    if hasattr(service, "_knowledge_qa_llm") and service._knowledge_qa_llm is not None:
        llm_client = service._knowledge_qa_llm
        llm_answer_text, llm_fallback_reason = llm_client.generate_answer(
            user_query=user_query,
            question_type=question_type,
            module_name=module_name,
            module_hint=module_hint,
            retrieval_queries=state.get("retrieval_queries", []),
            evidence_hits=evidence_hits,
        )
        latest_status = getattr(llm_client, "last_call_status", {}) or {}
        if isinstance(latest_status, dict):
            llm_call_status = dict(latest_status)
        llm_call_status.setdefault("model", llm_client.config.model)
        llm_call_status.setdefault("invoked", True)

    if llm_answer_text:
        if _is_code_location_query(service, user_query) and code_hits and not _answer_mentions_code_anchor(llm_answer_text, code_hits):
            llm_fallback_reason = "llm_missing_code_anchor"
            llm_call_status.update(
                {
                    "status": "success_discarded",
                    "reason": "llm_missing_code_anchor",
                }
            )
            final_answer = _build_code_location_fallback(module_name, module_hint, code_hits)
        else:
            llm_mode = "llm"
            final_answer = llm_answer_text
    else:
        if _is_code_location_query(service, user_query) and code_hits:
            final_answer = _build_code_location_fallback(module_name, module_hint, code_hits)
        else:
            final_answer = _build_general_fallback(
                module_name=module_name,
                module_hint=module_hint,
                question_type=question_type,
                evidence_hits=evidence_hits,
            )

    return {
        "response_kind": "knowledge_qa",
        "task_stage": "knowledge_qa",
        "status": "completed",
        "next_action": "completed",
        "answer": final_answer,
        "analysis": {
            "summary": "知识问答已完成",
            "module": module_name,
            "confidence": "medium",
            "transition_type": transition_type,
            "task_stage": "knowledge_qa",
            "generation_mode": llm_mode,
            "question_type": question_type,
            "evidence_count": len(evidence_hits),
            "wiki_evidence_count": len(wiki_hits),
            "code_evidence_count": len(code_hits),
            "llm_enabled": bool(
                hasattr(service, "_knowledge_qa_llm")
                and service._knowledge_qa_llm is not None
                and service._knowledge_qa_llm.config.enabled
            ),
            "llm_available": bool(
                hasattr(service, "_knowledge_qa_llm")
                and service._knowledge_qa_llm is not None
                and service._knowledge_qa_llm.is_available
            ),
            "llm_model": (
                service._knowledge_qa_llm.config.model
                if hasattr(service, "_knowledge_qa_llm") and service._knowledge_qa_llm is not None
                else None
            ),
            "llm_fallback_reason": llm_fallback_reason,
            "llm_call_status": llm_call_status,
            "highlights": [
                "知识问答优先尝试使用 LLM 基于证据生成答案",
                "代码定位类问题会额外检查答案里是否包含代码锚点",
                "最终 Markdown 三段式格式由 finalize_response 节点统一收口",
            ],
        },
        "node_trace": service._trace(state, "knowledge_answer", f"module={module_name}"),
    }
