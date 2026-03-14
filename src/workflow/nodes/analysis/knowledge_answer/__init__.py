# -*- coding: utf-8 -*-
"""Knowledge answer node."""
from __future__ import annotations

import re
from typing import Any

from workflow.common.evidence import collect_evidence_hits
from workflow.llm.llm_client import CommonLLMRequest
from workflow.llm.llm_prompt_utils import build_evidence_block, looks_like_reasoning_dump, resolve_system_prompt
from workflow.common.func_utils import normalize_source_type


QA_SYSTEM_PROMPT_TEMPLATE = (
    "你是企业知识问答助手。"
    "必须严格基于提供的证据回答，不补充证据外事实。"
    "输出中文，结构尽量为：结论 -> 依据。"
)

QA_USER_PROMPT_TEMPLATE = """【用户问题】
{user_query}

【当前主模块】
- module_name: {module_name}
- module_hint: {module_hint}

【相关模块】
{related_modules_block}

【检索证据（按相关性排序）】
{evidence_block}
"""


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords if keyword)


def _infer_question_type(service: Any, user_query: str) -> str:
    normalized = str(user_query or "").lower().strip()
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
    if any(token in normalized for token in ("为什么", "为何", "原因", "how to", "why")):
        return "reason"
    if any(token in normalized for token in ("有哪些", "哪些", "包括什么", "包含什么", "分别是", "列出", "清单")):
        return "list"
    return "general"


def _is_code_location_query(service: Any, user_query: str) -> bool:
    normalized = str(user_query or "").lower().strip()
    code_location_terms = service.domain_profile.query_rewrite.intent_terms.get("code_location", ())
    if code_location_terms:
        return _contains_any(normalized, code_location_terms)
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
    return _contains_any(normalized, fallback)


def _format_line_range(item: dict[str, Any]) -> str:
    start_line = item.get("start_line")
    end_line = item.get("end_line")
    if start_line is not None and end_line is not None:
        return f"{start_line}-{end_line}"
    if start_line is not None:
        return str(start_line)
    return "unknown"


def _answer_mentions_code_anchor(answer_text: str, code_hits: list[dict[str, Any]]) -> bool:
    normalized_answer = str(answer_text or "").lower()
    for item in code_hits[:6]:
        path = str(item.get("path", "")).strip().lower()
        symbol_name = str(item.get("symbol_name", "")).strip().lower()
        if path and (path in normalized_answer or path.split("/")[-1] in normalized_answer):
            return True
        if symbol_name and symbol_name in normalized_answer:
            return True
    return False


def _extract_points_from_hits(evidence_hits: list[dict[str, Any]], *, max_points: int = 6) -> list[str]:
    points: list[str] = []
    for item in evidence_hits:
        source_type = normalize_source_type(item.get("source_type"))
        excerpt = re.sub(r"\s+", " ", str(item.get("excerpt", "")).strip())
        if source_type == "code":
            path = str(item.get("path", "")).strip() or "unknown_path"
            symbol_name = str(item.get("symbol_name", "")).strip() or str(item.get("section", "")).strip() or "unknown_symbol"
            points.append(f"代码位置：`{path}` -> `{symbol_name}` (line: {_format_line_range(item)})")
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
    lines: list[str] = []
    for item in code_hits[:3]:
        path = str(item.get("path", "")).strip() or "unknown_path"
        symbol_name = str(item.get("symbol_name", "")).strip() or str(item.get("section", "")).strip() or "unknown_symbol"
        lines.append(f"{len(lines) + 1}. `{path}` -> `{symbol_name}` (line: {_format_line_range(item)})")
    if not lines:
        lines = [
            f"1. 当前问题主要落在 `{module_name}`。",
            f"2. 当前模块说明：{module_hint or '--'}。",
            "3. 暂无可直接定位的代码证据，建议补充函数名、文件名或调用链关键词后重试。",
        ]
    return "\n".join(lines)


def _build_general_fallback(
    *,
    module_name: str,
    module_hint: str,
    question_type: str,
    evidence_hits: list[dict[str, Any]],
) -> str:
    points = _extract_points_from_hits(evidence_hits, max_points=6)
    if not points:
        return (
            f"当前问题主要落在 `{module_name}`。"
            f" 模块说明：{module_hint or '--'}。"
            " 当前没有命中足够证据，建议补充更具体的业务词、字段名或代码符号后重试。"
        )

    if question_type == "list":
        return "\n".join(f"{index}. {point}" for index, point in enumerate(points[:5], start=1))
    if question_type in {"reason", "formula"}:
        return "\n".join(f"{index}. {point}" for index, point in enumerate(points[:4], start=1))
    return "；".join(points[:3])


def _default_llm_call_status() -> dict[str, Any]:
    return {
        "status": "not_configured",
        "invoked": False,
        "request_sent": False,
        "attempts": 0,
        "latency_ms": 0,
        "reason": None,
        "model": None,
    }


def _is_calibration_query(service: Any, normalized_query: str) -> bool:
    terms = service.domain_profile.answering.calibration_terms
    if terms:
        return any(token in normalized_query for token in terms)
    return any(token in normalized_query for token in ("校准", "pctr", "pcvr", "ctr", "cvr"))


def _is_bid_entry_query(service: Any, normalized_query: str) -> bool:
    profile = service.domain_profile.answering
    bid_terms = profile.bid_terms or ("出价", "计费", "报价", "pricing", "bid")
    entry_terms = profile.bid_entry_terms or ("入口", "函数", "实现")
    has_bid = any(token in normalized_query for token in bid_terms)
    has_entry = any(token in normalized_query for token in entry_terms)
    return has_bid and has_entry


def _default_entry_symbol(service: Any) -> str:
    return service.domain_profile.answering.default_entry_symbol or "main_entry"


def _is_reason_query(normalized_query: str) -> bool:
    return any(token in normalized_query for token in ("原因", "为什么", "为何", "导致", "怎么会", "why"))


def _build_related_modules_block(related_modules: list[dict[str, Any]]) -> str:
    """
    构建相关模块展示文本。

    说明：
        最终给 LLM 的提示词仍然保留“主模块”概念，
        相关模块只作为辅助信息追加，帮助模型理解跨模块问题的上下文边界。
    """
    if not related_modules:
        return "- 无"

    rows: list[str] = []
    for item in related_modules[:3]:
        module_name = str(item.get("module_name", "")).strip()
        module_hint = str(item.get("module_hint", "")).strip() or "--"
        if not module_name:
            continue
        rows.append(f"- module_name: {module_name}")
        rows.append(f"  module_hint: {module_hint}")
    return "\n".join(rows) if rows else "- 无"


def _enforce_structured_output(answer: str, *, question_type: str) -> str:
    if not answer:
        return ""
    if question_type not in {"list", "reason", "formula"}:
        return answer

    has_list = bool(re.search(r"(?m)^\s*(?:\d+\.|[-*])\s+", answer))
    if has_list:
        return answer

    sentences = [segment.strip() for segment in re.split(r"[。\n,，!?！？]", answer) if segment.strip()]
    if not sentences:
        return answer

    top_sentences = sentences[:4]
    list_body = "\n".join(f"{index}. {text}" for index, text in enumerate(top_sentences, start=1))
    if question_type == "list":
        section_title = "要点清单"
    elif question_type == "reason":
        section_title = "主要原因"
    else:
        section_title = "公式与关键变量"
    return f"{answer}\n\n**{section_title}**\n{list_body}"


def _validate_qa_answer(answer: str) -> tuple[bool, str | None]:
    if not answer.strip():
        return False, "empty_answer"
    if looks_like_reasoning_dump(answer):
        return False, "empty_answer:reasoning_dump"
    return True, None


def _run_llm_for_qa(
    service: Any,
    *,
    user_query: str,
    question_type: str,
    module_name: str,
    module_hint: str,
    related_modules: list[dict[str, Any]],
    evidence_hits: list[dict[str, Any]],
) -> tuple[str | None, str | None, dict[str, Any]]:
    llm_client = getattr(service, "_llm_client", None)
    if llm_client is None:
        return None, None, _default_llm_call_status()

    system_prompt = resolve_system_prompt(
        env_key="WORKFLOW_QA_LLM_SYSTEM_PROMPT",
        default_prompt=QA_SYSTEM_PROMPT_TEMPLATE,
        domain_profile=getattr(service, "domain_profile", None),
    )

    request = CommonLLMRequest(
        node_name="knowledge_answer",
        system_prompt=system_prompt,
        user_prompt=QA_USER_PROMPT_TEMPLATE.format(
            user_query=user_query,
            module_name=module_name,
            module_hint=module_hint,
            related_modules_block=_build_related_modules_block(related_modules),
            evidence_block=build_evidence_block(evidence_hits),
        ),
        evidence_count=len(evidence_hits),
        require_evidence=True,
        log_namespace="workflow.llm_qa",
        metadata={
            "module_name": module_name,
            "question_type": question_type,
            "related_module_count": len(related_modules),
            "user_query_preview": user_query[:120],
        },
        normalize_answer=lambda text: _enforce_structured_output(text, question_type=question_type),
        validate_answer=_validate_qa_answer,
    )

    if hasattr(llm_client, "generate_with_status"):
        return llm_client.generate_with_status(request)

    if hasattr(llm_client, "generate_answer_with_status"):
        return llm_client.generate_answer_with_status(
            user_query=user_query,
            question_type=question_type,
            module_name=module_name,
            module_hint=module_hint,
            related_modules=related_modules,
            evidence_hits=evidence_hits,
        )

    answer_text, fallback_reason = llm_client.generate_answer(
        user_query=user_query,
        question_type=question_type,
        module_name=module_name,
        module_hint=module_hint,
        related_modules=related_modules,
        evidence_hits=evidence_hits,
    )
    status = getattr(llm_client, "last_call_status", {}) or {}
    call_status = dict(status) if isinstance(status, dict) else _default_llm_call_status()
    return answer_text, fallback_reason, call_status


def run(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    module_name = state["module_name"]
    module_hint = state["module_hint"]
    related_modules = list(state.get("related_modules", []) or [])
    user_query = str(state.get("user_query", ""))
    question_type = _infer_question_type(service, user_query)

    wiki_hits = list(state.get("wiki_hits", []))
    code_hits = list(state.get("code_hits", []))
    evidence_hits = collect_evidence_hits(state)

    llm_mode = "fallback_rule"
    llm_fallback_reason: str | None = None
    llm_call_status = _default_llm_call_status()
    final_answer: str

    llm_answer_text, llm_fallback_reason, llm_call_status = _run_llm_for_qa(
        service,
        user_query=user_query,
        question_type=question_type,
        module_name=module_name,
        module_hint=module_hint,
        related_modules=related_modules,
        evidence_hits=evidence_hits,
    )

    llm_client = getattr(service, "_llm_client", None)
    llm_call_status.setdefault("model", getattr(getattr(llm_client, "config", None), "model", None))
    llm_call_status.setdefault("invoked", bool(llm_client is not None))

    if llm_answer_text:
        if service.domain_profile.is_code_location_query(user_query) and code_hits and not _answer_mentions_code_anchor(llm_answer_text, code_hits):
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
        if service.domain_profile.is_code_location_query(user_query) and code_hits:
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
        "status": "completed",
        "answer": final_answer,
        "analysis": {
            "summary": "知识问答已完成",
            "module": module_name,
            "related_modules": related_modules,
            "confidence": "medium",
            "generation_mode": llm_mode,
            "question_type": question_type,
            "evidence_count": len(evidence_hits),
            "wiki_evidence_count": len(wiki_hits),
            "code_evidence_count": len(code_hits),
            "llm_enabled": bool(llm_client is not None and llm_client.config.enabled),
            "llm_available": bool(llm_client is not None and llm_client.is_available),
            "llm_model": (llm_client.config.model if llm_client is not None else None),
            "llm_fallback_reason": llm_fallback_reason,
            "llm_call_status": llm_call_status,
            "highlights": [
                "知识问答优先尝试使用 LLM 基于证据生成答案",
                "代码定位类问题会额外检查答案是否包含代码锚点",
                "最终 Markdown 三段式格式由 finalize_response 节点统一收口",
            ],
        },
        "node_trace": service._trace(
            state,
            "knowledge_answer",
            f"module={module_name}, related={len(related_modules)}",
        ),
    }
