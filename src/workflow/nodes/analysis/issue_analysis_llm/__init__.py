# -*- coding: utf-8 -*-
"""
该模块实现工作流节点`issue_analysis_llm` 的处理逻辑，负责读取状态并输出增量结果。
"""
from __future__ import annotations

import re
from typing import Any

from workflow.utils import normalize_source_type


_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*]\s+|\d+[.)、]\s+)(.+)$")

_SECTION_LABELS: dict[str, tuple[str, ...]] = {
    "root_cause": ("根因判断", "根因", "结论"),
    "risks": ("风险提示", "风险"),
    "fix_plan": ("修复建议", "修复方案", "处理建议", "建议"),
    "verification_steps": ("验证步骤", "验证", "回归验证"),
}


def _collect_evidence_hits(state: dict[str, Any]) -> list[dict[str, Any]]:
    """
    收集并标准化当前流程依赖的输入数据。
    
    参数:
        state: 工作流状态字典，包含会话上下文与中间结果。
    
    返回:
        返回类型为 `list[dict[str, Any]]` 的处理结果。
    """
    citations = list(state.get("citations", []) or [])
    if citations:
        rows: list[dict[str, Any]] = []
        for item in citations:
            row = dict(item)
            row["source_type"] = normalize_source_type(row.get("source_type", row.get("source", "")))
            rows.append(row)
        return rows

    hits: list[dict[str, Any]] = []
    for key, source_type in (("wiki_hits", "wiki"), ("code_hits", "code"), ("case_hits", "case")):
        for item in list(state.get(key, []) or []):
            row = dict(item)
            row["source_type"] = normalize_source_type(row.get("source_type", source_type))
            hits.append(row)
    return hits


def _trim_item(text: str) -> str:
    """
    内部辅助函数，负责`trim item` 相关处理。
    
    参数:
        text: 待处理的文本内容。
    
    返回:
        返回类型为 `str` 的处理结果。
    """
    value = str(text or "").strip()
    if not value:
        return ""
    match = _LIST_ITEM_RE.match(value)
    if match:
        value = match.group(1).strip()
    if value.startswith("#"):
        return ""
    return value.strip(" \t-*:：")


def _dedup_items(items: list[str], *, max_items: int) -> list[str]:
    """
    内部辅助函数，负责`dedup items` 相关处理。
    
    参数:
        items: 列表参数，用于承载批量输入数据。
    
    返回:
        返回类型为 `list[str]` 的处理结果。
    """
    seen: set[str] = set()
    result: list[str] = []
    for raw in items:
        value = _trim_item(raw)
        if not value:
            continue
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
        if len(result) >= max_items:
            break
    return result


def _match_section_heading(line: str) -> tuple[str | None, str]:
    """
    内部辅助函数，负责`match section heading` 相关处理。
    
    参数:
        line: 输入参数，用于控制当前处理逻辑。
    
    返回:
        返回类型为 `tuple[str | None, str]` 的处理结果。
    """
    text = str(line or "").strip()
    if not text:
        return None, ""
    normalized = _trim_item(text)
    if not normalized:
        return None, ""

    for section, labels in _SECTION_LABELS.items():
        for label in labels:
            if normalized.startswith(label):
                payload = normalized[len(label) :].lstrip(":： ").strip()
                return section, payload
    return None, normalized


def _parse_issue_analysis_answer(answer: str) -> dict[str, Any]:
    """
    解析输入内容并转换为结构化数据。
    
    参数:
        answer: 输入参数，用于控制当前处理逻辑。
    
    返回:
        返回类型为 `dict[str, Any]` 的处理结果。
    """
    root_cause = ""
    risks: list[str] = []
    fix_plan: list[str] = []
    verification_steps: list[str] = []
    fallback_items: list[str] = []

    current_section: str | None = None
    for raw_line in str(answer or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        matched_section, payload = _match_section_heading(line)
        if matched_section is not None:
            current_section = matched_section
            if payload:
                if matched_section == "root_cause" and not root_cause:
                    root_cause = payload
                elif matched_section == "risks":
                    risks.append(payload)
                elif matched_section == "fix_plan":
                    fix_plan.append(payload)
                elif matched_section == "verification_steps":
                    verification_steps.append(payload)
            continue

        item = _trim_item(payload)
        if not item:
            continue
        if current_section == "root_cause":
            if not root_cause:
                root_cause = item
            else:
                fallback_items.append(item)
            continue
        if current_section == "risks":
            risks.append(item)
            continue
        if current_section == "fix_plan":
            fix_plan.append(item)
            continue
        if current_section == "verification_steps":
            verification_steps.append(item)
            continue
        fallback_items.append(item)

    fallback_items = _dedup_items(fallback_items, max_items=8)
    risks = _dedup_items(risks, max_items=3)
    fix_plan = _dedup_items(fix_plan, max_items=4)
    verification_steps = _dedup_items(verification_steps, max_items=3)

    if not root_cause and fallback_items:
        root_cause = fallback_items[0]
    if not risks and len(fallback_items) > 1:
        risks = fallback_items[1:3]
    if not fix_plan and len(fallback_items) > 1:
        fix_plan = fallback_items[1:4]
    if not verification_steps and len(fallback_items) > 2:
        verification_steps = fallback_items[-2:]

    parsed: dict[str, Any] = {}
    if root_cause:
        parsed["root_cause"] = root_cause
    if risks:
        parsed["risks"] = risks
    if fix_plan:
        parsed["fix_plan"] = fix_plan
    if verification_steps:
        parsed["verification_steps"] = verification_steps
    return parsed


def _build_issue_analysis_prompt(user_query: str) -> str:
    """
    构建当前步骤所需的数据结构或文本内容。
    
    参数:
        user_query: 用户输入问题文本。
    
    返回:
        返回类型为 `str` 的处理结果。
    """
    base_query = str(user_query or "").strip()
    requirement = (
        "\n\n请严格按以下结构输出：\n"
        "根因判断：一句话\n"
        "风险提示：2条\n"
        "修复建议：3条\n"
        "验证步骤：2条\n"
        "只基于证据回答，不补充证据外事实。"
    )
    return f"{base_query}{requirement}"


def run(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """
    执行`issue_analysis_llm` 节点主流程，基于输入状态计算并返回状态增量。
    
    参数:
        service: 工作流服务对象，提供检索、路由、日志与配置能力。
        state: 工作流状态字典，包含会话上下文与中间结果。
    
    返回:
        返回类型为 `dict[str, Any]` 的处理结果。
    """
    module_name = state["module_name"]
    module_hint = state["module_hint"]
    user_query = str(state.get("user_query", ""))
    evidence_hits = _collect_evidence_hits(state)
    retrieval_queries = list(state.get("retrieval_queries", []) or [])

    analysis = dict(state.get("analysis") or {})
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

    if hasattr(service, "_knowledge_qa_llm") and service._knowledge_qa_llm is not None:
        llm_client = service._knowledge_qa_llm
        issue_query = _build_issue_analysis_prompt(user_query)
        llm_text, llm_fallback_reason = llm_client.generate_answer(
            user_query=issue_query,
            question_type="reason",
            module_name=module_name,
            module_hint=module_hint,
            retrieval_queries=retrieval_queries,
            evidence_hits=evidence_hits,
        )
        latest_status = getattr(llm_client, "last_call_status", {}) or {}
        if isinstance(latest_status, dict):
            llm_call_status = dict(latest_status)
        llm_call_status.setdefault("model", llm_client.config.model)
        llm_call_status.setdefault("invoked", True)

        if llm_text:
            parsed = _parse_issue_analysis_answer(llm_text)
            analysis.update(parsed)
            analysis["issue_analysis_llm_summary"] = llm_text
            llm_mode = "llm"

    analysis["issue_analysis_generation_mode"] = llm_mode
    analysis["issue_analysis_llm_fallback_reason"] = llm_fallback_reason
    analysis["llm_call_status"] = llm_call_status

    return {
        "analysis": analysis,
        "node_trace": service._trace(state, "issue_analysis_llm", f"mode={llm_mode}"),
    }
