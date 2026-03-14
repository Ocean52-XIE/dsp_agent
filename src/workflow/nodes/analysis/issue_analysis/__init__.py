# -*- coding: utf-8 -*-
"""Unified issue-analysis node."""
from __future__ import annotations

import re
from typing import Any

from workflow.common.evidence import collect_evidence_hits
from workflow.common.llm_client import CommonLLMRequest
from workflow.common.llm_prompt_utils import build_evidence_block, looks_like_reasoning_dump, resolve_system_prompt
from workflow.common.node_trace import append_node_trace


ISSUE_SYSTEM_PROMPT_TEMPLATE = (
    "你是企业问题分析助手。"
    "必须严格基于提供的证据回答，不补充证据外事实。"
    "输出中文，优先给出结构化结果。"
)

ISSUE_USER_PROMPT_TEMPLATE = """【待分析问题】
{user_query}

【当前模块】
- module_name: {module_name}
- module_hint: {module_hint}

【检索语句】
{retrieval_queries}

【检索证据（按相关性排序）】
{evidence_block}

请严格按以下结构输出：
根因判断：一句话
风险提示：2-3条
修复建议：2-4条
验证步骤：2-3条
"""

_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*]\s+|\d+[.)、]\s+)(.+)$")
_SECTION_HINTS = {
    "root_cause": ("根因判断", "根因", "结论"),
    "risks": ("风险提示", "风险"),
    "fix_plan": ("修复建议", "修复方案", "处理建议", "建议"),
    "verification_steps": ("验证步骤", "验证", "回归验证"),
}


def _strip_item(text: str) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    match = _LIST_ITEM_RE.match(value)
    if match:
        value = match.group(1).strip()
    if value.startswith("#"):
        return ""
    return value.strip(" \t-*：:")


def _dedup(items: list[str], max_items: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in items:
        value = _strip_item(raw)
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
        if len(out) >= max_items:
            break
    return out


def _match_heading(line: str) -> tuple[str | None, str]:
    normalized = _strip_item(line)
    if not normalized:
        return None, ""
    for section, labels in _SECTION_HINTS.items():
        for label in labels:
            if normalized.startswith(label):
                payload = normalized[len(label) :].lstrip("：:").strip()
                return section, payload
    return None, normalized


def _parse_llm_answer(answer: str) -> dict[str, Any]:
    root_cause = ""
    risks: list[str] = []
    fix_plan: list[str] = []
    verification_steps: list[str] = []
    fallback: list[str] = []

    current: str | None = None
    for raw in str(answer or "").splitlines():
        line = raw.strip()
        if not line:
            continue

        section, payload = _match_heading(line)
        if section is not None:
            current = section
            if payload:
                if section == "root_cause" and not root_cause:
                    root_cause = payload
                elif section == "risks":
                    risks.append(payload)
                elif section == "fix_plan":
                    fix_plan.append(payload)
                elif section == "verification_steps":
                    verification_steps.append(payload)
            continue

        item = _strip_item(line)
        if not item:
            continue
        if current == "root_cause":
            if not root_cause:
                root_cause = item
            else:
                fallback.append(item)
        elif current == "risks":
            risks.append(item)
        elif current == "fix_plan":
            fix_plan.append(item)
        elif current == "verification_steps":
            verification_steps.append(item)
        else:
            fallback.append(item)

    fallback = _dedup(fallback, 8)
    risks = _dedup(risks, 3)
    fix_plan = _dedup(fix_plan, 4)
    verification_steps = _dedup(verification_steps, 3)

    if not root_cause and fallback:
        root_cause = fallback[0]
    if not risks and len(fallback) > 1:
        risks = fallback[1:3]
    if not fix_plan and len(fallback) > 1:
        fix_plan = fallback[1:4]
    if not verification_steps and len(fallback) > 2:
        verification_steps = fallback[-2:]

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


def _validate_issue_answer(answer: str) -> tuple[bool, str | None]:
    if not answer.strip():
        return False, "empty_answer"
    if looks_like_reasoning_dump(answer):
        return False, "empty_answer:reasoning_dump"
    return True, None


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


def _run_llm_if_available(service: Any, state: dict[str, Any], analysis: dict[str, Any]) -> None:
    module_name = str(state.get("module_name", "") or "")
    module_hint = str(state.get("module_hint", "") or "")
    user_query = str(state.get("user_query", "") or "")
    retrieval_queries = list(state.get("retrieval_queries", []) or [])
    evidence_hits = collect_evidence_hits(state)

    llm_mode = "fallback_rule"
    llm_fallback_reason: str | None = None
    llm_call_status = _default_llm_call_status()

    llm_client = getattr(service, "_knowledge_qa_llm", None)
    if llm_client is not None:
        system_prompt = resolve_system_prompt(
            env_key="WORKFLOW_ISSUE_LLM_SYSTEM_PROMPT",
            default_prompt=ISSUE_SYSTEM_PROMPT_TEMPLATE,
            domain_profile=getattr(service, "domain_profile", None),
        )
        retrieval_text = "\n".join(f"- {item}" for item in retrieval_queries) if retrieval_queries else "- none"
        request = CommonLLMRequest(
            node_name="issue_analysis",
            system_prompt=system_prompt,
            user_prompt=ISSUE_USER_PROMPT_TEMPLATE.format(
                user_query=user_query,
                module_name=module_name,
                module_hint=module_hint,
                retrieval_queries=retrieval_text,
                evidence_block=build_evidence_block(evidence_hits),
            ),
            evidence_count=len(evidence_hits),
            require_evidence=True,
            log_namespace="workflow.llm_issue_analysis",
            metadata={
                "module_name": module_name,
                "retrieval_query_count": len(retrieval_queries),
                "user_query_preview": user_query[:120],
            },
            normalize_answer=lambda text: text.strip(),
            validate_answer=_validate_issue_answer,
        )

        if hasattr(llm_client, "generate_with_status"):
            llm_text, llm_fallback_reason, llm_call_status = llm_client.generate_with_status(request)
        elif hasattr(llm_client, "generate_issue_analysis_with_status"):
            llm_text, llm_fallback_reason, llm_call_status = llm_client.generate_issue_analysis_with_status(
                user_query=user_query,
                module_name=module_name,
                module_hint=module_hint,
                retrieval_queries=retrieval_queries,
                evidence_hits=evidence_hits,
            )
        else:
            llm_text, llm_fallback_reason = llm_client.generate_issue_analysis(
                user_query=user_query,
                module_name=module_name,
                module_hint=module_hint,
                retrieval_queries=retrieval_queries,
                evidence_hits=evidence_hits,
            )
            latest_status = getattr(llm_client, "last_call_status", {}) or {}
            if isinstance(latest_status, dict):
                llm_call_status = dict(latest_status)

        llm_call_status.setdefault("invoked", True)
        model_name = getattr(getattr(llm_client, "config", None), "model", None)
        if model_name and not llm_call_status.get("model"):
            llm_call_status["model"] = model_name

        if llm_text:
            analysis.update(_parse_llm_answer(llm_text))
            analysis["issue_analysis_llm_summary"] = llm_text
            llm_mode = "llm"

    analysis["issue_analysis_generation_mode"] = llm_mode
    analysis["issue_analysis_llm_fallback_reason"] = llm_fallback_reason
    analysis["llm_call_status"] = llm_call_status


def _apply_defaults(analysis: dict[str, Any]) -> None:
    analysis.setdefault("root_cause", "高概率是边界输入或重复回调场景未做保护，导致状态被错误覆盖。")
    analysis.setdefault(
        "risks",
        [
            "可能影响已有重试任务或补偿逻辑。",
            "需要确认上游调用方是否依赖旧的异常行为。",
        ],
    )
    analysis.setdefault(
        "fix_plan",
        [
            "在入口层增加空值和非法值校验。",
            "为状态变更逻辑增加幂等保护。",
            "补充与历史案例一致的回归测试。",
        ],
    )
    analysis.setdefault(
        "verification_steps",
        [
            "复现原始问题路径并验证错误不再出现。",
            "检查关键日志字段、状态转换和重试行为。",
        ],
    )
    analysis["need_user_confirmation"] = False


def run(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    module_name = str(state.get("module_name", "") or "")
    analysis = dict(state.get("analysis") or {})
    analysis.update(
        {
            "module": module_name,
            "confidence": "high" if float(state.get("domain_relevance", 0.0) or 0.0) >= 0.75 else "medium",
        }
    )

    _run_llm_if_available(service, state, analysis)
    _apply_defaults(analysis)

    root_cause = str(analysis.get("root_cause", "") or "").strip() or "--"
    fix_plan = [str(item) for item in list(analysis.get("fix_plan", []) or [])]
    verification_steps = [str(item) for item in list(analysis.get("verification_steps", []) or [])]
    fix_plan_text = "\n".join(f"{i}. {item}" for i, item in enumerate(fix_plan[:4], start=1)) or "1. --"
    verification_text = "\n".join(f"{i}. {item}" for i, item in enumerate(verification_steps[:3], start=1)) or "1. --"

    generation_mode = str(analysis.get("issue_analysis_generation_mode", "fallback_rule") or "fallback_rule")
    mode_line = "本轮结合 LLM 与证据完成分析。" if generation_mode == "llm" else "本轮使用规则策略完成分析。"

    return {
        "response_kind": "issue_analysis",
        "status": "completed",
        "answer": (
            f"本轮输入已进入问题分析链路。当前更可能定位在 `{module_name}`。\n"
            f"{mode_line}\n"
            f"根因判断：{root_cause}\n\n"
            f"修复建议：\n{fix_plan_text}\n\n"
            f"验证步骤：\n{verification_text}"
        ),
        "analysis": analysis,
        "node_trace": append_node_trace(
            state,
            "issue_analysis",
            f"module={module_name}, generation={generation_mode}",
        ),
    }
