# -*- coding: utf-8 -*-
"""Finalize response node."""
from __future__ import annotations

from typing import Any

from workflow.common.func_utils import normalize_source_type
from workflow.common.node_trace import append_node_trace


SCENE_LABELS = {
    "knowledge_qa": "知识问答",
    "issue_analysis": "问题分析",
    "code_generation": "代码实现建议",
    "out_of_scope": "领域外问题",
}


def _collect_evidence_hits(state: dict[str, Any]) -> list[dict[str, Any]]:
    citations = list(state.get("citations", []) or [])
    if citations:
        normalized_rows: list[dict[str, Any]] = []
        for item in citations:
            row = dict(item)
            row["source_type"] = normalize_source_type(row.get("source_type", row.get("source", "")))
            normalized_rows.append(row)
        return normalized_rows

    hits: list[dict[str, Any]] = []
    for key, source_type in (("wiki_hits", "wiki"), ("code_hits", "code"), ("case_hits", "case")):
        for item in list(state.get(key, []) or []):
            row = dict(item)
            row["source_type"] = normalize_source_type(row.get("source_type", source_type))
            hits.append(row)
    return hits


def _select_output_evidence_hits(all_hits: list[dict[str, Any]], *, max_items: int = 4) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in all_hits:
        key = (
            str(item.get("source_type", "")),
            str(item.get("path", "")),
            str(item.get("section", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        selected.append(item)
        if len(selected) >= max_items:
            break
    return selected


def _build_verbose_debug(state: dict[str, Any], graph_path: list[str], evidence_hits: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "module_name": str(state.get("module_name", "") or ""),
        "related_modules": list(state.get("related_modules", []) or []),
        "retrieval_queries": list(state.get("retrieval_queries", []) or []),
        "retrieval_plan": dict(state.get("retrieval_plan", {}) or {}),
        "retrieval_grades": {
            "wiki": str(state.get("wiki_retrieval_grade", "unknown") or "unknown"),
            "code": str(state.get("code_retrieval_grade", "unknown") or "unknown"),
            "case": str(state.get("case_retrieval_grade", "unknown") or "unknown"),
        },
        "retrieval_profiles": {
            "wiki": dict(state.get("wiki_retrieval_profile", {}) or {}),
            "code": dict(state.get("code_retrieval_profile", {}) or {}),
            "case": dict(state.get("case_retrieval_profile", {}) or {}),
            "fusion": dict(state.get("evidence_fusion_profile", {}) or {}),
        },
        "evidence_count": len(evidence_hits),
        "graph_path": graph_path,
        "node_trace": list(state.get("node_trace", []) or []),
    }


def _default_llm_call_status(service: Any, *, response_route: str) -> dict[str, Any]:
    llm_model = None
    if hasattr(service, "_llm_client") and service._llm_client is not None:
        llm_model = service._llm_client.config.model
    reason = "no_llm_in_route" if response_route != "knowledge_qa" else "llm_status_missing"
    return {
        "status": "not_invoked",
        "invoked": False,
        "request_sent": False,
        "attempts": 0,
        "latency_ms": 0,
        "reason": reason,
        "model": llm_model,
    }


def _passthrough_content(state: dict[str, Any]) -> str:
    raw_answer = state.get("answer")
    if isinstance(raw_answer, str):
        return raw_answer
    if raw_answer is None:
        return ""
    return str(raw_answer)


def run(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    graph_path = [item["node"] for item in state.get("node_trace", [])]
    debug_enabled = bool(getattr(service, "debug_verbose_enabled", False))
    response_kind = state.get("response_kind", state.get("route", "unknown"))
    response_route = state.get("route", "unknown")

    analysis_payload = state.get("analysis")
    analysis_output: dict[str, Any] | Any = analysis_payload
    if isinstance(analysis_payload, dict):
        analysis_output = dict(analysis_payload)
        if not debug_enabled:
            analysis_output.pop("llm_call_status", None)

    is_out_of_scope = (
        response_kind == "out_of_scope"
        or response_route == "out_of_scope"
        or state.get("status") == "out_of_scope"
    )
    all_evidence_hits = [] if is_out_of_scope else _collect_evidence_hits(state)
    evidence_hits = _select_output_evidence_hits(all_evidence_hits, max_items=4)

    assistant_message: dict[str, Any] = {
        "role": "assistant",
        "kind": response_kind,
        "intent": response_route,
        "status": state.get("status", "completed"),
        "content": _passthrough_content(state),
        "trace_id": state["trace_id"],
        "citations": evidence_hits,
        "analysis": analysis_output,
        "actions": [],
    }

    if debug_enabled:
        debug_analysis_payload = analysis_payload or {}
        llm_call_status = None
        if isinstance(debug_analysis_payload, dict):
            llm_call_status = debug_analysis_payload.get("llm_call_status")
        if not isinstance(llm_call_status, dict) or not llm_call_status:
            llm_call_status = _default_llm_call_status(service, response_route=response_route)
        assistant_message["debug"] = {
            "scene_label": SCENE_LABELS.get(assistant_message["kind"], assistant_message["kind"]),
            "domain_relevance": state.get("domain_relevance", 0.0),
            "latency_ms": 0,
            "route": state.get("route", "unknown"),
            "graph_backend": service.backend_name,
            "graph_path": graph_path,
            "verbose": _build_verbose_debug(state, graph_path, evidence_hits),
            "llm_call_status": llm_call_status,
        }

    return {
        "assistant_message": assistant_message,
        "node_trace": append_node_trace(state, "finalize_response", f"kind={assistant_message['kind']}"),
    }
