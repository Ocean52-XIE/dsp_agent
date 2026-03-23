# -*- coding: utf-8 -*-
"""Unified retrieval tool exposed to the Deep Agent."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from domain_profile import get_domain_profile
from retrievers.orchestration.fusion import run as merge_evidence_run
from retrievers.orchestration.code_flow import execute_code_retrieval
from retrievers.code.retriever import get_code_retriever
from retrievers.orchestration.wiki_flow import execute_wiki_retrieval
from retrievers.wiki.retriever import get_wiki_retriever


class DomainRetrieveInput(BaseModel):
    """Input schema for the unified domain retrieval tool."""

    query: str = Field(..., min_length=1, description="User query or rewritten query to search.")
    intent: Literal["knowledge_qa", "issue_analysis"] = Field(
        default="knowledge_qa",
        description="Task intent so the retriever can apply the right bias.",
    )
    retrieval_bias: Literal["hybrid", "wiki_first", "code_first"] = Field(
        default="hybrid",
        description="Preferred retrieval strategy.",
    )
    module_name: str | None = Field(
        default=None,
        description="Optional primary module name inferred by the router skill.",
    )
    related_modules: list[str] | None = Field(
        default=None,
        description="Optional related modules that should be expanded into retrieval queries.",
    )
    top_k: int = Field(default=6, ge=1, le=10, description="Maximum number of evidence items to return.")


class DomainRetrieveOutput(BaseModel):
    """Normalized retrieval response returned to the agent."""

    domain_id: str
    intent: str
    normalized_query: str
    retrieval_bias: str
    module_name: str
    related_modules: list[str]
    evidence: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    debug: dict[str, Any]


@dataclass
class _TraceProxy:
    """Tiny proxy so we can reuse the existing merge implementation."""

    def _trace(
        self,
        state: dict[str, Any],
        node_name: str,
        detail: str,
    ) -> list[dict[str, str]]:
        return [{"node": node_name, "summary": detail}]


def _dedupe_queries(queries: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in queries:
        normalized = " ".join(str(value or "").strip().split())
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def _resolve_module_name(query: str, module_name: str | None) -> tuple[str, list[str]]:
    profile = get_domain_profile()
    explicit_module = str(module_name or "").strip()
    if explicit_module:
        related = profile.infer_related_modules(query, primary_module_name=explicit_module, limit=2)
        return explicit_module, [item["module_name"] for item in related]
    inferred_module, _ = profile.infer_module(query)
    related = profile.infer_related_modules(query, primary_module_name=inferred_module, limit=2)
    return inferred_module, [item["module_name"] for item in related]


def _build_retrieval_plan(*, bias: str, top_k: int, related_modules: list[str]) -> dict[str, Any]:
    profile = get_domain_profile()
    preset_name = bias if bias in {"hybrid", "wiki_first", "code_first"} else "hybrid"
    preset = profile.retrieval.preset(preset_name)
    final_top_k = max(1, min(int(top_k), 10))
    max_per_source = dict(profile.retrieval.max_per_source)
    if preset_name == "code_first":
        max_per_source["code"] = max(int(max_per_source.get("code", 4)), 5)
    if preset_name == "wiki_first":
        max_per_source["wiki"] = max(int(max_per_source.get("wiki", 4)), 5)

    return {
        "strategy": preset_name,
        "enable_wiki": bool(profile.retrieval.enable_wiki),
        "enable_code": bool(profile.retrieval.enable_code),
        "wiki_top_k": int(preset.get("wiki_top_k", 4)),
        "code_top_k": int(preset.get("code_top_k", 4)),
        "final_top_k": final_top_k,
        "source_weights": dict(profile.retrieval.source_weights),
        "max_per_source": max_per_source,
        "intent_profile": {
            "is_code_intent": preset_name == "code_first",
            "is_code_location": preset_name == "code_first",
            "is_wiki_intent": preset_name == "wiki_first",
            "is_issue_analysis": False,
            "related_module_count": len(related_modules),
        },
    }


def _build_state(payload: DomainRetrieveInput) -> dict[str, Any]:
    module_name, inferred_related_modules = _resolve_module_name(payload.query, payload.module_name)
    explicit_related_modules = [str(item).strip() for item in (payload.related_modules or []) if str(item).strip()]
    related_modules = list(dict.fromkeys([*explicit_related_modules, *inferred_related_modules]))
    module_hint = get_domain_profile().module_hint(module_name)

    retrieval_queries = _dedupe_queries(
        [
            payload.query,
            f"{module_name} {payload.query}".strip(),
            *[f"{item} {payload.query}".strip() for item in related_modules],
        ]
    )

    retrieval_plan = _build_retrieval_plan(
        bias=payload.retrieval_bias,
        top_k=payload.top_k,
        related_modules=related_modules,
    )
    retrieval_plan["intent_profile"]["is_issue_analysis"] = payload.intent == "issue_analysis"

    return {
        "trace_id": "deep_agent_retrieve",
        "session_id": "deep_agent_retrieve",
        "user_query": payload.query,
        "module_name": module_name,
        "module_hint": module_hint,
        "related_modules": [
            {"module_name": item, "module_hint": get_domain_profile().module_hint(item)}
            for item in related_modules
        ],
        "retrieval_queries": retrieval_queries,
        "retrieval_plan": retrieval_plan,
    }


def _run_domain_retrieve_from_payload(payload: DomainRetrieveInput) -> dict[str, Any]:
    started_at = perf_counter()
    state = _build_state(payload)
    profile = get_domain_profile()

    wiki_retriever = get_wiki_retriever()
    code_retriever = get_code_retriever()
    if wiki_retriever is None and code_retriever is None:
        raise RuntimeError("Retrievers are not initialized")

    if wiki_retriever is not None:
        state.update(execute_wiki_retrieval(wiki_retriever, state))
    else:
        state.update({"wiki_hits": [], "wiki_retrieval_grade": "disabled"})

    if code_retriever is not None:
        state.update(execute_code_retrieval(code_retriever, state))
    else:
        state.update({"code_hits": [], "code_retrieval_grade": "disabled"})

    merged = merge_evidence_run(_TraceProxy(), state)
    citations = list(merged.get("citations", []))

    response = DomainRetrieveOutput(
        domain_id=profile.profile_id,
        intent=payload.intent,
        normalized_query=" | ".join(state["retrieval_queries"]),
        retrieval_bias=str(state["retrieval_plan"]["strategy"]),
        module_name=str(state["module_name"]),
        related_modules=[item["module_name"] for item in state.get("related_modules", [])],
        evidence=citations,
        citations=[
            {
                "source": item.get("source", item.get("source_type", "")),
                "source_type": item.get("source_type", item.get("source", "")),
                "path": item.get("path", ""),
                "title": item.get("title", ""),
                "section": item.get("section", ""),
                "score": item.get("score", 0.0),
                "excerpt": item.get("excerpt", ""),
                "symbol_name": item.get("symbol_name", ""),
                "start_line": item.get("start_line"),
                "end_line": item.get("end_line"),
            }
            for item in citations
        ],
        debug={
            "domain_id": profile.profile_id,
            "intent": payload.intent,
            "module_name": str(state["module_name"]),
            "retrieval_strategy": str(state["retrieval_plan"]["strategy"]),
            "requested_top_k": int(payload.top_k),
            "returned_citation_count": len(citations),
            "query_count": len(state["retrieval_queries"]),
            "wiki_hits": len(state.get("wiki_hits", [])),
            "code_hits": len(state.get("code_hits", [])),
            "wiki_grade": state.get("wiki_retrieval_grade", "unknown"),
            "code_grade": state.get("code_retrieval_grade", "unknown"),
            "fusion_profile": merged.get("evidence_fusion_profile", {}),
            "latency_ms": int((perf_counter() - started_at) * 1000),
        },
    )
    return response.model_dump()


async def _run_domain_retrieve_async(
    query: str,
    intent: str = "knowledge_qa",
    retrieval_bias: str = "hybrid",
    module_name: str | None = None,
    related_modules: list[str] | None = None,
    top_k: int = 6,
) -> dict[str, Any]:
    payload = DomainRetrieveInput(
        query=query,
        intent=intent,
        retrieval_bias=retrieval_bias,
        module_name=module_name,
        related_modules=related_modules,
        top_k=top_k,
    )
    return await asyncio.to_thread(_run_domain_retrieve_from_payload, payload)


def create_domain_retrieve_tool() -> StructuredTool:
    """Create the unified retrieval tool used by the Deep Agent."""
    return StructuredTool.from_function(
        coroutine=_run_domain_retrieve_async,
        name="domain_retrieve",
        description=(
            "Unified retrieval tool for the current domain. "
            "Use it when you need wiki or code evidence before answering."
        ),
        args_schema=DomainRetrieveInput,
    )
