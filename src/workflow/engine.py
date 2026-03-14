# -*- coding: utf-8 -*-
"""LangGraph workflow engine."""
from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from workflow.common.llm_client import WorkflowLLMClient
from workflow.common.node_trace import append_node_trace
from workflow.domain_profile import DomainProfile, get_domain_profile
from workflow.nodes.analysis.issue_analysis import run as issue_analysis_node
from workflow.nodes.analysis.knowledge_answer import run as knowledge_answer_node
from workflow.nodes.code_generation_flow.code_generation import run as code_generation_node
from workflow.nodes.code_generation_flow.load_code_context import run as load_code_context_node
from workflow.nodes.code_generation_flow.retrieve_code_context import run as retrieve_code_context_node
from workflow.nodes.control_response.finalize_response import run as finalize_response_node
from workflow.nodes.control_response.out_of_scope_response import run as out_of_scope_response_node
from workflow.nodes.retrieval_flow.merge_evidence import run as merge_evidence_node
from workflow.nodes.retrieval_flow.query_rewriter import run as query_rewriter_node
from workflow.nodes.retrieval_flow.retrieve_cases import run as retrieve_cases_node
from workflow.nodes.retrieval_flow.retrieve_code import run as retrieve_code_node
from workflow.nodes.retrieval_flow.retrieve_code.code_retriever import LocalCodeRetriever, parse_code_dirs_from_env
from workflow.nodes.retrieval_flow.retrieve_wiki import run as retrieve_wiki_node
from workflow.nodes.retrieval_flow.retrieve_wiki.wiki_retriever import MarkdownWikiRetriever
from workflow.nodes.routing_context.intent_routing import run as intent_routing_node
from workflow.nodes.routing_context.load_context import run as load_context_node
from workflow.runtime_logging import get_file_logger
from workflow.utils import env_bool

BACKEND_NAME = "langgraph"


class WorkflowState(TypedDict, total=False):
    """Shared state payload across graph nodes."""

    trace_id: str  # 当前请求链路的追踪 ID
    session_id: str  # 会话 ID，用于关联多轮对话
    user_query: str  # 用户本轮输入问题
    history: list[dict[str, Any]]  # 会话历史消息列表

    route: str  # 意图路由结果（走哪条分支）
    status: str  # 当前处理状态
    response_kind: str  # 最终响应类型
    domain_relevance: float  # 输入与领域的相关度分数

    module_name: str  # 当前识别到的模块名
    module_hint: str  # 模块提示信息（用于辅助路由与检索）
    active_topic_source: str  # 当前活跃主题的来源
    active_module_name: str  # 当前上下文中的活跃模块名
    last_analysis_result: dict[str, Any] | None  # 最近一次分析结果
    last_analysis_citations: list[dict[str, Any]]  # 最近一次分析引用

    retrieval_queries: list[str]  # 检索改写后查询词列表
    retrieval_plan: dict[str, Any]  # 检索计划（来源、策略等）
    wiki_hits: list[dict[str, Any]]  # wiki 检索命中文档
    wiki_retrieval_grade: str  # wiki 检索质量评级
    wiki_retrieval_profile: dict[str, Any]  # wiki 检索过程画像
    case_hits: list[dict[str, Any]]  # case 检索命中文档
    case_retrieval_grade: str  # case 检索质量评级
    case_retrieval_profile: dict[str, Any]  # case 检索过程画像
    code_hits: list[dict[str, Any]]  # code 检索命中文档
    code_retrieval_grade: str  # code 检索质量评级
    code_retrieval_profile: dict[str, Any]  # code 检索过程画像
    citations: list[dict[str, Any]]  # 融合后的引用列表
    evidence_fusion_profile: dict[str, Any]  # 证据融合过程画像

    analysis: dict[str, Any] | None  # 分析结果载荷
    answer: str  # 面向用户的最终回答文本
    node_trace: list[dict[str, str]]  # 节点执行轨迹
    assistant_message: dict[str, Any]  # 最终输出的助手消息结构


class WorkflowService:
    """Workflow orchestrator."""

    def __init__(self) -> None:
        self.backend_name = BACKEND_NAME
        self.debug_verbose_enabled = env_bool("WORKFLOW_DEBUG_VERBOSE", default=False)

        project_root = Path(__file__).resolve().parents[2]
        self._file_logger = get_file_logger(project_root=project_root)
        self.domain_profile: DomainProfile = get_domain_profile(project_root=project_root)

        wiki_dir = self.domain_profile.resolve_wiki_dir(project_root)
        self._wiki_retriever = MarkdownWikiRetriever(
            wiki_dir=wiki_dir,
            project_root=project_root,
            default_top_k=4,
            module_doc_hints=self.domain_profile.module_doc_hints(),
        )

        env_code_dirs = os.getenv("WORKFLOW_CODE_RETRIEVER_DIRS", "").strip()
        if env_code_dirs:
            code_dirs = parse_code_dirs_from_env(project_root=project_root)
        else:
            code_dirs = self.domain_profile.resolve_code_roots(project_root)

        self._code_retriever = LocalCodeRetriever(
            project_root=project_root,
            code_dirs=code_dirs,
            default_top_k=4,
        )
        self._knowledge_qa_llm = WorkflowLLMClient.from_env(prefix="WORKFLOW_QA_LLM")

        self._checkpointer = MemorySaver()
        self._graph = self._build_graph()

        self._file_logger.info(
            "workflow.service.initialized",
            backend=self.backend_name,
            domain_profile=self.domain_profile.profile_id,
            domain_display_name=self.domain_profile.display_name,
            debug_verbose_enabled=self.debug_verbose_enabled,
            logger_status=self._file_logger.status(),
            wiki_dir=str(wiki_dir),
            code_dirs=[str(path) for path in getattr(self._code_retriever, "code_dirs", [])],
            checkpointer_type=type(self._checkpointer).__name__,
        )

    def run_user_message(
        self,
        *,
        session_id: str,
        trace_id: str,
        user_query: str,
        history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        state: WorkflowState = {
            "trace_id": trace_id,
            "session_id": session_id,
            "user_query": user_query.strip(),
            "history": history,
            "node_trace": [],
        }
        return self._invoke(state)

    def _invoke(self, state: WorkflowState) -> dict[str, Any]:
        started_at = perf_counter()
        self._file_logger.info(
            "workflow.invoke.start",
            trace_id=state.get("trace_id", ""),
            session_id=state.get("session_id", ""),
            history_size=len(state.get("history", []) or []),
            user_query_preview=self._preview_text(state.get("user_query", ""), max_chars=120),
        )
        try:
            result = self._graph.invoke(state, config=self._invoke_config(state))
        except Exception as exc:
            latency_ms = int((perf_counter() - started_at) * 1000)
            self._file_logger.exception(
                "workflow.invoke.exception",
                trace_id=state.get("trace_id", ""),
                session_id=state.get("session_id", ""),
                latency_ms=latency_ms,
                error_type=type(exc).__name__,
            )
            raise

        assistant_message = dict(result["assistant_message"])
        latency_ms = int((perf_counter() - started_at) * 1000)

        debug_payload = assistant_message.get("debug")
        if isinstance(debug_payload, dict):
            debug_payload["latency_ms"] = latency_ms

        self._file_logger.info(
            "workflow.invoke.complete",
            trace_id=state.get("trace_id", ""),
            session_id=state.get("session_id", ""),
            latency_ms=latency_ms,
            response_kind=assistant_message.get("kind", "unknown"),
            response_status=assistant_message.get("status", "unknown"),
            actions=(assistant_message.get("actions") or []),
            citation_count=len(assistant_message.get("citations", []) or []),
            node_trace_count=len(result.get("node_trace", []) or []),
        )
        return assistant_message

    def _invoke_config(self, state: WorkflowState) -> dict[str, Any]:
        session_id = str(state.get("session_id", "") or "").strip() or "default_session"
        return {"configurable": {"thread_id": session_id}}

    def _build_graph(self) -> Any:
        graph = StateGraph(WorkflowState)

        graph.add_node("load_context", self._load_context)
        graph.add_node("intent_routing", self._intent_routing)

        graph.add_node("query_rewriter", self._query_rewriter)
        graph.add_node("retrieve_wiki", self._retrieve_wiki)
        graph.add_node("retrieve_cases", self._retrieve_cases)
        graph.add_node("retrieve_code", self._retrieve_code)
        graph.add_node("merge_evidence", self._merge_evidence)

        graph.add_node("knowledge_answer", self._knowledge_answer)
        graph.add_node("issue_analysis", self._issue_analysis)
        graph.add_node("out_of_scope_response", self._out_of_scope_response)

        graph.add_node("load_code_context", self._load_code_context)
        graph.add_node("retrieve_code_context", self._retrieve_code_context)
        graph.add_node("code_generation", self._code_generation)

        graph.add_node("finalize_response", self._finalize_response)

        graph.add_edge(START, "load_context")
        graph.add_edge("load_context", "intent_routing")

        graph.add_conditional_edges(
            "intent_routing",
            self._route_by_intent,
            {
                "knowledge_qa": "query_rewriter",
                "issue_analysis": "query_rewriter",
                "code_generation": "load_code_context",
                "out_of_scope": "out_of_scope_response",
            },
        )

        graph.add_edge("query_rewriter", "retrieve_wiki")
        graph.add_edge("retrieve_wiki", "retrieve_cases")
        graph.add_edge("retrieve_cases", "retrieve_code")
        graph.add_edge("retrieve_code", "merge_evidence")
        graph.add_conditional_edges(
            "merge_evidence",
            self._route_by_intent,
            {
                "knowledge_qa": "knowledge_answer",
                "issue_analysis": "issue_analysis",
                "code_generation": "load_code_context",
            },
        )

        graph.add_edge("knowledge_answer", "finalize_response")
        graph.add_edge("issue_analysis", "finalize_response")
        graph.add_edge("out_of_scope_response", "finalize_response")

        graph.add_edge("load_code_context", "retrieve_code_context")
        graph.add_edge("retrieve_code_context", "code_generation")
        graph.add_edge("code_generation", "finalize_response")

        graph.add_edge("finalize_response", END)
        return graph.compile(checkpointer=self._checkpointer)

    def _load_context(self, state: WorkflowState) -> dict[str, Any]:
        return self._run_node("load_context", load_context_node, state)

    def _intent_routing(self, state: WorkflowState) -> dict[str, Any]:
        return self._run_node("intent_routing", intent_routing_node, state)

    def _query_rewriter(self, state: WorkflowState) -> dict[str, Any]:
        return self._run_node("query_rewriter", query_rewriter_node, state)

    def _retrieve_wiki(self, state: WorkflowState) -> dict[str, Any]:
        return self._run_node("retrieve_wiki", retrieve_wiki_node, state)

    def _retrieve_cases(self, state: WorkflowState) -> dict[str, Any]:
        return self._run_node("retrieve_cases", retrieve_cases_node, state)

    def _retrieve_code(self, state: WorkflowState) -> dict[str, Any]:
        return self._run_node("retrieve_code", retrieve_code_node, state)

    def _merge_evidence(self, state: WorkflowState) -> dict[str, Any]:
        return self._run_node("merge_evidence", merge_evidence_node, state)

    def _knowledge_answer(self, state: WorkflowState) -> dict[str, Any]:
        return self._run_node("knowledge_answer", knowledge_answer_node, state)

    def _issue_analysis(self, state: WorkflowState) -> dict[str, Any]:
        return self._run_node("issue_analysis", issue_analysis_node, state)

    def _out_of_scope_response(self, state: WorkflowState) -> dict[str, Any]:
        return self._run_node("out_of_scope_response", out_of_scope_response_node, state)

    def _load_code_context(self, state: WorkflowState) -> dict[str, Any]:
        return self._run_node("load_code_context", load_code_context_node, state)

    def _retrieve_code_context(self, state: WorkflowState) -> dict[str, Any]:
        return self._run_node("retrieve_code_context", retrieve_code_context_node, state)

    def _code_generation(self, state: WorkflowState) -> dict[str, Any]:
        return self._run_node("code_generation", code_generation_node, state)

    def _finalize_response(self, state: WorkflowState) -> dict[str, Any]:
        return self._run_node("finalize_response", finalize_response_node, state)

    def _route_by_intent(self, state: WorkflowState) -> str:
        route = state["route"]
        self._file_logger.debug(
            "workflow.route.intent",
            trace_id=state.get("trace_id", ""),
            session_id=state.get("session_id", ""),
            route=route,
            citation_count=len(state.get("citations", []) or []),
        )
        return route

    def _preview_text(self, value: Any, *, max_chars: int = 80) -> str:
        text = str(value or "").strip()
        if len(text) <= max_chars:
            return text
        return f"{text[:max_chars]}..."

    def _summarize_node_updates(self, updates: dict[str, Any]) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        scalar_keys = (
            "route",
            "status",
            "response_kind",
            "module_name",
            "module_hint",
            "active_topic_source",
            "domain_relevance",
            "wiki_retrieval_grade",
            "case_retrieval_grade",
            "code_retrieval_grade",
        )
        for key in scalar_keys:
            if key not in updates:
                continue
            value = updates.get(key)
            summary[key] = self._preview_text(value, max_chars=120) if isinstance(value, str) else value

        if "retrieval_queries" in updates:
            summary["retrieval_query_count"] = len(updates.get("retrieval_queries", []) or [])
        if "wiki_hits" in updates:
            summary["wiki_hit_count"] = len(updates.get("wiki_hits", []) or [])
        if "case_hits" in updates:
            summary["case_hit_count"] = len(updates.get("case_hits", []) or [])
        if "code_hits" in updates:
            summary["code_hit_count"] = len(updates.get("code_hits", []) or [])
        if "citations" in updates:
            summary["citation_count"] = len(updates.get("citations", []) or [])
        if "node_trace" in updates:
            summary["node_trace_count"] = len(updates.get("node_trace", []) or [])
        if "assistant_message" in updates:
            assistant_message = dict(updates.get("assistant_message") or {})
            summary["assistant_kind"] = assistant_message.get("kind", "unknown")
            summary["assistant_status"] = assistant_message.get("status", "unknown")
            summary["assistant_action_count"] = len(assistant_message.get("actions", []) or [])
            summary["assistant_citation_count"] = len(assistant_message.get("citations", []) or [])
        return summary

    def _run_node(
        self,
        node_name: str,
        node_runner: Callable[[Any, dict[str, Any]], dict[str, Any]],
        state: WorkflowState,
    ) -> dict[str, Any]:
        started_at = perf_counter()
        self._file_logger.debug(
            "workflow.node.start",
            trace_id=state.get("trace_id", ""),
            session_id=state.get("session_id", ""),
            node=node_name,
            route=state.get("route", ""),
        )
        try:
            updates = node_runner(self, state)
        except Exception as exc:
            latency_ms = int((perf_counter() - started_at) * 1000)
            self._file_logger.exception(
                "workflow.node.exception",
                trace_id=state.get("trace_id", ""),
                session_id=state.get("session_id", ""),
                node=node_name,
                latency_ms=latency_ms,
                error_type=type(exc).__name__,
            )
            raise

        latency_ms = int((perf_counter() - started_at) * 1000)
        self._file_logger.info(
            "workflow.node.complete",
            trace_id=state.get("trace_id", ""),
            session_id=state.get("session_id", ""),
            node=node_name,
            latency_ms=latency_ms,
            updates=self._summarize_node_updates(updates),
        )
        return updates

    def runtime_log_status(self) -> dict[str, Any]:
        return self._file_logger.status()

    def _trace(self, state: WorkflowState, node: str, summary: str) -> list[dict[str, str]]:
        return append_node_trace(state, node, summary)
