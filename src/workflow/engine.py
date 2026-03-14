# -*- coding: utf-8 -*-
"""LangGraph workflow engine."""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, TypedDict
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from bootstrap.postgres_bootstrap import ensure_database_exists
from workflow.common.func_utils import env_bool, to_bool, to_int
from workflow.llm.llm_client import WorkflowLLMClient
from workflow.common.domain_profile import DomainProfile, get_domain_profile
from workflow.common.node_trace import append_node_trace
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
from workflow.common.runtime_logging import get_file_logger

BACKEND_NAME = "langgraph"


def _ensure_connect_timeout_in_dsn(dsn: str, timeout_seconds: int) -> str:
    """
    为 PostgreSQL DSN 注入 connect_timeout 参数（若未显式配置）。

    说明：
    1. PostgresSaver 的构造仅接收 conn_string，因此需要在 DSN 层完成超时参数注入；
    2. 若用户已在 DSN 中显式配置 connect_timeout，则保持用户配置优先，不做覆盖。
    """
    normalized_dsn = str(dsn or "").strip()
    if not normalized_dsn:
        return normalized_dsn

    parsed = urlsplit(normalized_dsn)
    query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
    if any(str(key).lower() == "connect_timeout" for key, _ in query_pairs):
        return normalized_dsn

    query_pairs.append(("connect_timeout", str(max(1, int(timeout_seconds)))))
    updated_query = urlencode(query_pairs)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, updated_query, parsed.fragment))


@dataclass
class WorkflowCheckpointerConfig:
    """
    Workflow Checkpointer 配置。

    说明：
    1. backend 支持 `memory` 与 `postgres` 两种模式；
    2. 当未显式指定 backend 时，会根据是否配置了 PostgreSQL DSN 自动决策；
    3. PostgreSQL DSN 支持独立配置，也支持复用 session/observability 的 DSN。
    """

    backend: str
    pg_enabled: bool
    pg_dsn: str
    pg_setup: bool
    pg_connect_timeout_seconds: int

    @classmethod
    def from_env(cls) -> "WorkflowCheckpointerConfig":
        """
        从环境变量解析 checkpointer 配置。

        环境变量优先级：
        1. `WORKFLOW_CHECKPOINTER_PG_DSN`
        2. `WORKFLOW_SESSION_PG_DSN`
        3. `WORKFLOW_OBS_PG_DSN`
        """
        explicit_backend = str(os.getenv("WORKFLOW_CHECKPOINTER_BACKEND", "") or "").strip().lower()
        explicit_dsn = str(os.getenv("WORKFLOW_CHECKPOINTER_PG_DSN", "") or "").strip()
        fallback_dsn = (
            str(os.getenv("WORKFLOW_SESSION_PG_DSN", "") or "").strip()
            or str(os.getenv("WORKFLOW_OBS_PG_DSN", "") or "").strip()
        )
        resolved_dsn = explicit_dsn or fallback_dsn

        if explicit_backend in {"memory", "postgres"}:
            backend = explicit_backend
        else:
            backend = "postgres" if resolved_dsn else "memory"

        pg_enabled_default = bool(resolved_dsn) and backend == "postgres"
        connect_timeout_raw = (
            str(os.getenv("WORKFLOW_CHECKPOINTER_PG_CONNECT_TIMEOUT_SECONDS", "") or "").strip()
            or str(os.getenv("WORKFLOW_SESSION_PG_CONNECT_TIMEOUT_SECONDS", "") or "").strip()
            or str(os.getenv("WORKFLOW_OBS_PG_CONNECT_TIMEOUT_SECONDS", "") or "").strip()
        )

        connect_timeout_seconds = max(1, to_int(connect_timeout_raw, 5))
        dsn_with_timeout = _ensure_connect_timeout_in_dsn(resolved_dsn, connect_timeout_seconds)

        return cls(
            backend=backend,
            pg_enabled=to_bool(os.getenv("WORKFLOW_CHECKPOINTER_PG_ENABLED"), pg_enabled_default),
            pg_dsn=dsn_with_timeout,
            pg_setup=to_bool(os.getenv("WORKFLOW_CHECKPOINTER_PG_SETUP"), True),
            pg_connect_timeout_seconds=connect_timeout_seconds,
        )


class WorkflowState(TypedDict, total=False):
    """Shared state payload across graph nodes."""

    # Request/session context
    trace_id: str
    session_id: str
    user_query: str
    original_user_query: str
    query_rewrite_mode: str
    history: list[dict[str, Any]]

    # Routing context
    route: str
    status: str
    response_kind: str
    domain_relevance: float
    module_name: str
    module_hint: str
    related_modules: list[dict[str, str]]
    active_topic_source: str
    active_module_name: str
    last_analysis_result: dict[str, Any] | None
    last_analysis_citations: list[dict[str, Any]]

    # Retrieval context
    retrieval_queries: list[str]
    retrieval_plan: dict[str, Any]
    wiki_hits: list[dict[str, Any]]
    wiki_retrieval_grade: str
    wiki_retrieval_profile: dict[str, Any]
    case_hits: list[dict[str, Any]]
    case_retrieval_grade: str
    case_retrieval_profile: dict[str, Any]
    code_hits: list[dict[str, Any]]
    code_retrieval_grade: str
    code_retrieval_profile: dict[str, Any]
    citations: list[dict[str, Any]]
    evidence_fusion_profile: dict[str, Any]

    # Output context
    analysis: dict[str, Any] | None
    answer: str
    node_trace: list[dict[str, str]]
    assistant_message: dict[str, Any]


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
            embedding_profile=self.domain_profile.retrieval.embedding,
            hybrid_weights_profile=self.domain_profile.retrieval.hybrid_weights,
            module_prior_boost=self.domain_profile.retrieval.module_prior_boost,
            reranker_profile=self.domain_profile.retrieval.reranker,
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
            reranker_profile=self.domain_profile.retrieval.reranker,
        )
        self._llm_client = WorkflowLLMClient.from_env(prefix="WORKFLOW_QA_LLM")

        self._checkpointer_context: ExitStack | None = None
        self._checkpointer_status: dict[str, Any] = {}
        self._checkpointer_config = WorkflowCheckpointerConfig.from_env()
        self._checkpointer = self._init_checkpointer()
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
            checkpointer=self.checkpointer_status(),
        )

    def _memory_checkpointer_with_status(
        self,
        *,
        reason: str,
        init_error: str | None = None,
    ) -> MemorySaver:
        """
        构造内存 checkpointer，并同步写入状态快照。

        说明：
        1. 该方法用于统一“回退到内存”时的状态结构；
        2. 保证 health/日志里能直接看到回退原因，便于排障。
        """
        self._checkpointer_status = {
            "requested_backend": self._checkpointer_config.backend,
            "backend": "memory",
            "active": True,
            "dsn_configured": bool(self._checkpointer_config.pg_dsn),
            "pg_enabled": bool(self._checkpointer_config.pg_enabled),
            "fallback_reason": reason,
            "init_error": init_error,
        }
        return MemorySaver()

    def _init_checkpointer(self) -> Any:
        """
        初始化 workflow checkpointer。

        行为约定：
        1. 若配置为 `memory`，直接使用 MemorySaver；
        2. 若配置为 `postgres`，优先初始化 PostgresSaver；
        3. PostgreSQL 初始化失败时自动回退 MemorySaver，并记录结构化日志。
        """
        config = self._checkpointer_config

        if config.backend == "memory":
            self._file_logger.info("workflow.checkpointer.memory.enabled", reason="backend_memory")
            return self._memory_checkpointer_with_status(reason="backend_memory")

        if not config.pg_enabled:
            self._file_logger.info("workflow.checkpointer.memory.enabled", reason="pg_disabled")
            return self._memory_checkpointer_with_status(reason="pg_disabled")

        if not config.pg_dsn:
            self._file_logger.warning("workflow.checkpointer.postgres.disabled", reason="empty_dsn")
            return self._memory_checkpointer_with_status(reason="empty_dsn")

        try:
            import psycopg  # type: ignore
            from langgraph.checkpoint.postgres import PostgresSaver  # type: ignore
        except Exception as exc:  # pragma: no cover - 依赖缺失
            self._file_logger.warning(
                "workflow.checkpointer.postgres.init_failed",
                reason="import_dependency_failed",
                error_type=type(exc).__name__,
            )
            return self._memory_checkpointer_with_status(
                reason="import_dependency_failed",
                init_error=f"import_dependency_failed:{exc}",
            )

        stack: ExitStack | None = None
        try:
            ensure_database_exists(
                psycopg_module=psycopg,
                dsn=config.pg_dsn,
                connect_timeout_seconds=config.pg_connect_timeout_seconds,
            )

            stack = ExitStack()
            postgres_checkpointer = stack.enter_context(PostgresSaver.from_conn_string(config.pg_dsn))
            if config.pg_setup:
                postgres_checkpointer.setup()
            self._checkpointer_context = stack
            self._checkpointer_status = {
                "requested_backend": config.backend,
                "backend": "postgres",
                "active": True,
                "dsn_configured": True,
                "pg_enabled": True,
                "pg_setup": bool(config.pg_setup),
                "connect_timeout_seconds": int(config.pg_connect_timeout_seconds),
                "fallback_reason": None,
                "init_error": None,
            }
            self._file_logger.info(
                "workflow.checkpointer.postgres.ready",
                connect_timeout_seconds=config.pg_connect_timeout_seconds,
                setup_ran=bool(config.pg_setup),
            )
            return postgres_checkpointer
        except Exception as exc:  # pragma: no cover - 数据库异常
            if stack is not None:
                stack.close()
            self._file_logger.warning(
                "workflow.checkpointer.postgres.init_failed",
                reason="bootstrap_or_setup_failed",
                error_type=type(exc).__name__,
            )
            return self._memory_checkpointer_with_status(
                reason="bootstrap_or_setup_failed",
                init_error=f"bootstrap_or_setup_failed:{exc}",
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
            "query_rewrite_mode",
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

        if "user_query" in updates:
            summary["user_query_preview"] = self._preview_text(updates.get("user_query", ""), max_chars=120)
        if "original_user_query" in updates:
            summary["original_user_query_preview"] = self._preview_text(updates.get("original_user_query", ""), max_chars=120)
        if "retrieval_queries" in updates:
            summary["retrieval_query_count"] = len(updates.get("retrieval_queries", []) or [])
        if "related_modules" in updates:
            summary["related_module_count"] = len(updates.get("related_modules", []) or [])
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

    def checkpointer_status(self) -> dict[str, Any]:
        """
        返回 checkpointer 当前状态，供 health 与排障使用。
        """
        return dict(self._checkpointer_status)

    def _trace(self, state: WorkflowState, node: str, summary: str) -> list[dict[str, str]]:
        return append_node_trace(state, node, summary)

