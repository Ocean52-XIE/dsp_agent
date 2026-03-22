# -*- coding: utf-8 -*-
"""LangGraph workflow engine.

主工作流负责：
1. 上下文加载（load_context）
2. 意图路由（intent_routing）
3. 调用子图（knowledge_qa / issue_analysis）
4. 代码生成（code_generation）
5. 响应收口（finalize_response）

检索节点已内聚到子图中，主工作流不再直接调用检索节点。

同步 Checkpointer 支持：
- MemorySaver：原生支持 invoke，无需额外处理
- PostgresSaver：使用 psycopg 同步连接，需要同步初始化
"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Callable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from workflow.common.func_utils import env_bool, to_bool, to_int
from workflow.common.node_trace import append_node_trace
from workflow.nodes.code_generation_flow.code_generation import run as code_generation_node
from workflow.nodes.code_generation_flow.load_code_context import run as load_code_context_node
from workflow.nodes.code_generation_flow.retrieve_code_context import run as retrieve_code_context_node
from workflow.nodes.control_response.finalize_response import run as finalize_response_node
from workflow.nodes.control_response.out_of_scope_response import run as out_of_scope_response_node
from workflow.nodes.routing_context.intent_routing import run as intent_routing_node
from workflow.nodes.routing_context.load_context import run as load_context_node
from log.runtime_logging import get_file_logger
from workflow.state import WorkflowState, create_initial_state


# ============================================================================
# PostgreSQL Bootstrap 工具函数（本地实现，避免依赖 bootstrap 模块）
# ============================================================================

def _extract_db_name_from_dsn(dsn: str) -> str:
    """从 DSN 中提取数据库名称"""
    parsed = urlsplit(dsn)
    db_name = (parsed.path or "").lstrip("/")
    return db_name


def _build_bootstrap_dsn(target_dsn: str, bootstrap_db: str) -> str:
    """构建 bootstrap 数据库的 DSN"""
    parsed = urlsplit(target_dsn)
    bootstrap_path = f"/{bootstrap_db.strip()}"
    return urlunsplit((parsed.scheme, parsed.netloc, bootstrap_path, parsed.query, parsed.fragment))


def _ensure_database_exists(
    *,
    psycopg_module: Any,
    dsn: str,
    connect_timeout_seconds: int,
) -> None:
    """确保目标数据库存在，不存在则创建

    该函数负责"数据库级别"的初始化（CREATE DATABASE）。
    """
    normalized_dsn = (dsn or "").strip()
    if not normalized_dsn:
        raise ValueError("empty_dsn")

    target_db = _extract_db_name_from_dsn(normalized_dsn)
    if not target_db:
        raise ValueError("invalid_dsn_missing_db_name")

    bootstrap_db = os.getenv("WORKFLOW_PG_BOOTSTRAP_DB", "postgres").strip() or "postgres"
    bootstrap_dsn = _build_bootstrap_dsn(normalized_dsn, bootstrap_db)

    # 使用 bootstrap 库检查并创建目标数据库
    with psycopg_module.connect(
        bootstrap_dsn,
        autocommit=True,
        connect_timeout=max(1, int(connect_timeout_seconds)),
    ) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname=%s", (target_db,))
            exists = cur.fetchone() is not None
            if exists:
                return

            sql_builder = getattr(psycopg_module, "sql", None)
            if sql_builder is None:
                raise RuntimeError("psycopg_sql_builder_unavailable")
            cur.execute(
                sql_builder.SQL("CREATE DATABASE {}").format(sql_builder.Identifier(target_db)),
            )


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


class WorkflowService:
    """Workflow orchestrator.

    所有依赖组件从全局单例获取，在 main.py 启动时已通过 init 模块初始化。
    Checkpointer 由 WorkflowService 内部管理，在构造时自动初始化。
    """

    # 图后端名称，用于标识当前使用的图框架
    backend_name: str = "langgraph"

    def __init__(self) -> None:
        self.debug_verbose_enabled = env_bool("WORKFLOW_DEBUG_VERBOSE", default=False)

        project_root = Path(__file__).resolve().parents[2]
        self._file_logger = get_file_logger(project_root=project_root)

        # 从全局单例获取组件
        from domain_profile import get_domain_profile

        self.domain_profile = get_domain_profile()

        # Checkpointer 相关属性初始化
        self._checkpointer: Any = None
        self._checkpointer_context: ExitStack | None = None
        self._checkpointer_status: dict[str, Any] = {}
        self._checkpointer_config = WorkflowCheckpointerConfig.from_env()

        # 初始化 checkpointer（内部管理）
        self._init_checkpointer()

        # 初始化子图（子图内部从全局单例获取 retriever）
        self._knowledge_qa_subgraph = None
        self._issue_analysis_subgraph = None
        self._init_subgraphs()

        # 构建图（传入已初始化的 checkpointer）
        self._graph = None
        self._graph = self._build_graph(checkpointer=self._checkpointer)

        # 输出初始化日志
        self._file_logger.info(
            "workflow.service.initialized",
            domain_profile=self.domain_profile.profile_id if self.domain_profile else "unknown",
            domain_display_name=self.domain_profile.display_name if self.domain_profile else "unknown",
            debug_verbose_enabled=self.debug_verbose_enabled,
            logger_status=self._file_logger.status(),
            checkpointer_type=type(self._checkpointer).__name__ if self._checkpointer else "None",
            checkpointer=self.checkpointer_status(),
            graph_built=self._graph is not None,
        )

    def _init_subgraphs(self) -> None:
        """初始化子图（启动时初始化）

        说明：
        1. 直接创建子图实例，无需注册中心；
        2. 子图从全局单例获取所有组件（LLMClient, SkillRegistry, ToolRegistry, DomainProfile, Retriever）；
        3. 不再传入 retriever 引用，子图内部直接从全局单例获取。
        """
        from workflow.subgraph.knowledge_qa import KnowledgeQAConfig, KnowledgeQASubgraph
        from workflow.subgraph.issue_analysis import IssueAnalysisConfig, IssueAnalysisSubgraph

        # 创建 knowledge_qa 子图（子图内部从全局单例获取 retriever）
        try:
            self._knowledge_qa_subgraph = KnowledgeQASubgraph(
                config=KnowledgeQAConfig(),
            )
            self._file_logger.info(
                "workflow.subgraph.initialized",
                subgraph_id="knowledge_qa"
            )
        except Exception as exc:
            self._file_logger.warning(
                "workflow.subgraph.init_failed",
                subgraph_id="knowledge_qa",
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

        # 创建 issue_analysis 子图（子图内部从全局单例获取 retriever）
        try:
            self._issue_analysis_subgraph = IssueAnalysisSubgraph(
                config=IssueAnalysisConfig(),
            )
            self._file_logger.info(
                "workflow.subgraph.initialized",
                subgraph_id="issue_analysis"
            )
        except Exception as exc:
            self._file_logger.warning(
                "workflow.subgraph.init_failed",
                subgraph_id="issue_analysis",
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

    def _memory_checkpointer_with_status(
        self,
        *,
        reason: str,
        init_error: str | None = None,
    ) -> MemorySaver:
        """构造内存 checkpointer，并同步写入状态快照。

        说明：
        1. 该方法用于统一"回退到内存"时的状态结构；
        2. 保证 health/日志里能直接看到回退原因，便于排障。
        3. MemorySaver 原生支持 invoke，无需额外处理。
        """
        self._checkpointer_status = {
            "requested_backend": self._checkpointer_config.backend,
            "backend": "memory",
            "active": True,
            "sync_supported": True,  # MemorySaver 支持 invoke
            "dsn_configured": bool(self._checkpointer_config.pg_dsn),
            "pg_enabled": bool(self._checkpointer_config.pg_enabled),
            "fallback_reason": reason,
            "init_error": init_error,
        }
        return MemorySaver()

    def _init_checkpointer(self) -> None:
        """初始化 workflow checkpointer（同步版本）。

        行为约定：
        1. 若配置为 `memory`，直接创建 MemorySaver；
        2. 若配置为 `postgres`，尝试初始化 PostgresSaver，失败则回退到 MemorySaver；
        3. 此方法在构造函数中调用，checkpointer 初始化完成后立即可用。
        """
        config = self._checkpointer_config

        # memory 模式
        if config.backend == "memory":
            self._checkpointer = self._memory_checkpointer_with_status(reason="backend_memory")
            self._file_logger.info("workflow.checkpointer.memory.enabled", reason="backend_memory")
            return

        # postgres 模式：尝试初始化 PostgresSaver
        if not config.pg_enabled:
            self._checkpointer = self._memory_checkpointer_with_status(reason="pg_disabled")
            self._file_logger.info("workflow.checkpointer.memory.enabled", reason="pg_disabled")
            return

        if not config.pg_dsn:
            self._checkpointer = self._memory_checkpointer_with_status(reason="empty_dsn")
            self._file_logger.warning("workflow.checkpointer.memory.enabled", reason="empty_dsn")
            return

        try:
            import psycopg  # type: ignore
            from langgraph.checkpoint.postgres import PostgresSaver  # type: ignore
        except Exception as exc:  # pragma: no cover - 依赖缺失
            self._checkpointer = self._memory_checkpointer_with_status(
                reason="import_dependency_failed",
                init_error=str(exc),
            )
            self._file_logger.warning(
                "workflow.checkpointer.postgres.import_failed",
                error_type=type(exc).__name__,
            )
            return

        try:
            # 确保数据库存在（同步操作）
            _ensure_database_exists(
                psycopg_module=psycopg,
                dsn=config.pg_dsn,
                connect_timeout_seconds=config.pg_connect_timeout_seconds,
            )

            # 创建同步上下文管理器
            self._checkpointer_context = ExitStack()

            # 使用 from_conn_string 创建 PostgresSaver（同步）
            checkpointer_cm = PostgresSaver.from_conn_string(config.pg_dsn)
            postgres_checkpointer = self._checkpointer_context.enter_context(checkpointer_cm)

            # 同步 setup
            if config.pg_setup:
                postgres_checkpointer.setup()

            # 设置 checkpointer
            self._checkpointer = postgres_checkpointer
            self._checkpointer_status = {
                "requested_backend": config.backend,
                "backend": "postgres",
                "active": True,
                "sync_supported": True,
                "dsn_configured": True,
                "pg_enabled": True,
                "pg_setup": bool(config.pg_setup),
                "connect_timeout_seconds": int(config.pg_connect_timeout_seconds),
                "fallback_reason": None,
                "init_error": None,
            }

            self._file_logger.info(
                "workflow.checkpointer.postgres.enabled",
                connect_timeout_seconds=config.pg_connect_timeout_seconds,
                setup_ran=bool(config.pg_setup),
            )

        except Exception as exc:  # pragma: no cover - 数据库异常
            if hasattr(self, "_checkpointer_context") and self._checkpointer_context:
                self._checkpointer_context.close()
                self._checkpointer_context = None

            self._checkpointer = self._memory_checkpointer_with_status(
                reason="postgres_init_failed",
                init_error=str(exc),
            )
            self._file_logger.warning(
                "workflow.checkpointer.postgres.init_failed",
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

    def close(self) -> None:
        """关闭资源。

        此方法应在应用关闭时调用。
        """
        if self._checkpointer_context:
            try:
                self._checkpointer_context.close()
                self._file_logger.info("workflow.checkpointer.close.complete")
            except Exception as exc:  # pragma: no cover
                self._file_logger.warning(
                    "workflow.checkpointer.close.failed",
                    error_type=type(exc).__name__,
                )
            finally:
                self._checkpointer_context = None

    def run_user_message(
        self,
        *,
        session_id: str,
        trace_id: str,
        user_query: str,
        history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """处理用户消息（同步版本）

        Args:
            session_id: 会话 ID
            trace_id: 追踪 ID
            user_query: 用户查询
            history: 对话历史

        Returns:
            助手响应消息
        """
        state: WorkflowState = {
            "trace_id": trace_id,
            "session_id": session_id,
            "user_query": user_query.strip(),
            "history": history,
            "node_trace": [],
        }
        return self._invoke(state)

    def _invoke(self, state: WorkflowState) -> dict[str, Any]:
        """执行工作流（同步版本）

        Args:
            state: 工作流状态

        Returns:
            助手响应消息

        Note:
            使用原生 invoke()，支持同步 checkpointer：
            - MemorySaver: 原生支持 invoke
            - PostgresSaver: 原生支持 invoke
        """
        started_at = perf_counter()
        self._file_logger.info(
            "workflow.invoke.start",
            trace_id=state.get("trace_id", ""),
            session_id=state.get("session_id", ""),
            history_size=len(state.get("history", []) or []),
            user_query_preview=self._preview_text(state.get("user_query", ""), max_chars=120),
        )
        try:
            # 使用原生 invoke，MemorySaver 和 PostgresSaver 都支持
            if self._graph is None:
                raise RuntimeError("Workflow graph not initialized")
            result = self._graph.invoke(
                state,
                config=self._invoke_config(state),
            )
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

    def _build_graph(self, checkpointer: Any = None) -> Any:
        """构建主工作流图

        主工作流结构：
        START -> load_context -> intent_routing -> [分支]
          - knowledge_qa: 调用 knowledge_qa 子图（内部包含检索和生成）
          - issue_analysis: 调用 issue_analysis 子图（内部包含检索和生成）
          - code_generation: load_code_context -> retrieve_code_context -> code_generation
          - out_of_scope: out_of_scope_response
        [所有分支] -> finalize_response -> END

        Returns:
            编译后的 LangGraph 图
        """
        # 使用传入的 checkpointer（如果提供），否则使用实例的
        effective_checkpointer = checkpointer if checkpointer is not None else self._checkpointer
        graph = StateGraph(WorkflowState)

        # 主流程节点
        graph.add_node("load_context", self._load_context)
        graph.add_node("intent_routing", self._intent_routing)

        # 子图节点（内部包含完整的检索和生成流程）
        graph.add_node("knowledge_answer", self._knowledge_answer)
        graph.add_node("issue_analysis", self._issue_analysis)

        # 代码生成节点
        graph.add_node("load_code_context", self._load_code_context)
        graph.add_node("retrieve_code_context", self._retrieve_code_context)
        graph.add_node("code_generation", self._code_generation)

        # 响应节点
        graph.add_node("out_of_scope_response", self._out_of_scope_response)
        graph.add_node("finalize_response", self._finalize_response)

        # 主流程边
        graph.add_edge(START, "load_context")
        graph.add_edge("load_context", "intent_routing")

        # 意图路由
        graph.add_conditional_edges(
            "intent_routing",
            self._route_by_intent,
            {
                "knowledge_qa": "knowledge_answer",
                "issue_analysis": "issue_analysis",
                "code_generation": "load_code_context",
                "out_of_scope": "out_of_scope_response",
            },
        )

        # 子图完成后进入收口
        graph.add_edge("knowledge_answer", "finalize_response")
        graph.add_edge("issue_analysis", "finalize_response")

        # 代码生成流程
        graph.add_edge("load_code_context", "retrieve_code_context")
        graph.add_edge("retrieve_code_context", "code_generation")
        graph.add_edge("code_generation", "finalize_response")

        # 超出范围响应
        graph.add_edge("out_of_scope_response", "finalize_response")

        # 收口
        graph.add_edge("finalize_response", END)
        return graph.compile(checkpointer=effective_checkpointer)

    def _load_context(self, state: WorkflowState) -> dict[str, Any]:
        return self._run_node("load_context", load_context_node, state)

    def _intent_routing(self, state: WorkflowState) -> dict[str, Any]:
        return self._run_node("intent_routing", intent_routing_node, state)

    def _knowledge_answer(self, state: WorkflowState) -> dict[str, Any]:
        return self._run_subgraph("knowledge_qa", self._knowledge_qa_subgraph, state)

    def _issue_analysis(self, state: WorkflowState) -> dict[str, Any]:
        return self._run_subgraph("issue_analysis", self._issue_analysis_subgraph, state)

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

    def _run_subgraph(
        self,
        subgraph_id: str,
        subgraph: Any,
        state: WorkflowState,
    ) -> dict[str, Any]:
        """执行子图（同步版本）

        子图内部包含完整的检索和生成流程。
        子图在构造时已持有 retriever，无需再传递 service。

        Args:
            subgraph_id: 子图 ID
            subgraph: 子图实例
            state: 工作流状态

        Returns:
            更新后的状态增量
        """
        started_at = perf_counter()
        self._file_logger.debug(
            "workflow.subgraph.start",
            trace_id=state.get("trace_id", ""),
            session_id=state.get("session_id", ""),
            subgraph_id=subgraph_id,
            route=state.get("route", ""),
        )
        try:
            # 子图在构造时已持有 retriever，直接调用 run 即可
            updates = subgraph.run(state)
        except Exception as exc:
            latency_ms = int((perf_counter() - started_at) * 1000)
            self._file_logger.exception(
                "workflow.subgraph.exception",
                trace_id=state.get("trace_id", ""),
                session_id=state.get("session_id", ""),
                subgraph_id=subgraph_id,
                latency_ms=latency_ms,
                error_type=type(exc).__name__,
            )
            raise

        latency_ms = int((perf_counter() - started_at) * 1000)
        self._file_logger.info(
            "workflow.subgraph.complete",
            trace_id=state.get("trace_id", ""),
            session_id=state.get("session_id", ""),
            subgraph_id=subgraph_id,
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
