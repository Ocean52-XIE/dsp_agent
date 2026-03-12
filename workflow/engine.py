from __future__ import annotations

"""基于 LangGraph 的多轮阶段式智能编排工作流。

这一版工作流相较于最初的“单轮分类”实现，有两个关键变化：

1. `intent` 仍然按“每一轮用户输入”重新判断，避免把整场会话固定死在某一种类型上。
2. `task_stage` 作为“跨轮任务状态”保留下来，用于表达当前会话正在经历
   `knowledge_qa -> issue_analysis -> confirm_code -> code_generation`
   这样的阶段升级过程。

节点内部当前依旧是 mock 数据实现，但状态字段、转场逻辑、条件分支和最终输出格式
都已经按照真实可扩展的工作流方式组织，后续只需要逐步替换节点内部能力即可。
"""

import os
from time import perf_counter
from pathlib import Path
from typing import Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph
from workflow.nodes.code_generation import run as code_generation_node
from workflow.nodes.conversation_transition import run as conversation_transition_node
from workflow.nodes.decline_code_generation_response import run as decline_code_generation_response_node
from workflow.nodes.domain_gate import run as domain_gate_node
from workflow.nodes.entry_router import run as entry_router_node
from workflow.nodes.finalize_response import run as finalize_response_node
from workflow.nodes.fix_plan import run as fix_plan_node
from workflow.nodes.intent_classifier import run as intent_classifier_node
from workflow.nodes.issue_localizer import run as issue_localizer_node
from workflow.nodes.knowledge_answer import run as knowledge_answer_node
from workflow.nodes.knowledge_answer.llm_qa import KnowledgeQALLMClient
from workflow.nodes.load_code_context import run as load_code_context_node
from workflow.nodes.load_context import run as load_context_node
from workflow.nodes.merge_evidence import run as merge_evidence_node
from workflow.nodes.out_of_scope_response import run as out_of_scope_response_node
from workflow.nodes.query_rewriter import run as query_rewriter_node
from workflow.nodes.retrieve_cases import run as retrieve_cases_node
from workflow.nodes.retrieve_code import run as retrieve_code_node
from workflow.nodes.retrieve_code.code_retriever import LocalCodeRetriever, parse_code_dirs_from_env
from workflow.nodes.retrieve_code_context import run as retrieve_code_context_node
from workflow.nodes.retrieve_wiki import run as retrieve_wiki_node
from workflow.nodes.retrieve_wiki.wiki_retriever import MarkdownWikiRetriever
from workflow.nodes.root_cause_analysis import run as root_cause_analysis_node
from workflow.runtime_logging import get_file_logger


def _env_bool(name: str, default: bool = False) -> bool:
    """读取 bool 环境变量并做兜底。"""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default

# 当前工程已经切换成真实 LangGraph，这里统一标识后端类型。
BACKEND_NAME = "langgraph"

# 当前工程已经切换成真实 langgraph，因此这里明确标识后端类型，
# 前端调试面板也会直接展示这个值。
BACKEND_NAME = "langgraph"


BACKEND_NAME = "langgraph"


class WorkflowState(TypedDict, total=False):
    """LangGraph 在节点间流转的共享状态。

    这份状态同时承担三类职责：

    1. 表达“本轮请求”的分类和执行信息。
    2. 表达“跨轮会话”的任务阶段和上下文记忆。
    3. 表达“最终响应”的结构化输出，便于 API 层和前端直接消费。
    """

    # 当前执行模式：
    # - message：普通用户消息，需重新做领域判定、意图识别和转场判断。
    # - code_generation：通过专门确认接口恢复，直接进入代码生成链路。
    mode: str
    # 单轮调用链路 ID，便于调试、日志串联和引用追踪。
    trace_id: str
    # 当前会话 ID，由 API 层传入。
    session_id: str
    # 本轮用户输入原文。
    user_query: str
    # 当前会话历史消息，供多轮上下文恢复使用。
    history: list[dict[str, Any]]
    # 在“确认生成代码”接口路径中，上一条问题分析消息会被直接带入图。
    source_message: dict[str, Any]

    # 本轮重新判断出的基础意图。
    route: str
    # 本轮真正执行的图路径：检索、代码生成或控制响应。
    execution_path: str
    # 当前轮相对上一轮的阶段转场类型。
    transition_type: str
    # 本轮执行完成后，会话应处于哪个任务阶段。
    task_stage: str

    # 当前响应状态与消息元信息。
    status: str
    response_kind: str
    next_action: str

    # 领域相关性分数与领域判断结果。
    domain_relevance: float
    is_domain_related: bool

    # 当前轮推断出的模块与说明。
    module_name: str
    module_hint: str

    # 跨轮恢复出的活动主题与任务上下文。
    active_topic: str
    active_topic_source: str
    active_task_stage: str
    active_module_name: str
    active_module_hint: str
    active_qa_context: dict[str, Any] | None
    active_issue_context: dict[str, Any] | None
    last_analysis_result: dict[str, Any] | None
    last_analysis_citations: list[dict[str, Any]]
    pending_action: str

    # 历史摘要与检索中间结果。
    history_summary: str
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

    # 统一分析对象与最终输出。
    analysis: dict[str, Any] | None
    answer: str
    node_trace: list[dict[str, str]]
    assistant_message: dict[str, Any]


class WorkflowService:
    """多轮阶段式 LangGraph 工作流服务。

    设计目标不是把每一轮消息都当成彼此独立的请求，而是：

    - 每轮都重新判断 `intent`，确保当前输入被正确理解；
    - 同时从历史消息恢复 `task_stage`，支持话题延续、阶段升级和切换主题；
    - 并保证代码生成始终依赖前置分析结果，而不是仅靠一句“给我代码”直接进入。
    """

    def __init__(self) -> None:
        self.backend_name = BACKEND_NAME
        # 系统调试开关（默认关闭）：
        # - false：保持当前精简 debug 输出；
        # - true：在最终响应中附加 debug.verbose 扩展调试信息。
        self.debug_verbose_enabled = _env_bool("WORKFLOW_DEBUG_VERBOSE", default=False)
        project_root = Path(__file__).resolve().parent.parent
        self._file_logger = get_file_logger(project_root=project_root)

        # 真实 Wiki 检索器：直接读取仓库内的 Markdown 文档作为语料。
        # 这里在服务初始化时构建轻量索引，后续每次请求直接检索，避免重复扫盘。
        self._wiki_retriever = MarkdownWikiRetriever(
            wiki_dir=project_root / "docs" / "wiki" / "ad_engine",
            project_root=project_root,
            default_top_k=4,
        )

        # 真实代码检索器：
        # - 默认索引仓库根目录 `codes/`（可通过环境变量 WORKFLOW_CODE_RETRIEVER_DIRS 覆盖）；
        # - 不再默认扫描整个仓库，避免把工具工程文件误纳入检索范围；
        # - 采用 Parent/Child 混合召回，返回可定位的代码证据。
        code_dirs = parse_code_dirs_from_env(project_root=project_root)
        self._code_retriever = LocalCodeRetriever(
            project_root=project_root,
            code_dirs=code_dirs,
            default_top_k=4,
        )

        # knowledge_answer 节点使用的 LLM 客户端：
        # - 未配置 API Key 时不会中断流程；
        # - 节点内部会自动降级到规则回答。
        self._knowledge_qa_llm = KnowledgeQALLMClient.from_env()
        self._graph = self._build_graph()
        self._file_logger.info(
            "workflow.service.initialized",
            backend=self.backend_name,
            debug_verbose_enabled=self.debug_verbose_enabled,
            logger_status=self._file_logger.status(),
            code_dirs=[str(path) for path in getattr(self._code_retriever, "code_dirs", [])],
        )

    def run_user_message(
        self,
        *,
        session_id: str,
        trace_id: str,
        user_query: str,
        history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """处理普通用户消息。

        这条入口对应前端输入框直接发出的消息。每次进入这里，都代表要重新做一轮：

        1. 会话上下文恢复；
        2. 领域判定；
        3. 意图识别；
        4. 阶段转场判断；
        5. 进入知识问答、问题分析或代码生成链路。
        """
        state: WorkflowState = {
            "mode": "message",
            "trace_id": trace_id,
            "session_id": session_id,
            "user_query": user_query.strip(),
            "history": history,
            "node_trace": [],
        }
        return self._invoke(state)

    def run_code_generation(
        self,
        *,
        session_id: str,
        trace_id: str,
        source_message: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """从问题分析结果恢复，直接进入代码生成链路。

        这条入口保留给“按钮确认继续生成代码”的场景。它不会重新做领域判定和检索问答，
        而是把上一条问题分析结论当作可信前置状态，直接继续向后执行。
        """
        state: WorkflowState = {
            "mode": "code_generation",
            "trace_id": trace_id,
            "session_id": session_id,
            "user_query": source_message.get("content", ""),
            "source_message": source_message,
            "history": history,
            "node_trace": [],
        }
        return self._invoke(state)

    def _invoke(self, state: WorkflowState) -> dict[str, Any]:
        """执行图并补齐统一调试信息。"""
        started_at = perf_counter()
        self._file_logger.info(
            "workflow.invoke.start",
            trace_id=state.get("trace_id", ""),
            session_id=state.get("session_id", ""),
            mode=state.get("mode", "message"),
            history_size=len(state.get("history", []) or []),
            user_query_preview=self._preview_text(state.get("user_query", ""), max_chars=120),
        )
        try:
            result = self._graph.invoke(state)
        except Exception as exc:
            latency_ms = int((perf_counter() - started_at) * 1000)
            self._file_logger.exception(
                "workflow.invoke.exception",
                trace_id=state.get("trace_id", ""),
                session_id=state.get("session_id", ""),
                mode=state.get("mode", "message"),
                latency_ms=latency_ms,
                error_type=type(exc).__name__,
            )
            raise
        assistant_message = dict(result["assistant_message"])
        latency_ms = int((perf_counter() - started_at) * 1000)

        # 注意：debug 字段现在由“系统调试开关”控制，默认可能不存在。
        # 因此这里只在 debug 已存在且是 dict 的情况下补 latency，避免 KeyError。
        debug_payload = assistant_message.get("debug")
        if isinstance(debug_payload, dict):
            debug_payload["latency_ms"] = latency_ms
        self._file_logger.info(
            "workflow.invoke.complete",
            trace_id=state.get("trace_id", ""),
            session_id=state.get("session_id", ""),
            mode=state.get("mode", "message"),
            latency_ms=latency_ms,
            response_kind=assistant_message.get("kind", "unknown"),
            response_status=assistant_message.get("status", "unknown"),
            next_action=(assistant_message.get("actions") or []),
            citation_count=len(assistant_message.get("citations", []) or []),
            node_trace_count=len(result.get("node_trace", []) or []),
        )
        return assistant_message

    def _build_graph(self) -> Any:
        """定义多轮阶段式图结构。

        图有两条大路径：

        1. `message`：普通消息路径，先恢复上下文，再判断这轮是延续、升级还是切换主题。
        2. `code_generation`：从确认节点恢复的快捷路径，直接装配分析上下文并生成代码建议。
        """
        graph = StateGraph(WorkflowState)

        # 入口与上下文恢复节点。
        graph.add_node("entry_router", self._entry_router)
        graph.add_node("load_context", self._load_context)
        graph.add_node("domain_gate", self._domain_gate)
        graph.add_node("intent_classifier", self._intent_classifier)
        graph.add_node("conversation_transition", self._conversation_transition)

        # 通用检索与证据聚合节点。
        graph.add_node("query_rewriter", self._query_rewriter)
        graph.add_node("retrieve_wiki", self._retrieve_wiki)
        graph.add_node("retrieve_cases", self._retrieve_cases)
        graph.add_node("retrieve_code", self._retrieve_code)
        graph.add_node("merge_evidence", self._merge_evidence)

        # 知识问答与问题分析节点。
        graph.add_node("knowledge_answer", self._knowledge_answer)
        graph.add_node("issue_localizer", self._issue_localizer)
        graph.add_node("root_cause_analysis", self._root_cause_analysis)
        graph.add_node("fix_plan", self._fix_plan)

        # 特殊响应节点。
        graph.add_node("decline_code_generation_response", self._decline_code_generation_response)
        graph.add_node("out_of_scope_response", self._out_of_scope_response)

        # 代码生成链路节点。
        graph.add_node("load_code_context", self._load_code_context)
        graph.add_node("retrieve_code_context", self._retrieve_code_context)
        graph.add_node("code_generation", self._code_generation)

        # 收敛输出节点。
        graph.add_node("finalize_response", self._finalize_response)

        # 图入口：先看本轮是普通消息还是确认后的代码生成恢复。
        graph.add_edge(START, "entry_router")
        graph.add_conditional_edges(
            "entry_router",
            self._route_by_mode,
            {
                "message": "load_context",
                "code_generation": "load_code_context",
            },
        )

        # 普通消息路径：
        # 先恢复会话上下文，再做领域判定；只有进入业务域，才继续做意图识别和阶段转场。
        graph.add_edge("load_context", "domain_gate")
        graph.add_conditional_edges(
            "domain_gate",
            self._route_by_domain_gate,
            {
                "in_scope": "intent_classifier",
                "out_of_scope": "out_of_scope_response",
            },
        )
        graph.add_edge("intent_classifier", "conversation_transition")
        graph.add_conditional_edges(
            "conversation_transition",
            self._route_by_execution_path,
            {
                "retrieval_flow": "query_rewriter",
                "code_generation_flow": "load_code_context",
                "decline_code_flow": "decline_code_generation_response",
            },
        )

        # 通用检索链路。知识问答和问题分析共用检索，只在证据融合之后再分叉。
        graph.add_edge("query_rewriter", "retrieve_wiki")
        graph.add_edge("retrieve_wiki", "retrieve_cases")
        graph.add_edge("retrieve_cases", "retrieve_code")
        graph.add_edge("retrieve_code", "merge_evidence")
        graph.add_conditional_edges(
            "merge_evidence",
            self._route_by_intent,
            {
                "knowledge_qa": "knowledge_answer",
                "issue_analysis": "issue_localizer",
            },
        )

        # 问答和问题分析的后半段。
        graph.add_edge("knowledge_answer", "finalize_response")
        graph.add_edge("issue_localizer", "root_cause_analysis")
        graph.add_edge("root_cause_analysis", "fix_plan")
        graph.add_edge("fix_plan", "finalize_response")
        graph.add_edge("decline_code_generation_response", "finalize_response")
        graph.add_edge("out_of_scope_response", "finalize_response")

        # 代码生成路径较短：装配分析上下文 -> 检索代码上下文 -> 生成 -> 收敛。
        graph.add_edge("load_code_context", "retrieve_code_context")
        graph.add_edge("retrieve_code_context", "code_generation")
        graph.add_edge("code_generation", "finalize_response")
        graph.add_edge("finalize_response", END)

        return graph.compile()

    def _entry_router(self, state: WorkflowState) -> dict[str, Any]:
        return self._run_node("entry_router", entry_router_node, state)

    def _load_context(self, state: WorkflowState) -> dict[str, Any]:
        return self._run_node("load_context", load_context_node, state)

    def _domain_gate(self, state: WorkflowState) -> dict[str, Any]:
        return self._run_node("domain_gate", domain_gate_node, state)

    def _intent_classifier(self, state: WorkflowState) -> dict[str, Any]:
        return self._run_node("intent_classifier", intent_classifier_node, state)

    def _conversation_transition(self, state: WorkflowState) -> dict[str, Any]:
        return self._run_node("conversation_transition", conversation_transition_node, state)

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

    def _issue_localizer(self, state: WorkflowState) -> dict[str, Any]:
        return self._run_node("issue_localizer", issue_localizer_node, state)

    def _root_cause_analysis(self, state: WorkflowState) -> dict[str, Any]:
        return self._run_node("root_cause_analysis", root_cause_analysis_node, state)

    def _fix_plan(self, state: WorkflowState) -> dict[str, Any]:
        return self._run_node("fix_plan", fix_plan_node, state)

    def _decline_code_generation_response(self, state: WorkflowState) -> dict[str, Any]:
        return self._run_node("decline_code_generation_response", decline_code_generation_response_node, state)

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

    def _route_by_mode(self, state: WorkflowState) -> str:
        """entry_router 的条件分支函数。"""
        route = state.get("mode", "message")
        self._file_logger.debug(
            "workflow.route.mode",
            trace_id=state.get("trace_id", ""),
            session_id=state.get("session_id", ""),
            route=route,
        )
        return route

    def _route_by_domain_gate(self, state: WorkflowState) -> str:
        """domain_gate 的条件分支函数。"""
        route = "in_scope" if state.get("is_domain_related") else "out_of_scope"
        self._file_logger.debug(
            "workflow.route.domain_gate",
            trace_id=state.get("trace_id", ""),
            session_id=state.get("session_id", ""),
            route=route,
            domain_relevance=state.get("domain_relevance", 0.0),
        )
        return route

    def _route_by_execution_path(self, state: WorkflowState) -> str:
        """根据转场节点的决定，进入检索、代码生成或控制响应路径。"""
        route = state.get("execution_path", "retrieval_flow")
        self._file_logger.debug(
            "workflow.route.execution_path",
            trace_id=state.get("trace_id", ""),
            session_id=state.get("session_id", ""),
            route=route,
            transition_type=state.get("transition_type", "unknown"),
        )
        return route

    def _route_by_intent(self, state: WorkflowState) -> str:
        """merge_evidence 之后按意图选择最终分支。"""
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
            "execution_path",
            "transition_type",
            "task_stage",
            "status",
            "next_action",
            "response_kind",
            "module_name",
            "module_hint",
            "active_task_stage",
            "active_topic_source",
            "is_domain_related",
            "domain_relevance",
            "wiki_retrieval_grade",
            "case_retrieval_grade",
            "code_retrieval_grade",
        )
        for key in scalar_keys:
            if key not in updates:
                continue
            value = updates.get(key)
            if isinstance(value, str):
                summary[key] = self._preview_text(value, max_chars=120)
            else:
                summary[key] = value

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
            task_stage=state.get("task_stage", ""),
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
        """在状态中追加节点轨迹，供前端 debug 面板渲染。"""
        return [*state.get("node_trace", []), {"node": node, "summary": summary}]

    def _build_history_summary(self, history: list[dict[str, Any]]) -> str:
        """构造轻量历史摘要。"""
        recent_questions = [
            message.get("content", "")[:24]
            for message in history
            if message.get("role") == "user"
        ][-4:]
        return " / ".join(recent_questions) if recent_questions else "无历史问题"

    def _latest_message_by(
        self,
        history: list[dict[str, Any]],
        predicate: Any,
    ) -> dict[str, Any] | None:
        """倒序查找满足条件的最近一条消息。"""
        for message in reversed(history):
            if predicate(message):
                return message
        return None

    def _derive_task_stage(self, message: dict[str, Any] | None) -> str:
        """根据最近一条关键助手消息恢复会话当前阶段。"""
        if not message:
            return "idle"
        if message.get("status") == "confirm_code":
            return "confirm_code"
        if message.get("kind") == "code_generation":
            return "code_generation"
        if message.get("intent") == "issue_analysis":
            return "issue_analysis"
        if message.get("intent") == "knowledge_qa":
            return "knowledge_qa"
        return "idle"

    def _extract_module_from_message(self, message: dict[str, Any] | None) -> tuple[str, str]:
        """从历史分析或问答消息中恢复活动模块。"""
        if not message:
            return "", ""
        analysis = message.get("analysis") or {}
        module_name = analysis.get("module", "")
        module_hint = self._infer_module(module_name)[1] if module_name else ""
        return module_name, module_hint

    def _is_pronoun_followup(self, text: str) -> bool:
        """识别“那它、这个、这块”这类依赖上下文的追问。"""
        pronouns = ("它", "这个", "这个问题", "那它", "那这个", "这块", "这里", "上面这个")
        return any(token in text for token in pronouns)

    def _looks_like_code_generation_request(self, text: str) -> bool:
        """识别用户是否在请求进入代码实现阶段。"""
        terms = (
            "给我代码",
            "给出代码",
            "直接给代码",
            "代码实现",
            "实现一下",
            "写一下代码",
            "补丁",
            "patch",
            "改代码",
            "修改代码",
            "直接修",
        )
        lowered = text.lower()
        return any(term in text or term in lowered for term in terms)

    def _looks_like_decline_code_request(self, text: str) -> bool:
        """识别用户是否在明确表示暂不进入代码实现。"""
        terms = ("不用代码", "先不用", "暂不需要", "不用了", "不需要代码", "先不要代码")
        return any(term in text for term in terms)

    def _is_same_topic(self, current_module: str, active_module: str | None, user_query: str) -> bool:
        """判断当前轮是否仍在延续上一个主题。"""
        if active_module and current_module == active_module:
            return True
        if active_module and self._is_pronoun_followup(user_query):
            return True
        return False

    def _infer_module(self, text: str) -> tuple[str, str]:
        """根据关键词粗略推断目标模块。

        当前仍然是轻量规则版，后续可替换成真实代码索引或路由模型。
        """
        if not text:
            return "ad-serving-orchestrator", "广告在线投放编排与请求链路推进"

        lowered = text.lower()
        # P0：模块路由词典补强（函数级强绑定）。
        # 目标：
        # 1. 避免“apply_diversity_penalty / apply_safety_penalty / rank_topn”被误路由到 bid/rate；
        # 2. 对函数定位问法优先按符号名直达模块，再走通用语义匹配。
        rerank_symbol_tokens = (
            "apply_diversity_penalty",
            "apply_safety_penalty",
            "rank_topn",
            "compute_rank_score",
            "rerank_engine.py",
            # 中文函数语义词，避免路由被落到 orchestrator/rate。
            "多样性惩罚",
            "安全惩罚",
            "频控惩罚",
            "最终排序分",
        )
        bid_symbol_tokens = (
            "compute_bid_for_request",
            "compute_ocpc_bid",
            "compute_alpha",
            "bid_optimizer.py",
        )
        rate_symbol_tokens = (
            "predict_ctr_cvr",
            "calibrate_probability",
            "rate_predictor.py",
        )
        recall_symbol_tokens = (
            "select_recall_candidates",
            "recall_service.py",
        )

        if any(token in lowered for token in rerank_symbol_tokens):
            return "rerank-engine", "多目标排序与约束重排链路"
        if any(token in lowered for token in bid_symbol_tokens):
            return "bid-optimizer", "实时出价、预算节奏与竞争策略链路"
        if any(token in lowered for token in rate_symbol_tokens):
            return "rate-prediction", "点击率/转化率预估与校准链路"
        if any(token in lowered for token in recall_symbol_tokens):
            return "ad-recall", "广告候选召回与过滤链路"

        # 全局约束类问题优先回到总体架构模块：
        # 例如“预算和风控约束一般包括哪些内容”，更偏全链路约束口径而非单一出价实现。
        if (
            any(token in lowered for token in ("约束", "全局", "总体", "架构"))
            and any(token in lowered for token in ("预算", "风控"))
            and all(token not in lowered for token in ("出价", "公式", "target_cpa", "target_roas"))
        ):
            return "ad-serving-orchestrator", "广告在线投放编排与请求链路推进"

        # 排障语义优先路由：
        # 1. 先识别“成本/胜率/冷启动/无量/掉量”等排障高频词；
        # 2. 再按更细粒度关键词映射到模块，避免大量问题都落到 orchestrator。
        is_troubleshoot_intent = any(
            token in lowered
            for token in (
                "排障",
                "排查",
                "故障",
                "异常",
                "定位",
                "冷启动",
                "无量",
                "掉量",
                "没量",
                "不稳定",
                "超标",
                "成本",
                "胜率",
                "联调",
                "实践建议",
                "手册",
            )
        )
        if is_troubleshoot_intent:
            # “联调实践/快速判断/手册”属于流程化排障引导，更贴近 orchestrator/手册语义。
            if any(token in lowered for token in ("联调", "实践建议", "快速判断", "排障手册")):
                return "ad-serving-orchestrator", "广告在线投放编排与请求链路推进"
            if any(token in lowered for token in ("高ecpm", "ecpm", "最终位", "精排", "排序", "重排", "rerank")):
                return "rerank-engine", "多目标排序与约束重排链路"
            if any(token in lowered for token in ("冷启动", "无量", "ctr", "cvr", "pctr", "pcvr", "预估", "校准")):
                return "rate-prediction", "点击率/转化率预估与校准链路"
            if any(token in lowered for token in ("成本", "超标", "胜率", "出价", "预算", "pacing", "竞价", "cpa", "roas")):
                return "bid-optimizer", "实时出价、预算节奏与竞争策略链路"
            if any(token in lowered for token in ("召回", "recall", "候选", "人群包")):
                return "ad-recall", "广告候选召回与过滤链路"

        # “高eCPM未进最终位”这类是典型精排问题，放在通用匹配前先行路由。
        if "ecpm" in lowered and any(token in lowered for token in ("最终位", "进入最终位", "没进", "未进")):
            return "rerank-engine", "多目标排序与约束重排链路"

        # 先匹配“出价”相关关键词：
        # 这是为了避免像 “target_cpa + pCVR 出价公式” 这种 query 被“两率关键词”提前截走。
        # 即便 query 同时包含 pCVR，也应优先归入出价模块（bid-optimizer）。
        if any(
            token in lowered
            for token in (
                "出价",
                "bid",
                "ocpc",
                "tcpa",
                "troas",
                "target_cpa",
                "target_roas",
                "cpa",
                "roas",
                "预算",
                "pacing",
                "公式",
                "出价公式",
            )
        ):
            return "bid-optimizer", "实时出价、预算节奏与竞争策略链路"

        # 再匹配两率预估相关关键词。
        if any(token in lowered for token in ("召回", "recall", "候选", "人群包")):
            return "ad-recall", "广告候选召回与过滤链路"
        if any(token in lowered for token in ("两率", "ctr", "cvr", "pctr", "pcvr", "预估", "校准")):
            return "rate-prediction", "点击率/转化率预估与校准链路"
        if any(token in lowered for token in ("精排", "排序", "rerank", "重排")):
            return "rerank-engine", "多目标排序与约束重排链路"
        if any(token in lowered for token in ("频控", "风控", "流量", "治理")):
            return "traffic-governor", "频控、风控与流量治理链路"

        return "ad-serving-orchestrator", "广告在线投放编排与请求链路推进"
