# -*- coding: utf-8 -*-
"""
LangGraph 工作流主引擎，负责构建节点图并调度会话流程。
"""
from __future__ import annotations


import os
from time import perf_counter
from pathlib import Path
from typing import Any, Callable, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from workflow.nodes.code_generation_flow.code_generation import run as code_generation_node
from workflow.nodes.routing_context.conversation_transition import run as conversation_transition_node
from workflow.nodes.control_response.decline_code_generation_response import run as decline_code_generation_response_node
from workflow.nodes.routing_context.domain_gate import run as domain_gate_node
from workflow.nodes.routing_context.entry_router import run as entry_router_node
from workflow.nodes.control_response.finalize_response import run as finalize_response_node
from workflow.nodes.analysis.fix_plan import run as fix_plan_node
from workflow.nodes.routing_context.intent_classifier import run as intent_classifier_node
from workflow.nodes.analysis.issue_analysis_llm import run as issue_analysis_llm_node
from workflow.nodes.analysis.issue_localizer import run as issue_localizer_node
from workflow.nodes.analysis.knowledge_answer import run as knowledge_answer_node
from workflow.nodes.analysis.knowledge_answer.llm_qa import KnowledgeQALLMClient
from workflow.nodes.code_generation_flow.load_code_context import run as load_code_context_node
from workflow.nodes.routing_context.load_context import run as load_context_node
from workflow.nodes.retrieval_flow.merge_evidence import run as merge_evidence_node
from workflow.nodes.control_response.out_of_scope_response import run as out_of_scope_response_node
from workflow.nodes.retrieval_flow.query_rewriter import run as query_rewriter_node
from workflow.nodes.retrieval_flow.retrieve_cases import run as retrieve_cases_node
from workflow.nodes.retrieval_flow.retrieve_code import run as retrieve_code_node
from workflow.nodes.retrieval_flow.retrieve_code.code_retriever import LocalCodeRetriever, parse_code_dirs_from_env
from workflow.nodes.code_generation_flow.retrieve_code_context import run as retrieve_code_context_node
from workflow.nodes.retrieval_flow.retrieve_wiki import run as retrieve_wiki_node
from workflow.nodes.retrieval_flow.retrieve_wiki.wiki_retriever import MarkdownWikiRetriever
from workflow.nodes.analysis.root_cause_analysis import run as root_cause_analysis_node
from workflow.domain_profile import DomainProfile, load_domain_profile
from workflow.runtime_logging import get_file_logger
from workflow.utils import env_bool


BACKEND_NAME = "langgraph"


class WorkflowState(TypedDict, total=False):
    """
    工作流状态定义，承载跨节点共享的上下文与中间结果。
    """

    # 当前请求入口模式：`message` 表示普通对话，`code_generation` 表示确认后代码生成链路。
    mode: str
    # 全链路追踪 ID，用于串联节点日志、前端调试信息与后端观测数据。
    trace_id: str
    # 会话唯一标识，同一会话的多轮请求通过该字段复用检查点和上下文。
    session_id: str
    # 当前轮用户原始问题文本（去首尾空白后的结果），是后续路由和检索的基础输入。
    user_query: str
    # 当前会话历史消息列表（user/assistant），用于多轮上下文恢复和阶段延续判断。
    history: list[dict[str, Any]]
    # 代码生成确认入口携带的来源消息，通常是上一轮分析结果消息的完整对象。
    source_message: dict[str, Any]

    # 意图路由结果：如 `knowledge_qa`、`issue_analysis`，决定进入哪条业务主链路。
    route: str
    # 实际执行路径标识：如检索问答流、代码生成流或拒绝流，用于图内条件分支。
    execution_path: str
    # 相对上一轮的会话迁移类型：如继续当前任务、从问答升级到分析、进入代码确认等。
    transition_type: str
    # 当前轮执行后写回的任务阶段：如 `knowledge_qa`、`issue_analysis`、`confirm_code`、`code_generation`。
    task_stage: str

    # 响应状态（例如 `ok`、`confirm_code` 等），用于前端展示和后续动作控制。
    status: str
    # 响应内容类别：例如知识回答、问题分析、代码生成提示等，用于 UI 分类型渲染。
    response_kind: str
    # 建议的下一步动作：例如继续追问、确认生成代码、补充信息等。
    next_action: str

    # 领域相关性分数，通常来自领域门控节点，表示用户问题与目标业务域的匹配强度。
    domain_relevance: float
    # 领域判定布尔结果，`True` 表示继续业务流程，`False` 表示触发越界兜底响应。
    is_domain_related: bool

    # 推断出的目标模块名（例如广告投放、召回、排序等），用于检索和回答聚焦。
    module_name: str
    # 对模块的补充提示文本，通常来自配置侧，用于增强检索 query 和回答上下文。
    module_hint: str

    # 当前会话激活主题，跨轮保存的“讨论对象”标识，便于处理省略主语的追问。
    active_topic: str
    # 激活主题的来源标识（例如来自分类器、历史消息或回填逻辑），用于调试可解释性。
    active_topic_source: str
    # 会话层激活任务阶段，用于跨轮延续任务上下文而不是每轮都从头开始。
    active_task_stage: str
    # 会话层激活模块名，用于短问句/指代问句场景的模块继承。
    active_module_name: str
    # 会话层激活模块提示，与 `active_module_name` 配套用于检索和回答约束。
    active_module_hint: str
    # 最近一次知识问答上下文快照，供后续追问复用关键背景信息。
    active_qa_context: dict[str, Any] | None
    # 最近一次问题分析上下文快照，供确认代码生成或深挖分析时继续使用。
    active_issue_context: dict[str, Any] | None
    # 最近一次问题分析的结构化结果（定位、根因、建议等），用于后续节点复用。
    last_analysis_result: dict[str, Any] | None
    # 最近一次问题分析引用的证据列表，便于后续回答保持证据一致性。
    last_analysis_citations: list[dict[str, Any]]
    # 待执行动作标识（如等待用户确认），用于流程控制和前端动作按钮联动。
    pending_action: str

    # 从历史消息提炼的简要摘要，降低提示词体积并保留关键上下文线索。
    history_summary: str
    # 查询改写节点产出的检索查询列表，作为 wiki/case/code 检索器的输入。
    retrieval_queries: list[str]
    # 检索计划配置（策略、top_k、权重、限额等），驱动后续检索和证据融合行为。
    retrieval_plan: dict[str, Any]
    # Wiki 检索命中列表，通常包含标题、路径、片段、分数等字段。
    wiki_hits: list[dict[str, Any]]
    # Wiki 检索质量等级（如 high/medium/low），用于融合和调试判断。
    wiki_retrieval_grade: str
    # Wiki 检索过程画像（候选规模、过滤统计、耗时等），用于可观测性分析。
    wiki_retrieval_profile: dict[str, Any]
    # 案例检索命中列表，用于补充真实排障经验和已知问题模式。
    case_hits: list[dict[str, Any]]
    # 案例检索质量等级，供融合策略和调试展示使用。
    case_retrieval_grade: str
    # 案例检索过程画像，记录召回/过滤/排序等关键统计信息。
    case_retrieval_profile: dict[str, Any]
    # 代码检索命中列表，包含路径、符号、行号区间、片段等可定位信息。
    code_hits: list[dict[str, Any]]
    # 代码检索质量等级，用于判断是否具备可回答/可定位的代码证据。
    code_retrieval_grade: str
    # 代码检索过程画像，记录检索器策略、命中统计和耗时。
    code_retrieval_profile: dict[str, Any]
    # 证据融合后选中的最终引用列表，是回答中“依据”部分的直接来源。
    citations: list[dict[str, Any]]
    # 融合阶段画像（候选规模、配额策略、各源占比等），用于调优和线上排障。
    evidence_fusion_profile: dict[str, Any]

    # 统一结构化分析对象（问答/问题分析产物），供最终响应组装节点消费。
    analysis: dict[str, Any] | None
    # 最终文本回答内容（若存在），由回答节点或兜底节点生成。
    answer: str
    # 节点执行轨迹列表，记录每个节点的摘要，供前端调试信息展示。
    node_trace: list[dict[str, str]]
    # 对外返回的标准化 assistant 消息对象（含状态、内容、动作、调试信息等）。
    assistant_message: dict[str, Any]


class WorkflowService:
    """
    工作流编排服务，负责路由、检索、分析与响应输出。
    """

    def __init__(self) -> None:
        """
        初始化工作流服务，包括检索器、LLM 客户端与状态图。
        
        参数:
            self: 当前对象实例。
        
        返回:
            无返回值。
        """
        self.backend_name = BACKEND_NAME
        self.debug_verbose_enabled = env_bool("WORKFLOW_DEBUG_VERBOSE", default=False)
        project_root = Path(__file__).resolve().parents[2]
        self._file_logger = get_file_logger(project_root=project_root)
        self.domain_profile: DomainProfile = load_domain_profile(project_root=project_root)

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

        self._knowledge_qa_llm = KnowledgeQALLMClient.from_env(domain_profile=self.domain_profile)
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
        """
        处理用户普通消息入口，并触发工作流执行。
        
        参数:
            self: 当前对象实例。
            session_id: 会话标识。
            trace_id: 链路追踪标识。
            user_query: 用户输入问题文本。
            history: 会话历史消息列表。
        
        返回:
            返回类型为 `dict[str, Any]` 的处理结果。
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
        source_message: dict[str, Any] | None,
        history: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        """
        处理确认后代码生成入口，并触发代码链路。
        
        参数:
            self: 当前对象实例。
            session_id: 会话标识。
            trace_id: 链路追踪标识。
            source_message: 来源消息对象。
            history: 会话历史消息列表。
        
        返回:
            返回类型为 `dict[str, Any]` 的处理结果。
        """
        resolved_source_message = dict(source_message or {})
        resolved_history = list(history or [])
        if not resolved_source_message or not resolved_history:
            checkpoint_state = self._load_checkpoint_state(session_id=session_id, mode="code_generation")
            if checkpoint_state:
                if not resolved_source_message:
                    latest_assistant = checkpoint_state.get("assistant_message") or {}
                    resolved_source_message = dict(latest_assistant)
                if not resolved_history:
                    resolved_history = list(checkpoint_state.get("history", []) or [])

        state: WorkflowState = {
            "mode": "code_generation",
            "trace_id": trace_id,
            "session_id": session_id,
            "user_query": resolved_source_message.get("content", ""),
            "source_message": resolved_source_message,
            "history": resolved_history,
            "node_trace": [],
        }
        return self._invoke(state)

    def _invoke(self, state: WorkflowState) -> dict[str, Any]:
        """
        调用 LangGraph 执行状态流转并输出最终响应。
        
        参数:
            self: 当前对象实例。
            state: 工作流状态字典，包含会话上下文与中间结果。
        
        返回:
            返回类型为 `dict[str, Any]` 的处理结果。
        """
        started_at = perf_counter()
        # 调用图执行前先记录入口日志，便于定位整轮请求链路。
        self._file_logger.info(
            "workflow.invoke.start",
            trace_id=state.get("trace_id", ""),
            session_id=state.get("session_id", ""),
            mode=state.get("mode", "message"),
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
                mode=state.get("mode", "message"),
                latency_ms=latency_ms,
                error_type=type(exc).__name__,
            )
            raise
        assistant_message = dict(result["assistant_message"])
        latency_ms = int((perf_counter() - started_at) * 1000)

        # 将总耗时内联到 debug 字段，便于前端调试面板直接展示。
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

    def _invoke_config(self, state: WorkflowState) -> dict[str, Any]:
        """
        构造图执行配置，注入线程级会话标识。
        
        参数:
            self: 当前对象实例。
            state: 工作流状态字典，包含会话上下文与中间结果。
        
        返回:
            返回类型为 `dict[str, Any]` 的处理结果。
        """
        session_id = str(state.get("session_id", "") or "").strip() or "default_session"
        return {
            "configurable": {
                "thread_id": session_id,
            }
        }

    def _load_checkpoint_state(self, *, session_id: str, mode: str) -> dict[str, Any] | None:
        """
        读取会话检查点状态，用于恢复上下文。
        
        参数:
            self: 当前对象实例。
            session_id: 会话标识。
            mode: 调用模式。
        
        返回:
            返回类型为 `dict[str, Any] | None` 的处理结果。
        """
        try:
            snapshot = self._graph.get_state(
                {
                    "configurable": {
                        "thread_id": session_id,
                    }
                }
            )
        except Exception:
            return None
        values = getattr(snapshot, "values", None)
        return dict(values) if isinstance(values, dict) else None

    def _build_graph(self) -> Any:
        """
        构建完整工作流图，包括节点注册与边路由规则。
        
        参数:
            self: 当前对象实例。
        
        返回:
            返回类型为 `Any` 的处理结果。
        """
        graph = StateGraph(WorkflowState)

        # 路由与上下文加载相关节点。
        graph.add_node("entry_router", self._entry_router)
        graph.add_node("load_context", self._load_context)
        graph.add_node("domain_gate", self._domain_gate)
        graph.add_node("intent_classifier", self._intent_classifier)
        graph.add_node("conversation_transition", self._conversation_transition)

        # 通用检索链路节点。
        graph.add_node("query_rewriter", self._query_rewriter)
        graph.add_node("retrieve_wiki", self._retrieve_wiki)
        graph.add_node("retrieve_cases", self._retrieve_cases)
        graph.add_node("retrieve_code", self._retrieve_code)
        graph.add_node("merge_evidence", self._merge_evidence)

        # 问答与问题分析节点。
        graph.add_node("knowledge_answer", self._knowledge_answer)
        graph.add_node("issue_localizer", self._issue_localizer)
        graph.add_node("issue_analysis_llm", self._issue_analysis_llm)
        graph.add_node("root_cause_analysis", self._root_cause_analysis)
        graph.add_node("fix_plan", self._fix_plan)

        # 控制型响应节点。
        graph.add_node("decline_code_generation_response", self._decline_code_generation_response)
        graph.add_node("out_of_scope_response", self._out_of_scope_response)

        # 代码生成链路节点。
        graph.add_node("load_code_context", self._load_code_context)
        graph.add_node("retrieve_code_context", self._retrieve_code_context)
        graph.add_node("code_generation", self._code_generation)

        graph.add_node("finalize_response", self._finalize_response)

        graph.add_edge(START, "entry_router")

        graph.add_conditional_edges(
            "entry_router",
            self._route_by_mode,
            {
                "message": "load_context",
                "code_generation": "load_code_context",
            },
        )

        # 普通消息先经过 domain/intention/path 分流。
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

        # 统一检索后按意图分到问答或问题分析。
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

        # 问题分析链路需要先定位、再经 LLM 增强、再给根因与修复建议。
        graph.add_edge("knowledge_answer", "finalize_response")

        graph.add_edge("issue_localizer", "issue_analysis_llm")
        graph.add_edge("issue_analysis_llm", "root_cause_analysis")
        graph.add_edge("root_cause_analysis", "fix_plan")
        graph.add_edge("fix_plan", "finalize_response")
        graph.add_edge("decline_code_generation_response", "finalize_response")
        graph.add_edge("out_of_scope_response", "finalize_response")

        graph.add_edge("load_code_context", "retrieve_code_context")
        graph.add_edge("retrieve_code_context", "code_generation")
        graph.add_edge("code_generation", "finalize_response")
        graph.add_edge("finalize_response", END)

        return graph.compile(checkpointer=self._checkpointer)

    def _entry_router(self, state: WorkflowState) -> dict[str, Any]:
        """
        执行` entry router`相关处理逻辑。
        
        参数:
            self: 当前对象实例。
            state: 工作流状态字典，包含会话上下文与中间结果。
        
        返回:
            返回类型为 `dict[str, Any]` 的处理结果。
        """
        return self._run_node("entry_router", entry_router_node, state)

    def _load_context(self, state: WorkflowState) -> dict[str, Any]:
        """
        执行` load context`相关处理逻辑。
        
        参数:
            self: 当前对象实例。
            state: 工作流状态字典，包含会话上下文与中间结果。
        
        返回:
            返回类型为 `dict[str, Any]` 的处理结果。
        """
        return self._run_node("load_context", load_context_node, state)

    def _domain_gate(self, state: WorkflowState) -> dict[str, Any]:
        """
        执行` domain gate`相关处理逻辑。
        
        参数:
            self: 当前对象实例。
            state: 工作流状态字典，包含会话上下文与中间结果。
        
        返回:
            返回类型为 `dict[str, Any]` 的处理结果。
        """
        return self._run_node("domain_gate", domain_gate_node, state)

    def _intent_classifier(self, state: WorkflowState) -> dict[str, Any]:
        """
        执行` intent classifier`相关处理逻辑。
        
        参数:
            self: 当前对象实例。
            state: 工作流状态字典，包含会话上下文与中间结果。
        
        返回:
            返回类型为 `dict[str, Any]` 的处理结果。
        """
        return self._run_node("intent_classifier", intent_classifier_node, state)

    def _conversation_transition(self, state: WorkflowState) -> dict[str, Any]:
        """
        执行` conversation transition`相关处理逻辑。
        
        参数:
            self: 当前对象实例。
            state: 工作流状态字典，包含会话上下文与中间结果。
        
        返回:
            返回类型为 `dict[str, Any]` 的处理结果。
        """
        return self._run_node("conversation_transition", conversation_transition_node, state)

    def _query_rewriter(self, state: WorkflowState) -> dict[str, Any]:
        """
        执行` query rewriter`相关处理逻辑。
        
        参数:
            self: 当前对象实例。
            state: 工作流状态字典，包含会话上下文与中间结果。
        
        返回:
            返回类型为 `dict[str, Any]` 的处理结果。
        """
        return self._run_node("query_rewriter", query_rewriter_node, state)

    def _retrieve_wiki(self, state: WorkflowState) -> dict[str, Any]:
        """
        执行` retrieve wiki`相关处理逻辑。
        
        参数:
            self: 当前对象实例。
            state: 工作流状态字典，包含会话上下文与中间结果。
        
        返回:
            返回类型为 `dict[str, Any]` 的处理结果。
        """
        return self._run_node("retrieve_wiki", retrieve_wiki_node, state)

    def _retrieve_cases(self, state: WorkflowState) -> dict[str, Any]:
        """
        执行` retrieve cases`相关处理逻辑。
        
        参数:
            self: 当前对象实例。
            state: 工作流状态字典，包含会话上下文与中间结果。
        
        返回:
            返回类型为 `dict[str, Any]` 的处理结果。
        """
        return self._run_node("retrieve_cases", retrieve_cases_node, state)

    def _retrieve_code(self, state: WorkflowState) -> dict[str, Any]:
        """
        执行` retrieve code`相关处理逻辑。
        
        参数:
            self: 当前对象实例。
            state: 工作流状态字典，包含会话上下文与中间结果。
        
        返回:
            返回类型为 `dict[str, Any]` 的处理结果。
        """
        return self._run_node("retrieve_code", retrieve_code_node, state)

    def _merge_evidence(self, state: WorkflowState) -> dict[str, Any]:
        """
        执行` merge evidence`相关处理逻辑。
        
        参数:
            self: 当前对象实例。
            state: 工作流状态字典，包含会话上下文与中间结果。
        
        返回:
            返回类型为 `dict[str, Any]` 的处理结果。
        """
        return self._run_node("merge_evidence", merge_evidence_node, state)

    def _knowledge_answer(self, state: WorkflowState) -> dict[str, Any]:
        """
        执行` knowledge answer`相关处理逻辑。
        
        参数:
            self: 当前对象实例。
            state: 工作流状态字典，包含会话上下文与中间结果。
        
        返回:
            返回类型为 `dict[str, Any]` 的处理结果。
        """
        return self._run_node("knowledge_answer", knowledge_answer_node, state)

    def _issue_localizer(self, state: WorkflowState) -> dict[str, Any]:
        """
        执行` issue localizer`相关处理逻辑。
        
        参数:
            self: 当前对象实例。
            state: 工作流状态字典，包含会话上下文与中间结果。
        
        返回:
            返回类型为 `dict[str, Any]` 的处理结果。
        """
        return self._run_node("issue_localizer", issue_localizer_node, state)

    def _issue_analysis_llm(self, state: WorkflowState) -> dict[str, Any]:
        """
        执行` issue analysis llm`相关处理逻辑。
        
        参数:
            self: 当前对象实例。
            state: 工作流状态字典，包含会话上下文与中间结果。
        
        返回:
            返回类型为 `dict[str, Any]` 的处理结果。
        """
        return self._run_node("issue_analysis_llm", issue_analysis_llm_node, state)

    def _root_cause_analysis(self, state: WorkflowState) -> dict[str, Any]:
        """
        执行` root cause analysis`相关处理逻辑。
        
        参数:
            self: 当前对象实例。
            state: 工作流状态字典，包含会话上下文与中间结果。
        
        返回:
            返回类型为 `dict[str, Any]` 的处理结果。
        """
        return self._run_node("root_cause_analysis", root_cause_analysis_node, state)

    def _fix_plan(self, state: WorkflowState) -> dict[str, Any]:
        """
        执行` fix plan`相关处理逻辑。
        
        参数:
            self: 当前对象实例。
            state: 工作流状态字典，包含会话上下文与中间结果。
        
        返回:
            返回类型为 `dict[str, Any]` 的处理结果。
        """
        return self._run_node("fix_plan", fix_plan_node, state)

    def _decline_code_generation_response(self, state: WorkflowState) -> dict[str, Any]:
        """
        执行` decline code generation response`相关处理逻辑。
        
        参数:
            self: 当前对象实例。
            state: 工作流状态字典，包含会话上下文与中间结果。
        
        返回:
            返回类型为 `dict[str, Any]` 的处理结果。
        """
        return self._run_node("decline_code_generation_response", decline_code_generation_response_node, state)

    def _out_of_scope_response(self, state: WorkflowState) -> dict[str, Any]:
        """
        执行` out of scope response`相关处理逻辑。
        
        参数:
            self: 当前对象实例。
            state: 工作流状态字典，包含会话上下文与中间结果。
        
        返回:
            返回类型为 `dict[str, Any]` 的处理结果。
        """
        return self._run_node("out_of_scope_response", out_of_scope_response_node, state)

    def _load_code_context(self, state: WorkflowState) -> dict[str, Any]:
        """
        执行` load code context`相关处理逻辑。
        
        参数:
            self: 当前对象实例。
            state: 工作流状态字典，包含会话上下文与中间结果。
        
        返回:
            返回类型为 `dict[str, Any]` 的处理结果。
        """
        return self._run_node("load_code_context", load_code_context_node, state)

    def _retrieve_code_context(self, state: WorkflowState) -> dict[str, Any]:
        """
        执行` retrieve code context`相关处理逻辑。
        
        参数:
            self: 当前对象实例。
            state: 工作流状态字典，包含会话上下文与中间结果。
        
        返回:
            返回类型为 `dict[str, Any]` 的处理结果。
        """
        return self._run_node("retrieve_code_context", retrieve_code_context_node, state)

    def _code_generation(self, state: WorkflowState) -> dict[str, Any]:
        """
        执行` code generation`相关处理逻辑。
        
        参数:
            self: 当前对象实例。
            state: 工作流状态字典，包含会话上下文与中间结果。
        
        返回:
            返回类型为 `dict[str, Any]` 的处理结果。
        """
        return self._run_node("code_generation", code_generation_node, state)

    def _finalize_response(self, state: WorkflowState) -> dict[str, Any]:
        """
        执行` finalize response`相关处理逻辑。
        
        参数:
            self: 当前对象实例。
            state: 工作流状态字典，包含会话上下文与中间结果。
        
        返回:
            返回类型为 `dict[str, Any]` 的处理结果。
        """
        return self._run_node("finalize_response", finalize_response_node, state)

    def _route_by_mode(self, state: WorkflowState) -> str:
        """
        执行` route by mode`相关处理逻辑。
        
        参数:
            self: 当前对象实例。
            state: 工作流状态字典，包含会话上下文与中间结果。
        
        返回:
            返回类型为 `str` 的处理结果。
        """
        route = state.get("mode", "message")
        self._file_logger.debug(
            "workflow.route.mode",
            trace_id=state.get("trace_id", ""),
            session_id=state.get("session_id", ""),
            route=route,
        )
        return route

    def _route_by_domain_gate(self, state: WorkflowState) -> str:
        """
        执行` route by domain gate`相关处理逻辑。
        
        参数:
            self: 当前对象实例。
            state: 工作流状态字典，包含会话上下文与中间结果。
        
        返回:
            返回类型为 `str` 的处理结果。
        """
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
        """
        执行` route by execution path`相关处理逻辑。
        
        参数:
            self: 当前对象实例。
            state: 工作流状态字典，包含会话上下文与中间结果。
        
        返回:
            返回类型为 `str` 的处理结果。
        """
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
        """
        执行` route by intent`相关处理逻辑。
        
        参数:
            self: 当前对象实例。
            state: 工作流状态字典，包含会话上下文与中间结果。
        
        返回:
            返回类型为 `str` 的处理结果。
        """
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
        """
        执行` preview text`相关处理逻辑。
        
        参数:
            self: 当前对象实例。
            value: 待预览输入值。
            max_chars: 最大预览字符数。
        
        返回:
            返回类型为 `str` 的处理结果。
        """
        text = str(value or "").strip()
        if len(text) <= max_chars:
            return text
        return f"{text[:max_chars]}..."

    def _summarize_node_updates(self, updates: dict[str, Any]) -> dict[str, Any]:
        """
        执行` summarize node updates`相关处理逻辑。
        
        参数:
            self: 当前对象实例。
            updates: 节点更新结果字典。
        
        返回:
            返回类型为 `dict[str, Any]` 的处理结果。
        """
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
        """
        统一执行单个节点并记录节点级日志。
        
        参数:
            self: 当前对象实例。
            node_name: 节点名称。
            node_runner: 节点执行函数。
            state: 工作流状态字典，包含会话上下文与中间结果。
        
        返回:
            返回类型为 `dict[str, Any]` 的处理结果。
        """
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
        """
        执行`runtime log status`相关处理逻辑。
        
        参数:
            self: 当前对象实例。
        
        返回:
            返回类型为 `dict[str, Any]` 的处理结果。
        """
        return self._file_logger.status()

    def _trace(self, state: WorkflowState, node: str, summary: str) -> list[dict[str, str]]:
        """
        追加节点追踪信息，返回新的 trace 列表。
        
        参数:
            self: 当前对象实例。
            state: 工作流状态字典，包含会话上下文与中间结果。
            node: 节点名称。
            summary: 节点摘要文本。
        
        返回:
            返回类型为 `list[dict[str, str]]` 的处理结果。
        """
        return [*state.get("node_trace", []), {"node": node, "summary": summary}]

    def _build_history_summary(self, history: list[dict[str, Any]]) -> str:
        """
        根据历史用户消息生成简要上下文摘要。
        
        参数:
            self: 当前对象实例。
            history: 会话历史消息列表。
        
        返回:
            返回类型为 `str` 的处理结果。
        """
        recent_questions = [
            message.get("content", "")[:24]
            for message in history
            if message.get("role") == "user"
        ][-4:]
        return " / ".join(recent_questions) if recent_questions else "暂无历史用户提问"

    def _latest_message_by(
        self,
        history: list[dict[str, Any]],
        predicate: Any,
    ) -> dict[str, Any] | None:
        """
        按条件逆序查找最近一条历史消息。
        
        参数:
            self: 当前对象实例。
            history: 会话历史消息列表。
            predicate: 过滤消息的判断函数。
        
        返回:
            返回类型为 `dict[str, Any] | None` 的处理结果。
        """
        for message in reversed(history):
            if predicate(message):
                return message
        return None

    def _derive_task_stage(self, message: dict[str, Any] | None) -> str:
        """
        根据历史消息推断当前任务阶段。
        
        参数:
            self: 当前对象实例。
            message: 单条消息对象或消息字典。
        
        返回:
            返回类型为 `str` 的处理结果。
        """
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
        """
        从历史消息中提取模块名与模块提示。
        
        参数:
            self: 当前对象实例。
            message: 单条消息对象或消息字典。
        
        返回:
            返回类型为 `tuple[str, str]` 的处理结果。
        """
        if not message:
            return "", ""
        analysis = message.get("analysis") or {}
        module_name = analysis.get("module", "")
        module_hint = self._infer_module(module_name)[1] if module_name else ""
        return module_name, module_hint

    def _is_pronoun_followup(self, text: str) -> bool:
        """
        判断是否为指代型追问。
        
        参数:
            self: 当前对象实例。
            text: 待处理文本。
        
        返回:
            返回类型为 `bool` 的处理结果。
        """
        pronouns = ("它", "这个", "这个问题", "那个", "那这个", "这块", "这里", "上面这个")
        return any(token in text for token in pronouns)

    def _looks_like_code_location_query(self, text: str) -> bool:
        """
        判断是否为代码定位类问题。
        
        参数:
            self: 当前对象实例。
            text: 待处理文本。
        
        返回:
            返回类型为 `bool` 的处理结果。
        """
        normalized = " ".join((text or "").strip().lower().split())
        if not normalized:
            return False

        code_location_terms = self.domain_profile.query_rewrite.intent_terms.get("code_location", ())
        if any(token and token.lower() in normalized for token in code_location_terms):
            return True

        fallback_terms = (
            "代码在哪里",
            "代码在哪",
            "代码位置",
            "实现在哪",
            "入口在哪",
            "哪个文件",
            "在哪个文件",
            "路径在哪",
            "哪一行",
            "line",
            "where is the code",
            "where is code",
        )
        return any(token in normalized for token in fallback_terms)

    def _looks_like_context_dependent_followup(self, text: str) -> bool:
        """
        判断是否为依赖上下文的追问。
        
        参数:
            self: 当前对象实例。
            text: 待处理文本。
        
        返回:
            返回类型为 `bool` 的处理结果。
        """
        if self._is_pronoun_followup(text):
            return True

        normalized = " ".join((text or "").strip().lower().split())
        compact = normalized.replace(" ", "")
        if not compact:
            return False

        if self._looks_like_code_location_query(normalized):
            return True

        has_code_anchor = any(token in normalized for token in ("代码", "函数", "文件", "路径", "实现", "入口", "行号"))
        has_where_anchor = any(token in normalized for token in ("哪里", "在哪", "哪个", "哪一", "位置"))
        return has_code_anchor and has_where_anchor and len(compact) <= 16

    def _looks_like_code_generation_request(self, text: str) -> bool:
        """
        判断是否为代码生成请求。
        
        参数:
            self: 当前对象实例。
            text: 待处理文本。
        
        返回:
            返回类型为 `bool` 的处理结果。
        """
        terms = (
            "给我代码",
            "给出代码",
            "直接给代码",
            "代码实现",
            "实现一个",
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
        """
        判断是否为拒绝代码生成请求。
        
        参数:
            self: 当前对象实例。
            text: 待处理文本。
        
        返回:
            返回类型为 `bool` 的处理结果。
        """
        terms = ("不用代码", "先不用", "暂不需要", "不用了", "不需要代码", "先不要代码")
        return any(term in text for term in terms)

    def _is_same_topic(self, current_module: str, active_module: str | None, user_query: str) -> bool:
        """
        判断当前问题是否与会话激活主题一致。
        
        参数:
            self: 当前对象实例。
            current_module: 当前轮推断的模块名称。
            active_module: 会话激活模块名称。
            user_query: 用户输入问题文本。
        
        返回:
            返回类型为 `bool` 的处理结果。
        """
        if active_module and current_module == active_module:
            return True
        if active_module and self._looks_like_context_dependent_followup(user_query):
            return True
        return False

    def _infer_module(self, text: str) -> tuple[str, str]:
        """
        根据关键词匹配推断模块名称与说明。
        
        参数:
            self: 当前对象实例。
            text: 待处理文本。
        
        返回:
            返回类型为 `tuple[str, str]` 的处理结果。
        """
        default_module = self.domain_profile.default_module
        default_hint = self.domain_profile.module_hint(default_module)
        if not text:
            return default_module, default_hint

        lowered = text.lower()
        modules = sorted(self.domain_profile.modules, key=lambda item: item.route_priority)

        # Symbol-level routing takes precedence for code-location style queries.
        for module in modules:
            if module.symbol_keywords and any(token.lower() in lowered for token in module.symbol_keywords):
                return module.name, module.hint

        best_module_name = default_module
        best_module_hint = default_hint
        best_score = 0
        best_priority = 10**9
        for module in modules:
            keyword_score = sum(1 for token in module.keywords if token and token.lower() in lowered)
            alias_score = sum(1 for token in module.aliases if token and token.lower() in lowered)
            score = keyword_score + alias_score
            if score <= 0:
                continue
            if score > best_score or (score == best_score and module.route_priority < best_priority):
                best_score = score
                best_priority = module.route_priority
                best_module_name = module.name
                best_module_hint = module.hint

        return best_module_name, best_module_hint
