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

from time import perf_counter
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

# 当前工程已经切换成真实 langgraph，因此这里明确标识后端类型，
# 前端调试面板也会直接展示这个值。
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
    wiki_hits: list[dict[str, Any]]
    case_hits: list[dict[str, Any]]
    code_hits: list[dict[str, Any]]
    citations: list[dict[str, Any]]

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
        self._graph = self._build_graph()

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
        result = self._graph.invoke(state)
        assistant_message = dict(result["assistant_message"])
        assistant_message["debug"]["latency_ms"] = int((perf_counter() - started_at) * 1000)
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
        """记录入口模式，便于前端调试看出当前是普通消息还是恢复执行。"""
        return {
            "node_trace": self._trace(state, "entry_router", f"mode={state.get('mode', 'message')}"),
        }

    def _load_context(self, state: WorkflowState) -> dict[str, Any]:
        """恢复多轮会话上下文。

        这个节点负责从历史消息中恢复“当前会话已经走到哪里了”，包括：

        - 当前活动阶段：刚做完 QA、刚做完分析，还是正在等待代码确认；
        - 当前活动模块：让“它、这个、这块”之类代词追问能复用上一轮主题；
        - 最近一次分析结果：用于文本形式“给我代码”时安全地升级到代码生成；
        - 历史摘要：为未来接真实记忆系统预留稳定接口。
        """
        history = state.get("history", [])
        user_query = state["user_query"]

        latest_assistant = self._latest_message_by(history, lambda message: message.get("role") == "assistant")
        latest_qa_message = self._latest_message_by(
            history,
            lambda message: message.get("role") == "assistant" and message.get("intent") == "knowledge_qa",
        )
        latest_issue_message = self._latest_message_by(
            history,
            lambda message: message.get("role") == "assistant"
            and message.get("kind") == "issue_analysis"
            and message.get("analysis") is not None,
        )
        latest_pending_confirmation = self._latest_message_by(
            history,
            lambda message: message.get("role") == "assistant" and message.get("status") == "confirm_code",
        )

        active_task_stage = self._derive_task_stage(latest_assistant)
        active_module_name, active_module_hint = self._extract_module_from_message(
            latest_issue_message or latest_qa_message
        )

        if self._is_pronoun_followup(user_query) and active_module_name:
            module_name = active_module_name
            module_hint = active_module_hint
            topic_source = "history_memory"
        else:
            module_name, module_hint = self._infer_module(user_query)
            topic_source = "current_query"

        return {
            "module_name": module_name,
            "module_hint": module_hint,
            "active_topic": active_module_name or module_name,
            "active_topic_source": topic_source,
            "active_task_stage": active_task_stage,
            "active_module_name": active_module_name or module_name,
            "active_module_hint": active_module_hint or module_hint,
            "active_qa_context": (latest_qa_message or {}).get("analysis"),
            "active_issue_context": (latest_issue_message or {}).get("analysis"),
            "last_analysis_result": (latest_issue_message or {}).get("analysis"),
            "last_analysis_citations": (latest_issue_message or {}).get("citations", []),
            "pending_action": "confirm_code" if latest_pending_confirmation else "completed",
            "history_summary": self._build_history_summary(history),
            "node_trace": self._trace(
                state,
                "load_context",
                f"active_stage={active_task_stage}, module={module_name}",
            ),
        }

    def _domain_gate(self, state: WorkflowState) -> dict[str, Any]:
        """判断输入是否属于当前系统的业务领域。"""
        text = state["user_query"]
        domain_terms = (
            "业务",
            "系统",
            "模块",
            "接口",
            "配置",
            "代码",
            "日志",
            "订单",
            "库存",
            "结算",
            "支付",
            "任务",
            "案例",
            "wiki",
            "规则",
            "修复",
            "补丁",
        )
        out_terms = ("天气", "股票", "旅游", "电影", "菜谱", "翻译", "写诗", "足球", "nba")
        domain_hits = sum(term in text for term in domain_terms)
        out_hits = sum(term in text for term in out_terms)
        relevance = 0.24 + domain_hits * 0.1
        if "报错" in text or "异常" in text or "失败" in text:
            relevance += 0.18
        if "是什么" in text or "规则" in text or "流程" in text:
            relevance += 0.12
        if state.get("active_task_stage") in {"knowledge_qa", "issue_analysis", "confirm_code", "code_generation"}:
            # 多轮场景下，像“那它报错 500 怎么排查”这种追问虽然当前句子里的业务词较少，
            # 但如果会话已经有明确主题，就应该让领域门控参考历史上下文而不是只看单句。
            relevance += 0.10
            if self._is_pronoun_followup(text):
                relevance += 0.16
        if state.get("pending_action") == "confirm_code" and self._looks_like_decline_code_request(text):
            # “先不用代码”属于会话控制语句，本身业务词很少，但它显然是在回复当前分析链路。
            relevance += 0.22
        if self._looks_like_code_generation_request(text):
            # “给我代码”这类请求虽然不一定带完整业务词，但在当前系统里仍属于领域内任务。
            relevance += 0.16
        if out_hits and domain_hits == 0:
            relevance = 0.14

        relevance = max(0.0, min(0.98, relevance))
        is_domain_related = relevance >= 0.45
        return {
            "domain_relevance": relevance,
            "is_domain_related": is_domain_related,
            "node_trace": self._trace(
                state,
                "domain_gate",
                f"relevance={relevance:.2f}, in_scope={is_domain_related}",
            ),
        }

    def _intent_classifier(self, state: WorkflowState) -> dict[str, Any]:
        """在领域内输入上重新判断本轮意图。

        这里判断的是“本轮 intent”，而不是整场会话 intent。
        即使上一轮是知识问答，本轮也完全可以被识别成问题分析或代码相关请求。
        """
        text = state["user_query"]
        lowered = text.lower()
        issue_terms = (
            "报错",
            "异常",
            "失败",
            "问题",
            "修复",
            "根因",
            "定位",
            "排查",
            "traceback",
            "error",
            "500",
            "超时",
            "补丁",
            "改代码",
            "代码实现",
        )
        route = "issue_analysis" if any(term in lowered for term in issue_terms) else "knowledge_qa"
        return {
            "route": route,
            "node_trace": self._trace(state, "intent_classifier", f"route={route}"),
        }

    def _conversation_transition(self, state: WorkflowState) -> dict[str, Any]:
        """判断当前会话是延续、升级还是切换主题。

        这是多轮阶段升级版工作流的核心节点。它负责回答两个问题：

        1. 本轮 intent 和上一轮阶段之间是什么关系？
        2. 这轮应该继续走检索/分析，还是直接升级到代码生成？

        设计原则如下：

        - `intent` 永远重新判断；
        - `task_stage` 依赖历史恢复；
        - 代码生成必须依赖已有分析结果；
        - 若检测到切换主题，则允许从当前轮重新开始一条新链路。
        """
        user_query = state["user_query"]
        active_stage = state.get("active_task_stage", "idle")
        current_route = state["route"]
        same_topic = self._is_same_topic(
            current_module=state["module_name"],
            active_module=state.get("active_module_name"),
            user_query=user_query,
        )
        wants_code = self._looks_like_code_generation_request(user_query)
        declines_code = self._looks_like_decline_code_request(user_query)
        has_prior_analysis = bool(state.get("last_analysis_result"))

        execution_path = "retrieval_flow"
        transition_type = f"start_{current_route}"
        next_stage = current_route

        if state.get("pending_action") == "confirm_code" and declines_code:
            execution_path = "decline_code_flow"
            transition_type = "decline_code_generation"
            next_stage = "issue_analysis"
        elif wants_code and has_prior_analysis:
            execution_path = "code_generation_flow"
            transition_type = "upgrade_to_code_generation"
            current_route = "issue_analysis"
            next_stage = "code_generation"
        elif wants_code and not has_prior_analysis:
            execution_path = "retrieval_flow"
            transition_type = "code_request_without_analysis"
            current_route = "issue_analysis"
            next_stage = "issue_analysis"
        else:
            if active_stage == "knowledge_qa" and current_route == "issue_analysis":
                transition_type = "upgrade_from_qa_to_issue_analysis"
                next_stage = "issue_analysis"
            elif active_stage in {"issue_analysis", "confirm_code", "code_generation"} and current_route == "knowledge_qa":
                transition_type = "fallback_to_contextual_qa" if same_topic else "switch_topic"
                next_stage = "knowledge_qa"
            elif active_stage in {"issue_analysis", "confirm_code", "code_generation"} and current_route == "issue_analysis":
                transition_type = "continue_issue_analysis" if same_topic else "switch_topic"
                next_stage = "issue_analysis"
            elif active_stage == "knowledge_qa" and current_route == "knowledge_qa":
                transition_type = "continue_knowledge_qa" if same_topic else "switch_topic"
                next_stage = "knowledge_qa"

        return {
            "route": current_route,
            "execution_path": execution_path,
            "transition_type": transition_type,
            "task_stage": next_stage,
            "node_trace": self._trace(
                state,
                "conversation_transition",
                f"transition={transition_type}, path={execution_path}",
            ),
        }

    def _query_rewriter(self, state: WorkflowState) -> dict[str, Any]:
        """根据本轮 intent 与当前阶段生成检索语句。"""
        module_name = state["module_name"]
        if state["route"] == "issue_analysis":
            queries = [
                state["user_query"],
                f"{module_name} 异常日志",
                f"{module_name} 历史案例",
                f"{module_name} handler 幂等",
            ]
            if state.get("transition_type") == "upgrade_from_qa_to_issue_analysis":
                # 从 QA 升级到分析时，补一条“从规则到故障”的桥接检索语句。
                queries.append(f"{module_name} 规则偏差 故障定位")
            if state.get("transition_type") == "code_request_without_analysis":
                # 用户直接要代码但没有前置分析时，先补问题分析证据。
                queries.append(f"{module_name} 修复方案 关键边界")
        else:
            queries = [
                state["user_query"],
                f"{module_name} 业务规则",
                f"{module_name} service 流程",
            ]
            if state.get("active_task_stage") in {"issue_analysis", "confirm_code", "code_generation"}:
                # 分析阶段回退到 QA 时，补一条“解释当前模块背景”的语句。
                queries.append(f"{module_name} 设计背景 业务口径")
        return {
            "retrieval_queries": queries,
            "node_trace": self._trace(state, "query_rewriter", f"queries={len(queries)}"),
        }

    def _retrieve_wiki(self, state: WorkflowState) -> dict[str, Any]:
        """模拟业务文档检索节点。"""
        module_name = state["module_name"]
        module_hint = state["module_hint"]
        if state["route"] == "issue_analysis":
            items = [
                {
                    "source_type": "wiki",
                    "title": "异常处理与回滚规范",
                    "path": "wiki/engineering/error-handling.md",
                    "score": 0.86,
                    "excerpt": "规范要求状态流转前做幂等判断，并记录关键日志字段。",
                },
                {
                    "source_type": "wiki",
                    "title": f"{module_name} 运行说明",
                    "path": f"wiki/modules/{module_name}.md",
                    "score": 0.82,
                    "excerpt": f"{module_hint} 是该模块的核心流程，涉及多处状态推进与回滚。",
                },
            ]
        else:
            items = [
                {
                    "source_type": "wiki",
                    "title": "业务规则与口径总览",
                    "path": "wiki/business-rules/overview.md",
                    "score": 0.94,
                    "excerpt": "业务规则通过状态流转、字段口径和边界条件定义对外行为。",
                },
                {
                    "source_type": "wiki",
                    "title": f"{module_name} 模块说明",
                    "path": f"wiki/modules/{module_name}.md",
                    "score": 0.87,
                    "excerpt": f"{module_hint} 是回答该问题时最相关的业务入口。",
                },
            ]
        return {
            "wiki_hits": items,
            "node_trace": self._trace(state, "retrieve_wiki", f"hits={len(items)}"),
        }

    def _retrieve_cases(self, state: WorkflowState) -> dict[str, Any]:
        """模拟历史案例检索节点。"""
        module_name = state["module_name"]
        if state["route"] == "issue_analysis":
            items = [
                {
                    "source_type": "case",
                    "title": f"{module_name} 历史案例复盘",
                    "path": f"cases/{module_name}/incident-retrospective.md",
                    "score": 0.92,
                    "excerpt": "同类问题曾由重复回调与边界值保护不足触发。",
                }
            ]
        else:
            items = [
                {
                    "source_type": "case",
                    "title": f"{module_name} 设计案例",
                    "path": f"cases/{module_name}/design-notes.md",
                    "score": 0.72,
                    "excerpt": "历史设计说明可用于补充模块职责和关键边界条件。",
                }
            ]
        return {
            "case_hits": items,
            "node_trace": self._trace(state, "retrieve_cases", f"hits={len(items)}"),
        }

    def _retrieve_code(self, state: WorkflowState) -> dict[str, Any]:
        """模拟代码与测试文件检索节点。"""
        module_name = state["module_name"]
        if state["route"] == "issue_analysis":
            items = [
                {
                    "source_type": "code",
                    "title": f"{module_name} 关键实现",
                    "path": f"services/{module_name}/handler.py",
                    "score": 0.89,
                    "excerpt": "入口函数对异常分支和重复调用缺少防御。",
                },
                {
                    "source_type": "code",
                    "title": f"{module_name} 单测",
                    "path": f"tests/{module_name}/test_handler.py",
                    "score": 0.8,
                    "excerpt": "现有单测未覆盖重复回调与空值边界场景。",
                },
            ]
        else:
            items = [
                {
                    "source_type": "code",
                    "title": f"{module_name} 模块入口",
                    "path": f"services/{module_name}/service.py",
                    "score": 0.87,
                    "excerpt": "模块职责主要集中在 service、validator 和 assembler 三段逻辑。",
                }
            ]
        return {
            "code_hits": items,
            "node_trace": self._trace(state, "retrieve_code", f"hits={len(items)}"),
        }

    def _merge_evidence(self, state: WorkflowState) -> dict[str, Any]:
        """把多源证据按分数排序后裁剪为前端可展示的 TopN。"""
        citations = sorted(
            [
                *state.get("wiki_hits", []),
                *state.get("case_hits", []),
                *state.get("code_hits", []),
            ],
            key=lambda item: item["score"],
            reverse=True,
        )[:5]
        return {
            "citations": citations,
            "node_trace": self._trace(state, "merge_evidence", f"citations={len(citations)}"),
        }

    def _knowledge_answer(self, state: WorkflowState) -> dict[str, Any]:
        """生成知识问答结果。"""
        module_name = state["module_name"]
        module_hint = state["module_hint"]
        transition_type = state.get("transition_type", "start_knowledge_qa")

        if transition_type == "fallback_to_contextual_qa":
            opening = "这轮输入被识别为围绕当前分析主题的补充问答。"
        elif transition_type == "continue_knowledge_qa":
            opening = "这轮输入延续了前面的知识问答主题。"
        elif transition_type == "switch_topic":
            opening = "这轮输入被判定为新的知识问答主题，因此已重新开始一条问答链路。"
        else:
            opening = "这轮输入被判定为业务知识问答。"

        return {
            "response_kind": "knowledge_qa",
            "task_stage": "knowledge_qa",
            "status": "completed",
            "next_action": "completed",
            "answer": (
                f"{opening}"
                f"结合文档和代码，当前主题主要落在 `{module_name}`，核心关注点是 {module_hint}。"
                "如果你继续追问规则、字段口径、流程或状态流转，当前上下文仍可直接复用。"
            ),
            "analysis": {
                "summary": "知识问答已完成",
                "module": module_name,
                "confidence": "medium",
                "transition_type": transition_type,
                "task_stage": "knowledge_qa",
                "highlights": [
                    "已完成领域判定、意图识别、上下文转场和统一检索",
                    "当前回答已绑定业务文档和代码入口作为依据",
                ],
            },
            "node_trace": self._trace(state, "knowledge_answer", f"module={module_name}"),
        }

    def _issue_localizer(self, state: WorkflowState) -> dict[str, Any]:
        """问题分析第一步：定位候选模块。"""
        module_name = state["module_name"]
        analysis = dict(state.get("analysis") or {})
        analysis.update(
            {
                "module": module_name,
                "confidence": "high" if state["domain_relevance"] >= 0.75 else "medium",
                "transition_type": state.get("transition_type"),
            }
        )
        return {
            "analysis": analysis,
            "node_trace": self._trace(state, "issue_localizer", f"module={module_name}"),
        }

    def _root_cause_analysis(self, state: WorkflowState) -> dict[str, Any]:
        """问题分析第二步：补充根因、风险和上下文。"""
        analysis = dict(state.get("analysis") or {})
        analysis.update(
            {
                "root_cause": "高概率是边界输入或重复回调场景未做保护，导致状态被错误覆盖。",
                "risks": [
                    "可能影响已有重试任务或补偿逻辑",
                    "需要确认上游调用方是否依赖旧的异常行为",
                ],
            }
        )
        return {
            "analysis": analysis,
            "node_trace": self._trace(state, "root_cause_analysis", "root_cause=state_protection_gap"),
        }

    def _fix_plan(self, state: WorkflowState) -> dict[str, Any]:
        """问题分析第三步：给出修复方案，并停在等待代码确认的阶段。"""
        analysis = dict(state.get("analysis") or {})
        transition_type = state.get("transition_type", "start_issue_analysis")
        analysis.update(
            {
                "fix_plan": [
                    "在入口层增加空值/非法值校验",
                    "为状态变更逻辑增加幂等保护",
                    "补充与历史案例一致的回归测试",
                ],
                "verification_steps": [
                    "复现原始问题路径并验证错误不再出现",
                    "检查关键日志字段、状态转换和重试行为",
                ],
                "need_user_confirmation": True,
                "task_stage": "confirm_code",
            }
        )

        if transition_type == "upgrade_from_qa_to_issue_analysis":
            opening = "这轮输入已从前面的知识问答升级为问题分析。"
        elif transition_type == "continue_issue_analysis":
            opening = "这轮输入延续了当前问题分析任务。"
        elif transition_type == "code_request_without_analysis":
            opening = "你这轮直接提出了代码实现诉求，但当前会话还缺少前置分析，因此我先补齐了问题分析。"
        elif transition_type == "switch_topic":
            opening = "这轮输入被视为新的问题分析主题，因此重新开始定位和分析。"
        else:
            opening = "这轮输入已进入问题分析链路。"

        return {
            "response_kind": "issue_analysis",
            "task_stage": "confirm_code",
            "status": "confirm_code",
            "next_action": "confirm_code",
            "answer": (
                f"{opening}"
                f"当前更可能定位在 `{state['module_name']}`。"
                "初步根因、修复方案和验证建议已经整理完成；"
                "如果需要，我可以继续生成代码实现建议。"
            ),
            "analysis": analysis,
            "node_trace": self._trace(state, "fix_plan", "next_action=confirm_code"),
        }

    def _decline_code_generation_response(self, state: WorkflowState) -> dict[str, Any]:
        """处理“暂不需要代码实现”的多轮控制响应。"""
        module_name = state.get("active_module_name") or state["module_name"]
        analysis = dict(state.get("last_analysis_result") or state.get("active_issue_context") or {})
        if analysis:
            analysis["task_stage"] = "issue_analysis"

        return {
            "route": "issue_analysis",
            "response_kind": "conversation_control",
            "task_stage": "issue_analysis",
            "status": "completed",
            "next_action": "completed",
            "citations": state.get("last_analysis_citations", []),
            "analysis": analysis or None,
            "answer": (
                f"已记录当前 `{module_name}` 只保留问题分析结论，不进入代码生成阶段。"
                "如果你后面改变主意，可以继续追问实现思路或再次明确提出需要代码实现。"
            ),
            "node_trace": self._trace(
                state,
                "decline_code_generation_response",
                "transition=decline_code_generation",
            ),
        }

    def _out_of_scope_response(self, state: WorkflowState) -> dict[str, Any]:
        """直接产出与当前系统无关的拒答结果。"""
        return {
            "route": "out_of_scope",
            "response_kind": "out_of_scope",
            "task_stage": "out_of_scope",
            "status": "out_of_scope",
            "next_action": "completed",
            "citations": [],
            "analysis": None,
            "answer": (
                "这轮输入在领域判定阶段被识别为与当前系统无关，因此没有进入后续的知识问答、问题分析或代码生成链路。"
                "你可以继续提业务规则、模块设计、接口流程、报错排查或修复方案相关的问题。"
            ),
            "node_trace": self._trace(state, "out_of_scope_response", "route=out_of_scope"),
        }

    def _load_code_context(self, state: WorkflowState) -> dict[str, Any]:
        """恢复代码生成所需的前置分析上下文。"""
        source_message = state.get("source_message") or {}
        analysis = source_message.get("analysis") or state.get("last_analysis_result") or {}
        citations = source_message.get("citations") or state.get("last_analysis_citations", [])

        module_name = analysis.get("module") or state.get("active_module_name") or state.get("module_name")
        if not module_name:
            module_name = "workflow-orchestrator"
        module_hint = self._infer_module(module_name)[1]
        return {
            "route": "issue_analysis",
            "execution_path": "code_generation_flow",
            "transition_type": state.get("transition_type", "resume_code_generation"),
            "task_stage": "code_generation",
            "module_name": module_name,
            "module_hint": module_hint,
            "analysis": analysis or None,
            "citations": citations,
            "node_trace": self._trace(state, "load_code_context", f"module={module_name}"),
        }

    def _retrieve_code_context(self, state: WorkflowState) -> dict[str, Any]:
        """模拟代码生成前的额外代码上下文检索。"""
        module_name = state["module_name"]
        items = [
            {
                "source_type": "code",
                "title": f"{module_name} 目标文件",
                "path": f"services/{module_name}/handler.py",
                "score": 0.91,
                "excerpt": "建议在入口函数前半段增加参数保护和幂等判断。",
            },
            {
                "source_type": "code",
                "title": f"{module_name} 测试文件",
                "path": f"tests/{module_name}/test_handler.py",
                "score": 0.84,
                "excerpt": "建议新增空值输入、重复回调和回滚路径三类测试。",
            },
        ]
        return {
            "code_hits": items,
            "citations": items,
            "node_trace": self._trace(state, "retrieve_code_context", f"hits={len(items)}"),
        }

    def _code_generation(self, state: WorkflowState) -> dict[str, Any]:
        """生成代码实现建议。

        当前仍是 mock 结果，但输出结构已经和前端的补丁摘要、文件列表、测试建议
        面板完全对齐。
        """
        module_name = state["module_name"]
        transition_type = state.get("transition_type", "resume_code_generation")

        if transition_type == "upgrade_to_code_generation":
            opening = "这轮输入触发了从分析结果到代码实现的阶段升级。"
        else:
            opening = "当前是从确认节点恢复的代码生成链路。"

        return {
            "response_kind": "code_generation",
            "task_stage": "code_generation",
            "status": "completed",
            "next_action": "completed",
            "answer": (
                f"{opening} 已继续生成 `{module_name}` 的实现建议。"
                "当前输出仍是 LangGraph 工作流中代码生成节点的 mock 结果，"
                "你可以先评审补丁摘要、涉及文件和测试建议，再接入真实代码生成器。"
            ),
            "analysis": {
                "summary": "代码实现建议已生成",
                "module": module_name,
                "task_stage": "code_generation",
                "transition_type": transition_type,
                "files": [
                    f"services/{module_name}/handler.py",
                    f"tests/{module_name}/test_handler.py",
                ],
                "patch_summary": [
                    "在入口逻辑补充参数兜底与重复回调保护",
                    "在状态流转前增加幂等判断",
                    "补充回归测试覆盖异常和重试路径",
                ],
                "test_plan": [
                    "新增空值输入的单元测试",
                    "新增重复回调幂等测试",
                    "验证异常路径下状态不被污染",
                ],
                "snippet": (
                    "def process_event(payload):\n"
                    "    if payload is None:\n"
                    "        raise ValueError('payload is required')\n\n"
                    "    if is_duplicate(payload.event_id):\n"
                    "        return build_idempotent_response()\n\n"
                    "    return apply_state_transition(payload)\n"
                ),
            },
            "node_trace": self._trace(state, "code_generation", f"module={module_name}"),
        }

    def _finalize_response(self, state: WorkflowState) -> dict[str, Any]:
        """把图中的结构化状态收敛成前端可直接消费的一条助手消息。"""
        graph_path = [item["node"] for item in state.get("node_trace", [])]
        assistant_message = {
            "role": "assistant",
            "kind": state.get("response_kind", state.get("route", "unknown")),
            "intent": state.get("route", "unknown"),
            "status": state.get("status", "completed"),
            "content": state["answer"],
            "trace_id": state["trace_id"],
            "citations": state.get("citations", []),
            "analysis": state.get("analysis"),
            "actions": (
                [{"type": "confirm_code_generation", "label": "需要代码实现"}]
                if state.get("next_action") == "confirm_code"
                else []
            ),
            "debug": {
                "domain_relevance": state.get("domain_relevance", 0.0),
                "latency_ms": 0,
                "route": state.get("route", "unknown"),
                "task_stage": state.get("task_stage", "unknown"),
                "active_task_stage": state.get("active_task_stage", "idle"),
                "transition_type": state.get("transition_type", "unknown"),
                "execution_path": state.get("execution_path", "unknown"),
                "next_action": state.get("next_action", "completed"),
                "graph_backend": self.backend_name,
                "graph_path": graph_path,
            },
        }
        return {
            "assistant_message": assistant_message,
            "node_trace": self._trace(state, "finalize_response", f"kind={assistant_message['kind']}"),
        }

    def _route_by_mode(self, state: WorkflowState) -> str:
        """entry_router 的条件分支函数。"""
        return state.get("mode", "message")

    def _route_by_domain_gate(self, state: WorkflowState) -> str:
        """domain_gate 的条件分支函数。"""
        return "in_scope" if state.get("is_domain_related") else "out_of_scope"

    def _route_by_execution_path(self, state: WorkflowState) -> str:
        """根据转场节点的决定，进入检索、代码生成或控制响应路径。"""
        return state.get("execution_path", "retrieval_flow")

    def _route_by_intent(self, state: WorkflowState) -> str:
        """merge_evidence 之后按意图选择最终分支。"""
        return state["route"]

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
            return "workflow-orchestrator", "业务流程编排与状态推进链路"
        if "inventory" in text or "库存" in text:
            return "inventory-lock", "锁库存与解锁回调链路"
        if "settlement" in text or "结算" in text or "订单" in text or "支付" in text:
            return "settlement-engine", "订单结算与折扣计算链路"
        if "权限" in text or "登录" in text or "鉴权" in text:
            return "auth-gateway", "登录态校验与权限判定链路"
        if "任务" in text or "调度" in text:
            return "job-scheduler", "调度任务与重试执行链路"
        return "workflow-orchestrator", "业务流程编排与状态推进链路"
