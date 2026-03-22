# -*- coding: utf-8 -*-
"""KnowledgeQA 子图引擎

完整的知识问答子图，内部包含：
- query_rewriter: 查询重写
- retrieve_wiki: Wiki 检索
- retrieve_code: 代码检索
- merge_evidence: 证据合并
- knowledge_answer: 知识回答生成

可被主工作流调用，提供端到端的知识问答能力。

## 状态设计（自动关联模式）

子图使用独立的 KnowledgeQAState，与主图 WorkflowState 通过同名字段自动关联。

LangGraph 自动处理：
- 输入阶段：主图同名字段自动传递给子图（trace_id, user_query, module_name 等）
- 输出阶段：子图同名字段自动合并回主图（answer, citations, node_trace 等）

子图节点直接从 state 读取字段，无需通过 _main_state。

## 生命周期

1. 子图启动时，LangGraph 自动从主图提取同名字段作为子图输入
2. 子图内部流转，各节点读写子图 state
3. 子图完成时，输出字段自动合并回主图 state
4. 子图 state 被丢弃，主图继续执行 finalize_response
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph

from workflow.subgraph.base import BaseSubgraph
from workflow.subgraph.knowledge_qa.config import KnowledgeQAConfig
from workflow.subgraph.knowledge_qa.state import (
    KnowledgeQAState,
    build_debug_info,
)
from workflow.nodes.agent_loop import BaseAgentLoopNode, AgentLoopNodeConfig

logger = logging.getLogger(__name__)


# ============================================================================
# 知识回答节点
# ============================================================================

class KnowledgeAnswerNode(BaseAgentLoopNode):
    """知识问答节点

    继承 BaseAgentLoopNode，实现知识问答场景：
    - 基于检索证据生成回答
    - 支持 Skill Tool 调用
    - 支持 Fallback 机制

    所有依赖从全局单例获取，无需外部传递。
    """

    def __init__(self) -> None:
        """初始化知识问答节点

        所有依赖从全局单例获取。
        """
        config = AgentLoopNodeConfig(
            node_name="knowledge_answer",
            response_kind="knowledge_qa",
            system_prompt_template=(
                "你是企业知识问答助手。"
                "必须严格基于提供的证据回答，不补充证据外事实。"
                "如果用户的问题需要查询实时数据或执行特定技能，请使用可用工具。"
                "输出中文，结构尽量为：结论 -> 依据。"
            ),
            user_prompt_template="""【用户问题】
{user_query}

【当前主模块】
- module_name: {module_name}
- module_hint: {module_hint}

【相关模块】
{related_modules_block}

【检索证据（按相关性排序）】
{evidence_block}
""",
            enable_skill_tool=True,
            max_iterations=3,
            require_evidence=True,
        )
        super().__init__(config=config)

    def _get_system_prompt(self, state: dict[str, Any]) -> str:
        """获取系统提示词

        优先从 domain_profile.prompts 获取，否则使用默认提示词。

        Args:
            state: 工作流状态
        """
        from domain_profile import get_domain_profile

        domain_profile = get_domain_profile()
        if domain_profile:
            # 优先从 prompts 字典获取 qa_system
            prompts = getattr(domain_profile, "prompts", None)
            if prompts and isinstance(prompts, dict):
                qa_system = prompts.get("qa_system")
                if qa_system:
                    return qa_system

        return """你是一个专业的知识问答助手。
请基于提供的检索证据，准确、清晰地回答用户的问题。

要求：
1. 优先使用检索证据中的信息
2. 如果证据不足以回答问题，请诚实说明
3. 回答应该简洁明了，重点突出
4. 如有引用，请标注来源"""

    def _get_user_prompt(
        self,
        state: dict[str, Any],
        user_query: str,
        module_name: str,
        module_hint: str,
        related_modules: list[dict[str, Any]],
        evidence_hits: list[dict[str, Any]],
    ) -> str:
        """获取用户提示词"""
        # 构建证据文本
        evidence_text = ""
        if evidence_hits:
            # 修复：按 fusion_rank 排序后再取前5个，保留多源平衡
            # 这样 wiki 证据(排名#2 和 #6) 也会被包含
            sorted_evidence = sorted(evidence_hits, key=lambda x: x.get("fusion_rank", 999))[:5]
            evidence_parts = []
            for i, hit in enumerate(sorted_evidence, 1):
                source = hit.get("source_type", hit.get("source", "unknown"))
                path = hit.get("path", "")
                section = hit.get("section", "")
                content = hit.get("content", "")[:500]
                evidence_parts.append(
                    f"[{i}] 来源: {source}\n"
                    f"    路径: {path}\n"
                    f"    章节: {section}\n"
                    f"    内容: {content}"
                )
            evidence_text = "\n\n".join(evidence_parts)

        # 构建模块提示
        module_context = ""
        if module_name:
            module_context = f"\n目标模块: {module_name}"
        if module_hint:
            module_context += f"\n模块提示: {module_hint}"

        return f"""请回答以下问题：

问题: {user_query}
{module_context}

相关证据:
{evidence_text if evidence_text else '无相关证据'}

请基于证据给出准确回答："""

    def _build_fallback(
        self,
        state: dict[str, Any],
        module_name: str,
        module_hint: str,
        related_modules: list[dict[str, Any]],
        evidence_hits: list[dict[str, Any]],
    ) -> str:
        """构建 Fallback 响应

        直接从子图 state 读取用户输入（已从主图自动接收）。
        """
        # 直接从子图 state 读取（已从主图自动接收）
        user_query = state.get("user_query", "")

        if not evidence_hits:
            return f"抱歉，未找到与「{user_query[:50]}」相关的知识文档。请尝试换一种表述或提供更多上下文。"

        return f"抱歉，无法基于现有证据回答「{user_query[:50]}」。请尝试提供更详细的问题描述。"


# ============================================================================
# 知识问答子图
# ============================================================================

class KnowledgeQASubgraph(BaseSubgraph):
    """知识问答子图

    完整的知识问答流程：
    query_rewriter -> retrieve_wiki -> retrieve_code -> merge_evidence -> knowledge_answer

    子图内部完成检索和生成，对外提供端到端的问答能力。

    所有 retriever 从全局单例获取，无需外部传递。

    Usage:
        subgraph = KnowledgeQASubgraph(config=KnowledgeQAConfig())
        result = subgraph.run(state)
    """

    def __init__(
        self,
        config: KnowledgeQAConfig | None = None,
        *,
        debug_enabled: bool = False,
    ) -> None:
        """初始化知识问答子图

        Args:
            config: 子图配置（可选，使用默认配置）
            debug_enabled: 是否启用调试模式（输出 debug_info）
        """
        if config is None:
            config = KnowledgeQAConfig()

        # 调用父类初始化（所有依赖从全局单例获取）
        super().__init__(config=config)

        # 创建 knowledge_answer 节点实例（内部获取单例）
        self._knowledge_answer_node = KnowledgeAnswerNode()

        # 调试模式开关
        self.debug_enabled = debug_enabled

    def _process_output(self, result: dict[str, Any]) -> dict[str, Any]:
        """处理子图输出

        自动关联模式下，LangGraph 会自动合并同名字段。
        此方法仅在 debug 模式下添加 debug_info。

        Args:
            result: 子图执行结果

        Returns:
            主图状态增量
        """
        if self.debug_enabled:
            # 复制结果并添加 debug_info
            output = dict(result)
            output["debug_info"] = build_debug_info(result)
            return output
        # 非调试模式，直接返回结果（LangGraph 自动合并同名字段）
        return result

    def build_graph(self) -> Any:
        """构建知识问答子图

        子图结构：
        START -> query_rewriter -> retrieve_wiki -> retrieve_code -> merge_evidence -> knowledge_answer -> END

        使用独立的 KnowledgeQAState，与主图 WorkflowState 通过同名字段自动关联。

        Returns:
            编译后的 LangGraph 图
        """
        graph = StateGraph(KnowledgeQAState)

        # 添加节点
        graph.add_node("query_rewriter", self._run_query_rewriter)
        graph.add_node("retrieve_wiki", self._run_retrieve_wiki)
        graph.add_node("retrieve_code", self._run_retrieve_code)
        graph.add_node("merge_evidence", self._run_merge_evidence)
        graph.add_node("knowledge_answer", self._run_knowledge_answer)

        # 添加边
        graph.add_edge(START, "query_rewriter")
        graph.add_edge("query_rewriter", "retrieve_wiki")
        graph.add_edge("retrieve_wiki", "retrieve_code")
        graph.add_edge("retrieve_code", "merge_evidence")
        graph.add_edge("merge_evidence", "knowledge_answer")
        graph.add_edge("knowledge_answer", END)

        return graph.compile()

    def _run_node_with_log(
        self,
        node_name: str,
        node_func: Callable[[], dict[str, Any]],
        state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """执行节点并打印日志

        Args:
            node_name: 节点名称
            node_func: 节点执行函数（无参数，使用闭包捕获状态）
            state: 子图状态（用于获取 trace_id）

        Returns:
            节点执行结果
        """
        start_time = time.time()
        trace_id = state.get("trace_id", "") if state else ""
        logger.debug(f"[Subgraph:knowledge_qa] 节点开始: {node_name}, trace_id={trace_id}")

        try:
            result = node_func()
            latency_ms = int((time.time() - start_time) * 1000)
            logger.info(
                f"[Subgraph:knowledge_qa] 节点完成: {node_name}, "
                f"trace_id={trace_id}, latency_ms={latency_ms}"
            )
            return result
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.error(
                f"[Subgraph:knowledge_qa] 节点失败: {node_name}, "
                f"trace_id={trace_id}, latency_ms={latency_ms}, error={e}"
            )
            raise

    def _run_query_rewriter(self, state: dict[str, Any]) -> dict[str, Any]:
        """执行查询重写节点

        直接从子图 state 读取用户输入（已从主图自动接收），构建检索计划。

        Args:
            state: 子图状态

        Returns:
            子图状态增量（retrieval_queries, retrieval_plan 等）
        """
        from domain_profile import get_domain_profile

        # 直接从子图 state 读取（已从主图自动接收）
        user_query = str(state.get("user_query", ""))
        module_name = str(state.get("module_name", ""))

        # 构建检索查询
        retrieval_queries = [user_query]
        if module_name:
            retrieval_queries.append(f"{module_name} {user_query}")

        # 从 domain_profile 获取检索配置
        domain_profile = get_domain_profile()
        retrieval_profile = domain_profile.retrieval if domain_profile else None

        # 默认值
        wiki_top_k = 5
        code_top_k = 5
        case_top_k = 2
        final_top_k = 6
        enable_wiki = True
        enable_code = True
        enable_cases = True
        source_weights = {"wiki": 1.0, "code": 1.0, "case": 0.6}
        max_per_source = {"wiki": 4, "code": 4, "case": 1}

        # 从 domain_profile 获取配置（如果存在）
        if retrieval_profile:
            # 获取 hybrid 预设的 top_k 配置
            try:
                preset = retrieval_profile.preset("hybrid")
                wiki_top_k = int(preset.get("wiki_top_k", wiki_top_k))
                code_top_k = int(preset.get("code_top_k", code_top_k))
                case_top_k = int(preset.get("case_top_k", case_top_k))
                final_top_k = int(preset.get("final_top_k", final_top_k))
            except Exception:
                pass

            # 获取启用开关
            enable_wiki = bool(getattr(retrieval_profile, "enable_wiki", enable_wiki))
            enable_code = bool(getattr(retrieval_profile, "enable_code", enable_code))
            enable_cases = bool(getattr(retrieval_profile, "enable_cases", enable_cases))

            # 获取权重配置
            if hasattr(retrieval_profile, "source_weights") and retrieval_profile.source_weights:
                source_weights = dict(retrieval_profile.source_weights)
            if hasattr(retrieval_profile, "max_per_source") and retrieval_profile.max_per_source:
                max_per_source = dict(retrieval_profile.max_per_source)

        # 构建完整的 retrieval_plan
        retrieval_plan = {
            "strategy": "hybrid",
            "enable_wiki": enable_wiki,
            "enable_code": enable_code,
            "enable_cases": enable_cases,
            "wiki_top_k": wiki_top_k,
            "code_top_k": code_top_k,
            "case_top_k": case_top_k,
            "final_top_k": final_top_k,
            "source_weights": source_weights,
            "max_per_source": max_per_source,
            "intent_profile": {
                "is_code_intent": False,
                "is_code_location": False,
                "is_wiki_intent": False,
                "is_issue_analysis": False,
            },
        }

        logger.info(
            f"[Subgraph:knowledge_qa] query_rewriter | "
            f"queries={retrieval_queries} | plan=enable_wiki={enable_wiki},enable_code={enable_code}"
        )

        return {
            "retrieval_queries": retrieval_queries,
            "retrieval_plan": retrieval_plan,
            "query_rewrite_mode": "original",
            "node_trace": self._trace(state, "query_rewriter", f"mode=original,strategy=hybrid"),
        }

    def _run_retrieve_wiki(self, state: dict[str, Any]) -> dict[str, Any]:
        """执行 Wiki 检索节点

        retrieval_plan 已由 query_rewriter 构建，无需临时构建。

        Args:
            state: 子图状态

        Returns:
            子图状态增量字典
        """
        from workflow.nodes.retrieval_flow.retrieve_wiki import run_with_retriever
        from workflow.nodes.retrieval_flow.retrieve_wiki.wiki_retriever import get_wiki_retriever

        return self._run_node_with_log(
            "retrieve_wiki",
            lambda: run_with_retriever(
                retriever=get_wiki_retriever(),
                state=state,
                trace_fn=self._trace,
            ),
            state=state,
        )

    def _run_retrieve_code(self, state: dict[str, Any]) -> dict[str, Any]:
        """执行代码检索节点

        retrieval_plan 已由 query_rewriter 构建，无需临时构建。

        Args:
            state: 子图状态

        Returns:
            子图状态增量字典
        """
        from workflow.nodes.retrieval_flow.retrieve_code import run_with_retriever
        from workflow.nodes.retrieval_flow.retrieve_code.code_retriever import get_code_retriever

        return self._run_node_with_log(
            "retrieve_code",
            lambda: run_with_retriever(
                retriever=get_code_retriever(),
                state=state,
                trace_fn=self._trace,
            ),
            state=state,
        )

    def _run_merge_evidence(self, state: dict[str, Any]) -> dict[str, Any]:
        """执行证据合并节点

        Args:
            state: 子图状态

        Returns:
            子图状态增量字典
        """
        from workflow.nodes.retrieval_flow.merge_evidence import run as merge_evidence_run

        # 构建 service 替代对象（merge_evidence 暂时仍需要 service 接口）
        service_proxy = _ServiceProxy(subgraph=self)

        return self._run_node_with_log(
            "merge_evidence",
            lambda: merge_evidence_run(service_proxy, state),
            state=state,
        )

    def _run_knowledge_answer(self, state: dict[str, Any]) -> dict[str, Any]:
        """执行知识回答节点"""
        start_time = time.time()
        trace_id = state.get("trace_id", "")
        logger.debug(f"[Subgraph:knowledge_qa] 节点开始: knowledge_answer, trace_id={trace_id}")

        try:
            # 构建 service 替代对象（knowledge_answer 节点需要）
            service_proxy = _ServiceProxy(subgraph=self)

            # 调用同步的 run 方法
            result = self._knowledge_answer_node.run(service_proxy, state)
            latency_ms = int((time.time() - start_time) * 1000)
            logger.info(
                f"[Subgraph:knowledge_qa] 节点完成: knowledge_answer, "
                f"trace_id={trace_id}, latency_ms={latency_ms}"
            )
            return result
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.error(
                f"[Subgraph:knowledge_qa] 节点失败: knowledge_answer, "
                f"trace_id={trace_id}, latency_ms={latency_ms}, error={e}"
            )
            raise


# ============================================================================
# Service 代理类（用于兼容需要 service 接口的节点）
# ============================================================================

class _ServiceProxy:
    """Service 代理类

    用于向需要 service 接口的节点提供必要的依赖。
    Retriever 从全局单例获取，不再依赖子图传递。
    """

    def __init__(self, subgraph: KnowledgeQASubgraph) -> None:
        """初始化代理

        Args:
            subgraph: 子图实例
        """
        self._subgraph = subgraph

        from workflow.nodes.retrieval_flow.retrieve_wiki.wiki_retriever import get_wiki_retriever
        from workflow.nodes.retrieval_flow.retrieve_code.code_retriever import get_code_retriever
        from workflow.nodes.retrieval_flow.retrieve_cases import get_case_retriever
        from domain_profile import get_domain_profile

        self._wiki_retriever = get_wiki_retriever()
        self._code_retriever = get_code_retriever()
        self._case_retriever = get_case_retriever()
        self._domain_profile = get_domain_profile()

    def _trace(
        self,
        state: dict[str, Any],
        node_name: str,
        detail: str,
    ) -> list[dict[str, str]]:
        """构建节点追踪

        Returns:
            节点追踪列表
        """
        return self._subgraph._trace(state, node_name, detail)
