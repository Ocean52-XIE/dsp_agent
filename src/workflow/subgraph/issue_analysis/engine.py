# -*- coding: utf-8 -*-
"""IssueAnalysis 子图引擎

完整的问题分析子图，内部包含：
- query_rewriter: 查询重写
- retrieve_wiki: Wiki 检索
- retrieve_code: 代码检索
- merge_evidence: 证据合并
- issue_analysis: 问题分析生成

可被主工作流调用，提供端到端的问题分析能力。

## 状态设计（自动关联模式）

子图使用独立的 IssueAnalysisState，与主图 WorkflowState 通过同名字段自动关联。

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
from workflow.subgraph.issue_analysis.config import IssueAnalysisConfig
from workflow.subgraph.issue_analysis.state import (
    IssueAnalysisState,
    build_debug_info,
)
from workflow.nodes.agent_loop import BaseAgentLoopNode, AgentLoopNodeConfig

logger = logging.getLogger(__name__)


# ============================================================================
# 问题分析节点
# ============================================================================

class IssueAnalysisNode(BaseAgentLoopNode):
    """问题分析节点

    继承 BaseAgentLoopNode，实现问题分析场景：
    - 分析用户遇到的问题/错误
    - 提供排查思路和解决方案
    - 支持 Skill Tool 调用
    - 支持 Fallback 机制

    所有依赖从全局单例获取，无需外部传递。
    """

    def __init__(self) -> None:
        """初始化问题分析节点

        所有依赖从全局单例获取。
        """
        config = AgentLoopNodeConfig(
            node_name="issue_analysis",
            response_kind="issue_analysis",
            # 提示词模板（节点会通过 _get_system_prompt/_get_user_prompt 覆盖）
            system_prompt_template=(
                "你是企业问题分析助手。"
                "请基于提供的检索证据和错误信息，帮助用户分析问题并提供排查思路。"
                "如果需要执行特定技能来获取诊断信息，请使用 skill_tool。"
                "输出中文，结构尽量为：问题定位 -> 排查步骤 -> 解决方案。"
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
            max_iterations=5,
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
            # 优先从 prompts 字典获取 issue_system
            prompts = getattr(domain_profile, "prompts", None)
            if prompts and isinstance(prompts, dict):
                issue_system = prompts.get("issue_system")
                if issue_system:
                    return issue_system

        return """你是一个专业的问题分析助手。
请基于提供的检索证据和错误信息，帮助用户分析问题并提供排查思路。

要求：
1. 优先使用检索证据中的信息
2. 提供清晰的排查步骤
3. 如果证据不足以分析问题，请诚实说明
4. 给出可能的解决方案"""

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
            evidence_parts = []
            for i, hit in enumerate(evidence_hits[:5], 1):
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

        return f"""请分析以下问题：

问题: {user_query}
{module_context}

相关证据:
{evidence_text if evidence_text else '无相关证据'}

请提供问题分析和排查建议："""

    def _build_fallback(
        self,
        state: dict[str, Any],
        module_name: str,
        module_hint: str,
        related_modules: list[dict[str, Any]],
        evidence_hits: list[dict[str, Any]],
    ) -> str:
        """构建 Fallback 响应"""
        user_query = state.get("user_query", "")

        if not evidence_hits:
            return f"抱歉，未找到与「{user_query[:50]}」相关的知识文档，无法提供排查建议。请尝试提供更多错误信息或日志。"

        return f"抱歉，基于现有证据无法分析「{user_query[:50]}」。请尝试提供更详细的错误信息、日志或堆栈跟踪。"


# ============================================================================
# 问题分析子图
# ============================================================================

class IssueAnalysisSubgraph(BaseSubgraph):
    """问题分析子图

    完整的问题分析流程：
    query_rewriter -> retrieve_wiki -> retrieve_code -> merge_evidence -> issue_analysis

    子图内部完成检索和生成，对外提供端到端的问题分析能力。

    所有 retriever 从全局单例获取，无需外部传递。

    Usage:
        subgraph = IssueAnalysisSubgraph(config=IssueAnalysisConfig())
        result = subgraph.run(state)
    """

    def __init__(
        self,
        config: IssueAnalysisConfig | None = None,
        *,
        debug_enabled: bool = False,
    ) -> None:
        """初始化问题分析子图

        Args:
            config: 子图配置（可选，使用默认配置）
            debug_enabled: 是否启用调试模式（输出 debug_info）
        """
        if config is None:
            config = IssueAnalysisConfig()

        # 调用父类初始化（所有依赖从全局单例获取）
        super().__init__(config=config)

        # 创建 issue_analysis 节点实例（内部获取单例）
        self._issue_analysis_node = IssueAnalysisNode()

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
        """构建问题分析子图

        子图结构：
        START -> query_rewriter -> retrieve_wiki -> retrieve_code -> merge_evidence -> issue_analysis -> END

        使用独立的 IssueAnalysisState，与主图 WorkflowState 通过同名字段自动关联。

        Returns:
            编译后的 LangGraph 图
        """
        graph = StateGraph(IssueAnalysisState)

        # 添加节点
        graph.add_node("query_rewriter", self._run_query_rewriter)
        graph.add_node("retrieve_wiki", self._run_retrieve_wiki)
        graph.add_node("retrieve_code", self._run_retrieve_code)
        graph.add_node("merge_evidence", self._run_merge_evidence)
        graph.add_node("issue_analysis", self._run_issue_analysis)

        # 添加边
        graph.add_edge(START, "query_rewriter")
        graph.add_edge("query_rewriter", "retrieve_wiki")
        graph.add_edge("retrieve_wiki", "retrieve_code")
        graph.add_edge("retrieve_code", "merge_evidence")
        graph.add_edge("merge_evidence", "issue_analysis")
        graph.add_edge("issue_analysis", END)

        return graph.compile()

    def run(self, state: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """执行子图（同步版本）

        Args:
            state: 工作流状态
            **kwargs: 其他参数

        Returns:
            执行结果
        """
        # 保存状态供节点使用
        self._current_state = state
        return super().run(state, **kwargs)

    def _run_node_with_log(
        self,
        node_name: str,
        node_func: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        """执行节点并打印日志（同步版本）

        Args:
            node_name: 节点名称
            node_func: 节点执行函数（无参数）

        Returns:
            节点执行结果
        """
        start_time = time.time()
        trace_id = self._current_state.get("trace_id", "")
        logger.debug(f"[Subgraph:issue_analysis] 节点开始: {node_name}, trace_id={trace_id}")

        try:
            result = node_func()
            latency_ms = int((time.time() - start_time) * 1000)
            logger.info(
                f"[Subgraph:issue_analysis] 节点完成: {node_name}, "
                f"trace_id={trace_id}, latency_ms={latency_ms}"
            )
            return result
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.error(
                f"[Subgraph:issue_analysis] 节点失败: {node_name}, "
                f"trace_id={trace_id}, latency_ms={latency_ms}, error={e}"
            )
            raise

    def _run_query_rewriter(self, state: dict[str, Any]) -> dict[str, Any]:
        """执行查询重写节点"""
        user_query = str(state.get("user_query", ""))
        module_name = str(state.get("module_name", ""))

        # 简化版：直接使用原始查询
        retrieval_queries = [user_query]

        # 如果有模块名，添加模块相关的查询
        if module_name:
            retrieval_queries.append(f"{module_name} {user_query}")

        return {
            "retrieval_queries": retrieval_queries,
            "query_rewrite_mode": "original",
            "node_trace": self._trace(state, "query_rewriter", "mode=original"),
        }

    def _run_retrieve_wiki(self, state: dict[str, Any]) -> dict[str, Any]:
        """执行 Wiki 检索节点"""
        from workflow.nodes.retrieval_flow.retrieve_wiki import run_with_retriever
        from workflow.nodes.retrieval_flow.retrieve_wiki.wiki_retriever import get_wiki_retriever

        # 构建 retrieval_plan（如果没有的话）
        if "retrieval_plan" not in state:
            state = {**state, "retrieval_plan": {"enable_wiki": True, "wiki_top_k": 5}}

        return self._run_node_with_log(
            "retrieve_wiki",
            lambda: run_with_retriever(
                retriever=get_wiki_retriever(),
                state=state,
                trace_fn=self._trace,
            ),
        )

    def _run_retrieve_code(self, state: dict[str, Any]) -> dict[str, Any]:
        """执行代码检索节点"""
        from workflow.nodes.retrieval_flow.retrieve_code import run_with_retriever
        from workflow.nodes.retrieval_flow.retrieve_code.code_retriever import get_code_retriever

        # 构建 retrieval_plan（如果没有的话）
        if "retrieval_plan" not in state:
            state = {**state, "retrieval_plan": {"enable_code": True, "code_top_k": 5}}
        else:
            state = {**state, "retrieval_plan": {**state["retrieval_plan"], "enable_code": True}}

        return self._run_node_with_log(
            "retrieve_code",
            lambda: run_with_retriever(
                retriever=get_code_retriever(),
                state=state,
                trace_fn=self._trace,
            ),
        )

    def _run_merge_evidence(self, state: dict[str, Any]) -> dict[str, Any]:
        """执行证据合并节点"""
        from workflow.nodes.retrieval_flow.merge_evidence import run as merge_evidence_run

        # 构建 service 替代对象（merge_evidence 暂时仍需要 service 接口）
        service_proxy = _ServiceProxy(subgraph=self)

        return self._run_node_with_log(
            "merge_evidence",
            lambda: merge_evidence_run(service_proxy, state),
        )

    def _run_issue_analysis(self, state: dict[str, Any]) -> dict[str, Any]:
        """执行问题分析节点"""
        start_time = time.time()
        trace_id = state.get("trace_id", "")
        logger.debug(f"[Subgraph:issue_analysis] 节点开始: issue_analysis, trace_id={trace_id}")

        try:
            # 构建 service 替代对象（issue_analysis 节点需要）
            service_proxy = _ServiceProxy(subgraph=self)

            # 调用同步的 run 方法
            result = self._issue_analysis_node.run(service_proxy, state)
            latency_ms = int((time.time() - start_time) * 1000)
            logger.info(
                f"[Subgraph:issue_analysis] 节点完成: issue_analysis, "
                f"trace_id={trace_id}, latency_ms={latency_ms}"
            )
            return result
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.error(
                f"[Subgraph:issue_analysis] 节点失败: issue_analysis, "
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

    def __init__(self, subgraph: IssueAnalysisSubgraph) -> None:
        """初始化代理

        Args:
            subgraph: 子图实例
        """
        self._subgraph = subgraph

        # 从全局单例获取 retriever
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
