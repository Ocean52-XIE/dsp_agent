# -*- coding: utf-8 -*-
"""Agent Finalize - 结果整理

参考 v1 finalize_response 逻辑，独立实现以支持 Agent 模式。

核心职责：
1. 格式化输出答案
2. 收集引用/证据
3. 构建响应结构
4. 添加调试信息
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from agent.state import AgentState, AgentStatus, ToolCallRecord

logger = logging.getLogger(__name__)


# 场景标签映射
SCENE_LABELS = {
    "knowledge_qa": "知识问答",
    "issue_analysis": "问题分析",
    "code_generation": "代码实现建议",
    "skill_execution": "技能执行",
    "out_of_scope": "领域外问题",
    "small_talk": "闲聊",
    "general": "通用处理",
}


@dataclass
class FinalizeConfig:
    """Finalize 配置

    Attributes:
        max_citations: 最大引用数量
        include_tool_calls: 是否包含工具调用记录
        include_debug_info: 是否包含调试信息
        truncate_answer: 是否截断答案
        max_answer_length: 最大答案长度
    """
    max_citations: int = 6
    include_tool_calls: bool = True
    include_debug_info: bool = False
    truncate_answer: bool = False
    max_answer_length: int = 8000

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FinalizeConfig":
        """从字典创建配置"""
        return cls(
            max_citations=int(data.get("max_citations", 4)),
            include_tool_calls=bool(data.get("include_tool_calls", True)),
            include_debug_info=bool(data.get("include_debug_info", False)),
            truncate_answer=bool(data.get("truncate_answer", False)),
            max_answer_length=int(data.get("max_answer_length", 8000)),
        )


@dataclass
class AgentResponse:
    """Agent 响应

    标准化的 Agent 输出格式。

    Attributes:
        role: 角色 (固定为 assistant)
        content: 响应内容
        kind: 场景类型
        intent: 意图类型
        status: 状态
        trace_id: 追踪 ID
        citations: 引用列表
        tool_calls: 工具调用记录
        analysis: 分析结果
        debug: 调试信息
        latency_ms: 总耗时
    """
    role: str = "assistant"
    content: str = ""
    kind: str = "general"
    intent: str = ""
    status: str = "completed"
    trace_id: str = ""
    citations: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    analysis: dict[str, Any] = field(default_factory=dict)
    debug: dict[str, Any] = field(default_factory=dict)
    latency_ms: int = 0
    success: bool = True
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        result = {
            "role": self.role,
            "content": self.content,
            "kind": self.kind,
            "intent": self.intent,
            "status": self.status,
            "trace_id": self.trace_id,
            "success": self.success,
        }

        if self.citations:
            result["citations"] = self.citations

        if self.tool_calls:
            result["tool_calls"] = self.tool_calls

        if self.analysis:
            result["analysis"] = self.analysis

        if self.debug:
            result["debug"] = self.debug

        if self.latency_ms > 0:
            result["latency_ms"] = self.latency_ms

        if self.error:
            result["error"] = self.error

        return result


class AgentFinalize:
    """Agent 结果整理器

    参考 v1 finalize_response 逻辑，独立实现以支持 Agent 模式。

    使用示例：
        finalize = AgentFinalize(config=FinalizeConfig())
        response = finalize.run(state, loop_result)
    """

    def __init__(self, config: FinalizeConfig | None = None):
        """初始化 Agent Finalize

        Args:
            config: Finalize 配置
        """
        self.config = config or FinalizeConfig()
        logger.info(
            f"[AgentFinalize] 初始化完成, "
            f"max_citations={self.config.max_citations}, "
            f"include_debug={self.config.include_debug_info}"
        )

    def run(
        self,
        state: AgentState,
        answer: str = "",
        tool_calls: list[ToolCallRecord] | None = None,
        error: str = "",
        latency_ms: int = 0,
    ) -> AgentResponse:
        """执行结果整理

        Args:
            state: Agent 状态
            answer: 最终答案
            tool_calls: 工具调用记录
            error: 错误信息
            latency_ms: 总耗时

        Returns:
            AgentResponse 响应对象
        """
        start_time = time.time()

        # 确定场景类型
        kind = self._determine_kind(state)
        intent = state.get("route", "general")

        # 获取答案
        content = answer or state.get("answer", "")
        if self.config.truncate_answer and len(content) > self.config.max_answer_length:
            content = content[:self.config.max_answer_length] + "..."
            logger.debug(f"[AgentFinalize] 答案已截断: {self.config.max_answer_length}")

        # 收集引用
        citations = self._collect_citations(state)

        # 整理工具调用记录
        tool_call_records = []
        if self.config.include_tool_calls and tool_calls:
            tool_call_records = [tc.to_dict() for tc in tool_calls]

        # 确定状态
        status = self._determine_status(state, error)
        success = not error and status != "failed"

        # 构建响应
        response = AgentResponse(
            role="assistant",
            content=content,
            kind=kind,
            intent=intent,
            status=status,
            trace_id=state.get("trace_id", ""),
            citations=citations,
            tool_calls=tool_call_records,
            success=success,
            error=error,
            latency_ms=latency_ms or int((time.time() - start_time) * 1000),
        )

        # 添加调试信息
        if self.config.include_debug_info:
            response.debug = self._build_debug_info(state, tool_calls)
            response.analysis = self._build_analysis(state)

        logger.info(
            f"[AgentFinalize] 响应构建完成, "
            f"kind={kind}, status={status}, "
            f"citations={len(citations)}, tool_calls={len(tool_call_records)}"
        )

        return response

    def _determine_kind(self, state: AgentState) -> str:
        """确定场景类型

        Args:
            state: Agent 状态

        Returns:
            场景类型字符串
        """
        # 优先使用 domain_route (新架构)
        domain_route = state.get("domain_route", "")
        if domain_route:
            # 映射 domain_route 字符串到场景类型
            domain_kind_mapping = {
                "out_of_scope": "out_of_scope",
                "small_talk": "small_talk",
                "pass_to_agent": "agent_loop",
                "module_routed": "module_routed",
            }
            return domain_kind_mapping.get(domain_route, domain_route)

        # 兼容旧架构的 route 字段
        route = state.get("route", "general")
        kind_mapping = {
            "out_of_scope": "out_of_scope",
            "small_talk": "small_talk",
            "knowledge_qa": "knowledge_qa",
            "issue_analysis": "issue_analysis",
            "code_generation": "code_generation",
            "skill_execution": "skill_execution",
            "general": "general",
        }

        return kind_mapping.get(route, route)

    def _determine_status(self, state: AgentState, error: str) -> str:
        """确定状态

        Args:
            state: Agent 状态
            error: 错误信息

        Returns:
            状态字符串
        """
        if error:
            return "failed"

        current_status = state.get("status", "")
        if current_status == AgentStatus.FAILED.value:
            return "failed"
        if current_status == AgentStatus.TIMEOUT.value:
            return "timeout"

        return "completed"

    def _collect_citations(self, state: AgentState) -> list[dict[str, Any]]:
        """收集引用

        Args:
            state: Agent 状态

        Returns:
            引用列表
        """
        # 优先使用已整理的 citations
        citations = list(state.get("citations", []) or [])
        if citations:
            return self._deduplicate_citations(citations)[:self.config.max_citations]

        # 从各个检索结果中收集
        all_hits: list[dict[str, Any]] = []

        for key, default_source in (
            ("wiki_hits", "wiki"),
            ("code_hits", "code"),
            ("case_hits", "case"),
        ):
            hits = list(state.get(key, []) or [])
            for item in hits:
                row = dict(item)
                if "source_type" not in row:
                    row["source_type"] = row.get("source", default_source)
                all_hits.append(row)

        return self._deduplicate_citations(all_hits)[:self.config.max_citations]

    def _deduplicate_citations(
        self,
        citations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """去重引用

        Args:
            citations: 引用列表

        Returns:
            去重后的引用列表
        """
        seen: set[tuple[str, str, str]] = set()
        result: list[dict[str, Any]] = []

        for item in citations:
            # 跳过 None 或非字典类型的项
            if item is None or not isinstance(item, dict):
                continue
            key = (
                str(item.get("source_type", "")),
                str(item.get("path", "")),
                str(item.get("section", "")),
            )
            if key in seen:
                continue
            seen.add(key)
            result.append(item)

        return result

    def _build_debug_info(
        self,
        state: AgentState,
        tool_calls: list[ToolCallRecord] | None,
    ) -> dict[str, Any]:
        """构建调试信息

        Args:
            state: Agent 状态
            tool_calls: 工具调用记录

        Returns:
            调试信息字典
        """
        return {
            "scene_label": SCENE_LABELS.get(
                state.get("route", "general"),
                state.get("route", "general"),
            ),
            "domain_relevance": state.get("domain_relevance", 0.0),
            "route": state.get("route", "unknown"),
            "active_skill_id": state.get("active_skill_id"),
            "tool_whitelist": state.get("tool_whitelist", []),
            "tool_call_count": len(tool_calls) if tool_calls else 0,
            "current_step": state.get("current_step", 0),
            "node_trace": state.get("debug_info", {}).get("node_trace", []),
        }

    def _build_analysis(self, state: AgentState) -> dict[str, Any]:
        """构建分析结果

        Args:
            state: Agent 状态

        Returns:
            分析结果字典
        """
        analysis = state.get("debug_info", {})

        # 添加路由信息
        if "router" not in analysis:
            analysis["router"] = {
                "route": state.get("route", ""),
                "domain_relevance": state.get("domain_relevance", 0.0),
                "active_skill_id": state.get("active_skill_id"),
            }

        return analysis

    def update_state(
        self,
        state: AgentState,
        response: AgentResponse,
    ) -> None:
        """更新状态（便捷方法）

        Args:
            state: Agent 状态
            response: 响应对象
        """
        state["answer"] = response.content
        state["status"] = response.status
        state["citations"] = response.citations

        if "debug_info" not in state:
            state["debug_info"] = {}
        state["debug_info"]["response"] = response.to_dict()
