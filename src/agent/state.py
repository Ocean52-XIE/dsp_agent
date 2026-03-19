# -*- coding: utf-8 -*-
"""Agent 状态定义

参考 v1 WorkflowState 设计，独立实现以支持：
- 动态工具调用循环
- Skill 激活状态
- 工具调用历史
- Checkpoint 持久化
"""
from dataclasses import dataclass, field
from enum import Enum
from time import perf_counter
from typing import Any, TypedDict


class AgentStatus(str, Enum):
    """Agent 状态枚举"""
    INIT = "init"                     # 初始化
    ROUTING = "routing"               # 路由中
    RUNNING = "running"               # Agent Loop 运行中
    TOOL_CALLING = "tool_calling"     # 工具调用中
    COMPLETED = "completed"           # 已完成
    FAILED = "failed"                 # 失败
    TIMEOUT = "timeout"               # 超时


@dataclass
class ToolCallRecord:
    """工具调用记录

    记录每次工具调用的详细信息，用于追踪和调试。

    Attributes:
        tool_name: 工具名称
        arguments: 调用参数
        result: 返回结果
        success: 是否成功
        error: 错误信息
        latency_ms: 耗时 (毫秒)
        timestamp: 时间戳
    """
    tool_name: str
    arguments: dict[str, Any]
    result: str = ""
    success: bool = True
    error: str = ""
    latency_ms: int = 0
    timestamp: float = field(default_factory=perf_counter)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "result": self.result[:500] if self.result else "",  # 截断结果
            "success": self.success,
            "error": self.error,
            "latency_ms": self.latency_ms,
        }


@dataclass
class AgentLoopResult:
    """Agent Loop 执行结果

    Agent 循环完成后的统一输出格式。

    Attributes:
        success: 是否成功
        answer: 最终答案
        tool_calls: 工具调用记录列表
        steps: 总步数
        active_skill: 激活的 Skill ID
        error: 错误信息
        latency_ms: 总耗时 (毫秒)
    """
    success: bool
    answer: str = ""
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    steps: int = 0
    active_skill: str | None = None
    error: str = ""
    latency_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "success": self.success,
            "answer": self.answer,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "steps": self.steps,
            "active_skill": self.active_skill,
            "error": self.error,
            "latency_ms": self.latency_ms,
        }


class AgentState(TypedDict, total=False):
    """Agent 状态

    参考 v1 WorkflowState 设计，但独立实现以支持 Agent 模式。

    状态字段分为五类：
    1. 会话上下文 - 请求级别信息
    2. Domain Router 结果 - 规则路由决策信息
    3. Agent Loop 状态 - 循环执行状态
    4. LLM 决策结果 - LLM 自决策信息
    5. 输出结果 - 最终输出

    Attributes:
        # 会话上下文
        trace_id: 追踪 ID
        session_id: 会话 ID
        user_query: 用户查询
        original_query: 原始查询 (改写前)
        history: 历史对话

        # Domain Router 结果 (规则层)
        domain_route: Domain Router 路由类型
        domain_relevance: 领域相关性分数
        matched_module: 命中的模块名 (MODULE_ROUTED)

        # Agent Loop 状态
        status: 当前状态
        messages: 对话消息列表
        tool_calls: 工具调用记录
        current_step: 当前步数
        is_complete: 是否完成

        # LLM 决策结果 (LLM 层，由 Agent Loop 填充)
        route: LLM 决定的路由目标
        active_skill_id: 激活的 Skill ID
        tool_whitelist: 可用工具白名单

        # 输出结果
        answer: 最终答案
        citations: 引用列表
        debug_info: 调试信息
    """
    # 会话上下文
    trace_id: str
    session_id: str
    user_query: str
    original_query: str
    history: list[dict[str, Any]]

    # Domain Router 结果 (规则层)
    domain_route: str
    domain_relevance: float
    matched_module: str | None

    # Agent Loop 状态
    status: str
    messages: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    current_step: int
    is_complete: bool

    # LLM 决策结果 (LLM 层)
    route: str
    active_skill_id: str | None
    tool_whitelist: list[str]

    # 输出结果
    answer: str
    citations: list[dict[str, Any]]
    debug_info: dict[str, Any]


def create_initial_state(
    trace_id: str,
    session_id: str,
    user_query: str,
    history: list[dict[str, Any]] | None = None,
) -> AgentState:
    """创建初始 Agent 状态

    Args:
        trace_id: 追踪 ID
        session_id: 会话 ID
        user_query: 用户查询
        history: 历史对话

    Returns:
        初始化的 AgentState
    """
    return AgentState(
        # 会话上下文
        trace_id=trace_id,
        session_id=session_id,
        user_query=user_query.strip(),
        original_query=user_query.strip(),
        history=history or [],

        # Domain Router 结果 (默认值)
        domain_route="",
        domain_relevance=0.0,
        matched_module=None,

        # Agent Loop 状态
        status=AgentStatus.INIT.value,
        messages=[],
        tool_calls=[],
        current_step=0,
        is_complete=False,

        # LLM 决策结果 (默认值，由 Agent Loop 填充)
        route="",
        active_skill_id=None,
        tool_whitelist=[],

        # 输出结果
        answer="",
        citations=[],
        debug_info={},
    )


def state_add_message(
    state: AgentState,
    role: str,
    content: str,
    tool_calls: list[dict[str, Any]] | None = None,
    tool_call_id: str | None = None,
) -> None:
    """向状态添加消息

    Args:
        state: Agent 状态
        role: 角色 (user/assistant/tool)
        content: 内容
        tool_calls: 工具调用 (assistant 角色时)
        tool_call_id: 工具调用 ID (tool 角色时)
    """
    message: dict[str, Any] = {"role": role, "content": content}

    if tool_calls:
        message["tool_calls"] = tool_calls

    if tool_call_id:
        message["tool_call_id"] = tool_call_id

    state["messages"].append(message)


def state_add_tool_call(
    state: AgentState,
    record: ToolCallRecord,
) -> None:
    """向状态添加工具调用记录

    Args:
        state: Agent 状态
        record: 工具调用记录
    """
    state["tool_calls"].append(record.to_dict())
