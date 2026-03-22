# -*- coding: utf-8 -*-
"""Agent 状态定义

简化版本：Agent 只负责 LLM 调用（含问答和工具使用）。
路由、证据、调试信息由 Workflow 层处理。

保留内容：
- AgentStatus: Agent 状态枚举（loop.py 和 finalize.py 导入使用）
- ToolCallRecord: 工具调用记录（loop.py 和 finalize.py 导入使用）
- AgentState: Agent 输入状态（最小化，仅 4 个字段）
"""
from dataclasses import dataclass, field
from enum import Enum
from time import perf_counter
from typing import Any, TypedDict


class AgentStatus(str, Enum):
    """Agent 状态枚举

    用于追踪 Agent 循环的当前状态。
    """
    INIT = "init"                     # 初始化
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
        """转换为字典

        Returns:
            包含工具调用信息的字典
        """
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "result": self.result[:500] if self.result else "",  # 截断结果
            "success": self.success,
            "error": self.error,
            "latency_ms": self.latency_ms,
        }


class AgentState(TypedDict, total=False):
    """Agent 状态 - 最小化版本

    Agent 只负责 LLM 调用（含问答和工具使用）。
    路由、证据、调试信息由 Workflow 层处理。

    保留字段（4个）：
        trace_id: 追踪 ID（日志用）
        user_query: 用户提示词（已构建）
        history: 对话历史
        tool_whitelist: 可用工具白名单

    设计说明：
        - 所有字段都是可选的（total=False）
        - AgentLoop 使用 state.get() 方式读取，兼容空值
        - 循环内部状态（messages, tool_calls 等）由 AgentLoop 内部管理
        - 输出通过 AgentLoopResult（定义在 loop.py）返回
    """
    # 必需输入
    trace_id: str              # 追踪 ID（日志用）
    user_query: str            # 用户提示词（已构建）
    history: list[dict[str, Any]]  # 对话历史

    # 可选配置
    tool_whitelist: list[str]  # 可用工具白名单
