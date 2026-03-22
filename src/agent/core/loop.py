# -*- coding: utf-8 -*-
"""Agent Loop - 核心 Agent 循环

参考 v1 节点逻辑，独立实现动态工具调用循环。

核心职责：
1. 接收 Router 分发的任务
2. 根据 Skill 加载对应工具
3. 调用 LLM 决定工具使用
4. 执行工具调用
5. 循环直到任务完成或达到限制

设计说明：
- 使用同步实现，支持端到端同步调用链
- LLM 调用使用 invoke_with_tools() 同步方法
- 工具执行使用 tool.invoke() 同步方法
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from agent.llm.client import LLMClient, LLMResponse, ToolCall
from agent.state import AgentState, AgentStatus, ToolCallRecord
from agent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class AgentLoopConfig:
    """Agent Loop 配置

    Attributes:
        max_steps: 最大循环步数
        timeout_seconds: 超时时间 (秒)
        system_prompt: 系统提示词
        max_message_pairs: 保留的最大消息对数（滑动窗口，0 表示不限制）
        max_tool_result_length: 单个工具结果最大长度（0 表示不限制）
    """
    max_steps: int = 10
    timeout_seconds: int = 120
    system_prompt: str = ""
    max_message_pairs: int = 5  # 保留最近 5 轮对话（每轮包含 assistant + tool 消息）
    max_tool_result_length: int = 4000  # 单个工具结果最大 4000 字符

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentLoopConfig":
        """从字典创建配置"""
        return cls(
            max_steps=data.get("max_steps", 10),
            timeout_seconds=data.get("timeout_seconds", 120),
            system_prompt=data.get("system_prompt", ""),
            max_message_pairs=data.get("max_message_pairs", 5),
            max_tool_result_length=data.get("max_tool_result_length", 4000),
        )


@dataclass
class AgentLoopResult:
    """Agent Loop 执行结果

    Attributes:
        success: 是否成功
        answer: 最终答案
        tool_calls: 工具调用记录
        steps: 总步数
        error: 错误信息
        latency_ms: 总耗时 (毫秒)
        is_timeout: 是否超时
    """
    success: bool
    answer: str = ""
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    steps: int = 0
    error: str = ""
    latency_ms: int = 0
    is_timeout: bool = False

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "success": self.success,
            "answer": self.answer,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "steps": self.steps,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "is_timeout": self.is_timeout,
        }


class AgentLoop:
    """Agent 循环

    参考 v1 节点逻辑，独立实现动态工具调用循环。

    使用示例：
        # 使用默认配置（从全局单例获取组件）
        loop = AgentLoop(config=AgentLoopConfig(max_steps=10))

        # 或显式传入组件
        loop = AgentLoop(
            llm_client=llm_client,
            tool_registry=tool_registry,
            config=AgentLoopConfig(max_steps=10),
        )

        result = loop.run(state)  # 同步调用
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        tool_registry: ToolRegistry | None = None,
        config: AgentLoopConfig | None = None,
    ):
        """初始化 Agent Loop

        优先级：参数传递 > 全局单例

        Args:
            llm_client: LLM 客户端（可选，默认从全局单例获取）
            tool_registry: 工具注册中心（可选，默认从全局单例获取）
            config: Agent Loop 配置
        """
        # 从全局单例获取组件（如果未通过参数传递）
        if llm_client is None:
            from agent.llm.client import get_llm_client
            llm_client = get_llm_client()

        if tool_registry is None:
            from agent.tools.registry import get_tool_registry
            tool_registry = get_tool_registry()

        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.config = config or AgentLoopConfig()

        logger.info(
            f"[AgentLoop] 初始化完成, "
            f"max_steps={self.config.max_steps}, "
            f"timeout={self.config.timeout_seconds}s, "
            f"tools={tool_registry.total_count}"
        )

    def run(self, state: AgentState) -> AgentLoopResult:
        """执行 Agent 循环（同步版本）

        Args:
            state: Agent 状态

        Returns:
            AgentLoopResult 执行结果
        """
        start_time = time.time()
        tool_calls_history: list[ToolCallRecord] = []

        logger.info(
            f"[AgentLoop] 开始执行, "
            f"trace_id={state.get('trace_id')}, "
            f"query={state.get('user_query', '')[:50]}..."
        )

        # 更新状态
        state["status"] = AgentStatus.RUNNING.value

        try:
            # 获取可用工具
            tools = self._get_available_tools(state)
            tool_schemas = [self._get_tool_schema(tool) for tool in tools]

            # 打印可用工具摘要日志
            local_tools = [t.name for t in tools if t.name in self.tool_registry.list_local_tool_names()]
            mcp_tools = [t.name for t in tools if t.name in self.tool_registry.list_mcp_tool_names()]
            logger.info(
                f"[AgentLoop] 当前可用工具: 总数={len(tools)}, "
                f"本地工具({len(local_tools)})={local_tools}, "
                f"MCP工具({len(mcp_tools)})={mcp_tools}"
            )

            # 构建初始消息
            messages = self._build_initial_messages(state)

            # 循环执行
            for step in range(1, self.config.max_steps + 1):
                state["current_step"] = step
                logger.debug(f"[AgentLoop] 步骤 {step}/{self.config.max_steps}")

                # 检查超时
                if time.time() - start_time > self.config.timeout_seconds:
                    logger.warning(f"[AgentLoop] 超时, steps={step}")
                    return AgentLoopResult(
                        success=False,
                        answer="任务执行超时，请简化问题或分步提问",
                        tool_calls=tool_calls_history,
                        steps=step,
                        is_timeout=True,
                        latency_ms=int((time.time() - start_time) * 1000),
                    )

                # 调用 LLM（同步）
                state["status"] = AgentStatus.RUNNING.value
                response = self._invoke_llm(messages, tool_schemas)

                # 检查是否完成 (无工具调用)
                if not response.has_tool_calls:
                    logger.info(f"[AgentLoop] 完成, steps={step}, latency={response.latency_ms}ms")
                    state["status"] = AgentStatus.COMPLETED.value
                    state["is_complete"] = True

                    return AgentLoopResult(
                        success=True,
                        answer=response.content,
                        tool_calls=tool_calls_history,
                        steps=step,
                        latency_ms=int((time.time() - start_time) * 1000),
                    )

                # 执行工具调用（同步）
                state["status"] = AgentStatus.TOOL_CALLING.value
                for tool_call in response.tool_calls:
                    record = self._execute_tool(tool_call, state)
                    tool_calls_history.append(record)

                    # 添加工具调用消息
                    messages.append({
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [tool_call.to_dict()],
                    })
                    messages.append({
                        "role": "tool",
                        "content": record.result,
                        "tool_call_id": tool_call.id,
                    })

                # 应用滑动窗口截断（防止消息历史无限增长）
                messages = self._truncate_messages(messages)

            # 达到最大步数
            logger.warning(f"[AgentLoop] 达到最大步数: {self.config.max_steps}")
            state["status"] = AgentStatus.COMPLETED.value

            return AgentLoopResult(
                success=True,
                answer=response.content if "response" in dir() else "任务执行完成，但未生成最终答案",
                tool_calls=tool_calls_history,
                steps=self.config.max_steps,
                latency_ms=int((time.time() - start_time) * 1000),
            )

        except Exception as e:
            logger.error(f"[AgentLoop] 执行失败: {e}")
            state["status"] = AgentStatus.FAILED.value

            return AgentLoopResult(
                success=False,
                error=str(e),
                tool_calls=tool_calls_history,
                steps=state.get("current_step", 0),
                latency_ms=int((time.time() - start_time) * 1000),
            )

    def _get_available_tools(self, state: AgentState):
        """获取可用工具

        优先级：
        1. Skill 定义的 tool_whitelist
        2. 所有工具（包括本地工具和 MCP 工具）

        Args:
            state: Agent 状态

        Returns:
            工具列表
        """
        whitelist = state.get("tool_whitelist", [])
        if whitelist:
            return self.tool_registry.get_tools(whitelist=whitelist)
        # 默认获取所有工具（包括本地工具和 MCP 工具）
        return self.tool_registry.get_tools(include_local=True, include_mcp=True)

    def _get_tool_schema(self, tool: Any) -> dict[str, Any]:
        """获取工具的 OpenAI schema

        支持两种类型的工具：
        1. 自定义工具（有 to_openai_schema 方法）
        2. LangChain StructuredTool（需要转换）

        Args:
            tool: 工具对象

        Returns:
            OpenAI 工具 schema 字典
        """
        # 优先使用自定义的 to_openai_schema 方法
        if hasattr(tool, "to_openai_schema") and callable(tool.to_openai_schema):
            return tool.to_openai_schema()

        # 回退到 LangChain 标准格式转换
        from langchain_core.utils.function_calling import convert_to_openai_function

        try:
            # LangChain 工具转换为 OpenAI function 格式
            func = convert_to_openai_function(tool)
            return {
                "type": "function",
                "function": func,
            }
        except Exception as e:
            logger.warning(f"[AgentLoop] 工具 {getattr(tool, 'name', 'unknown')} schema 转换失败: {e}")
            # 返回最基本的 schema
            return {
                "type": "function",
                "function": {
                    "name": getattr(tool, "name", "unknown"),
                    "description": getattr(tool, "description", ""),
                    "parameters": {"type": "object", "properties": {}},
                },
            }

    def _build_initial_messages(self, state: AgentState) -> list[dict[str, Any]]:
        """构建初始消息列表

        Args:
            state: Agent 状态

        Returns:
            消息列表
        """
        messages: list[dict[str, Any]] = []

        # 添加历史对话
        history = state.get("history", [])
        user_query = state.get("user_query", "")

        for msg in history:
            messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            })

        # 检查是否需要添加当前用户查询
        # 如果 history 的最后一条用户消息与 user_query 相同，则不重复添加
        last_user_msg = ""
        for msg in reversed(history):
            if msg.get("role") == "user":
                last_user_msg = msg.get("content", "")
                break

        # 只有当 history 中没有当前用户消息时才添加
        if not last_user_msg or last_user_msg != user_query:
            messages.append({
                "role": "user",
                "content": user_query,
            })

        return messages

    def _invoke_llm(
        self,
        messages: list[dict[str, Any]],
        tool_schemas: list[dict[str, Any]],
    ) -> LLMResponse:
        """调用 LLM（同步版本）

        Args:
            messages: 消息列表
            tool_schemas: 工具 Schema 列表

        Returns:
            LLM 响应
        """
        # 打印传递给 LLM 的工具列表
        tool_names_in_schema = [schema.get("function", {}).get("name", "unknown") for schema in tool_schemas]
        logger.info(
            f"[AgentLoop] 调用 LLM, 传递工具数={len(tool_schemas)}, "
            f"工具列表={tool_names_in_schema}"
        )

        return self.llm_client.invoke_with_tools(
            messages=messages,
            tools=tool_schemas,
            system_prompt=self.config.system_prompt or None,
        )

    def _execute_tool(
        self,
        tool_call: ToolCall,
        state: AgentState,
    ) -> ToolCallRecord:
        """执行工具调用（同步版本）

        Args:
            tool_call: 工具调用
            state: Agent 状态

        Returns:
            工具调用记录
        """
        start_time = time.time()
        tool_name = tool_call.name
        arguments = tool_call.args

        logger.debug(f"[AgentLoop] 执行工具: {tool_name}, args={arguments}")

        # 获取工具
        tool = self.tool_registry.get_tool(tool_name)
        if tool is None:
            error_msg = f"工具不存在: {tool_name}"
            logger.warning(f"[AgentLoop] {error_msg}")
            return ToolCallRecord(
                tool_name=tool_name,
                arguments=arguments,
                result="",
                success=False,
                error=error_msg,
                latency_ms=int((time.time() - start_time) * 1000),
            )

        # 执行工具（同步）
        try:
            # LangChain BaseTool.invoke() 同步执行
            result = tool.invoke(arguments)
            latency_ms = int((time.time() - start_time) * 1000)

            # 截断工具结果（防止结果过长撑爆 context window）
            result = self._truncate_tool_result(result)

            logger.debug(
                f"[AgentLoop] 工具执行成功: {tool_name}, "
                f"latency={latency_ms}ms, result_len={len(result) if result else 0}"
            )

            return ToolCallRecord(
                tool_name=tool_name,
                arguments=arguments,
                result=result if result else "",
                success=True,
                latency_ms=latency_ms,
            )
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            error_msg = str(e)
            logger.error(f"[AgentLoop] 工具执行失败: {tool_name}, error={error_msg}")

            return ToolCallRecord(
                tool_name=tool_name,
                arguments=arguments,
                result="",
                success=False,
                error=error_msg,
                latency_ms=latency_ms,
            )

    def _truncate_tool_result(self, result: str) -> str:
        """截断工具结果

        当工具结果过长时进行截断，防止撑爆 context window。

        Args:
            result: 工具返回结果

        Returns:
            截断后的结果
        """
        if not result:
            return result

        max_length = self.config.max_tool_result_length
        if max_length <= 0:
            return result

        if len(result) <= max_length:
            return result

        truncated = result[:max_length]
        truncated += f"\n\n... [结果已截断，原始长度: {len(result)} 字符]"

        logger.debug(
            f"[AgentLoop] 工具结果已截断: "
            f"原始长度={len(result)}, 截断后={len(truncated)}"
        )

        return truncated

    def _truncate_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """滑动窗口截断消息历史

        保留最近 N 轮对话，防止消息历史无限增长导致：
        1. 内存溢出
        2. Context Window 超限
        3. Token 消耗过大

        消息结构说明：
        - 每轮对话包含：assistant (tool_calls) + tool (results)
        - 始终保留第一条 user 消息（用户原始问题）
        - 始终保留最新的消息

        Args:
            messages: 消息列表

        Returns:
            截断后的消息列表
        """
        max_pairs = self.config.max_message_pairs
        if max_pairs <= 0:
            # 0 表示不限制
            return messages

        if len(messages) <= max_pairs * 2 + 2:
            # 消息数量在限制内，无需截断
            return messages

        # 分离 system 消息、user 消息和其他消息
        system_messages = [m for m in messages if m.get("role") == "system"]
        user_messages = [m for m in messages if m.get("role") == "user"]
        other_messages = [m for m in messages if m.get("role") not in ("system", "user")]

        # 保留第一条 user 消息（用户原始问题）
        first_user_msg = user_messages[0] if user_messages else None

        # 计算需要保留的消息数量（滑动窗口）
        # max_pairs 表示保留的轮数，每轮约 2 条消息（assistant + tool）
        max_other_messages = max_pairs * 2

        # 从最新的消息开始保留
        truncated_other = other_messages[-max_other_messages:] if len(other_messages) > max_other_messages else other_messages

        # 重建消息列表
        result: list[dict[str, Any]] = []

        # 添加 system 消息
        result.extend(system_messages)

        # 添加第一条 user 消息
        if first_user_msg:
            result.append(first_user_msg)

        # 添加截断后的其他消息
        result.extend(truncated_other)

        # 记录截断日志
        if len(messages) > len(result):
            logger.info(
                f"[AgentLoop] 消息历史截断: "
                f"原始数量={len(messages)}, 截断后={len(result)}, "
                f"保留轮数={max_pairs}"
            )

        return result
