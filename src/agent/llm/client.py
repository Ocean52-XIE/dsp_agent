# -*- coding: utf-8 -*-
"""LLM 客户端

参考 v1 WorkflowLLMClient 设计，独立实现以支持 Agent 模式：
- 支持 async 原生调用
- 支持工具调用 (Function Calling)
- 支持流式响应
- 详细的请求/响应日志
"""
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from agent.llm.config import LLMConfig
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

# LLM 日志相关常量
LLM_LOG_PROMPT_MAX_LENGTH = 2000
LLM_LOG_RESPONSE_MAX_LENGTH = 2000

# 用于匹配 LLM thinking 内容的正则表达式
_THINKING_PATTERN = re.compile(
    r"<(?:thinking|thinker|think)>[\s\S]*?</(?:thinking|thinker|think)>",
    re.IGNORECASE
)


def strip_thinking_content(text: str) -> str:
    """移除 LLM 响应中的 thinking 标签及其内容

    部分 LLM（如 DeepSeek-R1、Claude 等）会返回 <thinking>...</thinking> 格式的思考过程，
    该函数用于在返回给用户界面之前过滤掉这部分内容。

    Args:
        text: 原始响应文本

    Returns:
        移除 thinking 内容后的文本
    """
    if not text:
        return text
    cleaned = _THINKING_PATTERN.sub("", text)
    return cleaned.strip()


def _truncate_text(text: str, max_length: int = LLM_LOG_PROMPT_MAX_LENGTH) -> str:
    """截断文本用于日志显示

    Args:
        text: 原始文本
        max_length: 最大长度

    Returns:
        截断后的文本
    """
    if not text:
        return ""
    if len(text) <= max_length:
        return text
    return text[:max_length] + f"... (truncated, total {len(text)} chars)"


@dataclass
class ToolCall:
    """工具调用"""
    id: str
    name: str
    args: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "args": self.args,
        }


@dataclass
class LLMResponse:
    """LLM 响应

    Attributes:
        content: 文本内容
        tool_calls: 工具调用列表
        raw_response: 原始响应对象
        latency_ms: 耗时 (毫秒)
    """
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_response: Any = None
    latency_ms: int = 0

    @property
    def has_tool_calls(self) -> bool:
        """是否有工具调用"""
        return len(self.tool_calls) > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "latency_ms": self.latency_ms,
        }


class LLMClient:
    """LLM 客户端

    参考 v1 WorkflowLLMClient 设计，独立实现以支持 Agent 模式。

    使用示例：
        config = LLMConfig.from_env()
        client = LLMClient(config)

        # 普通调用
        response = await client.ainvoke("你好")

        # 带工具调用
        tools = tool_registry.get_all_tool_schemas()
        response = await client.ainvoke_with_tools(
            messages=[{"role": "user", "content": "查一下 CTR"}],
            tools=tools,
        )
    """

    def __init__(self, config: LLMConfig):
        """初始化 LLM 客户端

        Args:
            config: LLM 配置
        """
        self.config = config
        self._llm = self._create_llm()
        self._call_count = 0  # 调用计数器，用于生成唯一 call_id
        logger.info(f"[LLMClient] 初始化完成, model={config.model}")

    def _create_llm(self) -> ChatOpenAI:
        """创建 LangChain LLM 实例"""
        llm_kwargs: dict[str, Any] = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "timeout": self.config.timeout,
        }

        # 设置 API Key
        if self.config.api_key:
            llm_kwargs["api_key"] = self.config.api_key

        # 设置 Base URL (支持本地部署或代理)
        if self.config.base_url:
            llm_kwargs["base_url"] = self.config.base_url

        # 合并额外参数
        llm_kwargs.update(self.config.extra_params)

        return ChatOpenAI(**llm_kwargs)

    # =========================================================================
    # 日志记录方法
    # =========================================================================

    def _log_request(
        self,
        call_id: str,
        messages: list[dict[str, Any]] | None = None,
        lc_messages: list[BaseMessage] | None = None,
        system_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        simple_message: str | None = None,
    ) -> None:
        """记录 LLM 请求日志

        Args:
            call_id: 调用 ID
            messages: 原始消息列表 (可选)
            lc_messages: LangChain 消息列表 (可选)
            system_prompt: 系统提示词 (可选)
            tools: 工具列表 (可选)
            simple_message: 简单消息字符串 (可选)
        """
        # 构建请求摘要
        request_summary: dict[str, Any] = {
            "call_id": call_id,
            "model": self.config.model,
        }

        # 提取系统提示词
        system_content: str | None = None
        if system_prompt:
            system_content = system_prompt
        elif lc_messages:
            for msg in lc_messages:
                if isinstance(msg, SystemMessage):
                    system_content = str(msg.content)
                    break

        if system_content:
            request_summary["system_prompt"] = _truncate_text(system_content)

        # 提取用户消息
        user_messages: list[dict[str, str]] = []
        if simple_message:
            user_messages.append({"role": "user", "content": simple_message})
        elif messages:
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    user_messages.append({
                        "role": role,
                        "content": _truncate_text(str(content)),
                    })
        elif lc_messages:
            for msg in lc_messages:
                if isinstance(msg, HumanMessage):
                    user_messages.append({
                        "role": "user",
                        "content": _truncate_text(str(msg.content)),
                    })

        if user_messages:
            request_summary["user_messages"] = user_messages
            request_summary["user_message_count"] = len(user_messages)

        # 工具信息
        if tools:
            tool_names = [t.get("function", {}).get("name", "unknown") for t in tools]
            request_summary["tools"] = tool_names
            request_summary["tool_count"] = len(tools)

        logger.info(f"[LLMClient] request.started | {json.dumps(request_summary, ensure_ascii=False)}")

    def _log_response(
        self,
        call_id: str,
        response: LLMResponse,
        success: bool = True,
        error: str | None = None,
    ) -> None:
        """记录 LLM 响应日志

        Args:
            call_id: 调用 ID
            response: LLM 响应
            success: 是否成功
            error: 错误信息 (可选)
        """
        response_summary: dict[str, Any] = {
            "call_id": call_id,
            "success": success,
            "latency_ms": response.latency_ms,
        }

        if not success:
            response_summary["error"] = error
            logger.error(f"[LLMClient] request.failed | {json.dumps(response_summary, ensure_ascii=False)}")
            return

        # 响应内容
        if response.content:
            response_summary["content"] = _truncate_text(response.content, LLM_LOG_RESPONSE_MAX_LENGTH)
            response_summary["content_length"] = len(response.content)

        # 工具调用
        if response.tool_calls:
            response_summary["tool_calls"] = [
                {
                    "id": tc.id,
                    "name": tc.name,
                    "args": tc.args,
                }
                for tc in response.tool_calls
            ]
            response_summary["tool_call_count"] = len(response.tool_calls)

        logger.info(f"[LLMClient] request.completed | {json.dumps(response_summary, ensure_ascii=False)}")

    def _generate_call_id(self) -> str:
        """生成调用 ID"""
        self._call_count += 1
        return f"llm_call_{self._call_count:04d}"

    # =========================================================================
    # 同步调用方法
    # =========================================================================

    def invoke(self, message: str) -> LLMResponse:
        """同步调用 LLM

        Args:
            message: 用户消息

        Returns:
            LLMResponse 响应
        """
        call_id = self._generate_call_id()
        start_time = time.time()

        # 记录请求
        self._log_request(call_id, simple_message=message)

        try:
            response = self._llm.invoke([HumanMessage(content=message)])
            latency_ms = int((time.time() - start_time) * 1000)

            content = response.content if hasattr(response, "content") else str(response)
            content = strip_thinking_content(content)

            llm_response = LLMResponse(
                content=content,
                raw_response=response,
                latency_ms=latency_ms,
            )

            # 记录响应
            self._log_response(call_id, llm_response)

            return llm_response
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            error_response = LLMResponse(latency_ms=latency_ms)
            self._log_response(call_id, error_response, success=False, error=str(e))
            raise

    def invoke_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system_prompt: str | None = None,
    ) -> LLMResponse:
        """同步调用 LLM (带工具)

        Args:
            messages: 消息列表
            tools: 工具 Schema 列表
            system_prompt: 系统提示词 (可选)

        Returns:
            LLMResponse 响应
        """
        call_id = self._generate_call_id()
        start_time = time.time()

        try:
            # 构建消息
            lc_messages = self._build_messages(messages, system_prompt)

            # 记录请求
            self._log_request(
                call_id,
                messages=messages,
                lc_messages=lc_messages,
                system_prompt=system_prompt,
                tools=tools,
            )

            # 绑定工具
            llm_with_tools = self._llm.bind_tools(tools)

            # 调用
            response = llm_with_tools.invoke(lc_messages)
            latency_ms = int((time.time() - start_time) * 1000)

            # 解析响应
            content = response.content if hasattr(response, "content") else str(response)
            content = strip_thinking_content(content)

            # 解析工具调用
            tool_calls = self._parse_tool_calls(response)

            llm_response = LLMResponse(
                content=content,
                tool_calls=tool_calls,
                raw_response=response,
                latency_ms=latency_ms,
            )

            # 记录响应
            self._log_response(call_id, llm_response)

            return llm_response
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            error_response = LLMResponse(latency_ms=latency_ms)
            self._log_response(call_id, error_response, success=False, error=str(e))
            raise

    # =========================================================================
    # 异步调用方法
    # =========================================================================

    async def ainvoke(self, message: str) -> LLMResponse:
        """异步调用 LLM

        Args:
            message: 用户消息

        Returns:
            LLMResponse 响应
        """
        call_id = self._generate_call_id()
        start_time = time.time()

        # 记录请求
        self._log_request(call_id, simple_message=message)

        try:
            response = await self._llm.ainvoke([HumanMessage(content=message)])
            latency_ms = int((time.time() - start_time) * 1000)

            content = response.content if hasattr(response, "content") else str(response)
            content = strip_thinking_content(content)

            llm_response = LLMResponse(
                content=content,
                raw_response=response,
                latency_ms=latency_ms,
            )

            # 记录响应
            self._log_response(call_id, llm_response)

            return llm_response
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            error_response = LLMResponse(latency_ms=latency_ms)
            self._log_response(call_id, error_response, success=False, error=str(e))
            raise

    async def ainvoke_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system_prompt: str | None = None,
    ) -> LLMResponse:
        """异步调用 LLM (带工具)

        Args:
            messages: 消息列表
            tools: 工具 Schema 列表
            system_prompt: 系统提示词 (可选)

        Returns:
            LLMResponse 响应
        """
        call_id = self._generate_call_id()
        start_time = time.time()

        try:
            # 构建消息
            lc_messages = self._build_messages(messages, system_prompt)

            # 记录请求
            self._log_request(
                call_id,
                messages=messages,
                lc_messages=lc_messages,
                system_prompt=system_prompt,
                tools=tools,
            )

            # 绑定工具
            llm_with_tools = self._llm.bind_tools(tools)

            # 调用
            response = await llm_with_tools.ainvoke(lc_messages)
            latency_ms = int((time.time() - start_time) * 1000)

            # 解析响应
            content = response.content if hasattr(response, "content") else str(response)
            content = strip_thinking_content(content)

            # 解析工具调用
            tool_calls = self._parse_tool_calls(response)

            llm_response = LLMResponse(
                content=content,
                tool_calls=tool_calls,
                raw_response=response,
                latency_ms=latency_ms,
            )

            # 记录响应
            self._log_response(call_id, llm_response)

            return llm_response
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            error_response = LLMResponse(latency_ms=latency_ms)
            self._log_response(call_id, error_response, success=False, error=str(e))
            raise

    # =========================================================================
    # 辅助方法
    # =========================================================================

    def _build_messages(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str | None = None,
    ) -> list[BaseMessage]:
        """构建 LangChain 消息列表

        Args:
            messages: 原始消息列表
            system_prompt: 系统提示词

        Returns:
            LangChain 消息列表
        """
        lc_messages: list[BaseMessage] = []

        # 添加系统提示词
        if system_prompt:
            lc_messages.append(SystemMessage(content=system_prompt))

        # 转换消息
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                # 处理 assistant 消息中的 tool_calls
                tool_calls_data = msg.get("tool_calls")
                if tool_calls_data:
                    # 如果有 tool_calls，需要在 AIMessage 中包含
                    lc_messages.append(AIMessage(content=content, tool_calls=tool_calls_data))
                else:
                    lc_messages.append(AIMessage(content=content))
            elif role == "tool":
                lc_messages.append(ToolMessage(
                    content=content,
                    tool_call_id=msg.get("tool_call_id", ""),
                ))

        return lc_messages

    def _parse_tool_calls(self, response: Any) -> list[ToolCall]:
        """解析工具调用

        Args:
            response: LLM 响应对象

        Returns:
            ToolCall 列表
        """
        tool_calls: list[ToolCall] = []

        if not hasattr(response, "tool_calls"):
            return tool_calls

        for tc in response.tool_calls:
            # LangChain 的 tool_calls 格式: {"name": str, "args": dict, "id": str}
            tool_calls.append(ToolCall(
                id=tc.get("id", ""),
                name=tc.get("name", ""),
                args=tc.get("args", {}),
            ))

        return tool_calls

    @classmethod
    def from_env(cls, prefix: str = "AGENT_LLM") -> "LLMClient":
        """从环境变量创建客户端

        Args:
            prefix: 环境变量前缀

        Returns:
            LLMClient 实例
        """
        config = LLMConfig.from_env(prefix)
        return cls(config)


# ============================================================================
# 全局单例
# ============================================================================

_llm_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """获取全局 LLM 客户端单例

    注意：如果未通过 set_llm_client 设置，会使用默认配置创建客户端。
    建议在启动时通过 set_llm_client 设置正确的配置。
    """
    global _llm_client
    if _llm_client is None:
        # 使用默认配置创建
        _llm_client = LLMClient.from_env()
    return _llm_client


def set_llm_client(client: LLMClient) -> None:
    """设置全局 LLM 客户端（启动时初始化使用）

    Args:
        client: LLM 客户端实例
    """
    global _llm_client
    _llm_client = client


def reset_llm_client() -> None:
    """重置全局 LLM 客户端（用于测试）"""
    global _llm_client
    _llm_client = None
