# -*- coding: utf-8 -*-
"""LLM 客户端模块

提供 LLM 调用能力：
- LLMClient: LLM 客户端 (参考 v1 WorkflowLLMClient)
- LLMConfig: LLM 配置
- 支持工具调用
- 支持流式响应

全局单例：
- get_llm_client: 获取全局 LLM 客户端单例
- set_llm_client: 设置全局 LLM 客户端
- reset_llm_client: 重置全局 LLM 客户端
"""
from agent.llm.client import (
    LLMClient,
    LLMResponse,
    ToolCall,
    get_llm_client,
    set_llm_client,
    reset_llm_client,
)
from agent.llm.config import LLMConfig

__all__ = [
    "LLMClient",
    "LLMConfig",
    "LLMResponse",
    "ToolCall",
    "get_llm_client",
    "set_llm_client",
    "reset_llm_client",
]
