# -*- coding: utf-8 -*-
"""LLM 客户端初始化器

负责初始化 LLM 客户端：
- 读取 LLM 配置（API Key、Model 等）
- 创建 LLM 客户端实例
- 验证 LLM 连接（可选）
- 设置全局单例

设计原则：
1. 在程序启动时完成 LLM 客户端初始化
2. 支持配置热更新（通过环境变量）
3. 初始化后设置全局单例，其他模块通过单例访问
"""
from __future__ import annotations

import logging
import os
from typing import Any
from agent.llm import LLMClient, set_llm_client


logger = logging.getLogger(__name__)


# ============================================================================
# LLM 客户端初始化
# ============================================================================

def init_llm_client(
    prefix: str = "WORKFLOW_QA_LLM",
) -> None:
    """初始化 LLM 客户端并设置全局单例

    从环境变量读取配置，创建 LLM 客户端实例，
    并设置全局单例供其他模块访问。

    Args:
        prefix: 环境变量前缀（默认 "WORKFLOW_QA_LLM"）
    """

    client = LLMClient.from_env(prefix=prefix)

    # 设置全局单例
    set_llm_client(client)


def validate_llm_connection(client: Any) -> bool:
    """验证 LLM 连接（可选）

    发送一个简单的测试请求，验证 LLM 是否可用。

    Args:
        client: LLM 客户端实例

    Returns:
        是否可用
    """
    if client is None:
        return False

    # 检查客户端配置
    config = getattr(client, "config", None)
    if config is None:
        return False

    if not getattr(config, "enabled", False):
        logger.info("[LLMInit] LLM 未启用，跳过连接验证")
        return False

    # 检查 API Key
    api_key = getattr(config, "api_key", None)
    if not api_key:
        logger.warning("[LLMInit] LLM API Key 未配置")
        return False

    # 尝试检查可用性
    is_available = getattr(client, "is_available", False)
    if callable(is_available):
        is_available = is_available()

    if is_available:
        logger.info("[LLMInit] LLM 连接验证通过")
    else:
        logger.warning("[LLMInit] LLM 连接验证失败")

    return bool(is_available)


# ============================================================================
# 状态查询
# ============================================================================

def get_llm_status(client: Any) -> dict[str, Any]:
    """获取 LLM 客户端状态

    Args:
        client: LLM 客户端实例

    Returns:
        状态字典
    """
    if client is None:
        return {
            "status": "not_initialized",
            "enabled": False,
            "available": False,
        }

    config = getattr(client, "config", None)

    return {
        "status": "initialized",
        "enabled": getattr(config, "enabled", False) if config else False,
        "available": getattr(client, "is_available", False),
        "model": getattr(config, "model", None) if config else None,
        "provider": getattr(config, "provider", None) if config else None,
    }
