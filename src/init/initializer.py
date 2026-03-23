# -*- coding: utf-8 -*-
"""Initializer for the new deep-agent runtime."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _init_retrievers(domain_profile: Any, project_root: Path) -> None:
    """初始化检索器（调用子初始化器，单例在子初始化器中设置）"""

    try:
        from init.retriever_initializer import (
            init_wiki_retriever,
            init_code_retriever,
        )

        # 1. Wiki 检索器
        init_wiki_retriever(domain_profile, project_root)

        # 2. Code 检索器
        init_code_retriever(domain_profile, project_root)

    except Exception as e:
        logger.warning(f"  └─ 检索器初始化失败: {e}")


def initialize(
    project_root: Path | None = None,
    enable_mcp: bool = True,
    enable_retrievers: bool = True,
) ->None:
    """Initialize only the components still used by the deep-agent runtime."""
    logger.info("initialize.begin")

    from init.domain_profile_initializer import init_domain_profile

    domain_profile = init_domain_profile(project_root=project_root)
    if domain_profile is None:
        raise RuntimeError("私域配置加载失败，无法继续初始化")

    if enable_mcp:
        from init.mcp_initializer import initialize_mcp_system
        initialize_mcp_system(domain_profile)

    if enable_retrievers:
        _init_retrievers(domain_profile, project_root=project_root)

    _log_initialization_summary()


def _log_initialization_summary() -> None:
    """输出初始化摘要日志（从单例获取信息）"""
    from domain_profile import get_domain_profile
    from agent.mcp import get_mcp_client
    from retrievers.wiki.retriever import get_wiki_retriever
    from retrievers.code.retriever import get_code_retriever

    domain_profile = get_domain_profile()
    mcp_client = get_mcp_client()
    wiki_retriever = get_wiki_retriever()
    code_retriever = get_code_retriever()

    logger.info("")
    logger.info("=" * 60)
    logger.info("[Init] 初始化完成摘要")
    logger.info("=" * 60)
    if domain_profile:
        logger.info(f"  ├─ domain: {domain_profile.profile_id}")
    logger.info(f"  ├─ deep_agent_skills: {domain_profile.raw.get('deep_agents', {}).get('primary_skills', []) if domain_profile else []}")
    logger.info(f"  ├─ mcp: {'启用' if mcp_client else '未启用'}")
    logger.info(f"  ├─ wiki_retriever: {'启用' if wiki_retriever else '未启用'}")
    logger.info(f"  └─ code_retriever: {'启用' if code_retriever else '未启用'}")
    logger.info("=" * 60)


# ============================================================================
# 异步初始化函数（推荐用于 FastAPI 等异步框架）
# ============================================================================

async def initialize_async(
    project_root: Path | None = None,
    enable_mcp: bool = True,
    enable_retrievers: bool = True,
) -> None:
    """Async initialization for the deep-agent runtime."""
    logger.info("initialize_async.begin")

    from init.domain_profile_initializer import init_domain_profile

    domain_profile = init_domain_profile(project_root=project_root)
    if domain_profile is None:
        raise RuntimeError("私域配置加载失败，无法继续初始化")

    if enable_mcp:
        from init.mcp_initializer import initialize_mcp_system_async
        await initialize_mcp_system_async(domain_profile)

    if enable_retrievers:
        _init_retrievers(domain_profile, project_root=project_root)

    _log_initialization_summary()
