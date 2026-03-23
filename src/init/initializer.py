# -*- coding: utf-8 -*-
"""Initializer for the new deep-agent runtime."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _init_retrievers(domain_profile: Any, project_root: Path) -> None:
    """初始化检索器，检索器本身仍保持同步预热。"""
    try:
        from init.retriever_initializer import init_code_retriever, init_wiki_retriever

        init_wiki_retriever(domain_profile, project_root)
        init_code_retriever(domain_profile, project_root)
    except Exception as exc:
        logger.warning("retriever.initialize.failed: %s", type(exc).__name__)


def _log_initialization_summary() -> None:
    """输出初始化摘要日志。"""
    from agent.mcp import get_mcp_client
    from domain_profile import get_domain_profile
    from retrievers.code.retriever import get_code_retriever
    from retrievers.wiki.retriever import get_wiki_retriever

    domain_profile = get_domain_profile()
    mcp_client = get_mcp_client()
    wiki_retriever = get_wiki_retriever()
    code_retriever = get_code_retriever()

    logger.info("=" * 60)
    logger.info("[Init] initialization summary")
    if domain_profile:
        logger.info("domain=%s", domain_profile.profile_id)
        logger.info(
            "deep_agent_skills=%s",
            domain_profile.raw.get("deep_agents", {}).get("primary_skills", []),
        )
    logger.info("mcp_enabled=%s", bool(mcp_client))
    logger.info("wiki_retriever_enabled=%s", bool(wiki_retriever))
    logger.info("code_retriever_enabled=%s", bool(code_retriever))
    logger.info("=" * 60)


async def initialize_async(
    project_root: Path | None = None,
    enable_mcp: bool = True,
    enable_retrievers: bool = True,
) -> None:
    """异步初始化 deep-agent 运行时。"""
    logger.info("initialize_async.begin")

    from init.domain_profile_initializer import init_domain_profile

    domain_profile = init_domain_profile(project_root=project_root)
    if domain_profile is None:
        raise RuntimeError("领域配置加载失败，无法继续初始化")

    if enable_mcp:
        from init.mcp_initializer import initialize_mcp_system_async

        await initialize_mcp_system_async(domain_profile)

    if enable_retrievers:
        _init_retrievers(domain_profile, project_root=project_root)

    _log_initialization_summary()

