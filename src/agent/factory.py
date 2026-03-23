# -*- coding: utf-8 -*-
"""Factory for the new deep agent runtime."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langchain_openai import ChatOpenAI

from agent.config import DeepAgentConfig
from agent.llm_logging import LLMCallLoggingCallback
from agent.tools.mcp_tools import get_mcp_tools
from domain_profile import get_domain_profile
from retrievers.tools.domain_retrieve_tool import create_domain_retrieve_tool

logger = logging.getLogger(__name__)


def _build_model(*, config: DeepAgentConfig, project_root: Path) -> ChatOpenAI:
    kwargs: dict[str, Any] = {
        "model": config.model,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "timeout": config.timeout_seconds,
        "callbacks": [LLMCallLoggingCallback(project_root=project_root)],
    }
    if config.api_key:
        kwargs["api_key"] = config.api_key
    if config.base_url:
        kwargs["base_url"] = config.base_url
    return ChatOpenAI(**kwargs)


def create_agent(
    *,
    project_root: Path,
    checkpointer: Any | None = None,
) -> Any:
    """Create the single Deep Agent for the application."""
    try:
        from deepagents import create_deep_agent
        from deepagents.backends import FilesystemBackend
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "deepagents is not installed. Add the dependency and reinstall requirements."
        ) from exc

    domain_profile = get_domain_profile(project_root=project_root)
    config = DeepAgentConfig.from_domain_profile(
        domain_profile=domain_profile,
        project_root=project_root,
    )
    tools = [create_domain_retrieve_tool(), *get_mcp_tools()]
    model = _build_model(config=config, project_root=project_root)
    backend = FilesystemBackend(root_dir=str(config.backend_root), virtual_mode=True)
    logger.info(
        "Creating deep agent: domain=%s, skills_root=%s, tool_count=%s",
        domain_profile.profile_id,
        config.skills_root,
        len(tools),
    )
    agent = create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=config.system_prompt,
        skills=[config.skills_root],
        backend=backend,
        checkpointer=checkpointer,
    )
    setattr(
        agent,
        "_dsp_runtime_config",
        {
            "model": config.model,
            "skills_root": config.skills_root,
            "tool_count": len(tools),
        },
    )
    return agent
