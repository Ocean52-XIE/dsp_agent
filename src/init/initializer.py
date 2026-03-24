# -*- coding: utf-8 -*-
"""Initializer for the deep-agent runtime."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

from log import get_file_logger


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _domain_summary(domain_profile: Any, project_root: Path, *, latency_ms: int) -> dict[str, Any]:
    from domain_profile import resolve_domain_profile_path

    raw = getattr(domain_profile, "raw", {}) or {}
    deep_agents = raw.get("deep_agents", {}) if isinstance(raw, dict) else {}
    skills = list(deep_agents.get("primary_skills", []) or [])
    return {
        "domain": domain_profile.profile_id,
        "profile_id": domain_profile.profile_id,
        "display_name": getattr(domain_profile, "display_name", domain_profile.profile_id),
        "profile_path": str(resolve_domain_profile_path(project_root=project_root)),
        "skills": skills,
        "wiki_dir": str(domain_profile.resolve_wiki_dir(project_root)),
        "code_dirs": [str(path) for path in domain_profile.resolve_code_roots(project_root)],
        "module_names": [item.name for item in getattr(domain_profile, "modules", ())],
        "latency_ms": latency_ms,
    }


def _disabled_retriever_summary(kind: str, domain_profile: Any, project_root: Path) -> dict[str, Any]:
    if kind == "wiki":
        return {
            "active": False,
            "wiki_dir": str(domain_profile.resolve_wiki_dir(project_root)),
            "file_count": 0,
            "chunk_count": 0,
            "embedding_enabled": False,
            "embedding_model": None,
            "reranker_enabled": False,
            "reranker_model": None,
            "latency_ms": 0,
            "error_type": None,
            "reason": "disabled_by_initializer",
        }
    return {
        "active": False,
        "code_dirs": [str(path) for path in domain_profile.resolve_code_roots(project_root)],
        "indexed_file_count": 0,
        "parent_chunk_count": 0,
        "child_chunk_count": 0,
        "embedding_enabled": False,
        "embedding_model": None,
        "reranker_enabled": False,
        "reranker_model": None,
        "latency_ms": 0,
        "error_type": None,
        "reason": "disabled_by_initializer",
    }


def _init_retrievers(domain_profile: Any, project_root: Path) -> dict[str, dict[str, Any]]:
    from init.retriever_initializer import init_code_retriever, init_wiki_retriever

    runtime_logger = get_file_logger(project_root=project_root)
    runtime_logger.info(
        "init.retriever.wiki.begin",
        domain=domain_profile.profile_id,
        wiki_dir=str(domain_profile.resolve_wiki_dir(project_root)),
    )
    _, wiki_summary = init_wiki_retriever(domain_profile, project_root)
    runtime_logger.info(
        "init.retriever.wiki.completed",
        domain=domain_profile.profile_id,
        **wiki_summary,
    )

    runtime_logger.info(
        "init.retriever.code.begin",
        domain=domain_profile.profile_id,
        code_dirs=[str(path) for path in domain_profile.resolve_code_roots(project_root)],
    )
    _, code_summary = init_code_retriever(domain_profile, project_root)
    runtime_logger.info(
        "init.retriever.code.completed",
        domain=domain_profile.profile_id,
        **code_summary,
    )
    return {
        "wiki": wiki_summary,
        "code": code_summary,
    }


async def initialize_async(
    project_root: Path | None = None,
    enable_mcp: bool = True,
    enable_retrievers: bool = True,
) -> dict[str, Any]:
    """Initialize the runtime and return a structured startup summary."""
    resolved_root = (project_root or _default_project_root()).resolve()
    runtime_logger = get_file_logger(project_root=resolved_root)

    from init.domain_profile_initializer import init_domain_profile

    runtime_logger.info("init.domain_profile.begin")
    domain_started_at = perf_counter()
    domain_profile = init_domain_profile(project_root=resolved_root)
    if domain_profile is None:
        raise RuntimeError("领域配置加载失败，无法继续初始化")

    domain_summary = _domain_summary(
        domain_profile,
        resolved_root,
        latency_ms=int((perf_counter() - domain_started_at) * 1000),
    )
    runtime_logger.info("init.domain_profile.loaded", **domain_summary)

    if enable_mcp:
        from init.mcp_initializer import initialize_mcp_system_async

        mcp_summary = await initialize_mcp_system_async(domain_profile)
    else:
        mcp_summary = {
            "enabled": False,
            "servers": [],
            "tools": [],
            "server_tool_map": {},
            "latency_ms": 0,
            "reason": "disabled_by_initializer",
            "error_type": None,
        }
        runtime_logger.info("init.mcp.completed", domain=domain_profile.profile_id, **mcp_summary)

    retriever_summary = {
        "wiki": _disabled_retriever_summary("wiki", domain_profile, resolved_root),
        "code": _disabled_retriever_summary("code", domain_profile, resolved_root),
    }
    if enable_retrievers:
        retriever_summary = _init_retrievers(domain_profile, project_root=resolved_root)
    else:
        runtime_logger.info(
            "init.retriever.wiki.completed",
            domain=domain_profile.profile_id,
            **retriever_summary["wiki"],
        )
        runtime_logger.info(
            "init.retriever.code.completed",
            domain=domain_profile.profile_id,
            **retriever_summary["code"],
        )

    return {
        "domain": domain_summary,
        "mcp": mcp_summary,
        "retrievers": retriever_summary,
    }
