# -*- coding: utf-8 -*-
"""Retriever initializers."""
from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter
from typing import Any


def _summarize_wiki_retriever(
    retriever: Any | None,
    *,
    wiki_dir: Path,
    latency_ms: int,
    error_type: str = "",
    reason: str = "",
) -> dict[str, Any]:
    runtime_config = getattr(retriever, "runtime_config", None)
    reranker_profile = getattr(retriever, "_reranker_profile", None)
    return {
        "active": retriever is not None,
        "wiki_dir": str(wiki_dir),
        "file_count": len(getattr(retriever, "_chunk_ids_by_path", {}) or {}),
        "chunk_count": len(getattr(retriever, "_documents", []) or []),
        "embedding_enabled": bool(getattr(retriever, "_embedding_retriever", None)),
        "embedding_model": getattr(runtime_config, "embedding_model", None)
        if getattr(retriever, "_embedding_retriever", None) is not None
        else None,
        "reranker_enabled": bool(getattr(retriever, "_reranker", None)),
        "reranker_model": getattr(reranker_profile, "model", None)
        if getattr(retriever, "_reranker", None) is not None
        else None,
        "latency_ms": latency_ms,
        "error_type": error_type or None,
        "reason": reason or None,
    }


def _summarize_code_retriever(
    retriever: Any | None,
    *,
    code_dirs: list[Path],
    latency_ms: int,
    error_type: str = "",
    reason: str = "",
) -> dict[str, Any]:
    reranker_profile = getattr(retriever, "_reranker_profile", None)
    embedding_profile = getattr(retriever, "_embedding_profile", None)
    return {
        "active": retriever is not None,
        "code_dirs": [str(path) for path in code_dirs],
        "indexed_file_count": int(
            getattr(retriever, "_index_stats", {}).get("indexed_file_count", 0)
            if retriever is not None
            else 0
        ),
        "parent_chunk_count": len(getattr(retriever, "_parent_chunks", {}) or {}),
        "child_chunk_count": len(getattr(retriever, "_child_chunks", []) or []),
        "embedding_enabled": bool(getattr(retriever, "_embedding_retriever", None)),
        "embedding_model": getattr(embedding_profile, "model", None)
        if getattr(retriever, "_embedding_retriever", None) is not None
        else None,
        "reranker_enabled": bool(getattr(retriever, "_reranker", None)),
        "reranker_model": getattr(reranker_profile, "model", None)
        if getattr(retriever, "_reranker", None) is not None
        else None,
        "latency_ms": latency_ms,
        "error_type": error_type or None,
        "reason": reason or None,
    }


def init_wiki_retriever(
    domain_profile: Any,
    project_root: Path,
) -> tuple[Any | None, dict[str, Any]]:
    """Initialize the wiki retriever and return a startup summary."""
    from retrievers.wiki.retriever import (
        MarkdownWikiRetriever,
        set_wiki_retriever,
    )

    wiki_dir = domain_profile.resolve_wiki_dir(project_root)
    started_at = perf_counter()
    try:
        retriever = MarkdownWikiRetriever(
            wiki_dir=wiki_dir,
            project_root=project_root,
            default_top_k=4,
            module_doc_hints=domain_profile.module_doc_hints(),
            embedding_profile=domain_profile.retrieval.embedding,
            hybrid_weights_profile=domain_profile.retrieval.hybrid_weights,
            reranker_profile=domain_profile.retrieval.reranker,
        )
        set_wiki_retriever(retriever)
        latency_ms = int((perf_counter() - started_at) * 1000)
        return retriever, _summarize_wiki_retriever(
            retriever,
            wiki_dir=wiki_dir,
            latency_ms=latency_ms,
        )
    except Exception as exc:
        latency_ms = int((perf_counter() - started_at) * 1000)
        return None, _summarize_wiki_retriever(
            None,
            wiki_dir=wiki_dir,
            latency_ms=latency_ms,
            error_type=type(exc).__name__,
            reason="init_failed",
        )


def init_code_retriever(
    domain_profile: Any,
    project_root: Path,
) -> tuple[Any | None, dict[str, Any]]:
    """Initialize the code retriever and return a startup summary."""
    from retrievers.code.retriever import (
        LocalCodeRetriever,
        parse_code_dirs_from_env,
        set_code_retriever,
    )

    env_code_dirs = os.getenv("AGENT_CODE_RETRIEVER_DIRS", "").strip()
    code_dirs = (
        parse_code_dirs_from_env(project_root=project_root)
        if env_code_dirs
        else domain_profile.resolve_code_roots(project_root)
    ) or []

    started_at = perf_counter()
    try:
        retriever = LocalCodeRetriever(
            project_root=project_root,
            code_dirs=code_dirs,
            default_top_k=4,
            embedding_profile=domain_profile.retrieval.embedding,
            reranker_profile=domain_profile.retrieval.reranker,
        )
        set_code_retriever(retriever)
        latency_ms = int((perf_counter() - started_at) * 1000)
        return retriever, _summarize_code_retriever(
            retriever,
            code_dirs=code_dirs,
            latency_ms=latency_ms,
        )
    except Exception as exc:
        latency_ms = int((perf_counter() - started_at) * 1000)
        return None, _summarize_code_retriever(
            None,
            code_dirs=code_dirs,
            latency_ms=latency_ms,
            error_type=type(exc).__name__,
            reason="init_failed",
        )
