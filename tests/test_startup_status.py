# -*- coding: utf-8 -*-
from __future__ import annotations

import pytest

from init.retriever_initializer import _summarize_code_retriever
from init.database_initializer import _create_memory_checkpointer, get_database_status
from observability.async_postgres_store import PostgresObservabilityConfig, PostgresObservabilityStore
from session.async_postgres_session_store import PostgresSessionConfig, PostgresSessionStore


def test_get_database_status_reports_memory_fallback_metadata() -> None:
    checkpointer = _create_memory_checkpointer("postgres_init_failed")

    assert get_database_status(checkpointer) == {
        "backend": "memory",
        "status": "active",
        "reason": "postgres_init_failed",
        "fallback": True,
        "fallback_from": "postgres",
        "fallback_to": "memory",
        "selected_backend": "memory",
    }


@pytest.mark.asyncio
async def test_observability_store_status_includes_reason_when_disabled() -> None:
    store = await PostgresObservabilityStore.create(
        PostgresObservabilityConfig(
            enabled=False,
            dsn="",
            schema="public",
            connect_timeout_seconds=5,
            alert_window_minutes=30,
            alert_min_samples=20,
            alert_suppress_minutes=30,
            alert_empty_response_rate_max=0.05,
            alert_fallback_rate_max=0.25,
            alert_insufficient_rate_max=0.20,
            alert_p95_latency_ms_max=3000.0,
            alert_exact_like_pass_rate_min=0.70,
            pool_min_size=1,
            pool_max_size=10,
        )
    )

    assert store.status()["reason"] == "config_disabled"


@pytest.mark.asyncio
async def test_session_store_status_includes_reason_when_disabled() -> None:
    store = await PostgresSessionStore.create(
        PostgresSessionConfig(
            enabled=False,
            dsn="",
            schema="public",
            connect_timeout_seconds=5,
            pool_min_size=1,
            pool_max_size=10,
        )
    )

    assert store.status()["reason"] == "config_disabled"


def test_code_retriever_summary_uses_indexed_file_count_from_index_stats() -> None:
    retriever = type(
        "RetrieverStub",
        (),
        {
            "_index_stats": {"indexed_file_count": 5},
            "_parent_chunks": {"p1": object(), "p2": object(), "p3": object()},
            "_child_chunks": [object(), object(), object(), object()],
            "_embedding_retriever": object(),
            "_embedding_profile": type("EmbeddingProfile", (), {"model": "embed"})(),
            "_reranker": object(),
            "_reranker_profile": type("RerankerProfile", (), {"model": "rerank"})(),
        },
    )()

    summary = _summarize_code_retriever(
        retriever,
        code_dirs=[],
        latency_ms=12,
    )

    assert summary["indexed_file_count"] == 5
    assert summary["parent_chunk_count"] == 3
    assert summary["child_chunk_count"] == 4
