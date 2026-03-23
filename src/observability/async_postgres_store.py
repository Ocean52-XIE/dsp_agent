# -*- coding: utf-8 -*-
"""Async PostgreSQL observability store."""
from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from psycopg_pool import AsyncConnectionPool

from common.func_utils import normalize_source_type, to_bool, to_float, to_int
from init.database_initializer import ensure_database_exists
from log import get_file_logger


def _sanitize_identifier(value: str, default: str) -> str:
    normalized = (value or "").strip()
    if not normalized:
        return default
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", normalized):
        return normalized
    return default


@dataclass
class PostgresObservabilityConfig:
    """Configuration for the async observability store."""

    enabled: bool
    dsn: str
    schema: str
    connect_timeout_seconds: int
    alert_window_minutes: int
    alert_min_samples: int
    alert_suppress_minutes: int
    alert_empty_response_rate_max: float
    alert_fallback_rate_max: float
    alert_insufficient_rate_max: float
    alert_p95_latency_ms_max: float
    alert_exact_like_pass_rate_min: float
    pool_min_size: int
    pool_max_size: int

    @classmethod
    def from_env(cls) -> "PostgresObservabilityConfig":
        dsn = os.getenv("AGENT_OBS_PG_DSN", "").strip()
        enabled_default = bool(dsn)
        return cls(
            enabled=to_bool(os.getenv("AGENT_OBS_PG_ENABLED"), enabled_default),
            dsn=dsn,
            schema=_sanitize_identifier(os.getenv("AGENT_OBS_PG_SCHEMA", "public"), "public"),
            connect_timeout_seconds=max(1, to_int(os.getenv("AGENT_OBS_PG_CONNECT_TIMEOUT_SECONDS"), 5)),
            alert_window_minutes=max(5, to_int(os.getenv("AGENT_OBS_ALERT_WINDOW_MINUTES"), 30)),
            alert_min_samples=max(1, to_int(os.getenv("AGENT_OBS_ALERT_MIN_SAMPLES"), 20)),
            alert_suppress_minutes=max(1, to_int(os.getenv("AGENT_OBS_ALERT_SUPPRESS_MINUTES"), 30)),
            alert_empty_response_rate_max=max(
                0.0,
                min(1.0, to_float(os.getenv("AGENT_OBS_ALERT_EMPTY_RESPONSE_RATE_MAX"), 0.05)),
            ),
            alert_fallback_rate_max=max(
                0.0,
                min(1.0, to_float(os.getenv("AGENT_OBS_ALERT_FALLBACK_RATE_MAX"), 0.25)),
            ),
            alert_insufficient_rate_max=max(
                0.0,
                min(1.0, to_float(os.getenv("AGENT_OBS_ALERT_INSUFFICIENT_RATE_MAX"), 0.20)),
            ),
            alert_p95_latency_ms_max=max(
                1.0,
                to_float(os.getenv("AGENT_OBS_ALERT_P95_LATENCY_MS_MAX"), 3000.0),
            ),
            alert_exact_like_pass_rate_min=max(
                0.0,
                min(1.0, to_float(os.getenv("AGENT_OBS_ALERT_EXACT_LIKE_PASS_RATE_MIN"), 0.70)),
            ),
            pool_min_size=max(1, to_int(os.getenv("AGENT_OBS_PG_POOL_MIN_SIZE"), 1)),
            pool_max_size=max(1, to_int(os.getenv("AGENT_OBS_PG_POOL_MAX_SIZE"), 10)),
        )


class PostgresObservabilityStore:
    """Async PostgreSQL-backed observability store."""

    def __init__(
        self,
        config: PostgresObservabilityConfig,
        *,
        pool: AsyncConnectionPool | None = None,
        init_error: str | None = None,
    ) -> None:
        self.config = config
        self._pool = pool
        self._init_error = init_error
        self._logger = get_file_logger(project_root=Path(__file__).resolve().parents[3])

    @classmethod
    async def create_from_env(cls) -> "PostgresObservabilityStore":
        return await cls.create(PostgresObservabilityConfig.from_env())

    @classmethod
    async def create(cls, config: PostgresObservabilityConfig) -> "PostgresObservabilityStore":
        store = cls(config)
        if not config.enabled:
            store._logger.info("observability.store.disabled", reason="config_disabled")
            return store

        try:
            import psycopg  # type: ignore
        except Exception as exc:  # pragma: no cover
            config.enabled = False
            store._init_error = f"import_psycopg_failed:{exc}"
            store._logger.warning(
                "observability.store.init_failed",
                reason="import_psycopg_failed",
                error_type=type(exc).__name__,
            )
            return store

        try:
            await asyncio.to_thread(
                ensure_database_exists,
                psycopg_module=psycopg,
                dsn=config.dsn,
                connect_timeout_seconds=config.connect_timeout_seconds,
            )
            pool = AsyncConnectionPool(
                config.dsn,
                kwargs={
                    "autocommit": True,
                    "connect_timeout": config.connect_timeout_seconds,
                },
                min_size=config.pool_min_size,
                max_size=max(config.pool_min_size, config.pool_max_size),
                open=False,
            )
            await pool.open()
            await pool.wait()
            store._pool = pool
            await store.ensure_schema()
            store._logger.info(
                "observability.store.ready",
                schema=config.schema,
                connect_timeout_seconds=config.connect_timeout_seconds,
                pool_min_size=config.pool_min_size,
                pool_max_size=config.pool_max_size,
            )
            return store
        except Exception as exc:  # pragma: no cover
            config.enabled = False
            store._init_error = f"bootstrap_or_schema_failed:{exc}"
            store._logger.warning(
                "observability.store.init_failed",
                reason="bootstrap_or_schema_failed",
                error_type=type(exc).__name__,
            )
            if store._pool is not None:
                await store._pool.close()
                store._pool = None
            return store

    @property
    def is_active(self) -> bool:
        return bool(self.config.enabled and self._pool is not None)

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.config.enabled,
            "active": self.is_active,
            "schema": self.config.schema,
            "dsn_configured": bool(self.config.dsn),
            "init_error": self._init_error,
        }

    async def aclose(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    def _table(self, name: str) -> str:
        safe_name = _sanitize_identifier(name, name)
        return f"{self.config.schema}.{safe_name}"

    async def ensure_schema(self) -> None:
        if not self.is_active or self._pool is None:
            return
        request_table = self._table("qa_request_log")
        evidence_table = self._table("qa_evidence_log")
        feedback_table = self._table("qa_feedback_log")
        alert_table = self._table("qa_alert_event")
        snapshot_table = self._table("qa_metric_snapshot")
        ddl = f"""
        CREATE SCHEMA IF NOT EXISTS {self.config.schema};

        CREATE TABLE IF NOT EXISTS {request_table} (
            id BIGSERIAL PRIMARY KEY,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            turn_type TEXT NOT NULL,
            session_id TEXT NOT NULL,
            trace_id TEXT NOT NULL,
            message_id TEXT NOT NULL,
            user_query TEXT NOT NULL,
            assistant_kind TEXT NOT NULL,
            assistant_intent TEXT NOT NULL,
            assistant_status TEXT NOT NULL,
            task_stage TEXT NOT NULL,
            execution_path TEXT NOT NULL,
            transition_type TEXT NOT NULL,
            route TEXT NOT NULL,
            next_action TEXT NOT NULL,
            latency_ms INTEGER NOT NULL DEFAULT 0,
            domain_relevance DOUBLE PRECISION NOT NULL DEFAULT 0.0,
            is_qa_turn BOOLEAN NOT NULL DEFAULT FALSE,
            is_empty_response BOOLEAN NOT NULL DEFAULT FALSE,
            is_fallback BOOLEAN NOT NULL DEFAULT FALSE,
            contains_insufficient BOOLEAN NOT NULL DEFAULT FALSE,
            exact_like_pass BOOLEAN NOT NULL DEFAULT FALSE,
            llm_model TEXT NULL,
            llm_fallback_reason TEXT NULL,
            answer_length INTEGER NOT NULL DEFAULT 0,
            citation_count INTEGER NOT NULL DEFAULT 0,
            wiki_citation_count INTEGER NOT NULL DEFAULT 0,
            code_citation_count INTEGER NOT NULL DEFAULT 0,
            case_citation_count INTEGER NOT NULL DEFAULT 0,
            payload JSONB NOT NULL DEFAULT '{{}}'::jsonb
        );

        CREATE INDEX IF NOT EXISTS idx_qa_request_log_created_at ON {request_table}(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_qa_request_log_trace_id ON {request_table}(trace_id);
        CREATE INDEX IF NOT EXISTS idx_qa_request_log_session_id ON {request_table}(session_id);

        CREATE TABLE IF NOT EXISTS {evidence_table} (
            id BIGSERIAL PRIMARY KEY,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            session_id TEXT NOT NULL,
            trace_id TEXT NOT NULL,
            message_id TEXT NOT NULL,
            rank_no INTEGER NOT NULL,
            source_type TEXT NOT NULL,
            path TEXT NULL,
            title TEXT NULL,
            section TEXT NULL,
            score DOUBLE PRECISION NULL,
            symbol_name TEXT NULL,
            start_line INTEGER NULL,
            end_line INTEGER NULL,
            excerpt TEXT NULL,
            payload JSONB NOT NULL DEFAULT '{{}}'::jsonb
        );

        CREATE TABLE IF NOT EXISTS {feedback_table} (
            id BIGSERIAL PRIMARY KEY,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            session_id TEXT NOT NULL,
            trace_id TEXT NOT NULL,
            message_id TEXT NOT NULL,
            helpful BOOLEAN NOT NULL,
            reason_tag TEXT NULL,
            rating SMALLINT NULL,
            comment TEXT NULL,
            payload JSONB NOT NULL DEFAULT '{{}}'::jsonb
        );

        CREATE TABLE IF NOT EXISTS {alert_table} (
            id BIGSERIAL PRIMARY KEY,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            alert_type TEXT NOT NULL,
            severity TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            metric_value DOUBLE PRECISION NOT NULL,
            threshold DOUBLE PRECISION NOT NULL,
            window_minutes INTEGER NOT NULL,
            sample_size INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'open',
            detail JSONB NOT NULL DEFAULT '{{}}'::jsonb
        );

        CREATE TABLE IF NOT EXISTS {snapshot_table} (
            id BIGSERIAL PRIMARY KEY,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            window_minutes INTEGER NOT NULL,
            sample_size INTEGER NOT NULL,
            empty_response_rate DOUBLE PRECISION NOT NULL DEFAULT 0.0,
            fallback_rate DOUBLE PRECISION NOT NULL DEFAULT 0.0,
            insufficient_rate DOUBLE PRECISION NOT NULL DEFAULT 0.0,
            p95_latency_ms DOUBLE PRECISION NOT NULL DEFAULT 0.0,
            exact_like_pass_rate DOUBLE PRECISION NOT NULL DEFAULT 0.0,
            payload JSONB NOT NULL DEFAULT '{{}}'::jsonb
        );
        """
        started_at = perf_counter()
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(ddl)
        self._logger.info(
            "observability.store.ensure_schema.completed",
            schema=self.config.schema,
            latency_ms=int((perf_counter() - started_at) * 1000),
        )

    async def record_turn(
        self,
        *,
        turn_type: str,
        session_id: str,
        trace_id: str,
        message_id: str,
        user_query: str,
        assistant_message: dict[str, Any],
    ) -> None:
        if not self.is_active or self._pool is None:
            self._logger.debug("observability.store.record_turn.skipped", reason="inactive_store")
            return
        started_at = perf_counter()
        analysis = dict(assistant_message.get("analysis") or {})
        debug = dict(assistant_message.get("debug") or {})
        citations = list(assistant_message.get("citations") or [])
        content = str(assistant_message.get("content", "") or "")
        status = str(assistant_message.get("status", "unknown") or "unknown")
        intent = str(assistant_message.get("intent", "unknown") or "unknown")
        kind = str(assistant_message.get("kind", "unknown") or "unknown")
        generation_mode = str(analysis.get("generation_mode", "") or "")
        llm_fallback_reason = str(analysis.get("llm_fallback_reason", "") or "")
        is_qa_turn = intent == "knowledge_qa"
        is_empty_response = llm_fallback_reason.startswith("empty_answer")
        is_fallback = bool(generation_mode and generation_mode != "llm")
        contains_insufficient = ("当前证据不足" in content) or ("证据不足" in content)
        exact_like_pass = bool(
            is_qa_turn
            and status == "completed"
            and (not is_empty_response)
            and (not contains_insufficient)
            and len(content.strip()) >= 40
            and len(citations) > 0
        )
        source_counter = {"wiki": 0, "code": 0, "case": 0}
        for item in citations:
            source = normalize_source_type(item.get("source_type"))
            if source in source_counter:
                source_counter[source] += 1
        request_sql = f"""
        INSERT INTO {self._table("qa_request_log")} (
            turn_type, session_id, trace_id, message_id, user_query,
            assistant_kind, assistant_intent, assistant_status, task_stage,
            execution_path, transition_type, route, next_action,
            latency_ms, domain_relevance, is_qa_turn, is_empty_response,
            is_fallback, contains_insufficient, exact_like_pass,
            llm_model, llm_fallback_reason, answer_length, citation_count,
            wiki_citation_count, code_citation_count, case_citation_count, payload
        ) VALUES (
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s, %s::jsonb
        )
        """
        request_args = (
            turn_type,
            session_id,
            trace_id,
            message_id,
            user_query,
            kind,
            intent,
            status,
            str(debug.get("task_stage", "")),
            str(debug.get("execution_path", "")),
            str(debug.get("transition_type", "")),
            str(debug.get("route", "")),
            str(debug.get("next_action", "")),
            int(debug.get("latency_ms", 0) or 0),
            float(debug.get("domain_relevance", 0.0) or 0.0),
            is_qa_turn,
            is_empty_response,
            is_fallback,
            contains_insufficient,
            exact_like_pass,
            str(analysis.get("llm_model", "")),
            llm_fallback_reason,
            len(content.strip()),
            len(citations),
            int(source_counter["wiki"]),
            int(source_counter["code"]),
            int(source_counter["case"]),
            json.dumps({"analysis": analysis, "debug": debug}, ensure_ascii=False),
        )
        evidence_sql = f"""
        INSERT INTO {self._table("qa_evidence_log")} (
            session_id, trace_id, message_id, rank_no, source_type, path, title,
            section, score, symbol_name, start_line, end_line, excerpt, payload
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s::jsonb
        )
        """
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(request_sql, request_args)
                for rank_no, item in enumerate(citations, start=1):
                    await cur.execute(
                        evidence_sql,
                        (
                            session_id,
                            trace_id,
                            message_id,
                            rank_no,
                            normalize_source_type(item.get("source_type")),
                            str(item.get("path", "") or ""),
                            str(item.get("title", "") or ""),
                            str(item.get("section", "") or ""),
                            float(item.get("score", 0.0) or 0.0),
                            str(item.get("symbol_name", "") or ""),
                            item.get("start_line"),
                            item.get("end_line"),
                            str(item.get("excerpt", "") or "")[:2000],
                            json.dumps(item, ensure_ascii=False),
                        ),
                    )
        self._logger.debug(
            "observability.store.record_turn.completed",
            trace_id=trace_id,
            session_id=session_id,
            message_id=message_id,
            turn_type=turn_type,
            citation_count=len(citations),
            latency_ms=int((perf_counter() - started_at) * 1000),
        )

    async def record_feedback(
        self,
        *,
        session_id: str,
        trace_id: str,
        message_id: str,
        helpful: bool,
        reason_tag: str,
        rating: int | None,
        comment: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        if not self.is_active or self._pool is None:
            self._logger.debug("observability.store.record_feedback.skipped", reason="inactive_store")
            return
        sql = f"""
        INSERT INTO {self._table("qa_feedback_log")} (
            session_id, trace_id, message_id, helpful, reason_tag, rating, comment, payload
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
        """
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    sql,
                    (
                        session_id,
                        trace_id,
                        message_id,
                        helpful,
                        reason_tag,
                        rating,
                        comment,
                        json.dumps(payload or {}, ensure_ascii=False),
                    ),
                )

    async def get_summary(self, *, window_minutes: int) -> dict[str, Any]:
        if not self.is_active or self._pool is None:
            return self._empty_summary(window_minutes=window_minutes)
        window_minutes = max(1, int(window_minutes))
        sql = f"""
        SELECT
            COUNT(*) FILTER (WHERE is_qa_turn) AS qa_total,
            AVG(CASE WHEN is_qa_turn AND is_empty_response THEN 1.0 ELSE 0.0 END) AS empty_response_rate,
            AVG(CASE WHEN is_qa_turn AND is_fallback THEN 1.0 ELSE 0.0 END) AS fallback_rate,
            AVG(CASE WHEN is_qa_turn AND contains_insufficient THEN 1.0 ELSE 0.0 END) AS insufficient_rate,
            COALESCE(
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms)
                FILTER (WHERE is_qa_turn AND latency_ms IS NOT NULL),
                0.0
            ) AS p95_latency_ms,
            AVG(CASE WHEN is_qa_turn AND exact_like_pass THEN 1.0 ELSE 0.0 END) AS exact_like_pass_rate
        FROM {self._table("qa_request_log")}
        WHERE created_at >= NOW() - (%s || ' minutes')::INTERVAL
        """
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql, (window_minutes,))
                row = await cur.fetchone() or (0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return {
            "window_minutes": window_minutes,
            "sample_size": int(row[0] or 0),
            "empty_response_rate": float(row[1] or 0.0),
            "fallback_rate": float(row[2] or 0.0),
            "insufficient_rate": float(row[3] or 0.0),
            "p95_latency_ms": float(row[4] or 0.0),
            "exact_like_pass_rate": float(row[5] or 0.0),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "top_failed_queries": await self._get_top_failed_queries(window_minutes=window_minutes, limit=10),
        }

    async def list_alerts(self, *, limit: int = 50) -> list[dict[str, Any]]:
        if not self.is_active or self._pool is None:
            return []
        sql = f"""
        SELECT id, created_at, alert_type, severity, metric_name, metric_value, threshold,
               window_minutes, sample_size, status, detail
        FROM {self._table("qa_alert_event")}
        ORDER BY created_at DESC
        LIMIT %s
        """
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql, (max(1, int(limit)),))
                rows = await cur.fetchall() or []
        return [
            {
                "id": int(row[0]),
                "created_at": row[1].isoformat(timespec="seconds") if row[1] is not None else "",
                "alert_type": str(row[2] or ""),
                "severity": str(row[3] or ""),
                "metric_name": str(row[4] or ""),
                "metric_value": float(row[5] or 0.0),
                "threshold": float(row[6] or 0.0),
                "window_minutes": int(row[7] or 0),
                "sample_size": int(row[8] or 0),
                "status": str(row[9] or ""),
                "detail": row[10] or {},
            }
            for row in rows
        ]

    def _empty_summary(self, *, window_minutes: int) -> dict[str, Any]:
        return {
            "window_minutes": int(window_minutes),
            "sample_size": 0,
            "empty_response_rate": 0.0,
            "fallback_rate": 0.0,
            "insufficient_rate": 0.0,
            "p95_latency_ms": 0.0,
            "exact_like_pass_rate": 0.0,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "top_failed_queries": [],
        }

    async def _get_top_failed_queries(self, *, window_minutes: int, limit: int) -> list[dict[str, Any]]:
        if not self.is_active or self._pool is None:
            return []
        sql = f"""
        SELECT
            user_query,
            COUNT(*) AS total_count,
            SUM(CASE WHEN exact_like_pass THEN 1 ELSE 0 END) AS pass_count
        FROM {self._table("qa_request_log")}
        WHERE created_at >= NOW() - (%s || ' minutes')::INTERVAL
          AND is_qa_turn = TRUE
        GROUP BY user_query
        ORDER BY (COUNT(*) - SUM(CASE WHEN exact_like_pass THEN 1 ELSE 0 END)) DESC, COUNT(*) DESC
        LIMIT %s
        """
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql, (window_minutes, limit))
                rows = await cur.fetchall() or []
        items: list[dict[str, Any]] = []
        for row in rows:
            total_count = int(row[1] or 0)
            pass_count = int(row[2] or 0)
            fail_count = max(0, total_count - pass_count)
            if fail_count <= 0:
                continue
            items.append(
                {
                    "query": str(row[0] or ""),
                    "total_count": total_count,
                    "pass_count": pass_count,
                    "fail_count": fail_count,
                }
            )
        return items
