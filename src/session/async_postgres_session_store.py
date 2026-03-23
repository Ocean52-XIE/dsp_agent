# -*- coding: utf-8 -*-
"""Async PostgreSQL session store."""
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

from common.func_utils import to_bool, to_int
from init.database_initializer import ensure_database_exists
from log import get_file_logger


def _sanitize_identifier(value: str, default: str) -> str:
    normalized = (value or "").strip()
    if not normalized:
        return default
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", normalized):
        return normalized
    return default


def _to_iso(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds")
    text = str(value or "").strip()
    if not text:
        return datetime.now().isoformat(timespec="seconds")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed.isoformat(timespec="seconds")
    except ValueError:
        return text


@dataclass
class PostgresSessionConfig:
    """Configuration for the async session store."""

    enabled: bool
    dsn: str
    schema: str
    connect_timeout_seconds: int
    pool_min_size: int
    pool_max_size: int

    @classmethod
    def from_env(cls) -> "PostgresSessionConfig":
        explicit_dsn = os.getenv("WORKFLOW_SESSION_PG_DSN", "").strip()
        fallback_dsn = os.getenv("WORKFLOW_OBS_PG_DSN", "").strip()
        dsn = explicit_dsn or fallback_dsn
        enabled_default = bool(dsn)
        schema_env = (
            os.getenv("WORKFLOW_SESSION_PG_SCHEMA")
            or os.getenv("WORKFLOW_OBS_PG_SCHEMA")
            or "public"
        )
        return cls(
            enabled=to_bool(os.getenv("WORKFLOW_SESSION_PG_ENABLED"), enabled_default),
            dsn=dsn,
            schema=_sanitize_identifier(schema_env, "public"),
            connect_timeout_seconds=max(
                1,
                to_int(
                    os.getenv("WORKFLOW_SESSION_PG_CONNECT_TIMEOUT_SECONDS")
                    or os.getenv("WORKFLOW_OBS_PG_CONNECT_TIMEOUT_SECONDS"),
                    5,
                ),
            ),
            pool_min_size=max(1, to_int(os.getenv("WORKFLOW_SESSION_PG_POOL_MIN_SIZE"), 1)),
            pool_max_size=max(1, to_int(os.getenv("WORKFLOW_SESSION_PG_POOL_MAX_SIZE"), 10)),
        )


class PostgresSessionStore:
    """Async PostgreSQL-backed session store."""

    def __init__(
        self,
        config: PostgresSessionConfig,
        *,
        pool: AsyncConnectionPool | None = None,
        init_error: str | None = None,
    ) -> None:
        self.config = config
        self._pool = pool
        self._init_error = init_error
        self._logger = get_file_logger(project_root=Path(__file__).resolve().parents[3])

    @classmethod
    async def create_from_env(cls) -> "PostgresSessionStore":
        return await cls.create(PostgresSessionConfig.from_env())

    @classmethod
    async def create(cls, config: PostgresSessionConfig) -> "PostgresSessionStore":
        store = cls(config)
        if not config.enabled:
            store._logger.info("session.store.disabled", reason="config_disabled")
            return store

        try:
            import psycopg  # type: ignore
        except Exception as exc:  # pragma: no cover
            config.enabled = False
            store._init_error = f"import_psycopg_failed:{exc}"
            store._logger.warning(
                "session.store.init_failed",
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
                "session.store.ready",
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
                "session.store.init_failed",
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
        ddl = f"""
        CREATE SCHEMA IF NOT EXISTS {self.config.schema};

        CREATE TABLE IF NOT EXISTS {self._table("qa_session")} (
            session_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            status TEXT NOT NULL DEFAULT 'idle',
            messages JSONB NOT NULL DEFAULT '[]'::jsonb,
            payload JSONB NOT NULL DEFAULT '{{}}'::jsonb
        );

        CREATE INDEX IF NOT EXISTS idx_qa_session_updated_at
            ON {self._table("qa_session")}(updated_at DESC);
        """
        started_at = perf_counter()
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(ddl)
        self._logger.info(
            "session.store.ensure_schema.completed",
            schema=self.config.schema,
            latency_ms=int((perf_counter() - started_at) * 1000),
        )

    def _normalize_session(self, session: dict[str, Any]) -> dict[str, Any]:
        session_id = str(session.get("id", "") or "").strip()
        if not session_id:
            raise ValueError("session_id_required")
        title = str(session.get("title", "") or "").strip() or f"新会话{session_id[-8:]}"
        created_at = _to_iso(session.get("created_at"))
        updated_at = _to_iso(session.get("updated_at"))
        status = str(session.get("status", "idle") or "idle")
        raw_messages = session.get("messages") or []
        messages = raw_messages if isinstance(raw_messages, list) else []
        known_keys = {"id", "title", "created_at", "updated_at", "status", "messages"}
        payload = {k: v for k, v in session.items() if k not in known_keys}
        return {
            "id": session_id,
            "title": title,
            "created_at": created_at,
            "updated_at": updated_at,
            "status": status,
            "messages": messages,
            "payload": payload,
        }

    def _deserialize_session_row(self, row: tuple[Any, ...]) -> dict[str, Any]:
        session_id, title, created_at, updated_at, status, messages, payload = row
        parsed_messages = messages if isinstance(messages, list) else json.loads(messages or "[]")
        parsed_payload = payload if isinstance(payload, dict) else json.loads(payload or "{}")
        result: dict[str, Any] = {
            "id": str(session_id),
            "title": str(title),
            "created_at": _to_iso(created_at),
            "updated_at": _to_iso(updated_at),
            "status": str(status),
            "messages": parsed_messages if isinstance(parsed_messages, list) else [],
        }
        if isinstance(parsed_payload, dict):
            result.update(parsed_payload)
        return result

    async def save_session(self, session: dict[str, Any]) -> None:
        if not self.is_active or self._pool is None:
            self._logger.debug("session.store.save.skipped", reason="inactive_store")
            return
        started_at = perf_counter()
        normalized = self._normalize_session(session)
        sql = f"""
        INSERT INTO {self._table("qa_session")} (
            session_id, title, created_at, updated_at, status, messages, payload
        ) VALUES (
            %s, %s, %s::timestamptz, %s::timestamptz, %s, %s::jsonb, %s::jsonb
        )
        ON CONFLICT (session_id) DO UPDATE SET
            title = EXCLUDED.title,
            updated_at = EXCLUDED.updated_at,
            status = EXCLUDED.status,
            messages = EXCLUDED.messages,
            payload = EXCLUDED.payload
        """
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    sql,
                    (
                        normalized["id"],
                        normalized["title"],
                        normalized["created_at"],
                        normalized["updated_at"],
                        normalized["status"],
                        json.dumps(normalized["messages"], ensure_ascii=False),
                        json.dumps(normalized["payload"], ensure_ascii=False),
                    ),
                )
        self._logger.debug(
            "session.store.save.completed",
            session_id=normalized["id"],
            message_count=len(normalized["messages"]),
            latency_ms=int((perf_counter() - started_at) * 1000),
        )

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        if not self.is_active or self._pool is None:
            self._logger.debug("session.store.get.skipped", reason="inactive_store")
            return None
        started_at = perf_counter()
        sql = f"""
        SELECT session_id, title, created_at, updated_at, status, messages, payload
        FROM {self._table("qa_session")}
        WHERE session_id = %s
        LIMIT 1
        """
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql, (session_id,))
                row = await cur.fetchone()
        if row is None:
            self._logger.debug(
                "session.store.get.miss",
                session_id=session_id,
                latency_ms=int((perf_counter() - started_at) * 1000),
            )
            return None
        self._logger.debug(
            "session.store.get.hit",
            session_id=session_id,
            latency_ms=int((perf_counter() - started_at) * 1000),
        )
        return self._deserialize_session_row(row)

    async def list_sessions(self, *, limit: int = 200) -> list[dict[str, Any]]:
        if not self.is_active or self._pool is None:
            self._logger.debug("session.store.list.skipped", reason="inactive_store")
            return []
        started_at = perf_counter()
        safe_limit = max(1, min(limit, 1000))
        sql = f"""
        SELECT session_id, title, created_at, updated_at, status, messages, payload
        FROM {self._table("qa_session")}
        ORDER BY updated_at DESC
        LIMIT %s
        """
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql, (safe_limit,))
                rows = await cur.fetchall()
        sessions = [self._deserialize_session_row(row) for row in (rows or [])]
        self._logger.debug(
            "session.store.list.completed",
            limit=safe_limit,
            returned=len(sessions),
            latency_ms=int((perf_counter() - started_at) * 1000),
        )
        return sessions

    async def find_message(self, message_id: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
        if not self.is_active or self._pool is None:
            self._logger.debug("session.store.find_message.skipped", reason="inactive_store")
            return None
        started_at = perf_counter()
        sql = f"""
        SELECT session_id, title, created_at, updated_at, status, messages, payload
        FROM {self._table("qa_session")} s
        WHERE EXISTS (
            SELECT 1
            FROM jsonb_array_elements(s.messages) AS msg
            WHERE msg->>'id' = %s
        )
        ORDER BY updated_at DESC
        LIMIT 1
        """
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql, (message_id,))
                row = await cur.fetchone()
        if row is None:
            self._logger.debug(
                "session.store.find_message.miss",
                message_id=message_id,
                latency_ms=int((perf_counter() - started_at) * 1000),
            )
            return None
        session = self._deserialize_session_row(row)
        for message in session.get("messages", []):
            if str(message.get("id", "")) == message_id:
                self._logger.debug(
                    "session.store.find_message.hit",
                    message_id=message_id,
                    session_id=session.get("id", ""),
                    latency_ms=int((perf_counter() - started_at) * 1000),
                )
                return session, message
        return None
