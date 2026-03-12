from __future__ import annotations

"""PostgreSQL 会话存储模块。

职责边界：
1. 管理会话基础信息（id/title/status/时间戳）。
2. 管理会话消息数组（JSONB），用于恢复完整历史上下文。
3. 提供按 message_id 反查会话的能力，支撑 confirm-code/feedback 接口。

设计说明：
- 当前实现采用“一张会话表 + JSONB 消息数组”的轻量方案，便于快速落地。
- session_id/message_id/trace_id 推荐统一使用 UUID 形式，避免重启后发生 ID 冲突。
- 若数据库不可用，可由上层选择回退到内存模式（本模块只暴露状态，不做业务降级决策）。
"""

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from workflow.postgres_bootstrap import ensure_database_exists
from workflow.runtime_logging import get_file_logger


def _to_bool(raw_value: str | None, default: bool) -> bool:
    """将字符串环境变量转换为 bool。"""
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _to_int(raw_value: str | None, default: int) -> int:
    """将字符串环境变量转换为 int。"""
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def _sanitize_identifier(value: str, default: str) -> str:
    """校验 SQL 标识符（schema/table）。

    仅允许字母、数字、下划线，避免将不可信输入拼接进 SQL DDL。
    """
    normalized = (value or "").strip()
    if not normalized:
        return default
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", normalized):
        return normalized
    return default


def _to_iso(value: Any) -> str:
    """将时间值统一为 ISO 字符串（秒级）。"""
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
    """PostgreSQL 会话存储配置。"""

    enabled: bool
    dsn: str
    schema: str
    connect_timeout_seconds: int

    @classmethod
    def from_env(cls) -> "PostgresSessionConfig":
        """从环境变量读取配置。

        读取优先级：
        1. `WORKFLOW_SESSION_PG_*`（会话专用配置）
        2. `WORKFLOW_OBS_PG_*`（复用可观测数据库配置）
        """
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
            enabled=_to_bool(os.getenv("WORKFLOW_SESSION_PG_ENABLED"), enabled_default),
            dsn=dsn,
            schema=_sanitize_identifier(schema_env, "public"),
            connect_timeout_seconds=max(
                1,
                _to_int(
                    os.getenv("WORKFLOW_SESSION_PG_CONNECT_TIMEOUT_SECONDS")
                    or os.getenv("WORKFLOW_OBS_PG_CONNECT_TIMEOUT_SECONDS"),
                    5,
                ),
            ),
        )


class PostgresSessionStore:
    """基于 PostgreSQL 的会话存储实现。"""

    def __init__(self, config: PostgresSessionConfig) -> None:
        self.config = config
        self._psycopg: Any | None = None
        self._init_error: str | None = None
        self._logger = get_file_logger(project_root=Path(__file__).resolve().parents[2])

        if not self.config.enabled:
            self._logger.info("session.store.disabled", reason="config_disabled")
            return

        try:
            import psycopg  # type: ignore
        except Exception as exc:  # pragma: no cover - 依赖缺失
            self.config.enabled = False
            self._init_error = f"import_psycopg_failed:{exc}"
            self._logger.warning(
                "session.store.init_failed",
                reason="import_psycopg_failed",
                error_type=type(exc).__name__,
            )
            return

        self._psycopg = psycopg
        try:
            ensure_database_exists(
                psycopg_module=self._psycopg,
                dsn=self.config.dsn,
                connect_timeout_seconds=self.config.connect_timeout_seconds,
            )
            self.ensure_schema()
            self._logger.info(
                "session.store.ready",
                schema=self.config.schema,
                connect_timeout_seconds=self.config.connect_timeout_seconds,
            )
        except Exception as exc:  # pragma: no cover - 数据库异常
            self.config.enabled = False
            self._init_error = f"bootstrap_or_schema_failed:{exc}"
            self._logger.warning(
                "session.store.init_failed",
                reason="bootstrap_or_schema_failed",
                error_type=type(exc).__name__,
            )

    @classmethod
    def from_env(cls) -> "PostgresSessionStore":
        """基于环境变量创建实例。"""
        return cls(PostgresSessionConfig.from_env())

    @property
    def is_active(self) -> bool:
        """当前存储是否可用。"""
        return bool(self.config.enabled and self._psycopg is not None)

    def status(self) -> dict[str, Any]:
        """返回存储状态，便于 health 接口透出。"""
        return {
            "enabled": self.config.enabled,
            "active": self.is_active,
            "schema": self.config.schema,
            "dsn_configured": bool(self.config.dsn),
            "init_error": self._init_error,
        }

    def _table(self, name: str) -> str:
        """拼接 schema.table。"""
        safe_name = _sanitize_identifier(name, name)
        return f"{self.config.schema}.{safe_name}"

    def _connect(self) -> Any:
        """创建数据库连接。"""
        if self._psycopg is None:
            raise RuntimeError("psycopg_unavailable")
        return self._psycopg.connect(
            self.config.dsn,
            autocommit=True,
            connect_timeout=self.config.connect_timeout_seconds,
        )

    def ensure_schema(self) -> None:
        """自动建表与索引。"""
        if not self.is_active:
            return
        session_table = self._table("qa_session")
        ddl = f"""
        CREATE SCHEMA IF NOT EXISTS {self.config.schema};

        CREATE TABLE IF NOT EXISTS {session_table} (
            session_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            status TEXT NOT NULL DEFAULT 'idle',
            messages JSONB NOT NULL DEFAULT '[]'::jsonb,
            payload JSONB NOT NULL DEFAULT '{{}}'::jsonb
        );

        CREATE INDEX IF NOT EXISTS idx_qa_session_updated_at
            ON {session_table}(updated_at DESC);
        """
        started_at = perf_counter()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(ddl)
        self._logger.info(
            "session.store.ensure_schema.completed",
            schema=self.config.schema,
            latency_ms=int((perf_counter() - started_at) * 1000),
        )

    def _normalize_session(self, session: dict[str, Any]) -> dict[str, Any]:
        """归一化会话结构，避免脏数据落库。"""
        session_id = str(session.get("id", "") or "").strip()
        if not session_id:
            raise ValueError("session_id_required")

        title = str(session.get("title", "") or "").strip() or f"新会话 {session_id[-8:]}"
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

    def save_session(self, session: dict[str, Any]) -> None:
        """保存会话（不存在则插入，存在则更新）。"""
        if not self.is_active:
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
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
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
        except Exception as exc:
            self._logger.warning(
                "session.store.save.failed",
                session_id=normalized["id"],
                latency_ms=int((perf_counter() - started_at) * 1000),
                error_type=type(exc).__name__,
            )
            raise

    def _deserialize_session_row(self, row: tuple[Any, ...]) -> dict[str, Any]:
        """将 SQL 行反序列化为 API 侧会话结构。"""
        session_id, title, created_at, updated_at, status, messages, payload = row
        parsed_messages = messages
        if isinstance(parsed_messages, str):
            try:
                parsed_messages = json.loads(parsed_messages)
            except json.JSONDecodeError:
                parsed_messages = []
        if not isinstance(parsed_messages, list):
            parsed_messages = []

        parsed_payload = payload
        if isinstance(parsed_payload, str):
            try:
                parsed_payload = json.loads(parsed_payload)
            except json.JSONDecodeError:
                parsed_payload = {}
        if not isinstance(parsed_payload, dict):
            parsed_payload = {}

        result: dict[str, Any] = {
            "id": str(session_id),
            "title": str(title),
            "created_at": _to_iso(created_at),
            "updated_at": _to_iso(updated_at),
            "status": str(status),
            "messages": parsed_messages,
        }
        result.update(parsed_payload)
        return result

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """按 ID 获取完整会话。"""
        if not self.is_active:
            self._logger.debug("session.store.get.skipped", reason="inactive_store")
            return None
        started_at = perf_counter()
        sql = f"""
        SELECT session_id, title, created_at, updated_at, status, messages, payload
        FROM {self._table("qa_session")}
        WHERE session_id = %s
        LIMIT 1
        """
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (session_id,))
                    row = cur.fetchone()
        except Exception as exc:
            self._logger.warning(
                "session.store.get.failed",
                session_id=session_id,
                latency_ms=int((perf_counter() - started_at) * 1000),
                error_type=type(exc).__name__,
            )
            raise
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

    def list_sessions(self, *, limit: int = 200) -> list[dict[str, Any]]:
        """按更新时间倒序返回会话列表（包含消息体）。"""
        if not self.is_active:
            self._logger.debug("session.store.list.skipped", reason="inactive_store")
            return []
        started_at = perf_counter()
        sql = f"""
        SELECT session_id, title, created_at, updated_at, status, messages, payload
        FROM {self._table("qa_session")}
        ORDER BY updated_at DESC
        LIMIT %s
        """
        safe_limit = max(1, min(limit, 1000))
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (safe_limit,))
                    rows = cur.fetchall()
        except Exception as exc:
            self._logger.warning(
                "session.store.list.failed",
                limit=safe_limit,
                latency_ms=int((perf_counter() - started_at) * 1000),
                error_type=type(exc).__name__,
            )
            raise
        sessions = [self._deserialize_session_row(row) for row in rows]
        self._logger.debug(
            "session.store.list.completed",
            limit=safe_limit,
            returned=len(sessions),
            latency_ms=int((perf_counter() - started_at) * 1000),
        )
        return sessions

    def find_message(self, message_id: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
        """按 message_id 反查所属会话与消息体。"""
        if not self.is_active:
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
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (message_id,))
                    row = cur.fetchone()
        except Exception as exc:
            self._logger.warning(
                "session.store.find_message.failed",
                message_id=message_id,
                latency_ms=int((perf_counter() - started_at) * 1000),
                error_type=type(exc).__name__,
            )
            raise
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
        self._logger.debug(
            "session.store.find_message.miss_after_scan",
            message_id=message_id,
            session_id=session.get("id", ""),
            latency_ms=int((perf_counter() - started_at) * 1000),
        )
        return None
