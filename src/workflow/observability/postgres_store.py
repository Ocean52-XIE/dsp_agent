from __future__ import annotations

"""PostgreSQL 可观测性存储模块。

职责边界：
1. 记录单次问答请求明细（query/route/latency/LLM状态/证据命中）。
2. 记录证据明细（用于离线排障与召回分析）。
3. 记录用户反馈（helpful/unhelpful + 标签 + 文本）。
4. 计算窗口指标并按阈值触发告警事件。

设计原则：
- 低侵入：对上层只暴露 record_* / get_* 接口，不耦合业务节点实现细节。
- 可降级：数据库不可用时不阻断主链路，返回 status 供 health 接口观测。
- 可配置：全部连接与告警阈值通过环境变量驱动。
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
from workflow.utils import normalize_source_type, to_bool, to_float, to_int


def _sanitize_identifier(value: str, default: str) -> str:
    """校验 SQL 标识符（schema/table/column）。

    仅允许字母数字下划线，避免将用户输入直接拼接到 DDL 中。
    """
    normalized = (value or "").strip()
    if not normalized:
        return default
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", normalized):
        return normalized
    return default


@dataclass
class PostgresObservabilityConfig:
    """PostgreSQL 可观测性配置。"""

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

    @classmethod
    def from_env(cls) -> "PostgresObservabilityConfig":
        """从环境变量读取配置。"""
        dsn = os.getenv("WORKFLOW_OBS_PG_DSN", "").strip()
        enabled_default = bool(dsn)
        return cls(
            enabled=to_bool(os.getenv("WORKFLOW_OBS_PG_ENABLED"), enabled_default),
            dsn=dsn,
            schema=_sanitize_identifier(os.getenv("WORKFLOW_OBS_PG_SCHEMA", "public"), "public"),
            connect_timeout_seconds=max(1, to_int(os.getenv("WORKFLOW_OBS_PG_CONNECT_TIMEOUT_SECONDS"), 5)),
            alert_window_minutes=max(5, to_int(os.getenv("WORKFLOW_OBS_ALERT_WINDOW_MINUTES"), 30)),
            alert_min_samples=max(1, to_int(os.getenv("WORKFLOW_OBS_ALERT_MIN_SAMPLES"), 20)),
            alert_suppress_minutes=max(1, to_int(os.getenv("WORKFLOW_OBS_ALERT_SUPPRESS_MINUTES"), 30)),
            alert_empty_response_rate_max=max(
                0.0,
                min(1.0, to_float(os.getenv("WORKFLOW_OBS_ALERT_EMPTY_RESPONSE_RATE_MAX"), 0.05)),
            ),
            alert_fallback_rate_max=max(
                0.0,
                min(1.0, to_float(os.getenv("WORKFLOW_OBS_ALERT_FALLBACK_RATE_MAX"), 0.25)),
            ),
            alert_insufficient_rate_max=max(
                0.0,
                min(1.0, to_float(os.getenv("WORKFLOW_OBS_ALERT_INSUFFICIENT_RATE_MAX"), 0.20)),
            ),
            alert_p95_latency_ms_max=max(
                1.0,
                to_float(os.getenv("WORKFLOW_OBS_ALERT_P95_LATENCY_MS_MAX"), 3000.0),
            ),
            alert_exact_like_pass_rate_min=max(
                0.0,
                min(1.0, to_float(os.getenv("WORKFLOW_OBS_ALERT_EXACT_LIKE_PASS_RATE_MIN"), 0.70)),
            ),
        )


class PostgresObservabilityStore:
    """PostgreSQL 可观测性存储实现。"""

    def __init__(self, config: PostgresObservabilityConfig) -> None:
        self.config = config
        self._init_error: str | None = None
        self._psycopg: Any | None = None
        self._logger = get_file_logger(project_root=Path(__file__).resolve().parents[3])

        if not self.config.enabled:
            self._logger.info("observability.store.disabled", reason="config_disabled")
            return
        try:
            import psycopg  # type: ignore
        except Exception as exc:  # pragma: no cover - 依赖缺失时可观测
            self._init_error = f"import_psycopg_failed:{exc}"
            self.config.enabled = False
            self._logger.warning(
                "observability.store.init_failed",
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
                "observability.store.ready",
                schema=self.config.schema,
                connect_timeout_seconds=self.config.connect_timeout_seconds,
            )
        except Exception as exc:  # pragma: no cover - 数据库不可达时可观测
            self._init_error = f"bootstrap_or_schema_failed:{exc}"
            self.config.enabled = False
            self._logger.warning(
                "observability.store.init_failed",
                reason="bootstrap_or_schema_failed",
                error_type=type(exc).__name__,
            )

    @classmethod
    def from_env(cls) -> "PostgresObservabilityStore":
        """从环境变量创建存储对象。"""
        return cls(PostgresObservabilityConfig.from_env())

    @property
    def is_active(self) -> bool:
        """当前是否已激活可写数据库状态。"""
        return bool(self.config.enabled and self._psycopg is not None)

    def status(self) -> dict[str, Any]:
        """返回可观测模块状态，供 health 接口直接透出。"""
        return {
            "enabled": self.config.enabled,
            "active": self.is_active,
            "schema": self.config.schema,
            "dsn_configured": bool(self.config.dsn),
            "init_error": self._init_error,
        }

    def _connect(self) -> Any:
        """创建数据库连接（调用方负责 with 关闭）。"""
        if self._psycopg is None:
            raise RuntimeError("psycopg_unavailable")
        return self._psycopg.connect(
            self.config.dsn,
            autocommit=True,
            connect_timeout=self.config.connect_timeout_seconds,
        )

    def _table(self, name: str) -> str:
        """拼接带 schema 的表名。"""
        safe_name = _sanitize_identifier(name, name)
        return f"{self.config.schema}.{safe_name}"

    def ensure_schema(self) -> None:
        """创建可观测性所需表与索引。

        为了降低部署门槛，这里采用“应用启动自动建表”。
        """
        if not self.is_active:
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

        CREATE INDEX IF NOT EXISTS idx_qa_evidence_log_trace_id ON {evidence_table}(trace_id);
        CREATE INDEX IF NOT EXISTS idx_qa_evidence_log_created_at ON {evidence_table}(created_at DESC);

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

        CREATE INDEX IF NOT EXISTS idx_qa_feedback_log_created_at ON {feedback_table}(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_qa_feedback_log_trace_id ON {feedback_table}(trace_id);

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

        CREATE INDEX IF NOT EXISTS idx_qa_alert_event_created_at ON {alert_table}(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_qa_alert_event_type_status ON {alert_table}(alert_type, status, created_at DESC);

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

        CREATE INDEX IF NOT EXISTS idx_qa_metric_snapshot_created_at ON {snapshot_table}(created_at DESC);
        """
        started_at = perf_counter()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(ddl)
        self._logger.info(
            "observability.store.ensure_schema.completed",
            schema=self.config.schema,
            latency_ms=int((perf_counter() - started_at) * 1000),
        )

    def record_turn(
        self,
        *,
        turn_type: str,
        session_id: str,
        trace_id: str,
        message_id: str,
        user_query: str,
        assistant_message: dict[str, Any],
    ) -> None:
        """记录一条问答明细，并触发窗口告警计算。"""
        if not self.is_active:
            self._logger.debug("observability.store.record_turn.skipped", reason="inactive_store")
            return
        started_at = perf_counter()
        try:
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

            # exact_like_pass 是在线近似通过率，不等同离线严格 exact_correct。
            # 这里用于趋势告警，因此采用可在线计算的启发式规则。
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

            payload = {
                "analysis": analysis,
                "debug": debug,
            }
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
                source_counter["wiki"],
                source_counter["code"],
                source_counter["case"],
                json.dumps(payload, ensure_ascii=False),
            )

            evidence_sql = f"""
            INSERT INTO {self._table("qa_evidence_log")} (
                session_id, trace_id, message_id, rank_no, source_type, path, title, section,
                score, symbol_name, start_line, end_line, excerpt, payload
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb
            )
            """

            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(request_sql, request_args)
                    for rank_no, item in enumerate(citations, start=1):
                        cur.execute(
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
            self._evaluate_alerts()
            self._logger.debug(
                "observability.store.record_turn.completed",
                trace_id=trace_id,
                session_id=session_id,
                message_id=message_id,
                turn_type=turn_type,
                citation_count=len(citations),
                latency_ms=int((perf_counter() - started_at) * 1000),
            )
        except Exception as exc:
            # 可观测模块异常不应影响主请求链路。
            self._logger.warning(
                "observability.store.record_turn.failed",
                trace_id=trace_id,
                session_id=session_id,
                message_id=message_id,
                turn_type=turn_type,
                latency_ms=int((perf_counter() - started_at) * 1000),
                error_type=type(exc).__name__,
            )
            return

    def record_feedback(
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
        """记录人工反馈。"""
        if not self.is_active:
            self._logger.debug("observability.store.record_feedback.skipped", reason="inactive_store")
            return
        started_at = perf_counter()
        try:
            sql = f"""
            INSERT INTO {self._table("qa_feedback_log")} (
                session_id, trace_id, message_id, helpful, reason_tag, rating, comment, payload
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            """
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
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
            self._logger.debug(
                "observability.store.record_feedback.completed",
                trace_id=trace_id,
                session_id=session_id,
                message_id=message_id,
                helpful=helpful,
                latency_ms=int((perf_counter() - started_at) * 1000),
            )
        except Exception as exc:
            self._logger.warning(
                "observability.store.record_feedback.failed",
                trace_id=trace_id,
                session_id=session_id,
                message_id=message_id,
                latency_ms=int((perf_counter() - started_at) * 1000),
                error_type=type(exc).__name__,
            )
            return

    def get_summary(self, *, window_minutes: int) -> dict[str, Any]:
        """查询最近窗口指标摘要。"""
        if not self.is_active:
            self._logger.debug(
                "observability.store.get_summary.skipped",
                reason="inactive_store",
                window_minutes=int(window_minutes),
            )
            return self._empty_summary(window_minutes=window_minutes)
        started_at = perf_counter()
        try:
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
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (window_minutes,))
                    row = cur.fetchone() or (0, 0.0, 0.0, 0.0, 0.0, 0.0)

            sample_size = int(row[0] or 0)
            summary = {
                "window_minutes": window_minutes,
                "sample_size": sample_size,
                "empty_response_rate": float(row[1] or 0.0),
                "fallback_rate": float(row[2] or 0.0),
                "insufficient_rate": float(row[3] or 0.0),
                "p95_latency_ms": float(row[4] or 0.0),
                "exact_like_pass_rate": float(row[5] or 0.0),
                "generated_at": datetime.now().isoformat(timespec="seconds"),
            }
            summary["top_failed_queries"] = self._get_top_failed_queries(window_minutes=window_minutes, limit=10)
            self._logger.debug(
                "observability.store.get_summary.completed",
                window_minutes=window_minutes,
                sample_size=sample_size,
                latency_ms=int((perf_counter() - started_at) * 1000),
            )
            return summary
        except Exception as exc:
            self._logger.warning(
                "observability.store.get_summary.failed",
                window_minutes=int(window_minutes),
                latency_ms=int((perf_counter() - started_at) * 1000),
                error_type=type(exc).__name__,
            )
            return self._empty_summary(window_minutes=window_minutes)

    def list_alerts(self, *, limit: int = 50) -> list[dict[str, Any]]:
        """查询最近告警事件。"""
        if not self.is_active:
            self._logger.debug("observability.store.list_alerts.skipped", reason="inactive_store", limit=int(limit))
            return []
        started_at = perf_counter()
        try:
            sql = f"""
            SELECT id, created_at, alert_type, severity, metric_name, metric_value, threshold,
                   window_minutes, sample_size, status, detail
            FROM {self._table("qa_alert_event")}
            ORDER BY created_at DESC
            LIMIT %s
            """
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (max(1, int(limit)),))
                    rows = cur.fetchall() or []
            items: list[dict[str, Any]] = []
            for row in rows:
                items.append(
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
                )
            self._logger.debug(
                "observability.store.list_alerts.completed",
                limit=max(1, int(limit)),
                returned=len(items),
                latency_ms=int((perf_counter() - started_at) * 1000),
            )
            return items
        except Exception as exc:
            self._logger.warning(
                "observability.store.list_alerts.failed",
                limit=max(1, int(limit)),
                latency_ms=int((perf_counter() - started_at) * 1000),
                error_type=type(exc).__name__,
            )
            return []

    def _empty_summary(self, *, window_minutes: int) -> dict[str, Any]:
        """数据库不可用时的空摘要。"""
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

    def _get_top_failed_queries(self, *, window_minutes: int, limit: int) -> list[dict[str, Any]]:
        """统计最近窗口 exact_like 失败最多的 query。"""
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
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (window_minutes, limit))
                rows = cur.fetchall() or []
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

    def _evaluate_alerts(self) -> None:
        """基于窗口指标执行阈值告警。"""
        if not self.is_active:
            return
        summary = self.get_summary(window_minutes=self.config.alert_window_minutes)
        sample_size = int(summary.get("sample_size", 0))
        if sample_size < self.config.alert_min_samples:
            return

        checks: list[dict[str, Any]] = [
            {
                "alert_type": "empty_response_rate_high",
                "metric_name": "empty_response_rate",
                "metric_value": float(summary.get("empty_response_rate", 0.0)),
                "threshold": self.config.alert_empty_response_rate_max,
                "violate": float(summary.get("empty_response_rate", 0.0)) > self.config.alert_empty_response_rate_max,
                "severity": "high",
            },
            {
                "alert_type": "fallback_rate_high",
                "metric_name": "fallback_rate",
                "metric_value": float(summary.get("fallback_rate", 0.0)),
                "threshold": self.config.alert_fallback_rate_max,
                "violate": float(summary.get("fallback_rate", 0.0)) > self.config.alert_fallback_rate_max,
                "severity": "medium",
            },
            {
                "alert_type": "insufficient_rate_high",
                "metric_name": "insufficient_rate",
                "metric_value": float(summary.get("insufficient_rate", 0.0)),
                "threshold": self.config.alert_insufficient_rate_max,
                "violate": float(summary.get("insufficient_rate", 0.0)) > self.config.alert_insufficient_rate_max,
                "severity": "medium",
            },
            {
                "alert_type": "p95_latency_high",
                "metric_name": "p95_latency_ms",
                "metric_value": float(summary.get("p95_latency_ms", 0.0)),
                "threshold": self.config.alert_p95_latency_ms_max,
                "violate": float(summary.get("p95_latency_ms", 0.0)) > self.config.alert_p95_latency_ms_max,
                "severity": "high",
            },
            {
                "alert_type": "exact_like_pass_rate_low",
                "metric_name": "exact_like_pass_rate",
                "metric_value": float(summary.get("exact_like_pass_rate", 0.0)),
                "threshold": self.config.alert_exact_like_pass_rate_min,
                "violate": float(summary.get("exact_like_pass_rate", 0.0)) < self.config.alert_exact_like_pass_rate_min,
                "severity": "high",
            },
        ]

        for check in checks:
            if not check["violate"]:
                continue
            if self._has_recent_alert(alert_type=check["alert_type"], within_minutes=self.config.alert_suppress_minutes):
                continue
            self._insert_alert(
                alert_type=str(check["alert_type"]),
                severity=str(check["severity"]),
                metric_name=str(check["metric_name"]),
                metric_value=float(check["metric_value"]),
                threshold=float(check["threshold"]),
                sample_size=sample_size,
                summary=summary,
            )

        self._insert_snapshot(summary=summary)

    def _has_recent_alert(self, *, alert_type: str, within_minutes: int) -> bool:
        """判断近期是否已有同类型告警，用于告警抑制。"""
        sql = f"""
        SELECT 1
        FROM {self._table("qa_alert_event")}
        WHERE alert_type = %s
          AND created_at >= NOW() - (%s || ' minutes')::INTERVAL
          AND status = 'open'
        LIMIT 1
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (alert_type, within_minutes))
                return cur.fetchone() is not None

    def _insert_alert(
        self,
        *,
        alert_type: str,
        severity: str,
        metric_name: str,
        metric_value: float,
        threshold: float,
        sample_size: int,
        summary: dict[str, Any],
    ) -> None:
        """写入告警事件。"""
        sql = f"""
        INSERT INTO {self._table("qa_alert_event")} (
            alert_type, severity, metric_name, metric_value, threshold, window_minutes, sample_size, status, detail
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, 'open', %s::jsonb)
        """
        detail = {
            "summary": summary,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (
                        alert_type,
                        severity,
                        metric_name,
                        metric_value,
                        threshold,
                        self.config.alert_window_minutes,
                        sample_size,
                        json.dumps(detail, ensure_ascii=False),
                    ),
                )

    def _insert_snapshot(self, *, summary: dict[str, Any]) -> None:
        """写入窗口指标快照，便于做趋势图。"""
        sql = f"""
        INSERT INTO {self._table("qa_metric_snapshot")} (
            window_minutes, sample_size, empty_response_rate, fallback_rate, insufficient_rate,
            p95_latency_ms, exact_like_pass_rate, payload
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (
                        int(summary.get("window_minutes", 0)),
                        int(summary.get("sample_size", 0)),
                        float(summary.get("empty_response_rate", 0.0)),
                        float(summary.get("fallback_rate", 0.0)),
                        float(summary.get("insufficient_rate", 0.0)),
                        float(summary.get("p95_latency_ms", 0.0)),
                        float(summary.get("exact_like_pass_rate", 0.0)),
                        json.dumps(summary, ensure_ascii=False),
                    ),
                )
