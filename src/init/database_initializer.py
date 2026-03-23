# -*- coding: utf-8 -*-
"""数据库初始化器

负责初始化数据库相关组件：
- PostgreSQL 连接
- 数据库自动创建（bootstrap）
- LangGraph Checkpointer（用于会话持久化）

设计原则：
1. 在程序启动时完成数据库连接（按需）
2. 支持连接失败时的优雅降级
3. 提供统一的连接状态查询
4. 合并了原 bootstrap/postgres_bootstrap.py 的功能
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

logger = logging.getLogger(__name__)


# ============================================================================
# Bootstrap 工具函数（合并自 bootstrap/postgres_bootstrap.py）
# ============================================================================

def _extract_db_name_from_dsn(dsn: str) -> str:
    """从 DSN 中提取数据库名称

    Args:
        dsn: PostgreSQL 连接字符串

    Returns:
        数据库名称
    """
    parsed = urlsplit(dsn)
    db_name = (parsed.path or "").lstrip("/")
    return db_name


def _build_bootstrap_dsn(target_dsn: str, bootstrap_db: str) -> str:
    """构建 bootstrap 数据库的 DSN

    Args:
        target_dsn: 目标数据库 DSN
        bootstrap_db: bootstrap 数据库名称（默认 postgres）

    Returns:
        bootstrap 数据库的 DSN
    """
    parsed = urlsplit(target_dsn)
    bootstrap_path = f"/{bootstrap_db.strip()}"
    return urlunsplit((parsed.scheme, parsed.netloc, bootstrap_path, parsed.query, parsed.fragment))


def _ensure_database_exists(
    *,
    psycopg_module: Any,
    dsn: str,
    connect_timeout_seconds: int,
) -> None:
    """确保目标数据库存在，不存在则创建

    该函数负责"数据库级别"的初始化（CREATE DATABASE）。
    具体业务表结构初始化仍由各存储模块的 ensure_schema 负责。

    Args:
        psycopg_module: psycopg 模块
        dsn: PostgreSQL 连接字符串
        connect_timeout_seconds: 连接超时时间（秒）
    """
    normalized_dsn = (dsn or "").strip()
    if not normalized_dsn:
        raise ValueError("empty_dsn")

    target_db = _extract_db_name_from_dsn(normalized_dsn)
    if not target_db:
        raise ValueError("invalid_dsn_missing_db_name")

    bootstrap_db = os.getenv("AGENT_PG_BOOTSTRAP_DB", "postgres").strip() or "postgres"
    bootstrap_dsn = _build_bootstrap_dsn(normalized_dsn, bootstrap_db)

    # 使用 bootstrap 库检查并创建目标数据库
    with psycopg_module.connect(
        bootstrap_dsn,
        autocommit=True,
        connect_timeout=max(1, int(connect_timeout_seconds)),
    ) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname=%s", (target_db,))
            exists = cur.fetchone() is not None
            if exists:
                return

            sql_builder = getattr(psycopg_module, "sql", None)
            if sql_builder is None:
                raise RuntimeError("psycopg_sql_builder_unavailable")
            cur.execute(
                sql_builder.SQL("CREATE DATABASE {}").format(sql_builder.Identifier(target_db)),
            )
            logger.info(f"[DBInit] 数据库已创建: {target_db}")


# ============================================================================
# 配置类型
# ============================================================================

@dataclass
class DatabaseConfig:
    """数据库配置"""
    backend: str  # "memory" 或 "postgres"
    pg_dsn: str
    pg_enabled: bool
    pg_setup: bool
    connect_timeout_seconds: int


def _parse_database_config() -> DatabaseConfig:
    """从环境变量解析数据库配置"""
    from common.func_utils import to_bool, to_int

    explicit_backend = str(os.getenv("AGENT_CHECKPOINTER_BACKEND", "") or "").strip().lower()
    explicit_dsn = str(os.getenv("AGENT_CHECKPOINTER_PG_DSN", "") or "").strip()
    fallback_dsn = (
        str(os.getenv("AGENT_SESSION_PG_DSN", "") or "").strip()
        or str(os.getenv("AGENT_OBS_PG_DSN", "") or "").strip()
    )
    resolved_dsn = explicit_dsn or fallback_dsn

    if explicit_backend in {"memory", "postgres"}:
        backend = explicit_backend
    else:
        backend = "postgres" if resolved_dsn else "memory"

    pg_enabled_default = bool(resolved_dsn) and backend == "postgres"
    connect_timeout_raw = (
        str(os.getenv("AGENT_CHECKPOINTER_PG_CONNECT_TIMEOUT_SECONDS", "") or "").strip()
        or str(os.getenv("AGENT_SESSION_PG_CONNECT_TIMEOUT_SECONDS", "") or "").strip()
        or str(os.getenv("AGENT_OBS_PG_CONNECT_TIMEOUT_SECONDS", "") or "").strip()
    )

    connect_timeout_seconds = max(1, to_int(connect_timeout_raw, 5))
    dsn_with_timeout = _ensure_connect_timeout_in_dsn(resolved_dsn, connect_timeout_seconds)

    return DatabaseConfig(
        backend=backend,
        pg_dsn=dsn_with_timeout,
        pg_enabled=to_bool(os.getenv("AGENT_CHECKPOINTER_PG_ENABLED"), pg_enabled_default),
        pg_setup=to_bool(os.getenv("AGENT_CHECKPOINTER_PG_SETUP"), True),
        connect_timeout_seconds=connect_timeout_seconds,
    )


def _ensure_connect_timeout_in_dsn(dsn: str, timeout_seconds: int) -> str:
    """为 PostgreSQL DSN 注入 connect_timeout 参数"""
    normalized_dsn = str(dsn or "").strip()
    if not normalized_dsn:
        return normalized_dsn

    parsed = urlsplit(normalized_dsn)
    query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
    if any(str(key).lower() == "connect_timeout" for key, _ in query_pairs):
        return normalized_dsn

    query_pairs.append(("connect_timeout", str(max(1, int(timeout_seconds)))))
    updated_query = urlencode(query_pairs)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, updated_query, parsed.fragment))


# ============================================================================
# 数据库初始化
# ============================================================================

async def init_database_async() -> tuple[Any | None, Any | None]:
    """初始化数据库连接

    根据配置初始化 PostgreSQL 或内存 Checkpointer。

    Returns:
        (db_connection, checkpointer) 元组
    """
    config = _parse_database_config()

    # 内存模式
    if config.backend == "memory":
        logger.info("[DBInit] 使用内存 Checkpointer")
        return None, _create_memory_checkpointer("backend_memory")

    # PostgreSQL 禁用
    if not config.pg_enabled:
        logger.info("[DBInit] PostgreSQL 未启用，使用内存 Checkpointer")
        return None, _create_memory_checkpointer("pg_disabled")

    # 无 DSN
    if not config.pg_dsn:
        logger.warning("[DBInit] 无 PostgreSQL DSN，使用内存 Checkpointer")
        return None, _create_memory_checkpointer("empty_dsn")

    # 尝试连接 PostgreSQL
    return await _init_postgres_async(config)


def _create_memory_checkpointer(reason: str) -> Any:
    """创建内存 Checkpointer"""
    from langgraph.checkpoint.memory import MemorySaver

    logger.info(f"[DBInit] 创建内存 Checkpointer: reason={reason}")
    return MemorySaver()


async def _init_postgres_async(config: DatabaseConfig) -> tuple[Any | None, Any | None]:
    """初始化 PostgreSQL 连接

    Args:
        config: 数据库配置

    Returns:
        (connection, checkpointer) 元组
    """
    try:
        import psycopg
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    except ImportError as exc:
        logger.warning(
            f"[DBInit] PostgreSQL 依赖缺失，使用内存 Checkpointer: {exc}"
        )
        return None, _create_memory_checkpointer("import_dependency_failed")

    stack: AsyncExitStack | None = None
    try:
        # 确保数据库存在（使用本地函数）
        await asyncio.to_thread(
            _ensure_database_exists,
            psycopg_module=psycopg,
            dsn=config.pg_dsn,
            connect_timeout_seconds=config.connect_timeout_seconds,
        )

        # 创建 Checkpointer
        stack = AsyncExitStack()
        checkpointer = await stack.enter_async_context(
            AsyncPostgresSaver.from_conn_string(config.pg_dsn)
        )

        # 初始化表结构
        if config.pg_setup:
            await checkpointer.setup()

        logger.info(
            f"[DBInit] PostgreSQL Checkpointer 初始化完成: "
            f"timeout={config.connect_timeout_seconds}s, "
            f"setup={config.pg_setup}"
        )

        return stack, checkpointer

    except Exception as exc:
        if stack is not None:
            await stack.aclose()

        logger.warning(
            f"[DBInit] PostgreSQL 初始化失败，使用内存 Checkpointer: "
            f"error_type={type(exc).__name__}, error={exc}"
        )
        return None, _create_memory_checkpointer("postgres_init_failed")


# ============================================================================
# 状态查询
# ============================================================================

def get_database_status(checkpointer: Any) -> dict[str, Any]:
    """获取数据库状态

    Args:
        checkpointer: Checkpointer 实例

    Returns:
        状态字典
    """
    if checkpointer is None:
        return {
            "backend": "none",
            "status": "not_initialized",
        }

    from langgraph.checkpoint.memory import MemorySaver

    if isinstance(checkpointer, MemorySaver):
        return {
            "backend": "memory",
            "status": "active",
        }

    # PostgreSQL
    return {
        "backend": "postgres",
        "status": "active",
        "type": type(checkpointer).__name__,
    }


# ============================================================================
# 公开接口（用于其他模块直接调用）
# ============================================================================

def ensure_database_exists(
    *,
    psycopg_module: Any,
    dsn: str,
    connect_timeout_seconds: int,
) -> None:
    """确保目标数据库存在，不存在则创建（公开接口）

    该函数是对 _ensure_database_exists 的公开封装，
    供 observability、session 等模块使用。

    Args:
        psycopg_module: psycopg 模块
        dsn: PostgreSQL 连接字符串
        connect_timeout_seconds: 连接超时时间（秒）
    """
    _ensure_database_exists(
        psycopg_module=psycopg_module,
        dsn=dsn,
        connect_timeout_seconds=connect_timeout_seconds,
    )
