from __future__ import annotations

"""PostgreSQL 启动引导工具。

职责：
1. 在服务启动时根据业务 DSN 自动检查目标数据库是否存在。
2. 若不存在，尝试连接 bootstrap 数据库并自动创建目标数据库。

说明：
- 该模块只负责“数据库级别”的初始化（CREATE DATABASE）。
- 具体业务表结构初始化仍由各存储模块的 ensure_schema 负责。
"""

import os
from urllib.parse import urlsplit, urlunsplit


def _extract_db_name_from_dsn(dsn: str) -> str:
    """从 DSN 中提取目标数据库名称。"""
    parsed = urlsplit(dsn)
    db_name = (parsed.path or "").lstrip("/")
    return db_name


def _build_bootstrap_dsn(target_dsn: str, bootstrap_db: str) -> str:
    """把目标 DSN 替换为 bootstrap 数据库 DSN。"""
    parsed = urlsplit(target_dsn)
    bootstrap_path = f"/{bootstrap_db.strip()}"
    return urlunsplit((parsed.scheme, parsed.netloc, bootstrap_path, parsed.query, parsed.fragment))


def ensure_database_exists(
    *,
    psycopg_module: object,
    dsn: str,
    connect_timeout_seconds: int,
) -> None:
    """确保目标数据库存在。

    参数：
    - psycopg_module: 已导入的 psycopg 模块对象。
    - dsn: 业务目标数据库 DSN。
    - connect_timeout_seconds: 连接超时秒数。

    环境变量：
    - WORKFLOW_PG_BOOTSTRAP_DB: bootstrap 数据库名（默认 `postgres`）。
    """
    normalized_dsn = (dsn or "").strip()
    if not normalized_dsn:
        raise ValueError("empty_dsn")

    target_db = _extract_db_name_from_dsn(normalized_dsn)
    if not target_db:
        raise ValueError("invalid_dsn_missing_db_name")

    bootstrap_db = os.getenv("WORKFLOW_PG_BOOTSTRAP_DB", "postgres").strip() or "postgres"
    bootstrap_dsn = _build_bootstrap_dsn(normalized_dsn, bootstrap_db)

    # 使用 bootstrap 库检查并创建目标数据库。
    with psycopg_module.connect(  # type: ignore[attr-defined]
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

