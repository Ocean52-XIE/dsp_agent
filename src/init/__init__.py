# -*- coding: utf-8 -*-
"""Initialization exports for the new deep-agent stack."""
from __future__ import annotations

from init.initializer import (
    initialize,
    initialize_async,
)
from init.database_initializer import (
    ensure_database_exists,
    init_database,
    get_database_status,
)

__all__ = [
    # 主入口
    "initialize",
    "initialize_async",
    # 数据库初始化
    "ensure_database_exists",
    "init_database",
    "get_database_status",
]
