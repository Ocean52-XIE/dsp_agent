# -*- coding: utf-8 -*-
"""
该模块实现可观测性能力，负责日志、追踪或持久化记录。
"""
from __future__ import annotations

"""可观测性模块导出。"""

from workflow.observability.postgres_store import PostgresObservabilityConfig, PostgresObservabilityStore

__all__ = ["PostgresObservabilityConfig", "PostgresObservabilityStore"]
