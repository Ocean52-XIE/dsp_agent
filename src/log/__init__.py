# -*- coding: utf-8 -*-
"""日志模块。

提供统一的日志配置和管理：
- `setup_global_logging`: 设置全局日志配置
- `get_file_logger`: 获取文件日志器单例
- `AgentFileLogger`: Agent 文件日志器

Usage:
    from pathlib import Path
    from log import get_file_logger, setup_global_logging

    setup_global_logging(Path("/path/to/project"))
    logger = get_file_logger(project_root=Path("/path/to/project"))
    logger.info("retriever.event", key="value")
"""
from __future__ import annotations

from log.runtime_logging import (
    AgentFileLogger,
    get_file_logger,
    get_global_log_path,
    get_global_log_status,
    setup_global_logging,
)

__all__ = [
    "AgentFileLogger",
    "get_file_logger",
    "get_global_log_path",
    "get_global_log_status",
    "setup_global_logging",
]
