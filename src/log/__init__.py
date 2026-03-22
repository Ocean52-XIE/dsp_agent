# -*- coding: utf-8 -*-
"""日志模块

提供统一的日志配置和管理：
- setup_global_logging: 设置全局日志配置
- get_file_logger: 获取文件日志器单例
- WorkflowFileLogger: 工作流文件日志器

Usage:
    from log import get_file_logger, setup_global_logging
    from pathlib import Path

    # 在程序入口处初始化
    setup_global_logging(Path("/path/to/project"))

    # 获取日志器
    logger = get_file_logger(project_root=Path("/path/to/project"))
    logger.info("workflow.event", key="value")
"""
from __future__ import annotations

from log.runtime_logging import (
    WorkflowFileLogger,
    get_file_logger,
    get_global_log_path,
    get_global_log_status,
    setup_global_logging,
)

__all__ = [
    "WorkflowFileLogger",
    "get_file_logger",
    "get_global_log_path",
    "get_global_log_status",
    "setup_global_logging",
]
