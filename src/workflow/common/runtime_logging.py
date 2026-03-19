# -*- coding: utf-8 -*-
"""全局运行时日志配置

该模块实现统一的全局日志配置：
1. 所有模块共享同一个日志 handler
2. 日志统一输出到 agent.log
3. 支持 workflow 和 agent 模块的统一日志管理
"""
from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
from typing import Any

from workflow.common.func_utils import env_bool, env_int


# 全局共享的日志 handler
_GLOBAL_HANDLER: RotatingFileHandler | None = None
_GLOBAL_LOG_PATH: Path | None = None
_GLOBAL_LOG_CONFIGURED = False


def _configure_global_logging(project_root: Path) -> RotatingFileHandler | None:
    """配置全局日志 handler

    为所有模块创建共享的日志 handler，确保日志统一输出到 agent.log

    Args:
        project_root: 项目根目录

    Returns:
        配置好的 RotatingFileHandler，配置失败返回 None
    """
    global _GLOBAL_HANDLER, _GLOBAL_LOG_PATH, _GLOBAL_LOG_CONFIGURED

    if _GLOBAL_LOG_CONFIGURED:
        return _GLOBAL_HANDLER

    # 读取配置
    enabled = env_bool("WORKFLOW_FILE_LOG_ENABLED", True)
    if not enabled:
        _GLOBAL_LOG_CONFIGURED = True
        return None

    level_name = str(os.getenv("WORKFLOW_FILE_LOG_LEVEL", "INFO") or "INFO").upper()
    max_bytes = max(1024, env_int("WORKFLOW_FILE_LOG_MAX_BYTES", 5 * 1024 * 1024))
    backup_count = max(1, env_int("WORKFLOW_FILE_LOG_BACKUP_COUNT", 3))
    log_dir = Path(os.getenv("WORKFLOW_FILE_LOG_DIR", str(project_root / "logs")))
    # 固定使用 agent.log
    file_name = "agent.log"

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        _GLOBAL_LOG_PATH = log_dir / file_name

        # 创建全局 handler
        _GLOBAL_HANDLER = RotatingFileHandler(
            _GLOBAL_LOG_PATH,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
        )
        _GLOBAL_HANDLER.setFormatter(formatter)

        # 配置根 logger，让所有模块共享同一个 handler
        root_logger = logging.getLogger()
        root_logger.addHandler(_GLOBAL_HANDLER)
        root_logger.setLevel(getattr(logging, level_name, logging.INFO))

        # 为 workflow 和 agent 模块配置日志级别
        for module_prefix in ["workflow", "agent"]:
            module_logger = logging.getLogger(module_prefix)
            module_logger.setLevel(getattr(logging, level_name, logging.INFO))
            # 确保传播到根 logger
            module_logger.propagate = True

        _GLOBAL_LOG_CONFIGURED = True

        # 记录初始化日志
        init_logger = logging.getLogger("workflow.runtime")
        init_logger.info(
            "runtime_logging.initialized | "
            f"log_path={_GLOBAL_LOG_PATH}, level={level_name}"
        )

        return _GLOBAL_HANDLER

    except Exception as exc:
        _GLOBAL_LOG_CONFIGURED = True
        print(f"[runtime_logging] 配置日志失败: {exc}")
        return None


def get_global_log_path() -> Path | None:
    """获取全局日志文件路径

    Returns:
        日志文件路径，未配置返回 None
    """
    return _GLOBAL_LOG_PATH


def get_global_log_status() -> dict[str, Any]:
    """获取全局日志配置状态

    Returns:
        日志配置状态字典
    """
    return {
        "configured": _GLOBAL_LOG_CONFIGURED,
        "log_path": str(_GLOBAL_LOG_PATH) if _GLOBAL_LOG_PATH else None,
        "handler_active": _GLOBAL_HANDLER is not None,
    }


class WorkflowFileLogger:
    """工作流文件日志器

    封装日志操作，提供统一的日志接口。
    所有日志最终通过全局 handler 输出到 agent.log
    """

    def __init__(self, *, project_root: Path) -> None:
        """初始化日志器

        Args:
            project_root: 项目根目录
        """
        self.project_root = Path(project_root)
        self.enabled = env_bool("WORKFLOW_FILE_LOG_ENABLED", True)
        self.level_name = str(os.getenv("WORKFLOW_FILE_LOG_LEVEL", "INFO") or "INFO").upper()
        self.max_bytes = max(1024, env_int("WORKFLOW_FILE_LOG_MAX_BYTES", 5 * 1024 * 1024))
        self.backup_count = max(1, env_int("WORKFLOW_FILE_LOG_BACKUP_COUNT", 3))
        self.log_dir = Path(os.getenv("WORKFLOW_FILE_LOG_DIR", str(self.project_root / "logs")))
        # 固定使用 agent.log
        self.file_name = "agent.log"
        self.log_path = self.log_dir / self.file_name
        self.init_error = ""
        self._logger = logging.getLogger("workflow.runtime")
        self._configured = False
        self._configure()

    def _configure(self) -> None:
        """配置日志"""
        if not self.enabled:
            return

        # 使用全局配置
        handler = _configure_global_logging(self.project_root)
        if handler:
            self._configured = True
            self.log_path = _GLOBAL_LOG_PATH or self.log_path
        else:
            self.init_error = "全局日志配置失败"
            self._configured = False

    @property
    def is_active(self) -> bool:
        """判断日志器是否可用"""
        return self.enabled and self._configured and not self.init_error

    def status(self) -> dict[str, Any]:
        """获取日志器状态"""
        return {
            "enabled": self.enabled,
            "active": self.is_active,
            "level": self.level_name,
            "path": str(self.log_path),
            "init_error": self.init_error or None,
        }

    def _emit(self, level: str, event: str, **payload: Any) -> None:
        """发送日志"""
        if not self.is_active:
            return
        log_fn = getattr(self._logger, level, self._logger.info)
        if payload:
            message = f"{event} | {json.dumps(payload, ensure_ascii=False, default=str, separators=(',', ':'))}"
        else:
            message = event
        log_fn(message)

    def debug(self, event: str, **payload: Any) -> None:
        """记录 DEBUG 日志"""
        self._emit("debug", event, **payload)

    def info(self, event: str, **payload: Any) -> None:
        """记录 INFO 日志"""
        self._emit("info", event, **payload)

    def warning(self, event: str, **payload: Any) -> None:
        """记录 WARNING 日志"""
        self._emit("warning", event, **payload)

    def error(self, event: str, **payload: Any) -> None:
        """记录 ERROR 日志"""
        self._emit("error", event, **payload)

    def exception(self, event: str, **payload: Any) -> None:
        """记录异常日志"""
        if not self.is_active:
            return
        message = f"{event} | {json.dumps(payload, ensure_ascii=False, default=str, separators=(',', ':'))}" if payload else event
        self._logger.exception(message)


# 全局单例
_LOGGER_SINGLETON: WorkflowFileLogger | None = None


def get_file_logger(*, project_root: Path) -> WorkflowFileLogger:
    """获取文件日志器单例

    Args:
        project_root: 项目根目录

    Returns:
        WorkflowFileLogger 实例
    """
    global _LOGGER_SINGLETON
    if _LOGGER_SINGLETON is None:
        _LOGGER_SINGLETON = WorkflowFileLogger(project_root=project_root)
    return _LOGGER_SINGLETON


def setup_global_logging(project_root: Path) -> None:
    """设置全局日志配置

    可在程序入口处调用，提前初始化全局日志配置

    Args:
        project_root: 项目根目录
    """
    _configure_global_logging(project_root)
