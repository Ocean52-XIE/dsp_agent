from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
from typing import Any


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


class WorkflowFileLogger:
    """Minimal file logger for workflow runtime events."""

    def __init__(self, *, project_root: Path) -> None:
        self.project_root = Path(project_root)
        self.enabled = _env_bool("WORKFLOW_FILE_LOG_ENABLED", True)
        self.level_name = str(os.getenv("WORKFLOW_FILE_LOG_LEVEL", "INFO") or "INFO").upper()
        self.max_bytes = max(1024, _env_int("WORKFLOW_FILE_LOG_MAX_BYTES", 5 * 1024 * 1024))
        self.backup_count = max(1, _env_int("WORKFLOW_FILE_LOG_BACKUP_COUNT", 3))
        self.log_dir = Path(os.getenv("WORKFLOW_FILE_LOG_DIR", str(self.project_root / "logs")))
        self.file_name = str(os.getenv("WORKFLOW_FILE_LOG_FILE", "workflow.log") or "workflow.log").strip() or "workflow.log"
        self.log_path = self.log_dir / self.file_name
        self.init_error = ""
        self._logger = logging.getLogger("workflow.runtime")
        self._logger.propagate = False
        self._configured = False
        self._configure()

    def _configure(self) -> None:
        if not self.enabled:
            return
        if self._logger.handlers:
            self._configured = True
            return
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            handler = RotatingFileHandler(
                self.log_path,
                maxBytes=self.max_bytes,
                backupCount=self.backup_count,
                encoding="utf-8",
            )
            formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
            self._logger.setLevel(getattr(logging, self.level_name, logging.INFO))
            self._configured = True
        except Exception as exc:
            self.init_error = str(exc)
            self._configured = False

    @property
    def is_active(self) -> bool:
        return self.enabled and self._configured and not self.init_error

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "active": self.is_active,
            "level": self.level_name,
            "path": str(self.log_path),
            "init_error": self.init_error or None,
        }

    def _emit(self, level: str, event: str, **payload: Any) -> None:
        if not self.is_active:
            return
        log_fn = getattr(self._logger, level, self._logger.info)
        if payload:
            message = f"{event} | {json.dumps(payload, ensure_ascii=False, default=str, separators=(',', ':'))}"
        else:
            message = event
        log_fn(message)

    def debug(self, event: str, **payload: Any) -> None:
        self._emit("debug", event, **payload)

    def info(self, event: str, **payload: Any) -> None:
        self._emit("info", event, **payload)

    def warning(self, event: str, **payload: Any) -> None:
        self._emit("warning", event, **payload)

    def error(self, event: str, **payload: Any) -> None:
        self._emit("error", event, **payload)

    def exception(self, event: str, **payload: Any) -> None:
        if not self.is_active:
            return
        message = f"{event} | {json.dumps(payload, ensure_ascii=False, default=str, separators=(',', ':'))}" if payload else event
        self._logger.exception(message)


_LOGGER_SINGLETON: WorkflowFileLogger | None = None


def get_file_logger(*, project_root: Path) -> WorkflowFileLogger:
    global _LOGGER_SINGLETON
    if _LOGGER_SINGLETON is None:
        _LOGGER_SINGLETON = WorkflowFileLogger(project_root=project_root)
    return _LOGGER_SINGLETON
