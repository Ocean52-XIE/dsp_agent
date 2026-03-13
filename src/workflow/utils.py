from __future__ import annotations

import os
from typing import Any


_TRUE_VALUES = {"1", "true", "yes", "y", "on"}
_FALSE_VALUES = {"0", "false", "no", "n", "off"}


def to_bool(raw_value: str | None, default: bool) -> bool:
    """Convert a raw string value to bool with a default fallback."""
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    return default


def to_int(raw_value: str | None, default: int) -> int:
    """Convert a raw string value to int with a default fallback."""
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def to_float(raw_value: str | None, default: float) -> float:
    """Convert a raw string value to float with a default fallback."""
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


def env_bool(name: str, default: bool) -> bool:
    """Read an environment variable and parse it as bool."""
    return to_bool(os.getenv(name), default)


def env_int(name: str, default: int, minimum: int | None = None) -> int:
    """Read an environment variable and parse it as int."""
    value = to_int(os.getenv(name), default)
    if minimum is not None:
        return max(value, minimum)
    return value


def env_float(name: str, default: float, minimum: float | None = None) -> float:
    """Read an environment variable and parse it as float."""
    value = to_float(os.getenv(name), default)
    if minimum is not None:
        return max(value, minimum)
    return value


def normalize_source_type(raw_source: Any) -> str:
    """Normalize evidence source type to a stable enum value."""
    normalized = str(raw_source or "").strip().lower()
    if normalized.startswith("wiki"):
        return "wiki"
    if normalized.startswith("code"):
        return "code"
    if normalized.startswith("case"):
        return "case"
    return normalized or "unknown"

