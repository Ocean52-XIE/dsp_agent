# -*- coding: utf-8 -*-
"""
该模块实现`utils` 相关能力与辅助逻辑。
"""
from __future__ import annotations

import os
from typing import Any


_TRUE_VALUES = {"1", "true", "yes", "y", "on"}
_FALSE_VALUES = {"0", "false", "no", "n", "off"}


def to_bool(raw_value: str | None, default: bool) -> bool:
    """
    执行`to bool` 相关处理逻辑。
    
    参数:
        raw_value: 输入参数，用于控制当前处理逻辑。
        default: 输入参数，用于控制当前处理逻辑。
    
    返回:
        返回类型为 `bool` 的处理结果。
    """
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    return default


def to_int(raw_value: str | None, default: int) -> int:
    """
    执行`to int` 相关处理逻辑。
    
    参数:
        raw_value: 输入参数，用于控制当前处理逻辑。
        default: 输入参数，用于控制当前处理逻辑。
    
    返回:
        返回类型为 `int` 的处理结果。
    """
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def to_float(raw_value: str | None, default: float) -> float:
    """
    执行`to float` 相关处理逻辑。
    
    参数:
        raw_value: 输入参数，用于控制当前处理逻辑。
        default: 输入参数，用于控制当前处理逻辑。
    
    返回:
        返回类型为 `float` 的处理结果。
    """
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


def env_bool(name: str, default: bool) -> bool:
    """
    执行`env bool` 相关处理逻辑。
    
    参数:
        name: 输入参数，用于控制当前处理逻辑。
        default: 输入参数，用于控制当前处理逻辑。
    
    返回:
        返回类型为 `bool` 的处理结果。
    """
    return to_bool(os.getenv(name), default)


def env_int(name: str, default: int, minimum: int | None = None) -> int:
    """
    执行`env int` 相关处理逻辑。
    
    参数:
        name: 输入参数，用于控制当前处理逻辑。
        default: 输入参数，用于控制当前处理逻辑。
        minimum: 输入参数，用于控制当前处理逻辑。
    
    返回:
        返回类型为 `int` 的处理结果。
    """
    value = to_int(os.getenv(name), default)
    if minimum is not None:
        return max(value, minimum)
    return value


def env_float(name: str, default: float, minimum: float | None = None) -> float:
    """
    执行`env float` 相关处理逻辑。
    
    参数:
        name: 输入参数，用于控制当前处理逻辑。
        default: 输入参数，用于控制当前处理逻辑。
        minimum: 输入参数，用于控制当前处理逻辑。
    
    返回:
        返回类型为 `float` 的处理结果。
    """
    value = to_float(os.getenv(name), default)
    if minimum is not None:
        return max(value, minimum)
    return value


def normalize_source_type(raw_source: Any) -> str:
    """
    执行`normalize source type` 相关处理逻辑。
    
    参数:
        raw_source: 输入参数，用于控制当前处理逻辑。
    
    返回:
        返回类型为 `str` 的处理结果。
    """
    normalized = str(raw_source or "").strip().lower()
    if normalized.startswith("wiki"):
        return "wiki"
    if normalized.startswith("code"):
        return "code"
    if normalized.startswith("case"):
        return "case"
    return normalized or "unknown"

