# -*- coding: utf-8 -*-
"""API 配置模块

控制 API 版本路由和后端选择。

环境变量：
- API_BACKEND_VERSION: 后端版本 (v1 或 v2)，默认 v1
- API_DEBUG_VERBOSE: 是否开启详细调试日志
"""
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any


class BackendVersion(str, Enum):
    """后端版本枚举"""
    V1 = "v1"  # Workflow (现有实现)
    V2 = "v2"  # Agent (新架构)


def _as_str(value: Any, default: str = "") -> str:
    return str(value if value is not None else default).strip()


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes", "on")
    return bool(value)


@dataclass
class ApiConfig:
    """API 配置

    Attributes:
        backend_version: 后端版本 (v1 或 v2)
        debug_verbose: 是否开启详细调试
        v2_fallback_to_v1: v2 失败时是否回退到 v1
    """
    backend_version: BackendVersion = BackendVersion.V1
    debug_verbose: bool = False
    v2_fallback_to_v1: bool = True

    @classmethod
    def from_env(cls) -> "ApiConfig":
        """从环境变量加载配置

        环境变量：
        - API_BACKEND_VERSION: 后端版本 (v1/v2)
        - API_DEBUG_VERBOSE: 是否开启详细调试
        - API_V2_FALLBACK_TO_V1: v2 失败时是否回退到 v1
        """
        version_str = _as_str(os.getenv("API_BACKEND_VERSION", "v1")).lower()

        # 解析版本
        if version_str == "v2":
            backend_version = BackendVersion.V2
        else:
            backend_version = BackendVersion.V1

        debug_verbose = _as_bool(os.getenv("API_DEBUG_VERBOSE"), False)
        v2_fallback = _as_bool(os.getenv("API_V2_FALLBACK_TO_V1"), True)

        return cls(
            backend_version=backend_version,
            debug_verbose=debug_verbose,
            v2_fallback_to_v1=v2_fallback,
        )

    @property
    def use_v2(self) -> bool:
        """是否使用 v2 后端"""
        return self.backend_version == BackendVersion.V2

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "backend_version": self.backend_version.value,
            "debug_verbose": self.debug_verbose,
            "v2_fallback_to_v1": self.v2_fallback_to_v1,
        }


# 全局配置单例
_api_config: ApiConfig | None = None


def get_api_config() -> ApiConfig:
    """获取 API 配置单例

    Returns:
        ApiConfig 实例
    """
    global _api_config
    if _api_config is None:
        _api_config = ApiConfig.from_env()
    return _api_config


def reload_api_config() -> ApiConfig:
    """重新加载 API 配置

    Returns:
        新的 ApiConfig 实例
    """
    global _api_config
    _api_config = ApiConfig.from_env()
    return _api_config
