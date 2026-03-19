# -*- coding: utf-8 -*-
"""LLM 配置定义"""
from dataclasses import dataclass, field
import os
from typing import Any


@dataclass
class LLMConfig:
    """LLM 配置

    支持 OpenAI 兼容的 LLM API。

    Attributes:
        model: 模型名称
        base_url: API 基础 URL (支持本地部署或代理)
        api_key: API 密钥
        temperature: 温度参数
        max_tokens: 最大 token 数
        timeout: 请求超时 (秒)
        extra_params: 额外参数
    """
    model: str = "gpt-4o"
    base_url: str = ""
    api_key: str = ""
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 120
    extra_params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls, prefix: str = "AGENT_LLM") -> "LLMConfig":
        """从环境变量加载配置

        环境变量命名规则：{PREFIX}_{KEY}
        - AGENT_LLM_MODEL
        - AGENT_LLM_BASE_URL
        - AGENT_LLM_API_KEY
        - AGENT_LLM_TEMPERATURE
        - AGENT_LLM_MAX_TOKENS
        - AGENT_LLM_TIMEOUT

        Args:
            prefix: 环境变量前缀

        Returns:
            LLMConfig 实例
        """
        def get_env(key: str, default: str = "") -> str:
            return os.getenv(f"{prefix}_{key}", default)

        def get_env_float(key: str, default: float) -> float:
            value = get_env(key)
            try:
                return float(value) if value else default
            except ValueError:
                return default

        def get_env_int(key: str, default: int) -> int:
            value = get_env(key)
            try:
                return int(value) if value else default
            except ValueError:
                return default

        return cls(
            model=get_env("MODEL", "gpt-4o"),
            base_url=get_env("BASE_URL", ""),
            api_key=get_env("API_KEY", ""),
            temperature=get_env_float("TEMPERATURE", 0.7),
            max_tokens=get_env_int("MAX_TOKENS", 4096),
            timeout=get_env_int("TIMEOUT", 120),
        )
