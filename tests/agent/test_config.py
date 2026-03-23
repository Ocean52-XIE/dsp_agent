# -*- coding: utf-8 -*-
"""Deep Agent 配置测试。"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from agent.config import DeepAgentConfig


def test_deep_agent_config_prefers_deep_agent_env(monkeypatch) -> None:
    """优先读取 Deep Agent 专用环境变量。"""
    monkeypatch.setenv("DEEP_AGENT_MODEL", "gpt-5-mini")
    monkeypatch.setenv("DEEP_AGENT_API_KEY", "deep-key")
    monkeypatch.setenv("DEEP_AGENT_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("DEEP_AGENT_TEMPERATURE", "0.25")
    monkeypatch.setenv("DEEP_AGENT_MAX_TOKENS", "2048")
    monkeypatch.setenv("DEEP_AGENT_TIMEOUT_SECONDS", "90")

    profile = SimpleNamespace(
        profile_id="ad_engine",
        prompts={"deep_agent_system": "system prompt"},
        raw={"deep_agents": {"skills_root": "domain/ad_engine/skills"}},
    )

    config = DeepAgentConfig.from_domain_profile(
        domain_profile=profile,
        project_root=Path("."),
    )

    assert config.model == "gpt-5-mini"
    assert config.api_key == "deep-key"
    assert config.base_url == "https://example.com/v1"
    assert config.temperature == 0.25
    assert config.max_tokens == 2048
    assert config.timeout_seconds == 90
    assert config.system_prompt == "system prompt"
    assert config.skills_root == "/domain/ad_engine/skills"


def test_deep_agent_config_falls_back_to_agent_env(monkeypatch) -> None:
    """当 Deep Agent 环境变量为空时回退到历史 Agent 配置。"""
    monkeypatch.delenv("DEEP_AGENT_MODEL", raising=False)
    monkeypatch.delenv("DEEP_AGENT_API_KEY", raising=False)
    monkeypatch.delenv("DEEP_AGENT_BASE_URL", raising=False)
    monkeypatch.delenv("DEEP_AGENT_TEMPERATURE", raising=False)
    monkeypatch.delenv("DEEP_AGENT_MAX_TOKENS", raising=False)
    monkeypatch.delenv("DEEP_AGENT_TIMEOUT_SECONDS", raising=False)
    monkeypatch.setenv("AGENT_LLM_MODEL", "fallback-model")
    monkeypatch.setenv("AGENT_LLM_API_KEY", "fallback-key")
    monkeypatch.setenv("AGENT_LLM_BASE_URL", "https://fallback.test")
    monkeypatch.setenv("AGENT_LLM_TEMPERATURE", "bad-float")
    monkeypatch.setenv("AGENT_LLM_MAX_TOKENS", "bad-int")
    monkeypatch.setenv("AGENT_LLM_TIMEOUT", "77")

    profile = SimpleNamespace(
        profile_id="ad_engine",
        prompts={},
        raw={},
    )

    config = DeepAgentConfig.from_domain_profile(
        domain_profile=profile,
        project_root=Path("."),
    )

    assert config.model == "fallback-model"
    assert config.api_key == "fallback-key"
    assert config.base_url == "https://fallback.test"
    assert config.temperature == 0.1
    assert config.max_tokens == 4096
    assert config.timeout_seconds == 77
    assert config.skills_root == "/domain/ad_engine/skills"
    assert "domain_retrieve tool" in config.system_prompt
