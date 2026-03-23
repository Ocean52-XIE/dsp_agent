# -*- coding: utf-8 -*-
"""Tests for DeepAgentConfig."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from agent.config import DeepAgentConfig


def test_deep_agent_config_reads_agent_env(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_LLM_MODEL", "gpt-5-mini")
    monkeypatch.setenv("AGENT_LLM_API_KEY", "deep-key")
    monkeypatch.setenv("AGENT_LLM_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("AGENT_LLM_TEMPERATURE", "0.25")
    monkeypatch.setenv("AGENT_LLM_MAX_TOKENS", "2048")
    monkeypatch.setenv("AGENT_LLM_TIMEOUT_SECONDS", "90")

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


def test_deep_agent_config_uses_defaults_for_invalid_agent_env(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_LLM_MODEL", "fallback-model")
    monkeypatch.setenv("AGENT_LLM_API_KEY", "fallback-key")
    monkeypatch.setenv("AGENT_LLM_BASE_URL", "https://fallback.test")
    monkeypatch.setenv("AGENT_LLM_TEMPERATURE", "bad-float")
    monkeypatch.setenv("AGENT_LLM_MAX_TOKENS", "bad-int")
    monkeypatch.setenv("AGENT_LLM_TIMEOUT_SECONDS", "77")

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
