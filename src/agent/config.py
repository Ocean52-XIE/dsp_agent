# -*- coding: utf-8 -*-
"""Deep Agent runtime config."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any


def _first_env(*names: str, default: str = "") -> str:
    for name in names:
        value = str(os.getenv(name, "") or "").strip()
        if value:
            return value
    return default


def _as_float(value: str, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: str, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class DeepAgentConfig:
    """Minimal config required to create a Deep Agent."""

    model: str
    api_key: str
    base_url: str
    temperature: float
    max_tokens: int
    timeout_seconds: int
    system_prompt: str
    backend_root: Path
    skills_root: str

    @classmethod
    def from_domain_profile(
        cls,
        *,
        domain_profile: Any,
        project_root: Path,
    ) -> "DeepAgentConfig":
        prompts = getattr(domain_profile, "prompts", {}) or {}
        system_prompt = str(prompts.get("deep_agent_system", "") or "").strip()
        if not system_prompt:
            system_prompt = (
                "You are an ad engine deep agent. "
                "Use the available skills before answering domain requests. "
                "When evidence is needed, call the domain_retrieve tool. "
                "Do not fabricate citations, paths, or implementation details."
            )

        raw = getattr(domain_profile, "raw", {}) or {}
        deep_agents = raw.get("deep_agents", {}) if isinstance(raw, dict) else {}
        configured_skills_root = str(deep_agents.get("skills_root", "") or "").strip()
        if configured_skills_root:
            virtual_skills_root = "/" + configured_skills_root.strip("/").replace("\\", "/")
        else:
            virtual_skills_root = f"/domain/{domain_profile.profile_id}/skills"

        return cls(
            model=_first_env(
                "DEEP_AGENT_MODEL",
                "WORKFLOW_QA_LLM_MODEL",
                "AGENT_LLM_MODEL",
                default="gpt-4o-mini",
            ),
            api_key=_first_env(
                "DEEP_AGENT_API_KEY",
                "WORKFLOW_QA_LLM_API_KEY",
                "AGENT_LLM_API_KEY",
            ),
            base_url=_first_env(
                "DEEP_AGENT_BASE_URL",
                "WORKFLOW_QA_LLM_BASE_URL",
                "AGENT_LLM_BASE_URL",
            ),
            temperature=_as_float(
                _first_env(
                    "DEEP_AGENT_TEMPERATURE",
                    "WORKFLOW_QA_LLM_TEMPERATURE",
                    "AGENT_LLM_TEMPERATURE",
                    default="0.1",
                ),
                0.1,
            ),
            max_tokens=_as_int(
                _first_env(
                    "DEEP_AGENT_MAX_TOKENS",
                    "WORKFLOW_QA_LLM_MAX_TOKENS",
                    "AGENT_LLM_MAX_TOKENS",
                    default="4096",
                ),
                4096,
            ),
            timeout_seconds=_as_int(
                _first_env(
                    "DEEP_AGENT_TIMEOUT_SECONDS",
                    "WORKFLOW_QA_LLM_TIMEOUT",
                    "AGENT_LLM_TIMEOUT",
                    default="60",
                ),
                60,
            ),
            system_prompt=system_prompt,
            backend_root=project_root.resolve(),
            skills_root=virtual_skills_root,
        )
