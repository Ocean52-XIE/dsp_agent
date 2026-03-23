# -*- coding: utf-8 -*-
"""Tests for the deep agent factory."""
from __future__ import annotations

from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

from agent.factory import create_agent


def test_create_agent_assembles_deep_agent(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_LLM_MODEL", "factory-model")

    class _FakeFilesystemBackend:
        def __init__(self, *, root_dir: str, virtual_mode: bool) -> None:
            self.root_dir = root_dir
            self.virtual_mode = virtual_mode

    deepagents_module = ModuleType("deepagents")
    backends_module = ModuleType("deepagents.backends")
    created: dict[str, object] = {}

    def _fake_create_deep_agent(**kwargs):
        created.update(kwargs)
        return SimpleNamespace(agent="ok")

    deepagents_module.create_deep_agent = _fake_create_deep_agent
    backends_module.FilesystemBackend = _FakeFilesystemBackend
    monkeypatch.setitem(sys.modules, "deepagents", deepagents_module)
    monkeypatch.setitem(sys.modules, "deepagents.backends", backends_module)

    monkeypatch.setattr(
        "agent.factory.get_domain_profile",
        lambda project_root: SimpleNamespace(
            profile_id="ad_engine",
            prompts={"deep_agent_system": "system"},
            raw={"deep_agents": {"skills_root": "domain/ad_engine/skills"}},
        ),
    )
    monkeypatch.setattr("agent.factory.create_domain_retrieve_tool", lambda: "retrieve-tool")
    monkeypatch.setattr("agent.factory.get_mcp_tools", lambda: ["mcp-tool"])
    monkeypatch.setattr(
        "agent.factory._build_model",
        lambda *, config, project_root: {"model": config.model, "project_root": project_root},
    )

    agent = create_agent(project_root=Path("."), checkpointer="checkpoint")

    assert agent.agent == "ok"
    assert created["model"]["model"] == "factory-model"
    assert created["tools"] == ["retrieve-tool", "mcp-tool"]
    assert created["system_prompt"] == "system"
    assert created["skills"] == ["/domain/ad_engine/skills"]
    assert created["backend"].virtual_mode is True
    assert created["checkpointer"] == "checkpoint"
