# -*- coding: utf-8 -*-
"""测试基础配置。"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest


_TMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp" / "pytest"
_TMP_ROOT.mkdir(parents=True, exist_ok=True)
os.environ["TMP"] = str(_TMP_ROOT)
os.environ["TEMP"] = str(_TMP_ROOT)
os.environ["TMPDIR"] = str(_TMP_ROOT)
tempfile.tempdir = str(_TMP_ROOT)


def _ensure_src_on_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def pytest_configure(config) -> None:
    """在收集测试前确保可导入 `src/` 下的模块。"""
    _ensure_src_on_path()


_ensure_src_on_path()


@pytest.fixture(autouse=True)
def setup_env(monkeypatch) -> None:
    """设置测试默认环境变量。"""
    monkeypatch.setenv("TMP", str(_TMP_ROOT))
    monkeypatch.setenv("TEMP", str(_TMP_ROOT))
    monkeypatch.setenv("TMPDIR", str(_TMP_ROOT))
    monkeypatch.setenv("AGENT_LLM_MODEL", "gpt-4o")
    monkeypatch.setenv("AGENT_LLM_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("AGENT_LLM_API_KEY", "test-api-key")
    monkeypatch.setenv("AGENT_LLM_TEMPERATURE", "0.1")
    monkeypatch.setenv("AGENT_LLM_MAX_TOKENS", "4096")
    monkeypatch.setenv("AGENT_LLM_TIMEOUT", "120")
