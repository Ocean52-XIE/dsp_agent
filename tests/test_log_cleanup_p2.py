# -*- coding: utf-8 -*-
"""P2 启动日志清理回归测试。"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

import log as log_module
from agent.mcp.client import get_mcp_client, set_mcp_client
from retrievers.code.retriever import get_code_retriever, set_code_retriever
from retrievers.wiki.retriever import get_wiki_retriever, set_wiki_retriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("relative_path", "forbidden_fragment"),
    [
        ("src/retrievers/code/retriever.py", "print("),
        ("src/retrievers/core/cross_encoder_reranker.py", "print("),
    ],
)
def test_retriever_init_modules_do_not_use_console_prints(
    relative_path: str,
    forbidden_fragment: str,
) -> None:
    """初始化路径不应再混入控制台 print 输出。"""
    source = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
    assert forbidden_fragment not in source


def test_wiki_index_events_use_distinct_names() -> None:
    """Wiki 初始化日志应避免同名不同义事件。"""
    source = (PROJECT_ROOT / "src/retrievers/wiki/retriever.py").read_text(encoding="utf-8")

    assert source.count('"retriever.wiki.index.built"') == 1
    assert '"retriever.wiki.chunks.built"' in source


def test_retriever_event_namespaces_do_not_use_workflow_prefix() -> None:
    """Retriever 栈事件名应统一迁移出 workflow 前缀。"""
    paths = [
        "src/retrievers/core/embedding_retriever.py",
        "src/retrievers/core/cross_encoder_reranker.py",
        "src/retrievers/wiki/retriever.py",
        "src/retrievers/code/retriever.py",
        "src/log/__init__.py",
        "src/log/runtime_logging.py",
    ]

    for relative_path in paths:
        source = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
        assert "workflow." not in source


def test_log_module_exports_agent_file_logger_only() -> None:
    """日志模块对外只暴露新类名。"""
    assert hasattr(log_module, "AgentFileLogger")
    assert not hasattr(log_module, "WorkflowFileLogger")


@pytest.mark.parametrize(
    ("setter", "getter", "logger_name", "event_prefix"),
    [
        (set_mcp_client, get_mcp_client, "agent.mcp.client", "agent.mcp.singleton.updated"),
        (
            set_wiki_retriever,
            get_wiki_retriever,
            "retrievers.wiki.retriever",
            "wiki.retriever.singleton.updated",
        ),
        (
            set_code_retriever,
            get_code_retriever,
            "retrievers.code.retriever",
            "code.retriever.singleton.updated",
        ),
    ],
)
def test_singleton_updates_are_debug_logs(
    setter,
    getter,
    logger_name: str,
    event_prefix: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """单例更新属于实现细节，不应出现在 INFO 主视图。"""
    sentinel = object()

    with caplog.at_level(logging.INFO, logger=logger_name):
        setter(sentinel)
        setter(None)

    info_messages = [record.getMessage() for record in caplog.records if record.name == logger_name]
    assert info_messages == []

    caplog.clear()

    with caplog.at_level(logging.DEBUG, logger=logger_name):
        setter(sentinel)
        setter(None)

    debug_records = [record for record in caplog.records if record.name == logger_name]
    assert [record.levelno for record in debug_records] == [logging.DEBUG, logging.DEBUG]
    assert debug_records[0].getMessage() == f"{event_prefix} active=True"
    assert debug_records[1].getMessage() == f"{event_prefix} active=False"

    setter(None)
    assert getter() is None
