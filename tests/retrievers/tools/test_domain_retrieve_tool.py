# -*- coding: utf-8 -*-
from __future__ import annotations

from retrievers.tools.domain_retrieve_tool import DomainRetrieveInput, _build_state


class _StubRetrieval:
    enable_wiki = True
    enable_code = True
    max_per_source = {"wiki": 4, "code": 4}
    source_weights = {"wiki": 1.0, "code": 1.0}

    @staticmethod
    def preset(name: str) -> dict[str, int]:
        return {"wiki_top_k": 4, "code_top_k": 2}


class _StubProfile:
    retrieval = _StubRetrieval()
    profile_id = "ad_engine"

    def infer_related_modules(self, query: str, primary_module_name: str, limit: int = 2) -> list[dict[str, str]]:
        return [{"module_name": "rerank-engine"}]

    def infer_module(self, query: str) -> tuple[str, float]:
        return ("bid-optimizer", 0.9)

    def module_hint(self, module_name: str) -> str:
        return module_name


def test_build_state_uses_request_context_fields(monkeypatch) -> None:
    monkeypatch.setattr("retrievers.tools.domain_retrieve_tool.get_domain_profile", lambda: _StubProfile())

    payload = DomainRetrieveInput(query="eCPM 公式", intent="knowledge_qa", retrieval_bias="wiki_first")
    state = _build_state(
        payload,
        session_id="sess-ctx",
        trace_id="trace-ctx",
    )

    assert state["session_id"] == "sess-ctx"
    assert state["trace_id"] == "trace-ctx"
    assert state["module_name"] == "bid-optimizer"
    assert state["retrieval_queries"][0] == "eCPM 公式"
