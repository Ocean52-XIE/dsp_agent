# -*- coding: utf-8 -*-
from __future__ import annotations

from retrievers.orchestration.wiki_flow import _grade_wiki_hits, _should_retry_wiki


def test_grade_wiki_hits_uses_normalized_thresholds() -> None:
    assert _grade_wiki_hits([]) == "insufficient"
    assert _grade_wiki_hits([{"score": 0.9}, {"score": 0.7}]) == "high"
    assert _grade_wiki_hits([{"score": 0.6}]) == "medium"
    assert _grade_wiki_hits([{"score": 0.4}]) == "low"


def test_grade_wiki_hits_keeps_legacy_thresholds_for_large_scores() -> None:
    assert _grade_wiki_hits([{"score": 6.2}, {"score": 5.0}]) == "high"
    assert _grade_wiki_hits([{"score": 3.4}]) == "medium"
    assert _grade_wiki_hits([{"score": 2.2}]) == "low"


def test_should_retry_wiki_uses_adaptive_default_threshold() -> None:
    assert _should_retry_wiki("low", 0.96, -1.0) is False
    assert _should_retry_wiki("low", 0.42, -1.0) is True
    assert _should_retry_wiki("medium", 0.42, -1.0) is False
    assert _should_retry_wiki("low", 3.4, -1.0) is False
    assert _should_retry_wiki("low", 2.4, -1.0) is True


def test_should_retry_wiki_respects_explicit_env_override() -> None:
    assert _should_retry_wiki("low", 0.96, 0.98) is True
    assert _should_retry_wiki("low", 0.96, 0.90) is False
