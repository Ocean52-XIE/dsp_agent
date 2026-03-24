from __future__ import annotations

from pathlib import Path

from domain_profile.profile import DomainProfile


def test_domain_profile_from_dict_without_legacy_fields() -> None:
    payload = {
        "profile_id": "demo",
        "sources": {
            "wiki": {"root": "wiki"},
            "code": {"roots": ["codes"]},
        },
        "routing": {"default_module": "primary"},
        "modules": [
            {
                "name": "primary",
                "hint": "primary module",
                "route_priority": 10,
                "keywords": ["serve"],
                "symbol_keywords": ["entry.py"],
                "aliases": ["online-serving"],
                "wiki_hints": ["00-"],
            }
        ],
        "retrieval": {},
        "answering": {},
        "prompts": {},
    }

    profile = DomainProfile.from_dict(payload, domain_dir=Path.cwd())

    assert profile.profile_id == "demo"
    assert profile.routing == {"default_module": "primary"}
    assert not hasattr(profile, "domain_gate")
    assert not hasattr(profile, "routing_profile")
    assert not hasattr(profile, "query_rewrite")
    assert not hasattr(profile, "code_generation")
    assert not hasattr(profile, "eval")
    assert profile.infer_module("please inspect entry.py") == ("primary", "primary module")
