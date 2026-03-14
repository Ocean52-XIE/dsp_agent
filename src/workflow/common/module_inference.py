# -*- coding: utf-8 -*-
"""Module inference helpers shared by multiple nodes."""
from __future__ import annotations

from workflow.domain_profile import DomainProfile


def infer_module(text: str, *, domain_profile: DomainProfile) -> tuple[str, str]:
    """Infer target module name and hint from query text."""
    default_module = domain_profile.default_module
    default_hint = domain_profile.module_hint(default_module)
    if not text:
        return default_module, default_hint

    lowered = text.lower()
    modules = sorted(domain_profile.modules, key=lambda item: item.route_priority)

    # Symbol-level routing takes precedence for code-location style queries.
    for module in modules:
        if module.symbol_keywords and any(token.lower() in lowered for token in module.symbol_keywords):
            return module.name, module.hint

    best_module_name = default_module
    best_module_hint = default_hint
    best_score = 0
    best_priority = 10**9
    for module in modules:
        keyword_score = sum(1 for token in module.keywords if token and token.lower() in lowered)
        alias_score = sum(1 for token in module.aliases if token and token.lower() in lowered)
        score = keyword_score + alias_score
        if score <= 0:
            continue
        if score > best_score or (score == best_score and module.route_priority < best_priority):
            best_score = score
            best_priority = module.route_priority
            best_module_name = module.name
            best_module_hint = module.hint

    return best_module_name, best_module_hint
