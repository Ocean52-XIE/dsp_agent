# -*- coding: utf-8 -*-
"""Per-request runtime context shared across async and tool layers."""
from __future__ import annotations

from contextvars import ContextVar, Token


_REQUEST_CONTEXT: ContextVar[dict[str, str]] = ContextVar("request_context", default={})


def set_request_context(*, session_id: str = "", trace_id: str = "") -> Token:
    """Bind the current request identifiers to the active async context."""
    payload = {
        "session_id": str(session_id or "").strip(),
        "trace_id": str(trace_id or "").strip(),
    }
    return _REQUEST_CONTEXT.set(payload)


def reset_request_context(token: Token) -> None:
    """Restore the previous request context."""
    _REQUEST_CONTEXT.reset(token)


def get_request_context() -> dict[str, str]:
    """Return the current request context as a plain dict."""
    return dict(_REQUEST_CONTEXT.get({}) or {})
