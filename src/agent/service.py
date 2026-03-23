# -*- coding: utf-8 -*-
"""Single deep-agent service entrypoint."""
from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from typing import Any

from agent.factory import create_agent
from agent.result_parser import DeepAgentTurnResult, parse_agent_result
from init import get_database_status, init_database

logger = logging.getLogger(__name__)


class DeepAgentService:
    """Thin service wrapper around a single Deep Agent instance."""

    backend_name: str = "deepagents"

    def __init__(self, *, project_root: Path) -> None:
        self.project_root = project_root.resolve()
        self._database_resource, self._checkpointer = init_database()
        self._checkpointer_status = get_database_status(self._checkpointer)
        self._agent = create_agent(
            project_root=self.project_root,
            checkpointer=self._checkpointer,
        )

    def run_user_message(
        self,
        *,
        session_id: str,
        trace_id: str,
        user_query: str,
        history: list[dict[str, Any]],
    ) -> DeepAgentTurnResult:
        """Run the deep agent and return a normalized execution result."""
        started_at = perf_counter()
        messages = self._build_messages(user_query=user_query)
        logger.info(
            "Running deep agent: session_id=%s trace_id=%s history_size=%s",
            session_id,
            trace_id,
            len(history),
        )
        result = self._agent.invoke(
            {"messages": messages},
            config={"configurable": {"thread_id": session_id}},
        )
        logger.info("Deep agent finished: trace_id=%s", trace_id)
        return parse_agent_result(
            trace_id=trace_id,
            result=result,
            runtime_debug={
                "latency_ms": int((perf_counter() - started_at) * 1000),
                "checkpointer_backend": self._checkpointer_status.get("backend"),
                "agent_backend": self.backend_name,
            },
        )

    def close(self) -> None:
        """Compatibility hook for API lifespan."""
        if self._database_resource is not None and hasattr(self._database_resource, "close"):
            self._database_resource.close()
        return None

    def runtime_log_status(self) -> dict[str, Any]:
        return {"backend": self.backend_name}

    def checkpointer_status(self) -> dict[str, Any]:
        return dict(self._checkpointer_status)

    def _build_messages(
        self,
        *,
        user_query: str,
    ) -> list[dict[str, str]]:
        return [{"role": "user", "content": user_query.strip()}]
