from __future__ import annotations

from typing import Any


def run(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """记录入口模式，便于前端调试看出当前是普通消息还是恢复执行。"""
    return {
        "node_trace": service._trace(state, "entry_router", f"mode={state.get('mode', 'message')}"),
    }

