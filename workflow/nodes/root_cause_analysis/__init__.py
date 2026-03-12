from __future__ import annotations

from typing import Any


def run(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """问题分析第二步：补充根因、风险和上下文。"""
    analysis = dict(state.get("analysis") or {})
    analysis.update(
        {
            "root_cause": "高概率是边界输入或重复回调场景未做保护，导致状态被错误覆盖。",
            "risks": [
                "可能影响已有重试任务或补偿逻辑",
                "需要确认上游调用方是否依赖旧的异常行为",
            ],
        }
    )
    return {
        "analysis": analysis,
        "node_trace": service._trace(state, "root_cause_analysis", "root_cause=state_protection_gap"),
    }

