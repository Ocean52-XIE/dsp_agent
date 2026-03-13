# -*- coding: utf-8 -*-
"""
该模块实现工作流节点`root_cause_analysis` 的处理逻辑，负责读取状态并输出增量结果。
"""
from __future__ import annotations

from typing import Any


def run(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """
    执行`root_cause_analysis` 节点主流程，基于输入状态计算并返回状态增量。
    
    参数:
        service: 工作流服务对象，提供检索、路由、日志与配置能力。
        state: 工作流状态字典，包含会话上下文与中间结果。
    
    返回:
        返回类型为 `dict[str, Any]` 的处理结果。
    """
    analysis = dict(state.get("analysis") or {})
    analysis.setdefault("root_cause", "高概率是边界输入或重复回调场景未做保护，导致状态被错误覆盖。")
    analysis.setdefault(
        "risks",
        [
            "可能影响已有重试任务或补偿逻辑",
            "需要确认上游调用方是否依赖旧的异常行为",
        ],
    )
    return {
        "analysis": analysis,
        "node_trace": service._trace(state, "root_cause_analysis", "root_cause=state_protection_gap"),
    }
