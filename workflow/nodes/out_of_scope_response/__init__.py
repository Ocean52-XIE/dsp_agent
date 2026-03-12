from __future__ import annotations

from typing import Any


def run(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """直接产出与当前系统无关的拒答结果。"""
    return {
        "route": "out_of_scope",
        "response_kind": "out_of_scope",
        "task_stage": "out_of_scope",
        "status": "out_of_scope",
        "next_action": "completed",
        "citations": [],
        "analysis": None,
        "answer": (
            "这轮输入在领域判定阶段被识别为与当前系统无关，因此没有进入后续的知识问答、问题分析或代码生成链路。"
            "你可以继续提业务规则、模块设计、接口流程、报错排查或修复方案相关的问题。"
        ),
        "node_trace": service._trace(state, "out_of_scope_response", "route=out_of_scope"),
    }

