# -*- coding: utf-8 -*-
"""
该模块实现工作流节点`out_of_scope_response` 的处理逻辑，负责读取状态并输出增量结果。
"""
from __future__ import annotations

from typing import Any


def run(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """
    执行`out_of_scope_response` 节点主流程，基于输入状态计算并返回状态增量。
    
    参数:
        service: 工作流服务对象，提供检索、路由、日志与配置能力。
        state: 工作流状态字典，包含会话上下文与中间结果。
    
    返回:
        返回类型为 `dict[str, Any]` 的处理结果。
    """
    return {
        "route": "out_of_scope",
        "response_kind": "out_of_scope",
        "task_stage": "out_of_scope",
        "status": "out_of_scope",
        "next_action": "completed",
        "retrieval_queries": [],
        "retrieval_plan": {},
        "wiki_hits": [],
        "case_hits": [],
        "code_hits": [],
        "wiki_retrieval_grade": "disabled",
        "case_retrieval_grade": "disabled",
        "code_retrieval_grade": "disabled",
        "wiki_retrieval_profile": {},
        "case_retrieval_profile": {},
        "code_retrieval_profile": {},
        "evidence_fusion_profile": {},
        "citations": [],
        "analysis": None,
        "answer": (
            "这轮输入在领域判定阶段被识别为与当前系统无关，因此没有进入后续的知识问答、问题分析或代码生成链路。"
            "你可以继续提业务规则、模块设计、接口流程、报错排查或修复方案相关的问题。"
        ),
        "node_trace": service._trace(state, "out_of_scope_response", "route=out_of_scope"),
    }
