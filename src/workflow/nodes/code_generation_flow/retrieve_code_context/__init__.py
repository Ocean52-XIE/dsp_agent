# -*- coding: utf-8 -*-
"""
该模块实现工作流节点`retrieve_code_context` 的处理逻辑，负责读取状态并输出增量结果。
"""
from __future__ import annotations

from typing import Any


def run(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """
    执行`retrieve_code_context` 节点主流程，基于输入状态计算并返回状态增量。
    
    参数:
        service: 工作流服务对象，提供检索、路由、日志与配置能力。
        state: 工作流状态字典，包含会话上下文与中间结果。
    
    返回:
        返回类型为 `dict[str, Any]` 的处理结果。
    """
    module_name = state["module_name"]
    context_paths = service.domain_profile.build_code_context_paths(module_name)
    primary_path = context_paths[0] if context_paths else f"services/{service.domain_profile.profile_id}/{module_name}/handler.py"
    test_path = context_paths[1] if len(context_paths) > 1 else f"tests/{service.domain_profile.profile_id}/{module_name}/test_handler.py"

    items = [
        {
            "source_type": "code",
            "title": f"{module_name} target file",
            "path": primary_path,
            "score": 0.91,
            "excerpt": "建议在入口阶段增加参数完整性校验与兜底保护。",
        },
        {
            "source_type": "code",
            "title": f"{module_name} test file",
            "path": test_path,
            "score": 0.84,
            "excerpt": "建议新增关键边界与回归测试覆盖。",
        },
    ]
    return {
        "code_hits": items,
        "citations": items,
        "node_trace": service._trace(state, "retrieve_code_context", f"hits={len(items)}"),
    }
