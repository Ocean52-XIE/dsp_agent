# -*- coding: utf-8 -*-
"""
该模块实现工作流节点`code_generation` 的处理逻辑，负责读取状态并输出增量结果。
"""
from __future__ import annotations

from typing import Any


def run(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """
    执行`code_generation` 节点主流程，基于输入状态计算并返回状态增量。
    
    参数:
        service: 工作流服务对象，提供检索、路由、日志与配置能力。
        state: 工作流状态字典，包含会话上下文与中间结果。
    
    返回:
        返回类型为 `dict[str, Any]` 的处理结果。
    """
    module_name = state["module_name"]
    transition_type = state.get("transition_type", "resume_code_generation")

    if transition_type == "upgrade_to_code_generation":
        opening = "这一轮输入触发了从分析结果到代码实现的阶段升级。"
    else:
        opening = "当前是从确认节点恢复的代码生成链路。"

    suggested_files = service.domain_profile.build_code_context_paths(module_name)
    if not suggested_files:
        suggested_files = [
            f"services/{service.domain_profile.profile_id}/{module_name}/handler.py",
            f"tests/{service.domain_profile.profile_id}/{module_name}/test_handler.py",
        ]

    return {
        "response_kind": "code_generation",
        "task_stage": "code_generation",
        "status": "completed",
        "next_action": "completed",
        "answer": (
            f"{opening} 已继续生成 `{module_name}` 的实现建议。"
            " 当前输出仍是工作流中的 mock 结果；你可以先审阅补丁摘要、涉及文件和测试建议，再接入真实代码生成器。"
        ),
        "analysis": {
            "summary": "代码实现建议已生成",
            "module": module_name,
            "task_stage": "code_generation",
            "transition_type": transition_type,
            "files": suggested_files,
            "patch_summary": [
                "在入口逻辑补充参数兜底与重复回调保护",
                "在状态流转前增加幂等判断",
                "补充回归测试覆盖异常和重试路径",
            ],
            "test_plan": [
                "新增空值输入的单元测试",
                "新增重复回调幂等测试",
                "验证异常路径下状态不被污染",
            ],
            "snippet": (
                "def process_event(payload):\n"
                "    if payload is None:\n"
                "        raise ValueError('payload is required')\n\n"
                "    if is_duplicate(payload.event_id):\n"
                "        return build_idempotent_response()\n\n"
                "    return apply_state_transition(payload)\n"
            ),
        },
        "node_trace": service._trace(state, "code_generation", f"module={module_name}"),
    }
