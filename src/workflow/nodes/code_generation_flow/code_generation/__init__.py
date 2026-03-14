# -*- coding: utf-8 -*-
"""Code generation node."""
from __future__ import annotations

from typing import Any


def run(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    module_name = str(state.get("module_name", "") or service.domain_profile.default_module)
    has_analysis_context = bool(state.get("analysis") or state.get("last_analysis_result"))

    if has_analysis_context:
        opening = "本轮基于当前输入与已有分析证据，直接进入代码实现建议。"
    else:
        opening = "本轮根据你的输入直接进入代码实现建议。"

    suggested_files = service.domain_profile.build_code_context_paths(module_name)
    if not suggested_files:
        suggested_files = [
            f"services/{service.domain_profile.profile_id}/{module_name}/handler.py",
            f"tests/{service.domain_profile.profile_id}/{module_name}/test_handler.py",
        ]

    return {
        "response_kind": "code_generation",
        "status": "completed",
        "answer": (
            f"{opening} 已生成 `{module_name}` 的实现建议。"
            " 当前输出仍是工作流中的 mock 结果；你可以先审阅补丁摘要、涉及文件和测试建议，再接入真实代码生成器。"
        ),
        "analysis": {
            "summary": "代码实现建议已生成",
            "module": module_name,
            "has_analysis_context": has_analysis_context,
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
