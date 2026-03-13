# -*- coding: utf-8 -*-
"""
该模块实现工作流节点`fix_plan` 的处理逻辑，负责读取状态并输出增量结果。
"""
from __future__ import annotations

from typing import Any


def run(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """
    执行`fix_plan` 节点主流程，基于输入状态计算并返回状态增量。
    
    参数:
        service: 工作流服务对象，提供检索、路由、日志与配置能力。
        state: 工作流状态字典，包含会话上下文与中间结果。
    
    返回:
        返回类型为 `dict[str, Any]` 的处理结果。
    """
    analysis = dict(state.get("analysis") or {})
    transition_type = state.get("transition_type", "start_issue_analysis")

    analysis.setdefault(
        "fix_plan",
        [
            "在入口层增加空值/非法值校验",
            "为状态变更逻辑增加幂等保护",
            "补充与历史案例一致的回归测试",
        ],
    )
    analysis.setdefault(
        "verification_steps",
        [
            "复现原始问题路径并验证错误不再出现",
            "检查关键日志字段、状态转换和重试行为",
        ],
    )
    analysis["need_user_confirmation"] = True
    analysis["task_stage"] = "confirm_code"

    if transition_type == "upgrade_from_qa_to_issue_analysis":
        opening = "这轮输入已从前面的知识问答升级为问题分析。"
    elif transition_type == "continue_issue_analysis":
        opening = "这轮输入延续了当前问题分析任务。"
    elif transition_type == "code_request_without_analysis":
        opening = "你这轮直接提出了代码实现诉求，但当前会话还缺少前置分析，因此我先补齐了问题分析。"
    elif transition_type == "switch_topic":
        opening = "这轮输入被视为新的问题分析主题，因此重新开始定位和分析。"
    else:
        opening = "这轮输入已进入问题分析链路。"

    root_cause = str(analysis.get("root_cause", "") or "").strip()
    fix_plan = list(analysis.get("fix_plan", []) or [])
    verification_steps = list(analysis.get("verification_steps", []) or [])

    fix_plan_text = "\n".join(f"{index}. {item}" for index, item in enumerate(fix_plan[:4], start=1)) or "1. --"
    verification_text = (
        "\n".join(f"{index}. {item}" for index, item in enumerate(verification_steps[:3], start=1)) or "1. --"
    )
    llm_mode = str(analysis.get("issue_analysis_generation_mode", "fallback_rule") or "fallback_rule")
    mode_line = "本轮已结合 LLM 与证据完成分析。" if llm_mode == "llm" else "本轮使用规则策略完成分析。"

    return {
        "response_kind": "issue_analysis",
        "task_stage": "confirm_code",
        "status": "confirm_code",
        "next_action": "confirm_code",
        "answer": (
            f"{opening}"
            f"当前更可能定位在 `{state['module_name']}`。\n"
            f"{mode_line}\n"
            f"根因判断：{root_cause or '--'}\n\n"
            f"修复建议：\n{fix_plan_text}\n\n"
            f"验证步骤：\n{verification_text}\n\n"
            "如有需要，我可以继续生成代码实现建议。"
        ),
        "analysis": analysis,
        "node_trace": service._trace(state, "fix_plan", "next_action=confirm_code"),
    }
