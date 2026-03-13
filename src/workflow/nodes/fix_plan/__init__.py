from __future__ import annotations

from typing import Any


def run(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """问题分析第三步：给出修复方案，并停在等待代码确认的阶段。"""
    analysis = dict(state.get("analysis") or {})
    transition_type = state.get("transition_type", "start_issue_analysis")
    analysis.update(
        {
            "fix_plan": [
                "在入口层增加空值/非法值校验",
                "为状态变更逻辑增加幂等保护",
                "补充与历史案例一致的回归测试",
            ],
            "verification_steps": [
                "复现原始问题路径并验证错误不再出现",
                "检查关键日志字段、状态转换和重试行为",
            ],
            "need_user_confirmation": True,
            "task_stage": "confirm_code",
        }
    )

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

    return {
        "response_kind": "issue_analysis",
        "task_stage": "confirm_code",
        "status": "confirm_code",
        "next_action": "confirm_code",
        "answer": (
            f"{opening}"
            f"当前更可能定位在 `{state['module_name']}`。"
            "初步根因、修复方案和验证建议已经整理完成；"
            "如果需要，我可以继续生成代码实现建议。"
        ),
        "analysis": analysis,
        "node_trace": service._trace(state, "fix_plan", "next_action=confirm_code"),
    }

