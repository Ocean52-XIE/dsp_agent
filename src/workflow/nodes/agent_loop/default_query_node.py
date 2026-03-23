# -*- coding: utf-8 -*-
"""Default Query 节点

处理领域相关但无法明确分类的知识问答/问题分析类查询。
使用 AgentLoop 的完整能力（工具调用、多步推理）处理用户请求。

特点：
- 不强制要求检索证据（require_evidence=False）
- 允许更多迭代次数（max_iterations=5）
- 启用技能工具（enable_skill_tool=True）

路由决策流程中，当规则路由置信度不高且 LLM 兜底返回 other_query 时，
会路由到此节点进行处理。
"""
from __future__ import annotations

from typing import Any

from workflow.nodes.agent_loop.base import BaseAgentLoopNode
from workflow.nodes.agent_loop.config import AgentLoopNodeConfig


class DefaultQueryNode(BaseAgentLoopNode):
    """默认查询节点

    处理领域相关但无法明确分类为知识问答或问题分析的查询。
    这类查询通常需要灵活处理，可能涉及：
    - 综合分析（如"帮我分析一下最近的投放数据"）
    - 对比分析（如"对比一下两种出价策略的优劣"）
    - 建议生成（如"给出价策略一些建议"）
    - 总结归纳（如"总结一下召回模块的要点"）
    """

    # 系统提示词模板
    SYSTEM_PROMPT = """你是广告引擎领域的智能助手。

用户的问题与广告引擎领域相关，但不属于标准的知识问答或问题分析类型。

## 处理原则

1. **灵活应对**：根据问题的性质选择合适的处理方式
2. **善用工具**：可以使用检索工具查询知识，或使用技能工具执行操作
3. **直接回答**：如果已有信息足够，直接给出答案
4. **结构化输出**：对于分析类问题，提供清晰的结构和要点

## 回答要求

- 简洁专业，重点突出
- 如有数据支撑，请明确说明
- 如需假设，请明确标注
- 适当使用列表、分段等格式提升可读性

## 领域范围

- 广告投放策略（出价、pacing、预算分配）
- 召回与排序（候选召回、精排、重排）
- 两率预估（CTR、CVR 预测）
- 流量治理与策略调控
- 效果数据分析与问题排查"""

    def __init__(self) -> None:
        """初始化 DefaultQuery 节点"""
        config = AgentLoopNodeConfig(
            node_name="default_query",
            response_kind="default_query",
            enable_skill_tool=True,
            max_iterations=5,
            timeout_seconds=90,
            require_evidence=False,
            enable_fallback=True,
        )
        super().__init__(config=config)

    def _get_system_prompt(self, state: dict[str, Any]) -> str:
        """获取系统提示词

        Args:
            state: 工作流状态

        Returns:
            系统提示词
        """
        # 可以从 domain_profile 加载自定义提示词
        # 这里使用默认模板
        return self.SYSTEM_PROMPT

    def _get_user_prompt(
        self,
        state: dict[str, Any],
        user_query: str,
        module_name: str,
        module_hint: str,
        related_modules: list[dict[str, Any]],
        evidence_hits: list[dict[str, Any]],
    ) -> str:
        """获取用户提示词

        Args:
            state: 工作流状态
            user_query: 用户问题
            module_name: 模块名称
            module_hint: 模块提示
            related_modules: 相关模块列表
            evidence_hits: 证据列表

        Returns:
            用户提示词
        """
        prompt = f"用户问题：{user_query}"

        # 添加模块上下文
        if module_name and module_hint:
            prompt += f"\n\n相关模块：{module_name}（{module_hint}）"

        # 添加相关模块
        if related_modules:
            related_names = [
                m.get("module_name", m.get("name", ""))
                for m in related_modules[:3]
                if m.get("module_name") or m.get("name")
            ]
            if related_names:
                prompt += f"\n相关上下文：{', '.join(related_names)}"

        # 添加证据（如果有）
        if evidence_hits:
            evidence_summary = []
            for hit in evidence_hits[:3]:
                source = hit.get("source", "unknown")
                title = hit.get("title", hit.get("path", ""))
                if title:
                    evidence_summary.append(f"- [{source}] {title}")
            if evidence_summary:
                prompt += "\n\n相关参考：\n" + "\n".join(evidence_summary)

        prompt += "\n\n请根据问题的性质，提供合适的回答。"
        return prompt

    def _build_fallback(
        self,
        state: dict[str, Any],
        module_name: str,
        module_hint: str,
        related_modules: list[dict[str, Any]],
        evidence_hits: list[dict[str, Any]],
    ) -> str:
        """构建 Fallback 响应

        当 LLM 调用失败或返回无效答案时使用。

        Args:
            state: 工作流状态
            module_name: 模块名称
            module_hint: 模块提示
            related_modules: 相关模块列表
            evidence_hits: 证据列表

        Returns:
            Fallback 响应字符串
        """
        return (
            "抱歉，我暂时无法处理这个请求。\n\n"
            "您可以尝试：\n"
            "1. 换一种方式描述您的问题\n"
            "2. 提供更多上下文信息\n"
            "3. 将问题拆分为更具体的子问题\n\n"
            "如果问题持续存在，请联系技术支持。"
        )
