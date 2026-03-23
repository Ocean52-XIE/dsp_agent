# -*- coding: utf-8 -*-
"""LLM 路由决策模块

当规则路由置信度不高时，使用 LLM 进行兜底路由决策。
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# 有效的路由意图类型
VALID_ROUTING_INTENTS = ("knowledge_qa", "issue_analysis", "default_query", "out_of_scope")


@dataclass
class LLMRoutingResult:
    """LLM 路由结果"""
    intent: str  # "knowledge_qa" | "issue_analysis" | "default_query" | "out_of_scope"
    reason: str
    latency_ms: int
    raw_response: str
    success: bool = True


# 默认的系统提示词
DEFAULT_ROUTING_SYSTEM_PROMPT = """# 意图分类助手

你是一个专业的意图分类助手。你的任务是判断用户问题的意图类型。

## 领域背景

你是广告引擎领域的助手，领域范围包括：
- 广告投放策略（出价、pacing、预算分配）
- 召回与排序（候选召回、精排、重排）
- 两率预估（CTR、CVR 预测）
- 流量治理与策略调控
- 效果数据分析与问题排查

## 可选意图类型

1. **knowledge_qa**: 知识问答
   - 用户询问领域知识、概念解释、原理说明
   - 例如：什么是 CTR 预估？出价策略有哪些？

2. **issue_analysis**: 问题分析
   - 用户遇到具体问题需要排查
   - 包含错误信息、异常现象、性能问题
   - 例如：CTR 预估偏低怎么排查？为什么出价胜率下降？

3. **default_query**: 默认查询
   - 与领域相关但不是标准的知识问答或问题分析
   - 需要灵活处理的通用查询
   - 例如：帮我分析一下最近的投放数据、对比一下两种出价策略的优劣

4. **out_of_scope**: 领域无关
   - 与广告引擎领域完全无关的问题
   - 例如：今天天气怎么样？帮我写一首诗

## 判断原则

1. 优先考虑是否包含明确的错误/异常信号（报错、异常、失败等）
2. 如果是概念性询问，归类为 knowledge_qa
3. 如果需要灵活处理或综合分析，归类为 default_query
4. 明显与领域无关的，归类为 out_of_scope

## 输出格式

请直接返回 JSON 格式（不要包含 markdown 代码块）：
{"intent": "knowledge_qa|issue_analysis|default_query|out_of_scope", "reason": "简短的判断理由"}"""


class LLMRouter:
    """LLM 路由决策器

    当规则路由置信度不高时，调用 LLM 进行路由决策。
    """

    def __init__(
        self,
        llm_client: Any,
        system_prompt: str | None = None,
        timeout_seconds: int = 5,
        max_retries: int = 1,
    ) -> None:
        """初始化 LLM 路由器

        Args:
            llm_client: LLM 客户端实例
            system_prompt: 系统提示词（可选，使用默认值）
            timeout_seconds: 超时时间（秒）
            max_retries: 最大重试次数
        """
        self._llm_client = llm_client
        self._system_prompt = system_prompt or DEFAULT_ROUTING_SYSTEM_PROMPT
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries

    @classmethod
    def create_from_config(
        cls,
        llm_client: Any,
        routing_profile: Any,
    ) -> "LLMRouter":
        """从配置创建 LLM 路由器

        Args:
            llm_client: LLM 客户端
            routing_profile: RoutingProfile 配置

        Returns:
            LLMRouter 实例
        """
        return cls(
            llm_client=llm_client,
            timeout_seconds=getattr(routing_profile, "llm_routing_timeout_seconds", 5),
            max_retries=getattr(routing_profile, "llm_routing_max_retries", 1),
        )

    def route(
        self,
        user_query: str,
        domain_context: str = "",
        relevance: float = 0.0,
    ) -> LLMRoutingResult:
        """执行 LLM 路由决策

        Args:
            user_query: 用户查询
            domain_context: 领域上下文（模块名、相关模块等）
            relevance: 领域相关性分数（用于辅助判断）

        Returns:
            LLMRoutingResult: 路由结果
        """
        start_time = time.time()
        messages = self._build_messages(user_query, domain_context, relevance)

        last_error = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._llm_client.invoke(messages)
                raw_response = response.get("content", "") if isinstance(response, dict) else str(response)

                result = self._parse_response(raw_response)
                result.latency_ms = int((time.time() - start_time) * 1000)
                result.raw_response = raw_response

                logger.info(
                    f"[LLMRouter] 路由决策完成: intent={result.intent}, "
                    f"latency={result.latency_ms}ms, reason={result.reason}"
                )
                return result

            except Exception as e:
                last_error = e
                logger.warning(f"[LLMRouter] 尝试 {attempt + 1} 失败: {e}")

        # 所有重试失败，返回默认值
        latency_ms = int((time.time() - start_time) * 1000)
        logger.error(f"[LLMRouter] 所有重试失败: {last_error}")
        return LLMRoutingResult(
            intent="knowledge_qa",  # 默认路由到知识问答
            reason=f"llm_error: {last_error}",
            latency_ms=latency_ms,
            raw_response="",
            success=False,
        )

    def _build_messages(
        self,
        user_query: str,
        domain_context: str,
        relevance: float,
    ) -> list[dict[str, str]]:
        """构建消息

        Args:
            user_query: 用户查询
            domain_context: 领域上下文
            relevance: 领域相关性

        Returns:
            消息列表
        """
        user_content = f"用户问题：{user_query}"

        if domain_context:
            user_content += f"\n\n上下文信息：{domain_context}"

        if relevance > 0:
            user_content += f"\n\n领域相关性分数：{relevance:.2f}"

        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]

    def _parse_response(self, response: str) -> LLMRoutingResult:
        """解析 LLM 响应

        Args:
            response: LLM 原始响应

        Returns:
            LLMRoutingResult: 解析后的路由结果
        """
        try:
            # 尝试提取 JSON
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                intent = str(data.get("intent", "knowledge_qa")).strip().lower()

                # 验证 intent 有效性
                if intent not in VALID_ROUTING_INTENTS:
                    logger.warning(f"[LLMRouter] 无效的 intent: {intent}, 使用默认值 knowledge_qa")
                    intent = "knowledge_qa"

                return LLMRoutingResult(
                    intent=intent,
                    reason=str(data.get("reason", "")),
                    latency_ms=0,
                    raw_response=response,
                    success=True,
                )
        except json.JSONDecodeError as e:
            logger.warning(f"[LLMRouter] JSON 解析失败: {e}")
        except Exception as e:
            logger.warning(f"[LLMRouter] 解析响应失败: {e}")

        # 解析失败，返回默认值
        return LLMRoutingResult(
            intent="knowledge_qa",
            reason="parse_failed",
            latency_ms=0,
            raw_response=response,
            success=False,
        )


def create_llm_router(
    llm_client: Any,
    system_prompt: str | None = None,
    timeout_seconds: int = 5,
    max_retries: int = 1,
) -> LLMRouter:
    """创建 LLM 路由器的工厂函数

    Args:
        llm_client: LLM 客户端
        system_prompt: 系统提示词
        timeout_seconds: 超时时间
        max_retries: 最大重试次数

    Returns:
        LLMRouter 实例
    """
    return LLMRouter(
        llm_client=llm_client,
        system_prompt=system_prompt,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )
