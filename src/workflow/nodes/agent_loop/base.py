# -*- coding: utf-8 -*-
"""Agent Loop 节点基类

复用 agent.core.loop.AgentLoop 实现 LLM + Tool 调用能力：
- Agent 模式 LLM 调用（自动处理 tool calling 循环）
- Skill Tool 集成
- 规则 Fallback 机制
- 启动时初始化

子类需要实现：
- _get_system_prompt(): 获取系统提示词
- _get_user_prompt(): 获取用户提示词
- _build_fallback(): 构建 Fallback 响应
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from workflow.nodes.agent_loop.config import AgentLoopNodeConfig

logger = logging.getLogger(__name__)


class BaseAgentLoopNode(ABC):
    """Agent Loop 节点基类

    复用 agent.core.loop.AgentLoop 实现Agent 循环。

    Usage:
        class KnowledgeAnswerNode(BaseAgentLoopNode):
            def _get_system_prompt(self, state):
                return "你是知识问答助手..."

            def _get_user_prompt(self, state, **kwargs):
                return f"问题：{kwargs['user_query']}"

            def _build_fallback(self, state, **kwargs):
                return "无法回答该问题"

        # 在 WorkflowEngine 中初始化（启动时）
        node = KnowledgeAnswerNode(
            config=config
        )
    """

    def __init__(self, config: AgentLoopNodeConfig) -> None:
        """初始化节点

        所有依赖从全局单例获取，无需外部传递。

        Args:
            config: 节点配置
        """
        self.config = config

        # 从全局单例获取依赖
        from agent.llm.client import get_llm_client
        from agent.skills.registry import get_skill_registry
        from agent.skills.executor import get_skill_executor
        from agent.tools.registry import get_tool_registry

        self._llm_client = get_llm_client()
        self._skill_registry = get_skill_registry()
        self._skill_executor = get_skill_executor()
        self._tool_registry = get_tool_registry()

        logger.info(
            f"[{config.node_name}] AgentLoopNode 初始化完成, "
            f"enable_skill_tool={config.enable_skill_tool}"
        )

    def _create_agent_loop(self, system_prompt: str) -> Any:
        """创建 AgentLoop 实例

        复用 agent.core.loop.AgentLoop。
        AgentLoop 从全局单例获取 LLMClient 和 ToolRegistry。

        Args:
            system_prompt: 系统提示词

        Returns:
            AgentLoop 实例
        """
        from agent.core.loop import AgentLoop, AgentLoopConfig

        # 创建 AgentLoop 配置
        loop_config = AgentLoopConfig(
            max_steps=self.config.max_iterations,
            timeout_seconds=self.config.timeout_seconds,
            system_prompt=system_prompt,
        )

        # AgentLoop 从全局单例获取组件
        return AgentLoop(config=loop_config)

    def run(self, service: Any, state: dict[str, Any]) -> dict[str, Any]:
        """节点执行入口（同步版本）

        Args:
            service: WorkflowService 实例
            state: 工作流状态

        Returns:
            更新后的状态增量
        """
        # 1. 提取状态
        user_query = str(state.get("user_query", ""))
        module_name = str(state.get("module_name", ""))
        module_hint = str(state.get("module_hint", ""))
        related_modules = list(state.get("related_modules", []) or [])
        evidence_hits = self._collect_evidence(state)

        # 2. 尝试 LLM 调用（同步）
        llm_result = self._call_llm_with_agent(
            service=service,
            state=state,
            user_query=user_query,
            module_name=module_name,
            module_hint=module_hint,
            related_modules=related_modules,
            evidence_hits=evidence_hits,
        )

        llm_answer = llm_result.get("answer")
        llm_fallback_reason = llm_result.get("fallback_reason")
        llm_call_status = llm_result.get("call_status", self._default_call_status())

        # 3. 决定最终答案
        if llm_answer:
            final_answer = llm_answer
            generation_mode = self._determine_generation_mode(llm_call_status)
        else:
            # Fallback
            if self.config.enable_fallback:
                final_answer = self._build_fallback(
                    state=state,
                    module_name=module_name,
                    module_hint=module_hint,
                    related_modules=related_modules,
                    evidence_hits=evidence_hits,
                )
                generation_mode = "fallback_rule"
            else:
                final_answer = "抱歉，无法处理您的请求。"
                generation_mode = "no_response"

        # 4. 后处理
        final_answer = self._post_process(final_answer, state)

        # 5. 构建返回结果
        return self._build_result(
            service=service,
            state=state,
            answer=final_answer,
            generation_mode=generation_mode,
            llm_fallback_reason=llm_fallback_reason,
            llm_call_status=llm_call_status,
            evidence_hits=evidence_hits,
        )

    def _call_llm_with_agent(
        self,
        service: Any,
        state: dict[str, Any],
        user_query: str,
        module_name: str,
        module_hint: str,
        related_modules: list[dict[str, Any]],
        evidence_hits: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """使用 AgentLoop 调用 LLM（同步版本）

        复用 agent.core.loop.AgentLoop 执行 Agent 循环。

        Returns:
            {
                "answer": str | None,
                "fallback_reason": str | None,
                "call_status": dict,
            }
        """
        if self._llm_client is None:
            return {
                "answer": None,
                "fallback_reason": "llm_client_not_configured",
                "call_status": self._default_call_status(),
            }

        # 检查证据
        if self.config.require_evidence and len(evidence_hits) == 0:
            return {
                "answer": None,
                "fallback_reason": "no_evidence",
                "call_status": self._default_call_status(),
            }

        # 获取提示词
        system_prompt = self._get_system_prompt(state)
        user_prompt = self._get_user_prompt(
            state=state,
            user_query=user_query,
            module_name=module_name,
            module_hint=module_hint,
            related_modules=related_modules,
            evidence_hits=evidence_hits,
        )

        try:
            # 创建 AgentLoop 实例
            agent_loop = self._create_agent_loop(system_prompt)

            # 构建 Agent 状态（最小化字段）
            agent_state = {
                "trace_id": state.get("trace_id", ""),
                "user_query": user_prompt,  # 使用构建好的用户提示词
                "history": state.get("history", []),
                "tool_whitelist": self.config.tool_whitelist if self.config.tool_whitelist else None,
            }

            # 执行 Agent 循环（同步调用）
            result = agent_loop.run(agent_state)

            # 转换结果
            return {
                "answer": result.answer if result.success else None,
                "fallback_reason": None if result.success else result.error,
                "call_status": {
                    "status": "completed" if result.success else "failed",
                    "invoked": True,
                    "request_sent": True,
                    "attempts": result.steps,
                    "latency_ms": result.latency_ms,
                    "reason": None if result.success else result.error,
                    "tool_call_count": len(result.tool_calls),
                    "is_timeout": result.is_timeout,
                },
            }

        except Exception as e:
            logger.error(f"[{self.config.node_name}] AgentLoop 执行失败: {e}")
            return {
                "answer": None,
                "fallback_reason": f"agent_loop_error: {e}",
                "call_status": self._default_call_status(),
            }

    def _collect_evidence(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        """收集证据

        从 state 中收集 wiki_hits、code_hits 等证据。

        Args:
            state: 工作流状态

        Returns:
            证据列表
        """
        from workflow.common.evidence import collect_evidence_hits
        return collect_evidence_hits(state)

    def _default_call_status(self) -> dict[str, Any]:
        """默认调用状态

        Returns:
            默认的状态字典
        """
        return {
            "status": "not_configured",
            "invoked": False,
            "request_sent": False,
            "attempts": 0,
            "latency_ms": 0,
            "reason": None,
            "model": None,
        }

    def _determine_generation_mode(self, call_status: dict[str, Any]) -> str:
        """根据调用状态确定生成模式

        Args:
            call_status: LLM 调用状态

        Returns:
            生成模式字符串（llm / skill_tool）
        """
        iterations = call_status.get("attempts", 1)
        tool_call_count = call_status.get("tool_call_count", 0)
        if iterations > 1 or tool_call_count > 0:
            return "skill_tool"
        return "llm"

    def _post_process(self, answer: str, state: dict[str, Any]) -> str:
        """后处理

        子类可以覆盖此方法实现自定义后处理。

        Args:
            answer: 原始答案
            state: 工作流状态

        Returns:
            处理后的答案
        """
        # 应用标准化函数
        if self.config.normalize_answer_func:
            return self.config.normalize_answer_func(answer)
        return answer

    def _build_result(
        self,
        service: Any,
        state: dict[str, Any],
        answer: str,
        generation_mode: str,
        llm_fallback_reason: str | None,
        llm_call_status: dict[str, Any],
        evidence_hits: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """构建返回结果

        子类可以覆盖此方法实现自定义结果格式。

        Args:
            service: WorkflowService 实例
            state: 工作流状态
            answer: 最终答案
            generation_mode: 生成模式
            llm_fallback_reason: LLM 降级原因
            llm_call_status: LLM 调用状态
            evidence_hits: 证据列表

        Returns:
            状态增量字典
        """
        # 构建分析信息
        analysis: dict[str, Any] = {
            "summary": f"{self.config.node_name} 已完成",
            "generation_mode": generation_mode,
            "llm_fallback_reason": llm_fallback_reason,
            "llm_call_status": llm_call_status,
            "evidence_count": len(evidence_hits),
        }

        # 添加 LLM 可用性信息
        if self._llm_client:
            config = getattr(self._llm_client, "config", None)
            analysis["llm_model"] = getattr(config, "model", None) if config else None

        # 构建节点追踪
        node_trace = self._build_node_trace(
            service,
            state,
            f"generation={generation_mode}",
        )

        return {
            "response_kind": self.config.response_kind,
            "status": "completed",
            "answer": answer,
            "analysis": analysis,
            "node_trace": node_trace,
        }

    def _build_node_trace(self, service: Any, state: dict[str, Any], detail: str) -> list[dict[str, str]]:
        """构建节点追踪

        由于 node_trace 字段使用了 merge_lists reducer，节点只需返回新增的条目，
        LangGraph 会自动合并。

        Args:
            service: WorkflowService 实例
            state: 工作流状态
            detail: 追踪详情

        Returns:
            只包含新增条目的列表
        """
        trace_method = getattr(service, "_trace", None)
        if trace_method:
            return trace_method(state, self.config.node_name, detail)

        # 降级：只返回新增条目（由 merge_lists reducer 合并）
        return [{"node": self.config.node_name, "summary": detail}]

    # ========== 子类必须实现的抽象方法 ==========

    @abstractmethod
    def _get_system_prompt(self, state: dict[str, Any]) -> str:
        """获取系统提示词

        Args:
            state: 工作流状态

        Returns:
            系统提示词字符串
        """
        pass

    @abstractmethod
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
            用户提示词字符串
        """
        pass

    @abstractmethod
    def _build_fallback(
        self,
        state: dict[str, Any],
        module_name: str,
        module_hint: str,
        related_modules: list[dict[str, Any]],
        evidence_hits: list[dict[str, Any]],
    ) -> str:
        """构建 Fallback 响应

        当 LLM 调用失败或返回无效答案时，使用此方法生成兜底响应。

        Args:
            state: 工作流状态
            module_name: 模块名称
            module_hint: 模块提示
            related_modules: 相关模块列表
            evidence_hits: 证据列表

        Returns:
            Fallback 响应字符串
        """
        pass
