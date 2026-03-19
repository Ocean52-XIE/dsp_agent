# -*- coding: utf-8 -*-
"""Agent Engine - Agent 执行引擎

参考 v1 WorkflowEngine 设计，独立实现以支持 Agent 模式。

架构：
    Domain Router (规则层) → Agent Loop (LLM 层) → Finalize (结果整理)

核心职责：
1. 编排 Domain Router → Loop → Finalize 流程
2. 管理 LLM 客户端和工具注册中心
3. 协调 Skill 和 MCP 集成
4. 提供统一的执行入口

使用示例：
    engine = AgentEngine(config=EngineConfig())
    response = await engine.arun(
        user_query="查一下 CTR",
        session_id="xxx",
        trace_id="yyy",
    )
"""
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from agent.core.router import (
    DomainRouter,
    DomainRouterConfig,
    DomainRouterResult,
    DomainRouteType,
)
from agent.core.loop import AgentLoop, AgentLoopConfig, AgentLoopResult
from agent.core.finalize import AgentFinalize, FinalizeConfig, AgentResponse
from agent.llm.client import LLMClient
from agent.llm.config import LLMConfig
from agent.state import AgentState, AgentStatus, create_initial_state
from agent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class EngineConfig:
    """Engine 配置

    Attributes:
        max_steps: 最大循环步数
        timeout_seconds: 超时时间 (秒)
        debug_verbose: 是否开启详细调试
        llm_config: LLM 配置
        router_config: Domain Router 配置
        loop_config: Loop 配置
        finalize_config: Finalize 配置
    """
    max_steps: int = 10
    timeout_seconds: int = 120
    debug_verbose: bool = False
    llm_config: LLMConfig | None = None
    router_config: DomainRouterConfig | None = None
    loop_config: AgentLoopConfig | None = None
    finalize_config: FinalizeConfig | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EngineConfig":
        """从字典创建配置"""
        llm_config = None
        if "llm_config" in data:
            llm_config = LLMConfig(**data["llm_config"])

        router_config = None
        if "router_config" in data:
            router_config = DomainRouterConfig.from_dict(data["router_config"])

        loop_config = None
        if "loop_config" in data:
            loop_config = AgentLoopConfig.from_dict(data["loop_config"])

        finalize_config = None
        if "finalize_config" in data:
            finalize_config = FinalizeConfig.from_dict(data["finalize_config"])

        return cls(
            max_steps=int(data.get("max_steps", 10)),
            timeout_seconds=int(data.get("timeout_seconds", 120)),
            debug_verbose=bool(data.get("debug_verbose", False)),
            llm_config=llm_config,
            router_config=router_config,
            loop_config=loop_config,
            finalize_config=finalize_config,
        )


class AgentEngine:
    """Agent 执行引擎

    参考 v1 WorkflowEngine 设计，独立实现以支持 Agent 模式。

    架构流程：
        1. Domain Router (规则层) - 快速规则过滤
        2. Agent Loop (LLM 层) - LLM 自决策 + 工具调用
        3. Finalize (结果整理) - 格式化输出

    使用示例：
        engine = AgentEngine(config=EngineConfig())
        response = await engine.arun(
            user_query="查一下 CTR",
            session_id="xxx",
            trace_id="yyy",
        )
    """

    def __init__(
        self,
        config: EngineConfig | None = None,
        llm_client: LLMClient | None = None,
        tool_registry: ToolRegistry | None = None,
        skill_registry: Any | None = None,
    ):
        """初始化 Agent Engine

        Args:
            config: Engine 配置
            llm_client: LLM 客户端 (可选，不传则使用默认配置)
            tool_registry: 工具注册中心 (可选)
            skill_registry: Skill 注册中心 (可选)
        """
        self.config = config or EngineConfig()

        # 初始化 LLM 客户端
        self._llm_client = llm_client or LLMClient(
            self.config.llm_config or LLMConfig(model="gpt-4o")
        )

        # 初始化工具注册中心
        self._tool_registry = tool_registry or ToolRegistry()

        # 初始化 Skill 注册中心
        self._skill_registry = skill_registry

        # 初始化核心组件
        # 1. Domain Router (规则层，纯规则，无 LLM)
        self._router = DomainRouter(
            config=self.config.router_config or DomainRouterConfig(),
        )

        # 2. Agent Loop (LLM 层，LLM 自决策)
        self._loop = AgentLoop(
            llm_client=self._llm_client,
            tool_registry=self._tool_registry,
            config=self.config.loop_config or AgentLoopConfig(
                max_steps=self.config.max_steps,
                timeout_seconds=self.config.timeout_seconds,
            ),
        )

        # 3. Finalize (结果整理)
        self._finalize = AgentFinalize(
            config=self.config.finalize_config or FinalizeConfig(
                include_debug_info=self.config.debug_verbose,
            ),
        )

        logger.info(
            f"[AgentEngine] 初始化完成, "
            f"max_steps={self.config.max_steps}, "
            f"timeout={self.config.timeout_seconds}s"
        )

    @property
    def backend_name(self) -> str:
        """后端名称"""
        return "agent_engine"

    @property
    def llm_client(self) -> LLMClient:
        """LLM 客户端"""
        return self._llm_client

    @property
    def tool_registry(self) -> ToolRegistry:
        """工具注册中心"""
        return self._tool_registry

    @property
    def skill_registry(self) -> Any | None:
        """Skill 注册中心"""
        return self._skill_registry

    async def arun(
        self,
        user_query: str,
        session_id: str | None = None,
        trace_id: str | None = None,
        history: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AgentResponse:
        """异步执行 Agent

        Args:
            user_query: 用户查询
            session_id: 会话 ID (可选)
            trace_id: 追踪 ID (可选)
            history: 历史对话 (可选)
            **kwargs: 其他参数

        Returns:
            AgentResponse 响应对象
        """
        start_time = time.time()

        # 生成默认 ID
        if not trace_id:
            trace_id = uuid.uuid4().hex[:16]
        if not session_id:
            session_id = trace_id

        # 创建初始状态
        state = create_initial_state(
            trace_id=trace_id,
            session_id=session_id,
            user_query=user_query,
            history=history,
        )

        logger.info(
            f"[AgentEngine] 开始执行, "
            f"trace_id={trace_id}, "
            f"query={user_query[:50]}..."
        )

        try:
            # Phase 1: Domain Router (规则层，同步)
            router_result = self._run_router(state)
            state["status"] = AgentStatus.ROUTING.value

            # 检查是否可以直接响应 (out_of_scope, small_talk)
            if router_result.route in (
                DomainRouteType.OUT_OF_SCOPE,
                DomainRouteType.SMALL_TALK,
            ):
                return self._handle_direct_response(state, router_result, start_time)

            # 检查是否模块分流
            if router_result.route == DomainRouteType.MODULE_ROUTED:
                # TODO: 实现模块分流到专属 workflow
                logger.info(
                    f"[AgentEngine] 模块分流: {router_result.matched_module}"
                )

            # Phase 2: Agent Loop (LLM 层，异步)
            loop_result = await self._run_loop(state, router_result)

            # Phase 3: Finalize
            response = self._run_finalize(state, loop_result, start_time)

            return response

        except Exception as e:
            logger.error(f"[AgentEngine] 执行失败: {e}")
            return self._handle_error(state, str(e), start_time)

    def _run_router(self, state: AgentState) -> DomainRouterResult:
        """执行 Domain Router (规则层，同步)

        Domain Router 是纯规则实现，不做 LLM 调用。

        Args:
            state: Agent 状态

        Returns:
            DomainRouterResult 路由结果
        """
        logger.debug("[AgentEngine] 执行 Domain Router (规则层)...")

        user_query = str(state.get("user_query", "") or "")
        result = self._router.route(user_query)
        self._router.update_state(state, result)

        logger.info(
            f"[AgentEngine] Domain Router 完成: "
            f"route={result.route.value}, "
            f"relevance={result.domain_relevance:.2f}"
        )

        return result

    async def _run_loop(
        self,
        state: AgentState,
        router_result: DomainRouterResult,
    ) -> AgentLoopResult:
        """执行 Agent Loop (LLM 层)

        Agent Loop 中 LLM 自行决策：
        - 意图理解
        - Tool 选择
        - Skill 匹配

        Args:
            state: Agent 状态
            router_result: Domain Router 结果

        Returns:
            AgentLoopResult 循环结果
        """
        logger.debug("[AgentEngine] 执行 Agent Loop (LLM 层)...")

        result = await self._loop.run(state)

        logger.info(
            f"[AgentEngine] Agent Loop 完成: "
            f"success={result.success}, "
            f"steps={result.steps}, "
            f"tool_calls={len(result.tool_calls)}"
        )

        return result

    def _run_finalize(
        self,
        state: AgentState,
        loop_result: AgentLoopResult,
        start_time: float,
    ) -> AgentResponse:
        """执行结果整理

        Args:
            state: Agent 状态
            loop_result: 循环结果
            start_time: 开始时间

        Returns:
            AgentResponse 响应对象
        """
        logger.debug("[AgentEngine] 执行结果整理...")

        latency_ms = int((time.time() - start_time) * 1000)

        response = self._finalize.run(
            state=state,
            answer=loop_result.answer,
            tool_calls=loop_result.tool_calls,
            error=loop_result.error,
            latency_ms=latency_ms,
        )

        # 更新状态
        self._finalize.update_state(state, response)

        return response

    def _handle_direct_response(
        self,
        state: AgentState,
        router_result: DomainRouterResult,
        start_time: float,
    ) -> AgentResponse:
        """处理直接响应 (out_of_scope, small_talk)

        Args:
            state: Agent 状态
            router_result: Domain Router 结果
            start_time: 开始时间

        Returns:
            AgentResponse 响应对象
        """
        latency_ms = int((time.time() - start_time) * 1000)

        # 使用 Domain Router 提供的快速响应
        content = router_result.quick_response or ""

        response = AgentResponse(
            role="assistant",
            content=content,
            kind=router_result.route.value,
            intent=router_result.route.value,
            status="completed",
            trace_id=state.get("trace_id", ""),
            success=True,
            latency_ms=latency_ms,
        )

        logger.info(
            f"[AgentEngine] 直接响应: "
            f"route={router_result.route.value}, "
            f"latency={latency_ms}ms"
        )

        return response

    def _handle_error(
        self,
        state: AgentState,
        error: str,
        start_time: float,
    ) -> AgentResponse:
        """处理错误

        Args:
            state: Agent 状态
            error: 错误信息
            start_time: 开始时间

        Returns:
            AgentResponse 响应对象
        """
        latency_ms = int((time.time() - start_time) * 1000)
        state["status"] = AgentStatus.FAILED.value

        return AgentResponse(
            role="assistant",
            content="抱歉，处理您的请求时出现了问题，请稍后重试。",
            kind="error",
            intent="error",
            status="failed",
            trace_id=state.get("trace_id", ""),
            success=False,
            error=error,
            latency_ms=latency_ms,
        )

    def register_local_tool(self, tool: Any) -> None:
        """注册本地工具

        本地工具包括：
        - 业务工具 (如 web_search, code_retriever 等)
        - SkillManager (作为统一入口管理所有 Skill)

        Args:
            tool: 工具实例
        """
        self._tool_registry.register_local_tool(tool)
        logger.debug(f"[AgentEngine] 注册本地工具: {tool.name}")

    def register_mcp_tool(self, tool: Any) -> None:
        """注册 MCP 工具

        Args:
            tool: MCP 工具实例
        """
        self._tool_registry.register_mcp_tool(tool)
        logger.debug(f"[AgentEngine] 注册 MCP 工具: {tool.name}")

    def get_stats(self) -> dict[str, Any]:
        """获取引擎统计信息

        Returns:
            统计信息字典
        """
        return {
            "backend": self.backend_name,
            "llm_model": self._llm_client.config.model if self._llm_client else None,
            "tools": self._tool_registry.get_stats(),
            "config": {
                "max_steps": self.config.max_steps,
                "timeout_seconds": self.config.timeout_seconds,
                "debug_verbose": self.config.debug_verbose,
            },
        }

    def log_tool_summary(self) -> None:
        """打印工具加载摘要

        在所有工具注册完成后调用，打印当前引擎加载的工具摘要。
        """
        stats = self._tool_registry.get_stats()
        logger.info(
            f"[AgentEngine] 工具加载摘要: "
            f"本地工具({stats['local_tools']})={stats['local_tool_names']}, "
            f"MCP工具({stats['mcp_tools']})={stats['mcp_tool_names']}, "
            f"总计={stats['total_tools']}"
        )
