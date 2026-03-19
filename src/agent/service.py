# -*- coding: utf-8 -*-
"""Agent Service - Agent 服务入口

参考 v1 WorkflowService 设计，独立实现以支持 Agent 模式。

核心职责：
1. 作为 Agent 模块对外统一入口
2. 管理引擎生命周期和依赖
3. 加载领域配置、Skill 和 MCP 工具
4. 提供便捷的调用接口

初始化流程：
1. 创建 AgentEngine 实例
2. 加载领域技能 (domain_root/skills/) 并注册 SkillManager 工具
3. 连接 MCP Server 并注册 MCP 工具

工具架构：
- 本地工具 (Local Tools): SkillManager (统一管理所有 Skill)
- MCP 工具 (MCP Tools): 从 MCP Server 动态获取

使用示例：
    service = AgentService.from_env()
    await service.ainitialize()  # 推荐在服务启动时调用
    response = await service.arun(
        user_query="查一下 CTR",
        session_id="xxx",
    )
"""
import json
import logging
import os
from pathlib import Path
from typing import Any

from agent.engine import AgentEngine, EngineConfig
from agent.core.router import DomainRouterConfig
from agent.core.loop import AgentLoopConfig
from agent.core.finalize import FinalizeConfig
from agent.llm.client import LLMClient
from agent.llm.config import LLMConfig
from agent.state import AgentState
from agent.tools.registry import ToolRegistry
from agent.core.finalize import AgentResponse

logger = logging.getLogger(__name__)


def _load_domain_profile_dict(domain_root: Path | None) -> dict[str, Any] | None:
    """加载领域配置原始字典

    Args:
        domain_root: 领域配置根目录

    Returns:
        profile.json 的内容字典，加载失败返回 None
    """
    if not domain_root:
        return None

    profile_path = domain_root / "profile.json"
    if not profile_path.exists():
        logger.warning(f"[AgentService] profile.json 不存在: {profile_path}")
        return None

    try:
        with open(profile_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"[AgentService] 加载 profile.json 失败: {e}")
        return None


class AgentService:
    """Agent 服务入口

    参考 v1 WorkflowService 设计，独立实现以支持 Agent 模式。
    作为 Agent 模块对外统一入口。

    使用示例：
        service = AgentService.from_env()
        response = await service.arun(
            user_query="查一下 CTR",
            session_id="xxx",
        )
    """

    def __init__(
        self,
        engine: AgentEngine | None = None,
        config: EngineConfig | None = None,
        domain_root: Path | None = None,
        debug_verbose: bool = False,
    ):
        """初始化 Agent Service

        Args:
            engine: Agent Engine (可选，不传则自动创建)
            config: Engine 配置 (可选)
            domain_root: 领域配置目录 (可选，用于加载 MCP 配置)
            debug_verbose: 是否开启详细调试
        """
        self._engine = engine
        self._config = config or EngineConfig()
        self._domain_root = domain_root
        self._debug_verbose = debug_verbose
        self._initialized = False

        # MCP 客户端 (延迟初始化)
        self._mcp_client = None

        # Skill 注册中心 (延迟初始化)
        self._skill_registry = None

        # 延迟初始化
        logger.info(f"[AgentService] 创建实例 (延迟初始化), domain_root={domain_root}")

    # =========================================================================
    # 异步初始化方法 (推荐在服务启动时调用)
    # =========================================================================

    async def ainitialize(self) -> None:
        """异步初始化 - 加载 MCP 工具

        在服务启动时调用，完成 MCP 连接和工具注册。
        这是推荐的初始化方式，避免运行时竞争。

        使用示例：
            service = AgentService.from_env()
            await service.ainitialize()  # 启动时调用
        """
        if self._initialized:
            logger.debug("[AgentService] 已初始化，跳过")
            return

        logger.info("[AgentService] 开始异步初始化...")

        # 1. 确保 Engine 初始化
        if self._engine is None:
            self._engine = AgentEngine(config=self._config)

        # 2. 加载领域技能并初始化
        if self._domain_root:
            self._initialize_skills()

        # 3. 加载 MCP 配置并初始化
        if self._domain_root:
            await self._initialize_mcp()

        self._initialized = True
        logger.info("[AgentService] 异步初始化完成")

        # 打印工具加载摘要
        self.log_tool_summary()

    async def _initialize_mcp(self) -> None:
        """初始化 MCP 客户端并注册工具

        从 domain_root/mcp_servers/servers.yaml 加载配置，
        连接 MCP Server 并将工具注册到 ToolRegistry。
        """
        from agent.mcp import MCPClient

        try:
            logger.info(f"[AgentService] 初始化 MCP, domain_root={self._domain_root}")

            # 使用 MCPClient.from_domain() 简化初始化
            self._mcp_client = MCPClient.from_domain(self._domain_root)
            await self._mcp_client.initialize()

            # 注册 MCP 工具到 ToolRegistry
            adapters = self._mcp_client.get_tool_adapters()
            for adapter in adapters:
                self._engine.register_mcp_tool(adapter)

            logger.info(f"[AgentService] MCP 初始化完成, 工具数={len(adapters)}")

        except Exception as e:
            # MCP 初始化失败不应阻止服务启动
            logger.warning(f"[AgentService] MCP 初始化失败: {e}, 服务将以降级模式运行")
            self._mcp_client = None

    def _initialize_skills(self) -> None:
        """初始化 Skill 系统并注册 SkillManager 工具

        从 domain_root/skills/ 目录加载所有技能，
        创建 SkillManager 工具并注册到 ToolRegistry。

        设计说明：
        - SkillManager 作为一个统一的本地工具，内部管理所有 Skill 的调用
        - LLM 通过调用 skill_manager 来执行技能，参数为 skill_name 和 skill_args
        - 这种设计的优势：
          * 工具列表稳定，不因 Skill 数量变化而膨胀
          * 动态扩展：可以热加载新 Skill 而不需要重新注册工具
          * 统一入口：便于添加日志、监控、缓存等横切关注点
        """
        from agent.skills import SkillRegistry, SkillExecutor, create_skill_manager

        try:
            logger.info(f"[AgentService] 初始化 Skill 系统, domain_root={self._domain_root}")

            # 1. 创建 Skill 注册中心并加载技能
            self._skill_registry = SkillRegistry()
            skill_count = self._skill_registry.load_from_directory(self._domain_root)

            if skill_count == 0:
                logger.warning("[AgentService] 未加载任何技能，skill_manager 工具将不可用")
                self._skill_registry = None
                return

            # 2. 创建 SkillManager 工具
            executor = SkillExecutor()
            skill_manager = create_skill_manager(
                registry=self._skill_registry,
                executor=executor,
            )

            # 3. 注册到 ToolRegistry 作为本地工具
            self._engine.register_local_tool(skill_manager)

            # 记录加载的技能列表
            skill_ids = list(self._skill_registry.skills.keys())
            logger.info(
                f"[AgentService] Skill 初始化完成, "
                f"技能数={skill_count}, 技能列表={skill_ids}"
            )

        except Exception as e:
            # Skill 初始化失败不应阻止服务启动
            logger.warning(f"[AgentService] Skill 初始化失败: {e}, 服务将以降级模式运行")
            self._skill_registry = None

    async def ashutdown(self) -> None:
        """异步关闭 - 释放 MCP 资源

        在服务关闭时调用，确保 MCP 连接正确释放。
        """
        logger.info("[AgentService] 开始关闭...")

        if self._mcp_client:
            try:
                await self._mcp_client.shutdown()
                logger.info("[AgentService] MCP 连接已关闭")
            except Exception as e:
                logger.warning(f"[AgentService] MCP 关闭失败: {e}")
            finally:
                self._mcp_client = None

        self._initialized = False
        logger.info("[AgentService] 服务已关闭")

    # =========================================================================
    # 同步初始化方法 (向后兼容)
    # =========================================================================

    def _ensure_initialized(self) -> None:
        """确保服务已初始化"""
        if self._initialized:
            return

        if self._engine is None:
            self._engine = AgentEngine(config=self._config)

        self._initialized = True
        logger.info("[AgentService] 初始化完成")

        # 打印工具加载摘要
        self.log_tool_summary()

    def log_tool_summary(self) -> None:
        """打印工具加载摘要

        在服务初始化完成后调用，打印当前加载的工具摘要。
        """
        if self._engine:
            self._engine.log_tool_summary()

    @property
    def engine(self) -> AgentEngine:
        """获取 Agent Engine"""
        self._ensure_initialized()
        return self._engine

    @property
    def backend_name(self) -> str:
        """后端名称"""
        return "agent_service"

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
        self._ensure_initialized()
        return await self._engine.arun(
            user_query=user_query,
            session_id=session_id,
            trace_id=trace_id,
            history=history,
            **kwargs,
        )

    def run(
        self,
        user_query: str,
        session_id: str | None = None,
        trace_id: str | None = None,
        history: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AgentResponse:
        """同步执行 Agent

        注意：这是同步包装器，内部使用 asyncio 运行异步代码。

        Args:
            user_query: 用户查询
            session_id: 会话 ID (可选)
            trace_id: 追踪 ID (可选)
            history: 历史对话 (可选)
            **kwargs: 其他参数

        Returns:
            AgentResponse 响应对象
        """
        import asyncio
        return asyncio.run(self.arun(
            user_query=user_query,
            session_id=session_id,
            trace_id=trace_id,
            history=history,
            **kwargs,
        ))

    def register_local_tool(self, tool: Any) -> None:
        """注册本地工具

        本地工具包括：
        - 业务工具 (如 web_search, code_retriever 等)
        - SkillManager (作为统一入口管理所有 Skill)

        Args:
            tool: 工具实例
        """
        self._ensure_initialized()
        self._engine.register_local_tool(tool)

    def register_mcp_tool(self, tool: Any) -> None:
        """注册 MCP 工具

        Args:
            tool: MCP 工具实例
        """
        self._ensure_initialized()
        self._engine.register_mcp_tool(tool)

    def get_stats(self) -> dict[str, Any]:
        """获取服务统计信息

        Returns:
            统计信息字典
        """
        self._ensure_initialized()
        stats = {
            "backend": self.backend_name,
            "initialized": self._initialized,
            "debug_verbose": self._debug_verbose,
            "engine": self._engine.get_stats() if self._engine else None,
        }

        # 添加 Skill 统计信息
        if self._skill_registry:
            stats["skills"] = self._skill_registry.get_stats()

        return stats

    @classmethod
    def from_env(cls, prefix: str = "AGENT") -> "AgentService":
        """从环境变量创建服务

        环境变量：
        - AGENT_LLM_MODEL: LLM 模型名称
        - AGENT_LLM_BASE_URL: LLM API 基础 URL
        - AGENT_LLM_API_KEY: LLM API 密钥
        - AGENT_MAX_STEPS: 最大循环步数
        - AGENT_TIMEOUT_SECONDS: 超时时间
        - AGENT_DEBUG_VERBOSE: 是否开启详细调试
        - AGENT_DOMAIN_ROOT: 领域配置目录路径
        - AGENT_DOMAIN_ID: 领域 ID (默认 ad_engine)

        Args:
            prefix: 环境变量前缀

        Returns:
            AgentService 实例
        """
        # 读取 LLM 配置
        llm_config = LLMConfig.from_env(f"{prefix}_LLM")

        # 读取 Engine 配置
        max_steps = int(os.getenv(f"{prefix}_MAX_STEPS", "10"))
        timeout_seconds = int(os.getenv(f"{prefix}_TIMEOUT_SECONDS", "120"))
        debug_verbose = os.getenv(f"{prefix}_DEBUG_VERBOSE", "").lower() in ("true", "1", "yes")

        # 读取领域配置目录
        domain_root_env = os.getenv(f"{prefix}_DOMAIN_ROOT", "")
        domain_id = os.getenv(f"{prefix}_DOMAIN_ID", "ad_engine")

        # 确定领域配置根目录
        if domain_root_env:
            domain_root = Path(domain_root_env)
        else:
            # 默认使用项目根目录下的 domain/{domain_id}
            project_root = Path(__file__).resolve().parents[2]
            domain_root = project_root / "domain" / domain_id

        # 加载领域配置 (用于 Domain Router 配置)
        profile_dict = _load_domain_profile_dict(domain_root)
        router_config = None
        if profile_dict:
            router_config = DomainRouterConfig.from_profile(profile_dict)
            logger.info(
                f"[AgentService] 加载领域配置: domain_gate.enabled={router_config.enabled}"
            )

        # 构建配置
        engine_config = EngineConfig(
            max_steps=max_steps,
            timeout_seconds=timeout_seconds,
            debug_verbose=debug_verbose,
            llm_config=llm_config,
            router_config=router_config,
            finalize_config=FinalizeConfig(
                include_debug_info=debug_verbose,
            ),
        )

        logger.info(
            f"[AgentService] 从环境变量创建, "
            f"model={llm_config.model}, "
            f"max_steps={max_steps}, "
            f"timeout={timeout_seconds}s, "
            f"domain_root={domain_root}"
        )

        return cls(
            config=engine_config,
            domain_root=domain_root,
            debug_verbose=debug_verbose,
        )

    @classmethod
    def from_config(cls, config_path: str | Path) -> "AgentService":
        """从配置文件创建服务

        Args:
            config_path: 配置文件路径

        Returns:
            AgentService 实例
        """
        import json

        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        engine_config = EngineConfig.from_dict(config_data)

        return cls(
            config=engine_config,
            debug_verbose=config_data.get("debug_verbose", False),
        )


# 全局单例
_agent_service: AgentService | None = None


async def ainit_agent_service() -> AgentService:
    """初始化全局 Agent Service 单例（异步）

    在服务启动时调用，完成 MCP 连接和工具注册。
    这是推荐的初始化方式。

    使用示例：
        # 在 FastAPI lifespan 中
        await ainit_agent_service()

    Returns:
        AgentService 实例
    """
    global _agent_service
    if _agent_service is None:
        _agent_service = AgentService.from_env()
        await _agent_service.ainitialize()
    return _agent_service


def get_agent_service() -> AgentService:
    """获取已初始化的 Agent Service 单例

    注意：此函数假设 ainit_agent_service() 已经被调用。
    如果未初始化，将抛出 RuntimeError。

    Returns:
        AgentService 实例

    Raises:
        RuntimeError: 如果服务未初始化
    """
    global _agent_service
    if _agent_service is None:
        raise RuntimeError(
            "AgentService 未初始化，请先调用 await ainit_agent_service()"
        )
    return _agent_service
