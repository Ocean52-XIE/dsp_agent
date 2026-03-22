# -*- coding: utf-8 -*-
"""主初始化器

提供统一的程序初始化入口，协调各子初始化器完成所有组件的初始化。

设计原则：
1. 启动时初始化：所有组件在程序启动时完成初始化，避免延迟初始化
2. 模块化初始化：每类组件有独立的初始化方法
3. 单例访问：初始化后的组件通过全局单例访问，不在此保存结果
4. 配置隔离：src/agent 不直接感知私域配置
5. 详细日志：每个初始化方法打印详细的初始化信息

使用方式：
    # 同步初始化（不包含 MCP，MCP 需要异步上下文）
    from init import initialize

    initialize(
        domain_id="ad_engine",
        project_root=Path("/path/to/project"),
    )

    # 异步初始化（推荐用于 FastAPI 等异步框架）
    from init import initialize_async

    await initialize_async(
        domain_id="ad_engine",
        project_root=Path("/path/to/project"),
    )

    # 在其他模块中通过单例访问组件
    from domain_profile import get_domain_profile
    from agent.llm.client import get_llm_client
    from workflow.nodes.retrieval_flow.retrieve_wiki.wiki_retriever import get_wiki_retriever
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _init_retrievers(domain_profile: Any, project_root: Path) -> None:
    """初始化检索器（调用子初始化器，单例在子初始化器中设置）"""

    try:
        from init.retriever_initializer import (
            init_wiki_retriever,
            init_code_retriever,
            init_case_retriever,
        )

        # 1. Wiki 检索器
        init_wiki_retriever(domain_profile, project_root)

        # 2. Code 检索器
        init_code_retriever(domain_profile, project_root)

        # 3. Case 检索器（可选）
        init_case_retriever(domain_profile, project_root)

    except Exception as e:
        logger.warning(f"  └─ 检索器初始化失败: {e}")


def _register_tools() -> None:
    """注册工具到 ToolRegistry"""

    from agent.tools.registry import get_tool_registry
    from agent.skills import get_skill_registry, get_skill_executor
    from agent.mcp import get_mcp_client

    tool_registry = get_tool_registry()
    tool_registry.clear()

    # 1. 注册 Skill Tool
    skill_registry = get_skill_registry()
    skill_executor = get_skill_executor()

    if skill_registry and skill_executor and len(skill_registry.skills) > 0:
        from agent.skills import create_skill_manager
        skill_tool = create_skill_manager(skill_registry, skill_executor)
        if skill_tool:
            tool_registry.register_local_tool(skill_tool)

    # 2. 注册 MCP 工具
    mcp_client = get_mcp_client()
    if mcp_client:
        from init.mcp_initializer import load_mcp_tools
        mcp_tools = load_mcp_tools()
        if mcp_tools:
            for i, tool in enumerate(mcp_tools):
                tool_registry.register_mcp_tool(tool)


# ============================================================================
# 主初始化函数
# ============================================================================

def initialize(
    project_root: Path | None = None,
    enable_mcp: bool = True,
    enable_skills: bool = True,
    enable_retrievers: bool = True,
) ->None:
    """统一初始化入口

    在程序启动时调用，完成所有组件的初始化。
    初始化后的组件通过全局单例访问。

    Args:
        project_root: 项目根目录（默认自动检测）
        enable_mcp: 是否初始化 MCP
        enable_skills: 是否初始化 Skill 系统
        enable_retrievers: 是否初始化检索器

    Returns:
        初始化状态信息（不包含组件引用，组件通过单例访问）

    示例:
        # 在其他模块中访问组件
        from domain_profile import get_domain_profile
        from agent.llm.client import get_llm_client
        profile = get_domain_profile()
        llm = get_llm_client()
    """
    # 检测项目根目录
    logger.info("initialize.begin")

    # 阶段 1: 加载私域配置（设置 domain_profile 单例）
    from init.domain_profile_initializer import init_domain_profile

    domain_profile = init_domain_profile(project_root=project_root)
    if domain_profile is None:
        raise RuntimeError("私域配置加载失败，无法继续初始化")

    # 阶段 2: 初始化 LLM 客户端
    from init.llm_initializer import init_llm_client

    init_llm_client()

    # 阶段 3: 初始化 Skill 系统
    if enable_skills:
        from init.skill_initializer import init_skill_registry, init_skill_executor
        init_skill_registry(domain_profile)
        init_skill_executor()

    # 阶段 4: 初始化 MCP 系统
    if enable_mcp:
        from init.mcp_initializer import initialize_mcp_system, load_mcp_tools
        initialize_mcp_system(domain_profile)

    # 阶段 5: 初始化检索器（设置 retriever 单例）
    if enable_retrievers:
        _init_retrievers(domain_profile, project_root=project_root)

    # 注册工具
    _register_tools()

    # 输出初始化摘要
    _log_initialization_summary()


def _log_initialization_summary() -> None:
    """输出初始化摘要日志（从单例获取信息）"""
    from domain_profile import get_domain_profile
    from agent.skills import get_skill_registry
    from agent.mcp import get_mcp_client
    from workflow.nodes.retrieval_flow.retrieve_wiki.wiki_retriever import get_wiki_retriever
    from workflow.nodes.retrieval_flow.retrieve_code.code_retriever import get_code_retriever

    domain_profile = get_domain_profile()
    skill_registry = get_skill_registry()
    mcp_client = get_mcp_client()
    wiki_retriever = get_wiki_retriever()
    code_retriever = get_code_retriever()

    skill_count = len(skill_registry.skills) if skill_registry else 0

    logger.info("")
    logger.info("=" * 60)
    logger.info("[Init] 初始化完成摘要")
    logger.info("=" * 60)
    if domain_profile:
        logger.info(f"  ├─ domain: {domain_profile.profile_id}")
    logger.info(f"  ├─ skills: {skill_count} 个")
    logger.info(f"  ├─ mcp: {'启用' if mcp_client else '未启用'}")
    logger.info(f"  ├─ wiki_retriever: {'启用' if wiki_retriever else '未启用'}")
    logger.info(f"  └─ code_retriever: {'启用' if code_retriever else '未启用'}")
    logger.info("=" * 60)


# ============================================================================
# 异步初始化函数（推荐用于 FastAPI 等异步框架）
# ============================================================================

async def initialize_async(
    project_root: Path | None = None,
    enable_mcp: bool = True,
    enable_skills: bool = True,
    enable_retrievers: bool = True,
) -> None:
    """异步统一初始化入口

    在程序启动时调用，完成所有组件的初始化。
    使用异步上下文初始化 MCP 等需要异步的组件。

    Args:
        project_root: 项目根目录（默认自动检测）
        enable_mcp: 是否初始化 MCP
        enable_skills: 是否初始化 Skill 系统
        enable_retrievers: 是否初始化检索器

    示例:
        # 在 FastAPI lifespan 中使用
        from init import initialize_async

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            await initialize_async(enable_mcp=True)
            yield
    """
    # 检测项目根目录
    logger.info("initialize_async.begin")

    # 阶段 1: 加载私域配置（设置 domain_profile 单例）- 同步操作
    from init.domain_profile_initializer import init_domain_profile

    domain_profile = init_domain_profile(project_root=project_root)
    if domain_profile is None:
        raise RuntimeError("私域配置加载失败，无法继续初始化")

    # 阶段 2: 初始化 LLM 客户端 - 同步操作
    from init.llm_initializer import init_llm_client

    init_llm_client()

    # 阶段 3: 初始化 Skill 系统 - 同步操作
    if enable_skills:
        from init.skill_initializer import init_skill_registry, init_skill_executor
        init_skill_registry(domain_profile)
        init_skill_executor()

    # 阶段 4: 初始化 MCP 系统 - 异步操作
    if enable_mcp:
        from init.mcp_initializer import initialize_mcp_system_async
        await initialize_mcp_system_async(domain_profile)

    # 阶段 5: 初始化检索器（设置 retriever 单例）- 同步操作
    if enable_retrievers:
        _init_retrievers(domain_profile, project_root=project_root)

    # 注册工具
    _register_tools()

    # 输出初始化摘要
    _log_initialization_summary()
