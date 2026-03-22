# -*- coding: utf-8 -*-
"""程序初始化模块

提供统一的程序初始化入口，包括：
- 私域配置加载（profile.json、技能、MCP 等）
- Skill 系统初始化
- MCP 系统初始化
- 检索索引初始化
- 数据库初始化（按需）
- LLM 客户端初始化

设计原则：
1. 所有组件在程序启动时完成初始化，避免延迟初始化
2. 初始化后的组件通过全局单例访问（domain_profile, agent.llm.client 等）
3. 每类组件有独立的初始化器，便于管理和定位问题

使用示例：

    # 同步初始化（适用于脚本等场景）
    from init import initialize

    initialize(
        project_root=Path("/path/to/project"),
    )

    # 异步初始化（推荐用于 FastAPI 等异步框架）
    from init import initialize_async

    await initialize_async(
        project_root=Path("/path/to/project"),
    )

    # 在其他模块中通过单例访问组件
    from domain_profile import get_domain_profile
    from agent.llm.client import get_llm_client

    domain_profile = get_domain_profile()
    llm_client = get_llm_client()
"""
from __future__ import annotations

from init.initializer import (
    initialize,
    initialize_async,
)
from init.database_initializer import (
    ensure_database_exists,
    init_database,
    get_database_status,
)

__all__ = [
    # 主入口
    "initialize",
    "initialize_async",
    # 数据库初始化
    "ensure_database_exists",
    "init_database",
    "get_database_status",
]
