# -*- coding: utf-8 -*-
"""Skill 系统初始化器

负责初始化 Skill 系统的组件：
- SkillRegistry：技能注册中心
- SkillExecutor：技能执行器

设计原则：
1. 在程序启动时完成所有初始化
2. src/agent 模块无需感知私域配置
3. 通过 agent_loop_node 传递已初始化的组件
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# ============================================================================
# Skill Registry 初始化
# ============================================================================

def init_skill_registry(
    domain_profile: Any,
) -> None:
    """初始化技能注册中心

    从领域目录加载技能定义，注册到全局 SkillRegistry。

    Args:
        domain_profile: 私域配置实例（DomainProfile）

    Returns:
        SkillRegistry 实例，如果加载失败则返回 None
    """
    from agent.skills import get_skill_registry

    # 使用全局单例
    registry = get_skill_registry()

    # 从 domain_profile 获取领域目录
    domain_dir = domain_profile.domain_dir
    if not domain_dir.exists():
        logger.warning(f"[SkillInit] 领域目录不存在: {domain_dir}")
        return None

    try:
        count = registry.load_from_directory(domain_dir)
        logger.info(f"[SkillInit] 加载技能完成: count={count}, domain={domain_profile.profile_id}")

    except Exception as e:
        logger.warning(f"[SkillInit] 加载技能失败: {e}")


# ============================================================================
# Skill Executor 初始化
# ============================================================================

def init_skill_executor() -> None:
    """初始化技能执行器并设置全局单例"""
    from agent.skills import SkillExecutor, set_skill_executor

    try:
        executor = SkillExecutor()
        set_skill_executor(executor)
        logger.info("[SkillInit] SkillExecutor 初始化完成，已设置全局单例")

    except Exception as e:
        logger.warning(f"[SkillInit] SkillExecutor 初始化失败: {e}")
