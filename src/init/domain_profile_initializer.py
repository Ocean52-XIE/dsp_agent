# -*- coding: utf-8 -*-
"""私域配置初始化器

负责加载私域配置（DomainProfile）：
- 从 profile.json 加载配置
- 创建 DomainProfile 单例

设计原则：
1. 在程序启动时完成配置加载
2. 作为其他初始化器的依赖，必须在其他初始化器之前执行
3. 提供 DomainProfile 单例供其他模块使用
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ============================================================================
# DomainProfile 初始化
# ============================================================================

def init_domain_profile(
    *,
    project_root: Path | None = None,
) -> Any:
    """初始化私域配置

    加载 profile.json 并创建 DomainProfile 实例。
    此函数应该在所有其他初始化器之前调用。

    Args:
        domain_id: 领域 ID（默认 "ad_engine"）
        project_root: 项目根目录

    Returns:
        DomainProfile 实例
    """
    from domain_profile import get_domain_profile

    if project_root is None:
        logger.warning(
            f"[DomainProfileInit] 私域配置加载失败"
        )
        return None

    # 使用现有的加载逻辑
    profile = get_domain_profile(project_root=project_root)

    logger.info(
        f"[DomainProfileInit] 私域配置加载完成: "
        f"profile_id={profile.profile_id}, "
        f"display_name={profile.display_name}, "
        f"modules={len(profile.modules)}"
    )

    return profile

# ============================================================================
# 状态查询
# ============================================================================

def get_domain_profile_status(profile: Any) -> dict[str, Any]:
    """获取私域配置状态

    Args:
        profile: DomainProfile 实例

    Returns:
        状态字典
    """
    if profile is None:
        return {
            "status": "not_loaded",
        }

    return {
        "status": "loaded",
        "profile_id": getattr(profile, "profile_id", None),
        "display_name": getattr(profile, "display_name", None),
        "modules_count": len(getattr(profile, "modules", [])),
        "language": getattr(profile, "language", None),
    }
