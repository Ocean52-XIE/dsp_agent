# -*- coding: utf-8 -*-
"""私域配置管理模块

提供私域配置的加载、管理和访问功能：
- DomainProfile: 私域配置实例
- ModuleProfile: 模块配置
- RetrievalProfile: 检索配置
- AnsweringProfile: 回答配置
- QueryRewriteProfile: 查询改写配置
- DomainGateProfile: 领域门禁配置

使用示例：
    from domain_profile import get_domain_profile, DomainProfile

    # 获取私域配置（单例）
    profile = get_domain_profile()

    # 访问配置
    print(profile.profile_id)
    print(profile.display_name)
    print(profile.modules)
"""
from __future__ import annotations

from domain_profile.profile import (
    DomainProfile,
    ModuleProfile,
    RetrievalProfile,
    AnsweringProfile,
    QueryRewriteProfile,
    DomainGateProfile,
    EmbeddingProfile,
    RerankerProfile,
    get_domain_profile,
    set_domain_profile,
    load_domain_profile,
    reset_domain_profile_singleton,
)

__all__ = [
    # 核心类
    "DomainProfile",
    "ModuleProfile",
    # 配置类
    "RetrievalProfile",
    "AnsweringProfile",
    "QueryRewriteProfile",
    "DomainGateProfile",
    "EmbeddingProfile",
    "RerankerProfile",
    # 单例函数
    "get_domain_profile",
    "set_domain_profile",
    "load_domain_profile",
    # 重置函数
    "reset_domain_profile_singleton",
]
