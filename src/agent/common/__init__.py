# -*- coding: utf-8 -*-
"""通用工具模块

提供通用工具：
- DomainProfile: 领域配置
- logging: 日志工具
- utils: 工具函数
"""
from agent.common.domain_profile import DomainProfile
from agent.common.logging import get_logger

__all__ = [
    "DomainProfile",
    "get_logger",
]
