# -*- coding: utf-8 -*-
"""MCP Server 配置加载器

负责从 YAML 文件加载 MCP Server 配置，支持：
1. 从 domain 目录加载配置
2. 环境变量替换
3. 配置验证
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# 环境变量替换正则
_ENV_VAR_PATTERN = re.compile(r'\$\{([^}]+)\}')


def _substitute_env_vars(
    value: str,
    extra_vars: dict[str, str] | None = None,
) -> str:
    """替换字符串中的环境变量

    支持 ${VAR_NAME} 和 ${VAR_NAME:-default} 格式

    Args:
        value: 包含环境变量的字符串
        extra_vars: 额外的变量替换映射 (优先于环境变量)

    Returns:
        替换后的字符串
    """
    extra_vars = extra_vars or {}

    def replace(match):
        var_expr = match.group(1)
        # 支持默认值语法: ${VAR:-default}
        if ':-' in var_expr:
            var_name, default = var_expr.split(':-', 1)
            var_name = var_name.strip()
        else:
            var_name = var_expr
            default = ''

        # 优先使用额外变量
        if var_name in extra_vars:
            return extra_vars[var_name]

        return os.getenv(var_name, default)

    return _ENV_VAR_PATTERN.sub(replace, value)


def _process_config_values(
    obj: Any,
    extra_vars: dict[str, str] | None = None,
) -> Any:
    """递归处理配置值，替换环境变量

    Args:
        obj: 配置对象
        extra_vars: 额外的变量替换映射 (如 DOMAIN_ROOT, PROJECT_ROOT)

    Returns:
        处理后的配置对象
    """
    if isinstance(obj, str):
        return _substitute_env_vars(obj, extra_vars)
    elif isinstance(obj, dict):
        return {k: _process_config_values(v, extra_vars) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_process_config_values(item, extra_vars) for item in obj]
    return obj


@dataclass
class MCPServerConfig:
    """单个 MCP Server 配置

    Attributes:
        name: Server 名称
        transport: 传输模式 (stdio | sse | websocket)
        command: Stdio 模式下的命令
        args: Stdio 模式下的参数
        url: HTTP/SSE 模式下的 URL
        enabled: 是否启用
        env: 环境变量
        timeout: 超时时间 (秒)
    """
    name: str
    transport: str = "stdio"
    command: str = ""
    args: list[str] = field(default_factory=list)
    url: str = ""
    enabled: bool = True
    env: dict[str, str] = field(default_factory=dict)
    timeout: int = 30

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> "MCPServerConfig":
        """从字典创建配置

        Args:
            name: Server 名称
            data: 配置字典

        Returns:
            MCPServerConfig 实例
        """
        return cls(
            name=name,
            transport=data.get("transport", "stdio"),
            command=data.get("command", ""),
            args=data.get("args", []),
            url=data.get("url", ""),
            enabled=data.get("enabled", True),
            env=data.get("env", {}),
            timeout=data.get("timeout", 30),
        )

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "transport": self.transport,
            "command": self.command,
            "args": self.args,
            "url": self.url,
            "enabled": self.enabled,
            "env": self.env,
            "timeout": self.timeout,
        }

    def validate(self) -> list[str]:
        """验证配置

        Returns:
            错误消息列表，空列表表示验证通过
        """
        errors = []

        if self.transport not in ("stdio", "sse", "websocket"):
            errors.append(f"Invalid transport: {self.transport}")

        if self.transport == "stdio":
            if not self.command:
                errors.append("stdio transport requires 'command'")
        elif self.transport in ("sse", "websocket"):
            if not self.url:
                errors.append(f"{self.transport} transport requires 'url'")

        return errors


class MCPServerConfigLoader:
    """MCP Server 配置加载器

    从 YAML 文件加载 MCP Server 配置。

    使用示例：
        loader = MCPServerConfigLoader()
        config = loader.load_from_file(Path("mcp_servers.yaml"))
        config = loader.load_from_domain(Path("domain/ad_engine"))
    """

    def __init__(self):
        """初始化配置加载器"""
        self._configs: dict[str, MCPServerConfig] = {}

    def load_from_file(
        self,
        config_path: Path,
        extra_vars: dict[str, str] | None = None,
    ) -> dict[str, MCPServerConfig]:
        """从 YAML 文件加载配置

        Args:
            config_path: 配置文件路径
            extra_vars: 额外的变量替换映射 (如 DOMAIN_ROOT, PROJECT_ROOT)

        Returns:
            Server 名称到配置的映射
        """
        if not config_path.exists():
            logger.warning(f"[MCPServerConfigLoader] 配置文件不存在: {config_path}")
            return {}

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                raw_data = yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"[MCPServerConfigLoader] 加载配置文件失败: {e}")
            return {}

        # 处理环境变量和额外变量
        data = _process_config_values(raw_data, extra_vars)

        # 解析 servers 配置
        servers = data.get("servers", {})
        self._configs.clear()

        for name, server_data in servers.items():
            config = MCPServerConfig.from_dict(name, server_data)

            # 验证配置
            errors = config.validate()
            if errors:
                logger.warning(
                    f"[MCPServerConfigLoader] Server '{name}' 配置验证失败: {errors}"
                )
                continue

            self._configs[name] = config
            logger.debug(f"[MCPServerConfigLoader] 加载 Server 配置: {name}")

        logger.debug(
            f"[MCPServerConfigLoader] 加载完成, "
            f"servers={len(self._configs)}, "
            f"enabled={sum(1 for c in self._configs.values() if c.enabled)}"
        )

        return self._configs.copy()

    def load_from_domain(self, domain_root: Path) -> dict[str, MCPServerConfig]:
        """从领域目录加载配置

        查找 domain/<domain_id>/mcp_servers/servers.yaml

        Args:
            domain_root: 领域目录根路径

        Returns:
            Server 名称到配置的映射
        """
        config_path = domain_root / "mcp_servers" / "servers.yaml"

        # 设置路径变量用于配置替换
        # domain_root 格式: project_root/domain/<domain_id>
        domain_root_str = str(domain_root.resolve())
        project_root_str = str(domain_root.resolve().parent.parent)

        extra_vars = {
            "DOMAIN_ROOT": domain_root_str,
            "PROJECT_ROOT": project_root_str,
        }

        return self.load_from_file(config_path, extra_vars)

    def get_config(self, name: str) -> MCPServerConfig | None:
        """获取指定 Server 的配置

        Args:
            name: Server 名称

        Returns:
            配置对象，不存在返回 None
        """
        return self._configs.get(name)

    def get_all_configs(self) -> dict[str, MCPServerConfig]:
        """获取所有 Server 配置

        Returns:
            Server 名称到配置的映射
        """
        return self._configs.copy()

    def get_enabled_configs(self) -> dict[str, MCPServerConfig]:
        """获取所有启用的 Server 配置

        Returns:
            Server 名称到配置的映射
        """
        return {
            name: config
            for name, config in self._configs.items()
            if config.enabled
        }
