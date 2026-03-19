# -*- coding: utf-8 -*-
"""测试 MCP Server 配置加载器"""
import os
import sys
import tempfile
from pathlib import Path

# 确保可以导入 agent 模块
_project_root = Path(__file__).resolve().parents[3]
_src_path = _project_root / "src"
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

import pytest

from agent.mcp.config_loader import (
    MCPServerConfig,
    MCPServerConfigLoader,
)


class TestMCPServerConfig:
    """测试 MCPServerConfig 数据结构"""

    def test_create_default_config(self):
        """测试创建默认配置"""
        config = MCPServerConfig(name="test")

        assert config.name == "test"
        assert config.transport == "stdio"
        assert config.command == ""
        assert config.args == []
        assert config.url == ""
        assert config.enabled is True
        assert config.env == {}
        assert config.timeout == 30

    def test_create_full_config(self):
        """测试创建完整配置"""
        config = MCPServerConfig(
            name="codehub",
            transport="stdio",
            command="python",
            args=["server.py"],
            enabled=True,
            env={"API_KEY": "xxx"},
            timeout=60,
        )

        assert config.name == "codehub"
        assert config.transport == "stdio"
        assert config.command == "python"
        assert config.args == ["server.py"]
        assert config.enabled is True
        assert config.env == {"API_KEY": "xxx"}
        assert config.timeout == 60

    def test_from_dict(self):
        """测试从字典创建配置"""
        data = {
            "transport": "sse",
            "url": "http://localhost:8080/sse",
            "timeout": 45,
        }
        config = MCPServerConfig.from_dict("metrics", data)

        assert config.name == "metrics"
        assert config.transport == "sse"
        assert config.url == "http://localhost:8080/sse"
        assert config.timeout == 45

    def test_to_dict(self):
        """测试转换为字典"""
        config = MCPServerConfig(
            name="test",
            command="python",
            args=["a.py"],
        )
        result = config.to_dict()

        assert result["name"] == "test"
        assert result["command"] == "python"
        assert result["args"] == ["a.py"]

    def test_validate_stdio_missing_command(self):
        """测试 stdio 模式缺少 command"""
        config = MCPServerConfig(name="test", transport="stdio")
        errors = config.validate()

        assert len(errors) == 1
        assert "command" in errors[0]

    def test_validate_sse_missing_url(self):
        """测试 sse 模式缺少 url"""
        config = MCPServerConfig(name="test", transport="sse")
        errors = config.validate()

        assert len(errors) == 1
        assert "url" in errors[0]

    def test_validate_invalid_transport(self):
        """测试无效传输模式"""
        config = MCPServerConfig(name="test", transport="invalid")
        errors = config.validate()

        assert any("Invalid transport" in e for e in errors)

    def test_validate_valid_stdio(self):
        """测试有效的 stdio 配置"""
        config = MCPServerConfig(
            name="test",
            transport="stdio",
            command="python",
        )
        errors = config.validate()

        assert len(errors) == 0

    def test_validate_valid_sse(self):
        """测试有效的 sse 配置"""
        config = MCPServerConfig(
            name="test",
            transport="sse",
            url="http://localhost:8080/sse",
        )
        errors = config.validate()

        assert len(errors) == 0


class TestMCPServerConfigLoader:
    """测试 MCPServerConfigLoader"""

    def test_load_from_file_not_exists(self):
        """测试加载不存在的文件"""
        loader = MCPServerConfigLoader()
        result = loader.load_from_file(Path("/nonexistent/config.yaml"))

        assert result == {}

    def test_load_from_file_basic(self):
        """测试基本配置加载"""
        yaml_content = """
servers:
  codehub:
    transport: stdio
    command: python
    args:
      - server.py
    enabled: true
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(yaml_content)
            f.flush()
            temp_path = f.name

        try:
            loader = MCPServerConfigLoader()
            result = loader.load_from_file(Path(temp_path))

            assert "codehub" in result
            assert result["codehub"].command == "python"
            assert result["codehub"].args == ["server.py"]
        finally:
            os.unlink(temp_path)

    def test_load_from_file_multiple_servers(self):
        """测试加载多个 Server 配置"""
        yaml_content = """
servers:
  codehub:
    transport: stdio
    command: python
    args: ["codehub.py"]

  metrics:
    transport: sse
    url: http://localhost:8080/sse
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(yaml_content)
            f.flush()
            temp_path = f.name

        try:
            loader = MCPServerConfigLoader()
            result = loader.load_from_file(Path(temp_path))

            assert len(result) == 2
            assert "codehub" in result
            assert "metrics" in result
        finally:
            os.unlink(temp_path)

    def test_load_from_file_with_env_vars(self, monkeypatch):
        """测试环境变量替换"""
        monkeypatch.setenv("TEST_API_KEY", "secret-key")

        yaml_content = """
servers:
  test:
    transport: stdio
    command: python
    env:
      API_KEY: "${TEST_API_KEY}"
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(yaml_content)
            f.flush()
            temp_path = f.name

        try:
            loader = MCPServerConfigLoader()
            result = loader.load_from_file(Path(temp_path))

            assert "test" in result
            assert result["test"].env["API_KEY"] == "secret-key"
        finally:
            os.unlink(temp_path)

    def test_load_from_file_with_default_value(self, monkeypatch):
        """测试环境变量默认值"""
        # 不设置环境变量
        monkeypatch.delenv("NONEXISTENT_KEY", raising=False)

        yaml_content = """
servers:
  test:
    transport: stdio
    command: python
    env:
      API_KEY: "${NONEXISTENT_KEY:-default-value}"
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(yaml_content)
            f.flush()
            temp_path = f.name

        try:
            loader = MCPServerConfigLoader()
            result = loader.load_from_file(Path(temp_path))

            assert "test" in result
            assert result["test"].env["API_KEY"] == "default-value"
        finally:
            os.unlink(temp_path)

    def test_load_from_file_skip_invalid(self):
        """测试跳过无效配置"""
        yaml_content = """
servers:
  valid:
    transport: stdio
    command: python

  invalid:
    transport: invalid_transport
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(yaml_content)
            f.flush()
            temp_path = f.name

        try:
            loader = MCPServerConfigLoader()
            result = loader.load_from_file(Path(temp_path))

            # 只有 valid 配置被加载
            assert len(result) == 1
            assert "valid" in result
        finally:
            os.unlink(temp_path)

    def test_get_enabled_configs(self):
        """测试获取启用的配置"""
        yaml_content = """
servers:
  enabled1:
    transport: stdio
    command: python
    enabled: true

  enabled2:
    transport: stdio
    command: python

  disabled:
    transport: stdio
    command: python
    enabled: false
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(yaml_content)
            f.flush()
            temp_path = f.name

        try:
            loader = MCPServerConfigLoader()
            loader.load_from_file(Path(temp_path))
            enabled = loader.get_enabled_configs()

            assert len(enabled) == 2
            assert "enabled1" in enabled
            assert "enabled2" in enabled
            assert "disabled" not in enabled
        finally:
            os.unlink(temp_path)

    def test_load_from_domain(self):
        """测试从领域目录加载"""
        # 创建临时目录结构
        with tempfile.TemporaryDirectory() as tmpdir:
            domain_root = Path(tmpdir)
            mcp_dir = domain_root / "mcp_servers"
            mcp_dir.mkdir()

            yaml_content = """
servers:
  test:
    transport: stdio
    command: python
"""
            config_file = mcp_dir / "servers.yaml"
            config_file.write_text(yaml_content, encoding="utf-8")

            loader = MCPServerConfigLoader()
            result = loader.load_from_domain(domain_root)

            assert "test" in result
