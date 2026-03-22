# -*- coding: utf-8 -*-
"""Skill 执行器

执行两种类型的技能：
- Prompt Skill: 使用 Jinja2 渲染提示词模板
- Execution Skill: 执行脚本命令，通过 stdin 传递 JSON 参数

使用示例：
    from agent.skills.executor import SkillExecutor
    from agent.skills.base import Skill

    executor = SkillExecutor()

    # 执行 Prompt Skill
    result = executor.execute(skill, {"name": "产品A"}, "帮我写个广告文案")
    if result.success:
        print(result.prompt)

    # 执行 Execution Skill（参数通过 stdin 以 JSON 格式传递给脚本）
    result = executor.execute(skill, {"ad_type": "campaign", "limit": 10}, "查一下广告计划")
    if result.success:
        print(result.data)
"""

import json
import logging
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from jinja2 import Environment, Template

from agent.skills.base import ExecutionConfig, Skill

logger = logging.getLogger(__name__)


@dataclass
class ToolCallRecord:
    """工具调用记录

    记录 Execution Skill 的命令执行详情。

    Attributes:
        command: 执行的命令列表
        arguments: 输入参数
        success: 是否成功
        result: 返回结果
        error: 错误信息
        latency_ms: 耗时（毫秒）
    """
    command: list[str]
    arguments: dict[str, Any]
    success: bool
    result: Any = None
    error: str | None = None
    latency_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "command": self.command,
            "arguments": self.arguments,
            "success": self.success,
            "result": self.result if self.success else None,
            "error": self.error,
            "latency_ms": self.latency_ms,
        }


@dataclass
class SkillExecutionResult:
    """技能执行结果

    根据技能类型，返回不同的执行结果：
    - Prompt Skill: prompt 字段包含渲染后的提示词
    - Execution Skill: data 字段包含命令返回的数据

    Attributes:
        skill_id: 技能 ID
        skill_type: 技能类型
        success: 是否成功
        prompt: 渲染后的提示词（Prompt Skill）
        data: 命令返回的数据（Execution Skill）
        tool_calls: 工具调用记录列表
        error: 错误信息
        latency_ms: 执行耗时（毫秒）
    """
    skill_id: str
    skill_type: str
    success: bool

    # Prompt Skill 返回
    prompt: str | None = None

    # Execution Skill 返回
    data: Any = None
    tool_calls: list[ToolCallRecord] = field(default_factory=list)

    # 错误信息
    error: str | None = None
    latency_ms: int = 0

    def has_prompt(self) -> bool:
        """是否有提示词"""
        return self.success and self.prompt is not None

    def has_data(self) -> bool:
        """是否有数据"""
        return self.success and self.data is not None

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "skill_id": self.skill_id,
            "skill_type": self.skill_type,
            "success": self.success,
            "prompt": self.prompt,
            "data": self.data,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "error": self.error,
            "latency_ms": self.latency_ms,
        }


class SkillExecutor:
    """技能执行器

    统一处理 Prompt 和 Execution 两种类型技能的执行。

    Attributes:
        _jinja_env: Jinja2 环境实例
    """

    def __init__(self):
        """初始化技能执行器"""
        # 配置 Jinja2 环境
        self._jinja_env = Environment(
            variable_start_string="{{",
            variable_end_string="}}",
            block_start_string="{%",
            block_end_string="%}",
            comment_start_string="{#",
            comment_end_string="#}",
            trim_blocks=True,
            lstrip_blocks=True,
        )
        logger.info("[SkillExecutor] 初始化完成")

    def execute(
        self,
        skill: Skill,
        params: dict[str, Any],
        query: str = "",
    ) -> SkillExecutionResult:
        """执行技能

        根据技能类型选择执行方式。

        Args:
            skill: 技能对象
            params: 执行参数
            query: 用户原始查询（用于 Prompt 渲染上下文）

        Returns:
            SkillExecutionResult 执行结果
        """
        start_time = time.time()

        logger.info(
            f"[SkillExecutor] 开始执行: {skill.skill_id}, "
            f"类型={skill.skill_type}, params={list(params.keys())}"
        )

        try:
            if skill.is_prompt():
                result = self._execute_prompt(skill, params, query)
            elif skill.is_execution():
                result = self._execute_command(skill, params)
            else:
                result = SkillExecutionResult(
                    skill_id=skill.skill_id,
                    skill_type=skill.skill_type,
                    success=False,
                    error=f"不支持的技能类型: {skill.skill_type}",
                )

        except Exception as e:
            logger.error(f"[SkillExecutor] 执行异常: {skill.skill_id}, {e}")
            result = SkillExecutionResult(
                skill_id=skill.skill_id,
                skill_type=skill.skill_type,
                success=False,
                error=str(e),
            )

        result.latency_ms = int((time.time() - start_time) * 1000)

        logger.info(
            f"[SkillExecutor] 执行完成: {skill.skill_id}, "
            f"success={result.success}, latency={result.latency_ms}ms"
        )

        return result

    # ========== Prompt Skill 执行 ==========

    def _execute_prompt(
        self,
        skill: Skill,
        params: dict[str, Any],
        query: str,
    ) -> SkillExecutionResult:
        """执行 Prompt Skill：渲染模板

        Args:
            skill: 技能对象
            params: 参数
            query: 用户查询

        Returns:
            SkillExecutionResult
        """
        if not skill.prompt_template:
            return SkillExecutionResult(
                skill_id=skill.skill_id,
                skill_type="prompt",
                success=False,
                error="未定义 prompt_template",
            )

        try:
            # 构建模板上下文
            context = self._build_template_context(skill, params, query)

            # 渲染模板
            template = self._jinja_env.from_string(skill.prompt_template)
            prompt = template.render(**context)

            logger.info(
                f"[SkillExecutor] Prompt 渲染成功: {skill.skill_id}, "
                f"length={len(prompt)}"
            )

            return SkillExecutionResult(
                skill_id=skill.skill_id,
                skill_type="prompt",
                success=True,
                prompt=prompt,
            )

        except Exception as e:
            logger.error(f"[SkillExecutor] 模板渲染失败: {e}")
            return SkillExecutionResult(
                skill_id=skill.skill_id,
                skill_type="prompt",
                success=False,
                error=f"模板渲染失败: {e}",
            )

    def _build_template_context(
        self,
        skill: Skill,
        params: dict[str, Any],
        query: str,
    ) -> dict[str, Any]:
        """构建模板上下文

        Args:
            skill: 技能对象
            params: 参数
            query: 用户查询

        Returns:
            模板上下文字典
        """
        # 基础上下文
        context: dict[str, Any] = {
            "query": query,
            "params": params,
            "skill": skill,
            "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        # 展开参数（直接访问）
        context.update(params)

        # 加载参考文档
        if skill.references:
            context["references"] = {}
            for ref in skill.references:
                # 生成安全的键名
                key = ref.name.replace(".", "_").replace("/", "_").replace("\\", "_")
                try:
                    context["references"][key] = ref.load_content()
                except Exception as e:
                    logger.warning(f"[SkillExecutor] 加载参考文档失败 {ref.name}: {e}")
                    context["references"][key] = f"[加载失败: {e}]"

        return context

    # ========== Execution Skill 执行 ==========

    def _execute_command(
        self,
        skill: Skill,
        params: dict[str, Any],
    ) -> SkillExecutionResult:
        """执行 Execution Skill：通过 stdin 传递 JSON 参数

        脚本从 stdin 读取 JSON 格式的参数，自行处理默认值和验证。

        Args:
            skill: 技能对象
            params: 参数（LLM 传入的参数，可能不完整）

        Returns:
            SkillExecutionResult
        """
        exec_config = skill.execution
        if exec_config is None:
            return SkillExecutionResult(
                skill_id=skill.skill_id,
                skill_type="execution",
                success=False,
                error="未配置 execution",
            )

        # 1. 构建命令（不再渲染模板，直接使用配置的命令）
        try:
            cmd_args = self._build_command(exec_config)
        except Exception as e:
            return SkillExecutionResult(
                skill_id=skill.skill_id,
                skill_type="execution",
                success=False,
                error=f"命令构建失败: {e}",
            )

        # 2. 序列化参数为 JSON（通过 stdin 传递给脚本）
        params_json = json.dumps(params, ensure_ascii=False)

        # 3. 构建环境变量
        env = os.environ.copy()
        for key, value in exec_config.env.items():
            env[key] = self._resolve_env_value(value)

        # 4. 确定工作目录
        #    优先级：exec_config.cwd > skill 源文件所在目录 > 当前工作目录
        if exec_config.cwd:
            cwd = exec_config.cwd
        elif skill.source_path:
            cwd = str(skill.source_path.parent)
        else:
            cwd = None

        # 5. 执行命令，通过 stdin 传递 JSON 参数
        logger.info(
            f"[SkillExecutor] 执行命令: {' '.join(cmd_args)}, "
            f"cwd={cwd or '默认'}, params_json={params_json[:200]}"
        )

        try:
            # 使用 UTF-8 编码，通过 input 参数传递 JSON 到 stdin
            result = subprocess.run(
                cmd_args,
                input=params_json,  # JSON 通过 stdin 传递
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                cwd=cwd,
                env=env,
                timeout=exec_config.timeout,
            )

            if result.returncode != 0:
                # 空值保护：stderr 可能因解码失败而为 None
                error_msg = (result.stderr or "").strip() or f"退出码: {result.returncode}"
                logger.error(f"[SkillExecutor] 命令执行失败: {error_msg}")
                return SkillExecutionResult(
                    skill_id=skill.skill_id,
                    skill_type="execution",
                    success=False,
                    error=error_msg,
                    tool_calls=[ToolCallRecord(
                        command=cmd_args,
                        arguments=params,
                        success=False,
                        error=error_msg,
                    )],
                )

            # 6. 解析输出
            output = self._parse_output(
                result.stdout,
                exec_config.output,
                exec_config.extract,
            )

            logger.info(f"[SkillExecutor] 命令执行成功: {skill.skill_id}")

            return SkillExecutionResult(
                skill_id=skill.skill_id,
                skill_type="execution",
                success=True,
                data=output,
                tool_calls=[ToolCallRecord(
                    command=cmd_args,
                    arguments=params,
                    success=True,
                    result=output,
                )],
            )

        except subprocess.TimeoutExpired:
            logger.error(f"[SkillExecutor] 命令超时: {exec_config.timeout}s")
            return SkillExecutionResult(
                skill_id=skill.skill_id,
                skill_type="execution",
                success=False,
                error=f"命令执行超时 ({exec_config.timeout}s)",
                tool_calls=[ToolCallRecord(
                    command=cmd_args,
                    arguments=params,
                    success=False,
                    error="timeout",
                )],
            )
        except FileNotFoundError as e:
            logger.error(f"[SkillExecutor] 命令不存在: {e}")
            return SkillExecutionResult(
                skill_id=skill.skill_id,
                skill_type="execution",
                success=False,
                error=f"命令不存在: {cmd_args[0] if cmd_args else 'unknown'}",
                tool_calls=[ToolCallRecord(
                    command=cmd_args,
                    arguments=params,
                    success=False,
                    error=str(e),
                )],
            )
        except Exception as e:
            logger.error(f"[SkillExecutor] 命令执行异常: {e}")
            return SkillExecutionResult(
                skill_id=skill.skill_id,
                skill_type="execution",
                success=False,
                error=f"执行异常: {e}",
                tool_calls=[ToolCallRecord(
                    command=cmd_args,
                    arguments=params,
                    success=False,
                    error=str(e),
                )],
            )

    def _build_command(self, config: ExecutionConfig) -> list[str]:
        """构建命令参数列表（不渲染模板）

        JSON+stdin 模式下，命令不再包含参数模板，参数通过 stdin 传递。

        Args:
            config: 执行配置

        Returns:
            命令参数列表
        """
        if isinstance(config.command, list):
            # 数组形式：直接返回
            return list(config.command)
        else:
            # 字符串形式：使用 shlex 分割
            try:
                return shlex.split(config.command)
            except ValueError:
                return config.command.split()

    def _resolve_env_value(self, value: str) -> str:
        """解析环境变量值

        支持 ${ENV_VAR} 格式引用系统环境变量。

        Args:
            value: 环境变量值

        Returns:
            解析后的值
        """
        if "${" in value:
            def replace_env(match: re.Match) -> str:
                var_name = match.group(1)
                return os.environ.get(var_name, "")

            return re.sub(r'\$\{(\w+)\}', replace_env, value)

        return value

    def _parse_output(
        self,
        output: str,
        format: str,
        extract: str | None,
    ) -> Any:
        """解析命令输出

        Args:
            output: 命令输出字符串
            format: 输出格式 (json/text/raw)
            extract: JSON 提取路径

        Returns:
            解析后的数据
        """
        output = output.strip()

        if format == "raw":
            return {"raw": output}

        if format == "text":
            return {"text": output}

        if format == "json":
            try:
                data = json.loads(output)
                if extract:
                    # 支持点号路径提取，如 "data.records"
                    for key in extract.split("."):
                        if isinstance(data, dict) and key in data:
                            data = data[key]
                        elif isinstance(data, list) and key.isdigit():
                            data = data[int(key)]
                        else:
                            break
                return data
            except json.JSONDecodeError as e:
                logger.warning(f"[SkillExecutor] JSON 解析失败: {e}")
                return {"text": output, "parse_error": str(e)}

        # 默认返回文本
        return {"text": output}


# ============================================================================
# 全局单例
# ============================================================================

_skill_executor: SkillExecutor | None = None


def get_skill_executor() -> SkillExecutor:
    """获取全局技能执行器单例"""
    global _skill_executor
    if _skill_executor is None:
        _skill_executor = SkillExecutor()
    return _skill_executor


def set_skill_executor(executor: SkillExecutor) -> None:
    """设置全局技能执行器（启动时初始化使用）

    Args:
        executor: 技能执行器实例
    """
    global _skill_executor
    _skill_executor = executor


def reset_skill_executor() -> None:
    """重置全局技能执行器（用于测试）"""
    global _skill_executor
    _skill_executor = None
