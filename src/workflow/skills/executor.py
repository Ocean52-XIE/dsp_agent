"""技能执行器（One-Stage 简化版）

统一处理所有 skill 类型的执行，不依赖 LLM client。

支持类型：
- tool_execution: 执行工具，返回 data
- prompt_template: 渲染提示词，返回 prompt

使用示例：
    from src.workflow.skills.executor import SkillExecutor

    executor = SkillExecutor(external_registry)

    if plan.is_skill_call():
        result = executor.execute_plan(plan, query)

        if result.has_data():
            # tool_execution: data 交给 LLM 整理
            response = compose_with_llm(llm, query, result.data)
        elif result.has_prompt():
            # prompt_template: prompt 交给 LLM 生成
            response = llm.invoke(result.prompt)
"""
import importlib.util
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from jinja2 import Template

from src.workflow.skills.base import (
    SkillCallPlan,
    StandardSkill,
    StandardTool,
    ValidationResult,
)

logger = logging.getLogger(__name__)


@dataclass
class ToolCallRecord:
    """工具调用记录"""
    tool_name: str
    arguments: dict[str, Any]
    success: bool
    result: Any = None
    error: str | None = None
    latency_ms: int = 0


@dataclass
class SkillExecutionResult:
    """技能执行结果

    根据技能类型，返回不同的执行素材：
    - tool_execution: data 字段包含工具返回的数据
    - prompt_template: prompt 字段包含渲染后的提示词

    Attributes:
        skill_id: 技能 ID
        skill_type: 技能类型
        success: 是否成功
        data: 工具返回的数据（tool_execution 类型）
        prompt: 渲染后的提示词（prompt_template 类型）
        tool_calls: 工具调用记录列表
        error: 错误信息
        latency_ms: 执行耗时（毫秒）
    """
    skill_id: str
    skill_type: str
    success: bool
    data: dict[str, Any] | None = None      # tool_execution 返回的数据
    prompt: str | None = None                # prompt_template 返回的提示词
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    error: str | None = None
    latency_ms: int = 0

    def has_data(self) -> bool:
        """是否有工具数据"""
        return self.success and self.data is not None

    def has_prompt(self) -> bool:
        """是否有提示词"""
        return self.success and self.prompt is not None

    def to_dict(self) -> dict[str, Any]:
        """转换为字典

        Returns:
            结果字典
        """
        return {
            "skill_id": self.skill_id,
            "skill_type": self.skill_type,
            "success": self.success,
            "data": self.data,
            "prompt": self.prompt,
            "tool_calls": [
                {
                    "tool_name": tc.tool_name,
                    "success": tc.success,
                    "result": tc.result if tc.success else None,
                    "error": tc.error,
                    "latency_ms": tc.latency_ms,
                }
                for tc in self.tool_calls
            ],
            "error": self.error,
            "latency_ms": self.latency_ms,
        }


class SkillExecutor:
    """技能执行器

    统一处理所有 skill 类型的执行。
    不调用 LLM，只返回执行素材（data 或 prompt）。

    使用示例：
        executor = SkillExecutor(external_registry)

        if plan.is_skill_call():
            result = executor.execute_plan(plan, query)

            if result.has_data():
                # tool_execution: data 交给 LLM 整理
                response = compose_with_llm(llm, query, result.data)
            elif result.has_prompt():
                # prompt_template: prompt 交给 LLM 生成
                response = llm.invoke(result.prompt)
    """

    def __init__(self, external_registry: Any = None):
        """初始化技能执行器

        Args:
            external_registry: 外部系统注册中心实例（用于调用外部 API）
        """
        self.external_registry = external_registry
        self._tool_handlers: dict[str, dict[str, Callable]] = {}

        logger.info("[SkillExecutor] 初始化完成")

    def execute_plan(
        self,
        plan: SkillCallPlan,
        query: str = "",
    ) -> SkillExecutionResult:
        """执行技能调用计划

        统一入口，根据 skill_type 返回不同的执行素材。

        Args:
            plan: 技能调用计划（必须包含 skill 引用）
            query: 用户原始查询（用于 prompt 渲染）

        Returns:
            SkillExecutionResult 执行结果
            - tool_execution: data 字段包含工具返回的数据
            - prompt_template: prompt 字段包含渲染后的提示词
        """
        # 1. 检查 plan 状态
        if not plan.is_skill_call():
            return SkillExecutionResult(
                skill_id=plan.skill_name or "unknown",
                skill_type="unknown",
                success=False,
                error="Plan is not a skill call",
            )

        # 再次检查 skill 非空（类型检查器需要）
        skill = plan.skill
        if skill is None:
            return SkillExecutionResult(
                skill_id=plan.skill_name or "unknown",
                skill_type="unknown",
                success=False,
                error="Skill is None",
            )

        start_time = time.time()

        logger.info(
            f"[SkillExecutor] 执行技能: {skill.skill_id}, "
            f"类型={skill.skill_type}, input={plan.input}"
        )

        try:
            # 2. 校验参数
            validation = self._validate_input(skill, plan.input)
            if not validation.ok:
                return SkillExecutionResult(
                    skill_id=skill.skill_id,
                    skill_type=skill.skill_type,
                    success=False,
                    error=validation.message,
                )

            # 3. 根据类型执行
            if skill.is_tool_execution():
                result = self._execute_tool(skill, plan.input)
            elif skill.is_prompt_template():
                result = self._execute_prompt(skill, query, plan.input)
            else:
                # llm_orchestrated 暂不支持
                result = SkillExecutionResult(
                    skill_id=skill.skill_id,
                    skill_type=skill.skill_type,
                    success=False,
                    error=f"Unsupported skill type: {skill.skill_type}",
                )

        except Exception as e:
            logger.error(f"[SkillExecutor] 执行失败: {e}")
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

    def _execute_prompt(
        self,
        skill: StandardSkill,
        query: str,
        input: dict[str, Any],
    ) -> SkillExecutionResult:
        """执行 prompt_template 类型

        渲染提示词模板，返回渲染后的 prompt。

        Args:
            skill: 技能对象
            query: 用户查询
            input: 输入参数

        Returns:
            SkillExecutionResult（prompt 字段包含渲染后的提示词）
        """
        template_content = skill.prompt_template

        if not template_content:
            return SkillExecutionResult(
                skill_id=skill.skill_id,
                skill_type=skill.skill_type,
                success=False,
                error="No prompt template defined",
            )

        try:
            template = Template(template_content)

            # 构建模板上下文，避免参数冲突
            context = {
                "query": query,
                "input": input,
                "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            # 合并 input 中的字段，但 query/input/current_time 优先
            for key, value in input.items():
                if key not in context:
                    context[key] = value

            prompt = template.render(**context)

            logger.info(f"[SkillExecutor] 提示词渲染成功: {skill.skill_id}")

            return SkillExecutionResult(
                skill_id=skill.skill_id,
                skill_type=skill.skill_type,
                success=True,
                prompt=prompt,
            )

        except Exception as e:
            logger.error(f"[SkillExecutor] 模板渲染失败: {e}")
            return SkillExecutionResult(
                skill_id=skill.skill_id,
                skill_type=skill.skill_type,
                success=False,
                error=f"Template render error: {e}",
            )

    def _execute_tool(
        self,
        skill: StandardSkill,
        input: dict[str, Any],
    ) -> SkillExecutionResult:
        """执行 tool_execution 类型

        执行工具调用，返回工具数据。

        Args:
            skill: 技能对象
            input: 输入参数

        Returns:
            SkillExecutionResult（data 字段包含工具返回的数据）
        """
        tool_calls: list[ToolCallRecord] = []

        # 1. 确定要调用的工具
        tool_name = self._resolve_tool_name(skill, input)
        tool = self._get_internal_tool(skill, tool_name)

        if tool is None:
            return SkillExecutionResult(
                skill_id=skill.skill_id,
                skill_type=skill.skill_type,
                success=False,
                error=f"Tool not found: {tool_name}",
                tool_calls=tool_calls,
            )

        # 2. 构建参数
        tool_params = self._build_tool_params(tool, input)

        # 3. 执行
        call_record = self._call_tool(skill, tool, tool_params)
        tool_calls.append(call_record)

        # 4. 返回结果
        if call_record.success:
            logger.info(f"[SkillExecutor] 工具执行成功: {tool_name}")
            return SkillExecutionResult(
                skill_id=skill.skill_id,
                skill_type=skill.skill_type,
                success=True,
                data=call_record.result,
                tool_calls=tool_calls,
            )
        else:
            return SkillExecutionResult(
                skill_id=skill.skill_id,
                skill_type=skill.skill_type,
                success=False,
                error=call_record.error,
                tool_calls=tool_calls,
            )

    def _validate_input(
        self,
        skill: StandardSkill,
        input: dict[str, Any],
    ) -> ValidationResult:
        """校验输入参数

        Args:
            skill: 技能对象
            input: 输入参数

        Returns:
            ValidationResult 校验结果
        """
        schema = skill.param_schema

        if not schema:
            return ValidationResult(ok=True)

        # 检查必填字段
        required = schema.get("required", [])
        missing = [f for f in required if f not in input or input[f] is None]
        if missing:
            return ValidationResult(
                ok=False,
                message=f"缺少必填参数: {', '.join(missing)}",
                missing_fields=missing,
            )

        return ValidationResult(ok=True)

    def _resolve_tool_name(
        self,
        skill: StandardSkill,
        input: dict[str, Any]
    ) -> str:
        """根据 input 确定要调用的工具名称

        Args:
            skill: 技能对象
            input: 输入参数

        Returns:
            工具名称
        """
        tool_mapping = skill.get_tool_mapping()

        # 尝试从 tool_mapping 中匹配
        for key, tool_name in tool_mapping.items():
            # 检查 input 中是否有匹配的参数值
            for param_name, param_value in input.items():
                if str(param_value).lower() == key.lower():
                    return tool_name

        # 返回默认工具
        default_tool = skill.get_default_tool()
        if default_tool:
            return default_tool

        # 如果只有一个工具，使用它
        if len(skill.tools) == 1:
            return skill.tools[0].name

        # 无法确定
        logger.warning(f"[SkillExecutor] 无法确定工具: {skill.skill_id}")
        return ""

    def _get_internal_tool(
        self,
        skill: StandardSkill,
        tool_name: str
    ) -> StandardTool | None:
        """获取技能内部工具

        Args:
            skill: 技能对象
            tool_name: 工具名称

        Returns:
            工具对象，如不存在返回 None
        """
        for tool in skill.tools:
            if tool.name == tool_name:
                return tool
        return None

    def _build_tool_params(
        self,
        tool: StandardTool,
        input: dict[str, Any]
    ) -> dict[str, Any]:
        """构建工具参数

        从 input 中提取工具需要的参数。

        Args:
            tool: 工具对象
            input: 输入参数

        Returns:
            工具参数字典
        """
        params = {}

        for param_name in tool.parameters.keys():
            if param_name in input:
                params[param_name] = input[param_name]

        return params

    def _call_tool(
        self,
        skill: StandardSkill,
        tool: StandardTool,
        arguments: dict[str, Any]
    ) -> ToolCallRecord:
        """执行单个工具

        Args:
            skill: 技能对象
            tool: 工具对象
            arguments: 工具参数

        Returns:
            工具调用记录
        """
        start_time = time.time()

        logger.info(f"[SkillExecutor] 执行工具: {tool.name}, 参数: {arguments}")

        try:
            # 1. 尝试从 handler.py 获取处理器
            handler = self._get_tool_handler(skill, tool.name)

            if handler:
                # 使用自定义处理器
                result = handler(**arguments)
            else:
                # 2. 尝试调用外部系统
                result = self._call_external_system(tool, arguments)

            latency_ms = int((time.time() - start_time) * 1000)

            # 处理返回结果
            if isinstance(result, dict):
                success = result.get("success", True)
                data = result.get("data", result) if success else None
                error = result.get("error") if not success else None
            else:
                success = True
                data = result
                error = None

            return ToolCallRecord(
                tool_name=tool.name,
                arguments=arguments,
                success=success,
                result=data,
                error=error,
                latency_ms=latency_ms,
            )

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.error(f"[SkillExecutor] 工具执行失败: {tool.name}, error={e}")

            return ToolCallRecord(
                tool_name=tool.name,
                arguments=arguments,
                success=False,
                error=str(e),
                latency_ms=latency_ms,
            )

    def _get_tool_handler(
        self,
        skill: StandardSkill,
        tool_name: str
    ) -> Callable | None:
        """获取工具处理器

        从 skill 的 handler.py 中加载工具处理器。

        Args:
            skill: 技能对象
            tool_name: 工具名称

        Returns:
            处理器函数，如不存在返回 None
        """
        # 检查缓存
        if skill.skill_id in self._tool_handlers:
            return self._tool_handlers[skill.skill_id].get(tool_name)

        # 加载 handler.py
        handler_path = skill.get_handler_script_path()
        if handler_path is None:
            return None

        try:
            spec = importlib.util.spec_from_file_location(
                f"skill_{skill.skill_id}_handler",
                handler_path
            )
            if spec is None or spec.loader is None:
                return None

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # 获取 TOOLS 字典
            handlers = getattr(module, "TOOLS", {})
            self._tool_handlers[skill.skill_id] = handlers

            return handlers.get(tool_name)

        except Exception as e:
            logger.error(f"[SkillExecutor] 加载处理器失败: {e}")
            return None

    def _call_external_system(
        self,
        tool: StandardTool,
        arguments: dict[str, Any]
    ) -> Any:
        """调用外部系统

        根据工具的 handler_config 调用外部系统。

        Args:
            tool: 工具对象
            arguments: 工具参数

        Returns:
            调用结果
        """
        handler_config = tool.handler_config

        external_system = handler_config.get("external_system")
        connector_method = handler_config.get("connector_method")

        if not external_system or not connector_method:
            raise ValueError(
                f"Tool {tool.name} has no external_system or connector_method configured"
            )

        if self.external_registry is None:
            raise ValueError("External registry not configured")

        # 获取连接器
        connector = self.external_registry.get_connector(external_system)
        if connector is None:
            raise ValueError(f"External system not found: {external_system}")

        # 调用方法
        method = getattr(connector, connector_method, None)
        if method is None:
            raise ValueError(
                f"Connector {external_system} has no method {connector_method}"
            )

        return method(**arguments)
