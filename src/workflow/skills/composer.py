"""Response Composer - 组装技能执行结果为 LLM 输入内容

职责：
- 将 SkillExecutionResult 转换为给 LLM 的内容
- 不调用 LLM，只做内容组装
- 统一成功和失败的格式

使用示例：
    composer = ResponseComposer()

    result = executor.execute_plan(plan, query)

    # 获取给 LLM 的内容
    llm_content = composer.compose(query, result)

    # 调用方决定如何使用 LLM
    if llm_content:
        response = llm.invoke(llm_content)
    else:
        response = "执行失败"
"""
import json
import logging

from src.workflow.skills.executor import SkillExecutionResult

logger = logging.getLogger(__name__)


# LLM 整理提示词模板
DATA_COMPOSE_PROMPT = """用户问题：{query}

技能 {skill_name} 的执行结果：
{data}

请基于以上执行结果，回答用户的问题。要求：
1. 使用自然语言，简洁清晰
2. 突出关键信息
3. 如果数据为空或异常，如实说明
"""

ERROR_COMPOSE_PROMPT = """用户问题：{query}

技能 {skill_name} 执行失败。
错误信息：{error}

请向用户解释情况，并给出可能的解决建议。"""


class ResponseComposer:
    """响应组装器

    纯粹的内容组装器，不依赖 LLM client。
    根据 SkillExecutionResult 的类型和状态，组装给 LLM 的内容。

    使用示例：
        composer = ResponseComposer()

        result = executor.execute_plan(plan, query)

        # 获取给 LLM 的内容
        llm_content = composer.compose(query, result)

        # 调用方决定如何使用 LLM
        if llm_content:
            response = llm.invoke(llm_content)
        else:
            response = "执行失败"
    """

    def __init__(self):
        """初始化

        不需要任何依赖。
        """
        logger.info("[ResponseComposer] 初始化完成")

    def compose(
        self,
        query: str,
        result: SkillExecutionResult,
    ) -> str | None:
        """组装执行结果为 LLM 输入内容

        Args:
            query: 用户原问题
            result: 技能执行结果

        Returns:
            给 LLM 的内容，如果无法组装则返回 None
            - prompt_template: 返回渲染后的 prompt
            - tool_execution 成功: 返回 data + query 组装的 prompt
            - tool_execution 失败: 返回错误信息 prompt 或 None
        """
        # 1. prompt_template 类型：直接返回 prompt
        if result.has_prompt():
            logger.info(f"[ResponseComposer] prompt_template: {result.skill_id}")
            return result.prompt

        # 2. tool_execution 类型：根据成功/失败组装
        if result.has_data():
            return self._compose_data(query, result)

        # 3. 执行失败：组装错误信息
        if not result.success:
            return self._compose_error(query, result)

        # 4. 无内容可组装
        logger.warning(f"[ResponseComposer] 无内容可组装: {result.skill_id}")
        return None

    def compose_direct_response(
        self,
        query: str,
        result: SkillExecutionResult,
    ) -> str | None:
        """组装直接返回给用户的响应（不经过 LLM）

        用于错误情况或简单结果。

        Args:
            query: 用户原问题
            result: 技能执行结果

        Returns:
            直接返回给用户的字符串，或 None（需要 LLM 处理）
        """
        # 执行失败：直接返回友好错误
        if not result.success:
            return self._format_friendly_error(result)

        return None

    def _compose_data(
        self,
        query: str,
        result: SkillExecutionResult,
    ) -> str:
        """组装工具执行数据为 LLM prompt

        Args:
            query: 用户问题
            result: 执行结果

        Returns:
            给 LLM 的 prompt
        """
        # 格式化数据
        if isinstance(result.data, dict):
            data_str = json.dumps(result.data, ensure_ascii=False, indent=2)
        else:
            data_str = str(result.data)

        logger.info(f"[ResponseComposer] 组装数据: skill={result.skill_id}")

        return DATA_COMPOSE_PROMPT.format(
            query=query,
            skill_name=result.skill_id,
            data=data_str,
        )

    def _compose_error(
        self,
        query: str,
        result: SkillExecutionResult,
    ) -> str | None:
        """组装错误信息为 LLM prompt

        Args:
            query: 用户问题
            result: 执行结果

        Returns:
            给 LLM 的 prompt，或 None（让调用方直接处理）
        """
        error_msg = result.error or "未知错误"

        logger.info(f"[ResponseComposer] 组装错误: skill={result.skill_id}, error={error_msg}")

        return ERROR_COMPOSE_PROMPT.format(
            query=query,
            skill_name=result.skill_id,
            error=error_msg,
        )

    def _format_friendly_error(
        self,
        result: SkillExecutionResult,
    ) -> str:
        """格式化友好的错误信息

        Args:
            result: 执行结果

        Returns:
            用户友好的错误提示
        """
        error_msg = result.error or "未知错误"

        # 根据错误类型生成友好提示
        if "not found" in error_msg.lower() or "不存在" in error_msg:
            return "抱歉，系统暂时无法处理您的请求。"
        elif "缺少" in error_msg or "参数" in error_msg:
            return f"参数不完整：{error_msg}。请补充必要信息后重试。"
        else:
            return f"处理失败：{error_msg}"
