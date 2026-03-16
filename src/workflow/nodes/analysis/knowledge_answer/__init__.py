# -*- coding: utf-8 -*-
"""Knowledge answer node with Agent mode skill_tool integration.

在现有知识问答基础上，增加 skill_tool 支持（Agent 模式）。

Agent 模式：
- 创建 SkillTool 作为 LangChain Tool
- 使用 llm_client.generate_with_agent() 自动处理 tool calling 循环
- LLM 自动决定是否调用 skill_tool，执行后自动获取最终答案
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from workflow.common.evidence import collect_evidence_hits
from workflow.common.func_utils import normalize_source_type
from workflow.llm.llm_client import CommonLLMRequest, CommonLLMResult
from workflow.llm.llm_prompt_utils import build_evidence_block, looks_like_reasoning_dump, resolve_system_prompt

logger = logging.getLogger(__name__)


# ============================================================================
# Skill 组件延迟初始化（Agent 模式）
# ============================================================================

# 模块级缓存（应用启动时初始化一次）
_skill_components: dict[str, Any] | None = None


def _get_skill_components() -> dict[str, Any] | None:
    """获取 skill 组件（延迟初始化，Agent 模式）

    Returns:
        包含 registry, executor, skill_tool 的字典，或 None
    """
    global _skill_components

    if _skill_components is not None:
        return _skill_components

    try:
        from src.workflow.skills import SkillExecutor, create_skill_tool
        from src.workflow.skills.registry import get_skill_registry

        # 尝试获取全局 registry（可能在其他地方已初始化）
        registry = get_skill_registry()

        # 如果 registry 为空，尝试加载技能
        if len(registry.skills) == 0:
            domain_root = Path("domain/ad_engine")
            if domain_root.exists():
                registry.load_from_directory(domain_root)
                logger.info(f"[KnowledgeAnswer] 加载技能: {len(registry.skills)} 个")

        # 如果仍然没有技能，跳过 skill 功能
        if len(registry.skills) == 0:
            logger.info("[KnowledgeAnswer] 无可用技能，跳过 skill_tool 绑定")
            return None

        # 创建执行器
        executor = SkillExecutor(external_registry=None)

        # 创建 LangChain Tool（用于 Agent 模式）
        skill_tool = create_skill_tool(registry, executor)

        _skill_components = {
            "registry": registry,
            "executor": executor,
            "skill_tool": skill_tool,
        }

        logger.info(
            f"[KnowledgeAnswer] Skill 组件初始化完成（Agent 模式）: "
            f"技能数={len(registry.skills)}"
        )

        return _skill_components

    except Exception as e:
        logger.warning(f"[KnowledgeAnswer] Skill 组件初始化失败: {e}")
        return None


# ============================================================================
# 提示词模板
# ============================================================================

QA_SYSTEM_PROMPT_TEMPLATE = (
    "你是企业知识问答助手。"
    "必须严格基于提供的证据回答，不补充证据外事实。"
    "如果用户的问题需要查询实时数据或执行特定技能，请使用 skill_tool。"
    "输出中文，结构尽量为：结论 -> 依据。"
)

QA_USER_PROMPT_TEMPLATE = """【用户问题】
{user_query}

【当前主模块】
- module_name: {module_name}
- module_hint: {module_hint}

【相关模块】
{related_modules_block}

【检索证据（按相关性排序）】
{evidence_block}
"""


# ============================================================================
# 辅助函数
# ============================================================================


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords if keyword)


def _infer_question_type(service: Any, user_query: str) -> str:
    normalized = str(user_query or "").lower().strip()
    formula_terms = {
        "公式",
        "怎么算",
        "如何计算",
        "计算方式",
        "calculate",
        "formula",
        "bid",
    }
    formula_terms.update(service.domain_profile.answering.bid_terms)
    if any(token in normalized for token in formula_terms if token):
        return "formula"
    if any(token in normalized for token in ("为什么", "为何", "原因", "how to", "why")):
        return "reason"
    if any(token in normalized for token in ("有哪些", "哪些", "包括什么", "包含什么", "分别是", "列出", "清单")):
        return "list"
    return "general"


def _is_code_location_query(service: Any, user_query: str) -> bool:
    normalized = str(user_query or "").lower().strip()
    code_location_terms = service.domain_profile.query_rewrite.intent_terms.get("code_location", ())
    if code_location_terms:
        return _contains_any(normalized, code_location_terms)
    fallback = (
        "哪个函数",
        "什么函数",
        "入口函数",
        "哪个文件",
        "文件路径",
        "哪一行",
        "where",
        "which function",
    )
    return _contains_any(normalized, fallback)


def _format_line_range(item: dict[str, Any]) -> str:
    start_line = item.get("start_line")
    end_line = item.get("end_line")
    if start_line is not None and end_line is not None:
        return f"{start_line}-{end_line}"
    if start_line is not None:
        return str(start_line)
    return "unknown"


def _answer_mentions_code_anchor(answer_text: str, code_hits: list[dict[str, Any]]) -> bool:
    normalized_answer = str(answer_text or "").lower()
    for item in code_hits[:6]:
        path = str(item.get("path", "")).strip().lower()
        symbol_name = str(item.get("symbol_name", "")).strip().lower()
        if path and (path in normalized_answer or path.split("/")[-1] in normalized_answer):
            return True
        if symbol_name and symbol_name in normalized_answer:
            return True
    return False


def _extract_points_from_hits(evidence_hits: list[dict[str, Any]], *, max_points: int = 6) -> list[str]:
    points: list[str] = []
    for item in evidence_hits:
        source_type = normalize_source_type(item.get("source_type"))
        excerpt = re.sub(r"\s+", " ", str(item.get("excerpt", "")).strip())
        if source_type == "code":
            path = str(item.get("path", "")).strip() or "unknown_path"
            symbol_name = str(item.get("symbol_name", "")).strip() or str(item.get("section", "")).strip() or "unknown_symbol"
            points.append(f"代码位置：`{path}` -> `{symbol_name}` (line: {_format_line_range(item)})")
            if excerpt:
                points.append(f"代码摘要：{excerpt[:140]}")
        else:
            title = str(item.get("title", "")).strip() or str(item.get("path", "")).strip() or "未命名证据"
            if excerpt:
                points.append(f"{title}：{excerpt[:140]}")
            else:
                points.append(title)
        if len(points) >= max_points:
            break
    return points[:max_points]


def _build_code_location_fallback(module_name: str, module_hint: str, code_hits: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for item in code_hits[:3]:
        path = str(item.get("path", "")).strip() or "unknown_path"
        symbol_name = str(item.get("symbol_name", "")).strip() or str(item.get("section", "")).strip() or "unknown_symbol"
        lines.append(f"{len(lines) + 1}. `{path}` -> `{symbol_name}` (line: {_format_line_range(item)})")
    if not lines:
        lines = [
            f"1. 当前问题主要落在 `{module_name}`。",
            f"2. 当前模块说明：{module_hint or '--'}。",
            "3. 暂无可直接定位的代码证据，建议补充函数名、文件名或调用链关键词后重试。",
        ]
    return "\n".join(lines)


def _build_general_fallback(
    *,
    module_name: str,
    module_hint: str,
    question_type: str,
    evidence_hits: list[dict[str, Any]],
) -> str:
    points = _extract_points_from_hits(evidence_hits, max_points=6)
    if not points:
        return (
            f"当前问题主要落在 `{module_name}`。"
            f" 模块说明：{module_hint or '--'}。"
            " 当前没有命中足够证据，建议补充更具体的业务词、字段名或代码符号后重试。"
        )

    if question_type == "list":
        return "\n".join(f"{index}. {point}" for index, point in enumerate(points[:5], start=1))
    if question_type in {"reason", "formula"}:
        return "\n".join(f"{index}. {point}" for index, point in enumerate(points[:4], start=1))
    return "；".join(points[:3])


def _default_llm_call_status() -> dict[str, Any]:
    return {
        "status": "not_configured",
        "invoked": False,
        "request_sent": False,
        "attempts": 0,
        "latency_ms": 0,
        "reason": None,
        "model": None,
    }


def _build_related_modules_block(related_modules: list[dict[str, Any]]) -> str:
    """构建相关模块展示文本。"""
    if not related_modules:
        return "- 无"

    rows: list[str] = []
    for item in related_modules[:3]:
        module_name = str(item.get("module_name", "")).strip()
        module_hint = str(item.get("module_hint", "")).strip() or "--"
        if not module_name:
            continue
        rows.append(f"- module_name: {module_name}")
        rows.append(f"  module_hint: {module_hint}")
    return "\n".join(rows) if rows else "- 无"


def _enforce_structured_output(answer: str, *, question_type: str) -> str:
    if not answer:
        return ""
    if question_type not in {"list", "reason", "formula"}:
        return answer

    has_list = bool(re.search(r"(?m)^\s*(?:\d+\.|[-*])\s+", answer))
    if has_list:
        return answer

    sentences = [segment.strip() for segment in re.split(r"[。\n,，!?！？]", answer) if segment.strip()]
    if not sentences:
        return answer

    top_sentences = sentences[:4]
    list_body = "\n".join(f"{index}. {text}" for index, text in enumerate(top_sentences, start=1))
    if question_type == "list":
        section_title = "要点清单"
    elif question_type == "reason":
        section_title = "主要原因"
    else:
        section_title = "公式与关键变量"
    return f"{answer}\n\n**{section_title}**\n{list_body}"


def _validate_qa_answer(answer: str) -> tuple[bool, str | None]:
    if not answer.strip():
        return False, "empty_answer"
    if looks_like_reasoning_dump(answer):
        return False, "empty_answer:reasoning_dump"
    return True, None


# ============================================================================
# LLM 调用函数
# ============================================================================


def _run_llm_with_agent(
    service: Any,
    *,
    user_query: str,
    question_type: str,
    module_name: str,
    module_hint: str,
    related_modules: list[dict[str, Any]],
    evidence_hits: list[dict[str, Any]],
) -> tuple[str | None, str | None, dict[str, Any]]:
    """使用 Agent 模式调用 LLM（自动处理 skill_tool）

    Agent 模式流程：
    1. 调用 LLM（绑定 skill_tool）
    2. 如果 LLM 返回 tool_calls，自动执行 skill_tool
    3. 将执行结果发回 LLM
    4. LLM 生成最终答案

    Args:
        service: WorkflowService 实例
        user_query: 用户问题
        question_type: 问题类型
        module_name: 模块名称
        module_hint: 模块提示
        related_modules: 相关模块
        evidence_hits: 证据命中列表

    Returns:
        (answer, fallback_reason, call_status) 元组
    """
    llm_client = getattr(service, "_llm_client", None)
    if llm_client is None:
        return None, None, _default_llm_call_status()

    system_prompt = resolve_system_prompt(
        env_key="WORKFLOW_QA_LLM_SYSTEM_PROMPT",
        default_prompt=QA_SYSTEM_PROMPT_TEMPLATE,
        domain_profile=getattr(service, "domain_profile", None),
    )

    # 获取 skill_tool（LangChain Tool）
    skill_components = _get_skill_components()
    tools = None
    if skill_components is not None:
        tools = [skill_components["skill_tool"]]
        logger.info(f"[KnowledgeAnswer] Agent 模式：绑定 skill_tool")

    # 构建请求
    request = CommonLLMRequest(
        node_name="knowledge_answer",
        system_prompt=system_prompt,
        user_prompt=QA_USER_PROMPT_TEMPLATE.format(
            user_query=user_query,
            module_name=module_name,
            module_hint=module_hint,
            related_modules_block=_build_related_modules_block(related_modules),
            evidence_block=build_evidence_block(evidence_hits),
        ),
        evidence_count=len(evidence_hits),
        require_evidence=True,
        log_namespace="workflow.llm_qa",
        metadata={
            "module_name": module_name,
            "question_type": question_type,
            "related_module_count": len(related_modules),
            "user_query_preview": user_query[:120],
            "skill_tool_enabled": tools is not None,
        },
        normalize_answer=lambda text: _enforce_structured_output(text, question_type=question_type),
        validate_answer=_validate_qa_answer,
    )

    # 使用 Agent 模式（自动处理 tool calling 循环）
    if tools and hasattr(llm_client, "generate_with_agent"):
        result = llm_client.generate_with_agent(request, tools=tools, max_iterations=3)
        return result.answer, result.fallback_reason, dict(result.call_status)

    # 降级：使用普通模式
    if hasattr(llm_client, "generate_with_status"):
        return llm_client.generate_with_status(request)

    if hasattr(llm_client, "generate_answer_with_status"):
        return llm_client.generate_answer_with_status(
            user_query=user_query,
            question_type=question_type,
            module_name=module_name,
            module_hint=module_hint,
            related_modules=related_modules,
            evidence_hits=evidence_hits,
        )

    answer_text, fallback_reason = llm_client.generate_answer(
        user_query=user_query,
        question_type=question_type,
        module_name=module_name,
        module_hint=module_hint,
        related_modules=related_modules,
        evidence_hits=evidence_hits,
    )
    status = getattr(llm_client, "last_call_status", {}) or {}
    call_status = dict(status) if isinstance(status, dict) else _default_llm_call_status()
    return answer_text, fallback_reason, call_status


# ============================================================================
# Node 入口
# ============================================================================


def run(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """知识问答节点入口（Agent 模式）

    Agent 模式下，LLM 自动决定是否调用 skill_tool：
    1. LLM 基于 evidence 直接生成答案，或
    2. LLM 调用 skill_tool，执行技能后自动生成答案

    Args:
        service: WorkflowService 实例
        state: 工作流状态

    Returns:
        更新后的状态增量
    """
    module_name = state["module_name"]
    module_hint = state["module_hint"]
    related_modules = list(state.get("related_modules", []) or [])
    user_query = str(state.get("user_query", ""))
    question_type = _infer_question_type(service, user_query)

    wiki_hits = list(state.get("wiki_hits", []))
    code_hits = list(state.get("code_hits", []))
    evidence_hits = collect_evidence_hits(state)

    llm_mode = "fallback_rule"
    llm_fallback_reason: str | None = None
    llm_call_status = _default_llm_call_status()
    final_answer: str

    # 使用 Agent 模式调用 LLM
    llm_answer_text, llm_fallback_reason, llm_call_status = _run_llm_with_agent(
        service,
        user_query=user_query,
        question_type=question_type,
        module_name=module_name,
        module_hint=module_hint,
        related_modules=related_modules,
        evidence_hits=evidence_hits,
    )

    llm_client = getattr(service, "_llm_client", None)
    llm_call_status.setdefault("model", getattr(getattr(llm_client, "config", None), "model", None))
    llm_call_status.setdefault("invoked", bool(llm_client is not None))

    # 处理 LLM 回答
    if llm_answer_text:
        # 检查是否是代码定位问题且答案缺少代码锚点
        if service.domain_profile.is_code_location_query(user_query) and code_hits and not _answer_mentions_code_anchor(llm_answer_text, code_hits):
            llm_fallback_reason = "llm_missing_code_anchor"
            llm_call_status.update(
                {
                    "status": "success_discarded",
                    "reason": "llm_missing_code_anchor",
                }
            )
            final_answer = _build_code_location_fallback(module_name, module_hint, code_hits)
        else:
            # 判断是否使用了 skill_tool（根据 call_status 的 iterations）
            iterations = llm_call_status.get("attempts", 1)
            if iterations > 1:
                llm_mode = "skill_tool"
            else:
                llm_mode = "llm"
            final_answer = llm_answer_text
    else:
        # 无 LLM 回答，使用 fallback
        if service.domain_profile.is_code_location_query(user_query) and code_hits:
            final_answer = _build_code_location_fallback(module_name, module_hint, code_hits)
        else:
            final_answer = _build_general_fallback(
                module_name=module_name,
                module_hint=module_hint,
                question_type=question_type,
                evidence_hits=evidence_hits,
            )

    # 构建返回结果
    analysis: dict[str, Any] = {
        "summary": "知识问答已完成",
        "module": module_name,
        "related_modules": related_modules,
        "confidence": "medium",
        "generation_mode": llm_mode,
        "question_type": question_type,
        "evidence_count": len(evidence_hits),
        "wiki_evidence_count": len(wiki_hits),
        "code_evidence_count": len(code_hits),
        "llm_enabled": bool(llm_client is not None and llm_client.config.enabled),
        "llm_available": bool(llm_client is not None and llm_client.is_available),
        "llm_model": (llm_client.config.model if llm_client is not None else None),
        "llm_fallback_reason": llm_fallback_reason,
        "llm_call_status": llm_call_status,
        "highlights": [
            "知识问答使用 Agent 模式，自动处理 skill_tool 调用",
            "LLM 自动决定是否需要调用技能获取实时数据",
            "代码定位类问题会额外检查答案是否包含代码锚点",
            "最终 Markdown 三段式格式由 finalize_response 节点统一收口",
        ],
    }

    return {
        "response_kind": "knowledge_qa",
        "status": "completed",
        "answer": final_answer,
        "analysis": analysis,
        "node_trace": service._trace(
            state,
            "knowledge_answer",
            f"module={module_name}, related={len(related_modules)}",
        ),
    }
