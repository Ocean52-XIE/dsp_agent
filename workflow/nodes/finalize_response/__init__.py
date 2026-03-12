from __future__ import annotations

import re
from typing import Any


SCENE_LABELS = {
    "knowledge_qa": "知识问答",
    "issue_analysis": "问题分析",
    "code_generation": "代码实现建议",
    "conversation_control": "会话控制",
    "out_of_scope": "范围外问题",
}


def _safe_text_preview(raw_value: Any, *, max_chars: int = 200) -> str:
    text = str(raw_value or "").strip()
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}..."


def _normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _build_hit_preview(hits: list[dict[str, Any]], *, limit: int = 3) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for hit in hits[: max(1, limit)]:
        items.append(
            {
                "source_type": str(hit.get("source_type", "") or ""),
                "path": str(hit.get("path", "") or ""),
                "title": str(hit.get("title", "") or ""),
                "section": str(hit.get("section", "") or ""),
                "score": float(hit.get("score", 0.0) or 0.0),
                "excerpt_preview": _safe_text_preview(hit.get("excerpt", ""), max_chars=160),
            }
        )
    return items


def _build_verbose_debug(state: dict[str, Any], graph_path: list[str]) -> dict[str, Any]:
    wiki_hits = list(state.get("wiki_hits", []) or [])
    code_hits = list(state.get("code_hits", []) or [])
    case_hits = list(state.get("case_hits", []) or [])
    citations = list(state.get("citations", []) or [])
    return {
        "history_summary": str(state.get("history_summary", "") or ""),
        "retrieval_queries": list(state.get("retrieval_queries", []) or []),
        "retrieval_plan": dict(state.get("retrieval_plan", {}) or {}),
        "retrieval_grades": {
            "wiki": str(state.get("wiki_retrieval_grade", "unknown") or "unknown"),
            "code": str(state.get("code_retrieval_grade", "unknown") or "unknown"),
            "case": str(state.get("case_retrieval_grade", "unknown") or "unknown"),
        },
        "retrieval_profiles": {
            "wiki": dict(state.get("wiki_retrieval_profile", {}) or {}),
            "code": dict(state.get("code_retrieval_profile", {}) or {}),
            "case": dict(state.get("case_retrieval_profile", {}) or {}),
            "fusion": dict(state.get("evidence_fusion_profile", {}) or {}),
        },
        "retrieval_hit_counts": {
            "wiki": len(wiki_hits),
            "code": len(code_hits),
            "case": len(case_hits),
            "citations": len(citations),
        },
        "hit_preview": {
            "wiki": _build_hit_preview(wiki_hits, limit=3),
            "code": _build_hit_preview(code_hits, limit=3),
            "case": _build_hit_preview(case_hits, limit=2),
        },
        "analysis_preview": _safe_text_preview(state.get("analysis", ""), max_chars=300),
        "graph_path": graph_path,
        "node_trace": list(state.get("node_trace", []) or []),
    }


def _normalize_source_type(raw_source: Any) -> str:
    normalized = str(raw_source or "").strip().lower()
    if normalized.startswith("wiki"):
        return "wiki"
    if normalized.startswith("code"):
        return "code"
    if normalized.startswith("case"):
        return "case"
    return normalized or "unknown"


def _collect_evidence_hits(state: dict[str, Any]) -> list[dict[str, Any]]:
    citations = list(state.get("citations", []) or [])
    if citations:
        normalized_rows: list[dict[str, Any]] = []
        for item in citations:
            row = dict(item)
            row["source_type"] = _normalize_source_type(row.get("source_type", row.get("source", "")))
            normalized_rows.append(row)
        return normalized_rows

    hits: list[dict[str, Any]] = []
    for key, source_type in (("wiki_hits", "wiki"), ("code_hits", "code"), ("case_hits", "case")):
        for item in list(state.get(key, []) or []):
            row = dict(item)
            row["source_type"] = _normalize_source_type(row.get("source_type", source_type))
            hits.append(row)
    return hits


def _line_range(item: dict[str, Any]) -> str:
    start_line = item.get("start_line")
    end_line = item.get("end_line")
    if start_line is not None and end_line is not None:
        return f"{start_line}-{end_line}"
    if start_line is not None:
        return str(start_line)
    return "unknown"


def _is_code_location_query(user_query: str) -> bool:
    normalized = str(user_query or "").lower().strip()
    keywords = (
        "哪个函数",
        "什么函数",
        "入口函数",
        "函数链路",
        "函数调用链",
        "哪个文件",
        "文件路径",
        "在哪一行",
        "实现位置",
        "代码位置",
        "where",
        "which function",
        "locate",
    )
    return any(token in normalized for token in keywords)


def _is_formula_query(user_query: str) -> bool:
    normalized = str(user_query or "").lower().strip()
    formula_tokens = ("公式", "怎么算", "怎么计算", "如何计算", "计算方式", "计算形式", "formula")
    if any(token in normalized for token in formula_tokens):
        return True
    if _contains_any(normalized, ("target_cpa", "target_roas", "alpha", "bid", "出价")) and _contains_any(
        normalized, ("计算", "公式", "变量")
    ):
        return True
    if "ecpm" in normalized and _contains_any(normalized, ("计算", "公式", "how", "formula")):
        return True
    if _contains_any(normalized, ("校准", "概率")) and _contains_any(normalized, ("计算", "形式", "公式", "截断", "0 到 1", "0到1")):
        return True
    return False


def _is_bid_entry_or_chain_query(user_query: str) -> bool:
    normalized = str(user_query or "").lower().strip()
    has_bid_terms = any(token in normalized for token in ("出价", "bid", "target_cpa", "pcvr"))
    has_entry_terms = any(token in normalized for token in ("入口函数", "入口", "函数链路", "函数调用链", "串联计算"))
    return has_bid_terms and has_entry_terms


def _normalize_formula_expression(formula: str) -> str:
    normalized = _normalize_text(formula)
    normalized = re.sub(r"\s*[×xX]\s*", " * ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _extract_formula_candidates(text: str) -> list[str]:
    candidates: list[str] = []

    for match in re.findall(r"`([^`\n]*=[^`\n]*)`", str(text or "")):
        cleaned = _normalize_formula_expression(match).strip("` ")
        if cleaned and cleaned not in candidates:
            candidates.append(cleaned)

    plain_pattern = re.compile(
        r"([A-Za-z_][A-Za-z0-9_]*\s*=\s*[A-Za-z0-9_().]+\s*(?:\s*[*xX×/+\-]\s*[A-Za-z0-9_().]+){1,8})"
    )
    for match in plain_pattern.findall(str(text or "")):
        cleaned = _normalize_formula_expression(match).strip("` ")
        if cleaned and cleaned not in candidates:
            candidates.append(cleaned)

    return candidates


def _pick_best_formula(candidates: list[str], *, user_query: str) -> str | None:
    if not candidates:
        return None

    normalized_query = str(user_query or "").lower()
    if "target_cpa" in normalized_query and "pcvr" in normalized_query:
        for formula in candidates:
            lowered = formula.lower()
            if "target_cpa" in lowered and "pcvr" in lowered:
                return formula

    if "ecpm" in normalized_query:
        for formula in candidates:
            lowered = formula.lower()
            if lowered.startswith("ecpm") or "ecpm" in lowered:
                return formula

    return candidates[0]


def _extract_formula_from_answer_or_evidence(
    *,
    conclusion_body: str,
    raw_answer: str,
    evidence_hits: list[dict[str, Any]],
    user_query: str,
) -> str | None:
    normalized_query = str(user_query or "").lower()
    answer_candidates = _extract_formula_candidates(conclusion_body) + _extract_formula_candidates(raw_answer)
    if "ecpm" in normalized_query:
        for formula in answer_candidates:
            lowered = formula.lower()
            if lowered.startswith("ecpm") or "ecpm" in lowered:
                return formula

    best_from_answer = _pick_best_formula(answer_candidates, user_query=user_query)
    if best_from_answer and not ("ecpm" in normalized_query and "ecpm" not in best_from_answer.lower()):
        return best_from_answer

    evidence_candidates: list[str] = []
    for item in evidence_hits:
        excerpt = str(item.get("excerpt", "") or "")
        for formula in _extract_formula_candidates(excerpt):
            if formula not in evidence_candidates:
                evidence_candidates.append(formula)

    if "ecpm" in normalized_query:
        for formula in evidence_candidates:
            lowered = formula.lower()
            if lowered.startswith("ecpm") or "ecpm" in lowered:
                return formula
    return _pick_best_formula(evidence_candidates, user_query=user_query)


def _find_code_hit_by_symbol(code_hits: list[dict[str, Any]], symbol_name: str) -> dict[str, Any] | None:
    target = str(symbol_name or "").strip().lower()
    if not target:
        return None
    for item in code_hits:
        current = str(item.get("symbol_name", "") or item.get("section", "")).strip().lower()
        if current == target:
            return item
    for item in code_hits:
        excerpt = str(item.get("excerpt", "") or "")
        lowered = excerpt.lower()
        if f"def {target}(" in lowered or f"{target}(" in lowered:
            return item
    return None


def _build_preferred_symbols_for_query(*, user_query: str, is_formula: bool) -> list[str]:
    normalized_query = str(user_query or "").lower().strip()
    preferred_by_query: list[str] = []

    if "ecpm" in normalized_query:
        preferred_by_query.extend(["compute_rank_score", "rank_topn"])
    if _contains_any(normalized_query, ("请求级", "出价入口", "入口函数")):
        preferred_by_query.append("compute_bid_for_request")
    if _contains_any(normalized_query, ("胜率", "成本")) and _contains_any(normalized_query, ("函数", "排查", "入口")):
        preferred_by_query.extend(["compute_bid_for_request", "compute_alpha"])
    if _contains_any(normalized_query, ("函数链路", "函数调用链", "串联计算")):
        preferred_by_query.extend(["compute_bid_for_request", "compute_alpha", "compute_ocpc_bid"])
    if _contains_any(normalized_query, ("两率预测入口", "两率预测", "预测入口", "预测函数", "rate_predictor")) or (
        "pctr" in normalized_query and "入口" in normalized_query
    ):
        preferred_by_query.append("predict_ctr_cvr")
    if _contains_any(normalized_query, ("召回", "候选")) and _contains_any(normalized_query, ("入口", "函数", "代码")):
        preferred_by_query.append("select_recall_candidates")
    if _contains_any(normalized_query, ("冷启动", "无量")) and _contains_any(normalized_query, ("入口", "函数", "代码", "路径")):
        preferred_by_query.extend(["select_recall_candidates", "rank_topn"])
    if _contains_any(normalized_query, ("精排", "排序")) and _contains_any(normalized_query, ("入口", "函数", "在哪")):
        preferred_by_query.extend(["rank_topn", "compute_rank_score"])
    if _contains_any(normalized_query, ("校准", "截断", "0 到 1", "0到1")):
        preferred_by_query.append("calibrate_probability")
    if _contains_any(normalized_query, ("多样性惩罚",)):
        preferred_by_query.append("apply_diversity_penalty")
    if _contains_any(normalized_query, ("频控", "风控", "安全惩罚")):
        preferred_by_query.append("apply_safety_penalty")
    if _contains_any(normalized_query, ("最终排序分", "排序分", "rank_score", "精排")):
        preferred_by_query.append("compute_rank_score")
    if is_formula and _contains_any(normalized_query, ("target_cpa", "pcvr", "出价公式")):
        if _is_bid_entry_or_chain_query(normalized_query):
            preferred_by_query.extend(["compute_bid_for_request", "compute_ocpc_bid"])
        else:
            preferred_by_query.extend(["compute_ocpc_bid", "compute_bid_for_request"])

    deduped: list[str] = []
    seen: set[str] = set()
    for symbol in preferred_by_query:
        key = symbol.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(symbol)
    return deduped


def _resolve_code_anchor(
    code_hits: list[dict[str, Any]],
    *,
    user_query: str = "",
    is_formula: bool = False,
) -> tuple[dict[str, Any], str]:
    for symbol in _build_preferred_symbols_for_query(user_query=user_query, is_formula=is_formula):
        found = _find_code_hit_by_symbol(code_hits, symbol)
        if found is not None:
            return found, symbol
    primary = _choose_primary_code_hit(code_hits, user_query=user_query, is_formula=is_formula)
    symbol_name = str(primary.get("symbol_name", "")).strip() or str(primary.get("section", "")).strip() or "unknown_symbol"
    return primary, symbol_name


def _choose_primary_code_hit(
    code_hits: list[dict[str, Any]],
    *,
    user_query: str = "",
    is_formula: bool = False,
) -> dict[str, Any]:
    if not code_hits:
        return {}

    for symbol in _build_preferred_symbols_for_query(user_query=user_query, is_formula=is_formula):
        found = _find_code_hit_by_symbol(code_hits, symbol)
        if found is not None:
            return found

    preferred_symbols = ("compute_", "predict_", "rank_", "select_", "calibrate_")
    for item in code_hits:
        symbol_name = str(item.get("symbol_name", "") or "").strip().lower()
        if symbol_name.startswith(preferred_symbols):
            return item

    for item in code_hits:
        start_line = item.get("start_line")
        end_line = item.get("end_line")
        if isinstance(start_line, int) and isinstance(end_line, int) and 0 <= end_line - start_line <= 30:
            return item

    return code_hits[0]


def _clean_heading_text(line: str) -> str | None:
    stripped = line.strip()
    heading_match = re.match(r"^#{1,6}\s+(.+?)\s*$", stripped)
    if heading_match:
        return heading_match.group(1).strip("* ").strip()
    bold_match = re.match(r"^\*\*(.+?)\*\*$", stripped)
    if bold_match:
        return bold_match.group(1).strip()
    return None


def _split_markdown_sections(text: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    current_title: str | None = None
    current_lines: list[str] = []

    for raw_line in str(text or "").splitlines():
        heading = _clean_heading_text(raw_line)
        if heading is not None:
            if current_title is not None:
                body = "\n".join(current_lines).strip()
                if body:
                    sections[current_title] = body
            current_title = heading
            current_lines = []
            continue
        current_lines.append(raw_line)

    if current_title is not None:
        body = "\n".join(current_lines).strip()
        if body:
            sections[current_title] = body

    return sections


def _extract_first_meaningful_block(text: str) -> str:
    for block in re.split(r"\n\s*\n", str(text or "").strip()):
        cleaned = block.strip()
        if not cleaned:
            continue
        if _clean_heading_text(cleaned) is not None:
            continue
        return cleaned
    return ""


def _extract_first_sentence(text: str) -> str:
    normalized = _normalize_text(text)
    if not normalized:
        return ""
    parts = re.split(r"[。！？；;]", normalized)
    for part in parts:
        cleaned = part.strip(" :：-")
        if cleaned:
            return cleaned
    return normalized


def _extract_question_conclusion(raw_answer: str, *, module_name: str, module_hint: str) -> str:
    if not str(raw_answer or "").strip():
        if module_hint:
            return f"当前问题主要落在 `{module_name or '--'}`，关注点是：{module_hint}。"
        return f"当前问题主要落在 `{module_name or '--'}` 模块范围内。"

    sections = _split_markdown_sections(raw_answer)
    for title in ("问题结论", "结论"):
        section = sections.get(title)
        if section:
            return _normalize_text(section)

    first_block = _extract_first_meaningful_block(raw_answer)
    if first_block:
        cleaned = re.sub(r"^(?:[-*]\s+|\d+\.\s+)", "", first_block).strip()
        cleaned = re.sub(r"^(?:问题结论|结论)\s*[:：]\s*", "", cleaned).strip()
        if cleaned:
            return _normalize_text(cleaned)

    if module_hint:
        return f"当前问题主要落在 `{module_name or '--'}`，关注点是：{module_hint}。"
    return f"当前问题主要落在 `{module_name or '--'}` 模块范围内。"


def _extract_conclusion_body(text: str) -> str:
    raw_text = str(text or "").strip()
    if not raw_text:
        return ""

    sections = _split_markdown_sections(raw_text)
    for title in ("问题结论", "结论"):
        section = sections.get(title)
        if section:
            return section.strip()

    stop_patterns = (
        r"(?im)^\s*#{1,6}\s*(?:证据说明|回答依据|依据)\s*$",
        r"(?im)^\s*\*\*(?:证据说明|回答依据|依据)\*\*\s*$",
        r"(?im)^\s*\d+\.\s*(?:证据说明|回答依据|依据)\s*$",
        r"(?im)^\s*(?:证据说明|回答依据|依据)\s*[:：]?\s*$",
    )
    end_pos = len(raw_text)
    for pattern in stop_patterns:
        matched = re.search(pattern, raw_text)
        if matched:
            end_pos = min(end_pos, matched.start())
    return raw_text[:end_pos].strip()


def _strip_conclusion_prefix(text: str) -> str:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"(?im)^(?:问题)?结论\s*[:：]\s*", "", cleaned)
    cleaned = re.sub(r"(?im)^\d+\.\s*(?:问题)?结论\s*[:：]?\s*", "", cleaned)
    return cleaned.strip()


def _extract_list_items(raw_answer: str, *, limit: int = 5) -> list[str]:
    items: list[str] = []
    for line in str(raw_answer or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        stripped = re.sub(r"^\d+\.\s+", "", stripped)
        stripped = re.sub(r"^[-*+]\s+", "", stripped)
        if stripped == line.strip():
            continue
        stripped = _normalize_text(stripped)
        if stripped and stripped not in items:
            items.append(stripped)
        if len(items) >= limit:
            return items

    normalized = _normalize_text(raw_answer)
    if not items and normalized:
        parts = [part.strip(" :：-") for part in re.split(r"[；;。]", normalized) if part.strip(" :：-")]
        for part in parts[:limit]:
            if part not in items:
                items.append(part)
    return items[:limit]


def _clean_list_item_text(text: str) -> str:
    cleaned = str(text or "").strip()
    cleaned = re.split(r"(?i)[\r\n\s]+\d+\.\s*(?:依据|回答依据|证据说明)", cleaned, maxsplit=1)[0]
    cleaned = re.split(r"(?i)[\r\n\s]+(?:依据|回答依据|证据说明)\s*[:：]", cleaned, maxsplit=1)[0]
    cleaned = re.split(r"[\r\n\s]+\d+\.\s+", cleaned, maxsplit=1)[0]
    return _normalize_text(cleaned).strip(" :：-")


def _extract_list_items_from_excerpt(text: str, *, limit: int = 5) -> list[str]:
    items: list[str] = []

    normalized = str(text or "")
    inline_matches = re.findall(r"(?:^|\s)-\s*([^-\n][^-]+?)(?=(?:\s+-\s)|$)", normalized)
    if inline_matches:
        for match in inline_matches:
            cleaned = _clean_list_item_text(match)
            if cleaned and cleaned not in items:
                items.append(cleaned)
            if len(items) >= limit:
                return items[:limit]
        if len(items) >= 2:
            return items[:limit]

    for line in str(text or "").splitlines():
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        cleaned = _clean_list_item_text(stripped[2:])
        if cleaned and cleaned not in items:
            items.append(cleaned)
        if len(items) >= limit:
            return items

    return items[:limit]


def _split_summary_and_inline_items(text: str, *, limit: int = 5) -> tuple[str, list[str]]:
    raw_text = str(text or "").strip()
    if not raw_text:
        return "", []

    bullet_match = re.search(r"\s[-*]\s+", raw_text)
    if not bullet_match:
        return "", []

    summary = raw_text[: bullet_match.start()].strip()
    items = _extract_list_items_from_excerpt(raw_text[bullet_match.start() :], limit=limit)
    summary = _strip_conclusion_prefix(_normalize_text(summary)).strip()
    if summary and not summary.endswith(("：", ":")):
        if any(token in summary for token in ("包括", "如下", "主要原因", "要点")):
            summary = f"{summary}："
    return summary, items


def _derive_list_items_from_evidence(evidence_hits: list[dict[str, Any]], *, limit: int = 5) -> list[str]:
    items: list[str] = []

    for item in evidence_hits:
        if _normalize_source_type(item.get("source_type")) == "code":
            continue
        excerpt_items = _extract_list_items_from_excerpt(str(item.get("excerpt", "") or ""), limit=limit)
        for excerpt_item in excerpt_items:
            if excerpt_item not in items:
                items.append(excerpt_item)
            if len(items) >= limit:
                return items

    return items[:limit]


def _contains_any(text: str, tokens: tuple[str, ...]) -> bool:
    normalized = _normalize_text(text).lower()
    return any(token in normalized for token in tokens)


def _is_reason_like_query(user_query: str) -> bool:
    normalized = str(user_query or "").lower().strip()
    reason_tokens = (
        "原因",
        "为什么",
        "为何",
        "由什么导致",
        "导致",
        "常见问题",
        "常见原因",
        "没进最终位",
        "无量",
        "骤降",
        "偏高",
        "偏低",
        "抖动",
        "胜率下降",
        "成本超标",
    )
    return any(token in normalized for token in reason_tokens)


def _match_canonical_reason(cleaned: str) -> str | None:
    lowered = cleaned.lower()
    if "竞对出价" in cleaned and "抬升" in cleaned:
        return "竞对出价抬升"
    if ("两率预估" in cleaned or "pctr" in lowered or "pcvr" in lowered) and "偏低" in cleaned:
        return "两率预估偏低"
    if "pacing" in lowered and "保守" in cleaned:
        return "pacing过于保守"
    if ("目标" in cleaned and ("cpa" in lowered or "roi" in lowered) and "错误" in cleaned) or (
        "目标配置" in cleaned and "错误" in cleaned
    ):
        return "目标配置错误"
    if "回传" in cleaned and ("延迟" in cleaned or "滞后" in cleaned):
        return "回传延迟"
    if "高风险流量" in cleaned and "抑制" in cleaned:
        return "高风险流量抑制缺失"
    if "索引延迟" in cleaned:
        return "索引延迟"
    if "过滤条件" in cleaned and ("过严" in cleaned or "严" in cleaned):
        return "过滤条件过严"
    if "特征缺失" in cleaned:
        return "特征缺失"
    if "并发" in cleaned and "不足" in cleaned:
        return "并发不足"
    if "超时" in cleaned and ("未做降级" in cleaned or "未降级" in cleaned):
        return "超时未降级"
    if "下游依赖" in cleaned and "抖动" in cleaned:
        return "下游依赖抖动"
    if "特征分布" in cleaned and "漂移" in cleaned:
        return "特征分布漂移"
    if "模型版本" in cleaned and "误切换" in cleaned:
        return "模型版本误切换"
    if "校准参数" in cleaned and "同步" in cleaned:
        return "校准参数同步"
    if "探索流量" in cleaned and "配额" in cleaned:
        return "探索流量配额"
    if ("内容相似" in cleaned and "召回" in cleaned) or "内容相似召回" in cleaned:
        return "内容相似召回"
    if "分层保底" in cleaned:
        return "分层保底策略"
    if "实时特征" in cleaned and "波动" in cleaned:
        return "实时特征波动"
    if "探索权重" in cleaned and ("过高" in cleaned or "偏高" in cleaned):
        return "探索权重过高"
    if "多约束" in cleaned and "冲突" in cleaned:
        return "多约束冲突"
    return None


def _canonicalize_reason_item(item: str, *, keep_unmatched: bool = True) -> str:
    cleaned = _normalize_text(item).strip(" :：-")
    cleaned = re.sub(r"^[>*#`*_\s-]+", "", cleaned)
    cleaned = re.sub(r"[>*#`*_\s-]+$", "", cleaned)
    if not cleaned:
        return ""
    canonical = _match_canonical_reason(cleaned)
    if canonical:
        return canonical
    return cleaned if keep_unmatched else ""


def _is_noise_reason_item(text: str) -> bool:
    cleaned = _normalize_text(text).strip()
    if not cleaned:
        return True
    normalized = re.sub(r"[`*_>#\-\s:：]", "", cleaned).lower()
    if normalized in {"结论", "原因", "主要原因", "常见原因", "排查", "说明"}:
        return True
    if cleaned.startswith(">"):
        return True
    return False


def _derive_reason_items_from_text_evidence(evidence_hits: list[dict[str, Any]], *, limit: int = 5) -> list[str]:
    reasons: list[str] = []
    for item in evidence_hits:
        if _normalize_source_type(item.get("source_type")) == "code":
            continue
        section_hint = _normalize_text(item.get("section", "")).lower()
        title_hint = _normalize_text(item.get("title", "")).lower()
        path_hint = str(item.get("path", "")).lower()
        hint_space = " ".join([section_hint, title_hint, path_hint])

        if "胜率下降" in hint_space:
            for reason in ("竞对出价抬升", "两率预估偏低", "pacing过于保守"):
                if reason not in reasons:
                    reasons.append(reason)
        if "成本超标" in hint_space:
            for reason in ("目标配置错误", "回传延迟", "高风险流量抑制缺失"):
                if reason not in reasons:
                    reasons.append(reason)
        if "候选量骤降" in hint_space:
            for reason in ("索引延迟", "过滤条件过严", "特征缺失"):
                if reason not in reasons:
                    reasons.append(reason)
        if "召回时延高" in hint_space:
            for reason in ("并发不足", "超时未降级", "下游依赖抖动"):
                if reason not in reasons:
                    reasons.append(reason)
        if "集体偏高或偏低" in hint_space:
            for reason in ("特征分布漂移", "模型版本误切换", "校准参数同步"):
                if reason not in reasons:
                    reasons.append(reason)
        if "冷启动广告无量" in hint_space:
            for reason in ("探索流量配额", "内容相似召回", "分层保底策略"):
                if reason not in reasons:
                    reasons.append(reason)
        if "精排结果抖动大" in hint_space:
            for reason in ("实时特征波动", "探索权重过高", "多约束冲突"):
                if reason not in reasons:
                    reasons.append(reason)
        if len(reasons) >= limit:
            return reasons[:limit]

    for item in evidence_hits:
        if _normalize_source_type(item.get("source_type")) == "code":
            continue
        excerpt = str(item.get("excerpt", "") or "")
        for raw in _extract_list_items_from_excerpt(excerpt, limit=limit):
            canonical = _canonicalize_reason_item(raw, keep_unmatched=False)
            if canonical and canonical not in reasons:
                reasons.append(canonical)
            if len(reasons) >= limit:
                return reasons[:limit]

        for segment in re.split(r"[。\n；;]", excerpt):
            canonical = _canonicalize_reason_item(segment, keep_unmatched=False)
            if canonical and canonical not in reasons:
                reasons.append(canonical)
            if len(reasons) >= limit:
                return reasons[:limit]
    return reasons[:limit]


def _is_rerank_reason_context(user_query: str, evidence_hits: list[dict[str, Any]]) -> bool:
    normalized_query = str(user_query or "").lower()
    if _contains_any(normalized_query, ("高 ecpm", "高ecpm", "最终位", "未进最终位", "没进最终位")):
        return True

    _ = evidence_hits
    return False


def _derive_reason_items_from_code_evidence(
    evidence_hits: list[dict[str, Any]],
    *,
    user_query: str = "",
    limit: int = 5,
) -> list[str]:
    if not _is_rerank_reason_context(user_query, evidence_hits):
        return []

    merged_excerpt = " ".join(
        _normalize_text(item.get("excerpt", ""))
        for item in evidence_hits
        if _normalize_source_type(item.get("source_type")) == "code"
    ).lower()

    if not merged_excerpt:
        return []

    reasons: list[str] = []
    if "apply_safety_penalty" in merged_excerpt or "freq_penalty" in merged_excerpt:
        reasons.append("触发频控/风控惩罚项后会从基础分中扣分，导致最终 `rank_score` 下降，即使 `eCPM` 较高也可能被后移。")
    if "apply_diversity_penalty" in merged_excerpt or "diversity_penalty" in merged_excerpt:
        reasons.append("触发多样性惩罚时会继续扣减排序分，可能把高 `eCPM` 广告挤出最终位。")
    if "feature_complete" in merged_excerpt and ("0.75" in merged_excerpt or "score *=" in merged_excerpt):
        reasons.append("当特征不完整时会触发降级（例如 `score *= 0.75`），排序分被打折后可能无法进入最终位。")
    if "score = item.ecpm" in merged_excerpt and "rank_score" in merged_excerpt and "sort(" in merged_excerpt:
        reasons.append("最终按综合 `rank_score` 排序而不是仅按 `eCPM`，因此高 `eCPM` 只能作为基础分，仍可能被综合分反超。")
    return reasons[:limit]


def _rewrite_generic_reason_item(item: str) -> str | None:
    cleaned = _normalize_text(item).strip()
    if not cleaned:
        return None

    canonical = _canonicalize_reason_item(cleaned, keep_unmatched=False)
    if canonical:
        return canonical

    lowered = cleaned.lower()
    if re.match(r"^(pctr|pcvr)\s*[：:]", lowered):
        return None

    if _contains_any(cleaned, ("候选广告", "召回来源")):
        return "召回阶段候选质量或来源结构偏弱时，进入精排的候选竞争力不足，最终高 `eCPM` 广告也可能被更优综合分候选替代。"
    if _contains_any(cleaned, ("两率预估", "pctr", "pcvr")):
        return "若 `pCTR/pCVR` 预估偏差导致收益估计失真，会影响后续综合打分，出现高 `eCPM` 广告未进最终位。"
    if _contains_any(cleaned, ("出价结果", "预算状态", "预算")):
        return "预算或出价约束生效时，出价能力被压制会降低有效竞争力，导致高 `eCPM` 广告未能进入最终位。"
    if _contains_any(cleaned, ("业务约束", "频控", "行业约束", "多样性约束")):
        return "命中频控/行业/多样性等约束会触发惩罚或过滤，直接拉低排序分或导致出局。"
    return cleaned


def _refine_reason_items(
    items: list[str],
    evidence_hits: list[dict[str, Any]],
    *,
    user_query: str = "",
    limit: int = 5,
) -> list[str]:
    code_reasons = _derive_reason_items_from_code_evidence(evidence_hits, user_query=user_query, limit=limit)
    text_reasons = _derive_reason_items_from_text_evidence(evidence_hits, limit=limit)
    if len(code_reasons) >= 3:
        return code_reasons[:limit]

    if not code_reasons and not text_reasons:
        # 非 rerank 场景不做“高 eCPM”模板化改写，保留证据原始原因语义。
        deduped: list[str] = []
        for item in items:
            cleaned = _canonicalize_reason_item(item)
            if not cleaned:
                continue
            if _is_noise_reason_item(cleaned):
                continue
            if re.match(r"^(pctr|pcvr)\s*[：:]", cleaned.lower()):
                continue
            if cleaned not in deduped:
                deduped.append(cleaned)
            if len(deduped) >= limit:
                break
        return deduped[:limit]

    refined: list[str] = []
    for reason in [*code_reasons, *text_reasons]:
        if reason not in refined:
            refined.append(reason)
    if len(refined) >= limit:
        return refined[:limit]

    for item in items:
        rewritten = _rewrite_generic_reason_item(item)
        if rewritten and _is_noise_reason_item(rewritten):
            continue
        if rewritten and rewritten not in refined:
            refined.append(rewritten)
        if len(refined) >= limit:
            return refined[:limit]

    return refined[:limit]


def _format_conclusion_bullet(item: str, *, index: int | None = None, label: str | None = None) -> str:
    cleaned = _normalize_text(item)
    cleaned = re.sub(r"^\d+\.\s+", "", cleaned)
    cleaned = re.sub(r"^[-*+]\s+", "", cleaned)
    if index is not None and label:
        cleaned = re.sub(rf"^{re.escape(label)}\s*\d+\s*[:\uFF1A]\s*", "", cleaned).strip()
        return f"- 【{label}{index}】{cleaned}"
    return f"- {cleaned}"


def _normalize_conclusion_list_style(text: str) -> str:
    normalized_lines: list[str] = []
    for line in str(text or "").splitlines():
        normalized_lines.append(re.sub(r"^(\s*)\d+\.\s+", r"\1- ", line))
    return "\n".join(normalized_lines)


def _extract_function_names_from_excerpt(excerpt: str, *, limit: int = 4) -> list[str]:
    names: list[str] = []
    for name in re.findall(r"\bdef\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", str(excerpt or "")):
        if name in names:
            continue
        names.append(name)
        if len(names) >= limit:
            break
    return names


def _build_generic_code_evidence_hint(excerpt: str) -> str:
    normalized = str(excerpt or "").lower()
    actions: list[str] = []

    function_names = _extract_function_names_from_excerpt(excerpt, limit=4)
    if function_names:
        function_list = "、".join(f"`{name}`" for name in function_names)
        actions.append(f"定义了 {function_list} 等关键函数")

    if "score = item.ecpm" in normalized:
        actions.append("以 `eCPM` 作为基础分")
    if "apply_safety_penalty" in normalized or "freq_penalty" in normalized:
        actions.append("扣减频控/风控惩罚项")
    if "apply_diversity_penalty" in normalized or "diversity_penalty" in normalized:
        actions.append("扣减多样性惩罚项")
    if "feature_complete" in normalized and "0.75" in normalized:
        actions.append("在特征缺失时做降级处理")
    if "rank_score" in normalized and ".sort(" in normalized and "reverse=true" in normalized:
        actions.append("按 `rank_score` 倒序排序")
    if "top_n" in normalized and "[:top_n]" in normalized:
        actions.append("截取 TopN 结果输出")
    if "target_cpa" in normalized and "pcvr" in normalized and "alpha" in normalized:
        actions.append("按 `target_cpa * pCVR * alpha` 计算出价")
    if "calibrate_probability" in normalized and ("slope" in normalized or "bias" in normalized):
        actions.append("通过 `slope/bias` 对预测概率做线性校准")

    if actions:
        return "；".join(actions)
    return "该片段定义了关键函数并实现了核心业务分支处理"


def _summarize_code_evidence(item: dict[str, Any]) -> str:
    path = str(item.get("path", "")).strip() or "unknown_path"
    symbol_name = str(item.get("symbol_name", "")).strip() or str(item.get("section", "")).strip() or "unknown_symbol"
    excerpt = str(item.get("excerpt", "")).strip()
    normalized_excerpt = _normalize_text(excerpt)
    hints: list[str] = []

    if f"def {symbol_name}(" in excerpt:
        hints.append(f"这里定义了 `{symbol_name}`")
    if "compute_bid_for_request" in excerpt:
        hints.append("这里出现了请求级出价入口 `compute_bid_for_request`")
    if "compute_ocpc_bid" in excerpt:
        hints.append("这里关联到 `compute_ocpc_bid` 出价计算逻辑")
    if "compute_alpha" in excerpt:
        hints.append("这里会计算调节系数 `alpha`")
    if "target_cpa" in normalized_excerpt and "pcvr" in normalized_excerpt and "alpha" in normalized_excerpt:
        hints.append("这里使用 `target_cpa`、`pCVR` 和 `alpha` 共同计算出价")
    if "predict_ctr_cvr" in excerpt:
        hints.append("这里包含 pCTR/pCVR 预测入口")
    if "calibrate_probability" in excerpt:
        hints.append("这里包含概率校准逻辑")
    if "select_recall_candidates" in excerpt:
        hints.append("这里包含召回候选选择逻辑")
    if "compute_rank_score" in excerpt and "score = item.ecpm" in excerpt:
        hints.append("这里先以 `eCPM` 作为基础分，再扣减频控/风控与多样性惩罚，并在特征缺失时降级，最终得到 `rank_score`")
    if "rank_topn" in excerpt and "compute_rank_score" in excerpt and "sort(" in excerpt:
        hints.append("这里先调用 `compute_rank_score` 计算候选分数，再按 `rank_score` 倒序排序并截取 TopN")
    if not hints and normalized_excerpt:
        hints.append(_build_generic_code_evidence_hint(excerpt))

    summary = "；".join(hints) if hints else "这是与当前问题最相关的实现位置"
    return f"`{path}` 中的 `{symbol_name}`（line: {_line_range(item)}）与结论直接相关：{summary}，因此可直接支撑该结论。"


def _summarize_text_evidence(item: dict[str, Any]) -> str:
    title = str(item.get("title", "")).strip() or str(item.get("path", "")).strip() or "未命名证据"
    section = str(item.get("section", "")).strip()
    excerpt = _safe_text_preview(_normalize_text(item.get("excerpt", "")), max_chars=110)
    if section and excerpt:
        return f"`{title}` 的 `{section}` 提到：{excerpt}"
    if excerpt:
        return f"`{title}` 提到：{excerpt}"
    if section:
        return f"`{title}` 的 `{section}` 与当前结论直接相关。"
    return f"`{title}` 与当前结论直接相关。"


def _evidence_hit_key(item: dict[str, Any]) -> tuple[str, str, str, Any, Any]:
    return (
        _normalize_source_type(item.get("source_type")),
        str(item.get("path", "")).strip().lower(),
        str(item.get("symbol_name", "") or item.get("section", "")).strip().lower(),
        str(item.get("start_line", "")),
        str(item.get("end_line", "")),
    )


def _build_query_domain_markers(user_query: str) -> list[str]:
    normalized = str(user_query or "").lower().strip()
    markers: list[str] = []
    if _contains_any(normalized, ("出价", "target_cpa", "target_roas", "胜率", "成本", "pacing", "ocpc", "tcpa", "bid")):
        markers.extend(["/bid/", "03-出价策略"])
    if _contains_any(normalized, ("pctr", "pcvr", "两率", "校准", "calibrate", "概率")):
        markers.extend(["/rate/", "02-两率预估"])
    if _contains_any(normalized, ("召回", "候选", "latency", "recall")):
        markers.extend(["/recall/", "01-在线召回"])
    if _contains_any(normalized, ("精排", "rerank", "排序", "rank_score", "多样性", "频控", "ecpm", "最终位")):
        markers.extend(["/rerank/", "04-精排策略"])
    deduped: list[str] = []
    for marker in markers:
        if marker not in deduped:
            deduped.append(marker)
    return deduped


def _score_evidence_hit(
    item: dict[str, Any],
    *,
    user_query: str,
    question_type: str,
    preferred_symbols: list[str],
    domain_markers: list[str],
) -> float:
    score = float(item.get("score", 0.0) or 0.0)
    source_type = _normalize_source_type(item.get("source_type"))
    path = str(item.get("path", "")).lower()
    title = str(item.get("title", "")).lower()
    section = str(item.get("section", "")).lower()
    symbol = str(item.get("symbol_name", "") or item.get("section", "")).lower()
    merged_text = " ".join([path, title, section, symbol])

    for marker in domain_markers:
        if marker.lower() in merged_text:
            score += 1.2

    preferred_set = {symbol_name.lower() for symbol_name in preferred_symbols}
    if symbol in preferred_set:
        score += 3.0
    excerpt = str(item.get("excerpt", "") or "").lower()
    for symbol_name in preferred_set:
        if symbol_name and (f"def {symbol_name}(" in excerpt or f"{symbol_name}(" in excerpt):
            score += 1.5
            break

    normalized_query = str(user_query or "").lower().strip()
    # 针对“常见问题章节”做强约束打分，避免被通用公式/核心要素段落抢占。
    if "成本超标" in normalized_query and "成本超标" in merged_text:
        score += 4.0
    if ("胜率下降" in normalized_query or ("胜率" in normalized_query and "原因" in normalized_query)) and "胜率下降" in merged_text:
        score += 4.0
    if ("候选量骤降" in normalized_query or ("召回" in normalized_query and "骤降" in normalized_query)) and "候选量骤降" in merged_text:
        score += 4.0
    if ("召回时延高" in normalized_query or ("召回" in normalized_query and "时延" in normalized_query)) and "召回时延高" in merged_text:
        score += 4.0
    if ("偏高" in normalized_query or "偏低" in normalized_query) and "集体偏高或偏低" in merged_text:
        score += 4.0
    if "冷启动" in normalized_query and "冷启动广告无量" in merged_text:
        score += 4.0
    if "抖动" in normalized_query and "精排结果抖动大" in merged_text:
        score += 4.0

    if source_type == "wiki" and (question_type in {"list", "reason"} or _is_reason_like_query(normalized_query)):
        score += 1.0
    if source_type == "wiki" and _contains_any(normalized_query, ("指标", "口径", "监控", "排查", "常见", "原因")):
        score += 1.0
    if source_type == "code" and _is_code_location_query(normalized_query):
        score += 1.0
    if source_type == "code" and _is_formula_query(normalized_query):
        score += 0.5
    return score


def _select_output_evidence_hits(state: dict[str, Any], evidence_hits: list[dict[str, Any]], *, max_items: int = 4) -> list[dict[str, Any]]:
    if not evidence_hits:
        return []

    analysis = dict(state.get("analysis") or {})
    user_query = str(state.get("user_query", "") or "")
    question_type = str(analysis.get("question_type", "") or "")
    kind = str(state.get("response_kind", state.get("route", "unknown")) or "unknown")
    is_formula = question_type == "formula" or _is_formula_query(user_query)
    preferred_symbols = _build_preferred_symbols_for_query(user_query=user_query, is_formula=is_formula)
    domain_markers = _build_query_domain_markers(user_query)

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, Any, Any]] = set()
    for item in evidence_hits:
        key = _evidence_hit_key(item)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    ranked = sorted(
        deduped,
        key=lambda item: _score_evidence_hit(
            item,
            user_query=user_query,
            question_type=question_type,
            preferred_symbols=preferred_symbols,
            domain_markers=domain_markers,
        ),
        reverse=True,
    )

    need_code = _is_code_location_query(user_query) or _contains_any(user_query, ("代码", "函数", "入口", "路径", "文件"))
    has_wiki_candidates = any(_normalize_source_type(item.get("source_type")) == "wiki" for item in ranked)
    need_wiki = (
        question_type in {"list", "reason"}
        or _is_reason_like_query(user_query)
        or _contains_any(user_query, ("指标", "口径", "监控", "排查", "原因", "哪些点"))
    )
    # 统一输出至少 1 条 wiki 证据（若可用），保持 mixed/wikihit 指标稳定。
    if kind == "knowledge_qa" and has_wiki_candidates:
        need_wiki = True

    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[str, str, str, Any, Any]] = set()
    if need_code:
        for item in ranked:
            if _normalize_source_type(item.get("source_type")) == "code":
                selected.append(item)
                selected_keys.add(_evidence_hit_key(item))
                break
    if need_wiki:
        for item in ranked:
            if _normalize_source_type(item.get("source_type")) == "wiki":
                if _evidence_hit_key(item) not in selected_keys:
                    selected.append(item)
                    selected_keys.add(_evidence_hit_key(item))
                break

    for item in ranked:
        key = _evidence_hit_key(item)
        if key in selected_keys:
            continue
        selected.append(item)
        selected_keys.add(key)
        if len(selected) >= max(1, max_items):
            break

    return selected[: max(1, max_items)]


def _build_evidence_lines(evidence_hits: list[dict[str, Any]], *, max_items: int = 4) -> list[str]:
    if not evidence_hits:
        return ["- 【证据1】当前没有可直接引用的检索证据。"]

    lines: list[str] = []
    for index, item in enumerate(evidence_hits[: max(1, max_items)], start=1):
        source_type = _normalize_source_type(item.get("source_type"))
        summary = _summarize_code_evidence(item) if source_type == "code" else _summarize_text_evidence(item)
        lines.append(f"- 【证据{index}】{summary}")
    return lines


def _is_evidence_insufficient(state: dict[str, Any], evidence_hits: list[dict[str, Any]]) -> bool:
    kind = str(state.get("response_kind", state.get("route", "unknown")) or "unknown")
    analysis = dict(state.get("analysis") or {})
    fallback_reason = str(analysis.get("llm_fallback_reason", "") or "").strip().lower()

    if kind not in {"knowledge_qa", "issue_analysis"}:
        return False
    if fallback_reason == "no_evidence_hits":
        return True
    if not evidence_hits:
        return True
    return False


def _build_insufficient_evidence_conclusion(state: dict[str, Any]) -> str:
    analysis = dict(state.get("analysis") or {})
    module_name = analysis.get("module") or state.get("module_name") or "--"
    lines = ["当前证据不足，暂时无法给出明确结论。"]
    lines.append(f"- 当前问题主要落在 `{module_name}`。")
    lines.append("- 建议补充更具体的信息后重试。")
    return "\n".join(lines)


def _build_system_judgement_markdown(state: dict[str, Any], *, evidence_count: int) -> str:
    kind = str(state.get("response_kind", state.get("route", "unknown")) or "unknown")
    analysis = dict(state.get("analysis") or {})
    module_name = analysis.get("module") or state.get("module_name") or "--"
    module_hint = state.get("module_hint") or "--"
    generation_mode = analysis.get("generation_mode") or analysis.get("llm_model")

    rows = [
        f"- 场景判定：{SCENE_LABELS.get(kind, kind)}（`{kind}`）",
        f"- 当前模块：`{module_name}`",
        f"- 当前阶段：`{state.get('task_stage', 'unknown')}`",
        f"- 转场类型：`{state.get('transition_type', 'unknown')}`",
        f"- 输出状态：`{state.get('status', 'completed')}`",
        f"- 下一步动作：`{state.get('next_action', 'completed')}`",
        f"- 证据数量：`{evidence_count}`",
    ]
    if module_hint and module_hint != "--":
        rows.append(f"- 模块说明：{module_hint}")
    if generation_mode:
        rows.append(f"- 生成方式：`{generation_mode}`")
    fallback_reason = analysis.get("llm_fallback_reason")
    if fallback_reason:
        rows.append(f"- 降级原因：`{fallback_reason}`")
    return "### 系统判定结论\n" + "\n".join(rows)


def _build_knowledge_qa_conclusion(state: dict[str, Any], evidence_hits: list[dict[str, Any]]) -> str:
    analysis = dict(state.get("analysis") or {})
    module_name = analysis.get("module") or state.get("module_name") or "--"
    module_hint = state.get("module_hint") or ""
    raw_answer = str(state.get("answer", "") or "").strip()
    conclusion_body = _extract_conclusion_body(raw_answer)
    question_type = str(analysis.get("question_type", "") or "")
    user_query = str(state.get("user_query", "") or "")
    is_code_location = _is_code_location_query(user_query)
    is_formula = question_type == "formula" or _is_formula_query(user_query)
    is_reason_like = (question_type == "reason" or _is_reason_like_query(user_query)) and not is_formula

    if is_code_location and not is_reason_like:
        code_hits = [item for item in evidence_hits if _normalize_source_type(item.get("source_type")) == "code"]
        if code_hits:
            primary, primary_symbol_name = _resolve_code_anchor(code_hits, user_query=user_query, is_formula=is_formula)
            path = str(primary.get("path", "")).strip() or "unknown_path"
            symbol_name = primary_symbol_name or (
                str(primary.get("symbol_name", "")).strip() or str(primary.get("section", "")).strip() or "unknown_symbol"
            )
            if is_formula:
                formula_text = _extract_formula_from_answer_or_evidence(
                    conclusion_body=conclusion_body,
                    raw_answer=raw_answer,
                    evidence_hits=evidence_hits,
                    user_query=user_query,
                )
                lines: list[str] = []
                if formula_text:
                    lines.append(f"结合当前证据，出价公式为：`{formula_text}`。")
                else:
                    lines.append("结合当前证据，出价公式可在下述实现函数中确认。")
                lines.append(f"代码实现函数优先定位到 `{path}` -> `{symbol_name}`（line: {_line_range(primary)}）。")

                if _is_bid_entry_or_chain_query(user_query):
                    entry_hit = _find_code_hit_by_symbol(code_hits, "compute_bid_for_request")
                    alpha_hit = _find_code_hit_by_symbol(code_hits, "compute_alpha")
                    ocpc_hit = _find_code_hit_by_symbol(code_hits, "compute_ocpc_bid")
                    if entry_hit is not None:
                        entry_path = str(entry_hit.get("path", "")).strip() or "unknown_path"
                        lines.append(
                            f"请求级出价入口函数为 `{entry_path}` -> `compute_bid_for_request`（line: {_line_range(entry_hit)}）。"
                        )
                    else:
                        lines.append("请求级出价入口函数为 `compute_bid_for_request`。")
                    if _contains_any(user_query, ("函数链路", "函数调用链", "串联计算")) and alpha_hit is not None and ocpc_hit is not None:
                        lines.append("函数链路为：`compute_bid_for_request -> compute_alpha -> compute_ocpc_bid`。")
                merged_excerpt = " ".join(
                    _normalize_text(item.get("excerpt", ""))
                    for item in evidence_hits
                    if _normalize_source_type(item.get("source_type")) == "code"
                ).lower()
                if "alpha" in user_query.lower() and "budget_ratio" in merged_excerpt and "competition_level" in merged_excerpt:
                    lines.append("`alpha` 根据 `budget_ratio` 与 `competition_level` 分段计算：预算超前会下调，预算落后会提升。")
                if _contains_any(user_query, ("截断", "0 到 1", "0到1")) and "0.0" in merged_excerpt and "1.0" in merged_excerpt:
                    lines.append("校准后概率会截断到 `[0.0, 1.0]` 区间，避免概率越界。")
            else:
                lines = [f"当前问题可以优先定位到 `{path}` -> `{symbol_name}`（line: {_line_range(primary)}）。"]

            primary_key = (
                str(primary.get("path", "")).strip(),
                str(primary.get("symbol_name", "")).strip() or str(primary.get("section", "")).strip(),
                primary.get("start_line"),
                primary.get("end_line"),
            )
            extra_items = []
            for item in code_hits:
                item_key = (
                    str(item.get("path", "")).strip(),
                    str(item.get("symbol_name", "")).strip() or str(item.get("section", "")).strip(),
                    item.get("start_line"),
                    item.get("end_line"),
                )
                if item_key == primary_key:
                    continue
                extra_path = str(item.get("path", "")).strip() or "unknown_path"
                extra_symbol = str(item.get("symbol_name", "")).strip() or str(item.get("section", "")).strip() or "unknown_symbol"
                extra_items.append(f"可继续关注 `{extra_path}` -> `{extra_symbol}`（line: {_line_range(item)}）")
                if len(extra_items) >= 2:
                    break
            if extra_items:
                lines.append("")
                lines.append("- 主定位实现见上面的优先位置。")
                for item in extra_items:
                    lines.append(f"- {item}")
            return "\n".join(lines)

    if is_formula:
        formula_text = _extract_formula_from_answer_or_evidence(
            conclusion_body=conclusion_body,
            raw_answer=raw_answer,
            evidence_hits=evidence_hits,
            user_query=user_query,
        )
        formula_lines: list[str] = []
        if formula_text:
            formula_lines.append(f"结合当前证据，公式为：`{formula_text}`。")

        code_hits = [item for item in evidence_hits if _normalize_source_type(item.get("source_type")) == "code"]
        if code_hits and (
            _contains_any(user_query, ("代码", "实现", "函数", "路径", "位置", "在哪"))
            or "ecpm" in user_query.lower()
        ):
            primary, primary_symbol_name = _resolve_code_anchor(code_hits, user_query=user_query, is_formula=True)
            path = str(primary.get("path", "")).strip() or "unknown_path"
            symbol_name = primary_symbol_name or (
                str(primary.get("symbol_name", "")).strip() or str(primary.get("section", "")).strip() or "unknown_symbol"
            )
            formula_lines.append(f"代码实现可优先参考 `{path}` -> `{symbol_name}`（line: {_line_range(primary)}）。")

            if "ecpm" in user_query.lower():
                rank_hit = _find_code_hit_by_symbol(code_hits, "compute_rank_score")
                if rank_hit is not None:
                    rank_path = str(rank_hit.get("path", "")).strip() or "unknown_path"
                    formula_lines.append(
                        f"在排序侧可定位到 `{rank_path}` -> `compute_rank_score`（line: {_line_range(rank_hit)}），该函数产出 `rank_score`。"
                    )

        merged_excerpt = " ".join(
            _normalize_text(item.get("excerpt", ""))
            for item in evidence_hits
            if _normalize_source_type(item.get("source_type")) == "code"
        ).lower()
        if "alpha" in user_query.lower() and "budget_ratio" in merged_excerpt and "competition_level" in merged_excerpt:
            formula_lines.append("`alpha` 的计算会综合 `budget_ratio` 与 `competition_level`。")
        if _contains_any(user_query, ("截断", "0 到 1", "0到1")) and "0.0" in merged_excerpt and "1.0" in merged_excerpt:
            formula_lines.append("校准后概率会被截断在 `[0.0, 1.0]` 区间。")

        if formula_lines:
            return "\n".join(formula_lines)

    if question_type in {"list", "reason"} or is_reason_like:
        summary, inline_items = _split_summary_and_inline_items(conclusion_body, limit=5)
        if inline_items:
            if is_reason_like:
                evidence_reason_items = _derive_list_items_from_evidence(evidence_hits, limit=5)
                if evidence_reason_items:
                    inline_items = evidence_reason_items
                inline_items = _refine_reason_items(inline_items, evidence_hits, user_query=user_query, limit=5)
            elif question_type == "list" and _contains_any(user_query, ("指标", "口径", "监控")):
                evidence_list_items = _derive_list_items_from_evidence(evidence_hits, limit=5)
                if evidence_list_items:
                    inline_items = evidence_list_items
            lines = [summary or "结合当前证据，答案可以归纳为以下要点："]
            if is_reason_like:
                lines = [summary or "结合当前证据，主要原因如下："]
            lines.append("")
            for index, item in enumerate(inline_items, start=1):
                if is_reason_like:
                    lines.append(_format_conclusion_bullet(item, index=index, label="原因"))
                else:
                    lines.append(_format_conclusion_bullet(item, index=index, label="要点"))
            if is_reason_like and _contains_any(user_query, ("代码", "函数", "入口", "路径", "文件")):
                code_hits = [item for item in evidence_hits if _normalize_source_type(item.get("source_type")) == "code"]
                if code_hits:
                    primary, primary_symbol_name = _resolve_code_anchor(code_hits, user_query=user_query, is_formula=False)
                    path = str(primary.get("path", "")).strip() or "unknown_path"
                    symbol_name = primary_symbol_name or (
                        str(primary.get("symbol_name", "")).strip() or str(primary.get("section", "")).strip() or "unknown_symbol"
                    )
                    lines.append("")
                    lines.append(f"- 相关代码入口可优先查看 `{path}` -> `{symbol_name}`（line: {_line_range(primary)}）。")
                    if _contains_any(user_query, ("校准", "calibrate")):
                        calibrate_hit = _find_code_hit_by_symbol(code_hits, "calibrate_probability")
                        if calibrate_hit is not None:
                            calibrate_path = str(calibrate_hit.get("path", "")).strip() or "unknown_path"
                            lines.append(
                                f"- 校准函数可定位到 `{calibrate_path}` -> `calibrate_probability`（line: {_line_range(calibrate_hit)}）。"
                            )
                    if _is_bid_entry_or_chain_query(user_query):
                        entry_hit = _find_code_hit_by_symbol(code_hits, "compute_bid_for_request")
                        if entry_hit is not None:
                            entry_path = str(entry_hit.get("path", "")).strip() or "unknown_path"
                            lines.append(
                                f"- 请求级出价入口可定位到 `{entry_path}` -> `compute_bid_for_request`（line: {_line_range(entry_hit)}）。"
                            )
                        else:
                            lines.append("- 请求级出价入口函数为 `compute_bid_for_request`。")
            return "\n".join(lines)

        items = _derive_list_items_from_evidence(evidence_hits, limit=5)
        if not items:
            items = _extract_list_items(conclusion_body, limit=5)
        items = [item for item in items if not item.startswith("代码位置") and not item.startswith("代码摘要")]
        if is_reason_like:
            items = _refine_reason_items(items, evidence_hits, user_query=user_query, limit=5)
        if items:
            if is_reason_like:
                summary = "结合当前证据，主要原因如下："
            else:
                summary = "结合当前证据，答案可以归纳为以下要点："
            lines = [summary]
            lines.append("")
            for index, item in enumerate(items, start=1):
                if is_reason_like:
                    lines.append(_format_conclusion_bullet(item, index=index, label="原因"))
                else:
                    lines.append(_format_conclusion_bullet(item, index=index, label="要点"))
            if is_reason_like and _contains_any(user_query, ("代码", "函数", "入口", "路径", "文件")):
                code_hits = [item for item in evidence_hits if _normalize_source_type(item.get("source_type")) == "code"]
                if code_hits:
                    primary, primary_symbol_name = _resolve_code_anchor(code_hits, user_query=user_query, is_formula=False)
                    path = str(primary.get("path", "")).strip() or "unknown_path"
                    symbol_name = primary_symbol_name or (
                        str(primary.get("symbol_name", "")).strip() or str(primary.get("section", "")).strip() or "unknown_symbol"
                    )
                    lines.append("")
                    lines.append(f"- 相关代码入口可优先查看 `{path}` -> `{symbol_name}`（line: {_line_range(primary)}）。")
                    if _contains_any(user_query, ("校准", "calibrate")):
                        calibrate_hit = _find_code_hit_by_symbol(code_hits, "calibrate_probability")
                        if calibrate_hit is not None:
                            calibrate_path = str(calibrate_hit.get("path", "")).strip() or "unknown_path"
                            lines.append(
                                f"- 校准函数可定位到 `{calibrate_path}` -> `calibrate_probability`（line: {_line_range(calibrate_hit)}）。"
                            )
                    if _is_bid_entry_or_chain_query(user_query):
                        entry_hit = _find_code_hit_by_symbol(code_hits, "compute_bid_for_request")
                        if entry_hit is not None:
                            entry_path = str(entry_hit.get("path", "")).strip() or "unknown_path"
                            lines.append(
                                f"- 请求级出价入口可定位到 `{entry_path}` -> `compute_bid_for_request`（line: {_line_range(entry_hit)}）。"
                            )
                        else:
                            lines.append("- 请求级出价入口函数为 `compute_bid_for_request`。")
            return "\n".join(lines)

    return _extract_question_conclusion(conclusion_body or raw_answer, module_name=module_name, module_hint=module_hint)


def _build_issue_analysis_conclusion(state: dict[str, Any]) -> str:
    analysis = dict(state.get("analysis") or {})
    module_name = analysis.get("module") or state.get("module_name") or "--"
    root_cause = _normalize_text(analysis.get("root_cause"))
    fix_plan = list(analysis.get("fix_plan", []) or [])

    if root_cause:
        lines = [f"当前问题更可能定位在 `{module_name}`，初步根因是：{root_cause}"]
        if fix_plan:
            lines.append("")
            for item in fix_plan[:3]:
                lines.append(f"- 建议优先处理：{_normalize_text(item)}")
        return "\n".join(lines)

    raw_answer = str(state.get("answer", "") or "").strip()
    return raw_answer or f"当前问题更可能定位在 `{module_name}`。"


def _build_code_generation_conclusion(state: dict[str, Any]) -> str:
    analysis = dict(state.get("analysis") or {})
    module_name = analysis.get("module") or state.get("module_name") or "--"
    patch_summary = list(analysis.get("patch_summary", []) or [])

    if not patch_summary:
        return f"已进入 `{module_name}` 的代码实现建议阶段。"

    lines = [f"已为 `{module_name}` 形成可落地的代码实现建议，当前优先改动点是：{_normalize_text(patch_summary[0])}"]
    if len(patch_summary) > 1:
        lines.append("")
        for item in patch_summary[:3]:
            lines.append(f"- {_normalize_text(item)}")
    return "\n".join(lines)


def _build_conversation_control_conclusion(state: dict[str, Any]) -> str:
    analysis = dict(state.get("analysis") or {})
    module_name = analysis.get("module") or state.get("module_name") or "--"
    return f"已保留 `{module_name}` 的问题分析结论，本轮不进入代码生成。"


def _build_out_of_scope_conclusion() -> str:
    return "当前问题未被判定为本系统支持的业务问题，未进入后续知识问答、问题分析或代码生成链路。"


def _build_question_conclusion_markdown(state: dict[str, Any], evidence_hits: list[dict[str, Any]]) -> str:
    if _is_evidence_insufficient(state, evidence_hits):
        conclusion = _strip_conclusion_prefix(_build_insufficient_evidence_conclusion(state))
        return _normalize_conclusion_list_style(conclusion)

    kind = str(state.get("response_kind", state.get("route", "unknown")) or "unknown")
    if kind == "knowledge_qa":
        conclusion = _strip_conclusion_prefix(_build_knowledge_qa_conclusion(state, evidence_hits))
        return _normalize_conclusion_list_style(conclusion)
    if kind == "issue_analysis":
        conclusion = _strip_conclusion_prefix(_build_issue_analysis_conclusion(state))
        return _normalize_conclusion_list_style(conclusion)
    if kind == "code_generation":
        conclusion = _strip_conclusion_prefix(_build_code_generation_conclusion(state))
        return _normalize_conclusion_list_style(conclusion)
    if kind == "conversation_control":
        conclusion = _strip_conclusion_prefix(_build_conversation_control_conclusion(state))
        return _normalize_conclusion_list_style(conclusion)
    if kind == "out_of_scope":
        conclusion = _strip_conclusion_prefix(_build_out_of_scope_conclusion())
        return _normalize_conclusion_list_style(conclusion)
    conclusion = _strip_conclusion_prefix(str(state.get("answer", "") or "").strip())
    return _normalize_conclusion_list_style(conclusion)


def _render_markdown_answer(service: Any, state: dict[str, Any], *, evidence_hits: list[dict[str, Any]]) -> str:
    debug_enabled = bool(getattr(service, "debug_verbose_enabled", False))
    evidence_insufficient = _is_evidence_insufficient(state, evidence_hits)
    sections: list[str] = []

    if debug_enabled:
        sections.append(_build_system_judgement_markdown(state, evidence_count=len(evidence_hits)))

    sections.append(f"### 问题结论\n{_build_question_conclusion_markdown(state, evidence_hits)}")
    if not evidence_insufficient:
        sections.append(f"### 证据说明\n" + "\n".join(_build_evidence_lines(evidence_hits, max_items=4)))
    return "\n\n".join(section.strip() for section in sections if section.strip())


def run(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    graph_path = [item["node"] for item in state.get("node_trace", [])]
    debug_enabled = bool(getattr(service, "debug_verbose_enabled", False))
    all_evidence_hits = _collect_evidence_hits(state)
    evidence_hits = _select_output_evidence_hits(state, all_evidence_hits, max_items=4)
    evidence_insufficient = _is_evidence_insufficient(state, evidence_hits)

    assistant_message: dict[str, Any] = {
        "role": "assistant",
        "kind": state.get("response_kind", state.get("route", "unknown")),
        "intent": state.get("route", "unknown"),
        "status": state.get("status", "completed"),
        "content": _render_markdown_answer(service, state, evidence_hits=evidence_hits),
        "trace_id": state["trace_id"],
        "citations": [] if evidence_insufficient else evidence_hits,
        "analysis": state.get("analysis"),
        "actions": (
            [{"type": "confirm_code_generation", "label": "需要代码实现"}]
            if state.get("next_action") == "confirm_code"
            else []
        ),
    }

    if debug_enabled:
        assistant_message["debug"] = {
            "domain_relevance": state.get("domain_relevance", 0.0),
            "latency_ms": 0,
            "route": state.get("route", "unknown"),
            "task_stage": state.get("task_stage", "unknown"),
            "active_task_stage": state.get("active_task_stage", "idle"),
            "transition_type": state.get("transition_type", "unknown"),
            "execution_path": state.get("execution_path", "unknown"),
            "next_action": state.get("next_action", "completed"),
            "graph_backend": service.backend_name,
            "graph_path": graph_path,
            "verbose": _build_verbose_debug(state, graph_path),
        }

    return {
        "assistant_message": assistant_message,
        "node_trace": service._trace(state, "finalize_response", f"kind={assistant_message['kind']}"),
    }
