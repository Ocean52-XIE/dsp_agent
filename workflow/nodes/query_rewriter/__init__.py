from __future__ import annotations

"""检索改写与多源路由规划节点。"""

from typing import Any


SYNONYM_EXPANSION_MAP: dict[str, tuple[str, ...]] = {
    "在线投放": ("广告投放", "在线链路"),
    "核心链路": ("核心流程", "主链路"),
    "指标": ("关键指标", "监控指标", "业务指标"),
    "口径": ("定义", "统计口径", "计算口径"),
    "召回": ("候选召回", "候选生成"),
    "两率预估": ("ctr cvr 预估", "点击率转化率预估"),
    "出价": ("竞价", "bid", "出价策略"),
    "精排": ("排序", "重排", "rerank"),
    "排障": ("排查", "故障定位", "问题定位"),
}

ABBR_EXPANSION_MAP: dict[str, str] = {
    "ctr": "点击率",
    "cvr": "转化率",
    "pctr": "预估点击率",
    "pcvr": "预估转化率",
    "ecpm": "千次展示收益",
    "cpa": "单次转化成本",
    "roas": "广告投入产出比",
}

# 函数定位类关键词（P0）：
# 用于识别“请给我函数名/文件路径/实现位置”这类强代码定位查询，
# 这类问题在融合阶段应明显提升 code 侧权重，避免被 wiki 总览文档干扰。
CODE_LOCATION_KEYWORDS: tuple[str, ...] = (
    "哪个函数",
    "什么函数",
    "函数名",
    "入口函数",
    "实现函数",
    "文件路径",
    "哪个文件",
    "在哪个文件",
    "哪一行",
    "行号",
    "路径在哪",
    "定位",
    "定义在哪里",
    "where",
    "which function",
    "function",
)

# 项目内高频函数锚点词典（P0）：
# 目标是让 query_rewriter 在“符号名直接问法”下快速识别为 code-location 场景。
CODE_SYMBOL_HINTS: tuple[str, ...] = (
    "compute_bid_for_request",
    "compute_ocpc_bid",
    "compute_alpha",
    "predict_ctr_cvr",
    "calibrate_probability",
    "select_recall_candidates",
    "compute_rank_score",
    "rank_topn",
    "apply_diversity_penalty",
    "apply_safety_penalty",
)

# 中文短语到代码符号的别名映射（P0 多样性惩罚专项）：
# 目标：当用户用中文问“多样性惩罚/安全惩罚/排序函数”时，
# 检索 query 中仍能出现明确函数名，提高 code pattern 命中率。
SYMBOL_ALIAS_MAP: dict[str, tuple[str, ...]] = {
    "多样性惩罚": ("apply_diversity_penalty", "rerank_engine.py"),
    "安全惩罚": ("apply_safety_penalty", "rerank_engine.py"),
    "频控惩罚": ("apply_safety_penalty", "rerank_engine.py"),
    "排序函数": ("compute_rank_score", "rank_topn", "rerank_engine.py"),
    "最终排序分": ("compute_rank_score", "rerank_engine.py"),
    "请求级出价入口": ("compute_bid_for_request", "bid_optimizer.py"),
    "出价入口函数": ("compute_bid_for_request", "bid_optimizer.py"),
    "出价入口": ("compute_bid_for_request", "bid_optimizer.py"),
    "两率预测入口": ("predict_ctr_cvr", "rate_predictor.py"),
    "预测入口函数": ("predict_ctr_cvr", "rate_predictor.py"),
}


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    """判断文本是否命中任一关键词。"""
    return any(keyword in text for keyword in keywords)


def _infer_code_location_intent(normalized_query: str) -> bool:
    """识别“函数/文件定位”强意图。

    设计说明：
    1. 单纯出现“代码/实现”不等于函数定位，因此这里单独抽一个更严格的判定；
    2. 只要命中“定位表达”或“已知符号名”，就将本轮问题视为 code-location；
    3. 该信号会直接影响后续 strategy/source_weights/max_per_source。
    """
    if _contains_any(normalized_query, CODE_LOCATION_KEYWORDS):
        return True
    return _contains_any(normalized_query, CODE_SYMBOL_HINTS)


def _infer_query_flags(user_query: str) -> dict[str, bool]:
    """根据用户原始问题识别检索意图，用于收敛 query 构造。"""
    normalized = user_query.lower().strip()
    code_location = _infer_code_location_intent(normalized)
    return {
        "metric": _contains_any(normalized, ("指标", "口径", "监控", "统计", "ctr", "cvr", "ecpm", "roi", "winrate")),
        "pipeline": _contains_any(normalized, ("链路", "流程", "阶段", "环节", "步骤", "顺序")),
        "architecture": _contains_any(normalized, ("架构", "总体", "全景", "模块关系")),
        # 排障意图扩展：覆盖“成本/胜率/冷启动/掉量”等高频问题描述，
        # 让这类查询稳定触发 troubleshoot 分支。
        "troubleshoot": _contains_any(
            normalized,
            (
                "排障",
                "排查",
                "故障",
                "异常",
                "无量",
                "抖动",
                "定位",
                "成本",
                "超标",
                "胜率",
                "冷启动",
                "掉量",
                "没量",
                "无消耗",
                "不稳定",
                "联调",
                "实践建议",
                "手册",
            ),
        ),
        "code": _contains_any(
            normalized,
            ("代码", "函数", "类", "实现", "定义", "在哪个文件", "哪一行", "trace", "stack", "参数", "how to fix"),
        ),
        # 与“code”区分：code_location 仅代表“强定位诉求”，用于强收敛路由。
        "code_location": code_location,
    }


def _module_alias_queries(module_name: str) -> list[str]:
    """将模块英文名映射成业务语义查询，补充中文语料召回。"""
    mapping: dict[str, list[str]] = {
        "ad-recall": ["在线召回", "召回阶段 候选生成"],
        "rate-prediction": ["两率预估", "pctr pcvr 预估校准"],
        "bid-optimizer": ["出价策略", "ocpc tcpa 出价"],
        "rerank-engine": ["精排策略", "重排 多目标排序"],
        "traffic-governor": ["频控风控", "流量治理 频次控制"],
        "ad-serving-orchestrator": ["广告在线投放", "核心链路 架构流程"],
    }
    return mapping.get(module_name, [])


def _expand_queries(base_queries: list[str], *, original_user_query: str) -> list[str]:
    """对基础 query 做受控扩展，避免模板 query 触发级联扩展。"""
    merged = " ".join(base_queries).lower()
    original = original_user_query.lower().strip()
    expanded = list(base_queries)

    for anchor, aliases in SYNONYM_EXPANSION_MAP.items():
        trigger_terms = (anchor, *aliases)
        # 仅允许原始用户问句触发同义词扩展，减少“扩展再扩展”的噪声。
        if any(term in original for term in trigger_terms):
            expanded.extend([anchor, *aliases])

    for abbr, canonical in ABBR_EXPANSION_MAP.items():
        if abbr in merged:
            expanded.append(canonical)

    deduped: list[str] = []
    seen: set[str] = set()
    for query in expanded:
        normalized = " ".join(query.strip().split())
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped[:16]


def _expand_symbol_alias_queries(user_query: str) -> list[str]:
    """把中文代码语义短语扩展成函数符号 query。"""
    lowered = user_query.lower().strip()
    expanded: list[str] = []
    for phrase, aliases in SYMBOL_ALIAS_MAP.items():
        if phrase in lowered:
            expanded.extend(list(aliases))
    return expanded


def _expand_reason_section_queries(user_query: str) -> list[str]:
    """为“原因排查类”问题补充章节级检索锚点，提升 wiki 原因证据命中率。"""
    lowered = user_query.lower().strip()
    expanded: list[str] = []

    if "胜率" in lowered and _contains_any(lowered, ("原因", "下降", "排查", "导致")):
        expanded.append("出价策略 6.1 胜率下降 竞对出价抬升 两率预估偏低 pacing过于保守")
        expanded.append("compute_bid_for_request compute_alpha bid_optimizer.py")
    if "成本超标" in lowered or ("成本" in lowered and "超标" in lowered):
        expanded.append("出价策略 6.2 成本超标 目标配置错误 回传延迟 高风险流量抑制")
        expanded.append("compute_alpha compute_ocpc_bid bid_optimizer.py")
    if "候选量骤降" in lowered or ("召回" in lowered and "骤降" in lowered):
        expanded.append("在线召回 4.1 候选量骤降 索引延迟 过滤条件过严 特征缺失")
    if "召回时延高" in lowered or ("召回" in lowered and "时延" in lowered):
        expanded.append("在线召回 4.2 召回时延高 并发不足 超时未降级 下游依赖抖动")
    if any(token in lowered for token in ("偏高", "偏低")) and any(token in lowered for token in ("pctr", "pcvr", "ctr", "cvr")):
        expanded.append("两率预估 5.1 pCTR pCVR 集体偏高或偏低 特征分布漂移 模型版本误切换 校准参数同步")
    if "冷启动" in lowered and "无量" in lowered:
        expanded.append("两率预估 5.2 冷启动广告无量 探索流量配额 内容相似召回 分层保底策略")
        expanded.append("select_recall_candidates rank_topn recall_service.py rerank_engine.py")
    if "精排结果抖动" in lowered or ("精排" in lowered and "抖动" in lowered):
        expanded.append("精排策略 6.1 精排结果抖动大 实时特征波动 探索权重过高 多约束冲突")
    return expanded


def _expand_metric_section_queries(user_query: str) -> list[str]:
    """为“指标+定位”问题补充章节级指标锚点。"""
    lowered = user_query.lower().strip()
    expanded: list[str] = []
    if "召回" in lowered and any(token in lowered for token in ("指标", "质量")):
        expanded.append("在线召回 3. 召回质量指标 Recall@N CandidateCount RecallLatencyP99 FilterDropRate")
    if "精排" in lowered and any(token in lowered for token in ("指标", "监控")):
        expanded.append("精排策略 5. 关键监控 排序时延 P95 P99 TopN 点击率 多样性指标 探索流量占比")
    return expanded


def _build_retrieval_plan(
    *,
    user_query: str,
    route: str,
    transition_type: str,
    query_flags: dict[str, bool],
) -> dict[str, Any]:
    """根据问题语义构建多源检索计划。"""
    normalized = user_query.lower().strip()
    is_code_intent = bool(query_flags.get("code")) or _contains_any(
        normalized,
        ("代码", "函数", "类", "实现", "定义", "在哪个文件", "哪一行", "异常", "报错", "trace", "stack", "参数", "how to fix"),
    )
    is_code_location = bool(query_flags.get("code_location"))
    is_reason_query = _contains_any(normalized, ("原因", "为什么", "为何", "导致", "常见", "排查"))
    is_reason_with_location = is_reason_query and is_code_location
    is_metric_query = _contains_any(normalized, ("指标", "口径", "监控", "recall@", "candidatecount", "latency", "p99"))
    is_metric_with_location = is_metric_query and is_code_location
    is_wiki_intent = bool(query_flags.get("architecture") or query_flags.get("pipeline") or query_flags.get("metric"))
    if not is_wiki_intent:
        is_wiki_intent = _contains_any(normalized, ("架构", "流程", "阶段", "指标", "口径", "规则", "原理", "背景", "说明", "为什么"))
    is_issue_analysis = route == "issue_analysis"

    plan: dict[str, Any] = {
        "strategy": "hybrid",
        "enable_wiki": True,
        "enable_code": True,
        "enable_cases": False,
        "wiki_top_k": 4,
        "code_top_k": 4,
        "case_top_k": 2,
        "final_top_k": 6,
        "max_per_source": {"wiki": 4, "code": 4, "case": 1},
        "source_weights": {"wiki": 1.0, "code": 1.0, "case": 0.6},
        # 将当前轮意图标签透传到后续节点，避免重复猜测语义。
        "intent_profile": {
            "is_code_intent": is_code_intent,
            "is_code_location": is_code_location,
            "is_wiki_intent": is_wiki_intent,
            "is_issue_analysis": is_issue_analysis,
        },
        "reasons": [],
    }

    if is_issue_analysis:
        plan["code_top_k"] = 6
        plan["wiki_top_k"] = 3
        plan["source_weights"]["code"] = 1.15
        plan["source_weights"]["wiki"] = 0.95
        plan["reasons"].append("issue_analysis 场景提升 code 召回与融合权重")

    # P0：函数/文件定位类问题直接走 code_first（优先级高于普通 code_intent）。
    if is_reason_with_location or is_metric_with_location:
        plan["strategy"] = "hybrid"
        plan["code_top_k"] = max(int(plan["code_top_k"]), 5)
        plan["wiki_top_k"] = max(int(plan["wiki_top_k"]), 5)
        plan["source_weights"]["code"] = max(float(plan["source_weights"]["code"]), 1.15)
        plan["source_weights"]["wiki"] = max(float(plan["source_weights"]["wiki"]), 1.15)
        plan["max_per_source"]["code"] = max(int(plan["max_per_source"]["code"]), 4)
        plan["max_per_source"]["wiki"] = max(int(plan["max_per_source"]["wiki"]), 4)
        if is_reason_with_location:
            plan["reasons"].append("检测到原因解释+函数定位混合诉求，提升 wiki/code 双侧召回")
        else:
            plan["reasons"].append("检测到指标问答+函数定位混合诉求，提升 wiki/code 双侧召回")
    elif is_code_location:
        plan["strategy"] = "code_first"
        plan["code_top_k"] = max(int(plan["code_top_k"]), 7)
        plan["wiki_top_k"] = min(int(plan["wiki_top_k"]), 2)
        plan["source_weights"]["code"] = max(float(plan["source_weights"]["code"]), 1.35)
        plan["source_weights"]["wiki"] = min(float(plan["source_weights"]["wiki"]), 0.8)
        # 配额层面也收紧 wiki，防止融合阶段出现 wiki 刷屏。
        plan["max_per_source"]["code"] = max(int(plan["max_per_source"]["code"]), 5)
        plan["max_per_source"]["wiki"] = min(int(plan["max_per_source"]["wiki"]), 2)
        plan["reasons"].append("检测到函数/文件定位意图，强制采用 code_first")
    elif is_code_intent and not is_wiki_intent:
        plan["strategy"] = "code_first"
        plan["code_top_k"] = max(int(plan["code_top_k"]), 6)
        plan["wiki_top_k"] = min(int(plan["wiki_top_k"]), 2)
        plan["source_weights"]["code"] = max(float(plan["source_weights"]["code"]), 1.2)
        plan["source_weights"]["wiki"] = min(float(plan["source_weights"]["wiki"]), 0.9)
        plan["reasons"].append("查询语义偏代码定位，采用 code_first")
    elif is_wiki_intent and not is_code_intent:
        plan["strategy"] = "wiki_first"
        plan["wiki_top_k"] = max(int(plan["wiki_top_k"]), 6)
        plan["code_top_k"] = min(int(plan["code_top_k"]), 2)
        plan["source_weights"]["wiki"] = max(float(plan["source_weights"]["wiki"]), 1.15)
        plan["source_weights"]["code"] = min(float(plan["source_weights"]["code"]), 0.9)
        plan["reasons"].append("查询语义偏业务知识，采用 wiki_first")
    else:
        plan["strategy"] = "hybrid"
        plan["reasons"].append("查询语义混合，采用多源并行召回")

    if transition_type == "upgrade_from_qa_to_issue_analysis":
        plan["source_weights"]["code"] = max(float(plan["source_weights"]["code"]), 1.25)
        plan["reasons"].append("从 QA 升级为排障，进一步提升 code 权重")
    return plan


def run(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """生成检索查询语句，并输出多源路由计划。"""
    module_name = state["module_name"]
    user_query = state["user_query"]
    lowered_query = user_query.lower().strip()
    route = state.get("route", "knowledge_qa")
    transition_type = state.get("transition_type", "start_knowledge_qa")
    flags = _infer_query_flags(user_query)

    if route == "issue_analysis":
        base_queries = [
            user_query,
            f"{module_name} 异常日志",
            f"{module_name} 排查案例",
            f"{module_name} 联调排障",
        ]
        # 当用户明确问到链路阶段时，再补模块别名，避免把“架构总览”固定推到前面。
        if flags["pipeline"] or flags["architecture"]:
            base_queries.extend(_module_alias_queries(module_name))
        if transition_type == "upgrade_from_qa_to_issue_analysis":
            base_queries.append(f"{module_name} 指标口径偏差 故障定位")
        if transition_type == "code_request_without_analysis":
            base_queries.append(f"{module_name} 修复方案 关键边界")
    else:
        base_queries = [user_query]
        # P0：函数定位语义补符号别名。
        # 例如“多样性惩罚项对应哪个函数”会补 `apply_diversity_penalty`。
        base_queries.extend(_expand_symbol_alias_queries(user_query))
        # 原因排查类问题补章节级锚点，减少只命中代码片段导致“只给函数不说原因”。
        base_queries.extend(_expand_reason_section_queries(user_query))
        # 指标+定位问题补章节级指标锚点。
        base_queries.extend(_expand_metric_section_queries(user_query))
        # 对“预算+风控+约束”这类全局约束问题补总体架构锚点，
        # 避免只命中局部策略文档而错过总体架构说明。
        if "约束" in user_query and ("预算" in user_query or "风控" in user_query):
            base_queries.append("广告引擎 总体架构 约束条件")
        # 若已识别为排障问题，则不再追加“在线流程/核心链路”模板，
        # 避免把问题误导向 00-总体架构文档，压制模块级排障文档。
        if flags["pipeline"] and not flags["troubleshoot"]:
            base_queries.append(f"{module_name} 在线流程 核心链路")
        if flags["metric"]:
            base_queries.append(f"{module_name} 指标口径 监控体系")
            # 变更：移除“指标问题默认补总体架构关键指标”。
            # 该模板会让 00-总体架构过度前置，压制更垂直的模块文档（两率/出价/精排）。
            # 如需总体视角，交给 architecture 意图分支单独触发。
            # 例外：若是“广告引擎全局指标”类问题，且模块仍是 orchestrator，则补一条总体指标语义。
            if (
                module_name == "ad-serving-orchestrator"
                and not flags["troubleshoot"]
                and ("广告引擎" in user_query or "在线投放" in user_query)
            ):
                base_queries.append("广告引擎 总体架构 关键指标")
            # 针对精排指标问题补专属锚点，降低被“两率指标”文档抢占 Top1 的概率。
            if module_name == "rerank-engine":
                base_queries.append("精排策略 监控指标 排序质量")
                base_queries.append("compute_rank_score rank_topn rerank_engine.py")
                base_queries.append("排序时延 P95 P99 TopN 点击率 多样性指标 探索流量占比")
            # 指标问题补充一条模块别名，恢复适度召回宽度。
            base_queries.extend(_module_alias_queries(module_name)[:1])
        if flags["troubleshoot"]:
            # 避免把“联调排障手册”固定顶到第一，优先使用模块级排查语义。
            base_queries.append(f"{module_name} 模块排查 异常定位")
            # 仅在“明确手册导向”问题里，才补排障手册锚点：
            # - 例如“联调实践建议 / 快速判断 / 手册怎么用”；
            # - 对“冷启动无量 / 候选骤降 / 两率波动”这类模块排查问题不强加手册锚点。
            explicit_handbook_intent = any(
                token in lowered_query
                for token in ("联调", "实践建议", "快速判断", "手册", "排障清单")
            )
            if explicit_handbook_intent:
                base_queries.append("联调排障手册 故障定位 排查清单")
            base_queries.extend(_module_alias_queries(module_name)[:1])
        if flags["architecture"]:
            base_queries.append(f"{module_name} 总体架构 模块关系")
            base_queries.extend(_module_alias_queries(module_name))
        # 常规知识问答给一条轻量兜底语义，不再默认塞入架构相关模板语句。
        if not any(flags.values()):
            base_queries.append(f"{module_name} 业务说明")
        if state.get("active_task_stage") in {"issue_analysis", "confirm_code", "code_generation"}:
            base_queries.append(f"{module_name} 设计背景 业务口径")

    queries = _expand_queries(base_queries, original_user_query=user_query)
    # 精排指标场景下，剔除过泛的“指标”短 query，降低两率文档被误抬升概率。
    if module_name == "rerank-engine" and flags["metric"]:
        blocked_metric_queries = {"指标", "关键指标", "监控指标", "业务指标"}
        queries = [query for query in queries if query not in blocked_metric_queries]
    retrieval_plan = _build_retrieval_plan(
        user_query=user_query,
        route=route,
        transition_type=transition_type,
        query_flags=flags,
    )
    trace_summary = (
        f"queries={len(queries)},strategy={retrieval_plan['strategy']},"
        f"wiki_top_k={retrieval_plan['wiki_top_k']},code_top_k={retrieval_plan['code_top_k']}"
    )
    return {
        "retrieval_queries": queries,
        "retrieval_plan": retrieval_plan,
        "node_trace": service._trace(state, "query_rewriter", trace_summary),
    }
