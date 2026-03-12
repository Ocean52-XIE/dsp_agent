from __future__ import annotations

import re
from typing import Any

# 广告引擎领域关键词：命中后提升 in-scope 相关度。
DOMAIN_TERMS: tuple[str, ...] = (
    # 主业务链路
    "广告",
    "投放",
    "召回",
    "两率",
    "预估",
    "出价",
    "精排",
    "重排",
    "竞价",
    "预算",
    "pacing",
    "ecpm",
    "ocpc",
    "tcpa",
    "troas",
    "roas",
    "cpa",
    "ctr",
    "cvr",
    "pctr",
    "pcvr",
    "target_cpa",
    "target_roas",
    # 工程与排障语义
    "链路",
    "模块",
    "指标",
    "口径",
    "监控",
    "日志",
    "回传",
    "排障",
    "排查",
    "异常",
    "定位",
    "修复",
    "函数",
    "文件",
    "代码",
    "实现",
    "检索",
    "wiki",
)

# 常见闲聊/寒暄短语：用于快速判定为 out-of-scope。
SMALL_TALK_EXACT: set[str] = {
    "哈",
    "哈哈",
    "哈哈哈",
    "呵呵",
    "嘿嘿",
    "嗯",
    "哦",
    "啊",
    "呀",
    "在吗",
    "你好",
    "hi",
    "hello",
    "ok",
    "thanks",
    "谢谢",
}

SMALL_TALK_SUBSTR: tuple[str, ...] = (
    "笑死",
    "晚安",
    "早安",
    "早上好",
    "中午好",
    "晚上好",
)

# 明确偏离系统领域的问题词。
OFFTOPIC_TERMS: tuple[str, ...] = (
    "天气",
    "股价",
    "股票",
    "基金",
    "彩票",
    "电影",
    "翻译",
    "作文",
    "诗",
    "旅游",
    "菜谱",
    "八卦",
    "星座",
)

# 代码定位类问法特征（函数名、路径、文件名等）。
_CODE_HINT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}\s*(\(|\.py\b|/)")
# 仅由语气词/笑声/标点组成的输入。
_LAUGH_LIKE_RE = re.compile(r"^[哈呵嘿啊呀哦嗯~\s!！?？.,，。…]+$")


def _normalize(text: str) -> str:
    """对输入做轻量归一化，统一小写和空白。"""
    return " ".join((text or "").strip().lower().split())


def _is_small_talk(text: str) -> bool:
    """判断是否属于闲聊输入。"""
    if not text:
        return True
    if text in SMALL_TALK_EXACT:
        return True
    if any(token in text for token in SMALL_TALK_SUBSTR):
        return True
    return _LAUGH_LIKE_RE.fullmatch(text) is not None


def _count_hits(text: str, terms: tuple[str, ...]) -> int:
    """统计关键词命中数。"""
    return sum(1 for term in terms if term in text)


def _compute_domain_relevance(state: dict[str, Any]) -> tuple[float, bool, str]:
    """计算领域相关度，并输出判定摘要用于 debug。"""
    user_query = str(state.get("user_query", "") or "")
    normalized = _normalize(user_query)

    # 规则 1：纯闲聊直接拦截。
    if _is_small_talk(normalized):
        return 0.0, False, "small_talk"

    domain_hits = _count_hits(normalized, DOMAIN_TERMS)
    off_hits = _count_hits(normalized, OFFTOPIC_TERMS)
    code_hint = _CODE_HINT_RE.search(user_query) is not None

    # 规则 2：基础相关度 = 领域词命中 + 代码形态 hint。
    relevance = min(1.0, domain_hits * 0.22 + (0.22 if code_hint else 0.0))
    # 单一强信号保底：
    # - 只要命中一个领域词（如“ocpc/召回/出价”），至少视为弱 in-scope；
    # - 只要命中代码定位形态（函数名/文件路径），也至少视为弱 in-scope。
    if domain_hits > 0:
        relevance = max(relevance, 0.62)
    if code_hint:
        relevance = max(relevance, 0.58)

    # 规则 3：若是“指代式追问”且来自历史上下文，补充上下文加分。
    if str(state.get("active_topic_source", "")) == "history_memory":
        relevance = min(1.0, relevance + 0.18)

    # 规则 4：明显无关且无领域信号，直接判定领域外。
    if off_hits > 0 and domain_hits == 0 and not code_hint:
        return max(0.0, relevance - 0.45), False, f"offtopic_hits={off_hits}"

    # 规则 5：极短输入且无领域信号，视为领域外（例如“哈哈”“嗯？”）。
    compact = normalized.replace(" ", "")
    if len(compact) <= 4 and domain_hits == 0 and not code_hint:
        return max(0.0, relevance - 0.25), False, "short_non_domain"

    is_domain_related = relevance >= 0.50
    reason = (
        f"domain_hits={domain_hits}, off_hits={off_hits}, code_hint={int(code_hint)}, "
        f"source={state.get('active_topic_source', 'current_query')}"
    )
    return relevance, is_domain_related, reason


def run(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """领域门控：决定本轮是否进入业务链路。"""
    relevance, is_domain_related, reason = _compute_domain_relevance(state)
    return {
        "domain_relevance": relevance,
        "is_domain_related": is_domain_related,
        "node_trace": service._trace(
            state,
            "domain_gate",
            f"relevance={relevance:.2f}, in_scope={is_domain_related}, {reason}",
        ),
    }
