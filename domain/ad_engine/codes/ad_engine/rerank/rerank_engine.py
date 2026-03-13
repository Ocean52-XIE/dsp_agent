"""精排策略 mock 代码语料。"""

from dataclasses import dataclass
from typing import Any


@dataclass
class RankItem:
    """精排候选项。"""

    ad_id: str
    ecpm: float
    freq_penalty: float
    diversity_penalty: float
    feature_complete: bool


def apply_diversity_penalty(item: RankItem) -> float:
    """多样性惩罚项。"""
    return item.diversity_penalty * 0.6


def apply_safety_penalty(item: RankItem) -> float:
    """频控/风控惩罚项。"""
    return item.freq_penalty * 0.8


def compute_rank_score(item: RankItem) -> float:
    """最终排序分。

    该逻辑解释了“高 eCPM 广告为何未进最终位”：
    - 频控惩罚
    - 多样性惩罚
    - 特征缺失降级
    """
    score = item.ecpm
    score -= apply_safety_penalty(item)
    score -= apply_diversity_penalty(item)
    if not item.feature_complete:
        score *= 0.75
    return score


def rank_topn(items: list[RankItem], *, top_n: int) -> list[dict[str, Any]]:
    """返回最终 TopN。"""
    scored = []
    for item in items:
        scored.append({"ad_id": item.ad_id, "rank_score": compute_rank_score(item)})
    scored.sort(key=lambda record: record["rank_score"], reverse=True)
    return scored[:top_n]

