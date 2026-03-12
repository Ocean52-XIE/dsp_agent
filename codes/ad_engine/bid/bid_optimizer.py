"""出价策略 mock 代码语料。"""

from dataclasses import dataclass


@dataclass
class BidContext:
    """实时出价上下文。"""

    target_cpa: float
    budget_ratio: float
    competition_level: float
    trace_id: str


def compute_alpha(*, budget_ratio: float, competition_level: float) -> float:
    """根据预算进度与竞争强度计算调节因子 alpha。"""
    if budget_ratio > 1.05:
        # 预算超前消耗：下调 alpha，限制激进拿量。
        return 0.85
    if budget_ratio < 0.75:
        # 预算落后消耗：上调 alpha，提高竞争能力。
        return 1.15
    if competition_level > 0.8:
        return 1.05
    return 1.0


def compute_ocpc_bid(*, target_cpa: float, pcvr: float, alpha: float) -> float:
    """oCPC 简化出价公式。

    bid = target_cpa * pcvr * alpha
    """
    bid = target_cpa * pcvr * alpha
    if bid < 0:
        return 0.0
    return bid


def compute_bid_for_request(context: BidContext, pcvr: float) -> float:
    """请求级出价入口。"""
    alpha = compute_alpha(
        budget_ratio=context.budget_ratio,
        competition_level=context.competition_level,
    )
    return compute_ocpc_bid(
        target_cpa=context.target_cpa,
        pcvr=pcvr,
        alpha=alpha,
    )

