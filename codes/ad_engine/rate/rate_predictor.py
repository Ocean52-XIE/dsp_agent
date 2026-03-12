"""两率预估 mock 代码语料。"""

from dataclasses import dataclass
from typing import Any


@dataclass
class PredictFeatures:
    """预估特征容器。"""

    user_vector: list[float]
    ad_vector: list[float]
    context_vector: list[float]
    trace_id: str


def predict_ctr_cvr(features: PredictFeatures, model_client: Any) -> dict[str, float]:
    """预测 pCTR / pCVR。

    注意：
    - 线上推理后需要做概率校准；
    - 校准版本应与模型版本一致，避免偏差放大。
    """
    raw = model_client.infer(
        {
            "user_vector": features.user_vector,
            "ad_vector": features.ad_vector,
            "context_vector": features.context_vector,
        }
    )
    pctr = calibrate_probability(raw.get("pctr", 0.0), slope=0.96, bias=0.01)
    pcvr = calibrate_probability(raw.get("pcvr", 0.0), slope=1.02, bias=-0.005)
    return {"pctr": pctr, "pcvr": pcvr}


def calibrate_probability(value: float, *, slope: float, bias: float) -> float:
    """概率校准函数。"""
    calibrated = value * slope + bias
    if calibrated < 0:
        return 0.0
    if calibrated > 1:
        return 1.0
    return calibrated

