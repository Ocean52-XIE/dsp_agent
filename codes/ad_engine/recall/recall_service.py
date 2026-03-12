"""在线召回 mock 代码语料。

该文件用于检索评测，不参与生产逻辑。
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class RecallContext:
    """召回上下文。"""

    user_id: str
    request_id: str
    trace_id: str
    region: str
    scene: str


def select_recall_candidates(context: RecallContext, index_client: Any) -> list[dict[str, Any]]:
    """根据用户上下文选取候选广告。

    关键点：
    - 先按 region/scene 做预过滤；
    - 再按用户标签拉取人群包；
    - 最后按 source_tag 打标，给精排做多样性约束。
    """
    filter_query = {
        "region": context.region,
        "scene": context.scene,
        "user_id": context.user_id,
    }
    raw_candidates = index_client.search(filter_query, top_n=800)
    results: list[dict[str, Any]] = []
    for item in raw_candidates:
        if item.get("creative_status") != "online":
            continue
        results.append(
            {
                "ad_id": item["ad_id"],
                "campaign_id": item["campaign_id"],
                "source_tag": item.get("source_tag", "rule_recall"),
            }
        )
    return results

