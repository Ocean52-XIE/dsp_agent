"""广告指标查询工具处理器

本模块提供 query_ad_metrics 技能的工具处理函数。

使用方式：
    被 SkillExecutor 通过 importlib 动态加载，
    从 TOOLS 字典中获取对应工具的处理函数。

工具列表：
    - get_ad_metrics: 获取广告指标数据
"""
import logging
import random
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)


def get_ad_metrics(
    metric_type: str,
    time_range: str = "7d",
    ad_group_id: str | None = None,
    campaign_id: str | None = None,
    dimension: str = "day",
) -> dict[str, Any]:
    """获取广告指标数据

    从广告数据平台查询指定类型的指标数据。
    支持多维度筛选和时间范围查询。

    Args:
        metric_type: 指标类型（ctr/cvr/cost/impression/click/conversion/all）
        time_range: 时间范围，如 "7d" 表示最近7天
        ad_group_id: 广告组 ID（可选）
        campaign_id: 计划 ID（可选）
        dimension: 聚合维度（day/hour/campaign/ad_group/creative）

    Returns:
        包含指标数据的字典，格式如下：
        {
            "success": True,
            "data": {
                "metric_type": "ctr",
                "time_range": "7d",
                "dimension": "day",
                "filter": {...},
                "records": [...],
                "summary": {...}
            }
        }
    """
    logger.info(
        f"[get_ad_metrics] 查询指标: type={metric_type}, "
        f"time_range={time_range}, dimension={dimension}"
    )

    try:
        # 解析时间范围
        days = _parse_time_range(time_range)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # 构建筛选条件
        filters = {}
        if ad_group_id:
            filters["ad_group_id"] = ad_group_id
        if campaign_id:
            filters["campaign_id"] = campaign_id

        # 生成模拟数据（实际场景中应调用真实数据源）
        records = _generate_mock_data(
            metric_type=metric_type,
            start_date=start_date,
            end_date=end_date,
            dimension=dimension,
        )

        # 计算汇总统计
        summary = _calculate_summary(records, metric_type)

        result = {
            "success": True,
            "data": {
                "metric_type": metric_type,
                "time_range": time_range,
                "dimension": dimension,
                "filter": filters if filters else None,
                "query_time": datetime.now().isoformat(),
                "records": records,
                "summary": summary,
            }
        }

        logger.info(f"[get_ad_metrics] 查询成功，返回 {len(records)} 条记录")
        return result

    except Exception as e:
        logger.error(f"[get_ad_metrics] 查询失败: {e}")
        return {
            "success": False,
            "error": str(e),
        }


def _parse_time_range(time_range: str) -> int:
    """解析时间范围字符串

    Args:
        time_range: 时间范围字符串，如 "7d", "30d", "1d"

    Returns:
        天数
    """
    if time_range.endswith("d"):
        return int(time_range[:-1])
    elif time_range.endswith("h"):
        return 1  # 小时级按1天处理
    else:
        return 7  # 默认7天


def _generate_mock_data(
    metric_type: str,
    start_date: datetime,
    end_date: datetime,
    dimension: str,
) -> list[dict[str, Any]]:
    """生成模拟数据

    实际场景中应替换为真实的数据查询逻辑。

    Args:
        metric_type: 指标类型
        start_date: 开始日期
        end_date: 结束日期
        dimension: 聚合维度

    Returns:
        数据记录列表
    """
    records = []
    current_date = start_date

    # 不同指标的基准值范围
    metric_ranges = {
        "ctr": (0.015, 0.035),      # 点击率 1.5%-3.5%
        "cvr": (0.02, 0.08),        # 转化率 2%-8%
        "cost": (5000, 20000),      # 消耗 5000-20000
        "impression": (100000, 500000),  # 展示量
        "click": (2000, 10000),     # 点击量
        "conversion": (100, 500),   # 转化量
    }

    while current_date <= end_date:
        # 根据指标类型生成数据
        if metric_type == "all":
            # 全部指标
            record = {
                "date": current_date.strftime("%Y-%m-%d"),
                "ctr": round(random.uniform(*metric_ranges["ctr"]), 4),
                "cvr": round(random.uniform(*metric_ranges["cvr"]), 4),
                "cost": round(random.uniform(*metric_ranges["cost"]), 2),
                "impression": random.randint(*metric_ranges["impression"]),
                "click": random.randint(*metric_ranges["click"]),
                "conversion": random.randint(*metric_ranges["conversion"]),
            }
        else:
            # 单一指标
            value_range = metric_ranges.get(metric_type, (0, 1))
            if metric_type in ["ctr", "cvr"]:
                value = round(random.uniform(*value_range), 4)
            elif metric_type == "cost":
                value = round(random.uniform(*value_range), 2)
            else:
                value = random.randint(*value_range)

            record = {
                "date": current_date.strftime("%Y-%m-%d"),
                "value": value,
            }

            # 添加环比变化
            if records:
                prev_value = records[-1].get("value", value)
                if prev_value > 0:
                    change_rate = round((value - prev_value) / prev_value, 4)
                else:
                    change_rate = 0
                record["change_rate"] = change_rate
            else:
                record["change_rate"] = 0

        records.append(record)
        current_date += timedelta(days=1)

    return records


def _calculate_summary(
    records: list[dict[str, Any]],
    metric_type: str,
) -> dict[str, Any]:
    """计算汇总统计

    Args:
        records: 数据记录列表
        metric_type: 指标类型

    Returns:
        汇总统计数据
    """
    if not records:
        return {}

    if metric_type == "all":
        # 多指标汇总
        summary = {}
        for field in ["ctr", "cvr", "cost", "impression", "click", "conversion"]:
            values = [r[field] for r in records if field in r]
            if values:
                summary[f"{field}_avg"] = round(sum(values) / len(values), 4)
                summary[f"{field}_max"] = max(values)
                summary[f"{field}_min"] = min(values)
        return summary
    else:
        # 单指标汇总
        values = [r.get("value", 0) for r in records]
        return {
            "avg_value": round(sum(values) / len(values), 4) if values else 0,
            "max_value": max(values) if values else 0,
            "min_value": min(values) if values else 0,
            "total_records": len(records),
        }


# 工具注册表
# SkillExecutor 会通过 TOOLS 字典查找对应的处理函数
TOOLS = {
    "get_ad_metrics": get_ad_metrics,
}
