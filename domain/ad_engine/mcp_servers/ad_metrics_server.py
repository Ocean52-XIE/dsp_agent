#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""广告指标查询 MCP Server

使用 FastMCP 框架实现的 MCP Server，提供广告指标查询功能。

可用工具：
1. query_metrics - 查询广告指标
2. list_available_metrics - 列出可用指标
3. get_metric_trend - 获取指标趋势

启动方式：
    fastmcp run ad_metrics_server.py:mcp

或者直接运行：
    python ad_metrics_server.py
"""
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Any

from fastmcp import FastMCP

# 配置日志
import sys
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,  # MCP 使用 stdin/stdout，日志输出到 stderr
)
logger = logging.getLogger("ad_metrics_server")

# =============================================================================
# FastMCP Server 实例
# =============================================================================

mcp = FastMCP(
    name="ad_metrics",
    version="1.0.0",
)

# =============================================================================
# Mock 数据
# =============================================================================

# 模拟的广告指标数据
MOCK_METRICS_DATA = {
    "ctr": {
        "name": "CTR",
        "description": "点击率 (Click-Through Rate)",
        "unit": "%",
        "base_value": 2.5,
        "variance": 0.5,
    },
    "cvr": {
        "name": "CVR",
        "description": "转化率 (Conversion Rate)",
        "unit": "%",
        "base_value": 1.2,
        "variance": 0.3,
    },
    "cpm": {
        "name": "CPM",
        "description": "千次展示成本 (Cost Per Mille)",
        "unit": "元",
        "base_value": 15.0,
        "variance": 3.0,
    },
    "cpc": {
        "name": "CPC",
        "description": "单次点击成本 (Cost Per Click)",
        "unit": "元",
        "base_value": 0.6,
        "variance": 0.15,
    },
    "cpa": {
        "name": "CPA",
        "description": "单次转化成本 (Cost Per Action)",
        "unit": "元",
        "base_value": 50.0,
        "variance": 10.0,
    },
    "ecpm": {
        "name": "eCPM",
        "description": "有效千次展示收入 (Effective CPM)",
        "unit": "元",
        "base_value": 18.0,
        "variance": 4.0,
    },
    "spend": {
        "name": "消耗",
        "description": "广告消耗金额",
        "unit": "元",
        "base_value": 10000.0,
        "variance": 2000.0,
    },
    "impressions": {
        "name": "展示数",
        "description": "广告展示次数",
        "unit": "次",
        "base_value": 666667.0,
        "variance": 100000.0,
    },
    "clicks": {
        "name": "点击数",
        "description": "广告点击次数",
        "unit": "次",
        "base_value": 16667.0,
        "variance": 3000.0,
    },
    "conversions": {
        "name": "转化数",
        "description": "广告转化次数",
        "unit": "次",
        "base_value": 200.0,
        "variance": 50.0,
    },
}

# 模拟的维度数据
MOCK_DIMENSIONS = {
    "campaign": ["夏季大促", "品牌推广", "新品上市", "日常投放"],
    "ad_group": ["搜索广告", "信息流广告", "开屏广告", "视频广告"],
    "creative": ["图片A", "图片B", "视频A", "视频B"],
    "region": ["北京", "上海", "广州", "深圳", "杭州"],
}


def generate_mock_value(metric_key: str, date_str: str) -> float:
    """生成 Mock 指标值

    使用日期作为种子，确保同一日期的值一致。

    Args:
        metric_key: 指标键名
        date_str: 日期字符串

    Returns:
        模拟的指标值
    """
    metric_info = MOCK_METRICS_DATA.get(metric_key, {})
    base_value = metric_info.get("base_value", 1.0)
    variance = metric_info.get("variance", 0.1)

    # 使用日期和指标名生成确定性随机偏移
    seed = f"{date_str}_{metric_key}"
    hash_val = int(hashlib.md5(seed.encode()).hexdigest()[:8], 16)
    offset = ((hash_val % 1000) / 1000 - 0.5) * 2 * variance

    return round(base_value + offset, 4)


def generate_trend_data(
    metric_key: str,
    start_date: str,
    end_date: str,
) -> list[dict[str, Any]]:
    """生成趋势数据

    Args:
        metric_key: 指标键名
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        趋势数据列表
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    trend = []
    current = start
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        value = generate_mock_value(metric_key, date_str)
        trend.append({
            "date": date_str,
            "value": value,
        })
        current += timedelta(days=1)

    return trend


# =============================================================================
# MCP Tools 定义
# =============================================================================

@mcp.tool()
def query_metrics(
    start_date: str,
    end_date: str,
    metrics: list[str],
    dimensions: dict[str, str] | None = None,
) -> dict[str, Any]:
    """查询指定时间范围的广告指标数据。支持多个指标同时查询。

    Args:
        start_date: 开始日期，格式：YYYY-MM-DD
        end_date: 结束日期，格式：YYYY-MM-DD
        metrics: 要查询的指标列表，可选：ctr, cvr, cpm, cpc, cpa, ecpm, spend, impressions, clicks, conversions
        dimensions: 筛选维度（可选），支持 campaign, ad_group, region

    Returns:
        包含查询信息和指标数据的字典
    """
    logger.info(f"[query_metrics] 查询指标: {metrics}, 日期范围: {start_date} ~ {end_date}")

    # 验证日期格式
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(f"日期格式错误，应为 YYYY-MM-DD: {e}")

    # 验证指标
    valid_metrics = set(MOCK_METRICS_DATA.keys())
    invalid_metrics = set(metrics) - valid_metrics
    if invalid_metrics:
        raise ValueError(f"无效的指标: {invalid_metrics}，可用指标: {valid_metrics}")

    days = (end - start).days + 1

    # 生成结果
    result = {
        "query_info": {
            "start_date": start_date,
            "end_date": end_date,
            "days": days,
            "dimensions": dimensions if dimensions else None,
        },
        "metrics": {},
        "summary": {},
    }

    for metric_key in metrics:
        metric_info = MOCK_METRICS_DATA[metric_key]
        daily_values = []

        current = start
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            value = generate_mock_value(metric_key, date_str)
            daily_values.append({"date": date_str, "value": value})
            current += timedelta(days=1)

        total_value = sum(d["value"] for d in daily_values)
        avg_value = total_value / len(daily_values) if daily_values else 0

        result["metrics"][metric_key] = {
            "name": metric_info["name"],
            "unit": metric_info["unit"],
            "daily": daily_values,
            "total": round(total_value, 2) if metric_key in ["spend", "impressions", "clicks", "conversions"] else None,
            "average": round(avg_value, 4),
        }

        result["summary"][metric_key] = {
            "avg": round(avg_value, 4),
            "unit": metric_info["unit"],
        }

    return result


@mcp.tool()
def list_available_metrics() -> dict[str, Any]:
    """列出所有可用的广告指标及其说明。

    Returns:
        包含所有可用指标信息的字典
    """
    logger.info("[list_available_metrics] 列出所有可用指标")

    metrics_list = []
    for key, info in MOCK_METRICS_DATA.items():
        metrics_list.append({
            "key": key,
            "name": info["name"],
            "description": info["description"],
            "unit": info["unit"],
        })

    return {
        "metrics": metrics_list,
        "total": len(metrics_list),
        "usage": "使用 query_metrics 工具查询具体指标数据，metrics 参数传入 key 列表",
    }


@mcp.tool()
def get_metric_trend(
    metric: str,
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    """获取指定指标在时间范围内的趋势数据，包含每日明细。

    Args:
        metric: 指标名称，如 ctr, cvr, cpm 等
        start_date: 开始日期，格式：YYYY-MM-DD
        end_date: 结束日期，格式：YYYY-MM-DD

    Returns:
        包含趋势数据和统计信息的字典
    """
    logger.info(f"[get_metric_trend] 获取趋势: {metric}, 日期范围: {start_date} ~ {end_date}")

    # 验证指标
    if metric not in MOCK_METRICS_DATA:
        valid_metrics = list(MOCK_METRICS_DATA.keys())
        raise ValueError(f"无效指标: {metric}，可用指标: {valid_metrics}")

    # 验证日期
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(f"日期格式错误: {e}")

    metric_info = MOCK_METRICS_DATA[metric]
    trend_data = generate_trend_data(metric, start_date, end_date)

    # 计算统计信息
    values = [d["value"] for d in trend_data]
    avg_value = sum(values) / len(values) if values else 0
    max_value = max(values) if values else 0
    min_value = min(values) if values else 0

    return {
        "metric": {
            "key": metric,
            "name": metric_info["name"],
            "unit": metric_info["unit"],
        },
        "period": {
            "start_date": start_date,
            "end_date": end_date,
            "days": len(trend_data),
        },
        "trend": trend_data,
        "statistics": {
            "average": round(avg_value, 4),
            "max": round(max_value, 4),
            "min": round(min_value, 4),
            "max_date": max(trend_data, key=lambda x: x["value"])["date"],
            "min_date": min(trend_data, key=lambda x: x["value"])["date"],
        },
    }


if __name__ == "__main__":
    # 启动 MCP Server
    mcp.run()
