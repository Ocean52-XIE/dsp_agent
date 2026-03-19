# -*- coding: utf-8 -*-
"""广告信息查询脚本

这是 query_ad_info skill 的执行脚本，用于模拟查询广告系统信息。

使用方式（JSON+stdin 模式）：
    echo '{"ad_type": "campaign", "status": "active", "limit": 10}' | python scripts/query_ad.py
    echo '{"ad_type": "adgroup", "ad_id": "12345"}' | python scripts/query_ad.py
    echo '{"ad_type": "creative", "limit": 20}' | python scripts/query_ad.py

输入格式：JSON（从 stdin 读取）
{
    "ad_type": "campaign|adgroup|creative",  // 必填
    "ad_id": "xxx",                           // 可选，指定则查详情
    "status": "active|paused|archived|all",   // 可选，默认 active
    "limit": 10                               // 可选，默认 10
}

返回格式：JSON
"""

import json
import random
import sys
from datetime import datetime, timedelta
from typing import Any


def generate_campaign(campaign_id: str) -> dict[str, Any]:
    """生成模拟的广告计划数据"""
    statuses = ["active", "paused", "archived"]
    names = [
        "双十一大促计划",
        "品牌推广计划",
        "新品上市计划",
        "用户增长计划",
        "节日营销计划",
    ]

    return {
        "id": campaign_id,
        "name": random.choice(names),
        "status": random.choice(statuses),
        "budget": random.uniform(10000, 100000),
        "spent": random.uniform(5000, 50000),
        "impressions": random.randint(100000, 1000000),
        "clicks": random.randint(1000, 10000),
        "conversions": random.randint(10, 500),
        "start_date": (datetime.now() - timedelta(days=random.randint(1, 30))).strftime("%Y-%m-%d"),
        "end_date": (datetime.now() + timedelta(days=random.randint(1, 30))).strftime("%Y-%m-%d"),
        "created_at": datetime.now().isoformat(),
    }


def generate_adgroup(adgroup_id: str) -> dict[str, Any]:
    """生成模拟的广告组数据"""
    statuses = ["active", "paused"]
    names = [
        "核心用户组",
        "潜在客户组",
        "高价值用户组",
        "新用户转化组",
        "复购用户组",
    ]
    targeting_types = ["interest", "behavior", "custom", "lookalike"]

    return {
        "id": adgroup_id,
        "name": random.choice(names),
        "status": random.choice(statuses),
        "campaign_id": f"campaign_{random.randint(1000, 9999)}",
        "bid": random.uniform(0.5, 5.0),
        "bid_strategy": random.choice(["CPC", "CPM", "OCPM"]),
        "targeting": {
            "type": random.choice(targeting_types),
            "age_range": f"{random.randint(18, 35)}-{random.randint(36, 60)}",
            "gender": random.choice(["all", "male", "female"]),
            "interests": ["科技", "购物", "娱乐"][:random.randint(1, 3)],
        },
        "impressions": random.randint(10000, 100000),
        "clicks": random.randint(100, 1000),
        "ctr": round(random.uniform(0.01, 0.05), 4),
        "created_at": datetime.now().isoformat(),
    }


def generate_creative(creative_id: str) -> dict[str, Any]:
    """生成模拟的创意数据"""
    statuses = ["active", "paused", "pending_review", "rejected"]
    titles = [
        "限时特惠，不容错过！",
        "品质生活，从这里开始",
        "新品首发，抢先体验",
        "超级品牌日，爆款直降",
        "会员专享，更多福利",
    ]

    return {
        "id": creative_id,
        "title": random.choice(titles),
        "description": "这是一条精彩的广告创意，吸引你的目光。",
        "status": random.choice(statuses),
        "adgroup_id": f"adgroup_{random.randint(1000, 9999)}",
        "format": random.choice(["image", "video", "carousel"]),
        "image_url": f"https://example.com/images/creative_{creative_id}.jpg",
        "landing_page": f"https://example.com/landing/{creative_id}",
        "impressions": random.randint(1000, 50000),
        "clicks": random.randint(10, 500),
        "ctr": round(random.uniform(0.01, 0.05), 4),
        "created_at": datetime.now().isoformat(),
    }


def query_list(ad_type: str, status: str, limit: int) -> dict[str, Any]:
    """查询广告列表"""
    items = []
    generators = {
        "campaign": generate_campaign,
        "adgroup": generate_adgroup,
        "creative": generate_creative,
    }

    generator = generators.get(ad_type, generate_campaign)

    for i in range(limit):
        item_id = f"{ad_type}_{random.randint(10000, 99999)}"
        item = generator(item_id)

        # 如果指定了状态过滤（非 all），只返回匹配的
        if status != "all" and item.get("status") != status:
            continue

        items.append(item)

    return {
        "success": True,
        "data": {
            "type": ad_type,
            "total": len(items),
            "status_filter": status,
            "items": items,
        },
        "timestamp": datetime.now().isoformat(),
    }


def query_detail(ad_type: str, ad_id: str) -> dict[str, Any]:
    """查询单个广告详情"""
    generators = {
        "campaign": generate_campaign,
        "adgroup": generate_adgroup,
        "creative": generate_creative,
    }

    generator = generators.get(ad_type, generate_campaign)
    item = generator(ad_id)

    return {
        "success": True,
        "data": {
            "type": ad_type,
            "item": item,
        },
        "timestamp": datetime.now().isoformat(),
    }


def main():
    """主函数：从 stdin 读取 JSON 参数

    脚本从 stdin 读取 JSON 格式的参数，自行处理默认值和验证。
    """
    # 1. 从 stdin 读取 JSON
    params = {}
    try:
        params_text = sys.stdin.read()
        if params_text.strip():
            params = json.loads(params_text)
    except json.JSONDecodeError as e:
        error_result = {
            "success": False,
            "error": f"JSON 解析失败: {e}",
            "timestamp": datetime.now().isoformat(),
        }
        print(json.dumps(error_result, ensure_ascii=False))
        sys.exit(1)

    # 2. 定义默认值
    defaults = {
        "ad_type": None,       # 必填，无默认值
        "ad_id": None,         # 可选
        "status": "active",    # 默认值
        "limit": 10,           # 默认值
    }

    # 3. 合并参数（用户参数覆盖默认值）
    merged = {**defaults, **params}

    # 4. 验证必填参数
    if not merged["ad_type"]:
        error_result = {
            "success": False,
            "error": "ad_type 是必填参数",
            "valid_types": ["campaign", "adgroup", "creative"],
            "timestamp": datetime.now().isoformat(),
        }
        print(json.dumps(error_result, ensure_ascii=False))
        sys.exit(1)

    # 验证 ad_type 枚举值
    valid_types = ["campaign", "adgroup", "creative"]
    if merged["ad_type"] not in valid_types:
        error_result = {
            "success": False,
            "error": f"无效的 ad_type: {merged['ad_type']}",
            "valid_types": valid_types,
            "timestamp": datetime.now().isoformat(),
        }
        print(json.dumps(error_result, ensure_ascii=False))
        sys.exit(1)

    # 5. 执行查询
    try:
        if merged["ad_id"]:
            result = query_detail(merged["ad_type"], merged["ad_id"])
        else:
            result = query_list(merged["ad_type"], merged["status"], merged["limit"])

        print(json.dumps(result, ensure_ascii=False))
        sys.exit(0)

    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }
        print(json.dumps(error_result, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
