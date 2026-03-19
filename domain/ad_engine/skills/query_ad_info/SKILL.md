---
skill_id: query_ad_info
display_name: 广告信息查询
description: 查询广告系统的广告计划、广告组、创意等信息
version: 1.0.0
tags:
  - 广告
  - 查询
  - ad
  - campaign
  - 创意

skill_type: execution

trigger:
  keywords:
    - 查询广告
    - 广告计划
    - 广告组
    - 创意
    - campaign
    - adgroup
    - creative
    - 广告信息
  patterns:
    - "查.*广告"
    - "获取.*广告.*信息"
    - ".*广告.*详情"
  priority: 30

params:
  ad_type:
    type: string
    description: 广告类型
    enum: [campaign, adgroup, creative]
  ad_id:
    type: string
    description: 广告 ID（可选，不传则返回列表）
  status:
    type: string
    description: 状态过滤
    enum: [active, paused, archived, all]
  limit:
    type: integer
    description: 返回数量限制

execution:
  # 调用 Python 脚本，参数通过 stdin 以 JSON 格式传递
  command:
    - "python"
    - "scripts/query_ad.py"
  timeout: 30
  output: json

examples:
  - user: "查询所有活跃的广告计划"
    params:
      ad_type: campaign
      status: active
  - user: "查看广告组 12345 的详情"
    params:
      ad_type: adgroup
      ad_id: "12345"
  - user: "获取最近 20 条创意"
    params:
      ad_type: creative
      limit: 20
---

查询广告系统中的广告计划、广告组、创意等信息的技能。

## 输入参数

参数通过 stdin 以 JSON 格式传递，脚本会自动处理默认值。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| ad_type | string | 是 | - | 广告类型 (campaign/adgroup/creative) |
| ad_id | string | 否 | - | 广告 ID，指定则查详情 |
| status | string | 否 | active | 状态过滤 |
| limit | integer | 否 | 10 | 返回数量限制 |

## 支持的广告类型

| 类型 | 说明 | 主要字段 |
|------|------|----------|
| campaign | 广告计划 | 名称、预算、时间范围、状态 |
| adgroup | 广告组 | 名称、出价、定向、状态 |
| creative | 创意 | 标题、描述、图片、状态 |

## 使用场景

1. 查询广告计划列表或详情
2. 查询广告组信息和定向设置
3. 查询创意内容和审核状态

## 返回格式

返回 JSON 格式的结果：

```json
{
  "success": true,
  "data": {
    "type": "campaign",
    "total": 10,
    "items": [...]
  },
  "timestamp": "..."
}
```
