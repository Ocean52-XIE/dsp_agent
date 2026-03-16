---
# 技能元数据
name: query_ad_metrics
description: 查询广告投放指标数据，支持多维度筛选和时间范围查询。可获取 CTR、CVR、消耗、展示量、点击量等核心广告指标。
version: 1.0.0
author: ad_engine_team
tags:
  - 广告指标
  - 数据查询
  - 监控
enabled: true

# 技能类型
type: tool_execution

# 触发配置
trigger:
  keywords:
    - 查询指标
    - 广告数据
    - CTR
    - CVR
    - 消耗
    - 展示量
    - 点击量
    - 转化
    - 报表
    - 数据看板
    - 指标查询
  patterns:
    - "查询.*指标"
    - "获取.*数据"
    - "统计.*报表"
    - '\d+天.*数据'
  priority: 15

# 参数定义
params:
  required:
    - metric_type
  properties:
    metric_type:
      type: string
      description: 要查询的指标类型
      enum:
        - ctr          # 点击率
        - cvr          # 转化率
        - cost         # 消耗
        - impression   # 展示量
        - click        # 点击量
        - conversion   # 转化量
        - all          # 全部指标
      required: true
    time_range:
      type: string
      description: 时间范围，支持相对时间（如 7d, 30d）或绝对日期（如 2024-01-01~2024-01-31）
      default: "7d"
    ad_group_id:
      type: string
      description: 广告组 ID，可选，用于筛选特定广告组
    campaign_id:
      type: string
      description: 计划 ID，可选，用于筛选特定计划
    dimension:
      type: string
      description: 数据聚合维度
      enum:
        - day          # 按天
        - hour         # 按小时
        - campaign     # 按计划
        - ad_group     # 按广告组
        - creative     # 按创意
      default: "day"

# 工具定义
tools:
  - name: get_ad_metrics
    description: 从广告数据平台获取指标数据
    parameters:
      metric_type:
        type: string
        description: 指标类型（ctr/cvr/cost/impression/click/conversion/all 或中文别名如 点击率/转化率/消耗）
      time_range:
        type: string
        description: 时间范围（如 7d/30d 或自然语言如 yesterday/昨天/今天）
        default: "7d"
      # 别名参数（兼容 LLM 可能使用的不同参数名）
      metric:
        type: string
        description: 指标类型的别名（同 metric_type）
      date:
        type: string
        description: 时间范围的别名（同 time_range）
      ad_group_id:
        type: string
        description: 广告组 ID
      campaign_id:
        type: string
        description: 计划 ID
      # 别名参数（兼容 LLM 可能使用的不同参数名）
      plan_id:
        type: string
        description: 计划 ID 的别名（同 campaign_id）
      dimension:
        type: string
        description: 聚合维度
        default: "day"
    # 外部系统调用配置（可选，用于连接真实数据源）
    external_system: ad_data_platform
    connector_method: query_metrics

# 执行配置
execution:
  default_tool: get_ad_metrics

# 调用示例
examples:
  - user: "查询最近7天的CTR数据"
    arguments:
      metric_type: "ctr"
      time_range: "7d"
      dimension: "day"
  - user: "获取广告组12345的消耗数据"
    arguments:
      metric_type: "cost"
      ad_group_id: "12345"
      time_range: "30d"
  - user: "查看昨天的全量指标"
    arguments:
      metric_type: "all"
      time_range: "1d"
      dimension: "day"
---

# 查询广告指标

这是一个工具执行类技能，用于从广告数据平台查询各类广告投放指标。

## 功能说明

本技能支持查询以下指标：

| 指标类型 | 说明 | 计算公式 |
|---------|------|---------|
| CTR | 点击率 | 点击量 / 展示量 |
| CVR | 转化率 | 转化量 / 点击量 |
| Cost | 消耗 | 累计消耗金额 |
| Impression | 展示量 | 广告展示次数 |
| Click | 点击量 | 广告点击次数 |
| Conversion | 转化量 | 转化事件次数 |

## 使用场景

1. **日常监控**：查询核心指标的日常表现
2. **问题排查**：当指标异常时获取详细数据
3. **报表生成**：批量获取多维度数据用于报表

## 返回数据格式

工具执行后将返回如下格式的数据：

```json
{
  "success": true,
  "data": {
    "metric_type": "ctr",
    "time_range": "7d",
    "dimension": "day",
    "records": [
      {
        "date": "2024-01-01",
        "value": 0.0234,
        "change_rate": 0.05
      }
    ],
    "summary": {
      "avg_value": 0.0256,
      "max_value": 0.0312,
      "min_value": 0.0198
    }
  }
}
```
