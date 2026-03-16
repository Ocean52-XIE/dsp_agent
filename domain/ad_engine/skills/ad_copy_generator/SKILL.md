---
# 技能元数据
name: ad_copy_generator
description: 根据产品信息和投放目标生成广告文案建议，支持多种文案风格和平台适配。
version: 1.0.0
author: ad_engine_team
tags:
  - 广告文案
  - 创意生成
  - 内容创作
enabled: true

# 技能类型
type: prompt_template

# 触发配置
trigger:
  keywords:
    - 生成文案
    - 广告语
    - 创意文案
    - 文案优化
    - 标题生成
    - 推广语
    - 宣传语
  patterns:
    - "帮我.*文案"
    - "写.*广告"
    - "生成.*标题"
    - "优化.*创意"
  priority: 12

# 参数定义
params:
  required:
    - product_name
    - target_audience
  properties:
    product_name:
      type: string
      description: 产品/服务名称
      required: true
    product_description:
      type: string
      description: 产品/服务详细描述
      default: ""
    target_audience:
      type: string
      description: 目标受众描述
      required: true
    selling_points:
      type: string
      description: 核心卖点，多个卖点用逗号分隔
      default: ""
    tone:
      type: string
      description: 文案风格
      enum:
        - professional   # 专业严谨
        - casual        # 轻松活泼
        - emotional     # 情感共鸣
        - urgent        # 紧迫促销
        - humorous      # 幽默风趣
      default: "professional"
    platform:
      type: string
      description: 投放平台
      enum:
        - feed          # 信息流
        - search        # 搜索广告
        - video         # 视频广告
        - banner        # 横幅广告
        - social        # 社交媒体
      default: "feed"
    max_length:
      type: integer
      description: 文案最大字数限制
      default: 50
    count:
      type: integer
      description: 生成文案数量
      default: 3

# 工具定义（prompt_template 类型通常不需要工具）
tools: []

# 执行配置
execution: {}

# 调用示例
examples:
  - user: "帮我为一款智能手表生成信息流广告文案"
    arguments:
      product_name: "智能运动手表"
      product_description: "支持心率监测、睡眠追踪、运动记录，续航7天"
      target_audience: "25-35岁注重健康的都市白领"
      selling_points: "长续航,精准监测,时尚外观"
      tone: "casual"
      platform: "feed"
      count: 3
  - user: "生成一个电商促销的搜索广告标题"
    arguments:
      product_name: "春季女装新品"
      target_audience: "18-30岁女性"
      tone: "urgent"
      platform: "search"
      max_length: 30
---

# 广告文案生成提示词模板

你是一位资深的广告创意文案专家，拥有丰富的数字营销经验。现在需要根据以下信息生成高质量的广告文案。

## 产品信息

- **产品名称**：{{ product_name }}
{% if product_description %}
- **产品描述**：{{ product_description }}
{% endif %}
{% if selling_points %}
- **核心卖点**：{{ selling_points }}
{% endif %}

## 投放配置

- **目标受众**：{{ target_audience }}
- **投放平台**：{{ platform }}
- **文案风格**：{{ tone }}
- **字数限制**：{{ max_length }} 字以内

## 平台特性参考

{% if platform == "feed" %}
**信息流广告特点**：
- 需要在用户浏览内容时自然融入
- 标题要抓住注意力，正文要简洁有力
- 建议使用数字、疑问句、对比等技巧
{% elif platform == "search" %}
**搜索广告特点**：
- 关键词匹配度高
- 突出卖点和促销信息
- 标题简短有力，通常不超过30字
{% elif platform == "video" %}
**视频广告特点**：
- 前3秒要抓住用户注意力
- 文案要有画面感和节奏感
- 配合视频内容设计悬念或高潮
{% elif platform == "banner" %}
**横幅广告特点**：
- 极简设计，核心信息突出
- 行动号召明确
- 通常只有一句话的空间
{% elif platform == "social" %}
**社交媒体特点**：
- 口语化、接地气
- 可以使用表情和话题标签
- 鼓励互动和分享
{% endif %}

## 风格指南

{% if tone == "professional" %}
**专业严谨风格**：
- 使用准确的数据和专业术语
- 逻辑清晰，条理分明
- 传递信任感和权威感
{% elif tone == "casual" %}
**轻松活泼风格**：
- 使用日常口语和网络流行语
- 营造轻松愉快的氛围
- 拉近与用户的距离
{% elif tone == "emotional" %}
**情感共鸣风格**：
- 触发情感共鸣点
- 讲述场景故事
- 使用感性词汇
{% elif tone == "urgent" %}
**紧迫促销风格**：
- 强调限时、限量
- 使用紧迫性词汇
- 突出优惠力度
{% elif tone == "humorous" %}
**幽默风趣风格**：
- 使用双关、反讽等修辞
- 创造意想不到的转折
- 让用户会心一笑
{% endif %}

## 输出要求

请生成 {{ count }} 条不同的广告文案，每条文案需要包含：

1. **标题**（10-20字，吸引注意力）
2. **正文**（{{ max_length }} 字以内，突出卖点）
3. **行动号召**（引导用户下一步行为）

## 输出格式

```
【文案 1】
标题：xxx
正文：xxx
行动号召：xxx

【文案 2】
标题：xxx
正文：xxx
行动号召：xxx

【文案 3】
标题：xxx
正文：xxx
行动号召：xxx
```

---

## 当前时间

{{ current_time }}

## 用户原始需求

{{ query }}
