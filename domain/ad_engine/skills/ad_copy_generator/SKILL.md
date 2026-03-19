# domain/ad_engine/skills/ad_copy_generator/SKILL.md
---
skill_id: ad_copy_generator
display_name: 广告文案生成器
description: 根据产品信息和目标受众生成吸引人的广告文案
version: 1.0.0
tags:
  - 文案
  - 创意
  - 广告
  - 营销

skill_type: prompt

trigger:
  keywords:
    - 生成文案
    - 写广告
    - 文案创作
    - 广告语
    - 宣传文案
    - 营销文案
  patterns:
    - "帮我.*写.*文案"
    - "生成.*广告"
    - "创作.*文案"
  priority: 20

params:
  product_name:
    type: string
    description: 产品名称
    required: true
  target_audience:
    type: string
    description: 目标受众
    required: true
  style:
    type: string
    description: 文案风格
    enum: [formal, casual, creative]
    default: creative
  key_points:
    type: array
    items:
      type: string
    description: 需要突出的卖点

examples:
  - user: "帮我给新款手机写个广告文案，目标受众是年轻人"
    params:
      product_name: "新款智能手机"
      target_audience: "年轻人"
      style: "creative"
  - user: "生成一个护肤品的正式广告"
    params:
      product_name: "护肤精华"
      style: "formal"
  - user: "帮我写一个运动鞋的营销文案"
    params:
      product_name: "运动跑鞋"
      target_audience: "运动爱好者"
      key_points:
        - "轻便舒适"
        - "减震科技"
        - "时尚设计"
---

你是一位资深的广告文案创意总监，擅长创作引人注目的广告文案。

## 任务
为以下产品创作广告文案：

**产品名称**: {{ product_name }}
**目标受众**: {{ target_audience }}
{% if style %}
**文案风格**: {% if style == 'formal' %}正式专业{% elif style == 'casual' %}轻松活泼{% else %}创意新颖{% endif %}
{% endif %}

{% if key_points %}
**需要突出的卖点**:
{% for point in key_points %}
- {{ point }}
{% endfor %}
{% endif %}

## 要求
1. 文案长度控制在 50-100 字
2. 突出产品核心价值
3. 符合目标受众的语言习惯
4. 具有吸引力和记忆点
5. 避免使用夸张或虚假宣传

## 参考信息
{% if references.copywriting_guide %}
{{ references.copywriting_guide }}
{% endif %}

请直接输出广告文案，无需解释。
