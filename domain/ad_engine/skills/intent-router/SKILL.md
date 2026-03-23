---
name: intent-router
description: 用于所有广告引擎领域请求。先判断问题是否属于广告引擎领域，再决定它属于知识问答还是问题分析，并给出推荐检索策略。
---

# Intent Router

## 什么时候使用

当用户提出任何广告引擎领域问题时，先使用本 skill 判断：
1. 该问题是否在广告引擎领域范围内
2. 问题类型是 `knowledge_qa` 还是 `issue_analysis`
3. 检索更适合 `wiki_first`、`code_first` 还是 `hybrid`
4. 是否能从问题中识别明确模块名

## 路由规则

### 归为 `knowledge_qa`

满足以下一种或多种情况时，优先归为 `knowledge_qa`：
- 解释概念、指标、口径、流程、架构
- 询问某模块职责、链路、实现位置
- 询问函数、类、文件、参数定义
- 询问“是什么”“怎么实现”“在哪个文件”

### 归为 `issue_analysis`

满足以下一种或多种情况时，优先归为 `issue_analysis`：
- 异常、失败、错误、超时
- 胜率下降、成本超标、掉量、抖动
- 冷启动无量、召回骤降、精排异常
- 明确要求“排查”“定位问题”“给出原因和修复建议”

### 归为超范围

当问题明显不涉及广告投放、召回、排序、两率预估、出价、监控、排障、代码实现时，直接判为超范围。

## 检索策略建议

- `wiki_first`:
  概念解释、架构说明、业务流程、案例经验
- `code_first`:
  代码定位、函数实现、文件位置、参数定义
- `hybrid`:
  问题分析、需要同时参考 wiki 和 code 的复杂请求

## 输出要求

在内部先形成如下判断，再继续调用后续 skill：

```json
{
  "domain_scope": "in_scope | out_of_scope",
  "intent": "knowledge_qa | issue_analysis",
  "retrieval_bias": "wiki_first | code_first | hybrid",
  "module_name": "可选",
  "related_modules": ["可选"]
}
```
