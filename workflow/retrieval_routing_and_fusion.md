# 多源路由与融合设计说明

本文档描述当前版本在检索阶段新增的两项能力：

1. 多源路由（`query_rewriter` 产出 `retrieval_plan`）
2. 多源融合（`merge_evidence` 按权重和配额融合证据）

## 1. 核心目标

- 避免“所有来源固定同权重”的粗糙策略。
- 根据问题语义动态偏向 Wiki 或 Code。
- 在融合阶段控制来源配额，避免单源刷屏。
- 提供可诊断的融合画像，便于持续调参。

## 2. retrieval_plan 结构

`query_rewriter` 输出示例：

```json
{
  "strategy": "hybrid",
  "enable_wiki": true,
  "enable_code": true,
  "enable_cases": false,
  "wiki_top_k": 4,
  "code_top_k": 6,
  "final_top_k": 6,
  "max_per_source": {
    "wiki": 4,
    "code": 4,
    "case": 1
  },
  "source_weights": {
    "wiki": 0.95,
    "code": 1.15,
    "case": 0.6
  },
  "intent_profile": {
    "is_code_intent": true,
    "is_code_location": true,
    "is_wiki_intent": false,
    "is_issue_analysis": false
  },
  "reasons": [
    "检测到函数/文件定位意图，强制采用 code_first"
  ]
}
```

## 3. 路由策略

### 3.1 strategy 含义

- `wiki_first`：业务口径/流程/指标类问题优先 Wiki。
- `code_first`：实现定位/报错排查类问题优先 Code。
- `hybrid`：语义混合，双源并行召回。

### 3.3 函数定位专项（P0）

- 对“哪个函数/哪个文件/哪一行/入口函数”等问法，`query_rewriter` 会打上 `is_code_location=true`。
- 该标签会触发：
  - `strategy=code_first`
  - `code_top_k` 提升，`wiki_top_k` 收敛为保底
  - `source_weights` 向 code 倾斜
  - `max_per_source` 中 wiki 配额收敛为 1~2
- 目标是降低泛化 wiki 文档对函数定位问法的干扰。

### 3.2 节点行为

- `retrieve_wiki`：读取 `enable_wiki` 与 `wiki_top_k`。
- `retrieve_code`：读取 `enable_code` 与 `code_top_k`。
- `retrieve_cases`：读取 `enable_cases`，当前默认关闭。

## 4. 融合策略

融合节点 `merge_evidence` 的计算流程：

1. 候选展开  
收集 `wiki_hits`、`code_hits`、`case_hits`。

2. 候选打分  
`fusion_score = base_score * source_weight + rank_bonus + grade_bias + intent_bias`

其中 `intent_bias` 在 `code_location/wiki_first` 场景下生效，用于放大路由意图对最终融合排序的影响。

3. 去重  
按 `(source_type, path, section)` 去重，避免重复证据占位。

4. 配额选择  
先按 `max_per_source` 做第一轮选择，保证多源覆盖。

5. 放宽补齐  
如果仍不足 `final_top_k`，从溢出池按分数补齐。

## 5. 观测字段

融合节点会输出 `evidence_fusion_profile`：

- `strategy`
- `final_top_k`
- `source_weights`
- `max_per_source`
- `input_counts`（各来源输入条数）
- `candidate_count`
- `selected_count`
- `selected_counts_by_source`

每条引用会附带：

- `fusion_score`
- `fusion_rank`
- `fusion_debug`（base_score、权重、grade_bias 等）

## 6. 调参建议

1. 先看召回，再调融合  
若某源召回为空，先优化该源召回，不要直接调融合权重。

2. 先看来源占比，再调权重  
若融合结果单源占比过高，优先下调该源 `source_weights` 或下调 `max_per_source`。

3. 对排障场景单独调参  
`issue_analysis` 可以提高 `code_top_k` 与 code 权重，避免只有口径证据没有实现证据。

4. 启用低置信自动重试  
当 `wiki/code` 首轮结果为 `low/insufficient` 时，先扩 TopK 与扩展查询后再融合，可显著降低漏召回。
