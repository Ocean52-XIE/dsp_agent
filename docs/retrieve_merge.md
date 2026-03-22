# 检索融合实现文档

> 本文档详细描述了 DSP Agent 中多源检索融合模块（merge_evidence）的实现架构和融合策略。

## 目录

- [1. 概述](#1-概述)
- [2. 架构设计](#2-架构设计)
- [3. 融合流程](#3-融合流程)
- [4. 分数归一化协调](#4-分数归一化协调)
- [5. 融合评分公式](#5-融合评分公式)
- [6. 配额选择策略](#6-配额选择策略)
- [7. 复杂度分析与简化建议](#7-复杂度分析与简化建议)

---

## 1. 概述

`merge_evidence` 节点负责将 Wiki、Code、Case 三种来源的检索结果融合为统一的证据列表，供下游分析节点使用。

### 1.1 核心职责

| 职责 | 描述 |
|-----|------|
| **多源融合** | 合并 wiki_hits、code_hits、case_hits |
| **分数归一化** | 检测并处理分数范围差异 |
| **融合打分** | 综合分数、排名、质量等级、意图偏置 |
| **配额控制** | 按来源配额选择最终证据 |

### 1.2 文件位置

```
src/workflow/nodes/retrieval_flow/merge_evidence/
└── __init__.py    # 融合节点实现
```

---

## 2. 架构设计

### 2.1 整体流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         证据融合流程                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  输入                                                                    │
│  ────                                                                   │
│  wiki_hits: [{source_type, path, score, ...}]                           │
│  code_hits: [{source_type, path, score, ...}]                           │
│  case_hits: [{source_type, path, score, ...}]                           │
│  retrieval_plan: {strategy, source_weights, max_per_source, ...}        │
│                                                                         │
│                              ↓                                          │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │              Step 1: 分数归一化（智能检测）                        │    │
│  ├─────────────────────────────────────────────────────────────────┤    │
│  │  for each source in [wiki, code, case]:                         │    │
│  │      if 分数已在 [0, 1]:  直接使用（RRF 归一化）                  │    │
│  │      else:              min-max 归一化兜底                       │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│                              ↓                                          │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │              Step 2: 融合打分                                     │    │
│  ├─────────────────────────────────────────────────────────────────┤    │
│  │  fused_score = normalized_score × source_weight                  │    │
│  │              + rank_bonus                                        │    │
│  │              + grade_bias                                        │    │
│  │              + intent_bias                                       │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│                              ↓                                          │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │              Step 3: 配额选择                                     │    │
│  ├─────────────────────────────────────────────────────────────────┤    │
│  │  Round 1: 严格按 source 配额选择，保证多源覆盖                    │    │
│  │  Round 2: 如果不足，从 overflow 补齐                              │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│                              ↓                                          │
│                                                                         │
│  输出                                                                    │
│  ────                                                                   │
│  citations: [{source_type, path, fusion_score, fusion_rank, ...}]       │
│  evidence_fusion_profile: {strategy, selected_counts, ...}              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 融合流程

### 3.1 检索计划归一化

```python
def _normalize_retrieval_plan(state: dict) -> dict:
    """归一化检索计划，确保所有参数都有默认值"""
    return {
        "strategy": "hybrid",           # 融合策略
        "final_top_k": 5,               # 最终返回数量
        "source_weights": {             # 来源权重
            "wiki": 1.0,
            "code": 1.0,
            "case": 0.6,
        },
        "max_per_source": {             # 每来源最大数量
            "wiki": 5,
            "code": 5,
            "case": 1,
        },
        "intent_profile": {},           # 意图配置
    }
```

### 3.2 意图覆盖

```python
# 意图相关的来源偏置
INTENT_SOURCE_BIAS = {
    "code_location": {"code": 0.25, "wiki": -0.1, "case": 0.0},
    "wiki_first": {"wiki": 0.15, "code": -0.05, "case": 0.0},
}

# 质量等级偏置
GRADE_BIAS = {
    "high": 0.35,
    "medium": 0.15,
    "low": 0.0,
    "insufficient": -0.1,
    "disabled": -0.2,
}
```

---

## 4. 分数归一化协调

### 4.1 智能检测机制

```python
def _normalize_scores_within_source(hits: list, source: str) -> list:
    """
    智能分数归一化：
    1. 检测分数是否已在 [0, 1] 范围（RRF 归一化特征）
    2. 已归一化：直接使用
    3. 未归一化：执行 min-max 归一化兜底
    """
    scores = [hit.get("score", 0.0) for hit in hits]
    max_score = max(scores)
    min_score = min(scores)

    # 检测是否已归一化（RRF 后的特征）
    already_normalized = min_score >= -0.001 and max_score <= 1.001

    if already_normalized:
        # 直接使用，只需裁剪浮点误差
        for hit, score in zip(hits, scores):
            hit["normalized_score"] = max(0.0, min(1.0, score))
    else:
        # min-max 归一化兜底
        for hit, score in zip(hits, scores):
            hit["normalized_score"] = (score - min_score) / (max_score - min_score)
```

### 4.2 跨检索器协调

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      跨检索器归一化协调                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Wiki 检索器                           Code 检索器                       │
│  ───────────                           ───────────                       │
│  BM25 ─→ RRF(k=60) ─┐                  BM25 ─→ RRF(k=60) ─┐             │
│  Embed ─→ RRF(k=60) ─┼─→ [0,1]        TFIDF ─→ RRF(k=60) ─┼─→ [0,1]    │
│  Lexical ─→ 直接 ───┘                  Embed ─→ RRF(k=60) ─┤             │
│                                        Lexical ─→ 直接 ───┘              │
│                                                                         │
│                              ↓                                          │
│                                                                         │
│                    merge_evidence 融合节点                               │
│                    ────────────────────                                  │
│                    检测分数是否已归一化                                   │
│                    ├─ 是 [0,1]：直接使用 ✓                               │
│                    └─ 否：min-max 归一化兜底                             │
│                                                                         │
│                              ↓                                          │
│                                                                         │
│                    fused_score = normalized_score × weight + biases     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. 融合评分公式

### 5.1 完整公式

```python
fused_score = normalized_score × source_weight
            + rank_bonus           # 0.12 / rank
            + grade_bias            # 质量等级加成
            + intent_bias           # 意图偏置
```

### 5.2 各组件说明

| 组件 | 范围 | 说明 |
|------|------|------|
| normalized_score | [0, 1] | 检索器输出的归一化分数 |
| source_weight | [0, ∞) | 来源权重（可配置） |
| rank_bonus | [0.12, ∞) | 排名加成：0.12 / rank |
| grade_bias | [-0.2, 0.35] | 质量等级偏置 |
| intent_bias | [-0.1, 0.25] | 意图偏置 |

### 5.3 调试信息

每个证据项包含 `fusion_debug` 字段：

```python
{
    "fusion_debug": {
        "raw_score": 0.85,           # 原始分数
        "normalized_score": 0.85,    # 归一化分数
        "source_weight": 1.0,        # 来源权重
        "rank_bonus": 0.12,          # 排名加成
        "grade": "high",             # 质量等级
        "grade_bias": 0.35,          # 等级偏置
        "intent_bias": 0.0,          # 意图偏置
        "fused_score": 1.32,         # 最终融合分数
    }
}
```

---

## 6. 配额选择策略

### 6.1 两轮选择

```python
# Round 1: 严格配额
for item in candidates:
    source = item["source_type"]
    if source_counts[source] >= max_per_source[source]:
        overflow.append(item)  # 超出配额，放入溢出区
        continue
    selected.append(item)
    source_counts[source] += 1

# Round 2: 溢出补齐
for item in overflow:
    if len(selected) >= final_top_k:
        break
    selected.append(item)
```

### 6.2 去重策略

```python
def _dedup_key(item: dict) -> tuple:
    """去重键：(source_type, path, section)"""
    return (
        item.get("source_type", ""),
        item.get("path", ""),
        item.get("section", ""),
    )
```

---

## 7. 简化实施记录（已完成）

### 7.1 简化前后对比

| 维度 | 简化前 | 简化后 | 减少 |
|------|--------|--------|------|
| **检索路径** | 6 (BM25+TFIDF+Ens+Embed+RG+Pat) | 3 (BM25+Embed+Pat) | -50% |
| **评分组件** | 7 | 4 | -43% |
| **配置参数** | 20+ | 12 | -40% |
| **外部依赖** | ripgrep | 无 | -100% |

### 7.2 已移除的组件

| 组件 | 移除原因 |
|------|----------|
| **TFIDF** | 与 BM25 功能重复，BM25 已能覆盖词项匹配需求 |
| **Ensemble** | 本身是 BM25+TFIDF 的融合，移除 TFIDF 后无意义 |
| **RG (ripgrep)** | 外部依赖，Embedding 的语义理解能力已能覆盖其功能 |

### 7.3 简化收益

1. **部署简化**：无需安装 ripgrep，纯 Python 实现
2. **参数减少**：调优难度大幅降低
3. **分数统一**：所有路径使用 RRF(k=60) 归一化，分数可比
4. **维护成本**：代码量减少，逻辑更清晰

### 7.4 简化后的架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        简化后的检索融合架构                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Code 检索器（简化后）                                                  │
│  ───────────────────────                                               │
│  检索路径：BM25 + Embedding + Pattern                                   │
│  归一化：统一 RRF(k=60)，输出分数 [0, 1]                                 │
│                                                                         │
│  Wiki 检索器                                                            │
│  ───────────                                                             │
│  检索路径：BM25 + Embedding + Lexical                                   │
│  归一化：统一 RRF(k=60)，输出分数 [0, 1]                                 │
│                                                                         │
│                              ↓                                          │
│                                                                         │
│  融合层 (merge_evidence)                                                │
│  ──────────────────────                                                │
│  检测分数是否已归一化 → 直接使用                                         │
│  融合打分：normalized_score × source_weight + rank_bonus + biases        │
│  配额选择：按 source 配额 + 融合分数排序                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.5 后续优化建议

如果担心激进方案影响效果，可以分阶段简化：

**Phase 1**：移除明确冗余的组件
- 移除 Code 的 TFIDF（BM25 已覆盖）
- 移除 Code 的 Ensemble（本身就是 BM25+TFIDF）
- 统一配置默认值

**Phase 2**：评估 RG 的必要性
- 对比 `rg_first` vs `no_rg` 的检索效果
- 如果差异 < 5%，考虑移除 RG

**Phase 3**：简化分块
- 评估父子块的实际收益
- 考虑合并为单层分块

### 7.4 RG 是否可以去掉？

**分析**：

| 维度 | 保留 RG | 移除 RG |
|------|---------|---------|
| **精确匹配** | ✓ 路径/行级精确匹配 | BM25 已能覆盖 |
| **语义理解** | ✗ 无 | Embedding 已能覆盖 |
| **部署复杂度** | 需要安装 ripgrep | 无外部依赖 |
| **性能** | 快（命令行工具） | 略慢（模型推理） |
| **维护成本** | 高（跨平台兼容） | 低 |

**建议**：

1. **如果 Embedding 效果良好**：可以移除 RG
   - Embedding 的语义理解能力可以覆盖 RG 的精确匹配
   - 简化部署和维护

2. **如果 Embedding 效果一般**：保留 RG 作为补充
   - RG 在精确匹配场景下仍有优势
   - 可以设置 `rg_strategy=no_rg` 作为默认，需要时开启

3. **中间方案**：将 RG 功能内嵌
   - 不依赖外部 ripgrep
   - 使用 Python 的 `re` 模块实现类似功能

### 7.5 推荐的简化路线图

```
Phase 1 (立即)              Phase 2 (短期)              Phase 3 (中期)
─────────────              ─────────────              ─────────────
移除 TFIDF                  评估 RG 必要性              简化分块策略
移除 Ensemble               移除或内嵌 RG               统一配置结构
统一 RRF 参数               简化融合层                  性能优化
│                          │                          │
↓                          ↓                          ↓
参数减少 20%               参数减少 40%               参数减少 60%
复杂度降低                 维护成本降低                部署简化
```

---

## 附录

### A. 相关文件

| 文件 | 说明 |
|-----|------|
| [merge_evidence/__init__.py](src/workflow/nodes/retrieval_flow/merge_evidence/__init__.py) | 融合节点实现 |
| [wiki_retriever.py](src/workflow/nodes/retrieval_flow/retrieve_wiki/wiki_retriever.py) | Wiki 检索器 |
| [code_retriever.py](src/workflow/nodes/retrieval_flow/retrieve_code/code_retriever.py) | Code 检索器 |

### B. 配置环境变量

```bash
# 融合层配置
export WORKFLOW_RETRIEVAL_STRATEGY=hybrid
export WORKFLOW_RETRIEVAL_FINAL_TOP_K=5
export WORKFLOW_RETRIEVAL_SOURCE_WIKI_WEIGHT=1.0
export WORKFLOW_RETRIEVAL_SOURCE_CODE_WEIGHT=1.0
export WORKFLOW_RETRIEVAL_SOURCE_CASE_WEIGHT=0.6
```

### C. 输出结构

```python
{
    "citations": [
        {
            "source_type": "wiki",
            "path": "wiki/00-总体架构.md",
            "title": "广告引擎总体架构",
            "section": "召回阶段",
            "score": 0.85,
            "normalized_score": 0.85,
            "fusion_score": 1.32,
            "fusion_rank": 1,
            "fusion_debug": {...}
        }
    ],
    "evidence_fusion_profile": {
        "strategy": "hybrid",
        "final_top_k": 5,
        "input_counts": {"wiki": 8, "code": 12, "case": 2},
        "candidate_count": 22,
        "selected_count": 5,
        "selected_counts_by_source": {"wiki": 2, "code": 2, "case": 1}
    }
}
```
