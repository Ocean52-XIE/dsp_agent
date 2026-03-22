# Code 检索实现文档

> 本文档详细描述了 DSP Agent 中 Code 检索模块的实现架构、核心组件和配置方式。

## 目录

- [1. 概述](#1-概述)
- [2. 架构设计](#2-架构设计)
- [3. 核心组件](#3-核心组件)
- [4. 分块策略](#4-分块策略)
- [5. 检索流程](#5-检索流程)
- [6. 评分机制](#6-评分机制)
- [7. 配置说明](#7-配置说明)
- [8. 简化历程](#8-简化历程)

---

## 1. 概述

Code 检索模块负责从代码库中检索与用户查询相关的代码片段。采用**多级分块 + 精简召回 + 统一评分**的架构。

### 1.1 核心特性

| 特性 | 描述 |
|-----|------|
| **父子分块** | 代码块分为父块（符号级）和子块（滑动窗口），提供多粒度检索 |
| **精简召回** | BM25 + Embedding + Pattern 三路召回，移除冗余路径 |
| **RRF 归一化** | 统一使用 RRF(k=60) 归一化，确保所有分数在 [0, 1] 范围 |
| **符号感知** | 识别函数/类/方法等符号结构 |
| **零外部依赖** | 移除 ripgrep 外部依赖，简化部署 |

### 1.2 文件结构

```
src/workflow/nodes/retrieval_flow/retrieve_code/
├── __init__.py              # 节点入口
└── code_retriever.py        # 核心检索器实现
```

---

## 2. 架构设计

### 2.1 整体流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Code 检索流程（简化版）                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  输入                                                                    │
│  ────                                                                   │
│  user_query: "compute_bid 函数在哪个文件"                                │
│  module_name: "ad-engine"                                               │
│                                                                         │
│                              ↓                                          │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                     代码解析阶段                                  │    │
│  ├─────────────────────────────────────────────────────────────────┤    │
│  │  代码文件 ──→ AST 解析 ──→ 父块(符号) + 子块(滑动窗口)           │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│                              ↓                                          │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                     三路召回阶段                                  │    │
│  ├─────────────────────────────────────────────────────────────────┤    │
│  │  Query ──→ BM25 召回 ──┐                                         │    │
│  │        ──→ Embedding ──┼──→ 候选集合并                           │    │
│  │        ──→ Pattern ────┘                                         │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│                              ↓                                          │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                     子块评分阶段（RRF 归一化）                     │    │
│  ├─────────────────────────────────────────────────────────────────┤    │
│  │  child_score = lexical × 0.20                                    │    │
│  │              + bm25_score × 0.35                                 │    │
│  │              + embedding_score × 0.40                            │    │
│  │              + pattern_score × 0.20                              │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│                              ↓                                          │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                     父块聚合阶段                                  │    │
│  ├─────────────────────────────────────────────────────────────────┤    │
│  │  同一父块的子块 ──→ 聚合打分 ──→ 最终分数                          │    │
│  │  final_score = best_child_score                                  │    │
│  │              + avg_top2_scores × 0.25                            │    │
│  │              + best_pattern × 0.15                               │    │
│  │              + avg_pattern × 0.08                                │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│                              ↓                                          │
│                                                                         │
│  输出                                                                    │
│  ────                                                                   │
│  code_hits: [{path, symbol_name, score, excerpt, ...}]                 │
│  code_retrieval_grade: "high" | "medium" | "low"                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 核心组件

### 3.1 CodeParentChunk 父块

```python
@dataclass
class CodeParentChunk:
    """代码父块（符号级别）"""
    parent_id: str           # 父块唯一标识
    source_path: Path        # 源文件路径
    language: str            # 编程语言
    chunk_type: str          # 类型：function/class/method/file
    symbol_name: str         # 符号名称（函数名/类名）
    signature: str           # 函数签名
    start_line: int          # 起始行号
    end_line: int            # 结束行号
    content: str             # 完整内容
    normalized_text: str     # 归一化文本
    normalized_path: str     # 归一化路径
    normalized_symbol: str   # 归一化符号名
```

### 3.2 CodeChildChunk 子块

```python
@dataclass
class CodeChildChunk:
    """代码子块（滑动窗口）"""
    child_id: str            # 子块唯一标识
    parent_id: str           # 关联的父块 ID
    source_path: Path        # 源文件路径
    language: str            # 编程语言
    chunk_type: str          # 类型
    symbol_name: str         # 符号名称
    signature: str           # 函数签名
    start_line: int          # 起始行号
    end_line: int            # 结束行号
    content: str             # 内容（约 36 行）
    normalized_text: str     # 归一化文本
    normalized_path: str     # 归一化路径
    normalized_symbol: str   # 归一化符号名
```

### 3.3 CodeRetrieverRuntimeConfig 配置（简化版）

```python
@dataclass
class CodeRetrieverRuntimeConfig:
    """Code 检索器运行时配置（简化版）"""
    # 基础配置
    default_top_k: int = 4
    max_child_candidates: int = 64
    max_results_per_path: int = 2

    # BM25 检索权重（RRF 归一化后）
    bm25_weight: float = 0.35

    # 模式匹配权重
    pattern_weight: float = 0.20
    parent_best_pattern_weight: float = 0.15
    parent_avg_pattern_weight: float = 0.08

    # 质量评级阈值
    min_final_score: float = 0.30
    grade_high_top1_threshold: float = 0.85
    grade_medium_top1_threshold: float = 0.55

    # Embedding 配置（默认启用）
    enable_embedding: bool = True
    embedding_model: str = "BAAI/bge-base-zh-v1.5"
    embedding_device: str = "cpu"
    embedding_top_k: int = 4
    embedding_persist_root: str = ".vectorstore_code"
    embedding_weight: float = 0.40

    # RRF 常量：k=60 是信息检索领域验证的标准参数
    RRF_K: int = 60
```

---

## 4. 分块策略

### 4.1 双层分块

```
代码文件
    │
    ├── 父块（符号级）
    │   ├── 函数 compute_bid() [第 10-50 行]
    │   ├── 类 BidCalculator [第 52-120 行]
    │   └── 函数 validate_request() [第 122-145 行]
    │
    └── 子块（滑动窗口）
        ├── child_001: [第 10-45 行] (36行 + 8行重叠)
        ├── child_002: [第 28-63 行]
        └── child_003: [第 46-81 行]
```

**设计理由**：
- **父块**：提供符号级别的上下文，用于最终结果展示
- **子块**：提供细粒度检索，提高召回精度

### 4.2 分块参数

| 参数 | 值 | 说明 |
|------|-----|------|
| CHILD_CHUNK_LINES | 36 | 子块行数 |
| CHILD_CHUNK_OVERLAP | 8 | 子块重叠行数 |
| MAX_FILE_SIZE_BYTES | 256KB | 最大文件大小 |
| EXCERPT_MAX_LINES | 18 | 摘要最大行数 |

---

## 5. 检索流程

### 5.1 三路召回详解

| 路径 | 索引类型 | 归一化 | 说明 |
|------|----------|--------|------|
| **BM25** | LangChain BM25Retriever | RRF(k=60) | 词项精确匹配 |
| **Embedding** | Chroma 向量库 | RRF(k=60) | 语义向量匹配 |
| **Pattern** | 运行时计算 | 直接 [0,1] | 精确标识符匹配 |

### 5.2 召回流程

```
1. BM25 召回
   ────────────
   基于 LangChain BM25Retriever 进行词项匹配
   输出：候选子块列表（按 BM25 分数排序）

2. Embedding 召回
   ────────────
   基于 Chroma 向量库进行语义匹配
   输出：候选子块列表（按向量相似度排序）

3. Pattern 匹配
   ────────────
   基于查询中的标识符进行精确匹配
   匹配维度：符号名、路径、字段名

4. 候选合并
   ────────────
   合并三路召回结果，去重
   使用 RRF(k=60) 对排名进行归一化
```

---

## 6. 评分机制

### 6.1 RRF 归一化

所有检索路径统一使用 RRF(k=60) 归一化：

```python
def _rank_score(self, rank: int | None) -> float:
    """RRF(k=60) 归一化"""
    if rank is None:
        return 0.0
    k = 60
    return k / (rank + k)
```

**RRF 分数范围**：
- Rank 1: 60/61 ≈ 0.984
- Rank 5: 60/65 ≈ 0.923
- Rank 10: 60/70 ≈ 0.857
- Rank 30: 60/90 ≈ 0.667

### 6.2 子块评分公式

```python
# 有 Embedding 时（默认）
child_score = lexical × 0.20              # 词法匹配 [0, 0.20]
            + bm25_score × 0.35           # BM25 [0, 0.35]
            + embedding_score × 0.40      # Embedding [0, 0.40]
            + pattern_score × 0.20        # Pattern [0, 0.20]

# 无 Embedding 时
child_score = lexical × 0.20
            + bm25_score × 0.70           # BM25 权重提升
            + pattern_score × 0.20
```

### 6.3 父块聚合评分

```python
final_score = best_child_score                 # 最佳子块分数
            + avg(top2_scores) × 0.25          # Top2 子块平均
            + best_pattern_score × 0.15        # 最佳模式分数
            + avg(top2_pattern) × 0.08         # Top2 模式平均
```

### 6.4 各分数贡献范围

| 分数组件 | 范围 | 说明 |
|----------|------|------|
| lexical × 0.20 | [0, 0.20] | 词法覆盖率 |
| bm25_score × 0.35 | [0, 0.35] | RRF 归一化后 |
| embedding_score × 0.40 | [0, 0.40] | RRF 归一化后 |
| pattern_score × 0.20 | [0, 0.20] | 精确匹配 |

---

## 7. 配置说明

### 7.1 环境变量

```bash
# 基础配置
export WORKFLOW_CODE_RETRIEVER_TOP_K=4
export WORKFLOW_CODE_RETRIEVER_MAX_CHILD_CANDIDATES=64

# BM25 权重
export WORKFLOW_CODE_RETRIEVER_BM25_WEIGHT=0.35

# 模式匹配权重
export WORKFLOW_CODE_RETRIEVER_PATTERN_WEIGHT=0.20

# Embedding 配置（默认启用）
export WORKFLOW_CODE_EMBEDDING_ENABLED=true
export WORKFLOW_CODE_EMBEDDING_WEIGHT=0.40
export WORKFLOW_CODE_EMBEDDING_MODEL=BAAI/bge-base-zh-v1.5
```

### 7.2 质量评级

| 等级 | Top1 分数阈值 | 说明 |
|------|---------------|------|
| high | ≥ 0.85 | 高置信度命中 |
| medium | ≥ 0.55 | 中等置信度 |
| low | < 0.55 | 低置信度 |

---

## 8. 简化历程

### 8.1 简化前后对比

| 维度 | 简化前 | 简化后 | 减少 |
|------|--------|--------|------|
| **检索路径** | 6 条 | 3 条 | -50% |
| **评分组件** | 7 个 | 4 个 | -43% |
| **配置参数** | 20+ | 12 | -40% |
| **外部依赖** | ripgrep | 无 | -100% |

### 8.2 移除的组件

| 组件 | 移除原因 |
|------|----------|
| **TFIDF** | 与 BM25 功能重复 |
| **Ensemble** | 本身是 BM25+TFIDF 的融合 |
| **RG (ripgrep)** | 外部依赖，Embedding 已能覆盖其功能 |

### 8.3 简化收益

1. **部署简化**：无需安装 ripgrep，纯 Python 实现
2. **参数减少**：调优难度大幅降低
3. **分数统一**：所有路径使用 RRF(k=60) 归一化，分数可比
4. **维护成本**：代码量减少，逻辑更清晰

---

## 附录

### A. 相关文件

| 文件 | 说明 |
|-----|------|
| [code_retriever.py](src/workflow/nodes/retrieval_flow/retrieve_code/code_retriever.py) | 核心检索器 |
| [__init__.py](src/workflow/nodes/retrieval_flow/retrieve_code/__init__.py) | 节点入口 |

### B. 支持的语言

```python
SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx",
    ".go", ".java", ".sql",
    ".yaml", ".yml", ".toml", ".ini", ".conf", ".sh"
}
```

### C. 测试文件

| 文件 | 说明 |
|-----|------|
| [test_code_embedding.py](tests/workflow/nodes/retrieval_flow/retrieve_code/test_code_embedding.py) | Embedding 测试 |
| [test_rrf_normalization.py](tests/workflow/nodes/retrieval_flow/test_rrf_normalization.py) | RRF 归一化测试 |
