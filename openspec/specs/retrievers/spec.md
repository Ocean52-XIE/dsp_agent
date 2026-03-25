# Retrievers Capability Spec

## 概述

多源检索系统，支持 Wiki 文档和代码的混合检索、融合排序和统一工具暴露。

## 核心能力

### 1. Wiki 检索 (wiki/retriever.py)

| 组件 | 说明 |
|------|------|
| `MarkdownWikiRetriever` | Wiki 文档检索器 |
| `search()` | 执行检索（BM25 + Embedding + Lexical 三级召回） |
| `HybridScoreWeights` | 混合评分权重配置 |

**检索流程**:
```
Query -> BM25 召回 -> Embedding 召回 -> Lexical 召回 -> 融合排序 -> 结果
```

**配置字段**:
```python
@dataclass
class WikiRetrieverRuntimeConfig:
    top_k: int = 10
    hybrid_weights: HybridScoreWeights
    reranker_enabled: bool = True
```

### 2. 代码检索 (code/retriever.py)

| 组件 | 说明 |
|------|------|
| `LocalCodeRetriever` | 代码检索器 |
| `search()` | 执行检索（BM25 + Embedding + Pattern 三级召回） |
| `CodeParentChunk` | 父代码块（类/函数级别） |
| `CodeChildChunk` | 子代码块（行级别） |

**代码块结构**:
```python
@dataclass
class CodeParentChunk:
    parent_id: str
    symbol_name: str
    signature: str
    content: str
    start_line: int
    end_line: int
    file_path: str
```

### 3. 融合排序 (orchestration/fusion.py)

| 函数 | 说明 |
|------|------|
| `run()` | 证据融合主函数 |
| `_build_candidate_items()` | 构建候选集并计算融合分数 |
| `_select_with_quota()` | 按配额选择最终结果 |

**融合策略**:
```
┌─────────────────────────────────────────────────────────────┐
│                    Fusion Pipeline                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Wiki Results ──┐                                           │
│                 │     ┌──────────────────┐                  │
│  Code Results ──┼────►│ Build Candidates │                  │
│                 │     └────────┬─────────┘                  │
│                 │              │                            │
│                 │              ▼                            │
│                 │     ┌──────────────────┐                  │
│                 │     │  Calculate       │                  │
│                 │     │  Fusion Score    │                  │
│                 │     │  (grade + intent)│                  │
│                 │     └────────┬─────────┘                  │
│                 │              │                            │
│                 │              ▼                            │
│                 │     ┌──────────────────┐                  │
│                 └────►│ Select by Quota  │                  │
│                       │ (max_per_source) │                  │
│                       └────────┬─────────┘                  │
│                                │                            │
│                                ▼                            │
│                       Final Citations                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**偏差配置**:
```python
GRADE_BIAS = {"high": 1.2, "medium": 1.0, "low": 0.8}
INTENT_SOURCE_BIAS = {"code": 1.1, "wiki": 1.0}
```

### 4. 统一检索工具 (tools/domain_retrieve_tool.py)

| 组件 | 说明 |
|------|------|
| `create_domain_retrieve_tool()` | 创建统一检索工具 |
| `_run_domain_retrieve_async()` | 异步执行检索 |
| `DomainRetrieveInput` | 输入模型 |
| `DomainRetrieveOutput` | 输出模型 |

**输入模型**:
```python
class DomainRetrieveInput:
    query: str
    module_hint: str | None = None
    intent: str | None = None
```

**输出模型**:
```python
class DomainRetrieveOutput:
    citations: list[Citation]
    debug: dict | None = None
```

### 5. 核心组件 (core/)

| 组件 | 说明 |
|------|------|
| `EmbeddingRetriever` | 向量检索器 |
| `CrossEncoderReranker` | Cross-Encoder 重排器 |

## Citation 结构

```python
@dataclass
class Citation:
    source: str           # 来源标识
    source_type: str      # "wiki" | "code"
    path: str             # 文件路径
    title: str | None     # 标题
    section: str | None   # 章节名
    score: float          # 相关性分数
    excerpt: str          # 摘录内容
    symbol_name: str | None    # 符号名（代码）
    start_line: int | None     # 起始行（代码）
    end_line: int | None       # 结束行（代码）
```

## 数据流

```
┌─────────────────────────────────────────────────────────────┐
│                  domain_retrieve_tool                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: query, module_hint, intent                          │
│         │                                                   │
│         ├──────────────────────────────┐                   │
│         │                              │                   │
│         ▼                              ▼                   │
│  ┌─────────────────┐          ┌─────────────────┐         │
│  │  Wiki Flow      │          │  Code Flow      │         │
│  │  (wiki_flow.py) │          │  (code_flow.py) │         │
│  └────────┬────────┘          └────────┬────────┘         │
│           │                            │                   │
│           ▼                            ▼                   │
│  ┌─────────────────┐          ┌─────────────────┐         │
│  │ WikiRetriever   │          │ CodeRetriever   │         │
│  │ .search()       │          │ .search()       │         │
│  └────────┬────────┘          └────────┬────────┘         │
│           │                            │                   │
│           └──────────┬─────────────────┘                   │
│                      │                                     │
│                      ▼                                     │
│           ┌─────────────────┐                              │
│           │   Fusion.run()  │                              │
│           └────────┬────────┘                              │
│                    │                                       │
│                    ▼                                       │
│           Output: Citation[]                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 全局单例

- Wiki 检索器: `retrievers.wiki.retriever.get_wiki_retriever()`
- 代码检索器: `retrievers.code.retriever.get_code_retriever()`

## 约束

1. Citation 必须包含: `source`, `source_type`, `path`, `score`, `excerpt`
2. 代码定位类结果保留: `symbol_name`, `start_line`, `end_line`
3. 检索相关改动统一落在 `src/retrievers/`
4. 新增检索策略需在 `orchestration/` 中实现
