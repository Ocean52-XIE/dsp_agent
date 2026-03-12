# 检索回调优补丁说明（2026-03）

本文档说明本轮针对 `query_rewriter` 与 `retrieve_wiki` 的回调优改造点、参数项和评测结果。

## 1. 改造目标

1. 降低“总览文档（00-*）长期占据 Top1”的问题。
2. 保持检索覆盖面，避免过度收敛导致 Recall@K 下滑。
3. 排障问题优先命中模块文档，不被手册类文档单点主导。

## 2. 代码改造点

### 2.1 query_rewriter（`workflow/nodes/query_rewriter/__init__.py`）

1. 增加意图识别 flags（`metric/pipeline/architecture/troubleshoot/code`）。
2. 按意图生成基础 query，避免固定模板导致的宽泛召回。
3. 同义词扩展仅由原始用户问句触发，减少扩展级联噪声。
4. 指标与排障场景补充一条模块别名，平衡精度与召回宽度。

### 2.2 wiki_retriever（`workflow/nodes/retrieve_wiki/wiki_retriever.py`）

1. Stage2 重排新增 `doc_boost`：
- 排障意图命中 `05-*` 文档加分。
- 模块排障命中 `01~04-*` 文档加分。
- 架构意图命中 `00-*` 文档加分。

2. Stage2 重排增强 `doc_penalty`：
- 非架构/非链路/非指标问题，下调 `00-*` 文档权重。
- 排障问题下，对 `00-*` 增加额外惩罚。

3. 默认多样性策略：
- `max_chunks_per_doc` 默认调整为 `1`，降低单文档刷屏。

## 3. 新增或调整的配置参数

1. `WORKFLOW_WIKI_ARCH_DOC_PENALTY`  
说明：非架构类问题下 `00-*` 文档惩罚系数。  
默认：`1.2`

2. `WORKFLOW_WIKI_TROUBLESHOOT_DOC_BOOST`  
说明：排障问题命中 `05-*` 文档加分。  
默认：`0.2`

3. `WORKFLOW_WIKI_MAX_CHUNKS_PER_DOC`  
说明：每个文档最多保留切片数。  
默认：`1`

## 4. 本轮评测结果

使用命令：

```bash
python workflow/eval/run_retrieval_eval.py --config workflow/eval/config.template.json
```

结果：

- `recall@1 = 0.775`
- `recall@3 = 0.925`
- `recall@5 = 0.975`
- `mrr = 0.8625`
- `citation_hit_rate = 0.9`

## 5. 调参建议

1. 如果你希望进一步压低总览文档：
- 适当提高 `WORKFLOW_WIKI_ARCH_DOC_PENALTY`（例如 `1.4`）。

2. 如果你发现排障问题仍偏向手册文档：
- 先降低 `WORKFLOW_WIKI_TROUBLESHOOT_DOC_BOOST`（例如 `0.1`）。
- 再观察 `05-*` 在 Top1 的占比变化。

3. 如果你希望提升 Recall@3：
- 增加 `query_rewriter` 的检索语句数量上限（当前为 `16`）。
- 或增加 `WORKFLOW_WIKI_STAGE2_MIN_CANDIDATES`，扩大重排候选窗口。
