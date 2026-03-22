# Hard 样本三策略差异分析（30 + 30）

## 1) 数据集
- Wiki: `domain/ad_engine/eval/datasets/ad_engine_retrieval_eval_hard_30.jsonl`
- Code: `domain/ad_engine/eval/datasets/ad_engine_code_retrieval_eval_hard_30.jsonl`

## 2) 总体结果（摘录）
### Wiki
- `rg_first`: recall@1=0.300, recall@3=0.500, recall@5=0.900, mrr=0.481
- `rg_only`:  recall@1=0.400, recall@3=0.567, recall@5=0.667, mrr=0.497
- `no_rg`:    recall@1=0.267, recall@3=0.500, recall@5=0.933, mrr=0.465

### Code
- `rg_first`: recall@1=0.867, recall@3=0.967, recall@5=1.000, mrr=0.923, avg_latency_ms=38.553
- `rg_only`:  recall@1=0.900, recall@3=0.967, recall@5=0.967, mrr=0.928, avg_latency_ms=28.008
- `no_rg`:    recall@1=0.800, recall@3=0.967, recall@5=1.000, mrr=0.884, avg_latency_ms=6.438

## 3) 差异原因
### Wiki
1. `rg_only` Top1 更高，但长尾召回更差（recall@5 明显下降）
- 原因: hard 问法里大量是语义改写，`rg_only` 依赖词面命中，容易在 Top1 命中“词面近似文档”，但遇到弱词面样本时无法覆盖到正确文档。
- 证据样本: `hw_rerank_021`, `hw_rerank_022`, `hw_rerank_024`, `hw_rerank_025` 在 `rg_only` 下 recall@5=0，而 `no_rg` 下 recall@5=1。

2. `no_rg` recall@5 最高
- 原因: BM25/TFIDF 的语义近邻能力在弱词面样本上能把正确文档拉进 Top5，但排序头部不一定最好，所以 recall@1 较低。

3. `rg_first` 介于两者之间
- 原因: 语义召回保留了长尾覆盖，`rg` boost 提升了部分头部命中；但在少数样本中 boost 会把词面强但语义偏离的文档推前。

### Code
1. `rg_only` 在 Top1/Pattern/Highlight 更好
- 原因: hard code 样本仍包含大量可匹配的英文技术词（如 `timestamp`, `top_n`, `online`, `rank_score`），`rg` 行级命中对“实现定位”非常直接。

2. `no_rg` recall@5 能保持 1.0，但 Top1 更低
- 原因: 语义检索能把正确文件召回进候选，但头部排序更容易被相邻语义文件干扰（例如 trace/recall/rerank 之间共享词）。

3. `rg_only` 也有脆弱点
- 证据样本: `hc_rate_010` 在 `rg_only` 下 recall@5=0（空结果），而 `rg_first/no_rg` 均 recall@5=1。
- 原因: 该样本查询词更偏抽象表达，`rg` 词面不命中时缺乏语义兜底。

## 4) 结论建议
- Wiki 场景: 以覆盖率为目标优先 `rg_first`（或 `no_rg`），若只追求 Top1 可尝试 `rg_only` 但要接受长尾损失。
- Code 场景: 定位类问题可优先 `rg_first`；`rg_only` 可作为高精度快速模式，但需语义兜底机制防空召回。
