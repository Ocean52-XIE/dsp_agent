# retrieve_wiki Hybrid V1 设计说明

本文档描述 `retrieve_wiki` 在本次改造后的 P0 + P1.1 能力。

## 1. 代码位置

- 节点入口：`src/workflow/nodes/retrieve_wiki/__init__.py`
- 检索器实现：`src/workflow/nodes/retrieve_wiki/wiki_retriever.py`

## 2. P0：切片与召回优化

### 2.1 切片优化

1. 标题层级感知切片  
按 Markdown 标题层级维护 section，避免跨章节误拼。

2. 列表/表格/代码块保护  
对 `list/table/code/flow` 块做结构保护，不做粗暴按长度切分。

3. 邻接块回拼  
命中块为列表/表格时，自动拼接“导语块”；命中导语块时可拼接后续列表块。

### 2.2 查询扩展优化

1. 缩写归一（CTR/CVR/pCTR/pCVR/eCPM 等）
2. 同义词扩展（例如“核心链路/主流程/主链路”）
3. 指标词典补充（指标类问题自动补充高频指标术语）

## 3. P1.1：Hybrid 检索（BM25 + 向量 + 规则分）

Stage1 召回分三路：

- BM25 分：词项相关性
- 向量分：TF-IDF 稀疏向量余弦相似度
- 规则分：短语命中、标题/章节命中、模块词命中

融合公式（归一化后）：

`hybrid = w_bm25 * bm25_norm + w_vector * vector + w_lexical * lexical_norm`

然后映射到 `stage1_score`，再进入 Stage2 重排。

## 4. Stage2 重排

在 Stage1 候选窗口中加入：

- intent_score（指标/流程/排障/公式意图）
- title_section_score（标题/章节命中）
- chunk_type_score（块类型偏好）
- doc_penalty（目录块、短块、README 等惩罚）

最终分：

`final_score = stage1_score + intent + title_section + chunk_type - penalty`

## 5. 动态 TopK 与低置信重试（节点层）

`retrieve_wiki` 节点在首轮低置信时自动：

1. 扩大 `top_k`
2. 构造重试查询（模块核心链路/关键指标/口径）
3. 比较首轮与重试结果，按质量优先取更优结果

环境变量：

- `WORKFLOW_WIKI_RETRY_TOPK_MULTIPLIER`
- `WORKFLOW_WIKI_RETRY_MAX_TOPK`
- `WORKFLOW_WIKI_RETRY_MIN_TOP1`

## 6. Hybrid 权重配置

默认权重：

- `bm25=0.25`
- `vector=0.15`
- `lexical=0.60`

可配置方式：

1. 直接环境变量
- `WORKFLOW_WIKI_WEIGHT_BM25`
- `WORKFLOW_WIKI_WEIGHT_VECTOR`
- `WORKFLOW_WIKI_WEIGHT_LEXICAL`

2. 权重文件（推荐）
- `WORKFLOW_WIKI_HYBRID_WEIGHTS_PATH`

权重文件示例见：

`src/workflow/eval/wiki_hybrid_weights.template.json`

## 7. 回调优补丁（2026-03）

为解决“总览文档长期占据 Top1、排障问句被总览或手册误导”的问题，`wiki_retriever` 新增了文档先验重排能力：

1. 新增 `doc_boost` 正向加分
- 排障意图命中 `05-*` 文档时增加加分（可配置）。
- 链路意图命中 `01~04-*` 模块文档时增加加分。
- 架构意图命中 `00-*` 文档时增加加分。

2. 新增 `doc_penalty` 惩罚项
- 非架构/非链路问题下，降低 `00-*` 总览文档的排序权重。
- 排障问题下，对 `00-*` 追加惩罚，避免其压过模块排障信息。

3. 默认路径去重策略调整
- `max_chunks_per_doc` 默认从 `2` 调整为 `1`，优先提升路径级多样性。

4. 查询扩展收敛
- 同义词扩展仅由原始用户问句触发，避免模板 query 级联扩展导致召回漂移。

### 新增可配置参数

- `WORKFLOW_WIKI_ARCH_DOC_PENALTY`：非架构问句下 `00-*` 文档惩罚系数（默认 `1.2`）。
- `WORKFLOW_WIKI_TROUBLESHOOT_DOC_BOOST`：排障问句命中 `05-*` 文档时加分（默认 `0.2`）。
- `WORKFLOW_WIKI_MAX_CHUNKS_PER_DOC`：每文档最多保留切片数（默认 `1`）。
