# retrieve_code V1 设计与工程化说明

本文档描述 `retrieve_code` 第一版的工程化实现、运行参数、调试信息与评测方式。

## 1. 模块位置

- 节点入口：`workflow/nodes/retrieve_code/__init__.py`
- 检索器实现：`workflow/nodes/retrieve_code/code_retriever.py`
- 工作流接入：`workflow/engine.py`

## 2. 设计目标

`retrieve_code` 的目标不是“学术级最优检索”，而是先实现一个可落地、可解释、可调优的生产雏形：

1. 可落地：不依赖外部向量库即可运行。
2. 可解释：返回符号、路径、行号、片段高亮，不是黑盒分数。
3. 可调优：核心参数可通过环境变量调节，支持灰度。
4. 可观测：输出检索 profile，便于排查效果退化。

## 3. 检索链路（Parent/Child 混合召回）

代码检索由四步组成：

1. Parent 切块  
按函数/类切 Parent；无法解析时退化到文件级 Parent。

2. Child 切块  
对 Parent 按行窗口切 Child（默认 36 行，重叠 8 行），提高局部命中能力。

3. Child 打分  
综合词法分、TF-IDF 语义分、pattern-aware 分（标识符/公式/定位意图）。

4. Parent 聚合重排  
按 Parent 聚合 Child 命中，使用“最佳子块分 + 平均贡献 + pattern 贡献”得到最终排序。

## 4. 返回结构（核心字段）

`retrieve_code` 输出 `code_hits`，每条命中通常包含：

- `path`：相对仓库路径
- `score`：排序分（用于融合前排序）
- `stage1_score`：阶段一分数（便于调试重排收益）
- `symbol_name` / `signature` / `chunk_type`
- `start_line` / `end_line`
- `excerpt`：带行号文本片段
- `excerpt_lines`：结构化行级片段
- `highlight_lines`：命中高亮行号
- `retrieval_debug`：匹配词、pattern 命中等调试信息

## 5. 工程化收口能力

### 5.1 运行时配置（环境变量）

`CodeRetrieverRuntimeConfig` 支持以下参数：

- `WORKFLOW_CODE_RETRIEVER_DIRS`：代码索引目录（逗号/分号分隔）
  - 未配置时默认索引仓库根目录 `codes/`，不会默认扫描整个仓库
- `WORKFLOW_CODE_RETRIEVER_TOP_K`
- `WORKFLOW_CODE_RETRIEVER_MAX_CHILD_CANDIDATES`
- `WORKFLOW_CODE_RETRIEVER_MAX_PER_PATH`
- `WORKFLOW_CODE_RETRIEVER_SEMANTIC_WEIGHT`
- `WORKFLOW_CODE_RETRIEVER_PATTERN_WEIGHT`
- `WORKFLOW_CODE_RETRIEVER_PARENT_BEST_PATTERN_WEIGHT`
- `WORKFLOW_CODE_RETRIEVER_PARENT_AVG_PATTERN_WEIGHT`
- `WORKFLOW_CODE_RETRIEVER_MIN_FINAL_SCORE`
- `WORKFLOW_CODE_RETRIEVER_GRADE_HIGH_TOP1_THRESHOLD`
- `WORKFLOW_CODE_RETRIEVER_GRADE_MEDIUM_TOP1_THRESHOLD`

### 5.2 检索质量分级

`retrieve_code` 节点会输出 `code_retrieval_grade`：

- `high`：Top1 分数高且命中量足够
- `medium`：Top1 分数中等
- `low`：有命中但相关度弱
- `insufficient`：无命中
- `disabled`：被路由计划关闭

### 5.4 P0 增强（符号索引 + 路径先验 + 动态重试）

1. 符号级索引  
建立 `symbol token -> parent_id` 倒排索引，用于函数/类/字段定位类问题加权。

2. 路径先验  
建立 `path token -> parent_id` 倒排索引，模块名 token 命中路径时提升排序稳定性。

3. 动态 TopK 与低置信重试  
`retrieve_code` 节点在首轮 `low/insufficient` 时自动：
- 扩大 `top_k`
- 加入定位型重试 query（实现/入口/文件路径）
- 采用“质量优先”策略选择首轮或重试结果

相关环境变量：
- `WORKFLOW_CODE_RETRY_TOPK_MULTIPLIER`
- `WORKFLOW_CODE_RETRY_MAX_TOPK`

### 5.3 检索画像

节点输出 `code_retrieval_profile`，通常包含：

- `latency_ms`
- `child_candidates`
- `parent_candidates`
- `selected_count`
- `hits`
- `top_k`
- `strategy`（来自 retrieval_plan）

## 6. 与 Wiki 检索的差异

`retrieve_code` 与 `retrieve_wiki` 的核心差异：

1. 语料结构不同  
Wiki 是段落文本；代码是符号和语法结构，必须保留行号和上下文窗口。

2. 查询意图不同  
代码问题更强调“定位实现位置”和“命中标识符”，pattern 分更关键。

3. 输出要求不同  
代码检索必须提供可复制定位信息（文件/函数/行号/高亮行），而不只是摘要段落。

## 7. 多源融合中的作用

在多源路由与融合版本中：

- `query_rewriter` 生成 `retrieval_plan`（`code_first` / `wiki_first` / `hybrid`）。
- `retrieve_code` 根据计划读取 `code_top_k` 与启停开关。
- `merge_evidence` 使用 `source_weights` 与 `max_per_source` 对 code 证据进行融合排序。

## 8. 评测脚本与指标

### 8.1 入口

- 配置：`workflow/eval/config.code.template.json`
- 数据集：`workflow/eval/datasets/ad_engine_code_retrieval_eval.jsonl`
- 脚本：`workflow/eval/run_code_retrieval_eval.py`
- 启动：`workflow/eval/run_code_retrieval_eval.ps1`

### 8.2 指标

- `recall@K`
- `mrr`
- `top1_path_accuracy`
- `symbol_hit_rate@K`
- `pattern_hit_rate@1` / `pattern_hit_rate@K`
- `highlight_hit_rate@K`
- `avg_latency_ms`

## 9. 下一步建议（V1 -> V2）

1. 接入真实业务代码语料（逐目录灰度索引）。
2. 引入 language-aware parser（多语言符号抽取）。
3. 在融合层引入 query-aware 动态权重学习。
4. 为代码检索单独构建线上回归集与阈值告警。
