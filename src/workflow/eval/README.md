# Retrieval Eval 使用说明

本目录提供“检索能力量化评估”所需的模板、代码、脚本和样本集。

当前 Wiki 检索链路已接入“二阶段重排”：

1. 第一阶段：关键词召回打分（含短语兜底、N-Gram 回退）；
2. 第二阶段：在候选集上按 query 意图做重排（指标/流程/排障/公式）；
3. 最终输出：再做文档多样性裁剪，得到 TopK。

## 文件结构

- `config.template.json`：评测配置模板
- `wiki_hybrid_weights.template.json`：Wiki Hybrid 权重模板（可用于 `WORKFLOW_WIKI_HYBRID_WEIGHTS_PATH`）
- `datasets/ad_engine_retrieval_eval.jsonl`：评测集样本（基于当前广告引擎 wiki 生成）
- `run_retrieval_eval.py`：评测主程序
- `run_retrieval_eval.ps1`：PowerShell 启动脚本
- `export_wiki_chunks.py`：切片导出工具（每个文档的 chunk 明细）
- `export_wiki_chunks.ps1`：切片导出脚本
- `config.code.template.json`：代码检索评测配置模板
- `datasets/ad_engine_code_retrieval_eval.jsonl`：代码检索评测集
- `run_code_retrieval_eval.py`：代码检索评测主程序
- `run_code_retrieval_eval.ps1`：代码检索评测 PowerShell 启动脚本
- `domain/ad_engine/codes/`：代码检索 mock 语料（稳定回归用）

## 指标定义

- `Recall@K`：TopK 检索结果是否命中 gold 文档（命中记 1，不命中记 0，再求平均）
- `MRR`：首个命中文档排名的倒数平均值
- `citation_hit_rate`：端到端 workflow 最终 `citations` 的 wiki 路径是否命中 gold

## Wiki Hybrid 权重配置

`retrieve_wiki` 已支持 BM25 + 向量 + 规则分的 Hybrid 融合，可通过以下方式调权：

1. 直接设置环境变量：
- `WORKFLOW_WIKI_WEIGHT_BM25`
- `WORKFLOW_WIKI_WEIGHT_VECTOR`
- `WORKFLOW_WIKI_WEIGHT_LEXICAL`
- `WORKFLOW_WIKI_ARCH_DOC_PENALTY`
- `WORKFLOW_WIKI_ARCH_DOC_LOCATION_PENALTY`（函数定位类 query 下额外压低 00-总体架构）

2. 使用权重文件：
- `WORKFLOW_WIKI_HYBRID_WEIGHTS_PATH`
- 模板文件：`src/workflow/eval/wiki_hybrid_weights.template.json`

## 运行方式

### 方式 1：直接运行 Python

```powershell
python src/workflow/eval/run_retrieval_eval.py --config src/workflow/eval/config.template.json
```

### 方式 2：运行 PowerShell 脚本

```powershell
.\src\workflow\eval\run_retrieval_eval.ps1
```

## 输出

评测报告会写入配置中的 `output_path`，默认：

`src/workflow/eval/results/latest_eval_report.json`

报告包含：

- 总样本数
- top_ks 配置
- 聚合指标（Recall@K、MRR、citation_hit_rate）
- 每条样本的检索结果与命中情况（可选）

说明：`top_hits` 中的 wiki 结果会额外包含 `stage1_score` 与 `rerank_features`，
可用于分析“是召回没命中，还是重排权重不合理”。

## 切片导出

### Python 方式

```powershell
python src/workflow/eval/export_wiki_chunks.py --wiki-dir domain/ad_engine/wiki
```

### PowerShell 方式

```powershell
.\src\workflow\eval\export_wiki_chunks.ps1
```

默认输出：

- `src/workflow/eval/results/wiki_chunk_export.json`
- `src/workflow/eval/results/wiki_chunk_export.md`

可用于每次切片后快速查看：

- 每个文档切了多少块
- 每块所在 section
- 块类型（paragraph/list/flow/code_or_inline）
- 内容预览

## 代码检索评测（Code Retrieval Eval）

用于评估 `retrieve_code` 第一版本（代码感知切块 + Parent/Child 混合召回）：

### 指标说明

- `recall@K`：TopK 检索结果是否命中 gold 代码路径
- `mrr`：首个命中 gold 路径排名倒数的平均值
- `top1_path_accuracy`：Top1 是否命中 gold 路径
- `symbol_hit_rate@K`：TopK 内是否命中 gold 符号名（函数/类）
- `pattern_hit_rate@1`：Top1 是否覆盖预期关键片段（更关注“首条结果可用性”）
- `pattern_hit_rate@K`：TopK 内是否覆盖预期关键片段（公式/字段/关键词）
- `highlight_hit_rate@K`：TopK 内高亮行（`excerpt_lines.is_hit=true`）是否覆盖预期关键片段
- `avg_latency_ms`：单条查询平均检索时延

### 运行方式

```powershell
python src/workflow/eval/run_code_retrieval_eval.py --config src/workflow/eval/config.code.template.json
```

或

```powershell
.\src\workflow\eval\run_code_retrieval_eval.ps1
```

默认输出：

- `src/workflow/eval/results/latest_code_eval_report.json`

## 答案质量评测（Answer Eval）

仅有 Recall@K/MRR 还不够。为评估“最终回复是否正确、是否符合预期”，新增了答案评测脚本：

- `config.answer.template.json`：答案评测配置模板
- `config.answer.code.template.json`：代码问答专项评测配置模板
- `config.answer.hybrid.template.json`：wiki+code 融合问答评测配置模板
- `config.answer.hybrid.smoke5.template.json`：5 条 hybrid 快速验证配置模板
- `datasets/ad_engine_answer_eval.jsonl`：答案评测样例集
- `datasets/ad_engine_answer_eval_code.jsonl`：代码问答专项样例集
- `datasets/ad_engine_answer_eval_hybrid.jsonl`：wiki+code 融合问答样例集
- `datasets/ad_engine_answer_eval_hybrid_smoke5.jsonl`：5 条 hybrid 快速验证样例集（保留全量集不变）
- `run_answer_eval.py`：答案评测主程序
- `run_answer_eval.ps1`：答案评测 PowerShell 启动脚本

### 指标说明

- `required_fact_coverage`：必答要点覆盖率
- `forbidden_claim_pass_rate`：禁忌结论通过率（未出现错误结论）
- `expected_mode_accuracy`：响应模式准确率（正常回答 / 证据不足）
- `citation_hit_rate`：引用命中率（是否引用到 gold 路径）
- `wiki_citation_hit_rate`：仅在有 `gold_wiki_paths` 的样本上统计 wiki 引用命中率
- `code_citation_hit_rate`：仅在有 `gold_code_paths` 的样本上统计 code 引用命中率
- `expected_source_hit_rate`：仅在有 `expected_sources` 的样本上统计来源覆盖命中率
- `overall_score`：按配置权重聚合后的总分
- `exact_correct_rate`：严格通过率（覆盖率达阈值 + 模式正确 + 禁忌不过线 + 可选引用命中）

新增数据格式（向后兼容旧格式）：
- `gold_wiki_paths`：期望命中的 wiki 路径列表
- `gold_code_paths`：期望命中的 code 路径列表
- `expected_sources`：期望至少出现的来源类型列表（如 `["code"]`）
- `expected_mode` 额外支持 `either`：用于“只评估检索/引用，不强约束回答形态”的场景

### 运行方式

```powershell
python src/workflow/eval/run_answer_eval.py --config src/workflow/eval/config.answer.template.json
```

或

```powershell
.\src\workflow\eval\run_answer_eval.ps1
```

代码专项与融合专项可通过配置切换：

```powershell
.\src\workflow\eval\run_answer_eval.ps1 -ConfigPath src/workflow/eval/config.answer.code.template.json
.\src\workflow\eval\run_answer_eval.ps1 -ConfigPath src/workflow/eval/config.answer.hybrid.template.json
.\src\workflow\eval\run_answer_eval.ps1 -ConfigPath src/workflow/eval/config.answer.hybrid.smoke5.template.json
```

默认输出：

- `src/workflow/eval/results/latest_answer_eval_report.json`

### 可选：LLM 评审器（默认关闭）

在规则指标外，还可以开启 LLM 评审语义正确性（`enable_llm_judge=true`），需要配置以下环境变量：

- `WORKFLOW_EVAL_JUDGE_LLM_ENABLED`
- `WORKFLOW_EVAL_JUDGE_LLM_BASE_URL`
- `WORKFLOW_EVAL_JUDGE_LLM_API_KEY`
- `WORKFLOW_EVAL_JUDGE_LLM_MODEL`
- `WORKFLOW_EVAL_JUDGE_LLM_TIMEOUT_SECONDS`
- `WORKFLOW_EVAL_JUDGE_LLM_TEMPERATURE`
- `WORKFLOW_EVAL_JUDGE_LLM_MAX_TOKENS`

### 评估“回答模型”本身（QA LLM）

`run_answer_eval.ps1` 已支持同时设置 QA LLM 环境变量。若希望评测结果反映“真实 LLM 回答”，请确认：

- `WORKFLOW_QA_LLM_ENABLED=true`
- `WORKFLOW_QA_LLM_API_KEY` 已配置

相关变量：

- `WORKFLOW_QA_LLM_ENABLED`
- `WORKFLOW_QA_LLM_BASE_URL`
- `WORKFLOW_QA_LLM_API_KEY`
- `WORKFLOW_QA_LLM_MODEL`
- `WORKFLOW_QA_LLM_TIMEOUT_SECONDS`
- `WORKFLOW_QA_LLM_TEMPERATURE`
- `WORKFLOW_QA_LLM_MAX_TOKENS`

建议（评测场景）：
- 将 QA/Judge 的 `MAX_TOKENS` 控制在 `1200~1800`，通常足够且更稳定。
- `run_answer_eval.ps1` 已默认设置为 `1600`。

### 评测稳定性（P0）

`run_answer_eval.py` 已支持 QA/Judge 调用重试（指数退避），可通过环境变量配置：

- `WORKFLOW_EVAL_RETRY_COUNT`（默认 `1`，即失败后额外重试一次）
- `WORKFLOW_EVAL_RETRY_BASE_DELAY_MS`（默认 `300`，指数退避基础时长）

## Hybrid 专项指标（新增）

`run_answer_eval.py` 针对 `ad_engine_answer_eval_hybrid.jsonl` 额外输出：

- `formula_structure_hit_rate`
- `reason_structure_hit_rate`
- `location_anchor_hit_rate`
- `mixed_dual_source_hit_rate`
- `hybrid_type_metrics`（按 `formula/reason/function_location/mixed_followup` 分桶）

建议结合 `per_case[].hybrid_type` 与 `per_case[].code_anchor_hit` 一起看，便于快速定位是“内容覆盖问题”还是“代码锚点表达问题”。

## Context Regression (Minimal)

Added a minimal regression check for multi-turn context continuity and out-of-scope state isolation.

- Python:

```powershell
python src/workflow/eval/run_context_regression.py --debug-verbose
```

- PowerShell wrapper:

```powershell
.\src\workflow\eval\run_context_regression.ps1 -DebugVerbose
```

Default report output:

- `src/workflow/eval/results/latest_context_regression_report.json`

## Issue Analysis Eval

Added a dedicated issue-analysis routing evaluation set and runner.

- Dataset:
  - `domain/ad_engine/eval/datasets/ad_engine_issue_analysis_eval.jsonl`
- Config template:
  - `src/workflow/eval/config.issue_analysis.template.json`
- Runner:
  - `python src/workflow/eval/run_issue_analysis_eval.py --config src/workflow/eval/config.issue_analysis.template.json`
- PowerShell:
  - `.\src\workflow\eval\run_issue_analysis_eval.ps1 -ConfigPath src/workflow/eval/config.issue_analysis.template.json`

Default report output:

- `src/workflow/eval/results/latest_issue_analysis_eval_report.json`
