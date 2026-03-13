# Hybrid Answer Eval（Wiki + Code）说明

## 1. 数据集

- 文件：`domain/ad_engine/eval/datasets/ad_engine_answer_eval_hybrid.jsonl`
- 样本量：30 条
- 覆盖四类题型（字段 `hybrid_type`）：
  - `formula`（公式问答）
  - `reason`（原因解释）
  - `function_location`（函数定位）
  - `mixed_followup`（混合追问）

每条样本包含：
- `gold_wiki_paths` / `gold_code_paths`
- `expected_sources`（建议为 `["wiki","code"]`）
- `expected_code_symbols`（用于评估答案里的代码锚点覆盖）

另外提供一个 5 条快速验证集（保留大集不变）：
- `domain/ad_engine/eval/datasets/ad_engine_answer_eval_hybrid_smoke5.jsonl`
- 用于日常快速回归，不替代 30 条全量评测。

## 2. 新增 hybrid 专项指标

在 `run_answer_eval.py` 中新增：

- `formula_structure_hit_rate`
  - 仅统计 `hybrid_type=formula`
  - 判断答案是否具备公式结构（如 `=`、`公式`、`bid =` 等）
- `reason_structure_hit_rate`
  - 仅统计 `hybrid_type=reason`
  - 判断答案是否具备原因结构（编号/项目符号/“原因”关键词）
- `location_anchor_hit_rate`
  - 仅统计 `hybrid_type=function_location`
  - 判断答案是否命中代码锚点（函数名或代码路径文件名）
- `mixed_dual_source_hit_rate`
  - 仅统计 `hybrid_type=mixed_followup`
  - 判断是否同时命中期望来源（wiki + code）

同时新增 `hybrid_type_metrics` 分桶结果：
- 每个题型输出 `cases`、`required_fact_coverage`、`expected_mode_accuracy`、`exact_correct_rate`。

## 3. 运行方式

```powershell
.\src\workflow\eval\run_answer_eval.ps1 -ConfigPath src/workflow/eval/config.answer.hybrid.template.json
.\src\workflow\eval\run_answer_eval.ps1 -ConfigPath src/workflow/eval/config.answer.hybrid.smoke5.template.json
```

## 4. 结果解读建议

- 先看全局指标：`overall_score`、`exact_correct_rate`、`required_fact_coverage`
- 再看专项指标：`formula_structure_hit_rate`、`location_anchor_hit_rate` 等
- 最后看分桶：`hybrid_type_metrics`，判断哪一类题型是当前主要短板

## 5. P0-P1 已落地优化（本轮）

- 函数定位链路强化（P0）
  - `query_rewriter`：函数/文件定位问法强制 `code_first`，wiki 仅保底。
  - `merge_evidence`：`is_code_location` 场景提高 code 权重并收紧 wiki 配额。
- 模块路由词典补强（P0）
  - `engine._infer_module` 增加函数符号强绑定：
    - `apply_diversity_penalty` / `apply_safety_penalty` / `rank_topn` -> `rerank-engine`。
- 回答模式稳定（P0）
  - `llm_qa` Prompt 增加软约束：已给出“锚点+依据”时不再附加“当前证据不足”段。
  - `knowledge_answer` 后处理增加冗余“证据不足”段清理逻辑。
- 评测运行稳定性（P0）
  - `run_answer_eval.ps1` 默认 `QA/Judge max_tokens=1600`。
  - `run_answer_eval.py` 为 QA 与 Judge 调用增加 1 次指数退避重试。
- 检索精调（P1）
  - `retrieve_code`：符号精确命中奖励显著高于语义匹配（location query 优先）。
  - `retrieve_wiki`：仅在定位类 query 下进一步下调 `00-总体架构` 权重。
