# RG Strategy Compare (30 Cases)

- Generated: 2026-03-13T15:20:29
- Wiki dataset: `domain/ad_engine/eval/datasets/ad_engine_retrieval_eval_hard_30.jsonl`
- Code dataset: `domain/ad_engine/eval/datasets/ad_engine_code_retrieval_eval_hard_30.jsonl`

## Wiki Metrics
| strategy | recall@1 | recall@3 | recall@5 | mrr |
|---|---|---|---|---|
| rg_first | 0.3 | 0.5 | 0.9 | 0.481429 |
| rg_only | 0.4 | 0.566667 | 0.666667 | 0.497222 |
| no_rg | 0.266667 | 0.5 | 0.933333 | 0.465079 |

## Code Metrics
| strategy | recall@1 | recall@3 | recall@5 | mrr | top1_path_accuracy | symbol_hit_rate@3 | pattern_hit_rate@1 | pattern_hit_rate@3 | highlight_hit_rate@3 | avg_latency_ms |
|---|---|---|---|---|---|---|---|---|---|---|
| rg_first | 0.866667 | 0.966667 | 1.0 | 0.923333 | 0.866667 | 0.7 | 0.766667 | 0.866667 | 0.866667 | 38.553 |
| rg_only | 0.9 | 0.966667 | 0.966667 | 0.927778 | 0.9 | 0.766667 | 0.866667 | 0.966667 | 0.966667 | 28.008 |
| no_rg | 0.8 | 0.966667 | 1.0 | 0.884444 | 0.8 | 0.7 | 0.7 | 0.866667 | 0.866667 | 6.438 |

