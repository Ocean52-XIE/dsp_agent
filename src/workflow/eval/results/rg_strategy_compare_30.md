# RG Strategy Compare (30 Cases)

- Generated: 2026-03-13T15:13:00
- Wiki dataset: `domain/ad_engine/eval/datasets/ad_engine_retrieval_eval_30.jsonl`
- Code dataset: `domain/ad_engine/eval/datasets/ad_engine_code_retrieval_eval_30.jsonl`

## Wiki Metrics
| strategy | recall@1 | recall@3 | recall@5 | mrr |
|---|---|---|---|---|
| rg_first | 0.5 | 0.833333 | 1.0 | 0.667778 |
| rg_only | 0.9 | 0.966667 | 0.966667 | 0.927778 |
| no_rg | 0.466667 | 0.8 | 1.0 | 0.641111 |

## Code Metrics
| strategy | recall@1 | recall@3 | recall@5 | mrr | top1_path_accuracy | symbol_hit_rate@3 | pattern_hit_rate@1 | pattern_hit_rate@3 | highlight_hit_rate@3 | avg_latency_ms |
|---|---|---|---|---|---|---|---|---|---|---|
| rg_first | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.9 | 37.041 |
| rg_only | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.9 | 26.415 |
| no_rg | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.9 | 6.101 |

