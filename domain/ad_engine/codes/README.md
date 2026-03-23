# Mock Code Corpus（用于 retrieve_code 验证）

该目录是代码检索第一版本的验证语料，不参与生产运行。

## 目录结构

- `recall/recall_service.py`：召回链路入口与候选选择
- `rate/rate_predictor.py`：两率预测与校准
- `bid/bid_optimizer.py`：oCPC 出价与 alpha 调节
- `rerank/rerank_engine.py`：精排打分（含频控/多样性惩罚）
- `common/tracing.py`：trace_id 记录与格式化

## 设计目的

1. 提供稳定的、可控的代码检索评测语料；
2. 避免仓库真实代码变更导致评测基线漂移；
3. 支持“路径命中 + 符号命中 + 公式命中”三类检索验证。

