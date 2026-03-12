# Knowledge QA LLM 接入说明

本文档说明 `knowledge_answer` 节点的 LLM 接入配置、降级机制，以及本轮新增的 P0 可靠性增强。

## 1. 代码位置

- 节点实现：`workflow/nodes/knowledge_answer/__init__.py`
- LLM 调用层：`workflow/nodes/knowledge_answer/llm_qa.py`
- 工作流初始化：`workflow/engine.py`（`WorkflowService.__init__`）

## 2. 关键行为

- LLM 优先：节点默认优先调用 LLM 生成答案。
- 自动降级：未配置 API Key、网络错误、超时、空回答时，自动降级为规则兜底。
- 多源证据：LLM 输入优先使用融合后的 `citations`（wiki + code + case），不再只依赖 `wiki_hits`。
- 代码定位兜底（P0）：
  - 当用户问题属于“函数/文件/行号定位”且存在 `code` 证据时，
  - 如果 LLM 回答未给出代码锚点（路径/符号/行号），系统会强制降级到“代码定位模板答案”。

## 3. Prompt 强约束（P0）

`llm_qa.py` 新增了场景化中文 Prompt 约束：
- “pCTR/pCVR 校准看什么”问题：要求至少覆盖 2 个校准指标关键词（AUC、LogLoss、Calibration Error、线上CTR/CVR偏差）。
- “target_cpa/pCVR 出价公式 + 函数定位”问题：要求优先输出入口函数 `compute_bid_for_request` 和文件路径。

## 4. 后处理兜底（P0）

`knowledge_answer/__init__.py` 新增后处理补齐逻辑：
- 校准指标覆盖兜底：若答案未出现校准指标关键词，则自动追加“校准指标补充”段。
- 出价入口函数覆盖兜底：若答案未出现 `compute_bid_for_request`，则自动追加“代码锚点补充”段（优先给入口函数）。

## 5. 环境变量

| 参数名 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `WORKFLOW_QA_LLM_ENABLED` | 否 | `true` | 是否启用 LLM 问答 |
| `WORKFLOW_QA_LLM_BASE_URL` | 否 | `https://api.openai.com/v1` | OpenAI 兼容接口地址前缀 |
| `WORKFLOW_QA_LLM_API_KEY` | 是（启用时） | 空 | API Key |
| `WORKFLOW_QA_LLM_MODEL` | 否 | `gpt-4.1-mini` | 模型名 |
| `WORKFLOW_QA_LLM_TIMEOUT_SECONDS` | 否 | `20` | 请求超时（秒） |
| `WORKFLOW_QA_LLM_TEMPERATURE` | 否 | `0.2` | 采样温度 |
| `WORKFLOW_QA_LLM_MAX_TOKENS` | 否 | `600` | 最大输出 token |

## 6. 常见降级原因（`analysis.llm_fallback_reason`）

- `llm_disabled`：显式关闭了 QA LLM。
- `missing_api_key`：缺少 API Key。
- `no_evidence_hits`：没有可用证据，拒绝空口作答。
- `timeout` / `http_error` / `unknown_error`：调用失败。
- `empty_answer`：LLM 返回空内容。
- `llm_missing_code_anchor`：代码定位问题中，LLM 未输出代码锚点，触发强制兜底。
