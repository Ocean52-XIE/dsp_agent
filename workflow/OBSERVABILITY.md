# 可观测性与告警（PostgreSQL）

本文档说明如何为当前 Agent 服务启用“观测、记录、告警”能力。

## 1. 功能范围

已实现能力：
- 请求明细落库（query / 路由 / latency / LLM状态 / 引用统计）
- 证据明细落库（每条 citation 单独记录）
- 用户反馈落库（helpful / reason_tag / rating / comment）
- 窗口指标计算与告警事件写入
- 可观测性 API：
  - `GET /api/health`（含 observability 状态）
  - `GET /api/observability/summary?window_minutes=60`
  - `GET /api/observability/alerts?limit=50`
  - `POST /api/messages/{message_id}/feedback`

## 2. 数据库表

启动后会自动建表（若配置开启）：
- `qa_request_log`：请求主表
- `qa_evidence_log`：证据明细
- `qa_feedback_log`：用户反馈
- `qa_alert_event`：告警事件
- `qa_metric_snapshot`：窗口指标快照

## 3. 启动配置

在 `start_agent.ps1` 中配置以下变量（已预留）：

- `OBS_PG_ENABLED`：是否启用（`true/false`）
- `OBS_PG_DSN`：PostgreSQL 连接串  
  示例：`postgresql://user:password@127.0.0.1:5432/dsp_agent`
- `OBS_PG_SCHEMA`：schema 名称（默认 `public`）
- `OBS_PG_CONNECT_TIMEOUT_SECONDS`：连接超时（秒）

告警参数：
- `OBS_ALERT_WINDOW_MINUTES`
- `OBS_ALERT_MIN_SAMPLES`
- `OBS_ALERT_SUPPRESS_MINUTES`
- `OBS_ALERT_EMPTY_RESPONSE_RATE_MAX`
- `OBS_ALERT_FALLBACK_RATE_MAX`
- `OBS_ALERT_INSUFFICIENT_RATE_MAX`
- `OBS_ALERT_P95_LATENCY_MS_MAX`
- `OBS_ALERT_EXACT_LIKE_PASS_RATE_MIN`

对应环境变量名（脚本会自动 export）：
- `WORKFLOW_OBS_PG_ENABLED`
- `WORKFLOW_OBS_PG_DSN`
- `WORKFLOW_OBS_PG_SCHEMA`
- `WORKFLOW_OBS_PG_CONNECT_TIMEOUT_SECONDS`
- `WORKFLOW_OBS_ALERT_*`

## 4. 指标定义（窗口）

当前窗口指标基于 `knowledge_qa` turn：
- `empty_response_rate`：空响应率（`llm_fallback_reason` 以 `empty_answer` 开头）
- `fallback_rate`：fallback 率（`generation_mode != llm`）
- `insufficient_rate`：回答包含“证据不足”占比
- `p95_latency_ms`：P95 延迟（ms）
- `exact_like_pass_rate`：在线近似通过率（启发式，不等于离线 exact_correct）

## 5. 注意事项

- 数据库异常不会中断主链路；可在 `GET /api/health` 查看 `observability.init_error`。
- 需要安装依赖：`psycopg[binary]`（已加入 `requirements.txt`）。
- 若未配置 DSN，系统会自动降级为“不开启落库”模式。

## 6. 自动初始化行为（启动时）

- 服务启动时会先尝试“自动建库”再“自动建表”：
  - 自动建库：若 DSN 指向的数据库不存在，会自动连接 bootstrap 库创建数据库。
  - 自动建表：会自动创建 observability 所需表与索引。
- bootstrap 库默认使用 `postgres`，可通过 `WORKFLOW_PG_BOOTSTRAP_DB` 覆盖。
- 若当前账号无 `CREATE DATABASE` 权限，health 中会出现 `observability.init_error`，需由 DBA 预建数据库后重启服务。

## 7. 运行日志文件（Workflow Runtime Logging）

除 PostgreSQL 可观测落库外，系统还支持将关键运行动作写入本地日志文件（默认目录 `logs/`）：

- 工作流主链路开始/结束、耗时与异常
- 各节点执行耗时与关键状态变化（路由、阶段、命中数、证据数）
- API 关键动作（会话创建、消息请求、代码确认、反馈等）

可用环境变量：

- `WORKFLOW_FILE_LOG_ENABLED`：是否启用文件日志（默认 `true`）
- `WORKFLOW_FILE_LOG_LEVEL`：日志级别（默认 `INFO`，可选 `DEBUG/INFO/WARNING/ERROR`）
- `WORKFLOW_FILE_LOG_DIR`：日志目录（默认 `logs`）
- `WORKFLOW_FILE_LOG_FILE`：日志文件名（默认 `workflow.log`）
- `WORKFLOW_FILE_LOG_MAX_BYTES`：单文件滚动大小（默认 `5242880`）
- `WORKFLOW_FILE_LOG_BACKUP_COUNT`：滚动保留份数（默认 `3`）

可通过 `GET /api/health` 的 `runtime_logging` 字段查看当前生效状态与日志文件路径。
