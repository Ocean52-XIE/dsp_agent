# 会话管理（PostgreSQL 持久化）

本文档说明会话管理能力的落地方式：  
在保留现有 API 结构不变的前提下，将会话从“仅内存”升级为“PostgreSQL 优先、内存兜底”。

## 1. 能力目标

- 为每个新会话生成**全局唯一 ID**（`sess_<uuid4>`）。
- 持久化会话基础信息：`id/title/status/created_at/updated_at`。
- 持久化会话消息列表，支持服务重启后恢复上下文。
- 支持按 `message_id` 反查所属会话，覆盖：
  - `POST /api/messages/{message_id}/confirm-code`
  - `POST /api/messages/{message_id}/feedback`

## 2. 表结构

会话表：`qa_session`

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `session_id` | `TEXT PRIMARY KEY` | 会话主键（全局唯一） |
| `title` | `TEXT` | 会话标题 |
| `created_at` | `TIMESTAMPTZ` | 创建时间 |
| `updated_at` | `TIMESTAMPTZ` | 更新时间 |
| `status` | `TEXT` | 会话状态（如 `idle/completed/out_of_scope`） |
| `messages` | `JSONB` | 会话消息数组（按时间顺序） |
| `payload` | `JSONB` | 扩展字段，便于后续平滑演进 |

索引：
- `idx_qa_session_updated_at`（按更新时间倒序列会话）

## 3. 环境变量

会话存储支持独立配置，也支持复用 observability 的 DSN。

优先读取：
1. `WORKFLOW_SESSION_PG_*`
2. `WORKFLOW_OBS_PG_*`

可用变量：
- `WORKFLOW_SESSION_PG_ENABLED`
- `WORKFLOW_SESSION_PG_DSN`
- `WORKFLOW_SESSION_PG_SCHEMA`
- `WORKFLOW_SESSION_PG_CONNECT_TIMEOUT_SECONDS`

如果未显式配置 `WORKFLOW_SESSION_PG_DSN`，会自动回退到 `WORKFLOW_OBS_PG_DSN`。

## 4. 代码位置

- 会话存储实现：`src/workflow/session/postgres_session_store.py`
- API 接入层调用：`src/api/main.py`

## 5. 运行时行为

- 当 `session_store.active=true`：会话读写全部走 PostgreSQL。
- 当 `session_store.active=false`：自动回退内存模式（便于本地开发）。
- 健康检查可查看状态：`GET /api/health` 的 `session_store` 字段。

## 6. 启动自动初始化

- 会话存储在启动阶段会自动执行：
  - 自动建库（若目标数据库不存在）。
  - 自动建表（`qa_session`）。
- 自动建库依赖当前数据库账号具备 `CREATE DATABASE` 权限。
- bootstrap 库默认是 `postgres`，可通过 `WORKFLOW_PG_BOOTSTRAP_DB` 修改。
