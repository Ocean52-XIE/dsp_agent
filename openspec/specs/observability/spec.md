# Observability Capability Spec

## 概述

观测系统，记录请求日志、证据日志、反馈日志，生成告警和统计摘要。

## 核心能力

### 1. 观测存储 (async_postgres_store.py)

| 组件 | 说明 |
|------|------|
| `PostgresObservabilityStore` | PostgreSQL 观测存储 |
| `PostgresObservabilityConfig` | 配置类 |
| `record_turn()` | 记录单次对话 |
| `record_feedback()` | 记录用户反馈 |
| `get_summary()` | 获取统计摘要 |
| `list_alerts()` | 列出告警 |
| `ensure_schema()` | 创建数据库表结构 |

**配置字段**:
```python
@dataclass
class PostgresObservabilityConfig:
    database_url: str
    table_prefix: str = "qa_"
    pool_min_size: int = 2
    pool_max_size: int = 10
```

## 数据表结构

### qa_request_log

| 字段 | 类型 | 说明 |
|------|------|------|
| session_id | VARCHAR | 会话 ID |
| trace_id | VARCHAR | 追踪 ID |
| message_id | VARCHAR | 消息 ID |
| user_query | TEXT | 用户查询 |
| assistant_kind | VARCHAR | 助手类型 |
| intent | VARCHAR | 意图 |
| status | VARCHAR | 状态 |
| latency_ms | INTEGER | 延迟毫秒 |
| citation_count | INTEGER | 引用数量 |
| created_at | TIMESTAMP | 创建时间 |

### qa_evidence_log

| 字段 | 类型 | 说明 |
|------|------|------|
| trace_id | VARCHAR | 追踪 ID |
| rank_no | INTEGER | 排名 |
| source_type | VARCHAR | 来源类型 (wiki/code) |
| path | VARCHAR | 文件路径 |
| title | VARCHAR | 标题 |
| score | FLOAT | 分数 |
| symbol_name | VARCHAR | 符号名 |
| start_line | INTEGER | 起始行 |
| end_line | INTEGER | 结束行 |
| excerpt | TEXT | 摘录 |

### qa_feedback_log

| 字段 | 类型 | 说明 |
|------|------|------|
| message_id | VARCHAR | 消息 ID |
| helpful | BOOLEAN | 是否有帮助 |
| reason_tag | VARCHAR | 原因标签 |
| rating | INTEGER | 评分 (1-5) |
| comment | TEXT | 评论 |
| created_at | TIMESTAMP | 创建时间 |

### qa_alert_event

| 字段 | 类型 | 说明 |
|------|------|------|
| alert_type | VARCHAR | 告警类型 |
| severity | VARCHAR | 严重程度 |
| metric_name | VARCHAR | 指标名 |
| metric_value | FLOAT | 指标值 |
| threshold | FLOAT | 阈值 |
| created_at | TIMESTAMP | 创建时间 |

### qa_metric_snapshot

| 字段 | 类型 | 说明 |
|------|------|------|
| window_minutes | INTEGER | 时间窗口 |
| sample_size | INTEGER | 样本量 |
| empty_response_rate | FLOAT | 空响应率 |
| fallback_rate | FLOAT | 降级率 |
| p95_latency_ms | INTEGER | P95 延迟 |

## 数据流

```
┌─────────────────────────────────────────────────────────────┐
│                 Observability Pipeline                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  API Request                                                │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              record_turn()                           │   │
│  │                                                      │   │
│  │   ┌──────────────┐  ┌──────────────┐               │   │
│  │   │Request Log   │  │Evidence Log  │               │   │
│  │   │(query, intent│  │(citations,   │               │   │
│  │   │latency...)   │  │scores...)    │               │   │
│  │   └──────────────┘  └──────────────┘               │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  User Feedback                                              │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              record_feedback()                       │   │
│  │                                                      │   │
│  │   ┌──────────────┐                                  │   │
│  │   │Feedback Log  │                                  │   │
│  │   │(helpful,     │                                  │   │
│  │   │rating...)    │                                  │   │
│  │   └──────────────┘                                  │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Periodic Analysis                                          │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         get_summary() / list_alerts()               │   │
│  │                                                      │   │
│  │   ┌──────────────┐  ┌──────────────┐               │   │
│  │   │Metric        │  │Alert Events  │               │   │
│  │   │Snapshot      │  │(threshold    │               │   │
│  │   │              │  │breaches)     │               │   │
│  │   └──────────────┘  └──────────────┘               │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/observability/summary` | 获取统计摘要 |
| GET | `/api/observability/alerts` | 获取告警列表 |

## 依赖关系

```
observability
  └── psycopg_pool (数据库连接池)
```

## 约束

1. 观测数据写入不应影响主请求性能
2. 告警阈值可在 `profile.json` 中配置
3. 数据库表结构变更需要兼容旧数据
