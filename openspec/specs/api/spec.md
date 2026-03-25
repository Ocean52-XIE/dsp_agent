# API Capability Spec

## 概述

FastAPI REST API 接口，提供会话管理、消息处理、反馈收集和观测查询等能力。

## 端点列表

### 健康检查

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 服务健康检查 |

### 会话管理

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/sessions` | 列出所有会话 |
| POST | `/api/sessions` | 创建新会话 |
| GET | `/api/sessions/{session_id}` | 获取指定会话 |

### 消息处理

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/messages` | 发送消息（核心接口） |
| GET | `/api/references/{trace_id}` | 获取引用详情 |

### 反馈

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/messages/{message_id}/feedback` | 提交消息反馈 |

### 配置与观测

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/config` | 获取服务配置 |
| GET | `/api/observability/summary` | 获取观测摘要 |
| GET | `/api/observability/alerts` | 获取告警列表 |

## 请求/响应模型

### MessageCreateRequest

```python
class MessageCreateRequest:
    session_id: str
    content: str
```

### MessageFeedbackRequest

```python
class MessageFeedbackRequest:
    helpful: bool | None = None
    reason_tag: str | None = None
    rating: int | None = None  # 1-5
    comment: str | None = None
```

### Message Response

```python
class MessageResponse:
    session: Session
    summary: str | None
    assistant_message_id: str
    citations: list[Citation]
    analysis: dict | None
    debug: dict | None
```

## 数据流

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI App                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  POST /api/messages                                         │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────────────────────────────┐                   │
│  │       DeepAgentService              │                   │
│  │      run_user_message_async()       │                   │
│  └────────────────┬────────────────────┘                   │
│                   │                                         │
│                   ▼                                         │
│  ┌─────────────────────────────────────┐                   │
│  │       MessageMapper                 │                   │
│  │      to_assistant_message()         │                   │
│  └────────────────┬────────────────────┘                   │
│                   │                                         │
│         ┌────────┼────────┐                                │
│         ▼        ▼        ▼                                │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                   │
│  │ Session  │ │Observab- │ │  HTTP    │                   │
│  │  Store   │ │  ility   │ │ Response │                   │
│  └──────────┘ └──────────┘ └──────────┘                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 生命周期

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时
    await init.initialize_async()
    yield
    # 关闭时清理资源
```

## 依赖关系

```
api
  ├── agent.service (DeepAgentService)
  ├── session (会话存储)
  ├── observability (观测记录)
  └── init (初始化)
```

## 约束

1. `/api/messages` 响应结构必须保持兼容
2. Citation 必须包含: `source`, `source_type`, `path`, `score`, `excerpt`
3. 代码定位类结果尽量保留: `symbol_name`, `start_line`, `end_line`
4. 不在 API 层写业务检索策略
