# Session Capability Spec

## 概述

会话管理系统，提供会话持久化、会话摘要和结构化记忆能力。

## 核心能力

### 1. 会话存储 (async_postgres_session_store.py)

| 组件 | 说明 |
|------|------|
| `PostgresSessionStore` | PostgreSQL 会话存储 |
| `PostgresSessionConfig` | 配置类 |
| `save_session()` | 保存会话 |
| `get_session()` | 获取会话 |
| `list_sessions()` | 列出会话 |
| `find_message()` | 查找消息 |

**配置字段**:
```python
@dataclass
class PostgresSessionConfig:
    database_url: str
    table_name: str = "sessions"
    pool_min_size: int = 2
    pool_max_size: int = 10
```

### 2. 结构化记忆 (conversation_memory.py)

| 函数 | 说明 |
|------|------|
| `build_conversation_memory()` | 从对话构建结构化记忆 |
| `render_conversation_memory()` | 渲染记忆为文本（注入 Agent 上下文） |
| `merge_summary_memory_updates()` | 合并 LLM 摘要更新 |
| `normalize_conversation_memory()` | 标准化记忆结构 |

**ConversationMemory 结构**:
```python
@dataclass
class ConversationMemory:
    current_topic: str | None           # 当前话题
    module_name: str | None             # 涉及模块
    related_modules: list[str]          # 相关模块
    entities: list[str]                 # 提及实体
    active_issue: str | None            # 活跃问题
    referenced_paths: list[str]         # 引用路径
    referenced_symbols: list[str]       # 引用符号
    confirmed_facts: list[str]          # 已确认事实
    open_questions: list[str]           # 待解决问题
    last_intent: str | None             # 最近意图
```

### 3. 会话摘要 (conversation_summarizer.py)

| 组件 | 说明 |
|------|------|
| `ConversationSummarizer` | LLM 驱动的会话摘要生成器 |
| `refresh_summary()` | 刷新会话摘要 |
| `ConversationSummaryConfig` | 配置类 |
| `ConversationSummaryUpdate` | 摘要更新结果 |

**摘要更新结构**:
```python
@dataclass
class ConversationSummaryUpdate:
    summary: str                    # 摘要文本
    memory_updates: dict            # 记忆更新
    raw_text: str | None            # 原始 LLM 响应
```

## Session 结构

```python
@dataclass
class Session:
    id: str
    title: str | None
    created_at: datetime
    updated_at: datetime
    status: str  # "active" | "archived"
    messages: list[Message]
    conversation_summary: str | None
    conversation_memory: ConversationMemory | None
```

## 数据流

```
┌─────────────────────────────────────────────────────────────┐
│                    Session Pipeline                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              PostgresSessionStore                    │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │   │
│  │  │  save    │  │  get     │  │  list    │          │   │
│  │  └──────────┘  └──────────┘  └──────────┘          │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           ConversationSummarizer                     │   │
│  │                                                      │   │
│  │   refresh_summary() ──► LLM ──► Summary + Memory     │   │
│  │                                                      │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           ConversationMemory                         │   │
│  │                                                      │   │
│  │   build() ──► normalize() ──► merge() ──► render()   │   │
│  │                                                      │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           ▼                                 │
│              Agent Context (injected)                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 依赖关系

```
session
  ├── domain_profile (配置)
  ├── agent.config (LLM 配置)
  └── psycopg_pool (数据库连接池)
```

## 约束

1. 会话摘要和记忆通过 LLM 生成，需要处理 LLM 调用失败的情况
2. 记忆结构变更需要兼容旧数据
3. 数据库连接池需要正确管理生命周期
