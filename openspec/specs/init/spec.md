# Init Capability Spec

## 概述

统一初始化系统，管理领域配置、MCP、检索器、数据库的初始化顺序和生命周期。

## 核心能力

### 1. 主初始化器 (initializer.py)

| 函数 | 说明 |
|------|------|
| `initialize_async()` | 主初始化入口，返回启动摘要 |
| `_init_retrievers()` | 初始化检索器 |
| `_domain_summary()` | 生成领域配置摘要 |

**初始化顺序**:
```
┌─────────────────────────────────────────────────────────────┐
│                  Initialization Pipeline                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Domain Profile                                          │
│     │                                                       │
│     ▼                                                       │
│  2. Database (Checkpointer)                                 │
│     │                                                       │
│     ▼                                                       │
│  3. MCP System                                              │
│     │                                                       │
│     ▼                                                       │
│  4. Retrievers (Wiki + Code)                                │
│     │                                                       │
│     ▼                                                       │
│  5. Summary Report                                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2. 领域配置初始化 (domain_profile_initializer.py)

| 函数 | 说明 |
|------|------|
| `init_domain_profile()` | 初始化领域配置单例 |

### 3. MCP 初始化 (mcp_initializer.py)

| 函数 | 说明 |
|------|------|
| `initialize_mcp_system_async()` | 初始化 MCP 系统 |
| `init_mcp_client_async()` | 初始化 MCP 客户端 |
| `load_mcp_tools()` | 加载 MCP 工具 |

**MCP 初始化流程**:
```
┌─────────────────────────────────────────────────────────────┐
│                    MCP Initialization                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  domain/<id>/mcp_servers/*.json                             │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────────┐                                        │
│  │MCPServerConfig  │                                        │
│  │Loader           │ 加载配置                               │
│  └────────┬────────┘                                        │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────┐                                        │
│  │MCPClient        │                                        │
│  │.connect()       │ 连接 Server                            │
│  └────────┬────────┘                                        │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────┐                                        │
│  │load_mcp_tools() │ 发现工具                               │
│  └─────────────────┘                                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4. 检索器初始化 (retriever_initializer.py)

| 函数 | 说明 |
|------|------|
| `init_wiki_retriever()` | 初始化 Wiki 检索器 |
| `init_code_retriever()` | 初始化代码检索器 |
| `_summarize_wiki_retriever()` | 生成 Wiki 检索器摘要 |
| `_summarize_code_retriever()` | 生成代码检索器摘要 |

### 5. 数据库初始化 (database_initializer.py)

| 函数 | 说明 |
|------|------|
| `init_database_async()` | 初始化数据库（Checkpointer） |
| `get_database_status()` | 获取数据库状态 |
| `ensure_database_exists()` | 确保数据库存在 |

**数据库状态**:
```python
@dataclass
class DatabaseStatus:
    backend: str      # "postgres" | "memory"
    status: str       # "connected" | "fallback" | "error"
    reason: str | None
    fallback: bool
```

## 初始化摘要

```python
@dataclass
class InitializationSummary:
    domain: dict          # 领域配置摘要
    mcp: dict             # MCP 工具摘要
    retrievers: dict      # 检索器摘要
    database: dict        # 数据库状态
```

## 调用入口

```python
# src/api/main.py
@asynccontextmanager
async def lifespan(app: FastAPI):
    summary = await init.initialize_async()
    logger.info(f"Initialization complete: {summary}")
    yield
    # 清理资源
```

## 依赖关系

```
init
  ├── domain_profile (领域配置)
  ├── agent/mcp (MCP 客户端)
  ├── retrievers (检索器)
  └── session/observability (数据库)
```

## 约束

1. 所有全局初始化必须通过 `src/init/` 统一管理
2. 初始化失败时应有清晰的错误信息
3. 数据库连接失败应支持内存模式降级
4. 初始化顺序不能随意更改（存在依赖关系）
