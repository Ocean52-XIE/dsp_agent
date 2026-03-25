# Agent Capability Spec

## 概述

Deep Agent 运行时封装，提供统一的 Agent 服务入口，支持 MCP 工具集成和检索工具。

## 核心能力

### 1. Agent 服务 (service.py)

| 函数/类 | 说明 |
|---------|------|
| `DeepAgentService` | 核心服务类，封装 Agent 生命周期 |
| `run_user_message_async()` | 处理用户消息的主入口 |
| `_build_messages()` | 构建消息上下文（会话摘要 + 结构化记忆） |
| `_build_degraded_result()` | LLM 错误降级处理 |

### 2. Agent 工厂 (factory.py)

| 函数/类 | 说明 |
|---------|------|
| `create_agent()` | 创建 Deep Agent 实例 |
| `_build_model()` | 构建 ChatOpenAI 模型 |
| `_tool_name()` | 获取工具名称 |
| `_skill_names()` | 获取技能名称列表 |

### 3. 配置管理 (config.py)

| 函数/类 | 说明 |
|---------|------|
| `DeepAgentConfig` | Agent 配置数据类 |
| `from_domain_profile()` | 从领域配置加载配置 |

**配置字段**:
- `model`: 模型名称 (默认: claude-sonnet-4-6)
- `api_key`: API 密钥
- `base_url`: API 基础 URL
- `temperature`: 温度参数
- `max_tokens`: 最大 token 数
- `system_prompt`: 系统提示词
- `skills_root`: 技能根目录

### 4. 结果解析 (result_parser.py)

| 函数/类 | 说明 |
|---------|------|
| `DeepAgentTurnResult` | 标准化结果数据类 |
| `parse_agent_result()` | 解析 Agent 原始结果 |
| `extract_citations()` | 从工具消息提取引用 |
| `extract_answer()` | 提取最终答案 |

**结果字段**:
```python
@dataclass
class DeepAgentTurnResult:
    trace_id: str
    answer: str
    citations: list[Citation]
    message_count: int
    status: str  # "success" | "degraded" | "error"
    driver: str | None
    intent: str | None
    analysis: dict | None
    debug: dict | None
```

### 5. MCP 集成 (mcp/)

| 模块 | 说明 |
|------|------|
| `client.py` | MCPClient - 管理多 Server 连接和工具发现 |
| `config_loader.py` | MCPServerConfigLoader - 加载 MCP Server 配置 |
| `tool_adapter.py` | MCPToolAdapter - 适配 MCP 工具为 LangChain 工具 |

## 数据流

```
┌─────────────────────────────────────────────────────────────┐
│                    DeepAgentService                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  run_user_message_async()                                   │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────────┐                                        │
│  │ _build_messages │ ◄─── Session + ConversationMemory     │
│  └────────┬────────┘                                        │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────┐     ┌──────────────────┐              │
│  │   create_agent  │────►│   ChatOpenAI     │              │
│  └────────┬────────┘     └──────────────────┘              │
│           │               │                                 │
│           │               │  Tools:                         │
│           │               │  ├── domain_retrieve_tool       │
│           │               │  └── MCP tools                  │
│           │               │                                 │
│           ▼                                                 │
│  ┌─────────────────┐                                        │
│  │parse_agent_result│ ───► DeepAgentTurnResult              │
│  └─────────────────┘                                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 依赖关系

```
agent
  ├── domain_profile (配置)
  ├── init (MCP 客户端)
  ├── retrievers (检索工具)
  └── session (会话上下文)
```

## 全局单例

- MCP 客户端: `agent.mcp.get_mcp_client()`

## 约束

1. 新增 Agent tool 必须在 `src/agent/factory.py` 中装配
2. 错误处理必须支持降级模式 (`_build_degraded_result`)
3. Citation 结构必须包含: `source`, `source_type`, `path`, `score`, `excerpt`
