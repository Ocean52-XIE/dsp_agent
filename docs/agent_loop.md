# Agent Loop 模块设计文档

## 1. 概述

`src/agent` 模块是一个通用的 Agent 核心组件库，提供动态工具调用循环、MCP 协议集成、Skill 系统等核心能力。该模块独立于 Workflow 层，可被多种场景复用。

### 1.1 设计目标

- **通用性**：不绑定特定业务场景，可作为底层能力复用
- **可扩展**：支持动态加载工具和技能，无需修改核心代码
- **可观测**：完整的日志和状态追踪能力
- **同步优先**：使用同步 API 简化调用链，通过线程池包装异步操作

### 1.2 模块结构

```
src/agent/
├── __init__.py          # 模块入口，导出公共 API
├── state.py             # Agent 状态定义
├── core/
│   ├── loop.py          # AgentLoop 核心循环
│   └── finalize.py      # 结果整理器
├── llm/
│   ├── client.py        # LLM 客户端（支持 Function Calling）
│   └── config.py        # LLM 配置
├── mcp/
│   ├── client.py        # MCP 客户端（langchain-mcp-adapters 封装）
│   ├── tool_adapter.py  # MCP 工具适配器
│   └── config_loader.py # MCP 配置加载
├── tools/
│   └── registry.py      # 工具注册中心（本地工具 + MCP 工具）
└── skills/
    ├── base.py          # Skill 数据结构定义
    ├── loader.py        # Skill 加载器
    ├── registry.py      # Skill 注册中心
    ├── executor.py      # Skill 执行器
    └── manager.py       # SkillManager（LangChain Tool 封装）
```

---

## 2. 核心组件

### 2.1 AgentLoop（核心循环）

**职责**：执行 LLM + Tool 的动态调用循环，直到任务完成或达到限制。

**核心流程**：
```
┌──────────────────────────────────────────────────────┐
│                    AgentLoop                          │
│                                                       │
│  ┌─────────┐    ┌─────────┐    ┌─────────────────┐  │
│  │ 获取工具 │───>│ 调用 LLM │───>│ 检查是否有工具调用│  │
│  └─────────┘    └─────────┘    └────────┬────────┘  │
│                                          │           │
│                     ┌────────────────────┴──────┐    │
│                     │                           │    │
│               有工具调用                   无工具调用  │
│                     │                           │    │
│                     ▼                           ▼    │
│              ┌──────────┐              ┌──────────┐  │
│              │ 执行工具  │              │ 返回答案  │  │
│              └────┬─────┘              └──────────┘  │
│                   │                                   │
│                   └───────────────────────┐          │
│                                           │          │
│                     ◄─────────────────────┘          │
│                         (循环继续)                    │
└──────────────────────────────────────────────────────┘
```

**关键配置**：
```python
@dataclass
class AgentLoopConfig:
    max_steps: int = 10           # 最大循环步数
    timeout_seconds: int = 120    # 超时时间（秒）
    system_prompt: str = ""       # 系统提示词
```

**返回结果**：
```python
@dataclass
class AgentLoopResult:
    success: bool                    # 是否成功
    answer: str = ""                 # 最终答案
    tool_calls: list[ToolCallRecord] # 工具调用记录
    steps: int = 0                   # 总步数
    error: str = ""                  # 错误信息
    latency_ms: int = 0              # 总耗时
    is_timeout: bool = False         # 是否超时
```

### 2.2 AgentState（状态定义）

采用最小化设计，只保留必要字段：

```python
class AgentState(TypedDict, total=False):
    # 必需输入
    trace_id: str                       # 追踪 ID（日志用）
    user_query: str                     # 用户提示词（已构建）
    history: list[dict[str, Any]]       # 对话历史

    # 可选配置
    tool_whitelist: list[str]           # 可用工具白名单
```

**设计说明**：
- 所有字段可选（`total=False`），使用 `state.get()` 兼容空值
- 循环内部状态（messages, tool_calls）由 AgentLoop 内部管理
- 输出通过 AgentLoopResult 返回，不污染输入状态

### 2.3 ToolRegistry（工具注册中心）

**职责**：统一管理本地工具和 MCP 工具。

**工具分类**：
```
ToolRegistry
├── _local_tools: dict[str, BaseTool]
│   └── 本地 Python 工具（包括 SkillManager）
└── _mcp_tools: dict[str, BaseTool]
    └── MCP 协议工具（通过 MCPToolAdapter 适配）
```

**获取工具优先级**：
1. 白名单过滤（`tool_whitelist`）
2. 工具分组（`tool_groups`）
3. 默认获取所有（`include_local=True, include_mcp=True`）

### 2.4 LLMClient（LLM 客户端）

**职责**：封装 LangChain ChatOpenAI，支持工具调用。

**核心方法**：
- `invoke(message)` - 普通同步调用
- `invoke_with_tools(messages, tools)` - 带工具的同步调用
- `ainvoke(message)` - 异步调用
- `ainvoke_with_tools(messages, tools)` - 带工具的异步调用

**特性**：
- 自动移除 `<thinking>` 标签（兼容 DeepSeek-R1 等模型）
- 详细的请求/响应日志（可配置截断长度）
- 全局单例模式，支持环境变量配置

### 2.5 MCPClient（MCP 客户端）

**职责**：通过 langchain-mcp-adapters 连接多个 MCP Server。

**支持的传输协议**：
- `stdio` - 标准输入输出
- `sse` - Server-Sent Events
- `websocket` - WebSocket

**同步包装**：
```python
# 异步方法
await client.initialize()
result = await client.call_tool("tool_name", args)

# 同步包装
client.initialize_sync()
result = client.call_tool_sync("tool_name", args)
```

---

## 3. Skill 系统

### 3.1 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                       Skill 系统                              │
│                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐  │
│  │ SkillLoader │───>│ SkillRegistry│───>│ SkillExecutor   │  │
│  │ (加载)      │    │ (注册/索引)  │    │ (执行)          │  │
│  └─────────────┘    └─────────────┘    └────────┬────────┘  │
│                                                  │           │
│                              ┌───────────────────┴──────┐    │
│                              │                          │    │
│                        Prompt Skill              Execution Skill
│                              │                          │    │
│                              ▼                          ▼    │
│                      ┌──────────────┐          ┌──────────────┐
│                      │ Jinja2 渲染  │          │ subprocess   │
│                      │ 提示词模板   │          │ stdin 传参   │
│                      └──────────────┘          └──────────────┘
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                    SkillManager                          │ │
│  │            (LangChain Tool 封装，统一入口)                │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Skill 类型

**Prompt Skill**：
- 使用 Jinja2 渲染提示词模板
- 支持参考文档加载（延迟加载）
- 返回渲染后的提示词供 LLM 使用

**Execution Skill**：
- 通过 subprocess 执行命令
- 参数通过 stdin 以 JSON 格式传递
- 支持 json/text/raw 输出格式
- 支持环境变量注入和超时控制

### 3.3 SkillManager

将整个 Skill 系统封装为单个 LangChain Tool，LLM 通过调用 `skill_manager` 工具执行技能。

**优势**：
- 工具列表稳定，不因 Skill 数量变化而膨胀
- 动态扩展：可以热加载新 Skill
- 统一入口：便于添加日志、监控、缓存

**调用流程**：
```
LLM 返回 tool_calls: [{name: 'skill_manager', args: {skill_name: 'query_metrics', ...}}]
    │
    ▼
SkillManager._run(skill_name='query_metrics', query='...', params={...})
    │
    ▼
SkillExecutor.execute(skill, params, query)
    │
    ├── Prompt Skill: Jinja2 渲染 -> 返回提示词
    │
    └── Execution Skill: subprocess 执行 -> 返回数据
```

---

## 4. 与 Workflow 的集成

### 4.1 架构层次

```
┌─────────────────────────────────────────────────────────────────┐
│                        API Layer                                 │
│                     (FastAPI main.py)                            │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Workflow Layer                               │
│                   (LangGraph Engine)                             │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ load_context │──>│intent_routing│──>│   Subgraphs          │  │
│  └──────────────┘  └──────────────┘  │  ┌────────────────┐  │  │
│                                      │  │ knowledge_qa   │  │  │
│                                      │  │ issue_analysis │  │  │
│                                      │  └────────────────┘  │  │
│                                      └──────────┬───────────┘  │
│                                                 │              │
│                                                 ▼              │
│                                      ┌──────────────────────┐ │
│                                      │  finalize_response   │ │
│                                      └──────────────────────┘ │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Agent Layer                                │
│                    (本模块 src/agent)                            │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌───────────────────────┐  │
│  │ AgentLoop   │  │ ToolRegistry│  │    Skill System       │  │
│  │ (核心循环)  │  │ (工具管理)  │  │ (SkillManager 等)     │  │
│  └─────────────┘  └─────────────┘  └───────────────────────┘  │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌───────────────────────┐  │
│  │ LLMClient   │  │ MCPClient   │  │   AgentFinalize       │  │
│  │ (LLM 调用)  │  │ (MCP 协议)  │  │   (结果整理)          │  │
│  └─────────────┘  └─────────────┘  └───────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 BaseAgentLoopNode

Workflow 层通过 `BaseAgentLoopNode` 复用 Agent 能力：

```python
class BaseAgentLoopNode(ABC):
    """Agent Loop 节点基类"""

    def __init__(self, config: AgentLoopNodeConfig):
        # 从全局单例获取依赖
        self._llm_client = get_llm_client()
        self._skill_registry = get_skill_registry()
        self._tool_registry = get_tool_registry()

    def run(self, service, state) -> dict[str, Any]:
        # 1. 提取状态
        # 2. 调用 AgentLoop
        # 3. Fallback 处理
        # 4. 返回结果
```

**子类需要实现**：
- `_get_system_prompt(state)` - 系统提示词
- `_get_user_prompt(state, ...)` - 用户提示词
- `_build_fallback(state, ...)` - Fallback 响应

### 4.3 全局单例模式

所有核心组件采用全局单例，在启动时初始化：

```python
# 启动时初始化（src/init/）
llm_client = LLMClient.from_env()
set_llm_client(llm_client)

skill_registry = SkillRegistry()
skill_registry.load_from_directory(Path("domain/ad_engine"))
set_skill_registry(skill_registry)

tool_registry = ToolRegistry()
tool_registry.register_local_tool(create_skill_manager(skill_registry))
set_tool_registry(tool_registry)
```

---

## 5. 数据流

### 5.1 完整请求流程

```
用户请求 (FastAPI)
    │
    ▼
WorkflowService.run_user_message()
    │
    ├── state: {
    │     trace_id, session_id, user_query, history
    │   }
    │
    ▼
LangGraph 执行
    │
    ├── load_context: 加载上下文
    ├── intent_routing: 意图路由
    │
    ├── knowledge_qa 子图:
    │   ├── query_rewriter: 构建检索查询
    │   ├── retrieve_wiki: Wiki 检索
    │   ├── retrieve_code: 代码检索
    │   ├── merge_evidence: 证据合并
    │   │
    │   └── knowledge_answer (BaseAgentLoopNode):
    │       │
    │       ├── _get_system_prompt(): 获取系统提示词
    │       ├── _get_user_prompt(): 构建用户提示词
    │       │
    │       └── AgentLoop.run():
    │           │
    │           ├── 循环执行:
    │           │   ├── LLMClient.invoke_with_tools()
    │           │   │
    │           │   └── 如果有工具调用:
    │           │       ├── ToolRegistry.get_tool()
    │           │       └── tool.invoke()
    │           │
    │           └── 返回 AgentLoopResult
    │
    └── finalize_response: 构建最终响应
        │
        ▼
返回给用户
```

### 5.2 状态流转

```
WorkflowState (主图)
    │
    ├── LangGraph 自动传递同名字段
    │
    ▼
KnowledgeQAState (子图)
    │
    ├── 子图内部读写
    │
    ▼
AgentState (AgentLoop)
    │
    ├── 最小化输入
    │
    ▼
AgentLoopResult (输出)
    │
    ├── 合并回主图
    │
    ▼
最终响应
```

---

## 6. 配置说明

### 6.1 环境变量

**LLM 配置**（前缀 `AGENT_LLM_`）：
```
AGENT_LLM_MODEL=deepseek-chat
AGENT_LLM_API_KEY=sk-xxx
AGENT_LLM_BASE_URL=https://api.deepseek.com
AGENT_LLM_TEMPERATURE=0.7
AGENT_LLM_MAX_TOKENS=4096
AGENT_LLM_TIMEOUT=60
```

**MCP 配置**（`domain/<domain_id>/mcp_servers.json`）：
```json
{
  "servers": {
    "filesystem": {
      "transport": "stdio",
      "command": "mcp-server-filesystem",
      "args": ["/path/to/root"],
      "enabled": true
    }
  }
}
```

### 6.2 AgentLoopConfig

```python
config = AgentLoopConfig(
    max_steps=10,           # 最大循环步数
    timeout_seconds=120,    # 超时时间
    system_prompt="...",    # 系统提示词
)
```

### 6.3 FinalizeConfig

```python
config = FinalizeConfig(
    max_citations=4,            # 最大引用数量
    include_tool_calls=True,    # 包含工具调用记录
    include_debug_info=False,   # 包含调试信息
    truncate_answer=False,      # 截断答案
    max_answer_length=8000,     # 最大答案长度
)
```

---

## 7. 日志与可观测性

### 7.1 关键日志点

**AgentLoop**：
- `[AgentLoop] 初始化完成` - 启动信息
- `[AgentLoop] 当前可用工具` - 工具列表
- `[AgentLoop] 步骤 X/Y` - 循环进度
- `[AgentLoop] 完成/超时/失败` - 结果状态

**LLMClient**：
- `[LLMClient] request.started` - 请求开始（含工具列表）
- `[LLMClient] request.completed` - 请求完成（含耗时）
- `[LLMClient] request.failed` - 请求失败

**SkillManager**：
- `[SkillManager] 执行技能` - 技能调用
- `[SkillManager] 工具调用结果返回给LLM` - 结果预览

### 7.2 状态追踪

通过 `ToolCallRecord` 记录每次工具调用：

```python
@dataclass
class ToolCallRecord:
    tool_name: str
    arguments: dict[str, Any]
    result: str
    success: bool
    error: str
    latency_ms: int
    timestamp: float
```

---

## 8. 使用示例

### 8.1 直接使用 AgentLoop

```python
from agent import AgentLoop, AgentLoopConfig, AgentState
from agent.tools import get_tool_registry
from agent.llm import get_llm_client

# 创建 AgentLoop
loop = AgentLoop(config=AgentLoopConfig(
    max_steps=5,
    timeout_seconds=60,
    system_prompt="你是一个智能助手。",
))

# 构建状态
state: AgentState = {
    "trace_id": "trace-001",
    "user_query": "查询昨天的 CTR 数据",
    "history": [],
}

# 执行
result = loop.run(state)
print(result.answer)
```

### 8.2 创建自定义 AgentLoop 节点

```python
from workflow.nodes.agent_loop import BaseAgentLoopNode, AgentLoopNodeConfig

class MyCustomNode(BaseAgentLoopNode):
    def __init__(self):
        super().__init__(config=AgentLoopNodeConfig(
            node_name="my_custom_node",
            response_kind="custom",
            enable_skill_tool=True,
            max_iterations=5,
        ))

    def _get_system_prompt(self, state):
        return "你是自定义助手..."

    def _get_user_prompt(self, state, **kwargs):
        return f"问题: {kwargs['user_query']}"

    def _build_fallback(self, state, **kwargs):
        return "抱歉，无法处理您的请求。"
```

---

## 9. 注意事项

1. **同步优先**：AgentLoop 使用同步 API，MCP 异步操作通过线程池包装
2. **全局单例**：所有组件在启动时初始化，运行时直接获取
3. **最小状态**：AgentState 只保留必要字段，避免状态膨胀
4. **工具白名单**：可通过 `tool_whitelist` 限制可用工具
5. **Fallback 机制**：LLM 调用失败时有兜底响应
