# Agent Loop 系统潜在问题与优化方案分析

> 基于 `src/agent` 模块及整体架构的深度分析
> 分析日期：2026-03-22

---

## 目录

1. [架构层面问题](#1-架构层面问题)
2. [AgentLoop 问题](#2-agentloop-问题)
3. [LLMClient 问题](#3-llmclient-问题)
4. [ToolRegistry 问题](#4-toolregistry-问题)
5. [Skill 系统问题](#5-skill-系统问题)
6. [MCP Client 问题](#6-mcp-client-问题)
7. [Workflow 集成问题](#7-workflow-集成问题)
8. [可观测性问题](#8-可观测性问题)
9. [优化优先级总结](#9-优化优先级总结)

---

## 1. 架构层面问题

### 1.1 全局单例过度使用

**问题描述**：

几乎所有核心组件都使用全局单例模式：
- `_llm_client` (agent/llm/client.py)
- `_tool_registry` (agent/tools/registry.py)
- `_skill_registry` (agent/skills/registry.py)
- `_mcp_client` (agent/mcp/client.py)
- `_skill_executor` (agent/skills/executor.py)

**负面影响**：
- **测试困难**：需要手动 reset 单例，测试间可能相互影响
- **多租户场景无法支持**：无法为不同租户创建独立实例
- **运行时无法切换配置**：初始化后配置即固定

**代码示例**（当前实现）：
```python
# agent/llm/client.py
_llm_client: LLMClient | None = None

def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient.from_env()
    return _llm_client
```

**优化建议**：

使用依赖注入容器替代全局单例：

```python
# 推荐方案：使用 dependency_injector
from dependency_injector import containers, providers

class AgentContainer(containers.DeclarativeContainer):
    """Agent 模块依赖注入容器"""

    config = providers.Configuration()

    llm_client = providers.Singleton(
        LLMClient,
        config=config.llm,
    )

    tool_registry = providers.Singleton(ToolRegistry)

    skill_registry = providers.Singleton(
        SkillRegistry,
        domain_root=config.domain_root,
    )

    mcp_client = providers.Singleton(
        MCPClient.from_domain,
        domain_root=config.domain_root,
    )

    agent_loop = providers.Factory(
        AgentLoop,
        llm_client=llm_client,
        tool_registry=tool_registry,
    )
```

**改造收益**：
- 支持测试时 mock 替换
- 支持多租户隔离
- 依赖关系更清晰

---

### 1.2 同步/异步混合调用

**问题描述**：

`MCPClient` 内部使用 `asyncio`，但对外提供同步包装：

```python
# agent/mcp/client.py
def _run_async(coro: Any) -> Any:
    """在线程池中运行异步协程"""
    executor = _get_mcp_executor()

    def run_in_new_loop():
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    future = executor.submit(run_in_new_loop)
    return future.result()
```

**问题分析**：
1. 每次同步调用都创建新的 event loop，开销大
2. 线程池 `max_workers=4` 可能成为瓶颈
3. `loop.run_until_complete()` 在已有 loop 的上下文中会失败

**优化建议**：

**方案 A**：完全异步化（推荐）
```python
# 所有接口改为 async
class MCPClient:
    async def initialize(self) -> None: ...
    async def call_tool(self, tool_name: str, args: dict) -> MCPToolCallResult: ...

# 调用方也改为异步
async def handle_request():
    await mcp_client.initialize()
    result = await mcp_client.call_tool("tool_name", args)
```

**方案 B**：使用 nest_asyncio
```python
import nest_asyncio
nest_asyncio.apply()

def _run_async(coro: Any) -> Any:
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)
```

**方案 C**：增加线程池配置
```python
_mcp_executor = ThreadPoolExecutor(
    max_workers=int(os.getenv("MCP_THREAD_POOL_SIZE", "8")),
    thread_name_prefix="mcp_sync_"
)
```

---

### 1.3 状态定义分散

**问题描述**：

系统中存在三种状态定义：
- `AgentState` (agent/state.py)
- `WorkflowState` (workflow/state.py)
- `KnowledgeQAState` (workflow/subgraph/knowledge_qa/state.py)

**问题分析**：
- 字段定义可能不一致（如 `trace_id` 在各状态中的定义）
- 维护成本高，修改一处需要同步多处
- 缺少运行时类型校验

**优化建议**：

**方案 A**：统一基础状态 + 继承扩展
```python
from pydantic import BaseModel
from typing import Any

class BaseState(BaseModel):
    """基础状态（所有状态的父类）"""
    trace_id: str = ""
    session_id: str = ""

    class Config:
        extra = "allow"  # 允许额外字段

class AgentState(BaseState):
    """Agent 状态"""
    user_query: str = ""
    history: list[dict[str, Any]] = []
    tool_whitelist: list[str] = []

class WorkflowState(BaseState):
    """工作流状态"""
    user_query: str = ""
    route: str = ""
    node_trace: list[dict[str, str]] = []
```

**方案 B**：使用 TypedDict + 运行时校验
```python
from typing import TypedDict, Required

class AgentState(TypedDict, total=False):
    trace_id: Required[str]
    user_query: Required[str]
    history: list[dict[str, Any]]
    tool_whitelist: list[str]]

def validate_agent_state(state: dict) -> AgentState:
    """运行时校验"""
    if "trace_id" not in state:
        raise ValueError("trace_id is required")
    return cast(AgentState, state)
```

---

## 2. AgentLoop 问题

### 2.1 工具执行串行化

**问题描述**：

当 LLM 返回多个工具调用时，当前实现是串行执行的：

```python
# agent/core/loop.py:217-232
for tool_call in response.tool_calls:
    record = self._execute_tool(tool_call, state)
    tool_calls_history.append(record)

    # 添加工具调用消息
    messages.append({
        "role": "assistant",
        "content": "",
        "tool_calls": [tool_call.to_dict()],
    })
    messages.append({
        "role": "tool",
        "content": record.result,
        "tool_call_id": tool_call.id,
    })
```

**问题分析**：
- 如果 LLM 返回 3 个工具调用，每个耗时 1 秒，总耗时 3 秒
- 无法利用并行能力，性能浪费

**优化建议**：

**并行执行方案**：
```python
import concurrent.futures
from typing import Any

class AgentLoop:
    def _execute_tools_parallel(
        self,
        tool_calls: list[ToolCall],
        state: AgentState,
    ) -> list[ToolCallRecord]:
        """并行执行多个工具调用"""
        if len(tool_calls) <= 1:
            return [self._execute_tool(tc, state) for tc in tool_calls]

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(len(tool_calls), 5)
        ) as executor:
            future_to_tc = {
                executor.submit(self._execute_tool, tc, state): tc
                for tc in tool_calls
            }

            results = []
            for future in concurrent.futures.as_completed(future_to_tc):
                tc = future_to_tc[future]
                try:
                    record = future.result()
                    results.append((tc, record))
                except Exception as e:
                    # 单个失败不影响其他
                    results.append((tc, ToolCallRecord(
                        tool_name=tc.name,
                        arguments=tc.args,
                        success=False,
                        error=str(e),
                    )))

        # 保持原始顺序
        return [r for tc, r in sorted(results, key=lambda x: tool_calls.index(x[0]))]
```

**消息构建调整**：
```python
# 并行执行后批量添加消息
records = self._execute_tools_parallel(response.tool_calls, state)

# 批量构建 assistant 消息
messages.append({
    "role": "assistant",
    "content": "",
    "tool_calls": [tc.to_dict() for tc in response.tool_calls],
})

# 批量添加 tool 响应
for tc, record in zip(response.tool_calls, records):
    tool_calls_history.append(record)
    messages.append({
        "role": "tool",
        "content": record.result,
        "tool_call_id": tc.id,
    })
```

---

### 2.2 消息历史无限增长

**问题描述**：

每轮循环都追加消息，没有截断机制：

```python
# agent/core/loop.py:222-232
messages.append({
    "role": "assistant",
    "content": "",
    "tool_calls": [tool_call.to_dict()],
})
messages.append({
    "role": "tool",
    "content": record.result,  # 可能很长
    "tool_call_id": tool_call.id,
})
```

**问题分析**：
- 10 步循环可能产生 20+ 条消息
- 工具返回结果可能很长（如检索结果）
- 会快速撑爆 context window，导致 400 错误

**优化建议**：

**方案 A**：滑动窗口截断
```python
class AgentLoopConfig:
    max_steps: int = 10
    max_message_pairs: int = 5  # 保留最近 5 轮对话

def _truncate_messages(
    self,
    messages: list[dict],
    max_pairs: int,
) -> list[dict]:
    """保留最近的 N 轮对话"""
    # 保留第一条 system 消息
    system_messages = [m for m in messages if m.get("role") == "system"]
    other_messages = [m for m in messages if m.get("role") != "system"]

    # 保留最近的 N 对 (user + assistant/tool)
    # 每轮包含 1 条 assistant + 1-N 条 tool
    truncated = []
    pair_count = 0

    for msg in reversed(other_messages):
        truncated.insert(0, msg)
        if msg.get("role") == "user":
            pair_count += 1
            if pair_count >= max_pairs:
                break

    return system_messages + truncated
```

**方案 B**：结果长度限制
```python
class AgentLoopConfig:
    max_tool_result_length: int = 2000  # 单个工具结果最大长度

def _execute_tool(self, tool_call: ToolCall, state: AgentState) -> ToolCallRecord:
    record = ...  # 执行工具

    # 截断结果
    if len(record.result) > self.config.max_tool_result_length:
        record.result = record.result[:self.config.max_tool_result_length] + "\n...[truncated]"

    return record
```

**方案 C**：Token 预算控制
```python
import tiktoken

class AgentLoop:
    def __init__(self, ...):
        self._encoder = tiktoken.encoding_for_model("gpt-4")

    def _count_tokens(self, messages: list[dict]) -> int:
        total = 0
        for msg in messages:
            content = msg.get("content", "") or ""
            total += len(self._encoder.encode(content))
        return total

    def _truncate_by_token_budget(
        self,
        messages: list[dict],
        max_tokens: int,
    ) -> list[dict]:
        """按 token 预算截断"""
        while self._count_tokens(messages) > max_tokens:
            # 移除最早的非 system 消息
            for i, msg in enumerate(messages):
                if msg.get("role") != "system":
                    messages.pop(i)
                    break
        return messages
```

---

### 2.3 缺少重试机制

**问题描述**：

当前工具执行失败直接返回错误，没有重试：

```python
# agent/core/loop.py:433-445
except Exception as e:
    latency_ms = int((time.time() - start_time) * 1000)
    error_msg = str(e)
    logger.error(f"[AgentLoop] 工具执行失败: {tool_name}, error={error_msg}")

    return ToolCallRecord(
        tool_name=tool_name,
        arguments=arguments,
        result="",
        success=False,
        error=error_msg,
        latency_ms=latency_ms,
    )
```

**优化建议**：

使用 tenacity 添加重试机制：

```python
from tenacity import (
    retry,
    stop_after_attempt,
    retry_if_exception_type,
    wait_exponential,
    before_sleep_log,
)

class AgentLoopConfig:
    tool_retry_attempts: int = 3
    tool_retry_wait_seconds: float = 1.0

class AgentLoop:
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((TimeoutError, ConnectionError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _execute_tool_with_retry(
        self,
        tool_call: ToolCall,
        state: AgentState,
    ) -> ToolCallRecord:
        """带重试的工具执行"""
        return self._execute_tool(tool_call, state)

    def _execute_tool(self, tool_call: ToolCall, state: AgentState) -> ToolCallRecord:
        """原始工具执行（无重试）"""
        # ... 原有逻辑
```

**LLM 调用重试**：
```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((RateLimitError, ServiceUnavailableError)),
)
def _invoke_llm(self, messages, tool_schemas) -> LLMResponse:
    return self.llm_client.invoke_with_tools(
        messages=messages,
        tools=tool_schemas,
        system_prompt=self.config.system_prompt,
    )
```

---

## 3. LLMClient 问题

### 3.1 缺少速率限制和熔断

**问题描述**：

当前 LLMClient 没有任何速率限制或熔断保护：

```python
# agent/llm/client.py:335-396
def invoke_with_tools(self, messages, tools, system_prompt=None) -> LLMResponse:
    # 直接调用，无任何保护
    response = llm_with_tools.invoke(lc_messages)
```

**问题分析**：
- 并发请求可能导致 429 Too Many Requests
- 服务端故障时会持续重试，雪上加霜
- 没有降级策略

**优化建议**：

**方案 A**：Rate Limiter
```python
from ratelimit import limits, sleep_and_retry

class LLMConfig:
    rate_limit_calls: int = 60
    rate_limit_period: int = 60

class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        self._rate_limiter = RateLimiter(
            calls=config.rate_limit_calls,
            period=config.rate_limit_period,
        )

    @sleep_and_retry
    @limits(calls=60, period=60)
    def invoke_with_tools(self, messages, tools, system_prompt=None) -> LLMResponse:
        ...
```

**方案 B**：Circuit Breaker
```python
from circuitbreaker import circuit

class LLMClient:
    @circuit(failure_threshold=5, recovery_timeout=30)
    def invoke_with_tools(self, messages, tools, system_prompt=None) -> LLMResponse:
        try:
            response = self._llm.bind_tools(tools).invoke(lc_messages)
            return self._parse_response(response)
        except Exception as e:
            if "rate limit" in str(e).lower():
                raise RateLimitError(str(e))
            raise
```

**方案 C**：Token Bucket
```python
import time
from threading import Lock

class TokenBucket:
    def __init__(self, rate: float, capacity: int):
        self.rate = rate  # tokens/second
        self.capacity = capacity
        self.tokens = capacity
        self.last_time = time.time()
        self.lock = Lock()

    def acquire(self, tokens: int = 1) -> bool:
        with self.lock:
            now = time.time()
            elapsed = now - self.last_time
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_time = now

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def wait_and_acquire(self, tokens: int = 1):
        while not self.acquire(tokens):
            time.sleep(0.1)
```

---

### 3.2 Token 计数缺失

**问题描述**：

当前没有 token 计数能力，无法：
- 预估 API 调用成本
- 控制 context window 大小
- 日志中记录 token 使用量

**优化建议**：

集成 tiktoken 进行 token 估算：

```python
import tiktoken
from dataclasses import dataclass

@dataclass
class TokenUsage:
    """Token 使用统计"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        self._encoder = self._get_encoder(config.model)
        self._total_usage = TokenUsage()

    def _get_encoder(self, model: str):
        """获取对应模型的 encoder"""
        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            # 回退到 cl100k_base（GPT-4/3.5 通用）
            return tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, messages: list) -> int:
        """计算消息的 token 数"""
        total = 0
        for msg in messages:
            # 每条消息的基础开销
            total += 4  # role + content 结构

            content = msg.get("content", "") or ""
            if isinstance(content, str):
                total += len(self._encoder.encode(content))
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        total += len(self._encoder.encode(part["text"]))

        return total + 3  # 消息列表的额外开销

    def invoke_with_tools(self, messages, tools, system_prompt=None) -> LLMResponse:
        # 记录输入 token
        prompt_tokens = self._count_tokens(messages)

        response = self._llm.bind_tools(tools).invoke(messages)

        # 记录输出 token
        completion_tokens = len(self._encoder.encode(response.content or ""))

        # 更新统计
        self._total_usage.prompt_tokens += prompt_tokens
        self._total_usage.completion_tokens += completion_tokens
        self._total_usage.total_tokens += prompt_tokens + completion_tokens

        # 日志记录
        logger.info(
            f"[LLMClient] token_usage | "
            f"prompt={prompt_tokens}, completion={completion_tokens}, "
            f"total={self._total_usage.total_tokens}"
        )

        return self._parse_response(response)

    def get_token_usage(self) -> TokenUsage:
        """获取累计 token 使用统计"""
        return TokenUsage(
            prompt_tokens=self._total_usage.prompt_tokens,
            completion_tokens=self._total_usage.completion_tokens,
            total_tokens=self._total_usage.total_tokens,
        )
```

---

## 4. ToolRegistry 问题

### 4.1 工具获取日志过于冗长

**问题描述**：

每次获取工具都打印 INFO 级别日志：

```python
# agent/tools/registry.py:158-188
logger.info(
    f"[ToolRegistry] 获取默认工具: {len(result)} 个, "
    f"工具列表={sorted([t.name for t in result])}"
)
```

**问题分析**：
- AgentLoop 每步都会调用 `get_tools()`
- 高频调用时产生大量日志
- 影响日志可读性

**优化建议**：

```python
import time
from functools import lru_cache

class ToolRegistry:
    def __init__(self):
        self._last_log_time = 0
        self._log_interval = 60  # 60秒内不重复打印

    def get_tools(self, ...) -> list[BaseTool]:
        # ... 获取工具逻辑

        # 降级为 DEBUG 或采样打印
        current_time = time.time()
        if current_time - self._last_log_time > self._log_interval:
            logger.debug(
                f"[ToolRegistry] 获取工具: {len(result)} 个, "
                f"工具列表={sorted([t.name for t in result])}"
            )
            self._last_log_time = current_time

        return result
```

---

### 4.2 工具名称冲突处理

**问题描述**：

工具覆盖时只打印 warning：

```python
# agent/tools/registry.py:74-78
if tool.name in self._local_tools:
    logger.warning(f"[ToolRegistry] 覆盖本地工具: {tool.name}")

self._local_tools[tool.name] = tool
```

**问题分析**：
- MCP 工具和本地工具可能重名
- 意外覆盖可能导致错误行为
- 难以排查问题

**优化建议**：

添加严格模式和命名空间：

```python
class ToolRegistryConfig:
    strict_mode: bool = False  # 严格模式：冲突时抛出异常
    local_prefix: str = ""     # 本地工具前缀
    mcp_prefix: str = "mcp_"   # MCP 工具前缀

class ToolRegistry:
    def __init__(self, config: ToolRegistryConfig = None):
        self.config = config or ToolRegistryConfig()

    def register_local_tool(self, tool: BaseTool) -> None:
        name = tool.name
        if self.config.local_prefix:
            name = f"{self.config.local_prefix}{name}"

        if name in self._local_tools:
            if self.config.strict_mode:
                raise ValueError(f"工具名称冲突: {name}")
            logger.warning(f"[ToolRegistry] 覆盖本地工具: {name}")

        self._local_tools[name] = tool

    def register_mcp_tool(self, tool: BaseTool) -> None:
        # MCP 工具添加前缀，避免与本地工具冲突
        name = tool.name
        if self.config.mcp_prefix:
            name = f"{self.config.mcp_prefix}{name}"

        self._mcp_tools[name] = tool
```

---

## 5. Skill 系统问题

### 5.1 SkillManager 描述过长

**问题描述**：

SkillManager 的 description 是动态生成的，可能非常长：

```python
# agent/skills/manager.py:193-206
desc = f"""执行技能工具。当用户问题涉及以下场景时，**必须**使用此工具。

【触发关键词】
{keyword_hint}

【调用规则】
1. 根据用户问题选择合适的 skill_name
2. **必须**从用户问题中提取参数，通过 params 字段传递
...

【可用技能】
""" + "\n\n".join(skill_details)
```

**问题分析**：
- 如果有 10+ 个技能，描述可能有几千字
- 会消耗大量 token
- 增加 LLM 推理成本

**优化建议**：

**方案 A**：精简描述
```python
def _build_description_static(registry, candidate_skill_ids) -> str:
    """精简版描述"""
    skills = _get_candidate_skills(registry, candidate_skill_ids)

    # 只列出技能名称和简短描述
    skill_list = "\n".join([
        f"- {s.skill_id}: {s.description[:50]}..."
        for s in skills[:5]  # 最多 5 个
    ])

    return f"""执行技能工具。

可用技能:
{skill_list}

参数:
- skill_name: 技能名称
- query: 用户原始问题
- params: 提取的参数（可选）"""
```

**方案 B**：分阶段加载
```python
class SkillManager(BaseTool):
    name = "skill_manager"

    @property
    def description(self) -> str:
        # 返回简短描述
        return "执行技能工具。使用 list_skills 查看可用技能列表。"

    def _run(self, skill_name: str, query: str, params: dict = None, **kwargs) -> str:
        if skill_name == "list_skills":
            # 返回技能列表
            return self._list_skills()
        # 执行具体技能
        return self._execute_skill(skill_name, query, params)
```

---

### 5.2 Execution Skill 安全风险

**问题描述**：

Execution Skill 直接执行命令，没有沙箱隔离：

```python
# agent/skills/executor.py:370-380
result = subprocess.run(
    cmd_args,
    input=params_json,
    capture_output=True,
    text=True,
    encoding='utf-8',
    cwd=cwd,
    env=env,
    timeout=exec_config.timeout,
)
```

**问题分析**：
- 可以执行任意命令
- 环境变量可能包含敏感信息
- 没有权限控制

**优化建议**：

**方案 A**：命令白名单
```python
class ExecutionConfig:
    command: str | list[str]
    allowed_commands: list[str] = []  # 白名单

def _validate_command(self, config: ExecutionConfig) -> None:
    """校验命令是否在白名单中"""
    cmd = config.command[0] if isinstance(config.command, list) else config.command.split()[0]

    if config.allowed_commands and cmd not in config.allowed_commands:
        raise ValueError(f"命令不在白名单中: {cmd}")
```

**方案 B**：Docker 沙箱
```python
import docker

class SkillExecutor:
    def __init__(self):
        self._docker_client = docker.from_env()

    def _execute_in_docker(self, config: ExecutionConfig, params: dict) -> Any:
        """在 Docker 容器中执行"""
        container = self._docker_client.containers.run(
            image="skill-executor:latest",
            command=config.command,
            environment={k: self._mask_secrets(v) for k, v in config.env.items()},
            network_disabled=True,  # 禁用网络
            mem_limit="256m",       # 内存限制
            remove=True,
        )
        return container.decode('utf-8')
```

**方案 C**：敏感信息脱敏
```python
import re

SENSITIVE_PATTERNS = [
    r"sk-[a-zA-Z0-9]{20,}",  # API Keys
    r"[a-zA-Z0-9]{32,}",     # 长字符串（可能是密钥）
    r"password",
    r"secret",
    r"token",
]

def _mask_secrets(self, value: str) -> str:
    """脱敏敏感信息"""
    for pattern in SENSITIVE_PATTERNS:
        value = re.sub(pattern, "***MASKED***", value, flags=re.IGNORECASE)
    return value
```

---

### 5.3 缺少 Skill 版本管理

**问题描述**：

- Skill 加载后无法热更新
- 没有版本回滚能力
- 多版本共存困难

**优化建议**：

```python
from dataclasses import dataclass
from typing import Dict

@dataclass
class SkillVersion:
    """技能版本"""
    skill: Skill
    version: str
    loaded_at: float

class SkillRegistry:
    def __init__(self):
        self.skills: Dict[str, Skill] = {}
        self._versions: Dict[str, list[SkillVersion]] = {}  # skill_id -> versions

    def register(self, skill: Skill) -> None:
        """注册技能（支持版本管理）"""
        if skill.skill_id not in self._versions:
            self._versions[skill.skill_id] = []

        self._versions[skill.skill_id].append(SkillVersion(
            skill=skill,
            version=skill.version,
            loaded_at=time.time(),
        ))

        # 默认使用最新版本
        self.skills[skill.skill_id] = skill

        logger.info(
            f"[SkillRegistry] 注册技能: {skill.skill_id}@{skill.version}"
        )

    def get_skill(self, skill_id: str, version: str = None) -> Skill | None:
        """获取指定版本的技能"""
        if version is None:
            return self.skills.get(skill_id)

        versions = self._versions.get(skill_id, [])
        for v in versions:
            if v.version == version:
                return v.skill
        return None

    def rollback(self, skill_id: str) -> bool:
        """回滚到上一版本"""
        versions = self._versions.get(skill_id, [])
        if len(versions) < 2:
            return False

        # 移除最新版本
        versions.pop()
        # 设置为上一版本
        self.skills[skill_id] = versions[-1].skill

        logger.info(
            f"[SkillRegistry] 回滚技能: {skill_id} -> {versions[-1].version}"
        )
        return True

    def reload(self, skill_id: str) -> bool:
        """重新加载技能"""
        # 从文件重新加载
        from agent.skills.loader import SkillLoader
        # ... 实现热更新逻辑
```

---

## 6. MCP Client 问题

### 6.1 连接池管理缺失

**问题描述**：

每次获取工具都调用 `get_tools()`，没有连接复用：

```python
# agent/mcp/client.py:353-354
tools = await self._client.get_tools(server_name=server_name)
tool = next((t for t in tools if t.name == tool_name), None)
```

**优化建议**：

```python
class MCPClient:
    def __init__(self, server_configs: dict):
        self._connections: dict[str, Any] = {}  # server_name -> connection
        self._tool_cache: dict[str, list] = {}  # server_name -> tools

    async def initialize(self) -> None:
        """初始化所有连接"""
        for name, config in self._server_configs.items():
            if config.enabled:
                self._connections[name] = await self._connect(config)

    async def get_tools(self, server_name: str = None) -> list:
        """获取工具（带缓存）"""
        if server_name in self._tool_cache:
            return self._tool_cache[server_name]

        tools = await self._connections[server_name].get_tools()
        self._tool_cache[server_name] = tools
        return tools

    def invalidate_cache(self, server_name: str = None) -> None:
        """清除缓存"""
        if server_name:
            self._tool_cache.pop(server_name, None)
        else:
            self._tool_cache.clear()
```

---

### 6.2 错误处理不完善

**问题描述**：

服务器连接失败只打印日志，没有重连机制：

```python
# agent/mcp/client.py:283-288
except Exception as e:
    logger.error(
        f"[MCPClient] Failed to get tools from server '{server_name}': {e}"
    )
```

**优化建议**：

```python
from enum import Enum
from dataclasses import dataclass

class ServerStatus(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    FAILED = "failed"

@dataclass
class ServerHealth:
    status: ServerStatus
    last_error: str | None
    reconnect_attempts: int
    last_connected: float | None

class MCPClient:
    MAX_RECONNECT_ATTEMPTS = 3
    RECONNECT_DELAY = 5  # seconds

    def __init__(self, ...):
        self._server_health: dict[str, ServerHealth] = {}

    async def _connect_with_retry(self, name: str, config: MCPServerConfig) -> bool:
        """带重试的连接"""
        self._server_health[name] = ServerHealth(
            status=ServerStatus.CONNECTING,
            last_error=None,
            reconnect_attempts=0,
            last_connected=None,
        )

        for attempt in range(self.MAX_RECONNECT_ATTEMPTS):
            try:
                await self._do_connect(name, config)
                self._server_health[name].status = ServerStatus.CONNECTED
                self._server_health[name].last_connected = time.time()
                return True

            except Exception as e:
                self._server_health[name].reconnect_attempts += 1
                self._server_health[name].last_error = str(e)

                if attempt < self.MAX_RECONNECT_ATTEMPTS - 1:
                    logger.warning(
                        f"[MCPClient] 连接失败，{self.RECONNECT_DELAY}秒后重试: "
                        f"server={name}, attempt={attempt + 1}, error={e}"
                    )
                    await asyncio.sleep(self.RECONNECT_DELAY)

        self._server_health[name].status = ServerStatus.FAILED
        logger.error(f"[MCPClient] 连接失败，已达最大重试次数: server={name}")
        return False

    def get_server_health(self) -> dict[str, dict]:
        """获取服务器健康状态"""
        return {
            name: {
                "status": health.status.value,
                "last_error": health.last_error,
                "reconnect_attempts": health.reconnect_attempts,
                "last_connected": health.last_connected,
            }
            for name, health in self._server_health.items()
        }
```

---

## 7. Workflow 集成问题

### 7.1 子图与主图状态耦合

**问题描述**：

子图依赖主图同名字段自动传递，字段名变更可能导致数据丢失。

**优化建议**：

明确子图输入/输出接口：

```python
from pydantic import BaseModel

class SubgraphInput(BaseModel):
    """子图输入接口"""
    trace_id: str
    user_query: str
    module_name: str | None = None
    module_hint: str | None = None

class SubgraphOutput(BaseModel):
    """子图输出接口"""
    answer: str
    citations: list[dict] = []
    analysis: dict = {}
    node_trace: list[dict] = []

class BaseSubgraph:
    """子图基类"""

    def get_input_keys(self) -> list[str]:
        """声明需要的输入字段"""
        return ["trace_id", "user_query", "module_name", "module_hint"]

    def get_output_keys(self) -> list[str]:
        """声明输出的字段"""
        return ["answer", "citations", "analysis", "node_trace"]

    def extract_input(self, main_state: dict) -> SubgraphInput:
        """从主图状态提取输入"""
        return SubgraphInput(
            trace_id=main_state.get("trace_id", ""),
            user_query=main_state.get("user_query", ""),
            module_name=main_state.get("module_name"),
            module_hint=main_state.get("module_hint"),
        )

    def merge_output(self, output: SubgraphOutput, main_state: dict) -> dict:
        """合并输出到主图状态"""
        return {
            "answer": output.answer,
            "citations": output.citations,
            "analysis": output.analysis,
            "node_trace": main_state.get("node_trace", []) + output.node_trace,
        }
```

---

### 7.2 Checkpointer 初始化复杂

**问题描述**：

PostgreSQL Checkpointer 初始化逻辑复杂，错误处理分散。

**优化建议**：

提取独立的 CheckpointerFactory：

```python
from abc import ABC, abstractmethod
from typing import Any

class CheckpointerFactory(ABC):
    """Checkpointer 工厂基类"""

    @abstractmethod
    def create(self) -> Any:
        """创建 checkpointer"""
        pass

    @abstractmethod
    def close(self) -> None:
        """关闭资源"""
        pass

class MemoryCheckpointerFactory(CheckpointerFactory):
    """内存 Checkpointer 工厂"""

    def create(self) -> MemorySaver:
        return MemorySaver()

    def close(self) -> None:
        pass

class PostgresCheckpointerFactory(CheckpointerFactory):
    """PostgreSQL Checkpointer 工厂"""

    def __init__(self, config: WorkflowCheckpointerConfig):
        self.config = config
        self._checkpointer: PostgresSaver | None = None
        self._context: ExitStack | None = None

    def create(self) -> PostgresSaver:
        try:
            self._ensure_database()
            self._context = ExitStack()
            cm = PostgresSaver.from_conn_string(self.config.pg_dsn)
            self._checkpointer = self._context.enter_context(cm)
            if self.config.pg_setup:
                self._checkpointer.setup()
            return self._checkpointer
        except Exception as e:
            logger.error(f"PostgreSQL checkpointer 初始化失败: {e}")
            # 回退到内存
            return MemoryCheckpointerFactory().create()

    def close(self) -> None:
        if self._context:
            self._context.close()

    def _ensure_database(self) -> None:
        # ... 数据库确保逻辑

def create_checkpointer(config: WorkflowCheckpointerConfig) -> CheckpointerFactory:
    """工厂方法"""
    if config.backend == "memory":
        return MemoryCheckpointerFactory()
    elif config.backend == "postgres":
        return PostgresCheckpointerFactory(config)
    else:
        raise ValueError(f"不支持的 backend: {config.backend}")
```

---

## 8. 可观测性问题

### 8.1 缺少结构化日志

**问题描述**：

当前日志主要是文本格式，不利于日志聚合和分析。

**优化建议**：

使用 structlog 实现结构化日志：

```python
import structlog
from structlog.processors import JSONRenderer, TimeStamper

# 配置 structlog
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
)

logger = structlog.get_logger()

# 使用示例
class AgentLoop:
    def run(self, state: AgentState) -> AgentLoopResult:
        log = logger.bind(
            trace_id=state.get("trace_id"),
            session_id=state.get("session_id"),
        )

        log.info("agent_loop_started", user_query=state.get("user_query", "")[:50])

        for step in range(1, self.config.max_steps + 1):
            log.debug("agent_loop_step", step=step, max_steps=self.config.max_steps)
            # ...

        log.info(
            "agent_loop_completed",
            steps=steps,
            success=result.success,
            latency_ms=result.latency_ms,
            tool_calls=len(result.tool_calls),
        )

        return result
```

---

### 8.2 缺少指标收集

**问题描述**：

没有暴露 Prometheus 指标，无法监控系统健康状况。

**优化建议**：

集成 Prometheus 指标：

```python
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry

# 定义指标
AGENT_LOOP_TOTAL = Counter(
    'agent_loop_total',
    'Total agent loop executions',
    ['status']  # success, failed, timeout
)

AGENT_LOOP_LATENCY = Histogram(
    'agent_loop_latency_seconds',
    'Agent loop latency in seconds',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0]
)

AGENT_LOOP_STEPS = Histogram(
    'agent_loop_steps',
    'Number of steps in agent loop',
    buckets=[1, 2, 3, 5, 7, 10]
)

TOOL_CALL_TOTAL = Counter(
    'tool_call_total',
    'Total tool calls',
    ['tool_name', 'status']  # status: success, failed
)

TOOL_CALL_LATENCY = Histogram(
    'tool_call_latency_seconds',
    'Tool call latency in seconds',
    ['tool_name'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
)

LLM_CALL_TOTAL = Counter(
    'llm_call_total',
    'Total LLM calls',
    ['model', 'status']
)

LLM_TOKENS_USED = Counter(
    'llm_tokens_used_total',
    'Total tokens used',
    ['model', 'type']  # type: prompt, completion
)

# 使用示例
class AgentLoop:
    def run(self, state: AgentState) -> AgentLoopResult:
        start_time = time.time()

        try:
            result = self._run_loop(state)

            # 记录成功指标
            AGENT_LOOP_TOTAL.labels(status='success').inc()
            AGENT_LOOP_LATENCY.observe(time.time() - start_time)
            AGENT_LOOP_STEPS.observe(result.steps)

            return result

        except Exception as e:
            AGENT_LOOP_TOTAL.labels(status='failed').inc()
            raise

class ToolRegistry:
    def _execute_tool(self, tool_call, state) -> ToolCallRecord:
        start_time = time.time()

        try:
            record = self._do_execute(tool_call, state)

            TOOL_CALL_TOTAL.labels(
                tool_name=tool_call.name,
                status='success' if record.success else 'failed'
            ).inc()
            TOOL_CALL_LATENCY.labels(tool_name=tool_call.name).observe(
                time.time() - start_time
            )

            return record

        except Exception as e:
            TOOL_CALL_TOTAL.labels(
                tool_name=tool_call.name,
                status='failed'
            ).inc()
            raise

# FastAPI 集成
from prometheus_client import make_asgi_app

app = FastAPI()
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
```

---

## 9. 优化优先级总结

| 优先级 | 问题 | 影响 | 建议行动 | 预估工作量 |
|--------|------|------|----------|-----------|
| **P0** | 工具执行串行化 | 性能瓶颈 | 实现并行执行 | 2-3天 |
| **P0** | 消息历史无限增长 | 内存溢出/Token超限 | 添加截断策略 | 1-2天 |
| **P1** | LLM 速率限制缺失 | 429 错误 | 添加 Rate Limiter | 1天 |
| **P1** | Execution Skill 安全风险 | 安全漏洞 | 添加沙箱隔离 | 3-5天 |
| **P1** | 缺少重试机制 | 可靠性差 | 集成 tenacity | 1天 |
| **P2** | 全局单例过度 | 测试/多租户困难 | 考虑依赖注入 | 5-7天 |
| **P2** | 同步/异步混合 | 性能开销 | 统一异步化 | 3-5天 |
| **P2** | 缺少指标收集 | 可观测性差 | 集成 Prometheus | 2-3天 |
| **P3** | Token 计数缺失 | 成本无法预估 | 集成 tiktoken | 1天 |
| **P3** | 工具名称冲突 | 意外覆盖 | 添加命名空间 | 0.5天 |
| **P3** | 结构化日志缺失 | 日志分析困难 | 集成 structlog | 1-2天 |

---

## 附录：快速修复清单

### 可立即实施的修复（< 1天）

1. **添加消息截断**：
```python
# agent/core/loop.py
MAX_TOOL_RESULT_LENGTH = 2000
record.result = record.result[:MAX_TOOL_RESULT_LENGTH]
```

2. **降级工具日志**：
```python
# agent/tools/registry.py
logger.debug(...)  # 改为 debug 级别
```

3. **添加简单重试**：
```python
# agent/core/loop.py
from tenacity import retry, stop_after_attempt

@retry(stop=stop_after_attempt(3))
def _invoke_llm(self, ...):
    ...
```

### 需要规划的优化（1-2周）

1. 并行工具执行
2. 依赖注入容器
3. Prometheus 指标
4. Execution Skill 沙箱

### 长期优化项（> 1月）

1. 全面异步化
2. 多租户支持
3. 完整的可观测性平台
