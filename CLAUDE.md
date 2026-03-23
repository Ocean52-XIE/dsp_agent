# CLAUDE.md

本文件用于约束本仓库后续的代码生成与改动落位。

## 1. 当前架构

### 主链路

```text
启动: src/api/main.py -> lifespan() -> init.initialize_async() -> DeepAgentService.create_async()
请求: /api/messages -> DeepAgentService.run_user_message_async() -> agent.factory.create_agent()
工具: domain_retrieve + MCP tools
检索: wiki/code retrieval -> orchestration fusion -> citations
持久化: session + observability + checkpointer(memory/postgres)
```

### 当前真实入口

- 服务入口：`src/agent/service.py`
- Agent 装配：`src/agent/factory.py`
- 初始化入口：`src/init/initializer.py`
- 领域配置：`src/domain_profile/profile.py`
- 统一检索工具：`src/retrievers/tools/domain_retrieve_tool.py`
- MCP：`src/agent/mcp/` + `domain/<id>/mcp_servers/`

### 全局单例

- `domain_profile.get_domain_profile()`
- `agent.mcp.get_mcp_client()`
- `retrievers.wiki.retriever.get_wiki_retriever()`
- `retrievers.code.retriever.get_code_retriever()`

## 2. 目录职责

| 路径 | 职责 |
| --- | --- |
| `src/api/` | FastAPI 接口与协议映射 |
| `src/init/` | 领域配置、MCP、Retriever、Checkpointer 初始化 |
| `src/agent/` | Deep Agent 运行时封装 |
| `src/retrievers/` | Wiki/Code 检索、重试、融合、工具暴露 |
| `src/domain_profile/` | `profile.json` 解析、模块推断、路径解析 |
| `src/session/` | 会话存储 |
| `src/observability/` | 观测、反馈、告警 |
| `src/log/` | 日志初始化 |
| `src/web/` | 前端静态资源 |
| `domain/<id>/profile.json` | 领域路由、检索、提示词、deep_agents 配置 |
| `domain/<id>/skills/` | Deep Agent skills |
| `domain/<id>/wiki/` | Wiki 语料 |
| `domain/<id>/codes/` | Code 语料 |
| `domain/<id>/prompts/` | Prompt 模板 |
| `domain/<id>/mcp_servers/` | MCP server 配置 |
| `tests/` | pytest 测试 |

## 3. 改动原则

1. `src/api/` 不写业务检索策略，也不直接拼 Agent 工具。
2. `src/agent/` 负责运行时封装，不承载领域语料。
3. 领域规则优先放到 `domain/<id>/profile.json`，不要把阈值和路由硬编码进 service/api。
4. 新增 Agent tool 时，优先落在 `src/retrievers/tools/` 或 `src/agent/tools/`，并在 `src/agent/factory.py` 中显式装配。
5. 检索相关改动统一落在 `src/retrievers/orchestration/`、`src/retrievers/wiki/`、`src/retrievers/code/`。
6. citation 必须保留最少字段：`source`/`source_type`、`path`、`score`、`excerpt`；代码定位类结果尽量保留 `symbol_name`、`start_line`、`end_line`。
7. `/api/messages`、`/api/sessions`、`/api/references/{trace_id}`、`/api/messages/{message_id}/feedback` 的响应结构保持兼容。
8. 初始化改动统一进 `src/init/`，不要把全局初始化散落到 API 或工厂内部。
9. 保持类型标注、UTF-8、中文注释和关键日志。
10. 不要提交明文密钥；敏感配置统一走环境变量。

## 4. 变更落位

| 变更类型 | 落位 |
| --- | --- |
| API/响应结构 | `src/api/main.py`、`src/api/message_mapper.py` |
| Deep Agent 服务编排 | `src/agent/service.py` |
| Agent 创建/模型/工具/技能装配 | `src/agent/factory.py`、`src/agent/config.py` |
| MCP | `src/agent/mcp/` + `domain/<id>/mcp_servers/` |
| 初始化/Checkpointer | `src/init/` |
| DomainProfile | `src/domain_profile/profile.py` |
| Wiki/Code 检索 | `src/retrievers/wiki/`、`src/retrievers/code/` |
| 检索融合/重试 | `src/retrievers/orchestration/` |
| 统一检索工具 | `src/retrievers/tools/domain_retrieve_tool.py` |
| 会话/观测 | `src/session/`、`src/observability/` |
| 领域技能/提示词/语料 | `domain/<id>/` |

## 5. 提交前检查

- [ ] 是否仍然沿用当前 Deep Agent 架构，而不是重新引入旧 workflow 设计
- [ ] 新增初始化逻辑是否已接入 `src/init/`
- [ ] 新增 Agent tool 是否已在 `src/agent/factory.py` 装配
- [ ] citation 或 assistant message 改动后，是否同步检查 API 映射与兼容性
- [ ] 新配置是否优先放进 `profile.json` 或环境变量
- [ ] 是否补了对应 pytest 用例
- [ ] 是否避免提交明文敏感信息
