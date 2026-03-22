# CLAUDE.md

本文件用于约束本仓库后续的代码生成行为，目标是让改动可落位、可维护、可评估。

## 1. 目录职责

### 顶层目录

| 目录 | 职责 |
| --- | --- |
| `src/` | 生产代码（API、Agent、工作流、基础设施） |
| `domain/` | 领域数据（profile、wiki、codes、skills、prompts） |
| `tests/` | 自动化测试（pytest） |
| `docs/` | 设计文档 |
| `tools/` | 辅助脚本 |
| `logs/` | 运行日志 |

### `src/` 核心模块

| 目录 | 职责 |
| --- | --- |
| `api/` | FastAPI 接口层，只做协议转换和调用 WorkflowService |
| `init/` | 统一初始化入口，新增组件需补充对应初始化器 |
| `agent/` | **通用 Agent 能力**：AgentLoop、LLMClient、MCPClient、ToolRegistry、SkillRegistry |
| `workflow/` | **业务流程编排**：engine.py（主图）、nodes/（节点）、subgraph/（子图）、common/（工具） |
| `domain_profile/` | 领域配置管理，支持模块推断、查询归一化 |
| `retrievers/` | 检索器（向量检索、重排、融合） |
| `session/` | 会话存储（PostgreSQL/内存） |
| `observability/` | 可观测性（交互记录、反馈、告警） |
| `eval/` | 离线评测脚本 |

### `domain/` 领域目录

| 目录 | 职责 |
| --- | --- |
| `profile.json` | 领域配置中心（路由、检索、阈值、词表） |
| `wiki/` | 知识文档语料 |
| `codes/` | 代码检索语料 |
| `skills/` | 技能配置 |
| `mcp_servers/` | MCP Server 配置 |
| `prompts/` | 提示词模板 |

## 2. 核心架构

```
主工作流: load_context -> intent_routing -> [knowledge_qa | issue_analysis | code_generation | out_of_scope] -> finalize_response

初始化流程: DomainProfile -> LLM -> Skills -> MCP -> Retrievers -> ToolRegistry

全局单例: get_llm_client() | get_domain_profile() | get_skill_registry() | get_tool_registry() | get_mcp_client()
```

## 3. 代码生成约束

### 架构约束
1. **分层**：API 层不写业务推理，节点层不处理 HTTP
2. **Agent 隔离**：`src/agent/` 只提供通用能力，业务逻辑放 `src/workflow/`
3. **节点契约**：`run(service, state) -> dict`，返回状态增量
4. **图结构**：新增/删除节点需同步更新 `engine.py`

### 配置约束
5. **配置优先**：阈值、TopK 放 `profile.json`，密钥放环境变量
6. **领域隔离**：业务词表、提示词放 `domain/<domain_id>/`

### 质量约束
7. **可追踪**：检索命中需保留 source/path/score
8. **向后兼容**：保持 `/api/messages` 等接口响应结构
9. **测试覆盖**：代码覆盖率 ≥ 85%
10. **框架**：使用 LangChain/LangGraph 1.x

### 编码规范
11. 类型标注 + 中文注释 + 关键模块日志 + utf-8 编码

## 4. 变更落位

| 变更类型 | 落位 |
| --- | --- |
| API | `src/api/main.py` |
| 路由/检索策略 | `src/workflow/nodes/` + `src/retrievers/` |
| Agent 能力 | `src/agent/core/` 或 `src/agent/tools/` |
| 技能 | `domain/<id>/skills/` |
| MCP 工具 | `domain/<id>/mcp_servers/` |
| 新领域 | `domain/<new_id>/` 全套目录 |

## 5. 提交前自检

- [ ] 目录落位正确，未破坏分层边界
- [ ] 同步更新 `engine.py`、`state.py`
- [ ] 补充测试用例（`tests/`）
- [ ] 配置项有默认值（`profile.json` 或环境变量）
- [ ] 无明文敏感信息
- [ ] 新增全局组件已加入 `src/init/`
