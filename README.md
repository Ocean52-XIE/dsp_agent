# DSP Agent

面向垂直领域知识问答与问题分析的 Deep Agent 服务。当前版本已经统一到单一运行时架构：`Deep Agent + skills + domain_retrieve + MCP tools`，不再使用旧的多 workflow 编排链路。

## 当前架构

主链路如下：

```text
启动: src/api/main.py
  -> lifespan()
  -> init.initialize_async()
  -> DeepAgentService.create_async()

请求: /api/messages
  -> DeepAgentService.run_user_message_async()
  -> agent.factory.create_agent()

工具: domain_retrieve + MCP tools
检索: wiki/code retrieval -> fusion -> citations
持久化: session + observability + checkpointer(memory/postgres)
```

当前真实入口：

- `src/api/main.py`: FastAPI 接口与生命周期管理
- `src/init/initializer.py`: DomainProfile、MCP、Retriever、Checkpointer 初始化
- `src/agent/service.py`: Deep Agent 服务编排
- `src/agent/factory.py`: 模型、skills、tools 装配
- `src/retrievers/tools/domain_retrieve_tool.py`: 统一检索工具
- `src/domain_profile/profile.py`: 领域配置解析与模块推断

## 核心能力

- 统一承载领域问答、问题分析和代码定位类请求
- 基于领域 wiki 和代码语料做混合检索
- 通过 citations 返回证据路径、分数和摘要片段
- 支持 skill 驱动的工具调用与 MCP 扩展
- 提供 session、feedback 和 observability 接口

## 快速开始

### 环境要求

- Python 3.11+
- PostgreSQL 14+（可选；用于 session、observability、postgres checkpointer）

### 安装依赖

```bash
pip install -r requirements.txt
```

### 启动方式

Windows 下推荐直接使用启动脚本：

```powershell
.\start_agent.ps1
```

指定端口和领域目录：

```powershell
.\start_agent.ps1 -Port 8080 -DomainDir "domain/ad_engine"
```

也可以手动启动：

```powershell
$env:AGENT_DOMAIN_DIR = "domain/ad_engine"
$env:PYTHONPATH = "src"
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

服务启动后访问 `http://127.0.0.1:8000`。

### 关键环境变量

常用运行参数如下：

| 变量名 | 说明 | 默认值 |
| --- | --- | --- |
| `AGENT_DOMAIN_DIR` | 领域目录 | `domain/ad_engine` |
| `AGENT_DOMAIN_PROFILE_PATH` | 直接指定 `profile.json` 路径 | 空 |
| `AGENT_LLM_MODEL` | LLM 模型名 | `gpt-4o-mini` |
| `AGENT_LLM_BASE_URL` | LLM API 地址 | 空 |
| `AGENT_LLM_API_KEY` | LLM API Key | 空 |
| `AGENT_LLM_TEMPERATURE` | 温度参数 | `0.1` |
| `AGENT_LLM_MAX_TOKENS` | 最大输出 token | `4096` |
| `AGENT_LLM_TIMEOUT_SECONDS` | LLM 超时秒数 | `60` |
| `AGENT_MCP_ENABLED` | 是否启用 MCP | 由启动脚本设置 |
| `AGENT_CHECKPOINTER_BACKEND` | Checkpointer 后端 | 由启动脚本设置 |
| `AGENT_OBS_PG_DSN` | Observability PostgreSQL DSN | 空 |

说明：

- 不要在 README、脚本或代码中提交明文密钥。
- 如果需要生产部署，建议通过环境变量或安全配置中心注入敏感信息。

## 目录结构

```text
.
├── src/
│   ├── api/                  # FastAPI 接口与协议映射
│   ├── init/                 # 初始化入口
│   ├── agent/                # Deep Agent 运行时封装
│   ├── retrievers/           # Wiki/Code 检索与融合
│   ├── domain_profile/       # profile.json 解析与模块推断
│   ├── session/              # 会话存储
│   ├── observability/        # 观测、反馈、告警
│   ├── log/                  # 日志初始化
│   └── web/                  # 前端静态资源
├── domain/
│   ├── README.md             # 领域目录说明
│   └── ad_engine/            # 当前示例领域
│       ├── profile.json
│       ├── wiki/
│       ├── codes/
│       ├── prompts/
│       ├── skills/
│       └── mcp_servers/
├── docs/                     # 设计与检索说明
├── tests/                    # pytest 测试
└── start_agent.ps1           # Windows 启动脚本
```

## 领域配置

当前系统通过 `domain/<domain_id>/profile.json` 配置领域行为。现阶段真正参与主链路的字段主要有：

- `profile_id`
- `sources.wiki.root`
- `sources.code.roots`
- `routing.default_module`
- `modules`
- `retrieval`
- `prompts.deep_agent_system_path`
- `deep_agents.skills_root`

其中：

- `modules` 仍然是当前模块推断和 wiki 文档提示的核心配置
- `routing` 目前只保留 `default_module` 作为兜底
- `prompts` 现在只使用 `deep_agent_system_path`

领域目录的详细约束见 [domain/README.md](/d:/codes/dsp_agent/domain/README.md)。

## API 概览

当前主接口如下：

| 端点 | 方法 | 说明 |
| --- | --- | --- |
| `/` | GET | 返回前端首页 |
| `/api/health` | GET | 健康检查 |
| `/api/sessions` | GET | 列出会话 |
| `/api/sessions` | POST | 创建会话 |
| `/api/sessions/{session_id}` | GET | 获取会话详情 |
| `/api/messages` | POST | 发送用户消息并触发 Deep Agent |
| `/api/references/{trace_id}` | GET | 获取本次回答引用证据 |
| `/api/messages/{message_id}/feedback` | POST | 提交消息反馈 |
| `/api/config` | GET | 获取前端所需配置信息 |
| `/api/observability/summary` | GET | 获取观测摘要 |
| `/api/observability/alerts` | GET | 获取告警列表 |

示例请求：

```bash
curl -X POST http://127.0.0.1:8000/api/messages \
  -H "Content-Type: application/json" \
  -d '{"session_id":"sess_demo","content":"出价胜率下降怎么排查？"}'
```

## 工作机制

一次典型请求会经历以下阶段：

1. API 接收消息并创建或读取 session
2. `DeepAgentService` 组装本轮输入并调用 Deep Agent
3. Agent 根据 `deep_agent_system` 和 skills 选择是否调用 `domain_retrieve` 或 MCP 工具
4. `domain_retrieve` 基于领域 profile 做模块推断，执行 wiki/code 检索和融合
5. 服务层解析 agent 输出，返回 assistant message、trace 信息和 citations
6. session、feedback、observability 数据按配置写入存储


## 测试

```bash
pytest tests/
```

如果只验证本次架构相关改动，通常至少应覆盖：

```bash
python -m pytest tests/domain_profile/test_profile.py
python -m pytest tests/agent/test_factory.py tests/agent/test_config.py
```

## 相关文档

- [领域目录说明](/d:/codes/dsp_agent/domain/README.md)
- [Deep Agents 说明](/d:/codes/dsp_agent/docs/deep_agents.md)
- [Wiki Retrieval Doc](/d:/codes/dsp_agent/docs/wiki_retrieve.md)
- [Code Retrieval Doc](/d:/codes/dsp_agent/docs/code_retrieve.md)
- [Fusion Doc](/d:/codes/dsp_agent/docs/retrieve_merge.md)
- [仓库约束](/d:/codes/dsp_agent/AGENTS.md)

## 开发约束

提交改动前，建议至少检查以下几点：

- 是否仍然沿用当前 Deep Agent 架构
- 初始化逻辑是否集中在 `src/init/`
- 新工具是否在 `src/agent/factory.py` 显式装配
- 新配置是否优先放在 `profile.json` 或环境变量
- API 响应结构是否保持兼容
- 是否补了对应 pytest 用例

## License

MIT
