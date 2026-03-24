# 基于 Deep Agents 的单驱动重构方案

## 1. 设计目标

本方案直接放弃当前 `LangGraph 主图 + 子图 + AgentLoop 节点` 的组织方式，改为：

- **Deep Agents 作为唯一驱动层**
- **检索能力收敛为 1 个 tool**
- **路由能力收敛为 system prompt 中的内置规则**
- **知识问答收敛为 1 个 skill**
- **问题分析收敛为 1 个 skill**

目标系统不再保留主图，不再考虑与当前 workflow 的兼容，也不再让“路由、检索、生成”分别由固定节点编排，而是全部交由一个 Deep Agent 按 skill + tool 自主驱动。

## 2. 最终形态

### 2.1 核心原则

1. **只有一个主代理**
   API 请求进入后，直接交给一个通过 `create_deep_agent(...)` 创建的主代理执行。

2. **Skill 负责方法论**
   知识问答、问题分析都不是 Python 节点，而是 Skill；路由规则直接内置到主代理提示词中。

3. **Tool 负责能力调用**
   检索不再拆成 `query_rewriter / retrieve_wiki / retrieve_code / merge_evidence` 多个节点，而是合并为一个 `domain_retrieve` tool。

4. **代理自己决定执行路径**
   主代理收到问题后，先依据内置路由规则判断是：
   - 知识问答
   - 问题分析
   - 超范围

   然后再命中 `knowledge-qa` 或 `issue-analysis` skill，并按需调用 `domain_retrieve` tool。

5. **固定流程下沉到 skill/tool 内部**
   “先看模块、再改写查询、再混合检索、再归并证据”的流程，不再出现在主架构图里，而是成为 tool 内部实现或 skill 中的执行约束。

---

## 3. 目标架构

### 3.1 总体结构

```text
FastAPI API
  -> DeepAgentService
      -> create_deep_agent(...)
          -> skills
             - knowledge-qa
             - issue-analysis
          -> tools
             - domain_retrieve
             - MCP tools（可选）
          -> backend / workspace / memory
      -> agent.invoke(...)
      -> assistant_message
```

### 3.2 运行流程

```text
用户问题
  -> 主代理收到请求
  -> 根据 system prompt 中的内置路由规则判断任务属于 knowledge_qa / issue_analysis / out_of_scope
  -> 若 in-scope：
       -> 读取对应 skill
       -> 调用 domain_retrieve tool 获取证据
       -> 基于 skill 规则完成回答
  -> 输出统一响应
```

### 3.3 架构结论

最终系统中：

- 没有 `workflow/engine.py` 主图
- 没有 `knowledge_qa` 子图
- 没有 `issue_analysis` 子图
- 没有 `BaseAgentLoopNode`
- 没有“路由节点”
- 没有“检索节点”

只有：

- 一个 Deep Agent
- 两个核心 skills
- 一个核心 retrieval tool

---

## 4. 角色划分

### 4.1 主代理

主代理负责：

- 接收用户输入
- 匹配技能
- 调用工具
- 组织最终回答

主代理不再依赖外部主图做编排。

### 4.2 内置路由规则

职责：

- 识别请求属于知识问答还是问题分析
- 判断是否超出领域范围
- 给出推荐检索策略
- 决定后续应该启用哪个 skill

它不再是独立 skill，而是直接编译进主代理的 system prompt。

### 4.3 `knowledge-qa` skill

职责：

- 处理概念解释、架构说明、模块说明、实现说明、代码定位说明
- 告诉主代理何时调用 `domain_retrieve`
- 规定输出结构与引用方式

### 4.4 `issue-analysis` skill

职责：

- 处理异常、掉量、抖动、成本超标、胜率下降、冷启动无量等问题分析请求
- 指导主代理基于证据做假设、排查路径、结论和建议
- 规定问题分析的输出结构

### 4.5 `domain_retrieve` tool

职责：

- 接收一个统一检索请求
- 内部完成查询归一化、模块增强、wiki/code 混合检索、重排、证据归并
- 返回标准化证据列表给主代理

它是整个系统唯一的检索入口。

---

## 5. Deep Agent 的构建方式

### 5.1 创建方式

建议直接使用：

```python
agent = create_deep_agent(
    model=model,
    tools=[domain_retrieve, *mcp_tools],
    skills=[skill_root],
    system_prompt=system_prompt,
    backend=backend,
    checkpointer=checkpointer,
)
```

### 5.2 为什么不再保留主图

主图在当前系统里的作用主要是：

- 固定顺序执行
- 人工分叉路由
- 在末端调用 AgentLoop

但在新的目标里，这些职责都由 Deep Agent 原生承担：

- 技能匹配替代主图路由
- 工具调用替代检索节点链
- 主代理推理替代固定子图编排

因此继续保留主图只会让系统出现“双重编排层”，增加复杂度，没有收益。

---

## 6. Skill 设计

本方案的核心是把“业务流程知识”从代码节点迁移到 skill。

### 6.1 Skill 目录结构

建议将领域技能目录收敛为：

```text
domain/ad_engine/skills/
├── knowledge-qa/
│   ├── SKILL.md
│   └── references/
│       ├── answer_style.md
│       └── citation_rules.md
└── issue-analysis/
    ├── SKILL.md
    └── references/
        ├── analysis_framework.md
        └── troubleshooting_patterns.md
```

这两个 skill 是核心 skill。

其他现有技能可以后续决定保留为补充技能，但它们不再是核心驱动结构的一部分。

---

### 6.2 内置路由规则设计

#### 定位

路由规则不再以独立 skill 形式存在，而是直接编译进 `deep_agent_system.md`，作为主代理的内置判断逻辑。

#### 职责

1. 判断是否属于广告引擎领域
2. 判断是 `knowledge_qa` 还是 `issue_analysis`
3. 判断问题更偏 wiki 还是更偏 code
4. 判断是否需要主模块 / 相关模块增强
5. 决定下一步应该启用哪个业务 skill

#### 输入关注点

- 领域关键词
- 问题语气
- 是否带“为什么/是什么/怎么实现”
- 是否带“异常/排查/失败/下降/抖动/无量”
- 是否带符号名/文件名/函数名

#### 内部判断约定

建议主代理内部形成如下判断：

```json
{
  "domain_scope": "in_scope | out_of_scope",
  "intent": "knowledge_qa | issue_analysis",
  "retrieval_bias": "wiki_first | code_first | hybrid",
  "module_name": "xxx",
  "related_modules": ["a", "b"]
}
```

该结构不一定返回给用户，但应作为主代理内部执行约束。

---

### 6.3 `knowledge-qa` skill 设计

#### 定位

专门处理：

- 概念问答
- 模块架构说明
- 链路介绍
- 代码实现定位
- 参数/函数/文件位置说明

#### description 建议

> 用于广告引擎领域的知识问答、架构说明、模块说明、代码实现说明和源码定位类问题。需要时调用 `domain_retrieve` 工具检索 wiki 和 code 证据，并基于证据回答。

#### 核心规则

1. 先依据主代理的内置路由判断执行
2. 证据不足时必须调用 `domain_retrieve`
3. 回答必须基于证据，不允许编造
4. 涉及代码定位时优先强调：
   - 文件路径
   - 函数名 / 类名
   - 相关模块
5. 输出必须包含引用来源

#### 推荐输出结构

```text
结论

依据

涉及模块 / 文件

补充说明
```

#### 典型检索策略

- 架构说明：`wiki_first`
- 代码定位：`code_first`
- 模块解释：`hybrid`

---

### 6.4 `issue-analysis` skill 设计

#### 定位

专门处理：

- 异常定位
- 掉量分析
- 成本超标
- 胜率下降
- 精排抖动
- 冷启动无量
- 联调排障

#### description 建议

> 用于广告引擎领域的异常排查、问题诊断和原因分析类问题。需要调用 `domain_retrieve` 工具获取 wiki 与 code 证据，并按“现象 -> 可能原因 -> 排查步骤 -> 修复建议”组织输出。

#### 核心规则

1. 先明确现象，不直接跳结论
2. 证据不足时必须调用 `domain_retrieve`
3. 分析时要区分：
   - 已知证据支持的结论
   - 合理假设
   - 建议继续检查的方向
4. 不输出伪确定性结论
5. 输出必须给出排查步骤

#### 推荐输出结构

```text
问题判断

可能原因

建议排查步骤

修复建议

证据来源
```

#### 典型检索策略

- 问题分析默认 `hybrid`
- 若包含明确函数/文件/模块名，可偏 `code_first`
- 若更像经验排障，可偏 `wiki_first`

---

## 7. Tool 设计

### 7.1 唯一核心 Tool：`domain_retrieve`

本方案明确规定：当前系统的检索能力全部收敛为一个 tool。

它内部整合当前已有能力：

- query rewrite
- module infer
- wiki retrieval
- code retrieval
- hybrid fusion
- reranker
- evidence normalization

### 7.2 Tool 接口建议

```python
class DomainRetrieveInput(BaseModel):
    query: str
    intent: str = "knowledge_qa"
    retrieval_bias: str = "hybrid"
    module_name: str | None = None
    related_modules: list[str] | None = None
    top_k: int = 6
```

```python
class DomainRetrieveOutput(TypedDict):
    normalized_query: str
    retrieval_bias: str
    module_name: str
    evidence: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    debug: dict[str, Any]
```

### 7.3 Tool 的输入语义

| 字段 | 作用 |
| --- | --- |
| `query` | 用户原问题或 skill 整理后的查询 |
| `intent` | 当前任务类型：`knowledge_qa` / `issue_analysis` |
| `retrieval_bias` | `wiki_first` / `code_first` / `hybrid` |
| `module_name` | 已推断主模块 |
| `related_modules` | 补充模块 |
| `top_k` | 返回证据数 |

### 7.4 Tool 的输出语义

返回结果应统一包含：

- `source`
- `path`
- `title`
- `snippet`
- `score`
- `source_type`
- `section`

这样主代理和 skill 都不需要理解底层 retriever 差异。

### 7.5 Tool 内部实现

`domain_retrieve` 建议直接复用当前已有检索代码，但收口到一个入口：

```text
domain_retrieve()
  -> load domain profile
  -> normalize query
  -> infer/query enhance
  -> retrieve wiki
  -> retrieve code
  -> rerank/fuse
  -> normalize evidence
  -> return evidence
```

### 7.6 为什么必须是一个 tool

用户已经明确要求把当前检索能力封装为一个 tool，这个设计也有实际好处：

1. 主代理只需学习一个检索入口
2. Skill 不需要记忆多个检索工具差异
3. 路由 skill 只需要产出 bias，而不用关心底层执行细节
4. tool schema 更稳定

---

## 8. System Prompt 设计

### 8.1 主代理 System Prompt

主代理的系统提示词应只负责“全局角色和硬约束”，不要再塞入完整业务流程。

建议包含：

1. 你是广告引擎领域智能代理
2. 对领域请求优先使用 skills
3. 需要证据时调用 `domain_retrieve`
4. 必须给出引用
5. 不要编造代码路径或结论
6. 超范围时直接说明无法处理

### 8.2 业务方法论不放在 System Prompt

以下内容不放在主代理 system prompt：

- 路由规则细节
- 问答格式细节
- 问题分析方法论
- 排障模板

这些都放到 skill 中，避免主 prompt 膨胀。

---

## 9. 配置设计

### 9.1 `profile.json` 的职责

虽然不再保留 workflow 主图，但 `domain/ad_engine/profile.json` 仍然保留，并承担：

- 模块配置
- 领域词表
- 路由判断词表
- 检索配置
- prompt 路径
- Deep Agents 运行配置

### 9.2 新增 `deep_agents` 配置段

建议示例：

```json
{
  "deep_agents": {
    "enabled": true,
    "system_prompt_path": "prompts/deep_agent_system.md",
    "skills_root": "skills",
    "primary_skills": [
      "knowledge-qa",
      "issue-analysis"
    ],
    "tools": {
      "domain_retrieve": {
        "default_top_k": 6,
        "max_top_k": 10,
        "default_bias": "hybrid"
      }
    },
    "backend": {
      "workspace_root": "/workspace/",
      "artifact_root": "/artifacts/",
      "memory_root": "/memories/"
    }
  }
}
```

### 9.3 Prompt 文件建议

```text
domain/ad_engine/prompts/
├── deep_agent_system.md
├── qa_system.md
└── issue_system.md
```

其中：

- `deep_agent_system.md` 用于主代理全局约束
- `qa_system.md` 与 `issue_system.md` 可以作为 skill 的引用资料，而不是主 prompt

---

## 10. 代码目录设计

这一版目录设计完全围绕“单 Deep Agent 驱动”展开，不再保留 `workflow/` 作为业务编排层。

### 10.1 推荐项目目录

注意：这里采用的是**轻驱动设计**。  
因为 Deep Agents 已经原生支持：

- skills 目录加载
- skill 渐进披露
- tools 注入
- system prompt 注入
- backend / memory / checkpointer 配置

所以 `src/agent/` 不应该再人为拆出大量“框架包装层”。

```text
dsp_agent/
├── src/
│   ├── api/                           # FastAPI 接口层
│   │   ├── main.py
│   │   ├── schemas.py
│   │   └── deps.py
│   │
│   ├── agent/                         # Deep Agent 驱动层
│   │   ├── service.py                 # DeepAgentService，系统唯一业务入口
│   │   ├── factory.py                 # create_deep_agent(...) 的唯一装配入口
│   │   ├── config.py                  # agent 级配置解析
│   │   ├── message_mapper.py          # Deep Agent 输出 -> assistant_message
│   │   ├── tools/
│   │   │   ├── domain_retrieve.py     # 核心检索 tool
│   │   │   ├── models.py              # tool 输入输出 schema
│   │   │   └── mcp_tools.py           # MCP tools 加载与适配
│   │   └── mcp/                       # MCP 客户端与适配
│   │       ├── client.py
│   │       ├── config_loader.py
│   │       └── tool_adapter.py
│   │
│   ├── domain_profile/                # 领域配置加载与归一化
│   │   ├── __init__.py
│   │   └── profile.py
│   │
│   ├── retrievers/                    # 底层检索实现，仅供 domain_retrieve 复用
│   │   ├── embedding_retriever.py
│   │   ├── cross_encoder_reranker.py
│   │   ├── weighted_fusion.py
│   │   └── model_cache.py
│   │
│   ├── session/                       # 会话存储
│   │   ├── __init__.py
│   │   ├── memory_session_store.py
│   │   └── postgres_session_store.py
│   │
│   ├── observability/                 # 可观测性
│   │   ├── __init__.py
│   │   ├── tracer.py
│   │   ├── event_store.py
│   │   └── postgres_store.py
│   │
│   ├── init/                          # 启动初始化
│   │   ├── __init__.py
│   │   ├── initializer.py
│   │   ├── llm_initializer.py
│   │   ├── mcp_initializer.py
│   │   ├── retriever_initializer.py
│   │   └── service_initializer.py
│   │
│   └── log/
│       ├── __init__.py
│       └── runtime_logging.py
│
├── domain/
│   └── ad_engine/
│       ├── profile.json
│       ├── prompts/
│       │   ├── deep_agent_system.md
│       │   ├── qa_system.md
│       │   └── issue_system.md
│       ├── skills/
│       │   ├── knowledge-qa/
│       │   │   ├── SKILL.md
│       │   │   └── references/
│       │   └── issue-analysis/
│       │       ├── SKILL.md
│       │       └── references/
│       ├── wiki/
│       ├── codes/
│       ├── mcp_servers/
│       └── eval/
│
├── tests/
│   ├── api/
│   ├── agent/
│   │   ├── runtime/
│   │   ├── service/
│   │   ├── tools/
│   │   └── skills/
│   ├── domain_profile/
│   ├── retrievers/
│   └── integration/
│
├── docs/
├── tools/
└── logs/
```

### 10.2 顶层目录职责

| 目录 | 职责 |
| --- | --- |
| `src/api/` | 对外 HTTP 接口、请求响应模型、依赖注入 |
| `src/agent/` | Deep Agent 驱动层，负责创建代理、加载 skill、注册 tool、执行请求 |
| `src/domain_profile/` | 领域配置、模块推断、检索参数、prompt 路径解析 |
| `src/retrievers/` | 纯检索基础设施，不感知技能和 API |
| `src/session/` | 会话持久化 |
| `src/observability/` | tool/skill/request 级追踪与事件落库 |
| `src/init/` | 统一初始化入口 |
| `domain/` | 领域知识、skills、prompts、代码语料、MCP 配置 |
| `tests/` | 单元测试、集成测试、端到端测试 |

### 10.3 `src/agent/` 是否会过度设计

会。  
如果严格按 Deep Agents 原生能力来设计，`src/agent/` 应该尽量薄。

官方文档说明：

- `create_deep_agent` 的核心配置本身就只有 model、tools、system prompt、subagents、backends、skills、memory 等少数几项。  
  来源：<https://docs.langchain.com/oss/python/deepagents/customization>
- skills 可以直接通过目录路径传入，框架会在启动时读取 `SKILL.md` frontmatter，并在命中时按需读取完整 skill。  
  来源：<https://docs.langchain.com/oss/python/deepagents/skills>
- Deep Agents 已内置任务规划、文件系统工具与长上下文管理，不需要我们再人为搭一层“agent runtime orchestration”。  
  来源：<https://docs.langchain.com/oss/python/deepagents/overview>

所以不建议在 `src/agent/` 中再拆出：

- `skills/loader.py`
- `skills/resolver.py`
- `skills/validator.py` 作为主链路依赖
- `runtime/builder.py`
- `runtime/backend.py`
- `runtime/memory.py`
- `runtime/checkpointer.py`
- `runtime/settings.py`

这些文件只有在系统复杂度明显继续上升时才值得拆分。当前阶段会让目录比框架本身还重。

### 10.4 精简后的 `src/agent/` 职责

精简后，`src/agent/` 只保留三类职责：

1. **创建代理**
   - 从配置中读取 model、skills 根目录、system prompt、tools
   - 调用 `create_deep_agent(...)`

2. **提供工具**
   - `domain_retrieve`
   - MCP tools

3. **输出映射**
   - 把 deep agent 返回值映射成 API 层使用的 assistant message

### 10.5 精简后的关键文件建议

#### `src/agent/`

| 文件 | 职责 |
| --- | --- |
| `service.py` | `DeepAgentService`，接收消息并调用代理 |
| `factory.py` | 读取配置并创建 Deep Agent |
| `config.py` | 解析 agent 所需最少配置 |
| `message_mapper.py` | 将 agent 输出转换为 assistant message |

#### `src/agent/tools/`

| 文件 | 职责 |
| --- | --- |
| `domain_retrieve.py` | 唯一核心检索 tool |
| `models.py` | `domain_retrieve` 输入输出 schema |
| `mcp_tools.py` | MCP tools 适配与返回 |

### 10.6 为什么不单独建 `src/agent/skills/`

因为 Skill 的主体本来就应位于：

- `domain/<domain_id>/skills/<skill_name>/SKILL.md`

Deep Agents 会直接消费这个目录。  
如果再建一个 `src/agent/skills/`，很容易把“技能内容”与“技能加载代码”做成双中心，反而增加维护成本。

因此建议：

- **技能内容全部放在 `domain/`**
- `src/agent/` 只知道一个 `skills_root` 路径

### 10.7 为什么不单独建 `runtime/`

如果当前没有：

- 多 backend 路由
- 多 memory provider
- 多 agent profile
- 多 deployment mode

那么单独建 `runtime/` 往往是过早抽象。

当前最合理的做法是：

- 在 `factory.py` 中直接完成 `create_deep_agent(...)` 装配
- 未来真的复杂了，再把 `factory.py` 拆成 `runtime/`

### 10.5 `domain/` 目录设计

由于业务方法论主要通过 skill 表达，`domain/ad_engine/` 的重要性会高于现在。

建议职责如下：

| 路径 | 职责 |
| --- | --- |
| `profile.json` | 全局领域配置 |
| `prompts/deep_agent_system.md` | 主代理系统提示词 |
| `skills/knowledge-qa/` | 知识问答技能 |
| `skills/issue-analysis/` | 问题分析技能 |
| `wiki/` | 文档知识语料 |
| `codes/` | 代码检索语料 |
| `mcp_servers/` | 外部工具配置 |

### 10.8 不再保留的旧目录

新架构下建议直接移除：

- `src/workflow/`
- `src/agent/core/`
- `src/agent/skills/`（如果只是为了包装 Deep Agents skills）
- `src/agent/state.py` 中仅服务旧 AgentLoop 的部分

原因：

- `workflow/` 的职责已经被 Deep Agent 接管
- `agent/core/loop.py` 是旧 function-calling agent 循环
- 旧状态模型围绕节点编排设计，不适合单代理架构
- Deep Agents 的 skill 加载已经原生支持，不必重复包一层

---

## 11. 服务层设计

### 11.1 `DeepAgentService`

建议新增：

- `src/agent/app/service.py`

其职责：

1. 加载 domain profile
2. 创建 deep agent
3. 接收 session/history/user_query
4. 调用 `agent.invoke(...)`
5. 将结果映射为当前 API 所需的 assistant message

### 11.2 建议接口

```python
class DeepAgentService:
    def run_user_message(
        self,
        *,
        session_id: str,
        trace_id: str,
        user_query: str,
        history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        ...
```

这个接口可以直接替代当前 `WorkflowService.run_user_message(...)`。

---

## 12. API 层设计

### 12.1 API 保持简单

[src/api/main.py](/d:/codes/dsp_agent/src/api/main.py) 中不再初始化 workflow engine，而是初始化 `DeepAgentService`。

### 12.2 API 调用链

```text
POST /api/messages
  -> load session/history
  -> DeepAgentService.run_user_message(...)
  -> save assistant message
  -> record observability
```

API 层只做：

- 协议转换
- session 管理
- observability

不承担任何业务编排。

---

## 13. Observability 设计

Deep Agents 驱动后，观测重点从“节点执行”变成“skill / tool 执行”。

### 13.1 需要记录的事件

1. 请求级事件
   - trace_id
   - session_id
   - user_query

2. skill 命中事件
   - 匹配到的 skill 名称
   - skill 加载时机
   - skill 描述命中原因

3. tool 调用事件
   - tool 名称
   - 输入参数
   - latency
   - 输出摘要

4. 回答质量事件
   - citation 数量
   - 是否触发超范围
   - 是否检索成功

### 13.2 节点追踪改造

旧系统的 `node_trace` 应改为：

- `skill_trace`
- `tool_trace`

不再记录 `load_context -> retrieve_wiki -> merge_evidence` 这种节点轨迹。

---

## 14. Skill 内容设计建议

### 14.1 `knowledge-qa/SKILL.md`

应包含：

- 使用场景
- 何时调用 `domain_retrieve`
- 如何理解检索返回的证据
- 回答格式
- 引用格式
- 证据不足时的回应方式

### 14.2 `issue-analysis/SKILL.md`

应包含：

- 使用场景
- 如何将问题拆成“现象 / 原因 / 排查 / 修复”
- 何时调用 `domain_retrieve`
- 如何处理证据不足
- 如何区分“证据支持”和“假设”

---

## 15. `domain_retrieve` tool 的实现细节

### 15.1 推荐实现文件

- `src/agent/tools/domain_retrieve.py`

### 15.2 内部复用关系

可以直接复用当前已有模块：

- `src/domain_profile/profile.py`
- `src/retrievers/*`
- `src/workflow/common/evidence.py` 中可复用的证据归一化逻辑
- `src/workflow/nodes/retrieval_flow/*` 中可下沉的检索实现

但这些复用只存在于 tool 内部，不再暴露为 workflow 节点。

### 15.3 返回格式建议

```json
{
  "normalized_query": "召回候选量骤降 原因",
  "retrieval_bias": "hybrid",
  "module_name": "ad-recall",
  "evidence": [
    {
      "source": "wiki",
      "path": "domain/ad_engine/wiki/01-在线召回.md",
      "title": "在线召回",
      "section": "4.1 候选量骤降",
      "snippet": "...",
      "score": 0.88
    }
  ],
  "citations": [
    {
      "source": "wiki",
      "path": "domain/ad_engine/wiki/01-在线召回.md"
    }
  ]
}
```

---

## 16. 执行边界

### 16.1 本阶段只定义一个核心 tool

本方案明确只把“检索能力”封装为一个统一 tool。  
其他功能暂不拆成多个业务 tool，避免系统一开始又重新长回工具森林。

### 16.2 skill 与 tool 的边界

| 能力 | 归属 |
| --- | --- |
| 路由规则 | skill |
| 问答范式 | skill |
| 问题分析范式 | skill |
| wiki/code 混合检索 | tool |
| MCP 外部调用 | tool |
| API 响应组装 | service |

### 16.3 为什么路由不再单独做成 skill 或 tool

当前实现选择把路由规则直接放进主代理 system prompt，原因是：

- 路由本质是判断方法论，不是外部调用能力
- 它属于每次请求都要执行的高频逻辑，放进 prompt 比运行时再读 skill 更快
- 业务 skill 仍然保留给知识问答和问题分析，扩展性不受影响

---

## 17. 分阶段实施

### Phase 1：搭建 Deep Agent 主体

内容：

1. 引入 `deepagents`
2. 新建 `DeepAgentService`
3. 用 `create_deep_agent(...)` 跑通最小链路
4. API 层切到 `DeepAgentService`

产出：

- 主图完全下线
- 系统可由单个 deep agent 处理请求

### Phase 2：实现 `domain_retrieve` tool

内容：

1. 复用现有检索实现
2. 收敛为一个统一 tool
3. 统一返回 evidence/citations/debug

产出：

- 检索链路不再暴露为节点

### Phase 3：落地三个核心 skills

内容：

1. 编写内置路由规则
2. 编写 `knowledge-qa`
3. 编写 `issue-analysis`

产出：

- 路由、知识问答、问题分析全部 skill 化
- 路由规则内置化，知识问答与问题分析 skill 化

### Phase 4：清理旧代码

内容：

1. 删除 workflow engine / subgraph
2. 删除 AgentLoop 主通路
3. 删除旧节点

产出：

- 系统只保留 deep agent 架构

### Phase 5：增强观测与后续扩展

内容：

1. 增加 skill/tool 级 tracing
2. 接入 memory/backend
3. 按需引入更多 tools 或补充 skills

---

## 18. 测试方案

### 18.1 单元测试

重点覆盖：

- `domain_retrieve` 输入输出
- module 增强逻辑
- retrieval bias 行为
- skill 文件合法性校验

### 18.2 集成测试

重点覆盖：

1. 典型知识问答
2. 典型问题分析
3. 超范围问题
4. 代码定位问题
5. 模块识别与 bias 选择

### 18.3 验收标准

至少满足：

1. Deep Agent 能基于内置路由规则独立完成路由
2. 主代理能正确读取并使用 skill
3. `domain_retrieve` 能覆盖当前 wiki/code 检索能力
4. 输出仍可带引用
5. API 可稳定返回统一结构

---

## 19. 最终建议

本次重构的关键不是“把当前主图换成另一个图”，而是彻底改变系统驱动方式：

- 从 **节点编排驱动**
- 变成 **Deep Agent 驱动**

在这个目标下，最合理的抽象就是：

- **路由 = skill**
- **路由 = system prompt 中的内置规则**
- **知识问答 = skill**
- **问题分析 = skill**
- **检索 = tool**

也就是说，整个系统的业务核心应被压缩成一句话：

**一个主代理，先依据内置路由规则判断任务类型，再通过 `knowledge-qa` 或 `issue-analysis` 决定回答方法，并在需要证据时调用唯一的 `domain_retrieve` tool 完成检索。**

---

## 20. 参考资料

- Deep Agents Overview: https://docs.langchain.com/oss/python/deepagents/overview
- Deep Agents Customization: https://docs.langchain.com/oss/python/deepagents/customization
- Deep Agents Skills: https://docs.langchain.com/oss/python/deepagents/skills
- Deep Agents Subagents: https://docs.langchain.com/oss/python/deepagents/subagents
