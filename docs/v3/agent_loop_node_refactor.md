# Agent Loop Node 重构设计

## 1. 概述

### 1.1 目标

将 `knowledge_answer` 和 `issue_analysis` 两个节点的共性逻辑抽取到 `common_nodes/agent_loop_node/` 作为基础节点，实现：

1. **代码复用**：共享 LLM 调用、Skill Tool 集成、Fallback 逻辑
2. **场景扩展**：`knowledge_answer` 和 `issue_analysis` 从基础节点扩展
3. **启动时初始化**：通过 `src/init` 模块完成所有组件初始化，避免延迟初始化
4. **入口统一**：API 入口统一为 `workflow/engine.py`
5. **配置隔离**：`src/agent` 无需直接感知私域配置，通过 `agent_loop_node` 传递

### 1.2 设计原则

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          重构设计原则                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   1. 基础节点优先：agent_loop_node 提供通用 LLM + Skill 调用能力             │
│                                                                              │
│   2. 场景节点扩展：knowledge_answer / issue_analysis 继承并定制              │
│                                                                              │
│   3. 统一初始化：通过 src/init 模块在程序启动时完成所有初始化                │
│                                                                              │
│   4. 配置驱动：通过 NodeConfig 配置行为，无需修改代码                        │
│                                                                              │
│   5. 配置隔离：src/agent 不感知私域配置，通过 agent_loop_node 传递           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 架构设计

### 2.1 节点继承关系

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           节点继承关系图                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   common_nodes/agent_loop_node/                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  BaseAgentLoopNode                                                   │   │
│   │  ├── __init__(config: AgentLoopNodeConfig)                          │   │
│   │  ├── run(service, state) -> dict                                    │   │
│   │  ├── _init_components()          # 启动时初始化 Skill/Tool 组件      │   │
│   │  ├── _build_llm_request()        # 构建 LLM 请求                     │   │
│   │  ├── _call_llm_with_agent()      # Agent 模式调用 LLM                │   │
│   │  ├── _validate_answer()          # 校验 LLM 输出                     │   │
│   │  └── _build_fallback()           # 构建 Fallback 响应                │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                    ▲                                         │
│                                    │ extends                                 │
│                    ┌───────────────┴───────────────┐                        │
│                    │                               │                        │
│   nodes/analysis/knowledge_answer/    nodes/analysis/issue_analysis/        │
│   ┌─────────────────────────────┐    ┌─────────────────────────────┐        │
│   │  KnowledgeAnswerNode        │    │  IssueAnalysisNode          │        │
│   │  ├── _get_system_prompt()   │    │  ├── _get_system_prompt()   │        │
│   │  ├── _get_user_prompt()     │    │  ├── _get_user_prompt()     │        │
│   │  ├── _validate_answer()     │    │  ├── _validate_answer()     │        │
│   │  ├── _build_fallback()      │    │  ├── _build_fallback()      │        │
│   │  └── _post_process()        │    │  └── _parse_structured()    │        │
│   └─────────────────────────────┘    └─────────────────────────────┘        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 启动流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           启动时初始化流程                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   API 启动                                                                   │
│       │                                                                      │
│       ▼                                                                      │
│   WorkflowEngine.__init__()                                                  │
│       │                                                                      │
│       ├── _init_llm_client()           # 初始化 LLM 客户端                   │
│       │                                                                      │
│       ├── _init_skill_components()     # 初始化 Skill Registry/Executor     │
│       │                                                                      │
│       ├── _init_tool_registry()        # 初始化 Tool Registry                │
│       │                                                                      │
│       └── _init_nodes()                # 初始化所有节点实例                  │
│            │                                                                 │
│            ├── knowledge_answer_node = KnowledgeAnswerNode(config)          │
│            │        └── 已包含 Skill 组件引用（启动时初始化）                 │
│            │                                                                 │
│            └── issue_analysis_node = IssueAnalysisNode(config)              │
│                     └── 共享 Skill 组件引用（单例）                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 目录结构

### 3.1 统一初始化模块（src/init）

```
src/
├── init/                                # 【新增】程序初始化模块
│   ├── __init__.py                      # 导出 initialize()
│   ├── initializer.py                   # 主初始化器
│   │
│   ├── domain_config.py                 # 私域配置加载
│   │   ├── load_domain_profile()        # 加载 profile.json
│   │   ├── load_skill_config()          # 加载技能配置
│   │   └── load_mcp_config()            # 加载 MCP 配置
│   │
│   ├── skill_initializer.py             # Skill 初始化
│   │   ├── init_skill_registry()        # 初始化技能注册中心
│   │   ├── load_skills()                # 加载领域技能
│   │   └── init_skill_executor()        # 初始化技能执行器
│   │
│   ├── mcp_initializer.py               # MCP 初始化
│   │   ├── init_mcp_client()            # 初始化 MCP 客户端
│   │   └── load_mcp_tools()             # 加载 MCP 工具
│   │
│   ├── retriever_initializer.py         # 检索器初始化
│   │   ├── init_wiki_index()            # 初始化 Wiki 索引
│   │   ├── init_code_index()            # 初始化代码索引
│   │   └── init_case_index()            # 初始化案例索引
│   │
│   ├── database_initializer.py          # 数据库初始化
│   │   ├── init_postgres()              # 初始化 PostgreSQL（按需）
│   │   ├── ensure_database()            # 确保数据库存在
│   │   └── init_tables()                # 初始化表结构
│   │
│   └── llm_initializer.py               # LLM 初始化
│       ├── init_llm_client()            # 初始化 LLM 客户端
│       └── validate_llm_config()        # 验证 LLM 配置
```

### 3.2 初始化流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          程序启动初始化流程                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   API 启动 (main.py)                                                         │
│       │                                                                      │
│       ▼                                                                      │
│   src/init/initializer.py::initialize()                                      │
│       │                                                                      │
│       ├──────────────────────────────────────────────────────────────────┐  │
│       │  阶段 1: 加载私域配置                                              │  │
│       │  ────────────────────────────────────────────────────────────    │  │
│       │  │                                                                │  │
│       │  ├── load_domain_profile()                                        │  │
│       │  │   └── 读取 domain/{domain_id}/profile.json                     │  │
│       │  │   └── 解析路由词表、检索配置、提示词配置等                      │  │
│       │  │                                                                │  │
│       │  ├── load_skill_config()                                          │  │
│       │  │   └── 读取 domain/{domain_id}/skills/ 目录                     │  │
│       │  │   └── 解析 SKILL.md 文件                                       │  │
│       │  │                                                                │  │
│       │  └── load_mcp_config()                                            │  │
│       │      └── 读取 domain/{domain_id}/mcp/ 配置                        │  │
│       │      └── 解析 MCP Server 配置                                     │  │
│       │                                                                │  │
│       └──────────────────────────────────────────────────────────────────┘  │
│       │                                                                      │
│       ├──────────────────────────────────────────────────────────────────┐  │
│       │  阶段 2: 初始化 Skill 系统                                         │  │
│       │  ────────────────────────────────────────────────────────────    │  │
│       │  │                                                                │  │
│       │  ├── init_skill_registry()                                        │  │
│       │  │   └── 创建全局 SkillRegistry 单例                              │  │
│       │  │                                                                │  │
│       │  ├── load_skills()                                                │  │
│       │  │   └── 从领域目录加载技能定义                                   │  │
│       │  │   └── 解析技能参数、关键词、模式等                             │  │
│       │  │                                                                │  │
│       │  └── init_skill_executor()                                        │  │
│       │      └── 创建 SkillExecutor（注入外部 registry）                  │  │
│       │                                                                │  │
│       └──────────────────────────────────────────────────────────────────┘  │
│       │                                                                      │
│       ├──────────────────────────────────────────────────────────────────┐  │
│       │  阶段 3: 初始化 MCP 系统                                           │  │
│       │  ────────────────────────────────────────────────────────────    │  │
│       │  │                                                                │  │
│       │  ├── init_mcp_client()                                            │  │
│       │  │   └── 连接 MCP Server                                          │  │
│       │  │   └── 验证连接状态                                             │  │
│       │  │                                                                │  │
│       │  └── load_mcp_tools()                                             │  │
│       │      └── 获取可用工具列表                                         │  │
│       │      └── 注册到 ToolRegistry                                      │  │
│       │                                                                │  │
│       └──────────────────────────────────────────────────────────────────┘  │
│       │                                                                      │
│       ├──────────────────────────────────────────────────────────────────┐  │
│       │  阶段 4: 初始化检索索引                                            │  │
│       │  ────────────────────────────────────────────────────────────    │  │
│       │  │                                                                │  │
│       │  ├── init_wiki_index()                                            │  │
│       │  │   └── 加载/构建 Wiki 向量索引                                  │  │
│       │  │   └── 预热 BM25 索引                                           │  │
│       │  │                                                                │  │
│       │  ├── init_code_index()                                            │  │
│       │  │   └── 加载/构建代码向量索引                                    │  │
│       │  │   └── 解析代码符号表                                           │  │
│       │  │                                                                │  │
│       │  └── init_case_index()                                            │  │
│       │      └── 加载/构建案例向量索引                                    │  │
│       │                                                                │  │
│       └──────────────────────────────────────────────────────────────────┘  │
│       │                                                                      │
│       ├──────────────────────────────────────────────────────────────────┐  │
│       │  阶段 5: 初始化数据库（按需）                                      │  │
│       │  ────────────────────────────────────────────────────────────    │  │
│       │  │                                                                │  │
│       │  ├── init_postgres()                                              │  │
│       │  │   └── 检查 DSN 配置                                            │  │
│       │  │   └── 建立连接池                                               │  │
│       │  │                                                                │  │
│       │  ├── ensure_database()                                            │  │
│       │  │   └── 自动创建数据库（如不存在）                               │  │
│       │  │                                                                │  │
│       │  └── init_tables()                                                │  │
│       │      └── 创建必要的表结构                                         │  │
│       │      └── LangGraph checkpoint 表                                  │  │
│       │                                                                │  │
│       └──────────────────────────────────────────────────────────────────┘  │
│       │                                                                      │
│       ├──────────────────────────────────────────────────────────────────┐  │
│       │  阶段 6: 初始化 LLM 客户端                                         │  │
│       │  ────────────────────────────────────────────────────────────    │  │
│       │  │                                                                │  │
│       │  ├── init_llm_client()                                            │  │
│       │  │   └── 读取 LLM 配置（API Key、Model 等）                       │  │
│       │  │   └── 创建 LLM 客户端实例                                      │  │
│       │  │                                                                │  │
│       │  └── validate_llm_config()                                        │  │
│       │      └── 验证 API 连接（可选）                                    │  │
│       │      └── 预热模型缓存                                             │  │
│       │                                                                │  │
│       └──────────────────────────────────────────────────────────────────┘  │
│       │                                                                      │
│       ▼                                                                      │
│   WorkflowEngine.__init__()                                                  │
│       │                                                                      │
│       ├── 注入已初始化的组件（配置、Skill、MCP、索引、LLM）                  │
│       │                                                                      │
│       └── _init_agent_loop_nodes()                                          │
│            └── 初始化 knowledge_answer / issue_analysis 节点                │
│            └── 通过 agent_loop_node 传递配置，src/agent 无需感知私域配置     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 设计优势

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          src/init 设计优势                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   1. 配置隔离：                                                              │
│      - src/agent 不直接依赖私域配置                                          │
│      - 配置通过 agent_loop_node 传递                                        │
│      - agent 模块可独立测试和复用                                            │
│                                                                              │
│   2. 启动时初始化：                                                          │
│      - 所有组件在程序启动时完成初始化                                        │
│      - 避免首次请求时的延迟初始化开销                                        │
│      - 启动失败可立即发现，而非运行时                                        │
│                                                                              │
│   3. 模块化初始化：                                                          │
│      - 每类组件有独立的初始化器                                              │
│      - 初始化失败可精确定位                                                  │
│      - 支持按需初始化（如数据库可选）                                        │
│                                                                              │
│   4. 统一入口：                                                              │
│      - 所有初始化逻辑集中在 src/init                                         │
│      - 便于管理初始化顺序和依赖                                              │
│      - 启动日志统一输出                                                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.4 新目录结构

```
src/
├── init/                                # 【新增】程序初始化模块
│   ├── __init__.py                      # 导出 initialize()
│   ├── initializer.py                   # 主初始化器
│   ├── domain_config.py                 # 私域配置加载
│   ├── skill_initializer.py             # Skill 初始化
│   ├── mcp_initializer.py               # MCP 初始化
│   ├── retriever_initializer.py         # 检索器初始化
│   ├── database_initializer.py          # 数据库初始化
│   └── llm_initializer.py               # LLM 初始化
│
├── workflow/
│   ├── engine.py                       # 【入口】WorkflowEngine
│   │
│   ├── common_nodes/                   # 公共节点
│   │   ├── __init__.py
│   │   ├── finalize_response.py        # 结果整理节点
│   │   ├── out_of_scope_response.py    # 超范围响应节点
│   │   │
│   │   └── agent_loop_node/            # 【核心】Agent Loop 基础节点
│   │       ├── __init__.py             # 导出 BaseAgentLoopNode, AgentLoopNodeConfig
│   │       ├── base.py                 # BaseAgentLoopNode 基类
│   │       └── config.py               # AgentLoopNodeConfig 配置类
│   │
│   ├── nodes/
│   │   └── analysis/
│   │       ├── knowledge_answer/       # 知识问答节点（扩展 agent_loop_node）
│   │       │   ├── __init__.py         # 导出 KnowledgeAnswerNode, run()
│   │       │   └── node.py             # KnowledgeAnswerNode 类实现
│   │       │
│   │       └── issue_analysis/         # 问题分析节点（扩展 agent_loop_node）
│   │           ├── __init__.py         # 导出 IssueAnalysisNode, run()
│   │           └── node.py             # IssueAnalysisNode 类实现
│   │
│   └── common/                         # 通用工具
│       └── ...
│
├── agent/                              # Agent 核心组件（无需感知私域配置）
│   ├── llm/
│   │   └── client.py                   # LLMClient（通用）
│   ├── tools/
│   │   └── registry.py                 # ToolRegistry（通用）
│   ├── skills/
│   │   ├── registry.py                 # SkillRegistry（通用）
│   │   ├── executor.py                 # SkillExecutor（通用）
│   │   └── manager.py                  # SkillManager（通用）
│   └── mcp/
│       └── client.py                   # MCPClient（通用）
│
├── api/
│   └── main.py                         # FastAPI 入口（调用 init.initialize()）
│
└── bootstrap/
    └── postgres_bootstrap.py           # PostgreSQL 引导（保留）
```

### 3.5 配置隔离原则

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          配置隔离原则                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   src/agent/ 模块：                                                          │
│   ─────────────────                                                          │
│   - 不依赖任何私域配置（domain/*）                                            │
│   - 不读取 profile.json、SKILL.md 等                                         │
│   - 通过 agent_loop_node 接收已初始化的组件                                   │
│   - 可独立测试和复用到其他项目                                                │
│                                                                              │
│   src/init/ 模块：                                                           │
│   ─────────────────                                                          │
│   - 负责加载私域配置                                                         │
│   - 负责初始化 Skill、MCP、索引、数据库                                       │
│   - 将初始化结果注入到 WorkflowEngine                                         │
│   - 提供 initialize() 统一入口                                               │
│                                                                              │
│   agent_loop_node：                                                          │
│   ─────────────────                                                          │
│   - 作为 src/agent 和私域配置的桥梁                                          │
│   - 接收 src/init 初始化的组件                                               │
│   - 传递给 knowledge_answer / issue_analysis 节点                            │
│   - 节点无需直接访问私域配置                                                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 核心组件设计

### 4.1 AgentLoopNodeConfig（配置类）

```python
# src/workflow/common_nodes/agent_loop_node/config.py

from dataclasses import dataclass, field
from typing import Any, Callable

@dataclass
class AgentLoopNodeConfig:
    """Agent Loop 节点配置

    用于配置节点的行为，包括：
    - 提示词模板
    - 工具/技能白名单
    - 循环控制参数
    - 校验和后处理函数
    """

    # ========== 基础配置 ==========
    node_name: str                          # 节点名称
    response_kind: str                      # 响应类型（knowledge_qa / issue_analysis）

    # ========== 提示词配置 ==========
    system_prompt_template: str             # 系统提示词模板
    user_prompt_template: str               # 用户提示词模板
    system_prompt_env_key: str | None = None  # 环境变量覆盖键

    # ========== 工具约束 ==========
    tool_whitelist: list[str] = field(default_factory=list)
    skill_whitelist: list[str] = field(default_factory=list)
    enable_skill_tool: bool = True          # 是否启用 Skill Tool

    # ========== 循环控制 ==========
    max_iterations: int = 3                 # Agent 模式最大迭代次数
    timeout_seconds: int = 60               # 超时时间

    # ========== 校验配置 ==========
    require_evidence: bool = True           # 是否需要证据
    validate_answer_func: Callable[[str], tuple[bool, str | None]] | None = None
    normalize_answer_func: Callable[[str], str] | None = None

    # ========== Fallback 配置 ==========
    enable_fallback: bool = True            # 是否启用规则 Fallback
    fallback_func: Callable | None = None   # 自定义 Fallback 函数

    # ========== 元数据 ==========
    metadata: dict[str, Any] = field(default_factory=dict)
```

### 4.2 BaseAgentLoopNode（基类）

```python
# src/workflow/common_nodes/agent_loop_node/base.py

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from agent.llm.client import LLMClient
from agent.skills.registry import SkillRegistry
from agent.skills.executor import SkillExecutor
from workflow.common_nodes.agent_loop_node.config import AgentLoopNodeConfig

logger = logging.getLogger(__name__)


class BaseAgentLoopNode(ABC):
    """Agent Loop 节点基类

    提供：
    1. LLM 调用能力（Agent 模式）
    2. Skill Tool 集成
    3. Fallback 机制
    4. 启动时初始化

    子类需要实现：
    1. _get_system_prompt() - 获取系统提示词
    2. _get_user_prompt() - 获取用户提示词
    3. _validate_answer() - 校验 LLM 输出（可选）
    4. _build_fallback() - 构建 Fallback 响应
    5. _post_process() - 后处理（可选）
    """

    def __init__(
        self,
        config: AgentLoopNodeConfig,
        llm_client: LLMClient,
        skill_registry: SkillRegistry | None = None,
        skill_executor: SkillExecutor | None = None,
    ):
        """初始化节点

        Args:
            config: 节点配置
            llm_client: LLM 客户端（启动时注入）
            skill_registry: Skill 注册中心（启动时注入，可选）
            skill_executor: Skill 执行器（启动时注入，可选）
        """
        self.config = config
        self.llm_client = llm_client
        self.skill_registry = skill_registry
        self.skill_executor = skill_executor

        # 启动时初始化 Skill Tool
        self._skill_tool = None
        if config.enable_skill_tool and skill_registry and skill_executor:
            self._skill_tool = self._create_skill_tool()
            logger.info(
                f"[{config.node_name}] Skill Tool 初始化完成: "
                f"技能数={len(skill_registry.skills) if skill_registry else 0}"
            )

    def _create_skill_tool(self):
        """创建 Skill Tool（LangChain Tool）"""
        from agent.skills import create_skill_tool
        return create_skill_tool(self.skill_registry, self.skill_executor)

    def run(self, service: Any, state: dict[str, Any]) -> dict[str, Any]:
        """节点执行入口

        Args:
            service: WorkflowService 实例
            state: 工作流状态

        Returns:
            更新后的状态增量
        """
        # 1. 提取状态
        user_query = str(state.get("user_query", ""))
        module_name = str(state.get("module_name", ""))
        module_hint = str(state.get("module_hint", ""))
        evidence_hits = self._collect_evidence(state)

        # 2. 尝试 LLM 调用
        llm_answer, llm_fallback_reason, llm_call_status = self._call_llm_with_agent(
            service=service,
            state=state,
            user_query=user_query,
            module_name=module_name,
            module_hint=module_hint,
            evidence_hits=evidence_hits,
        )

        # 3. 决定最终答案
        if llm_answer:
            final_answer = llm_answer
            generation_mode = "llm"
            if llm_call_status.get("attempts", 1) > 1:
                generation_mode = "skill_tool"
        else:
            # Fallback
            if self.config.enable_fallback:
                final_answer = self._build_fallback(
                    state=state,
                    module_name=module_name,
                    module_hint=module_hint,
                    evidence_hits=evidence_hits,
                )
                generation_mode = "fallback_rule"
            else:
                final_answer = "抱歉，无法处理您的请求。"
                generation_mode = "no_response"

        # 4. 后处理
        final_answer = self._post_process(final_answer, state)

        # 5. 构建返回结果
        return self._build_result(
            service=service,
            state=state,
            answer=final_answer,
            generation_mode=generation_mode,
            llm_fallback_reason=llm_fallback_reason,
            llm_call_status=llm_call_status,
            evidence_hits=evidence_hits,
        )

    def _call_llm_with_agent(
        self,
        service: Any,
        state: dict[str, Any],
        user_query: str,
        module_name: str,
        module_hint: str,
        evidence_hits: list[dict[str, Any]],
    ) -> tuple[str | None, str | None, dict[str, Any]]:
        """使用 Agent 模式调用 LLM

        Returns:
            (answer, fallback_reason, call_status) 元组
        """
        from workflow.llm.llm_client import CommonLLMRequest
        from workflow.llm.llm_prompt_utils import build_evidence_block, resolve_system_prompt

        # 获取提示词
        system_prompt = self._get_system_prompt(service, state)
        user_prompt = self._get_user_prompt(
            service=service,
            state=state,
            user_query=user_query,
            module_name=module_name,
            module_hint=module_hint,
            evidence_hits=evidence_hits,
        )

        # 构建请求
        request = CommonLLMRequest(
            node_name=self.config.node_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            evidence_count=len(evidence_hits),
            require_evidence=self.config.require_evidence,
            log_namespace=f"workflow.{self.config.node_name}",
            metadata={
                "module_name": module_name,
                "user_query_preview": user_query[:120],
                "skill_tool_enabled": self._skill_tool is not None,
                **self.config.metadata,
            },
            normalize_answer=self.config.normalize_answer_func,
            validate_answer=self.config.validate_answer_func,
        )

        # 准备工具
        tools = [self._skill_tool] if self._skill_tool else None

        # 调用 LLM
        if tools and hasattr(self.llm_client, "generate_with_agent"):
            result = self.llm_client.generate_with_agent(
                request,
                tools=tools,
                max_iterations=self.config.max_iterations
            )
            return result.answer, result.fallback_reason, dict(result.call_status)

        # 降级：普通模式
        if hasattr(self.llm_client, "generate_with_status"):
            return self.llm_client.generate_with_status(request)

        return None, "llm_client_not_available", self._default_call_status()

    def _collect_evidence(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        """收集证据"""
        from workflow.common.evidence import collect_evidence_hits
        return collect_evidence_hits(state)

    def _default_call_status(self) -> dict[str, Any]:
        """默认调用状态"""
        return {
            "status": "not_configured",
            "invoked": False,
            "request_sent": False,
            "attempts": 0,
            "latency_ms": 0,
            "reason": None,
            "model": None,
        }

    # ========== 子类必须实现的抽象方法 ==========

    @abstractmethod
    def _get_system_prompt(self, service: Any, state: dict[str, Any]) -> str:
        """获取系统提示词"""
        pass

    @abstractmethod
    def _get_user_prompt(
        self,
        service: Any,
        state: dict[str, Any],
        user_query: str,
        module_name: str,
        module_hint: str,
        evidence_hits: list[dict[str, Any]],
    ) -> str:
        """获取用户提示词"""
        pass

    @abstractmethod
    def _build_fallback(
        self,
        state: dict[str, Any],
        module_name: str,
        module_hint: str,
        evidence_hits: list[dict[str, Any]],
    ) -> str:
        """构建 Fallback 响应"""
        pass

    # ========== 可选的钩子方法 ==========

    def _post_process(self, answer: str, state: dict[str, Any]) -> str:
        """后处理（子类可覆盖）"""
        return answer

    def _build_result(
        self,
        service: Any,
        state: dict[str, Any],
        answer: str,
        generation_mode: str,
        llm_fallback_reason: str | None,
        llm_call_status: dict[str, Any],
        evidence_hits: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """构建返回结果（子类可覆盖）"""
        return {
            "response_kind": self.config.response_kind,
            "status": "completed",
            "answer": answer,
            "analysis": {
                "summary": f"{self.config.node_name} 已完成",
                "generation_mode": generation_mode,
                "llm_fallback_reason": llm_fallback_reason,
                "llm_call_status": llm_call_status,
                "evidence_count": len(evidence_hits),
            },
            "node_trace": service._trace(
                state,
                self.config.node_name,
                f"generation={generation_mode}",
            ),
        }
```

### 4.3 KnowledgeAnswerNode（知识问答节点）

```python
# src/workflow/nodes/analysis/knowledge_answer/node.py

from __future__ import annotations

import logging
from typing import Any

from workflow.common_nodes.agent_loop_node.base import BaseAgentLoopNode
from workflow.common_nodes.agent_loop_node.config import AgentLoopNodeConfig

logger = logging.getLogger(__name__)


class KnowledgeAnswerNode(BaseAgentLoopNode):
    """知识问答节点

    扩展 BaseAgentLoopNode，定制：
    1. QA 专用提示词模板
    2. 代码定位问题的特殊校验
    3. 证据驱动的 Fallback
    """

    # 提示词模板
    SYSTEM_PROMPT_TEMPLATE = (
        "你是企业知识问答助手。"
        "必须严格基于提供的证据回答，不补充证据外事实。"
        "如果用户的问题需要查询实时数据或执行特定技能，请使用 skill_tool。"
        "输出中文，结构尽量为：结论 -> 依据。"
    )

    USER_PROMPT_TEMPLATE = """【用户问题】
{user_query}

【当前主模块】
- module_name: {module_name}
- module_hint: {module_hint}

【相关模块】
{related_modules_block}

【检索证据（按相关性排序）】
{evidence_block}
"""

    def __init__(
        self,
        llm_client,
        skill_registry=None,
        skill_executor=None,
        config: AgentLoopNodeConfig | None = None,
    ):
        if config is None:
            config = AgentLoopNodeConfig(
                node_name="knowledge_answer",
                response_kind="knowledge_qa",
                system_prompt_template=self.SYSTEM_PROMPT_TEMPLATE,
                user_prompt_template=self.USER_PROMPT_TEMPLATE,
                system_prompt_env_key="WORKFLOW_QA_LLM_SYSTEM_PROMPT",
                enable_skill_tool=True,
                max_iterations=3,
                require_evidence=True,
                validate_answer_func=self._validate_qa_answer,
            )
        super().__init__(config, llm_client, skill_registry, skill_executor)

    def _get_system_prompt(self, service: Any, state: dict[str, Any]) -> str:
        from workflow.llm.llm_prompt_utils import resolve_system_prompt
        return resolve_system_prompt(
            env_key=self.config.system_prompt_env_key,
            default_prompt=self.config.system_prompt_template,
            domain_profile=getattr(service, "domain_profile", None),
        )

    def _get_user_prompt(
        self,
        service: Any,
        state: dict[str, Any],
        user_query: str,
        module_name: str,
        module_hint: str,
        evidence_hits: list[dict[str, Any]],
    ) -> str:
        from workflow.llm.llm_prompt_utils import build_evidence_block

        related_modules = list(state.get("related_modules", []) or [])
        related_modules_block = self._build_related_modules_block(related_modules)

        return self.config.user_prompt_template.format(
            user_query=user_query,
            module_name=module_name,
            module_hint=module_hint,
            related_modules_block=related_modules_block,
            evidence_block=build_evidence_block(evidence_hits),
        )

    def _build_related_modules_block(self, related_modules: list[dict[str, Any]]) -> str:
        """构建相关模块展示文本"""
        if not related_modules:
            return "- 无"
        rows = []
        for item in related_modules[:3]:
            module_name = str(item.get("module_name", "")).strip()
            module_hint = str(item.get("module_hint", "")).strip() or "--"
            if module_name:
                rows.append(f"- module_name: {module_name}")
                rows.append(f"  module_hint: {module_hint}")
        return "\n".join(rows) if rows else "- 无"

    def _validate_qa_answer(self, answer: str) -> tuple[bool, str | None]:
        """校验 QA 答案"""
        from workflow.llm.llm_prompt_utils import looks_like_reasoning_dump

        if not answer.strip():
            return False, "empty_answer"
        if looks_like_reasoning_dump(answer):
            return False, "empty_answer:reasoning_dump"
        return True, None

    def _build_fallback(
        self,
        state: dict[str, Any],
        module_name: str,
        module_hint: str,
        evidence_hits: list[dict[str, Any]],
    ) -> str:
        """构建 Fallback 响应"""
        # 复用现有的 Fallback 逻辑
        # ...
        pass

    def _post_process(self, answer: str, state: dict[str, Any]) -> str:
        """后处理：应用结构化输出"""
        user_query = str(state.get("user_query", ""))
        question_type = self._infer_question_type(user_query)
        return self._enforce_structured_output(answer, question_type=question_type)

    def _infer_question_type(self, user_query: str) -> str:
        """推断问题类型"""
        # 复用现有逻辑
        pass

    def _enforce_structured_output(self, answer: str, question_type: str) -> str:
        """强制结构化输出"""
        # 复用现有逻辑
        pass


# 便捷函数入口
def run(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """节点入口函数（兼容现有调用方式）"""
    node = service._get_node("knowledge_answer")
    return node.run(service, state)
```

### 4.4 初始化模块设计

```python
# src/init/initializer.py

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class InitializationResult:
    """初始化结果"""
    domain_profile: Any                    # 私域配置
    skill_registry: Any | None             # Skill 注册中心
    skill_executor: Any | None             # Skill 执行器
    mcp_client: Any | None                 # MCP 客户端
    wiki_retriever: Any | None             # Wiki 检索器
    code_retriever: Any | None             # 代码检索器
    llm_client: Any | None                 # LLM 客户端
    db_connection: Any | None              # 数据库连接（可选）


def initialize(
    *,
    domain_id: str = "ad_engine",
    project_root: Path | None = None,
    enable_database: bool = True,
    enable_mcp: bool = True,
) -> InitializationResult:
    """统一初始化入口

    Args:
        domain_id: 领域 ID
        project_root: 项目根目录
        enable_database: 是否初始化数据库
        enable_mcp: 是否初始化 MCP

    Returns:
        InitializationResult 包含所有初始化的组件
    """
    if project_root is None:
        project_root = Path.cwd()

    logger.info(f"[Init] 开始初始化，domain_id={domain_id}")

    # 1. 加载私域配置
    from init.domain_config import load_domain_profile
    domain_profile = load_domain_profile(domain_id, project_root)
    logger.info(f"[Init] 私域配置加载完成: {domain_profile.profile_id}")

    # 2. 初始化 Skill 组件
    from init.skill_initializer import init_skill_registry, init_skill_executor
    skill_registry = init_skill_registry(domain_id, project_root)
    skill_executor = init_skill_executor() if skill_registry else None
    logger.info(f"[Init] Skill 初始化完成: 技能数={len(skill_registry.skills) if skill_registry else 0}")

    # 3. 初始化 MCP 组件（可选）
    mcp_client = None
    if enable_mcp:
        from init.mcp_initializer import init_mcp_client
        mcp_client = init_mcp_client(domain_id, project_root)
        logger.info(f"[Init] MCP 初始化完成: {'成功' if mcp_client else '跳过'}")

    # 4. 初始化检索索引
    from init.retriever_initializer import init_retrievers
    wiki_retriever, code_retriever = init_retrievers(domain_profile, project_root)
    logger.info(f"[Init] 检索器初始化完成")

    # 5. 初始化 LLM 客户端
    from init.llm_initializer import init_llm_client
    llm_client = init_llm_client()
    logger.info(f"[Init] LLM 客户端初始化完成")

    # 6. 初始化数据库（可选）
    db_connection = None
    if enable_database:
        from init.database_initializer import init_postgres
        db_connection = init_postgres()
        logger.info(f"[Init] 数据库初始化完成: {'成功' if db_connection else '跳过'}")

    result = InitializationResult(
        domain_profile=domain_profile,
        skill_registry=skill_registry,
        skill_executor=skill_executor,
        mcp_client=mcp_client,
        wiki_retriever=wiki_retriever,
        code_retriever=code_retriever,
        llm_client=llm_client,
        db_connection=db_connection,
    )

    logger.info("[Init] 所有组件初始化完成")
    return result
```

### 4.5 WorkflowEngine 初始化（使用 src/init）

```python
# src/workflow/engine.py（关键部分）

from init import initialize, InitializationResult

class WorkflowService:
    """Workflow orchestrator."""

    def __init__(self) -> None:
        self.backend_name = BACKEND_NAME

        # 1. 通过 src/init 统一初始化所有组件
        self._init_result: InitializationResult = initialize(
            domain_id="ad_engine",
            project_root=Path(__file__).resolve().parents[2],
            enable_database=True,
            enable_mcp=True,
        )

        # 2. 从初始化结果中获取组件
        self.domain_profile = self._init_result.domain_profile
        self._wiki_retriever = self._init_result.wiki_retriever
        self._code_retriever = self._init_result.code_retriever
        self._llm_client = self._init_result.llm_client
        self._skill_registry = self._init_result.skill_registry
        self._skill_executor = self._init_result.skill_executor
        self._mcp_client = self._init_result.mcp_client

        # 3. 初始化 Agent Loop 节点（启动时完成）
        self._init_agent_loop_nodes()

        # 4. 初始化 Checkpointer
        self._checkpointer = self._init_checkpointer()

        # 5. 构建工作流图
        self._graph = self._build_graph()

        logger.info(
            "workflow.service.initialized",
            domain_profile=self.domain_profile.profile_id,
            skill_count=len(self._skill_registry.skills) if self._skill_registry else 0,
            mcp_enabled=self._mcp_client is not None,
        )

    def _init_agent_loop_nodes(self) -> None:
        """初始化 Agent Loop 节点

        说明：
        src/agent 模块无需感知私域配置，
        所有配置通过 agent_loop_node 传递。
        """
        from workflow.nodes.analysis.knowledge_answer import init_node as init_knowledge_answer
        from workflow.nodes.analysis.issue_analysis import init_node as init_issue_analysis

        # 初始化 knowledge_answer 节点
        init_knowledge_answer(
            llm_client=self._llm_client,
            skill_registry=self._skill_registry,
            skill_executor=self._skill_executor,
        )

        # 初始化 issue_analysis 节点
        init_issue_analysis(
            llm_client=self._llm_client,
            skill_registry=self._skill_registry,
            skill_executor=self._skill_executor,
        )

        logger.info(
            "workflow.agent_loop_nodes.initialized",
            nodes=["knowledge_answer", "issue_analysis"],
        )
```

---

## 5. 配置示例

### 5.1 Domain Profile 配置

```json
// domain/ad_engine/profile.json
{
  "profile_id": "ad_engine",
  "display_name": "广告引擎",

  "nodes": {
    "knowledge_answer": {
      "system_prompt_append": "你是广告引擎领域的知识问答助手...",
      "max_iterations": 3,
      "enable_skill_tool": true,
      "skill_whitelist": ["query_code", "query_wiki"]
    },
    "issue_analysis": {
      "system_prompt_append": "你是广告引擎领域的问题分析专家...",
      "max_iterations": 5,
      "enable_skill_tool": true,
      "skill_whitelist": ["analyze_error", "query_metrics"]
    }
  }
}
```

---

## 6. 迁移计划

### 6.0 Phase 0: 创建 src/init 初始化模块

| 步骤 | 任务 | 文件 |
|------|------|------|
| 0.1 | 创建初始化模块骨架 | `src/init/__init__.py`, `src/init/initializer.py` |
| 0.2 | 创建私域配置加载器 | `src/init/domain_config.py` |
| 0.3 | 创建 Skill 初始化器 | `src/init/skill_initializer.py` |
| 0.4 | 创建 MCP 初始化器 | `src/init/mcp_initializer.py` |
| 0.5 | 创建检索器初始化器 | `src/init/retriever_initializer.py` |
| 0.6 | 创建数据库初始化器 | `src/init/database_initializer.py` |
| 0.7 | 创建 LLM 初始化器 | `src/init/llm_initializer.py` |

### 6.1 Phase 1: 创建基础节点

| 步骤 | 任务 | 文件 |
|------|------|------|
| 1.1 | 创建 AgentLoopNodeConfig | `common_nodes/agent_loop_node/config.py` |
| 1.2 | 创建 BaseAgentLoopNode | `common_nodes/agent_loop_node/base.py` |
| 1.3 | 更新 `__init__.py` | `common_nodes/agent_loop_node/__init__.py` |

### 6.2 Phase 2: 重构 knowledge_answer

| 步骤 | 任务 | 文件 |
|------|------|------|
| 2.1 | 创建 KnowledgeAnswerNode | `nodes/analysis/knowledge_answer/node.py` |
| 2.2 | 创建 KnowledgeAnswerConfig | `nodes/analysis/knowledge_answer/config.py` |
| 2.3 | 更新 `__init__.py` | `nodes/analysis/knowledge_answer/__init__.py` |
| 2.4 | 删除旧实现 | 移除原有代码 |

### 6.3 Phase 3: 重构 issue_analysis

| 步骤 | 任务 | 文件 |
|------|------|------|
| 3.1 | 创建 IssueAnalysisNode | `nodes/analysis/issue_analysis/node.py` |
| 3.2 | 创建 IssueAnalysisConfig | `nodes/analysis/issue_analysis/config.py` |
| 3.3 | 更新 `__init__.py` | `nodes/analysis/issue_analysis/__init__.py` |
| 3.4 | 删除旧实现 | 移除原有代码 |

### 6.5 Phase 5: 更新 WorkflowEngine

| 步骤 | 任务 | 文件 |
|------|------|------|
| 5.1 | 集成 src/init 初始化 | `workflow/engine.py` |
| 5.2 | 移除原有的 Skill 初始化逻辑 | `workflow/engine.py` |
| 5.3 | 更新节点调用方式 | `workflow/engine.py` |

### 6.6 Phase 6: 测试验证

| 步骤 | 任务 |
|------|------|
| 6.1 | 单元测试 |
| 6.2 | 集成测试 |
| 6.3 | API 端到端测试 |

---

## 7. 总结

### 7.1 核心变更

| 变更 | 说明 |
|------|------|
| **src/init 模块** | 统一的程序初始化入口，避免延迟初始化 |
| **配置隔离** | src/agent 不直接感知私域配置，通过 agent_loop_node 传递 |
| **BaseAgentLoopNode** | 抽取 LLM + Skill 调用的共性逻辑 |
| **启动时初始化** | 所有组件在程序启动时完成初始化 |
| **依赖注入** | LLM 客户端和 Skill 组件通过构造函数注入节点 |
| **入口统一** | API 入口保持为 `workflow/engine.py` |

### 7.2 优势

1. **代码复用**：`knowledge_answer` 和 `issue_analysis` 共享基础逻辑
2. **易于扩展**：新增场景只需继承 BaseAgentLoopNode
3. **启动时初始化**：避免请求时的延迟初始化开销
4. **配置隔离**：src/agent 可独立测试和复用
5. **配置驱动**：通过配置类定制行为
6. **模块化初始化**：每类组件有独立的初始化器，便于管理
