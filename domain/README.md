# Domain Layout

`domain/<domain_id>/` 用于承载单个领域的配置、知识语料、代码语料、skills 和 MCP 配置。

当前仓库已经统一到这条运行链路：

- `src/api/main.py` 负责 API 入口
- `src/init/initializer.py` 负责领域配置、Retriever、MCP 初始化
- `src/agent/factory.py` 负责 Deep Agent、skills、tools 装配
- `src/retrievers/tools/domain_retrieve_tool.py` 负责统一检索工具

这意味着 `domain/` 目录现在服务的是“单 Deep Agent + skills + domain_retrieve”的运行模式，而不是旧的多 workflow 节点编排。

## Directory Layout

```text
domain/
  README.md
  ad_engine/
    profile.json
    wiki/
      *.md
    codes/
      ...
    prompts/
      deep_agent_system.md
    skills/
      knowledge-qa/
      issue-analysis/
    mcp_servers/
      servers.yaml
```

最小可运行领域目录建议至少包含：

- `profile.json`
- `wiki/`
- `codes/`
- `prompts/deep_agent_system.md`
- `skills/`

## Startup

推荐通过领域目录启动：

```powershell
$env:AGENT_DOMAIN_DIR = "domain/ad_engine"
$env:PYTHONPATH = "src"
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

或：

```powershell
./start_agent.ps1 -DomainDir domain/ad_engine
```

当前领域定位优先级：

1. `AGENT_DOMAIN_PROFILE_PATH`
2. `AGENT_DOMAIN_DIR`
3. `AGENT_DOMAIN_PROFILE + AGENT_DOMAIN_PROFILE_DIR`
4. 默认 `domain/ad_engine/profile.json`

对应实现见 [profile.py](/d:/codes/dsp_agent/src/domain_profile/profile.py)。

## Current Profile Contract

下面这些字段仍然直接参与当前主链路。

### Top-level Fields

- `schema_version`
- `profile_id`
- `display_name`
- `language`
- `sources`
- `routing`
- `modules`
- `retrieval`
- `prompts`
- `deep_agents`

### `sources`

- `sources.wiki.root`
- `sources.code.roots`

说明：

- wiki 真实加载入口使用目录根路径
- code 真实加载入口使用根目录列表

### `routing`

当前真正生效的只有：

- `routing.default_module`

它用于模块规则推断失败时的兜底模块。

### `modules`

`modules` 仍然是当前主链路核心配置，直接参与：

- 主模块推断
- 相关模块推断
- wiki 文档提示

建议保留字段：

- `name`
- `hint`
- `route_priority`
- `keywords`
- `symbol_keywords`
- `aliases`
- `wiki_hints`

### `retrieval`

当前会被检索链路消费的字段：

- `presets`
- `source_weights`
- `max_per_source`
- `enable_wiki`
- `enable_code`
- `embedding`
- `reranker`
- `hybrid_weights`
- `module_prior_boost`

### `prompts`

当前只保留：

- `prompts.deep_agent_system_path`

`qa_system_path`、`issue_system_path` 已不再参与当前主链路。

### `deep_agents`

当前真正生效的只有：

- `deep_agents.skills_root`

skills 通过该目录挂到 Deep Agent。

## Legacy Fields

下面这些字段当前已经不参与运行时决策，属于遗留配置；如果继续精简，可以优先考虑删除。

- `sources.wiki.glob`
- `sources.code.include_ext`
- `sources.code.exclude_dirs`
- `fusion.*`
- `answering.*`
- `deep_agents.enabled`
- `deep_agents.primary_skills`
- `deep_agents.tools.*`

这些字段有些仍然保留在 `domain/ad_engine/profile.json` 中，但当前主链路并不会消费它们。

## Minimal Example

```json
{
  "schema_version": 1,
  "profile_id": "ad_engine",
  "display_name": "广告引擎",
  "language": "zh-CN",
  "sources": {
    "wiki": {
      "root": "wiki"
    },
    "code": {
      "roots": ["codes"]
    }
  },
  "routing": {
    "default_module": "ad-serving-orchestrator"
  },
  "modules": [
    {
      "name": "ad-serving-orchestrator",
      "hint": "广告在线投放编排与请求链路概览",
      "route_priority": 100,
      "keywords": ["广告引擎", "在线投放", "链路"],
      "aliases": ["广告在线投放"],
      "wiki_hints": ["00-", "总体架构"]
    }
  ],
  "retrieval": {
    "presets": {
      "hybrid": { "wiki_top_k": 4, "code_top_k": 4, "final_top_k": 6 },
      "wiki_first": { "wiki_top_k": 6, "code_top_k": 2, "final_top_k": 6 },
      "code_first": { "wiki_top_k": 2, "code_top_k": 7, "final_top_k": 6 }
    },
    "source_weights": {
      "wiki": 1.0,
      "code": 1.0
    },
    "max_per_source": {
      "wiki": 4,
      "code": 4
    },
    "enable_wiki": true,
    "enable_code": true
  },
  "prompts": {
    "deep_agent_system_path": "prompts/deep_agent_system.md"
  },
  "deep_agents": {
    "skills_root": "domain/ad_engine/skills"
  }
}
```

## Maintenance Notes

- 新增领域能力前，先确认它是否真的会被当前 Deep Agent 主链路消费
- 阈值、路由、检索参数优先放 `profile.json`
