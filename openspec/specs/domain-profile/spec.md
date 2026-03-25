# Domain Profile Capability Spec

## 概述

领域配置管理，解析 `profile.json`，提供模块推断、路径解析和全局配置访问。

## 核心能力

### 1. 配置类 (profile.py)

| 类 | 说明 |
|------|------|
| `DomainProfile` | 领域配置主类 |
| `ModuleProfile` | 模块配置 |
| `EmbeddingProfile` | 向量检索配置 |
| `RerankerProfile` | 重排器配置 |
| `RetrievalProfile` | 检索配置 |
| `AnsweringProfile` | 回答配置 |

### 2. 核心函数

| 函数 | 说明 |
|------|------|
| `get_domain_profile()` | 获取全局单例 |
| `load_domain_profile()` | 从 JSON 加载配置 |
| `resolve_domain_profile_path()` | 解析配置文件路径 |
| `infer_module()` | 推断查询所属模块 |
| `infer_related_modules()` | 推断相关模块 |

## 配置结构

### DomainProfile

```python
@dataclass
class DomainProfile:
    profile_id: str              # 配置 ID
    display_name: str            # 显示名称
    language: str                # 语言
    schema_version: str          # Schema 版本
    sources: SourcesConfig       # 数据源配置
    routing: RoutingConfig       # 路由配置
    modules: list[ModuleProfile] # 模块列表
    retrieval: RetrievalProfile  # 检索配置
    answering: AnsweringProfile  # 回答配置
    prompts: PromptsConfig       # 提示词配置
    domain_dir: Path             # 领域目录
    raw: dict                    # 原始 JSON
```

### ModuleProfile

```python
@dataclass
class ModuleProfile:
    name: str                    # 模块名
    hint: str                    # 提示信息
    route_priority: int          # 路由优先级
    keywords: list[str]          # 关键词
    symbol_keywords: list[str]   # 符号关键词
    aliases: list[str]           # 别名
    wiki_hints: list[str]        # Wiki 提示
```

### RetrievalProfile

```python
@dataclass
class RetrievalProfile:
    presets: dict                # 预设配置
    source_weights: dict         # 来源权重
    max_per_source: int          # 每源最大结果
    enable_wiki: bool            # 启用 Wiki
    enable_code: bool            # 启用代码
    embedding: EmbeddingProfile  # 向量配置
    reranker: RerankerProfile    # 重排配置
    hybrid_weights: dict         # 混合权重
```

## 模块推断

```
┌─────────────────────────────────────────────────────────────┐
│                    Module Inference                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Query: "如何使用 AgentService"                             │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────────┐                                        │
│  │ Keyword Match   │ ──► keywords, symbol_keywords          │
│  └────────┬────────┘                                        │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────┐                                        │
│  │ Alias Match     │ ──► aliases                            │
│  └────────┬────────┘                                        │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────┐                                        │
│  │ Route Priority  │ ──► 最高优先级胜出                     │
│  └────────┬────────┘                                        │
│           │                                                 │
│           ▼                                                 │
│  Result: "agent" module                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 配置文件位置

```
domain/
  <domain_id>/
    profile.json        # 领域配置
    prompts/            # 提示词模板
    wiki/               # Wiki 语料
    codes/              # 代码语料
    mcp_servers/        # MCP Server 配置
    skills/             # Deep Agent 技能
```

## 依赖关系

```
domain_profile
  └── (无依赖，被所有模块依赖)
```

## 全局单例

```python
# 获取领域配置
profile = get_domain_profile()

# 推断模块
module = profile.infer_module("如何使用 AgentService")

# 获取检索配置
retrieval = profile.retrieval
```

## 约束

1. 领域规则优先放到 `domain/<id>/profile.json`
2. 不把阈值和路由硬编码进 service/api
3. 配置变更需要向后兼容
4. 模块推断逻辑变更需要测试覆盖
