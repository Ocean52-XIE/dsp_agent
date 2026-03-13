# Domain Layout

每个领域独立放在 `domain/<domain_id>/` 下，目录至少包含：

- `profile.json`：领域配置。
- `wiki/`：知识文档语料。
- `codes/`：代码语料。
- `eval/datasets/`：离线评测数据集。

启动时只需指定领域目录，例如：

```powershell
$env:WORKFLOW_DOMAIN_DIR = "domain/ad_engine"
$env:PYTHONPATH = "src"
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

或：

```powershell
./start_agent.ps1 -DomainDir domain/ad_engine
```

# Domain Profile 配置与目录规范

本文档说明如何按“一个 domain 目录承载完整领域数据”的方式运行系统。

## 1. 目录结构

```text
domain/
  ad_engine/
    profile.json
    wiki/
      *.md
    codes/
      ...
    eval/
      datasets/
        *.jsonl
  logistics/
    profile.json
    wiki/
    codes/
    eval/
      datasets/
```

## 2. 启动方式

只需要指定 `domain` 子目录（领域目录）：

- 环境变量：`WORKFLOW_DOMAIN_DIR=domain/ad_engine`
- 或启动脚本参数：`./start_agent.ps1 -DomainDir domain/ad_engine`
- API 启动入口：`src/api/main.py`（`$env:PYTHONPATH='src'; python -m uvicorn api.main:app ...`）

系统会自动读取：

1. `domain/ad_engine/profile.json`
2. profile 中定义的 `sources.wiki.root`（相对 domain 目录）
3. profile 中定义的 `sources.code.roots`（相对 domain 目录）
4. profile 中定义的 `eval.*` 数据集路径（相对 domain 目录）

## 3. 解析优先级

1. `WORKFLOW_DOMAIN_PROFILE_PATH`（直接指定 profile 文件）
2. `WORKFLOW_DOMAIN_DIR`（指定领域目录）
3. `WORKFLOW_DOMAIN_PROFILE + WORKFLOW_DOMAIN_PROFILE_DIR`（兼容旧模式）
4. 默认：`domain/ad_engine/profile.json`

## 4. profile 字段清单（详细说明）

下面按“代码实际读取行为”说明字段。未特别说明时，路径字段都支持：
- 相对 `domain/<domain_id>/` 目录
- 或相对项目根目录
- 或绝对路径

### 4.1 顶层字段

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `schema_version` | `int` | 否 | `1` | profile 版本号，当前主要用于配置演进标识。 |
| `profile_id` | `string` | 否 | `ad_engine` | 领域唯一标识，用于默认模板路径渲染、日志标识等。 |
| `display_name` | `string` | 否 | `profile_id` | 领域展示名称，用于欢迎语、可观测输出。 |
| `language` | `string` | 否 | `zh-CN` | 领域默认语言偏好。 |
| `sources` | `object` | 否 | `{}` | 语料路径配置（wiki/code/cases）。 |
| `routing` | `object` | 否 | `{}` | 模块路由基础配置。 |
| `modules` | `array<object>` | 否 | `[]` | 模块清单（名称、关键词、优先级、符号词等）。 |
| `domain_gate` | `object` | 否 | 内置默认 | 领域内外判定与短 query 兜底规则。 |
| `query_rewrite` | `object` | 否 | 内置默认 | 检索改写词典、意图词、符号别名。 |
| `retrieval` | `object` | 否 | 内置默认 | 多源检索 TopK、权重、配额、开关。 |
| `answering` | `object` | 否 | 内置默认 | 问答器领域词与默认入口符号。 |
| `prompts` | `object` | 否 | `{}` | 领域 prompt，当前主要使用 `qa_system`。 |
| `code_generation` | `object` | 否 | `{}` | 代码生成上下文模板路径等。 |
| `ui` | `object` | 否 | `{}` | UI 文案配置，当前主要使用 `welcome_message`。 |
| `eval` | `object` | 否 | `{}` | 评测数据集路径映射（供 eval 脚本读取）。 |

### 4.2 `sources` 子字段

| 字段 | 类型 | 必填 | 默认值 | 生效情况 |
|---|---|---|---|---|
| `sources.wiki.root` | `string` | 否 | `wiki` | **已生效**。用于定位 Wiki 根目录。 |
| `sources.wiki.glob` | `string` | 否 | 无 | 预留字段，当前核心加载逻辑按 `.md` 递归扫描。 |
| `sources.code.roots` | `array<string>` | 否 | `["codes"]` | **已生效**。用于定位代码检索根目录列表。 |
| `sources.code.include_ext` | `array<string>` | 否 | 无 | 预留字段，当前扩展名过滤由代码检索器内置常量控制。 |
| `sources.code.exclude_dirs` | `array<string>` | 否 | 无 | 预留字段，当前排除目录由代码检索器内置常量控制。 |
| `sources.cases.path` | `string` | 否 | 无 | 预留/评测辅助字段，当前主链路默认不启用 `retrieve_cases`。 |

### 4.3 `routing` 子字段

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `routing.default_module` | `string` | 否 | `modules[0].name` 或 `default-module` | **已生效**。模块推断失败时的兜底模块。 |
| `routing.module_infer_strategy` | `string` | 否 | 无 | 预留策略标识，当前版本未直接读取。 |
| `routing.prefer_symbol_match` | `bool` | 否 | 无 | 预留策略标识，当前版本未直接读取。 |

### 4.4 `modules` 元素字段

每个模块对象支持如下字段：

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `name` | `string` | 是 | 无 | 模块唯一名。 |
| `hint` | `string` | 否 | `""` | 模块描述，参与解释/提示。 |
| `route_priority` | `int` | 否 | `100` | 路由优先级，值越小优先级越高。 |
| `keywords` | `array<string>` | 否 | `[]` | 文本关键词匹配。 |
| `symbol_keywords` | `array<string>` | 否 | `[]` | 符号级匹配词（函数/文件名），优先于普通关键词。 |
| `aliases` | `array<string>` | 否 | `[]` | 同义别名匹配。 |
| `wiki_hints` | `array<string>` | 否 | `[]` | Wiki 检索先验提示，参与模块相关文档 boost。 |

### 4.5 `domain_gate` 子字段

用于 `domain_gate` 节点判定是否领域内问题：

- 分值阈值类：
  - `threshold`（默认 `0.5`）
  - `weak_in_scope_min_score`（默认 `0.62`）
  - `weak_code_hint_min_score`（默认 `0.58`）
  - `history_memory_bonus`（默认 `0.18`）
  - `offtopic_penalty`（默认 `0.45`）
  - `short_query_penalty`（默认 `0.25`）
  - `short_query_max_len`（默认 `4`）
- 正则类：
  - `code_hint_regex`
  - `laugh_like_regex`
- 词表类：
  - `domain_terms`
  - `small_talk_exact`
  - `small_talk_substr`
  - `offtopic_terms`

### 4.6 `query_rewrite` 子字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `synonyms` | `object<string, array<string>>` | 同义词扩展，生成更多检索 query。 |
| `abbreviations` | `object<string, string>` | 缩写展开（如 `ctr -> 点击率`）。 |
| `symbol_aliases` | `object<string, array<string>>` | 概念到代码符号/文件名映射。 |
| `intent_terms` | `object<string, array<string>>` | 意图词表（如 `metric/pipeline/code/code_location`）。 |
| `query_templates` | `array<object>` | 模板化改写配置。 |

### 4.7 `retrieval` 子字段

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `presets.<name>.wiki_top_k` | `int` | `4` | 预设下 Wiki 召回数。 |
| `presets.<name>.code_top_k` | `int` | `4` | 预设下 Code 召回数。 |
| `presets.<name>.case_top_k` | `int` | `2` | 预设下 Case 召回数。 |
| `presets.<name>.final_top_k` | `int` | `6` | 融合后的最终证据数。 |
| `source_weights.wiki/code/case` | `float` | `1.0/1.0/0.6` | 融合权重。 |
| `max_per_source.wiki/code/case` | `int` | `4/4/1` | 每个来源的融合配额上限。 |
| `enable_wiki` | `bool` | `true` | 是否启用 wiki 检索。 |
| `enable_code` | `bool` | `true` | 是否启用 code 检索。 |
| `enable_cases` | `bool` | `false` | 是否启用 cases 检索（当前默认关闭）。 |

### 4.8 `answering` 子字段

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `calibration_terms` | `array<string>` | `[]` | 校准/两率相关识别词。 |
| `bid_terms` | `array<string>` | `[]` | 出价相关识别词。 |
| `bid_entry_terms` | `array<string>` | `[]` | “入口函数”类识别词。 |
| `default_entry_symbol` | `string` | `main_entry` | 默认入口符号兜底。 |

### 4.9 `prompts` / `code_generation` / `ui` / `eval`

- `prompts`
  - `qa_system`：**已生效**，用于问答系统提示词。
- `code_generation`
  - `file_templates`：**已生效**，用于构造代码上下文候选路径模板。
- `ui`
  - `welcome_message`：**已生效**，未配置时自动回退默认欢迎语。
- `eval`
  - 常用键：`retrieval_dataset`、`code_retrieval_dataset`、`answer_dataset`、`hybrid_answer_dataset`。
  - 说明：运行评测脚本时可按键读取对应数据集路径。

### 4.10 最小可用 profile 建议

若只想最小化配置并先跑通，建议至少包含：
- `profile_id`
- `sources.wiki.root`
- `sources.code.roots`
- `modules`（至少一个，含 `name`）
- `routing.default_module`
- `prompts.qa_system`

## 5. 当前示例

- 广告域：`domain/ad_engine/profile.json`
- 物流域：`domain/logistics/profile.json`

## 6. 验证命令

```powershell
python -m compileall src/workflow src/api
$env:WORKFLOW_DOMAIN_DIR='domain/ad_engine'; python -c "from workflow.engine import WorkflowService; s=WorkflowService(); print(s.domain_profile.profile_id)"
$env:WORKFLOW_DOMAIN_DIR='domain/logistics'; python -c "from workflow.engine import WorkflowService; s=WorkflowService(); print(s.domain_profile.profile_id)"
```
