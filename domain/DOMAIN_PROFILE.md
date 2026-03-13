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

## 4. profile 字段清单

- `schema_version`
- `profile_id`
- `display_name`
- `language`
- `sources`：`wiki/code/cases`
- `routing`
- `modules`
- `domain_gate`
- `query_rewrite`
- `retrieval`
- `answering`
- `code_generation`
- `prompts`
- `ui`
- `eval`

## 5. 当前示例

- 广告域：`domain/ad_engine/profile.json`
- 物流域：`domain/logistics/profile.json`

## 6. 验证命令

```powershell
python -m compileall src/workflow src/api
$env:WORKFLOW_DOMAIN_DIR='domain/ad_engine'; python -c "from workflow.engine import WorkflowService; s=WorkflowService(); print(s.domain_profile.profile_id)"
$env:WORKFLOW_DOMAIN_DIR='domain/logistics'; python -c "from workflow.engine import WorkflowService; s=WorkflowService(); print(s.domain_profile.profile_id)"
```
