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
