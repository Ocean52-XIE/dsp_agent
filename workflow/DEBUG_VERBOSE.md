# 系统调试开关说明

## 1. 配置项

- 环境变量：`WORKFLOW_DEBUG_VERBOSE`
- 默认值：`false`
- 取值：`true/false`（也支持 `1/0`, `yes/no`, `on/off`）

当该开关为 `false` 时，系统保持当前精简 `debug` 输出，不增加响应体体积。  
当该开关为 `true` 时，系统会在每条助手消息的 `debug` 中附加 `debug.verbose` 扩展调试信息。

## 2. 目前已接入的位置

- 工作流配置读取：`workflow/engine.py`
- 最终响应拼装：`workflow/nodes/finalize_response/__init__.py`
- 健康检查透出：`mock_api/main.py` 的 `GET /api/health`
- 启动脚本：`start_agent.ps1`
- 答案评测脚本：`workflow/eval/run_answer_eval.ps1`

## 3. 当前 verbose 输出内容（第一版）

开启后，`assistant_message.debug.verbose` 目前包含：

- `history_summary`
- `retrieval_queries`
- `retrieval_plan`
- `retrieval_grades`（wiki/code/case）
- `retrieval_profiles`（wiki/code/case/fusion）
- `retrieval_hit_counts`
- `hit_preview`（每个来源前几条命中摘要）
- `analysis_preview`
- `graph_path`
- `node_trace`

说明：以上为第一版字段，后续可以继续细化（例如按场景分层、按字段白名单开关等）。

## 3.1 调试开关对返回结构的影响

- `WORKFLOW_DEBUG_VERBOSE=false`：
  - 接口响应中不返回 `assistant_message.debug` 字段；
  - 前端应隐藏“调试信息”面板。
- `WORKFLOW_DEBUG_VERBOSE=true`：
  - 接口响应返回 `assistant_message.debug`；
  - 其中 `debug.verbose` 提供扩展排障信息。

## 3.2 QA 输出格式（knowledge_qa）

知识问答链路统一按三段式输出：

1. `系统判定结论`（仅调试开关开启时输出）
2. `问题结论`
3. `回答依据`

说明：

- 不再在系统输出中追加“代码锚点补充/入口函数补充”等附加段；
- `回答依据`由检索证据汇总生成，用于支撑“问题结论”。

## 4. 使用示例

### 服务启动脚本中开启

在 `start_agent.ps1` 里设置：

```powershell
$WORKFLOW_DEBUG_VERBOSE = "true"
```

### 评测脚本中开启

在 `workflow/eval/run_answer_eval.ps1` 里设置：

```powershell
$WORKFLOW_DEBUG_VERBOSE = "true"
```
