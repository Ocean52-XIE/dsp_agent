# AGENTS.md

本文件用于约束本仓库后续的代码生成行为，目标是让改动可落位、可维护、可评估。

## 1. 顶层目录职责

| 目录/文件 | 职责说明 | 代码生成约束 |
| --- | --- | --- |
| `src/` | 生产代码主目录（API、工作流、Web、基础设施） | 新功能默认放在这里，不要把业务代码放到 `docs/`、`tools/`、`test/` |
| `domain/` | 领域数据与配置（profile、wiki、代码语料、评测数据） | 领域知识/语料改动放这里，不要硬编码在 `src/workflow` |
| `docs/` | 设计文档与子系统说明 | 设计变更后同步文档；文档不作为运行时代码依赖 |
| `tools/` | 辅助脚本（提取、诊断、运维） | 工具脚本不能反向耦合主流程逻辑 |
| `test/` | 手工测试样例与临时验证文件 | 正式自动化测试优先放 `src/workflow/eval/` 或新增规范测试目录 |
| `logs/` | 运行日志输出目录 | 严禁将日志作为业务输入；不在代码中写死日志文件名之外的路径策略 |
| `start_agent.ps1` | 本地启动与环境变量注入脚本 | 只做启动配置，不承载业务逻辑 |
| `requirements.txt` | Python 依赖清单 | 新增依赖必须同步更新，并说明用途 |

## 2. `src/` 目录职责

| 目录/文件 | 职责说明 | 代码生成约束 |
| --- | --- | --- |
| `src/api/main.py` | FastAPI 接口层与请求编排入口 | 只做协议转换、参数校验、调用 `WorkflowService`；不要在 API 层写复杂检索/推理逻辑 |
| `src/workflow/engine.py` | LangGraph 编排与节点连接 | 负责图拓扑、节点注册与运行时日志；节点业务逻辑放 `nodes/` |
| `src/workflow/nodes/` | 各节点具体实现（路由、检索、分析、生成、收口） | 每个节点保持 `run(service, state) -> dict` 增量更新契约 |
| `src/workflow/common/` | 通用工具（state 辅助、日志、证据归一化等） | 可复用逻辑下沉这里，避免在节点内复制粘贴 |
| `src/workflow/llm/` | LLM 客户端与提示词工具 | 模型调用统一走该层，不在节点中直接 new 模型客户端 |
| `src/workflow/retrievers/` | 检索器共性算法与融合工具 | 检索算法改动优先集中在这里，保持节点层薄封装 |
| `src/workflow/session/` | 会话存储适配（PostgreSQL/回退） | 存储策略变更在此实现，不把 DB 细节散落到 API/节点 |
| `src/workflow/observability/` | 可观测性落库、聚合与告警 | 指标与告警逻辑集中管理，避免业务节点直接写 SQL |
| `src/workflow/eval/` | 离线评测脚本、模板配置、结果产物 | 评测脚本与生产逻辑解耦；结果文件可覆盖，不作为源码真相 |
| `src/bootstrap/` | 启动期基础设施引导（如 PG 初始化） | 只处理初始化，不承载请求时逻辑 |
| `src/web/` | 前端静态页面资源 | 前端改动不影响后端协议语义 |

## 3. `domain/` 目录职责

| 路径模式 | 职责说明 | 代码生成约束 |
| --- | --- | --- |
| `domain/<domain_id>/profile.json` | 领域配置中心（路由、检索、提示词、评测路径） | 优先改配置，不在代码中硬编码领域词表/阈值 |
| `domain/<domain_id>/wiki/` | 领域知识文档语料 | 事实性知识更新放这里 |
| `domain/<domain_id>/codes/` | 代码检索语料 | 仅用于检索与定位，避免混入无关工程文件 |
| `domain/<domain_id>/eval/datasets/` | 评测样本数据 | 新能力上线需补充对应评测样本 |
| `domain/<domain_id>/prompts/` | 领域提示词模板 | 提示词更新优先走文件，不硬编码长提示词 |

## 4. 代码生成硬约束（必须遵守）

1. 分层约束：API 层不写业务推理，节点层不直接处理 HTTP，存储层不反向依赖 API。
2. 节点契约：节点函数保持 `run(service, state) -> dict`，返回状态增量；不要直接破坏 state 结构。
3. 图结构约束：新增/删除节点时，必须同步更新 `src/workflow/engine.py` 的图连接。
4. 状态字段约束：新增 workflow 字段时，需同步更新 `WorkflowState`（`engine.py`）与最终收口逻辑（`finalize_response`）。
5. 配置优先：阈值、TopK、开关优先放 `profile.json` 或环境变量，不写魔法数字常量到业务路径。
6. 领域隔离：与具体业务域绑定的词典、路由词、提示词放 `domain/<domain_id>/`，不要放到通用模块。
7. 检索与融合：检索命中结构需保持可追踪（source/path/section/score 等）；融合后仍需可解释。
8. 可观测性：关键链路改动需要补齐 runtime log 与 observability 字段，至少能定位 route、latency、citations。
9. 向后兼容：`/api/messages`、`/api/references/{trace_id}`、`/api/messages/{message_id}/feedback` 响应结构默认保持兼容。
10. 安全约束：禁止在源码中新增明文密钥、DSN、Token，评测和启动等脚本除外；统一使用环境变量读取。
11. 依赖约束：新增第三方包需更新 `requirements.txt`，并确保本地可启动。
12. 编码与风格：Python 代码保持类型标注与清晰函数边界，避免单函数承载多阶段复杂逻辑。
13. 文件编码：所有文本文件使用 utf-8 格式的编码，做文件处理和字符串替换等场景时避免乱码。
14. 代码注释：生成的代码需要有详细的中文代码注释。
15. 日志输出：关键的模块需要有日志输出。

## 5. 变更落位规则（新增功能时）

| 变更类型 | 首选落位 |
| --- | --- |
| 新增 API | `src/api/main.py`（或拆分新 API 模块并在此挂载） |
| 新增路由策略 | `src/workflow/nodes/routing_context/` |
| 新增检索策略 | `src/workflow/nodes/retrieval_flow/` + `src/workflow/retrievers/` |
| 新增分析能力 | `src/workflow/nodes/analysis/` |
| 新增代码生成能力 | `src/workflow/nodes/code_generation_flow/` |
| 新增存储后端 | `src/workflow/session/` 或 `src/workflow/observability/` |
| 新增领域 | `domain/<new_domain>/` 全套目录 |
| 新增评测 | `src/workflow/eval/` + `domain/<domain_id>/eval/datasets/` |

## 6. 提交前自检清单

1. 改动是否放在正确目录，是否破坏分层边界。
2. 是否需要同步更新 `engine.py`、`NODES.md`、`README.md` 或相关设计文档。
3. 是否补充了最小评测或回归脚本验证（`src/workflow/eval/`）。
4. 是否引入配置项且已提供默认值/模板。
5. 是否包含潜在敏感信息（API Key、密码、内网地址）并已移除。
