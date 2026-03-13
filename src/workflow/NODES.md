# Workflow 节点说明

本文档说明 `workflow/nodes/` 目录下各节点的职责，以及当前版本的多源检索与融合行为。

## 1. 入口与路由类节点

- `entry_router`：判断本轮请求走普通消息链路，还是走“确认后代码生成”的恢复链路。
- `load_context`：从会话历史恢复活动主题、活动阶段、最近分析结果等上下文。
- `domain_gate`：领域门控，识别是否属于广告引擎业务域。
- `intent_classifier`：识别本轮意图（如 `knowledge_qa`、`issue_analysis`）。
- `conversation_transition`：结合意图与上下文判断本轮执行路径（检索问答 / 排障分析 / 代码生成）。
- `query_rewriter`：生成检索改写语句，并输出 `retrieval_plan`（多源路由策略）。
  - 支持同义词扩展、缩写归一（CTR/CVR/pCTR/pCVR 等）和指标词典补充。

## 2. 检索类节点

- `retrieve_wiki`：执行 Wiki 检索，输出 `wiki_hits`、`wiki_retrieval_grade`、`wiki_retrieval_profile`。
  - 低置信时自动重试（扩 TopK + 扩展查询语句）。
- `retrieve_code`：执行代码检索，输出 `code_hits`、`code_retrieval_grade`、`code_retrieval_profile`。
  - 低置信时自动重试（扩 TopK + 定位型查询增强）。
- `retrieve_cases`：案例检索占位节点；当前默认不启用，仅保留统一扩展接口。
- `merge_evidence`：多源证据融合节点。根据 `retrieval_plan` 执行：
  - 源权重加权（`source_weights`）
  - 源内配额控制（`max_per_source`）
  - 最终 TopK 裁剪（`final_top_k`）
  - 输出融合调试画像（`evidence_fusion_profile`）

### Wiki Hybrid 检索说明

- 代码位置：`workflow/nodes/retrieve_wiki/wiki_retriever.py`
- 召回分三路：
  - BM25 分（词项相关性）
  - 向量分（TF-IDF 余弦）
  - 规则分（短语命中、标题/章节命中、结构特征）
- 融合权重支持从 `WORKFLOW_WIKI_HYBRID_WEIGHTS_PATH` 加载。

## 3. 分析与生成类节点

- `knowledge_answer`：知识问答节点，LLM 优先，失败自动降级到规则兜底。
- `issue_localizer`：问题定位，给出候选模块和初步判断。
- `root_cause_analysis`：根因分析，补充风险与证据链。
- `fix_plan`：形成修复建议，并进入是否生成代码的确认阶段。
- `decline_code_generation_response`：处理“暂不生成代码”的控制类响应。
- `out_of_scope_response`：处理领域外输入。
- `load_code_context`：从上游分析消息中装配代码生成所需上下文。
- `retrieve_code_context`：为代码生成补充代码上下文（当前可按需扩展）。
- `code_generation`：输出代码实现建议/补丁建议。
- `finalize_response`：收敛最终响应结构，返回前端可直接消费的消息体。

## 4. 当前多源检索链路

当前默认链路为：

`query_rewriter -> retrieve_wiki -> retrieve_cases -> retrieve_code -> merge_evidence`

其中关键状态字段如下：

- `retrieval_queries`：检索改写语句列表。
- `retrieval_plan`：多源路由与融合计划（策略、开关、TopK、权重、配额）。
- `wiki_hits` / `code_hits` / `case_hits`：各检索源的原始命中。
- `wiki_retrieval_grade` / `code_retrieval_grade`：检索质量分级。
- `wiki_retrieval_profile` / `code_retrieval_profile`：检索耗时、命中量、候选量等诊断信息。
- `citations`：融合后的最终证据列表。
- `evidence_fusion_profile`：融合过程画像，用于调参与回归分析。

## 5. 相关代码入口

- 工作流定义：`workflow/engine.py`
- Wiki 检索器：`workflow/nodes/retrieve_wiki/wiki_retriever.py`
- Wiki Hybrid 设计文档：`workflow/retrieve_wiki_hybrid_v1.md`
- 代码检索器：`workflow/nodes/retrieve_code/code_retriever.py`
- 融合策略：`workflow/nodes/merge_evidence/__init__.py`
