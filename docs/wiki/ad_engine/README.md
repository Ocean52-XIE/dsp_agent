# 广告引擎 Wiki

本目录提供广告在线投放场景的基础业务知识文档，覆盖以下核心链路：

- 在线召回
- 两率预估（pCTR/pCVR）
- 出价策略
- 精排策略
- 联调与排障

这些文档会被 `workflow/nodes/retrieve_wiki/wiki_retriever.py` 作为检索语料直接加载，用于驱动 `workflow/engine.py` 的 `retrieve_wiki` 节点。

## 文档清单

- `00-广告引擎总体架构.md`
- `01-在线召回.md`
- `02-两率预估.md`
- `03-出价策略.md`
- `04-精排策略.md`
- `05-联调排障手册.md`
