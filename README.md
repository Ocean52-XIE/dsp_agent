python -m uvicorn mock_api.main:app --reload


如果用户一开始输入的内容并判别为知识问答，随着问答的进行，后面的内容识别为问题分析和代码实现，是否会对workflow的设计造成影响

会有影响，但不是推翻式影响，属于“要把工作流从单轮路由升级成多轮任务状态机”。

按你现在这套设计，workflow 本身并不要求“一个会话只能是知识问答”或“只能是问题分析”。当前实现里，每一轮用户输入都会重新走一次 domain_gate -> intent_classifier，所以从知识问答切到问题分析，在结构上是成立的。相关代码在 engine.py；接口层每次发消息也都是重新调用 WORKFLOW.run_user_message(...)，见 main.py。

真正会受影响的是这几个点：

不能把“会话类型”设计成固定值
当前更合理的是“每轮路由”，不是“整会话路由”。也就是第一轮是 knowledge_qa，第二轮完全可以切到 issue_analysis。

需要补“任务上下文”而不只是“消息历史”
如果用户前面问的是业务规则，后面说“那这个报错怎么修”，系统要知道“这个”指的是什么。
现在的 history_summary 还比较轻，只够演示；真实设计里要增加：

active_topic
active_issue_context
active_qa_context
pending_action
last_analysis_result
代码实现不能只靠当前一句话判断
代码生成最好不要靠用户突然说一句“给我代码”就直接进 code_generation。
更稳妥的是：

先有一轮 issue_analysis
产出模块、根因、修复方案
再进入 confirm_code
用户确认后才进 code_generation
这一点你现在的图设计是对的，run_code_generation(...) 就是单独路径。

需要增加“话题切换/延续”判断
比如：

“订单结算规则是什么？” -> 知识问答
“那它报错 500 怎么排查？” -> 问题分析
“直接给我修改代码” -> 代码生成
这三句其实是同一主题的连续升级。
但如果用户突然问“库存模块的锁定流程是什么”，那就可能是新任务了。
所以建议加一个节点：

conversation_transition_node
它负责判断本轮是：

延续当前任务
从 QA 升级到分析
从分析升级到代码生成
切换到新主题
结论是：

不会破坏你现在的 workflow 方向，但会要求你把设计从“单轮分类工作流”升级成“多轮可转阶段工作流”。
最核心的设计原则应该变成：

intent 是每轮判断的
task_state 是跨轮保存的
code_generation 必须依赖前置分析结果
会话里允许 knowledge_qa -> issue_analysis -> code_generation 逐步升级
如果你要，我下一步可以直接把现有 engine.py 改造成这种“多轮阶段升级版”workflow。


请帮忙把工作流改造成上面提到的多轮阶段升级版，节点内部的实现可以继续保持mock，注意代码需要详细的中文注释，同时请更新设计文档