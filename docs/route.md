# 路由能力优化设计方案

## Context

### 背景
当前路由实现（`intent_routing`节点）仅通过规则判断进行路由，当规则判断置信度不高时缺乏兜底机制。同时，现有路由结果缺少"其他查询"分支，导致某些领域相关但不适合知识问答/问题分析的查询无法被有效处理。

### 目标
1. **规则优先路由**：先通过规则进行路由判断
2. **LLM兜底决策**：规则置信度不高时调用LLM决策路由
3. **新增路由分支**：支持`default_query`分支，由AgentLoop处理通用查询
4. **路由结果类型**：领域无关（out_of_scope）、知识问答（knowledge_qa）、问题分析（issue_analysis）、默认查询（default_query）

---

## 设计方案

### 1. 置信度评估机制

#### 1.1 置信度计算公式

```python
def _compute_routing_confidence(
    relevance: float,
    domain_hits: int,
    code_hint: bool,
    intent_match_type: str,  # "explicit" | "heuristic" | "fallback"
    has_context: bool,
) -> float:
    """
    计算规则路由的置信度

    Args:
        relevance: 领域相关性分数 (0.0 ~ 1.0)
        domain_hits: 领域词命中数量
        code_hint: 是否有代码线索
        intent_match_type: 意图匹配类型
            - "explicit": 明确命中意图词（如 "报错"、"异常"）
            - "heuristic": 启发式推断
            - "fallback": 兜底到 knowledge_qa
        has_context: 是否有上下文（活跃模块或历史记忆）

    Returns:
        置信度分数 (0.0 ~ 1.0)
    """
    base_confidence = 0.5

    # 领域词命中加成 (每个命中 +0.1，上限 0.3)
    domain_bonus = min(0.3, domain_hits * 0.1)

    # 代码线索加成
    code_bonus = 0.15 if code_hint else 0.0

    # 意图匹配强度加成
    intent_bonus = {
        "explicit": 0.25,      # 明确命中意图词
        "heuristic": 0.10,     # 启发式推断
        "fallback": 0.0,       # 兜底
    }.get(intent_match_type, 0.0)

    # 上下文加成（有活跃模块或历史记忆）
    context_bonus = 0.1 if has_context else 0.0

    confidence = min(1.0, base_confidence + domain_bonus + code_bonus + intent_bonus + context_bonus)
    return confidence
```

#### 1.2 意图匹配类型判定

```python
def _determine_intent_match_type(
    user_query: str,
    route: str,
    intent_terms: dict[str, tuple[str, ...]],
) -> str:
    """
    判定意图匹配类型

    Returns:
        "explicit": 明确命中问题分析信号（报错、异常等）
        "heuristic": 启发式推断（排障词、代码生成词）
        "fallback": 兜底到 knowledge_qa
    """
    normalized = _normalize(user_query)

    # 检查问题分析信号
    if _ISSUE_SIGNAL_RE.search(normalized):
        return "explicit"

    # 检查排障词
    troubleshoot_terms = intent_terms.get("troubleshoot", ())
    if _contains_any(normalized, troubleshoot_terms):
        return "explicit"

    # 检查代码生成请求
    if _looks_like_code_generation_request(user_query):
        return "explicit"

    # 如果路由到 knowledge_qa 且没有明确信号，则为 fallback
    if route == "knowledge_qa":
        return "fallback"

    return "heuristic"
```

#### 1.3 LLM兜底触发条件

```python
# profile.json 配置项
{
    "routing": {
        "llm_fallback_enabled": true,
        "llm_fallback_confidence_threshold": 0.65,    # 置信度阈值
        "llm_fallback_relevance_range": [0.5, 0.75]   # 仅在此相关性范围内触发
    }
}

def _should_call_llm_routing(
    confidence: float,
    relevance: float,
    config: RoutingConfig,
) -> bool:
    """
    判断是否需要 LLM 兜底

    触发条件：
    1. confidence < 0.65 (置信度不高)
    2. 0.5 <= relevance <= 0.75 (领域相关性在中等范围)
       - relevance < 0.5: 直接走 out_of_scope，无需 LLM
       - relevance > 0.75: 高相关性，规则可信
    """
    if not config.llm_fallback_enabled:
        return False

    min_rel, max_rel = config.llm_fallback_relevance_range
    return (
        confidence < config.llm_fallback_confidence_threshold
        and min_rel <= relevance <= max_rel
    )
```

---

### 2. LLM路由决策

#### 2.1 提示词设计

**文件位置**：`domain/ad_engine/prompts/routing_llm.md`

```markdown
# 意图分类助手

你是一个专业的意图分类助手。你的任务是判断用户问题的意图类型。

## 领域背景

你是广告引擎领域的助手，领域范围包括：
- 广告投放策略（出价、 pacing、预算分配）
- 召回与排序（候选召回、精排、重排）
- 两率预估（CTR、CVR 预测）
- 流量治理与策略调控
- 效果数据分析与问题排查

## 可选意图类型

1. **knowledge_qa**: 知识问答
   - 用户询问领域知识、概念解释、原理说明
   - 例如："什么是 CTR 预估？"、"出价策略有哪些？"、"召回链路是怎样的？"

2. **issue_analysis**: 问题分析
   - 用户遇到具体问题需要排查
   - 包含错误信息、异常现象、性能问题
   - 例如："CTR 预估偏低怎么排查？"、"为什么出价胜率下降？"、"出现 500 错误"

3. **default_query**: 其他查询
   - 与领域相关但不是标准的知识问答或问题分析
   - 需要灵活处理的通用查询
   - 例如："帮我分析一下最近的投放数据"、"对比一下两种出价策略的优劣"

4. **out_of_scope**: 领域无关
   - 与广告引擎领域完全无关的问题
   - 例如："今天天气怎么样？"、"帮我写一首诗"

## 判断原则

1. 优先考虑是否包含明确的错误/异常信号（报错、异常、失败等）
2. 如果是概念性询问，归类为 knowledge_qa
3. 如果需要灵活处理或综合分析，归类为 default_query
4. 明显与领域无关的，归类为 out_of_scope

## 输出格式

请直接返回 JSON 格式（不要包含 markdown 代码块）：
{
    "intent": "knowledge_qa | issue_analysis | default_query | out_of_scope",
    "reason": "简短的判断理由"
}
```

#### 2.2 LLM路由模块设计

**文件位置**：`src/workflow/nodes/routing_context/llm_router.py`

```python
# -*- coding: utf-8 -*-
"""LLM 路由决策模块"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LLMRoutingResult:
    """LLM 路由结果"""
    intent: str  # "knowledge_qa" | "issue_analysis" | "default_query" | "out_of_scope"
    reason: str
    latency_ms: int
    raw_response: str


class LLMRouter:
    """LLM 路由决策器"""

    VALID_INTENTS = ("knowledge_qa", "issue_analysis", "default_query", "out_of_scope")

    def __init__(
        self,
        llm_client: Any,
        prompt_template: str,
        timeout_seconds: int = 5,
        max_retries: int = 1,
    ) -> None:
        self._llm_client = llm_client
        self._prompt_template = prompt_template
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries

    def route(self, user_query: str, domain_context: str = "") -> LLMRoutingResult:
        """
        执行 LLM 路由决策

        Args:
            user_query: 用户查询
            domain_context: 领域上下文（模块名、相关模块等）

        Returns:
            LLMRoutingResult
        """
        start_time = time.time()
        messages = self._build_messages(user_query, domain_context)

        last_error = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._llm_client.invoke(messages)
                raw_response = response.get("content", "")

                result = self._parse_response(raw_response)
                result.latency_ms = int((time.time() - start_time) * 1000)
                result.raw_response = raw_response

                logger.info(f"[LLMRouter] 路由决策完成: intent={result.intent}, latency={result.latency_ms}ms")
                return result

            except Exception as e:
                last_error = e
                logger.warning(f"[LLMRouter] 尝试 {attempt + 1} 失败: {e}")

        # 所有重试失败，返回默认值
        latency_ms = int((time.time() - start_time) * 1000)
        logger.error(f"[LLMRouter] 所有重试失败: {last_error}")
        return LLMRoutingResult(
            intent="knowledge_qa",  # 默认路由
            reason=f"llm_error: {last_error}",
            latency_ms=latency_ms,
            raw_response="",
        )

    def _build_messages(self, user_query: str, domain_context: str) -> list[dict[str, str]]:
        """构建消息"""
        user_content = f"用户问题：{user_query}"
        if domain_context:
            user_content += f"\n\n上下文信息：{domain_context}"

        return [
            {"role": "system", "content": self._prompt_template},
            {"role": "user", "content": user_content},
        ]

    def _parse_response(self, response: str) -> LLMRoutingResult:
        """解析 LLM 响应"""
        try:
            # 尝试提取 JSON
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                intent = data.get("intent", "knowledge_qa")

                # 验证 intent 有效性
                if intent not in self.VALID_INTENTS:
                    intent = "knowledge_qa"

                return LLMRoutingResult(
                    intent=intent,
                    reason=data.get("reason", ""),
                    latency_ms=0,
                    raw_response=response,
                )
        except Exception as e:
            logger.warning(f"[LLMRouter] 解析响应失败: {e}")

        # 解析失败，返回默认值
        return LLMRoutingResult(
            intent="knowledge_qa",
            reason="parse_failed",
            latency_ms=0,
            raw_response=response,
        )
```

#### 2.3 调用流程

```
intent_routing 节点
    │
    ├── 1. 规则路由计算
    │     ├── _compute_domain_relevance() -> (relevance, in_scope, reason)
    │     ├── _compute_routing_confidence() -> confidence
    │     └── _classify_in_scope_intent() -> rule_intent
    │
    ├── 2. LLM 兜底判断
    │     │
    │     ├── if not _should_call_llm_routing():
    │     │       返回 rule_intent
    │     │
    │     └── else:
    │           调用 LLMRouter.route()
    │           返回 llm_result.intent
    │
    └── 3. 返回结果
          ├── route: final_intent
          ├── routing_confidence: confidence
          ├── routing_method: "rule" | "llm"
          └── routing_intent_reason: reason
```

---

### 3. 新增路由分支：default_query

#### 3.1 节点实现

**文件位置**：`src/workflow/nodes/agent_loop/default_query_node.py`

```python
# -*- coding: utf-8 -*-
"""Other Query 节点

处理领域相关但无法明确分类的知识问答/问题分析类查询。
使用 AgentLoop 的完整能力（工具调用、多步推理）处理用户请求。
"""
from __future__ import annotations

from typing import Any

from workflow.nodes.agent_loop.base import BaseAgentLoopNode
from workflow.nodes.agent_loop.config import AgentLoopNodeConfig


class OtherQueryNode(BaseAgentLoopNode):
    """其他查询节点

    处理领域相关但无法明确分类为知识问答或问题分析的查询。
    特点：
    - 不强制要求检索证据（require_evidence=False）
    - 允许更多迭代次数（max_iterations=5）
    - 启用技能工具（enable_skill_tool=True）
    """

    def __init__(self) -> None:
        config = AgentLoopNodeConfig(
            node_name="default_query",
            response_kind="default_query",
            enable_skill_tool=True,
            max_iterations=5,
            timeout_seconds=90,
            require_evidence=False,
            enable_fallback=True,
        )
        super().__init__(config=config)

    def _get_system_prompt(self, state: dict[str, Any]) -> str:
        """获取系统提示词"""
        return """你是广告引擎领域的智能助手。

用户的问题与广告引擎领域相关，但不属于标准的知识问答或问题分析类型。

请根据问题的性质，灵活选择合适的处理方式：
1. 如果需要查询知识，可以使用检索工具
2. 如果需要执行操作，可以使用相应技能工具
3. 如果信息足够，直接回答即可

回答要求：
- 简洁专业，重点突出
- 如有数据支撑，请明确说明
- 如需假设，请明确标注"""

    def _get_user_prompt(
        self,
        state: dict[str, Any],
        user_query: str,
        module_name: str,
        module_hint: str,
        related_modules: list[dict[str, Any]],
        evidence_hits: list[dict[str, Any]],
    ) -> str:
        """获取用户提示词"""
        prompt = f"用户问题：{user_query}"

        if module_name and module_hint:
            prompt += f"\n\n相关模块：{module_name}（{module_hint}）"

        if related_modules:
            related_names = [m.get("name", "") for m in related_modules[:3]]
            prompt += f"\n相关上下文：{', '.join(related_names)}"

        prompt += "\n\n请根据问题的性质，提供合适的回答。"
        return prompt

    def _build_fallback(
        self,
        state: dict[str, Any],
        module_name: str,
        module_hint: str,
        related_modules: list[dict[str, Any]],
        evidence_hits: list[dict[str, Any]],
    ) -> str:
        """构建 Fallback 响应"""
        return "抱歉，我暂时无法处理这个请求，请尝试换一种方式描述您的问题，或者提供更多上下文信息。"
```

#### 3.2 与 AgentLoop 的集成

`OtherQueryNode` 继承 `BaseAgentLoopNode`，自动获得以下能力：

1. **Agent 循环**：通过 `agent.core.loop.AgentLoop` 实现 LLM + Tool 调用循环
2. **技能工具**：通过 `enable_skill_tool=True` 启用已注册的技能
3. **Fallback 机制**：LLM 调用失败时的兜底响应

---

### 4. 工作流图调整

#### 4.1 engine.py 修改

```python
# 新增节点
from workflow.nodes.agent_loop.default_query_node import OtherQueryNode

class WorkflowEngine:
    def __init__(self, ...):
        # ... 现有初始化 ...
        self._default_query_node = OtherQueryNode()

    def _build_graph(self, checkpointer: Any = None) -> Any:
        graph = StateGraph(WorkflowState)

        # 现有节点
        graph.add_node("load_context", self._load_context)
        graph.add_node("intent_routing", self._intent_routing)
        graph.add_node("knowledge_answer", self._knowledge_answer)
        graph.add_node("issue_analysis", self._issue_analysis)
        graph.add_node("out_of_scope_response", self._out_of_scope_response)
        graph.add_node("finalize_response", self._finalize_response)

        # 代码生成节点
        graph.add_node("load_code_context", self._load_code_context)
        graph.add_node("retrieve_code_context", self._retrieve_code_context)
        graph.add_node("code_generation", self._code_generation)

        # === 新增：default_query 节点 ===
        graph.add_node("default_query", self._default_query)

        # 主流程边
        graph.add_edge(START, "load_context")
        graph.add_edge("load_context", "intent_routing")

        # 意图路由（新增 default_query 分支）
        graph.add_conditional_edges(
            "intent_routing",
            self._route_by_intent,
            {
                "knowledge_qa": "knowledge_answer",
                "issue_analysis": "issue_analysis",
                "code_generation": "load_code_context",
                "out_of_scope": "out_of_scope_response",
                "default_query": "default_query",  # 新增
            },
        )

        # === 新增：default_query 到 finalize_response ===
        graph.add_edge("default_query", "finalize_response")

        # 其他边保持不变...

        return graph.compile(checkpointer=effective_checkpointer)

    def _default_query(self, state: dict[str, Any]) -> dict[str, Any]:
        """default_query 节点执行"""
        return self._default_query_node.run(self, state)
```

#### 4.2 状态字段扩展

**文件位置**：`src/workflow/state.py`

```python
class WorkflowState(TypedDict, total=False):
    # ... 现有字段 ...

    # === 新增：路由相关信息 ===
    routing_confidence: float           # 规则路由置信度 (0.0 ~ 1.0)
    routing_method: str                 # "rule" | "llm"
    llm_routing_result: dict[str, Any] | None  # LLM 路由原始结果（调试用）
    routing_intent_reason: str          # 路由判断理由
```

---

### 5. 配置扩展

#### 5.1 profile.json 新增配置项

```json
{
    "routing": {
        "default_module": "ad-serving-orchestrator",
        "module_infer_strategy": "keyword_then_symbol",
        "prefer_symbol_match": true,

        "llm_fallback_enabled": true,
        "llm_fallback_confidence_threshold": 0.65,
        "llm_fallback_relevance_range": [0.5, 0.75],
        "llm_routing_timeout_seconds": 5,
        "llm_routing_max_retries": 1
    },

    "domain_gate": {
        "enabled": false,
        "threshold": 0.5,
        "weak_in_scope_min_score": 0.62,
        "weak_code_hint_min_score": 0.58,
        "history_memory_bonus": 0.18,
        "offtopic_penalty": 0.45,
        "short_query_penalty": 0.25,
        "short_query_max_len": 4,
        "high_confidence_threshold": 0.75
    }
}
```

#### 5.2 DomainProfile 类扩展

**文件位置**：`src/domain_profile/profile.py`

```python
@dataclass(frozen=True)
class RoutingProfile:
    """路由配置"""
    default_module: str = ""
    module_infer_strategy: str = "keyword_then_symbol"
    prefer_symbol_match: bool = True

    # LLM 兜底配置
    llm_fallback_enabled: bool = True
    llm_fallback_confidence_threshold: float = 0.65
    llm_fallback_relevance_range: tuple[float, float] = (0.5, 0.75)
    llm_routing_timeout_seconds: int = 5
    llm_routing_max_retries: int = 1

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RoutingProfile":
        range_raw = payload.get("llm_fallback_relevance_range", [0.5, 0.75])
        return cls(
            default_module=_as_str(payload.get("default_module")),
            module_infer_strategy=_as_str(payload.get("module_infer_strategy"), "keyword_then_symbol"),
            prefer_symbol_match=bool(payload.get("prefer_symbol_match", True)),
            llm_fallback_enabled=bool(payload.get("llm_fallback_enabled", True)),
            llm_fallback_confidence_threshold=_as_float(payload.get("llm_fallback_confidence_threshold"), 0.65),
            llm_fallback_relevance_range=(
                float(range_raw[0]), float(range_raw[1])
            ) if isinstance(range_raw, (list, tuple)) and len(range_raw) >= 2 else (0.5, 0.75),
            llm_routing_timeout_seconds=_as_int(payload.get("llm_routing_timeout_seconds"), 5),
            llm_routing_max_retries=_as_int(payload.get("llm_routing_max_retries"), 1),
        )
```

---

## 执行流程图

```
                    ┌─────────────────┐
                    │  load_context   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ intent_routing  │
                    │  ┌───────────┐  │
                    │  │ 规则路由  │  │
                    │  │ 置信度    │  │
                    │  │ 计算      │  │
                    │  └─────┬─────┘  │
                    │        │        │
                    │  ┌─────▼─────┐  │
                    │  │置信度<0.65│  │
                    │  │且相关性在 │──┼──► LLM决策
                    │  │[0.5,0.75] │  │
                    │  └─────┬─────┘  │
                    └────────┼────────┘
                             │
         ┌───────────┬───────┼───────┬───────────┐
         │           │       │       │           │
    ┌────▼────┐ ┌────▼───┐ ┌─▼──┐ ┌──▼───┐ ┌────▼────┐
    │knowledge│ │ issue  │ │code│ │ out  │ │ other   │
    │   qa    │ │analysis│ │gen │ │scope │ │ query   │
    │ (子图)  │ │ (子图) │ │分支│ │      │ │(Agent)  │
    └────┬────┘ └───┬────┘ └─┬──┘ └──┬───┘ └────┬────┘
         │          │        │       │          │
         └──────────┴────────┴───────┴──────────┘
                             │
                    ┌────────▼────────┐
                    │finalize_response│
                    └─────────────────┘
```

---

## 文件修改清单

### 需要修改的文件

| 文件 | 修改内容 |
|------|---------|
| `src/workflow/nodes/routing_context/intent_routing/__init__.py` | 添加置信度计算、LLM路由决策逻辑 |
| `src/workflow/engine.py` | 添加 default_query 节点和路由分支 |
| `src/workflow/state.py` | 添加路由相关状态字段 |
| `src/domain_profile/profile.py` | 添加 RoutingProfile 配置类 |
| `domain/ad_engine/profile.json` | 添加 routing LLM兜底配置 |

### 需要新增的文件

| 文件 | 说明 |
|------|------|
| `src/workflow/nodes/agent_loop/default_query_node.py` | default_query 节点实现 |
| `src/workflow/nodes/routing_context/llm_router.py` | LLM路由决策模块 |
| `domain/ad_engine/prompts/routing_llm.md` | LLM路由提示词 |

---

## 实施步骤

1. **扩展配置**：修改 `profile.py` 和 `profile.json`，添加 RoutingProfile
2. **扩展状态**：修改 `state.py`，添加路由相关字段
3. **实现LLM路由**：创建 `llm_router.py` 模块
4. **创建提示词**：添加 `routing_llm.md`
5. **修改路由节点**：更新 `intent_routing/__init__.py`，集成置信度和LLM兜底
6. **创建 default_query 节点**：实现 `default_query_node.py`
7. **更新工作流引擎**：修改 `engine.py`，添加新节点和路由分支
8. **编写测试**：添加单元测试和集成测试
9. **验证**：运行端到端测试

---

## 验证方案

### 单元测试

```python
# tests/unit/test_routing_confidence.py

def test_confidence_high_with_explicit_intent():
    """明确意图信号时，置信度应该较高"""
    confidence = _compute_routing_confidence(
        relevance=0.7,
        domain_hits=3,
        code_hint=False,
        intent_match_type="explicit",
        has_context=True,
    )
    assert confidence >= 0.85

def test_confidence_low_with_fallback():
    """fallback 路由时，置信度应该较低"""
    confidence = _compute_routing_confidence(
        relevance=0.6,
        domain_hits=1,
        code_hint=False,
        intent_match_type="fallback",
        has_context=False,
    )
    assert confidence < 0.65

# tests/unit/test_llm_router.py

def test_parse_valid_json_response():
    """测试解析有效的 JSON 响应"""
    router = LLMRouter(...)
    result = router._parse_response('{"intent": "knowledge_qa", "reason": "test"}')
    assert result.intent == "knowledge_qa"
    assert result.reason == "test"

def test_parse_invalid_intent():
    """测试无效 intent 的处理"""
    router = LLMRouter(...)
    result = router._parse_response('{"intent": "invalid", "reason": "test"}')
    assert result.intent == "knowledge_qa"  # 默认值
```

### 集成测试场景

| 场景 | 输入 | 期望路由 | 期望方法 |
|------|------|---------|---------|
| 高置信度问题分析 | "为什么出价胜率下降，报错了" | issue_analysis | rule |
| 高置信度知识问答 | "CTR 预估的原理是什么" | knowledge_qa | rule |
| 低置信度触发LLM | "帮我看看最近的数据" | default_query | llm |
| 明确领域无关 | "今天天气怎么样" | out_of_scope | rule |

### 端到端测试

```bash
# 启动服务
python -m uvicorn src.api.main:app

# 测试 1: 知识问答
curl -X POST /api/messages -d '{"content": "出价策略有哪些？"}'
# 期望: route=knowledge_qa

# 测试 2: 问题分析
curl -X POST /api/messages -d '{"content": "CTR 预估报错了，怎么排查？"}'
# 期望: route=issue_analysis

# 测试 3: 其他查询（触发 LLM 路由）
curl -X POST /api/messages -d '{"content": "帮我分析一下最近的投放数据"}'
# 期望: route=default_query, routing_method=llm

# 测试 4: 领域无关
curl -X POST /api/messages -d '{"content": "今天天气怎么样？"}'
# 期望: route=out_of_scope
```

---

## 关键文件路径

| 组件 | 文件路径 |
|------|---------|
| 核心路由逻辑 | `src/workflow/nodes/routing_context/intent_routing/__init__.py` |
| 工作流引擎 | `src/workflow/engine.py` |
| 状态定义 | `src/workflow/state.py` |
| AgentLoop节点基类 | `src/workflow/nodes/agent_loop/base.py` |
| AgentLoop配置 | `src/workflow/nodes/agent_loop/config.py` |
| 领域配置管理 | `src/domain_profile/profile.py` |
| 领域配置文件 | `domain/ad_engine/profile.json` |
| 提示词目录 | `domain/ad_engine/prompts/` |
