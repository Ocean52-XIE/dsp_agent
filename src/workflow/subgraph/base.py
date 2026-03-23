# -*- coding: utf-8 -*-
"""Subgraph 基础类和配置

提供 Subgraph 的基础抽象和配置类：
- SubgraphConfig: 子图配置基类
- BaseSubgraph: 子图基类，提供通用能力

设计原则：
1. 子图是独立的工作流单元，可以复用
2. 子图通过 SubgraphRegistry 注册和管理
3. 子图可以包含多个节点，形成完整流程
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ============================================================================
# Subgraph 配置基类
# ============================================================================

@dataclass
class SubgraphConfig:
    """Subgraph 配置基类

    所有子图配置都应该继承此类。

    Attributes:
        subgraph_id: 子图唯一标识
        display_name: 显示名称
        description: 描述
        enable_skill_tool: 是否启用 Skill Tool
        max_iterations: Agent 模式最大迭代次数
        require_evidence: 是否需要证据
        enable_fallback: 是否启用降级
        metadata: 额外元数据
    """
    subgraph_id: str = "base"
    display_name: str = "Base Subgraph"
    description: str = ""
    enable_skill_tool: bool = True
    max_iterations: int = 3
    require_evidence: bool = True
    enable_fallback: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """初始化后处理"""
        # 确保 metadata 是字典
        if self.metadata is None:
            self.metadata = {}


# ============================================================================
# Subgraph 基类
# ============================================================================

class BaseSubgraph(ABC):
    """Subgraph 基类

    所有子图都应该继承此类。子图是独立的工作流单元，可以：
    1. 包含多个节点
    2. 构建自己的 LangGraph 图
    3. 被主工作流或其他子图调用

    Usage:
        class KnowledgeQASubgraph(BaseSubgraph):
            def __init__(self, config: KnowledgeQAConfig):
                super().__init__(config=config)
                # 所有依赖从全局单例获取

            def build_graph(self):
                # 构建子图
                graph = StateGraph(SubgraphState)
                # 添加节点和边
                return graph.compile()
    """

    def __init__(
        self,
        config: SubgraphConfig,
    ) -> None:
        """初始化子图

        所有组件从全局单例获取，无需手动传递。

        Args:
            config: 子图配置
        """
        self.config = config
        self._extra_kwargs: dict[str, Any] = {}

        # 子图图实例（立即构建）
        self._graph: Any | None = self.build_graph()

        logger.info(
            f"[Subgraph] {self.config.subgraph_id} 初始化完成, "
            f"graph_built={self._graph is not None}"
        )

    @abstractmethod
    def build_graph(self) -> Any:
        """构建子图

        子类必须实现此方法，返回编译后的 LangGraph 图。

        Returns:
            编译后的 LangGraph 图
        """
        pass

    def get_graph(self) -> Any:
        """获取子图实例

        Returns:
            编译后的 LangGraph 图
        """
        return self._graph

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """执行子图（同步版本）

        Args:
            state: 工作流状态

        Returns:
            更新后的状态增量
        """
        graph = self.get_graph()

        # 构建子图输入
        subgraph_input = self._prepare_input(state)

        try:
            # 执行子图（同步调用）
            result = graph.invoke(subgraph_input)
            return self._process_output(result)

        except Exception as e:
            logger.warning(
                f"[Subgraph] {self.config.subgraph_id} 执行失败: {e}"
            )
            return self._build_error_result(e)

    def _prepare_input(self, state: dict[str, Any]) -> dict[str, Any]:
        """准备子图输入

        子类可以覆盖此方法自定义输入映射。

        Args:
            state: 主工作流状态

        Returns:
            子图输入状态
        """
        return dict(state)

    def _process_output(self, result: dict[str, Any]) -> dict[str, Any]:
        """处理子图输出

        子类可以覆盖此方法自定义输出映射。

        Args:
            result: 子图执行结果

        Returns:
            主工作流状态增量
        """
        return dict(result)

    def _build_error_result(self, error: Exception) -> dict[str, Any]:
        """构建错误结果

        Args:
            error: 异常

        Returns:
            错误状态增量
        """
        return {
            "status": "error",
            "response_kind": self.config.subgraph_id,
            "answer": f"子图执行失败: {error}",
            "analysis": {
                "error": str(error),
                "error_type": type(error).__name__,
            },
        }

    def _trace(
        self,
        state: dict[str, Any],
        node_name: str,
        summary: str,
    ) -> list[dict[str, str]]:
        """构建节点追踪

        由于 node_trace 字段使用了 merge_lists reducer，节点只需返回新增的条目，
        LangGraph 会自动合并。

        Args:
            state: 工作流状态（用于兼容旧接口，实际不再使用）
            node_name: 节点名称
            summary: 节点摘要

        Returns:
            只包含新增条目的列表
        """
        return [{"node": node_name, "summary": summary}]


# ============================================================================
# Reducer 函数（用于 LangGraph 状态合并）
# ============================================================================

def merge_lists(left: list | None, right: list | None) -> list:
    """合并两个列表的 reducer 函数

    用于 LangGraph 状态中列表字段的合并。
    当多个节点返回同一列表字段时，将值合并而非覆盖。

    Args:
        left: 左侧列表（先前的状态值）
        right: 右侧列表（当前节点返回的值）

    Returns:
        合并后的新列表

    Examples:
        >>> merge_lists([1, 2], [3, 4])
        [1, 2, 3, 4]
        >>> merge_lists(None, [1, 2])
        [1, 2]
        >>> merge_lists([1, 2], None)
        [1, 2]
    """
    if left is None:
        return right or []
    if right is None:
        return left
    return list(left) + list(right)


def merge_dicts(left: dict | None, right: dict | None) -> dict:
    """合并两个字典的 reducer 函数（右侧优先）

    用于 LangGraph 状态中字典字段的合并。
    当多个节点返回同一字典字段时，合并字典，相同 key 时右侧覆盖左侧。

    Args:
        left: 左侧字典（先前的状态值）
        right: 右侧字典（当前节点返回的值）

    Returns:
        合并后的新字典

    Examples:
        >>> merge_dicts({"a": 1}, {"b": 2})
        {"a": 1, "b": 2}
        >>> merge_dicts({"a": 1}, {"a": 2, "b": 3})
        {"a": 2, "b": 3}
    """
    if left is None:
        return right or {}
    if right is None:
        return left
    return {**left, **right}
