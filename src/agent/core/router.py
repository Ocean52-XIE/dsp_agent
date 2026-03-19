# -*- coding: utf-8 -*-
"""Domain Router - 规则层路由

纯规则实现，不做 LLM 调用，用于快速过滤和分流。

核心职责：
1. 闲聊拦截 - 快速识别并响应闲聊
2. 领域门控 - 判断请求是否属于领域范围
3. 模块分流 - 路由到特定领域模块 (可选)

关键设计：
- 纯规则实现，无 LLM 调用，毫秒级响应
- 可完全禁用：enabled=false 时直接进入 Agent Loop
- 配置来源于 domain_profile.json
"""
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from agent.state import AgentState, AgentStatus

logger = logging.getLogger(__name__)


class DomainRouteType(str, Enum):
    """Domain Router 输出类型

    定义规则路由的可能结果。
    """
    # 拒绝类
    OUT_OF_SCOPE = "out_of_scope"      # 领域外，直接拒绝/转交
    SMALL_TALK = "small_talk"           # 闲聊，直接响应

    # 通过类
    PASS_TO_AGENT = "pass_to_agent"     # 通用请求，交给 Agent Loop
    MODULE_ROUTED = "module_routed"     # 命中特定模块，走专属 workflow


@dataclass
class DomainRouterConfig:
    """Domain Router 配置

    从 domain_profile.json 的 domain_gate 和 routing 节点加载。

    Attributes:
        enabled: 是否启用 Domain Router
        threshold: 领域相关性阈值
        domain_terms: 领域词汇表
        offtopic_terms: 领域外词汇表
        small_talk_exact: 闲聊精确匹配
        small_talk_substr: 闲聊子串匹配
        module_routing_enabled: 是否启用模块分流
        modules: 领域模块定义列表
    """
    # 开关
    enabled: bool = True                    # 是否启用 Domain Router

    # 领域门控
    threshold: float = 0.5                  # 领域相关性阈值
    domain_terms: tuple[str, ...] = ()      # 领域词汇
    offtopic_terms: tuple[str, ...] = ()    # 领域外词汇
    small_talk_exact: tuple[str, ...] = ()  # 闲聊精确匹配
    small_talk_substr: tuple[str, ...] = () # 闲聊子串匹配

    # 模块分流
    module_routing_enabled: bool = True     # 是否启用模块分流
    modules: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_profile(cls, profile: dict[str, Any]) -> "DomainRouterConfig":
        """从 domain_profile 加载配置

        Args:
            profile: domain_profile.json 的内容

        Returns:
            DomainRouterConfig 实例
        """
        gate = profile.get("domain_gate", {})
        routing = profile.get("routing", {})
        modules = profile.get("modules", [])

        def _as_tuple(value: Any) -> tuple[str, ...]:
            """转换为元组"""
            if isinstance(value, (list, tuple)):
                return tuple(str(item).strip() for item in value if str(item).strip())
            return ()

        def _as_float(value: Any, default: float) -> float:
            """转换为浮点数"""
            try:
                return float(value) if value is not None else default
            except (TypeError, ValueError):
                return default

        return cls(
            enabled=bool(gate.get("enabled", True)),
            threshold=_as_float(gate.get("threshold"), 0.5),
            domain_terms=_as_tuple(gate.get("domain_terms", [])),
            offtopic_terms=_as_tuple(gate.get("offtopic_terms", [])),
            small_talk_exact=_as_tuple(gate.get("small_talk_exact", [])),
            small_talk_substr=_as_tuple(gate.get("small_talk_substr", [])),
            module_routing_enabled=bool(routing.get("enabled", True)),
            modules=list(modules) if isinstance(modules, list) else [],
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DomainRouterConfig":
        """从字典创建配置（兼容旧接口）"""
        return cls(
            enabled=bool(data.get("enabled", True)),
            threshold=float(data.get("threshold", 0.5)),
            domain_terms=tuple(data.get("domain_terms", [])),
            offtopic_terms=tuple(data.get("offtopic_terms", [])),
            small_talk_exact=tuple(data.get("small_talk_exact", [])),
            small_talk_substr=tuple(data.get("small_talk_substr", [])),
            module_routing_enabled=bool(data.get("module_routing_enabled", True)),
            modules=list(data.get("modules", [])),
        )


@dataclass
class DomainRouterResult:
    """Domain Router 结果

    Attributes:
        route: 路由类型
        domain_relevance: 领域相关性分数 (0-1)
        matched_module: 命中的模块名 (MODULE_ROUTED)
        rejection_reason: 拒绝原因 (OUT_OF_SCOPE)
        quick_response: 快速响应 (闲聊/拒绝)
    """
    route: DomainRouteType = DomainRouteType.PASS_TO_AGENT
    domain_relevance: float = 0.0
    matched_module: str | None = None
    rejection_reason: str = ""
    quick_response: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "route": self.route.value,
            "domain_relevance": self.domain_relevance,
            "matched_module": self.matched_module,
            "rejection_reason": self.rejection_reason,
            "quick_response": self.quick_response,
        }


# 代码提示正则
_CODE_HINT_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]{2,}\s*(\(|\.py\b|/)",
    re.IGNORECASE,
)

# 笑声正则
_LAUGH_LIKE_RE = re.compile(
    r"^[哈哈呵呵嘻嘻啊哦嗯\s!？?.。~～]+$",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    """标准化文本"""
    return " ".join((text or "").strip().lower().split())


def _count_hits(text: str, terms: tuple[str, ...]) -> int:
    """计算命中词汇数"""
    return sum(1 for term in terms if term and term in text)


class DomainRouter:
    """私域规则路由器

    职责：
    1. 快速判断请求是否属于领域范围 (纯规则，无 LLM)
    2. 拦截闲聊和无关请求
    3. 分流到特定领域模块 (可选)

    特点：
    - 纯规则实现，毫秒级响应
    - 可完全禁用 (enabled=false)
    - 配置来源于 domain_profile.json

    使用示例：
        router = DomainRouter(config=DomainRouterConfig.from_profile(profile))
        result = router.route(user_query)

        if result.route == DomainRouteType.PASS_TO_AGENT:
            # 进入 Agent Loop
            ...
        elif result.route == DomainRouteType.OUT_OF_SCOPE:
            # 拒绝响应
            return result.quick_response
    """

    # 默认响应模板
    DEFAULT_SMALL_TALK_RESPONSE = "您好，我是广告引擎助手，有什么可以帮您的？"
    DEFAULT_OUT_OF_SCOPE_RESPONSE = "抱歉，这个问题超出了我的专业领域（广告引擎）。"

    def __init__(self, config: DomainRouterConfig | None = None):
        """初始化 Domain Router

        Args:
            config: Router 配置
        """
        self.config = config or DomainRouterConfig()

        logger.info(
            f"[DomainRouter] 初始化完成, "
            f"enabled={self.config.enabled}, "
            f"threshold={self.config.threshold}, "
            f"domain_terms={len(self.config.domain_terms)}, "
            f"modules={len(self.config.modules)}"
        )

    def route(self, user_query: str) -> DomainRouterResult:
        """执行规则路由 (同步，无 LLM 调用)

        Args:
            user_query: 用户查询

        Returns:
            DomainRouterResult: 路由结果
        """
        # 如果禁用，直接通过到 Agent
        if not self.config.enabled:
            logger.debug("[DomainRouter] 已禁用，直接通过到 Agent Loop")
            return DomainRouterResult(route=DomainRouteType.PASS_TO_AGENT)

        normalized = _normalize(user_query)

        # 1. 闲聊检测
        if self._is_small_talk(normalized):
            logger.debug(f"[DomainRouter] 闲聊: {user_query[:50]}")
            return DomainRouterResult(
                route=DomainRouteType.SMALL_TALK,
                quick_response=self.DEFAULT_SMALL_TALK_RESPONSE,
            )

        # 2. 领域相关性计算
        relevance, reason = self._compute_relevance(normalized, user_query)

        # 3. 领域外判断
        if relevance < self.config.threshold:
            logger.debug(
                f"[DomainRouter] 领域外: relevance={relevance:.2f}, reason={reason}"
            )
            return DomainRouterResult(
                route=DomainRouteType.OUT_OF_SCOPE,
                domain_relevance=relevance,
                rejection_reason=reason,
                quick_response=self.DEFAULT_OUT_OF_SCOPE_RESPONSE,
            )

        # 4. 模块分流 (可选)
        if self.config.module_routing_enabled:
            matched_module = self._match_module(user_query, normalized)
            if matched_module:
                logger.info(f"[DomainRouter] 模块分流: {matched_module}")
                return DomainRouterResult(
                    route=DomainRouteType.MODULE_ROUTED,
                    domain_relevance=relevance,
                    matched_module=matched_module,
                )

        # 5. 通过到 Agent Loop
        logger.info(f"[DomainRouter] 通过: relevance={relevance:.2f}")
        return DomainRouterResult(
            route=DomainRouteType.PASS_TO_AGENT,
            domain_relevance=relevance,
        )

    def route_from_state(self, state: AgentState) -> DomainRouterResult:
        """从 AgentState 执行路由 (便捷方法)

        Args:
            state: Agent 状态

        Returns:
            DomainRouterResult: 路由结果
        """
        user_query = str(state.get("user_query", "") or "")
        return self.route(user_query)

    def _is_small_talk(self, normalized: str) -> bool:
        """检查是否为闲聊

        Args:
            normalized: 标准化后的查询

        Returns:
            是否为闲聊
        """
        # 精确匹配
        if normalized in self.config.small_talk_exact:
            return True

        # 子串匹配
        if any(token in normalized for token in self.config.small_talk_substr):
            return True

        # 笑声模式匹配
        if _LAUGH_LIKE_RE.fullmatch(normalized):
            return True

        return False

    def _compute_relevance(
        self,
        normalized: str,
        original: str,
    ) -> tuple[float, str]:
        """计算领域相关性

        Args:
            normalized: 标准化后的查询
            original: 原始查询

        Returns:
            (相关性分数, 原因说明)
        """
        # 计算领域词汇命中
        domain_hits = _count_hits(normalized, self.config.domain_terms)
        off_hits = _count_hits(normalized, self.config.offtopic_terms)

        # 代码提示检测
        code_hint = _CODE_HINT_RE.search(original) is not None

        # 计算相关性分数
        relevance = min(1.0, domain_hits * 0.25 + (0.25 if code_hint else 0.0))

        # 领域词汇命中时保底分数
        if domain_hits > 0:
            relevance = max(relevance, 0.5)

        # 领域外词汇惩罚
        if off_hits > 0 and domain_hits == 0 and not code_hint:
            relevance = max(0.0, relevance - 0.3)
            return relevance, f"offtopic_hits={off_hits}"

        # 短查询惩罚
        compact = normalized.replace(" ", "")
        if len(compact) <= 4 and domain_hits == 0 and not code_hint:
            return max(0.0, relevance - 0.2), "short_non_domain"

        reason = f"domain_hits={domain_hits}, code_hint={int(code_hint)}"
        return relevance, reason

    def _match_module(self, original: str, normalized: str) -> str | None:
        """匹配领域模块

        Args:
            original: 原始查询
            normalized: 标准化后的查询

        Returns:
            匹配的模块名，或 None
        """
        best_match: str | None = None
        best_priority = -1

        for module in self.config.modules:
            module_name = module.get("name", "")
            keywords = module.get("keywords", [])
            priority = module.get("route_priority", 0)

            # 检查关键词匹配
            for kw in keywords:
                if kw and (kw in normalized or kw in original):
                    if priority > best_priority:
                        best_priority = priority
                        best_match = module_name
                    break

        return best_match

    def update_state(
        self,
        state: AgentState,
        result: DomainRouterResult,
    ) -> None:
        """更新状态（便捷方法）

        Args:
            state: Agent 状态
            result: 路由结果
        """
        state["domain_route"] = result.route.value
        state["domain_relevance"] = result.domain_relevance
        state["matched_module"] = result.matched_module
        state["status"] = AgentStatus.ROUTING.value

        # 更新 debug_info
        if "debug_info" not in state:
            state["debug_info"] = {}
        state["debug_info"]["domain_router"] = result.to_dict()


# 保持向后兼容的别名
AgentRouter = DomainRouter
RouterConfig = DomainRouterConfig
RouterResult = DomainRouterResult
RouteType = DomainRouteType
