# -*- coding: utf-8 -*-
"""测试 DomainRouter 规则路由器

Domain Router 是纯规则层，不做 LLM 调用：
1. 闲聊拦截 - 快速识别并响应闲聊
2. 领域门控 - 判断请求是否属于领域范围
3. 模块分流 - 路由到特定领域模块 (可选)
"""
import pytest

from agent.core.router import (
    DomainRouter,
    DomainRouterConfig,
    DomainRouterResult,
    DomainRouteType,
    # 向后兼容别名
    AgentRouter,
    RouterConfig,
    RouterResult,
    RouteType,
)
from agent.state import create_initial_state


class TestDomainRouteType:
    """测试 DomainRouteType 枚举"""

    def test_route_types(self):
        """测试路由类型"""
        assert DomainRouteType.OUT_OF_SCOPE.value == "out_of_scope"
        assert DomainRouteType.SMALL_TALK.value == "small_talk"
        assert DomainRouteType.PASS_TO_AGENT.value == "pass_to_agent"
        assert DomainRouteType.MODULE_ROUTED.value == "module_routed"

    def test_backward_compat_route_type(self):
        """测试向后兼容别名"""
        # RouteType 是 DomainRouteType 的别名
        assert RouteType is DomainRouteType


class TestDomainRouterConfig:
    """测试 DomainRouterConfig"""

    def test_create_config(self):
        """测试创建配置"""
        config = DomainRouterConfig(
            domain_terms=("ctr", "cvr", "广告"),
            threshold=0.4,
        )

        assert config.domain_terms == ("ctr", "cvr", "广告")
        assert config.threshold == 0.4
        assert config.enabled is True

    def test_config_from_dict(self):
        """测试从字典创建配置"""
        data = {
            "domain_terms": ["ctr", "cvr"],
            "offtopic_terms": ["天气"],
            "threshold": 0.35,
        }

        config = DomainRouterConfig.from_dict(data)

        assert config.domain_terms == ("ctr", "cvr")
        assert config.offtopic_terms == ("天气",)
        assert config.threshold == 0.35

    def test_config_defaults(self):
        """测试配置默认值"""
        config = DomainRouterConfig()

        assert config.enabled is True
        assert config.domain_terms == ()
        assert config.threshold == 0.5
        assert config.module_routing_enabled is True

    def test_config_from_profile(self):
        """测试从 profile 创建配置"""
        profile = {
            "domain_gate": {
                "enabled": True,
                "threshold": 0.6,
                "domain_terms": ["广告", "投放"],
                "offtopic_terms": ["天气"],
            },
            "routing": {
                "enabled": False,
            },
            "modules": [
                {"name": "ad-recall", "keywords": ["召回"]}
            ],
        }

        config = DomainRouterConfig.from_profile(profile)

        assert config.enabled is True
        assert config.threshold == 0.6
        assert config.domain_terms == ("广告", "投放")
        assert config.offtopic_terms == ("天气",)
        assert config.module_routing_enabled is False
        assert len(config.modules) == 1


class TestDomainRouterResult:
    """测试 DomainRouterResult"""

    def test_create_result(self):
        """测试创建路由结果"""
        result = DomainRouterResult(
            route=DomainRouteType.PASS_TO_AGENT,
            domain_relevance=0.8,
        )

        assert result.route == DomainRouteType.PASS_TO_AGENT
        assert result.domain_relevance == 0.8
        assert result.matched_module is None

    def test_result_to_dict(self):
        """测试结果转字典"""
        result = DomainRouterResult(
            route=DomainRouteType.MODULE_ROUTED,
            domain_relevance=0.9,
            matched_module="ad-recall",
            quick_response=None,
        )

        result_dict = result.to_dict()

        assert result_dict["route"] == "module_routed"
        assert result_dict["domain_relevance"] == 0.9
        assert result_dict["matched_module"] == "ad-recall"


class TestDomainRouter:
    """测试 DomainRouter"""

    def test_create_router(self, sample_router_config):
        """测试创建路由器"""
        router = DomainRouter(config=sample_router_config)

        assert router.config == sample_router_config

    def test_route_domain_query(self, sample_router_config):
        """测试领域内查询路由"""
        router = DomainRouter(config=sample_router_config)

        result = router.route("查一下 CTR 预估模块")

        assert result.route != DomainRouteType.OUT_OF_SCOPE
        assert result.domain_relevance > 0

    def test_route_small_talk(self, sample_router_config):
        """测试闲聊路由"""
        router = DomainRouter(config=sample_router_config)

        result = router.route("你好")

        assert result.route == DomainRouteType.SMALL_TALK
        assert result.quick_response is not None

    def test_route_offtopic(self, sample_router_config):
        """测试领域外查询"""
        router = DomainRouter(config=sample_router_config)

        result = router.route("今天天气如何")

        assert result.route == DomainRouteType.OUT_OF_SCOPE
        assert result.domain_relevance < sample_router_config.threshold
        assert result.quick_response is not None

    def test_route_module_routed(self, sample_router_config):
        """测试模块分流"""
        # 添加模块分流配置
        config = DomainRouterConfig(
            domain_terms=sample_router_config.domain_terms,
            threshold=0.5,
            module_routing_enabled=True,
            modules=[
                {"name": "ad-recall", "keywords": ["召回"], "route_priority": 10},
            ],
        )
        router = DomainRouter(config=config)

        result = router.route("召回模块怎么工作")

        # 应该命中领域词汇且有模块匹配
        assert result.route in [
            DomainRouteType.PASS_TO_AGENT,
            DomainRouteType.MODULE_ROUTED,
        ]

    def test_route_disabled(self, sample_router_config):
        """测试禁用路由器"""
        config = DomainRouterConfig(
            enabled=False,
            domain_terms=sample_router_config.domain_terms,
        )
        router = DomainRouter(config=config)

        result = router.route("今天天气如何")

        # 禁用后应该直接通过
        assert result.route == DomainRouteType.PASS_TO_AGENT

    def test_route_from_state(self, sample_router_config):
        """测试从 AgentState 路由"""
        router = DomainRouter(config=sample_router_config)
        state = create_initial_state(
            trace_id="test",
            session_id="test",
            user_query="CTR 预估",
        )

        result = router.route_from_state(state)

        assert result is not None
        assert result.route != DomainRouteType.OUT_OF_SCOPE

    def test_update_state(self, sample_router_config):
        """测试更新状态"""
        router = DomainRouter(config=sample_router_config)
        state = create_initial_state(
            trace_id="test",
            session_id="test",
            user_query="CTR 预估",
        )

        result = router.route("CTR 预估")
        router.update_state(state, result)

        assert state["domain_route"] == result.route.value
        assert state["domain_relevance"] == result.domain_relevance

    def test_route_with_code_hint(self, sample_router_config):
        """测试带代码提示的查询"""
        router = DomainRouter(config=sample_router_config)

        result = router.route("CTR_Estimator.py 这个类怎么用")

        # 代码提示应该提高相关性
        assert result.domain_relevance > 0

    def test_route_short_query(self, sample_router_config):
        """测试短查询"""
        router = DomainRouter(config=sample_router_config)

        result = router.route("abc")

        # 短查询可能会被惩罚
        assert result is not None


class TestDomainRouterEdgeCases:
    """测试 DomainRouter 边界情况"""

    def test_empty_query(self, sample_router_config):
        """测试空查询"""
        router = DomainRouter(config=sample_router_config)

        result = router.route("")

        assert result is not None

    def test_whitespace_query(self, sample_router_config):
        """测试空白查询"""
        router = DomainRouter(config=sample_router_config)

        result = router.route("   ")

        assert result is not None

    def test_very_long_query(self, sample_router_config):
        """测试超长查询"""
        router = DomainRouter(config=sample_router_config)
        long_query = "CTR " * 1000

        result = router.route(long_query)

        assert result is not None
        # 应该有较高的相关性（因为包含很多领域词）
        assert result.domain_relevance > 0

    def test_special_characters(self, sample_router_config):
        """测试特殊字符"""
        router = DomainRouter(config=sample_router_config)

        result = router.route("CTR 预估！@#$%^&*()")

        assert result is not None

    def test_mixed_language(self, sample_router_config):
        """测试混合语言"""
        router = DomainRouter(config=sample_router_config)

        result = router.route("How to use CTR estimation 模块")

        assert result is not None

    def test_laugh_pattern(self, sample_router_config):
        """测试笑声模式"""
        router = DomainRouter(config=sample_router_config)

        result = router.route("哈哈哈")

        assert result.route == DomainRouteType.SMALL_TALK


class TestBackwardCompatibility:
    """测试向后兼容性"""

    def test_agent_router_is_domain_router(self):
        """测试 AgentRouter 是 DomainRouter 的别名"""
        assert AgentRouter is DomainRouter

    def test_router_config_is_domain_router_config(self):
        """测试 RouterConfig 是 DomainRouterConfig 的别名"""
        assert RouterConfig is DomainRouterConfig

    def test_router_result_is_domain_router_result(self):
        """测试 RouterResult 是 DomainRouterResult 的别名"""
        assert RouterResult is DomainRouterResult

    def test_old_api_still_works(self, sample_router_config):
        """测试旧 API 仍然可用"""
        # 使用旧名称
        router = AgentRouter(config=RouterConfig.from_dict({
            "domain_terms": ["ctr"],
        }))

        result = router.route("CTR 怎么算")

        assert result is not None
