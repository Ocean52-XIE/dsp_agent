# -*- coding: utf-8 -*-
"""RRF 归一化策略测试用例

测试 Reciprocal Rank Fusion (RRF) 归一化策略：
1. RRF 分数计算正确性
2. 分数范围验证 [0, 1]
3. 不同检索路径分数可比性
4. Wiki 和 Code 检索器一致性

运行方式：
    pytest tests/workflow/nodes/retrieval_flow/test_rrf_normalization.py -v
"""
from __future__ import annotations

import pytest
from pathlib import Path
import sys

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))


# ==================== RRF 分数计算测试 ====================

class TestRRFScoreCalculation:
    """RRF 分数计算测试"""

    def test_rrf_formula_correctness(self):
        """测试 RRF 公式计算正确性"""
        from workflow.nodes.retrieval_flow.retrieve_wiki.wiki_retriever import HybridScoreWeights

        k = HybridScoreWeights.RRF_K  # k = 60

        # 验证 k 值
        assert k == 60, f"RRF k 值应为 60，实际为 {k}"

        # 计算 RRF 分数
        def rrf_score(rank: int) -> float:
            return k / (rank + k)

        # 验证关键排名的分数
        assert abs(rrf_score(1) - 0.984) < 0.001, "rank=1 分数应为 0.984"
        assert abs(rrf_score(5) - 0.923) < 0.001, "rank=5 分数应为 0.923"
        assert abs(rrf_score(10) - 0.857) < 0.001, "rank=10 分数应为 0.857"
        assert abs(rrf_score(50) - 0.545) < 0.001, "rank=50 分数应为 0.545"

    def test_rrf_score_range(self):
        """测试 RRF 分数范围 [0, 1]"""
        k = 60

        def rrf_score(rank: int) -> float:
            if rank is None:
                return 0.0
            return k / (rank + k)

        # rank=1 应该接近 1
        assert rrf_score(1) < 1.0
        assert rrf_score(1) > 0.98

        # rank 越大，分数越低
        for rank in [1, 5, 10, 20, 50, 100]:
            assert 0.0 <= rrf_score(rank) <= 1.0, f"rank={rank} 分数超出 [0,1] 范围"

        # rank=None 应该返回 0
        assert rrf_score(None) == 0.0

    def test_rrf_monotonic_decreasing(self):
        """测试 RRF 分数单调递减"""
        k = 60

        def rrf_score(rank: int) -> float:
            return k / (rank + k)

        prev_score = rrf_score(1)
        for rank in range(2, 100):
            current_score = rrf_score(rank)
            assert current_score < prev_score, f"rank={rank} 分数应小于 rank={rank-1}"
            prev_score = current_score


# ==================== Wiki 检索器归一化测试 ====================

class TestWikiRetrieverNormalization:
    """Wiki 检索器归一化测试"""

    def test_hybrid_weights_normalized(self):
        """测试混合权重归一化"""
        from workflow.nodes.retrieval_flow.retrieve_wiki.wiki_retriever import HybridScoreWeights

        weights = HybridScoreWeights()
        normalized = weights.normalized()

        # 权重应该非负
        assert normalized.bm25 >= 0
        assert normalized.embedding >= 0
        assert normalized.lexical >= 0

        # 权重总和应该为 1
        total = normalized.bm25 + normalized.embedding + normalized.lexical
        assert abs(total - 1.0) < 0.001, f"权重总和应为 1.0，实际为 {total}"

    def test_lexical_score_range(self):
        """测试词法覆盖率分数范围 [0, 1]"""
        from workflow.nodes.retrieval_flow.retrieve_wiki.wiki_retriever import MarkdownWikiRetriever

        # 创建模拟检索器
        retriever = MagicMock(spec=MarkdownWikiRetriever)

        # 测试完全覆盖
        text = "这是一段测试文本包含关键词"
        terms = ["测试", "关键词", "文本"]
        score = MarkdownWikiRetriever._lexical_match_score(retriever, text, terms)

        assert 0.0 <= score <= 1.0, f"词法分数超出范围: {score}"
        assert score > 0.9, "完全覆盖时分数应接近 1.0"

        # 测试部分覆盖
        partial_terms = ["测试", "不存在的词"]
        partial_score = MarkdownWikiRetriever._lexical_match_score(retriever, text, partial_terms)

        assert 0.0 <= partial_score <= 1.0
        assert abs(partial_score - 0.5) < 0.1, "部分覆盖时分数应约 0.5"

        # 测试无覆盖
        no_match_terms = ["不存在的词1", "不存在的词2"]
        no_match_score = MarkdownWikiRetriever._lexical_match_score(retriever, text, no_match_terms)

        assert no_match_score == 0.0, "无覆盖时分数应为 0"

    def test_module_boost_range(self):
        """测试模块加成分数范围 [0, 0.15]"""
        from workflow.nodes.retrieval_flow.retrieve_wiki.wiki_retriever import (
            MarkdownWikiRetriever, WikiChunk
        )

        # 创建模拟检索器
        retriever = MagicMock(spec=MarkdownWikiRetriever)
        retriever.module_doc_hints = {"test_module": ("hint1", "hint2")}
        retriever._to_relative_path = lambda p: str(p)

        # 创建测试块
        chunk = WikiChunk(
            chunk_id=1,
            source_path=Path("wiki/hint1_doc.md"),
            title="Hint1 文档",
            section="测试章节",
            chunk_type="paragraph",
            content="这是测试内容",
            normalized_text="测试内容",
        )

        # 计算模块加成
        boost = MarkdownWikiRetriever._module_prior_boost(
            retriever, chunk=chunk, module_name="test_module"
        )

        assert 0.0 <= boost <= 0.15, f"模块加成超出范围 [0, 0.15]: {boost}"


# ==================== Code 检索器归一化测试 ====================

class TestCodeRetrieverNormalization:
    """Code 检索器归一化测试"""

    def test_code_lexical_score_range(self):
        """测试代码词法分数范围 [0, 1]"""
        from workflow.nodes.retrieval_flow.retrieve_code.code_retriever import (
            LocalCodeRetriever, CodeChildChunk
        )

        # 创建模拟检索器
        retriever = MagicMock(spec=LocalCodeRetriever)
        retriever._normalize = lambda x: x.lower() if x else ""

        # 创建测试块
        child = CodeChildChunk(
            child_id="test_001",
            parent_id="parent_001",
            source_path=Path("codes/ad_engine.py"),
            language="python",
            chunk_type="function",
            symbol_name="compute_bid",
            signature="def compute_bid(request: Request) -> Bid",
            start_line=10,
            end_line=30,
            content="def compute_bid(request):\n    return calculate(request)",
            normalized_text="compute bid request calculate",
            normalized_path="ad engine",
            normalized_symbol="compute bid",
        )

        # 测试词法匹配
        patterns = {"identifiers": ["compute", "bid", "request"]}

        # 使用真实方法
        score, matched = LocalCodeRetriever._score_lexical(retriever, child, patterns)

        assert 0.0 <= score <= 1.0, f"词法分数超出范围: {score}"
        assert len(matched) > 0, "应该有匹配的词"

    def test_code_pattern_score_range(self):
        """测试代码模式分数范围 [0, 1]"""
        from workflow.nodes.retrieval_flow.retrieve_code.code_retriever import (
            LocalCodeRetriever, CodeChildChunk
        )

        retriever = MagicMock(spec=LocalCodeRetriever)

        # 创建测试块
        child = CodeChildChunk(
            child_id="test_001",
            parent_id="parent_001",
            source_path=Path("codes/ad_engine.py"),
            language="python",
            chunk_type="function",
            symbol_name="compute_bid",
            signature="def compute_bid(request: Request) -> Bid",
            start_line=10,
            end_line=30,
            content="def compute_bid(request):\n    return calculate(request)",
            normalized_text="compute bid request calculate",
            normalized_path="ad engine",
            normalized_symbol="compute bid",  # 归一化后是空格分隔
        )

        # 测试精确匹配（使用归一化后的标识符）
        patterns = {
            "exact_identifiers": ["compute bid"],  # 归一化后的标识符
            "is_location_query": False,
        }

        score, matched = LocalCodeRetriever._score_pattern(retriever, child, patterns)

        assert 0.0 <= score <= 1.0, f"模式分数超出范围: {score}"
        assert len(matched) > 0, "应该有匹配的模式"


# ==================== 分数可比性测试 ====================

class TestScoreComparability:
    """不同检索路径分数可比性测试"""

    def test_different_paths_scores_comparable(self):
        """测试不同检索路径分数可比"""
        k = 60

        def rrf_score(rank: int) -> float:
            return k / (rank + k)

        # 模拟不同检索器的排名
        bm25_rank = 1
        embedding_rank = 5
        lexical_score = 0.8  # 已经是 [0, 1] 范围

        bm25_score = rrf_score(bm25_rank)
        embedding_score = rrf_score(embedding_rank)

        # 所有分数都在 [0, 1] 范围内
        assert 0.0 <= bm25_score <= 1.0
        assert 0.0 <= embedding_score <= 1.0
        assert 0.0 <= lexical_score <= 1.0

        # 可以直接比较
        assert bm25_score > embedding_score  # rank 1 > rank 5

    def test_weighted_sum_meaningful(self):
        """测试加权求和有意义"""
        k = 60

        def rrf_score(rank: int) -> float:
            return k / (rank + k)

        # 模拟一个文档在不同检索器的排名
        bm25_score = rrf_score(3)  # 0.952
        embedding_score = rrf_score(1)  # 0.984
        lexical_score = 0.6  # 60% 覆盖率

        # 使用权重计算混合分数
        bm25_weight = 0.30
        embedding_weight = 0.50
        lexical_weight = 0.20

        final_score = (
            bm25_score * bm25_weight
            + embedding_score * embedding_weight
            + lexical_score * lexical_weight
        )

        # 最终分数应该在合理范围内
        assert 0.0 <= final_score <= 1.0, f"混合分数超出范围: {final_score}"

        # 混合分数应该反映各路检索的贡献
        # embedding 排名最好，权重最高，应该主导结果
        assert final_score > 0.85, "混合分数应该较高（embedding 排名 1）"


# ==================== 配置测试 ====================

class TestNormalizationConfig:
    """归一化配置测试"""

    def test_wiki_config_rrk_k_constant(self):
        """测试 Wiki 检索器 RRF_K 常量"""
        from workflow.nodes.retrieval_flow.retrieve_wiki.wiki_retriever import HybridScoreWeights

        assert hasattr(HybridScoreWeights, "RRF_K")
        assert HybridScoreWeights.RRF_K == 60

    def test_code_config_rrk_k_constant(self):
        """测试 Code 检索器 RRF_K 常量"""
        from workflow.nodes.retrieval_flow.retrieve_code.code_retriever import CodeRetrieverRuntimeConfig

        assert hasattr(CodeRetrieverRuntimeConfig, "RRF_K")
        assert CodeRetrieverRuntimeConfig.RRF_K == 60

    def test_boost_values_within_rrf_range(self):
        """测试权重值在合理范围内（简化版）"""
        from workflow.nodes.retrieval_flow.retrieve_wiki.wiki_retriever import (
            WikiRetrieverRuntimeConfig, HybridScoreWeights
        )
        from workflow.nodes.retrieval_flow.retrieve_code.code_retriever import CodeRetrieverRuntimeConfig

        # Wiki RRF_K 常量在 HybridScoreWeights 类中
        assert HybridScoreWeights.RRF_K == 60

        # Wiki 权重值应该合理
        wiki_config = WikiRetrieverRuntimeConfig()
        # 检查 HybridScoreWeights 默认值
        weights = HybridScoreWeights()
        assert 0.0 <= weights.bm25 <= 1.0
        assert 0.0 <= weights.embedding <= 1.0
        assert 0.0 <= weights.lexical <= 1.0

        # Code 权重值应该合理（简化后移除了 RG）
        code_config = CodeRetrieverRuntimeConfig()
        assert CodeRetrieverRuntimeConfig.RRF_K == 60
        assert 0.0 <= code_config.bm25_weight <= 1.0
        assert 0.0 <= code_config.embedding_weight <= 1.0
        assert 0.0 <= code_config.pattern_weight <= 1.0


# ==================== 模拟辅助 ====================

from unittest.mock import MagicMock


# ==================== 运行入口 ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
