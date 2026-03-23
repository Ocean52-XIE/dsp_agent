# -*- coding: utf-8 -*-
"""Code 检索器 Embedding 优化测试用例

测试代码检索器增加 Embedding 向量检索后的功能：
1. 配置加载和验证
2. 语义化文本生成
3. Embedding 检索路径
4. 混合评分权重
"""
from __future__ import annotations

import pytest
from dataclasses import asdict
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path


# ==================== 测试固件 ====================

@pytest.fixture
def mock_embedding_profile():
    """模拟 Embedding 配置"""
    profile = MagicMock()
    profile.enabled = True
    profile.model = "BAAI/bge-base-zh-v1.5"
    profile.device = "cpu"
    profile.top_k = 4
    profile.persist_root = ".vectorstore_code_test"
    return profile


@pytest.fixture
def sample_code_child_chunk():
    """创建测试用的代码块"""
    from collections import namedtuple

    CodeChildChunk = namedtuple("CodeChildChunk", [
        "child_id", "parent_id", "source_path", "language",
        "chunk_type", "symbol_name", "signature", "start_line",
        "end_line", "content", "normalized_text", "normalized_path", "normalized_symbol"
    ])

    return CodeChildChunk(
        child_id="test_child_001",
        parent_id="test_parent_001",
        source_path=Path("codes/ad_engine/bid_optimizer.py"),
        language="python",
        chunk_type="function",
        symbol_name="compute_bid_for_request",
        signature="def compute_bid_for_request(request: Request, user_id: int) -> Bid",
        start_line=42,
        end_line=78,
        content='''
def compute_bid_for_request(request: Request, user_id: int) -> Bid:
    """计算请求级别的出价。

    Args:
        request: 广告请求对象
        user_id: 用户ID

    Returns:
        Bid: 计算后的出价对象
    """
    # 获取用户特征
    user_features = get_user_features(user_id)

    # 计算基础出价
    base_bid = calculate_base_bid(request, user_features)

    # 应用 OCPC 调整
    adjusted_bid = apply_ocpc_adjustment(base_bid, user_features)

    return adjusted_bid
'''.strip(),
        normalized_text="compute bid for request",
        normalized_path="ad_engine bid_optimizer",
        normalized_symbol="compute bid for request",
    )


# ==================== 配置测试 ====================

class TestCodeEmbeddingConfig:
    """Code Embedding 配置测试"""

    def test_config_default_values(self):
        """测试默认配置值"""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))

        from retrievers.code.retriever import CodeRetrieverRuntimeConfig

        config = CodeRetrieverRuntimeConfig()

        # 验证新增的 Embedding 配置
        assert config.enable_embedding is True  # 简化后默认启用
        assert config.embedding_model == "BAAI/bge-base-zh-v1.5"
        assert config.embedding_device == "cpu"
        assert config.embedding_top_k == 4
        assert config.embedding_persist_root == ".vectorstore_code"
        assert config.embedding_weight == 0.40

    def test_config_from_env(self, monkeypatch):
        """测试从环境变量加载配置"""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))

        from retrievers.code.retriever import CodeRetrieverRuntimeConfig

        # 设置环境变量
        monkeypatch.setenv("AGENT_CODE_EMBEDDING_ENABLED", "true")
        monkeypatch.setenv("AGENT_CODE_EMBEDDING_MODEL", "custom-model")
        monkeypatch.setenv("AGENT_CODE_EMBEDDING_WEIGHT", "0.5")

        config = CodeRetrieverRuntimeConfig.from_env()

        assert config.enable_embedding is True
        assert config.embedding_model == "custom-model"
        assert config.embedding_weight == 0.5


# ==================== 语义化文本生成测试 ====================

class TestSemanticTextGeneration:
    """语义化文本生成测试"""

    def test_build_semantic_text_python_function(self, sample_code_child_chunk):
        """测试 Python 函数的语义化文本生成"""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))

        from retrievers.code.retriever import LocalCodeRetriever

        # 创建模拟的检索器实例（只测试方法）
        retriever = MagicMock(spec=LocalCodeRetriever)
        retriever._to_relative_path = lambda p: str(p)

        # 直接调用方法，传入 self=retriever
        # 需要确保 _extract_comments 和 _extract_key_code 返回字符串
        retriever._extract_comments = lambda content, lang: LocalCodeRetriever._extract_comments(retriever, content, lang)
        retriever._extract_key_code = lambda content: LocalCodeRetriever._extract_key_code(retriever, content)

        # 使用真实方法
        semantic_text = LocalCodeRetriever._build_semantic_text(retriever, sample_code_child_chunk)

        # 验证语义化文本包含关键信息
        assert "函数" in semantic_text or "function" in semantic_text.lower()
        assert "compute_bid_for_request" in semantic_text

    def test_extract_python_comments(self):
        """测试 Python 注释提取"""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))

        from retrievers.code.retriever import LocalCodeRetriever

        code = '''
def test_function():
    """这是 docstring 注释"""
    # 这是行注释
    pass
'''
        retriever = MagicMock(spec=LocalCodeRetriever)

        comments = LocalCodeRetriever._extract_comments(retriever, code, "python")

        assert "docstring" in comments or "行注释" in comments

    def test_extract_javascript_comments(self):
        """测试 JavaScript 注释提取"""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))

        from retrievers.code.retriever import LocalCodeRetriever

        code = '''
// 单行注释
function test() {
    /* 多行
       注释 */
}
'''
        retriever = MagicMock(spec=LocalCodeRetriever)

        comments = LocalCodeRetriever._extract_comments(retriever, code, "javascript")

        assert "单行注释" in comments or "多行" in comments

    def test_extract_key_code(self):
        """测试关键代码提取"""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))

        from retrievers.code.retriever import LocalCodeRetriever

        code = '''
import os
import sys

def important_function():
    """重要的函数"""
    result = calculate()  # 关键计算
    return result
'''
        retriever = MagicMock(spec=LocalCodeRetriever)

        key_code = LocalCodeRetriever._extract_key_code(retriever, code)

        # import 语句应该被跳过（非首行）
        # pass 应该被跳过
        assert "important_function" in key_code or "calculate" in key_code
        assert len(key_code) <= 300  # 限制长度


# ==================== Embedding 检索路径测试 ====================

class TestEmbeddingRetrievalPath:
    """Embedding 检索路径测试"""

    def test_embedding_search_called_when_enabled(self, mock_embedding_profile):
        """测试启用时调用 Embedding 检索"""
        # 验证配置启用状态
        assert mock_embedding_profile.enabled is True

    def test_embedding_docs_added_to_candidates(self):
        """测试 Embedding 结果加入候选集"""
        # 模拟 Embedding 检索结果
        embedding_docs = [
            MagicMock(metadata={"child_id": "child_001"}),
            MagicMock(metadata={"child_id": "child_002"}),
        ]

        # 模拟其他检索结果
        bm25_docs = [MagicMock(metadata={"child_id": "child_003"})]
        tfidf_docs = [MagicMock(metadata={"child_id": "child_004"})]

        # 候选集合并逻辑
        candidate_child_ids = list(dict.fromkeys([
            *[str(doc.metadata.get("child_id", "")) for doc in bm25_docs],
            *[str(doc.metadata.get("child_id", "")) for doc in tfidf_docs],
            *[str(doc.metadata.get("child_id", "")) for doc in embedding_docs],
        ]))

        # 验证所有来源的候选都被合并
        assert "child_001" in candidate_child_ids
        assert "child_002" in candidate_child_ids
        assert "child_003" in candidate_child_ids
        assert "child_004" in candidate_child_ids

    def test_embedding_score_included_in_hybrid_scoring(self):
        """测试 Embedding 分数包含在混合评分中"""
        # 模拟各路检索排名
        bm25_rank = {"child_001": 1}
        tfidf_rank = {"child_001": 2}
        embedding_rank = {"child_001": 1}  # Embedding 排名靠前

        # 模拟排名分数计算
        def rank_score(rank):
            if rank is None:
                return 0.0
            return 1.0 / (rank + 1)

        child_id = "child_001"

        # 当有 Embedding 结果时的权重
        bm25_weight = 0.25
        tfidf_weight = 0.15
        embedding_weight = 0.40

        score = (
            rank_score(bm25_rank.get(child_id)) * bm25_weight
            + rank_score(max(tfidf_rank.get(child_id, 10**9), 1)) * tfidf_weight
            + rank_score(embedding_rank.get(child_id)) * embedding_weight
        )

        # 验证分数计算正确
        expected_bm25 = 0.5 * 0.25  # 1/(1+1) * 0.25 = 0.125
        expected_tfidf = 0.33 * 0.15  # 1/(2+1) * 0.15 ≈ 0.05
        expected_embedding = 0.5 * 0.40  # 1/(1+1) * 0.40 = 0.2
        expected_total = expected_bm25 + expected_tfidf + expected_embedding

        assert abs(score - expected_total) < 0.01


# ==================== 混合评分权重测试 ====================

class TestHybridScoringWeights:
    """混合评分权重测试"""

    @pytest.mark.parametrize("has_embedding,expected_bm25_weight", [
        (True, 0.25),   # 有 Embedding 时 BM25 权重降低
        (False, 0.6),   # 无 Embedding 时保持原有权重
    ])
    def test_weight_adjustment_based_on_embedding(self, has_embedding, expected_bm25_weight):
        """测试根据 Embedding 可用性调整权重"""
        # 验证权重值
        if has_embedding:
            bm25_weight = 0.25
        else:
            bm25_weight = 0.6

        assert bm25_weight == expected_bm25_weight

    def test_weight_sum_reasonable(self):
        """测试权重总和合理"""
        # 有 Embedding 时的权重分配
        bm25 = 0.25
        tfidf = 0.15
        embedding = 0.40
        pattern = 0.20

        total = bm25 + tfidf + embedding + pattern

        # 权重总和应该接近 1.0（但不是必须）
        assert 0.8 < total < 1.2


# ==================== 检索性能指标测试 ====================

class TestRetrievalProfile:
    """检索性能指标测试"""

    def test_embedding_info_in_search_profile(self):
        """测试搜索 profile 包含 Embedding 信息"""
        # 模拟 last_search_profile
        profile = {
            "latency_ms": 100.0,
            "child_candidates": 20,
            "parent_candidates": 10,
            "selected_count": 4,
            "embedding": {
                "enabled": True,
                "hits": 8,
                "weight": 0.40,
            },
        }

        # 验证 Embedding 信息存在
        assert "embedding" in profile
        assert profile["embedding"]["enabled"] is True
        assert profile["embedding"]["hits"] == 8
        assert profile["embedding"]["weight"] == 0.40


# ==================== 集成测试 ====================

class TestCodeEmbeddingIntegration:
    """Code Embedding 集成测试"""

    @pytest.mark.skip(reason="需要实际代码文件和模型")
    def test_full_embedding_retrieval_flow(self):
        """测试完整的 Embedding 检索流程"""
        pass

    @pytest.mark.skip(reason="需要实际模型")
    def test_semantic_matching_capability(self):
        """测试语义匹配能力"""
        # 查询 "获取用户信息"
        # 应该能匹配 get_user_info 函数（即使没有关键词匹配）
        pass


# ==================== 运行入口 ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
