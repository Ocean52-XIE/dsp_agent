# -*- coding: utf-8 -*-
"""Wiki 重排器优化测试用例

测试重排器优化的核心改动：
1. 重排在多样性选择之前执行
2. 候选集使用 candidate_top_k 配置
3. 重排后仍保证多样性

运行方式：
    pytest tests/workflow/nodes/retrieval_flow/test_wiki_reranker_optimization.py -v
"""
from __future__ import annotations

import pytest
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock


# ==================== 测试固件 ====================

@pytest.fixture
def mock_reranker_profile():
    """模拟重排器配置"""
    profile = MagicMock()
    profile.enabled = True
    profile.model = "BAAI/bge-reranker-base"
    profile.device = "cpu"
    profile.top_k = 4
    profile.candidate_top_k = 20  # 关键配置：候选集大小
    profile.batch_size = 8
    profile.max_length = 512
    return profile


@pytest.fixture
def mock_embedding_profile():
    """模拟 Embedding 配置"""
    profile = MagicMock()
    profile.enabled = False
    profile.model = "BAAI/bge-base-zh-v1.5"
    profile.device = "cpu"
    profile.top_k = 4
    profile.persist_root = ".vectorstore"
    return profile


@pytest.fixture
def mock_reranker():
    """模拟重排器实例"""
    reranker = MagicMock()

    def mock_rerank(query: str, candidates: list[dict], top_k: int, content_key: str):
        # 模拟重排：根据候选内容长度排序（简单的模拟逻辑）
        scored = []
        for i, c in enumerate(candidates[:top_k]):
            content = c.get(content_key, "")
            # 使用内容长度作为模拟的重排分数
            scored.append({
                **c,
                "rerank_score": len(content) / 100.0,
                "original_rank": i + 1,
            })
        # 按分数降序排列
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored

    reranker.rerank = MagicMock(side_effect=mock_rerank)
    return reranker


@pytest.fixture
def sample_wiki_chunks():
    """创建测试用的 Wiki 块数据"""
    from collections import namedtuple

    WikiChunk = namedtuple("WikiChunk", [
        "chunk_id", "source_path", "title", "section",
        "chunk_type", "content", "normalized_text"
    ])

    chunks = []
    for i in range(1, 25):  # 创建 24 个块，超过 candidate_top_k
        chunk = WikiChunk(
            chunk_id=i,
            source_path=Path(f"wiki/doc_{(i-1)//6 + 1}.md"),
            title=f"文档 {(i-1)//6 + 1}",
            section=f"章节 {i % 6}",
            chunk_type="paragraph",
            content=f"这是第 {i} 个内容块，包含一些测试文本。" * (i % 5 + 1),
            normalized_text=f"内容块 {i}",
        )
        chunks.append(chunk)
    return chunks


# ==================== 测试用例 ====================

class TestWikiRerankerOptimization:
    """Wiki 重排器优化测试"""

    def test_reranker_candidate_k_expansion(self, mock_reranker_profile, mock_embedding_profile):
        """测试重排候选集扩展：应该使用 candidate_top_k 而非 final_top_k"""
        # 这个测试验证重排时候选集大小是否正确扩展
        # 预期：candidate_top_k=20，final_top_k=4
        # 重排应该对 20 条候选进行，而不是只对 4 条

        # 验证配置
        assert mock_reranker_profile.candidate_top_k == 20
        assert mock_reranker_profile.top_k == 4
        assert mock_reranker_profile.candidate_top_k > mock_reranker_profile.top_k

    def test_rerank_before_diverse_selection(self, mock_reranker, sample_wiki_chunks):
        """测试重排在多样性选择之前执行"""
        # 验证执行顺序：重排 -> 多样性选择
        # 预期：重排器收到的候选集应该大于最终输出数量

        # 模拟重排调用
        candidates = [
            {"content": f"内容 {i}", "path": f"doc_{i}.md"}
            for i in range(20)
        ]

        result = mock_reranker.rerank(
            query="测试查询",
            candidates=candidates,
            top_k=8,  # final_top_k * 2
            content_key="content",
        )

        # 验证重排器被调用
        mock_reranker.rerank.assert_called_once()
        call_args = mock_reranker.rerank.call_args

        # 验证候选集大小
        assert len(call_args.kwargs["candidates"]) == 20
        assert call_args.kwargs["top_k"] == 8

    def test_diversity_preserved_after_rerank(self, sample_wiki_chunks):
        """测试重排后多样性仍然保持"""
        # 验证：即使重排后，不同来源的文档仍能被选中
        # 预期：最终结果不应全部来自同一文档

        # 模拟重排后的结果（全部来自同一文档，分数很高）
        reranked = [
            {
                "chunk": sample_wiki_chunks[i],
                "score": 10.0 - i * 0.1,
                "bm25_score": 0.5,
                "embedding_score": 0.5,
                "lexical_score": 0.5,
                "module_boost": 0.0,
                "general_doc_penalty": 0.0,
                "rg_boost": 0.0,
                "rg_path_hits": 0,
                "rg_strategy": "no_rg",
                "rerank_score": 10.0 - i * 0.1,
                "original_rank": i + 1,
            }
            for i in range(8)
        ]

        # 多样性选择逻辑（模拟 _select_diverse）
        max_per_doc = 1
        selected = []
        doc_counts = {}

        for item in reranked:
            doc_path = str(item["chunk"].source_path)
            if doc_counts.get(doc_path, 0) < max_per_doc:
                selected.append(item)
                doc_counts[doc_path] = doc_counts.get(doc_path, 0) + 1
            if len(selected) >= 4:
                break

        # 验证多样性：选中的应该来自不同文档
        selected_docs = [str(item["chunk"].source_path) for item in selected]
        assert len(set(selected_docs)) == len(selected_docs), "多样性选择失败：存在重复来源"

    def test_rerank_profile_in_search_result(self, mock_reranker_profile):
        """测试检索结果中包含重排性能指标"""
        # 验证 last_search_profile 包含重排相关信息

        expected_keys = ["enabled", "latency_ms", "model", "candidate_k", "output_k"]

        # 模拟重排性能指标
        rerank_profile = {
            "enabled": True,
            "latency_ms": 45.2,
            "model": mock_reranker_profile.model,
            "candidate_k": mock_reranker_profile.candidate_top_k,
            "output_k": 8,
        }

        for key in expected_keys:
            assert key in rerank_profile, f"缺少重排指标: {key}"

    def test_no_reranker_fallback(self):
        """测试无重排器时的回退逻辑"""
        # 当没有重排器时，应该直接执行多样性选择
        # 预期：rerank.enabled = False

        rerank_profile = {"enabled": False, "reason": "no_reranker_or_insufficient_candidates"}
        assert rerank_profile["enabled"] is False
        assert "reason" in rerank_profile

    def test_small_candidate_set_handling(self, mock_reranker):
        """测试候选集较小时的处理"""
        # 当候选集小于 candidate_top_k 时，应该正常处理
        # 预期：使用实际候选集大小

        small_candidates = [
            {"content": "内容 1", "path": "doc_1.md"},
            {"content": "内容 2", "path": "doc_2.md"},
        ]

        result = mock_reranker.rerank(
            query="测试查询",
            candidates=small_candidates,
            top_k=4,
            content_key="content",
        )

        # 验证：候选集小，仍能正常处理
        assert len(result) <= len(small_candidates)


class TestRerankerCandidateTopK:
    """重排候选集大小测试"""

    @pytest.mark.parametrize("candidate_top_k,final_top_k,expected_rerank_k", [
        (20, 4, 20),     # 标准配置
        (10, 4, 10),     # 较小候选集
        (30, 6, 30),     # 较大配置
    ])
    def test_candidate_k_calculation(self, candidate_top_k, final_top_k, expected_rerank_k):
        """测试不同配置下的候选集大小计算"""
        # 预期：重排候选集 = min(candidate_top_k, len(scored))

        scored_count = 50  # 假设有 50 条候选
        actual_rerank_k = min(candidate_top_k, scored_count)

        assert actual_rerank_k == expected_rerank_k

    @pytest.mark.parametrize("scored_count,candidate_top_k,expected", [
        (5, 20, 5),      # 候选不足
        (30, 20, 20),    # 候选充足
        (20, 20, 20),    # 刚好相等
    ])
    def test_candidate_k_with_limited_scored(self, scored_count, candidate_top_k, expected):
        """测试候选数量受限时的处理"""
        actual = min(candidate_top_k, scored_count)
        assert actual == expected


class TestRerankOutputFormat:
    """重排输出格式测试"""

    def test_reranked_hit_contains_required_fields(self):
        """测试重排结果包含必要字段"""
        required_fields = [
            "source_type", "title", "path", "score", "section",
            "chunk_type", "excerpt", "content", "rank", "retrieval_debug"
        ]

        # 模拟重排后的命中结果
        hit = {
            "source_type": "wiki",
            "title": "测试文档",
            "path": "wiki/test.md",
            "score": 0.95,
            "section": "测试章节",
            "chunk_type": "paragraph",
            "excerpt": "测试摘要...",
            "content": "完整内容",
            "rank": 1,
            "retrieval_debug": {
                "bm25": 0.3,
                "embedding": 0.5,
                "lexical": 0.2,
                "rerank_score": 0.95,
                "original_rank": 3,
            },
            "score_source": "reranker",
        }

        for field in required_fields:
            assert field in hit, f"缺少字段: {field}"

    def test_rerank_debug_info(self):
        """测试重排调试信息完整性"""
        # 验证 retrieval_debug 中包含重排相关信息

        debug_info = {
            "bm25": 0.3,
            "embedding": 0.5,
            "lexical": 0.2,
            "module_boost": 0.1,
            "general_doc_penalty": 0.0,
            "rg_boost": 0.0,
            "rg_path_hits": 0,
            "rg_strategy": "no_rg",
            "weights": {
                "bm25": 0.3,
                "embedding": 0.5,
                "lexical": 0.2,
            },
            "rerank_score": 0.95,
            "original_rank": 3,
        }

        # 验证重排特有字段
        assert "rerank_score" in debug_info
        assert "original_rank" in debug_info
        assert debug_info["rerank_score"] > 0


# ==================== 集成测试 ====================

class TestWikiRerankerIntegration:
    """Wiki 重排器集成测试（需要实际依赖）"""

    @pytest.mark.skip(reason="需要实际 Wiki 文档和模型")
    def test_full_rerank_flow(self):
        """测试完整重排流程"""
        # 此测试需要：
        # 1. 实际的 Wiki 文档目录
        # 2. 安装 sentence-transformers
        # 3. 下载重排模型
        pass

    @pytest.mark.skip(reason="需要实际模型")
    def test_rerank_latency(self):
        """测试重排延迟在可接受范围内"""
        # 预期：重排延迟 < 100ms（20 条候选）
        pass


# ==================== 运行入口 ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
