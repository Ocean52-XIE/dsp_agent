# -*- coding: utf-8 -*-
"""模型缓存模块测试

测试目标：
1. 验证单例缓存正确工作
2. 验证线程安全性
3. 验证缓存统计准确性
4. 验证缓存清理功能
5. 验证本地模型路径支持 (cache_dir)
"""
from __future__ import annotations

import os
import shutil
import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest


_TEST_TMP_ROOT = Path.cwd() / ".tmp" / "model_cache_tests"
_TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)


def _new_test_dir() -> Path:
    """Create a writable temp directory inside the repo."""
    path = _TEST_TMP_ROOT / uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    return path


class TestModelCache:
    """模型缓存测试类"""

    def test_get_cache_stats_initial(self):
        """测试初始缓存统计为空"""
        from retrievers.core.model_cache import get_cache_stats, clear_cache

        # 清空缓存确保初始状态
        clear_cache()
        stats = get_cache_stats()

        assert stats["embedding"]["cache_size"] == 0
        assert stats["embedding"]["hits"] == 0
        assert stats["embedding"]["misses"] == 0
        assert stats["reranker"]["cache_size"] == 0
        assert stats["reranker"]["hits"] == 0
        assert stats["reranker"]["misses"] == 0

    def test_clear_cache(self):
        """测试清空缓存功能"""
        from retrievers.core.model_cache import (
            _embedding_cache,
            _reranker_cache,
            clear_cache,
        )

        # 添加一些模拟缓存
        with patch.dict(_embedding_cache, {"model1": MagicMock()}, clear=False):
            with patch.dict(_reranker_cache, {"model2": MagicMock()}, clear=False):
                clear_cache()

        # 验证缓存已清空
        assert len(_embedding_cache) == 0
        assert len(_reranker_cache) == 0


class TestResolveModelPath:
    """模型路径解析测试"""

    def test_resolve_model_path_huggingface_id(self):
        """测试解析 HuggingFace Hub ID"""
        from retrievers.core.model_cache import _resolve_model_path

        # Hub ID 应该直接返回
        result = _resolve_model_path("BAAI/bge-base-zh-v1.5", cache_dir=None)
        assert result == "BAAI/bge-base-zh-v1.5"

    def test_resolve_model_path_absolute_path(self):
        """测试解析绝对路径"""
        from retrievers.core.model_cache import _resolve_model_path

        tmpdir = _new_test_dir()
        try:
            model_path = tmpdir / "test-model"
            model_path.mkdir(parents=True, exist_ok=True)

            result = _resolve_model_path(str(model_path), cache_dir=None)
            assert result == str(model_path)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_resolve_model_path_cache_dir_direct(self):
        """测试从 cache_dir 直接加载模型"""
        from retrievers.core.model_cache import _resolve_model_path

        tmpdir = _new_test_dir()
        try:
            model_path = tmpdir / "BAAI" / "bge-base-zh-v1.5"
            model_path.mkdir(parents=True, exist_ok=True)

            result = _resolve_model_path("BAAI/bge-base-zh-v1.5", cache_dir=str(tmpdir))
            assert result == str(model_path)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_resolve_model_path_cache_dir_basename(self):
        """测试从 cache_dir 使用 basename 加载模型"""
        from retrievers.core.model_cache import _resolve_model_path

        tmpdir = _new_test_dir()
        try:
            model_path = tmpdir / "bge-base-zh-v1.5"
            model_path.mkdir(parents=True, exist_ok=True)

            result = _resolve_model_path("BAAI/bge-base-zh-v1.5", cache_dir=str(tmpdir))
            assert result == str(model_path)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_resolve_model_path_not_found_returns_original(self):
        """测试本地模型未找到时返回原始名称"""
        from retrievers.core.model_cache import _resolve_model_path

        tmpdir = _new_test_dir()
        try:
            result = _resolve_model_path("BAAI/bge-base-zh-v1.5", cache_dir=str(tmpdir))
            assert result == "BAAI/bge-base-zh-v1.5"
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestEmbeddingModelCache:
    """Embedding 模型缓存测试"""

    def test_get_embedding_model_singleton(self):
        """测试 Embedding 模型单例缓存"""
        from retrievers.core.model_cache import (
            _embedding_cache,
            clear_cache,
            get_embedding_model,
        )

        clear_cache()

        # 模拟 HuggingFaceEmbeddings
        mock_model = MagicMock()
        mock_model.model_name = "test-model"

        # Patch 在函数内部导入的位置
        with patch(
            "langchain_huggingface.HuggingFaceEmbeddings", return_value=mock_model
        ):
            # 首次获取
            model1 = get_embedding_model("test-model", "cpu")
            # 再次获取相同配置
            model2 = get_embedding_model("test-model", "cpu")

            # 应该是同一个实例
            assert model1 is model2
            # 缓存中应该只有一个条目
            assert len(_embedding_cache) == 1
            # 新的缓存键格式包含 cache_dir
            assert "test-model|cpu|default" in _embedding_cache

    def test_get_embedding_model_different_configs(self):
        """测试不同配置的 Embedding 模型各自缓存"""
        from retrievers.core.model_cache import (
            _embedding_cache,
            clear_cache,
            get_embedding_model,
        )

        clear_cache()

        mock_model_cpu = MagicMock()
        mock_model_cuda = MagicMock()

        with patch(
            "langchain_huggingface.HuggingFaceEmbeddings",
            side_effect=[mock_model_cpu, mock_model_cuda],
        ):
            model_cpu = get_embedding_model("test-model", "cpu")
            model_cuda = get_embedding_model("test-model", "cuda")

            # 不同配置应该是不同实例
            assert model_cpu is not model_cuda
            # 缓存中应该有两个条目
            assert len(_embedding_cache) == 2
            assert "test-model|cpu|default" in _embedding_cache
            assert "test-model|cuda|default" in _embedding_cache

    def test_get_embedding_model_cache_stats(self):
        """测试 Embedding 模型缓存统计"""
        from retrievers.core.model_cache import (
            clear_cache,
            get_cache_stats,
            get_embedding_model,
        )

        clear_cache()

        mock_model = MagicMock()

        with patch(
            "langchain_huggingface.HuggingFaceEmbeddings", return_value=mock_model
        ):
            # 首次获取（miss）
            get_embedding_model("test-model", "cpu")

            # 再次获取（hit）
            get_embedding_model("test-model", "cpu")

            stats = get_cache_stats()

            assert stats["embedding"]["hits"] == 1
            assert stats["embedding"]["misses"] == 1
            assert stats["embedding"]["cache_size"] == 1

    def test_get_embedding_model_custom_encode_kwargs(self):
        """测试自定义 encode_kwargs 参数"""
        from retrievers.core.model_cache import clear_cache, get_embedding_model

        clear_cache()

        mock_model = MagicMock()

        custom_kwargs = {"normalize_embeddings": False, "batch_size": 32}

        with patch(
            "langchain_huggingface.HuggingFaceEmbeddings", return_value=mock_model
        ) as mock_class:
            model = get_embedding_model("test-model", "cpu", custom_kwargs)

            # 验证调用参数
            mock_class.assert_called_once()
            call_kwargs = mock_class.call_args[1]
            assert call_kwargs["encode_kwargs"] == custom_kwargs

    def test_get_embedding_model_with_cache_dir(self):
        """测试使用 cache_dir 加载模型"""
        from retrievers.core.model_cache import (
            _embedding_cache,
            clear_cache,
            get_embedding_model,
        )

        clear_cache()

        mock_model = MagicMock()

        with patch(
            "langchain_huggingface.HuggingFaceEmbeddings", return_value=mock_model
        ) as mock_class:
            # 使用 cache_dir
            model = get_embedding_model("test-model", "cpu", cache_dir="/path/to/models")

            # 验证模型被加载
            assert model is mock_model
            # 验证缓存键包含 cache_dir
            assert "test-model|cpu|/path/to/models" in _embedding_cache

    def test_get_embedding_model_different_cache_dirs(self):
        """测试不同 cache_dir 的模型各自缓存"""
        from retrievers.core.model_cache import (
            _embedding_cache,
            clear_cache,
            get_embedding_model,
        )

        clear_cache()

        mock_model1 = MagicMock()
        mock_model2 = MagicMock()

        with patch(
            "langchain_huggingface.HuggingFaceEmbeddings",
            side_effect=[mock_model1, mock_model2],
        ):
            model1 = get_embedding_model("test-model", "cpu", cache_dir="/cache1")
            model2 = get_embedding_model("test-model", "cpu", cache_dir="/cache2")

            # 不同 cache_dir 应该是不同实例
            assert model1 is not model2
            # 缓存中应该有两个条目
            assert len(_embedding_cache) == 2


class TestRerankerModelCache:
    """Reranker 模型缓存测试"""

    def test_get_reranker_model_singleton(self):
        """测试 Reranker 模型单例缓存"""
        from retrievers.core.model_cache import (
            _reranker_cache,
            clear_cache,
            get_reranker_model,
        )

        clear_cache()

        mock_model = MagicMock()

        with patch(
            "sentence_transformers.CrossEncoder", return_value=mock_model
        ):
            # 首次获取
            model1 = get_reranker_model("test-reranker", "cpu", 512)
            # 再次获取相同配置
            model2 = get_reranker_model("test-reranker", "cpu", 512)

            # 应该是同一个实例
            assert model1 is model2
            # 缓存中应该只有一个条目
            assert len(_reranker_cache) == 1
            # 新的缓存键格式包含 cache_dir
            assert "test-reranker|cpu|512|default" in _reranker_cache

    def test_get_reranker_model_different_configs(self):
        """测试不同配置的 Reranker 模型各自缓存"""
        from retrievers.core.model_cache import (
            _reranker_cache,
            clear_cache,
            get_reranker_model,
        )

        clear_cache()

        mock_model_512 = MagicMock()
        mock_model_1024 = MagicMock()

        with patch(
            "sentence_transformers.CrossEncoder",
            side_effect=[mock_model_512, mock_model_1024],
        ):
            model_512 = get_reranker_model("test-reranker", "cpu", 512)
            model_1024 = get_reranker_model("test-reranker", "cpu", 1024)

            # 不同配置应该是不同实例
            assert model_512 is not model_1024
            # 缓存中应该有两个条目
            assert len(_reranker_cache) == 2

    def test_get_reranker_model_cache_stats(self):
        """测试 Reranker 模型缓存统计"""
        from retrievers.core.model_cache import (
            clear_cache,
            get_cache_stats,
            get_reranker_model,
        )

        clear_cache()

        mock_model = MagicMock()

        with patch(
            "sentence_transformers.CrossEncoder", return_value=mock_model
        ):
            # 首次获取（miss）
            get_reranker_model("test-reranker", "cpu")

            # 再次获取（hit）
            get_reranker_model("test-reranker", "cpu")

            stats = get_cache_stats()

            assert stats["reranker"]["hits"] == 1
            assert stats["reranker"]["misses"] == 1
            assert stats["reranker"]["cache_size"] == 1

    def test_get_reranker_model_with_cache_dir(self):
        """测试使用 cache_dir 加载 Reranker 模型"""
        from retrievers.core.model_cache import (
            _reranker_cache,
            clear_cache,
            get_reranker_model,
        )

        clear_cache()

        mock_model = MagicMock()

        with patch(
            "sentence_transformers.CrossEncoder", return_value=mock_model
        ):
            # 使用 cache_dir
            model = get_reranker_model(
                "test-reranker", "cpu", 512, cache_dir="/path/to/models"
            )

            # 验证模型被加载
            assert model is mock_model
            # 验证缓存键包含 cache_dir
            assert "test-reranker|cpu|512|/path/to/models" in _reranker_cache


class TestThreadSafety:
    """线程安全测试"""

    def test_concurrent_embedding_access(self):
        """测试并发访问 Embedding 缓存的线程安全性"""
        from retrievers.core.model_cache import (
            clear_cache,
            get_cache_stats,
            get_embedding_model,
        )

        clear_cache()

        mock_model = MagicMock()
        call_count = 0
        lock = threading.Lock()

        def create_model(*args, **kwargs):
            nonlocal call_count
            with lock:
                call_count += 1
            return mock_model

        with patch(
            "langchain_huggingface.HuggingFaceEmbeddings",
            side_effect=create_model,
        ):
            # 并发获取模型
            threads = []
            results = []

            def get_model():
                model = get_embedding_model("concurrent-test", "cpu")
                results.append(model)

            for _ in range(10):
                t = threading.Thread(target=get_model)
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            # 所有结果应该是同一个实例
            for result in results:
                assert result is mock_model

            # 模型应该只被创建一次
            assert call_count == 1

            stats = get_cache_stats()
            # 应该有 1 次 miss 和 9 次 hit
            assert stats["embedding"]["misses"] == 1
            assert stats["embedding"]["hits"] == 9


class TestWarmupFunctions:
    """预热功能测试"""

    def test_warmup_embedding_model(self):
        """测试 Embedding 模型预热"""
        from retrievers.core.model_cache import (
            clear_cache,
            get_cache_stats,
            warmup_embedding_model,
        )

        clear_cache()

        mock_model = MagicMock()
        mock_model.embed_query.return_value = [0.1, 0.2, 0.3]

        with patch(
            "langchain_huggingface.HuggingFaceEmbeddings", return_value=mock_model
        ):
            warmup_embedding_model("test-model", "cpu")

            # 验证预热时调用了 embed_query
            mock_model.embed_query.assert_called_once_with("warmup")

            # 验证模型已缓存
            stats = get_cache_stats()
            assert stats["embedding"]["cache_size"] == 1

    def test_warmup_embedding_model_with_cache_dir(self):
        """测试带 cache_dir 的 Embedding 模型预热"""
        from retrievers.core.model_cache import (
            clear_cache,
            get_cache_stats,
            warmup_embedding_model,
        )

        clear_cache()

        mock_model = MagicMock()
        mock_model.embed_query.return_value = [0.1, 0.2, 0.3]

        with patch(
            "langchain_huggingface.HuggingFaceEmbeddings", return_value=mock_model
        ):
            warmup_embedding_model("test-model", "cpu", cache_dir="/models")

            # 验证预热时调用了 embed_query
            mock_model.embed_query.assert_called_once_with("warmup")

            # 验证模型已缓存
            stats = get_cache_stats()
            assert stats["embedding"]["cache_size"] == 1

    def test_warmup_reranker_model(self):
        """测试 Reranker 模型预热"""
        from retrievers.core.model_cache import (
            clear_cache,
            get_cache_stats,
            warmup_reranker_model,
        )

        clear_cache()

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5]

        with patch(
            "sentence_transformers.CrossEncoder", return_value=mock_model
        ):
            warmup_reranker_model("test-reranker", "cpu")

            # 验证预热时调用了 predict
            mock_model.predict.assert_called_once()

            # 验证模型已缓存
            stats = get_cache_stats()
            assert stats["reranker"]["cache_size"] == 1

    def test_warmup_reranker_model_with_cache_dir(self):
        """测试带 cache_dir 的 Reranker 模型预热"""
        from retrievers.core.model_cache import (
            clear_cache,
            get_cache_stats,
            warmup_reranker_model,
        )

        clear_cache()

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5]

        with patch(
            "sentence_transformers.CrossEncoder", return_value=mock_model
        ):
            warmup_reranker_model("test-reranker", "cpu", cache_dir="/models")

            # 验证预热时调用了 predict
            mock_model.predict.assert_called_once()

            # 验证模型已缓存
            stats = get_cache_stats()
            assert stats["reranker"]["cache_size"] == 1


class TestModuleExports:
    """模块导出测试"""

    def test_exports_from_init(self):
        """测试从 __init__.py 正确导出所有函数"""
        from retrievers import (
            clear_cache,
            CrossEncoderReranker,
            CrossEncoderRerankerConfig,
            EmbeddingRetriever,
            EmbeddingRetrieverConfig,
            get_cache_stats,
            get_embedding_model,
            get_reranker_model,
            WeightedFusionRetriever,
        )

        # 验证所有导出都是可调用的
        assert callable(get_embedding_model)
        assert callable(get_reranker_model)
        assert callable(get_cache_stats)
        assert callable(clear_cache)


class TestConfigWithCacheDir:
    """配置类 cache_dir 支持测试"""

    def test_embedding_retriever_config_cache_dir_from_profile(self):
        """测试 EmbeddingRetrieverConfig 从 profile 读取 cache_dir"""
        from retrievers.core.embedding_retriever import EmbeddingRetrieverConfig
        from domain_profile import EmbeddingProfile

        profile = EmbeddingProfile(
            model="test-model",
            device="cpu",
            top_k=4,
            cache_dir="/custom/cache",
        )

        config = EmbeddingRetrieverConfig.from_profile(profile)
        assert config.cache_dir == "/custom/cache"

    def test_embedding_retriever_config_cache_dir_from_env(self):
        """测试 EmbeddingRetrieverConfig 从环境变量读取 cache_dir"""
        from retrievers.core.embedding_retriever import EmbeddingRetrieverConfig

        # 设置环境变量
        os.environ["WORKFLOW_EMBEDDING_CACHE_DIR"] = "/env/cache"

        try:
            config = EmbeddingRetrieverConfig.from_env()
            assert config.cache_dir == "/env/cache"
        finally:
            del os.environ["WORKFLOW_EMBEDDING_CACHE_DIR"]

    def test_reranker_config_cache_dir_from_profile(self):
        """测试 CrossEncoderRerankerConfig 从 profile 读取 cache_dir"""
        from retrievers.core.cross_encoder_reranker import CrossEncoderRerankerConfig
        from domain_profile import RerankerProfile

        profile = RerankerProfile(
            model="test-reranker",
            device="cpu",
            top_k=4,
            cache_dir="/custom/cache",
        )

        config = CrossEncoderRerankerConfig.from_profile(profile)
        assert config.cache_dir == "/custom/cache"

    def test_reranker_config_cache_dir_from_env(self):
        """测试 CrossEncoderRerankerConfig 从环境变量读取 cache_dir"""
        from retrievers.core.cross_encoder_reranker import CrossEncoderRerankerConfig

        # 设置环境变量
        os.environ["WORKFLOW_RERANKER_CACHE_DIR"] = "/env/cache"

        try:
            config = CrossEncoderRerankerConfig.from_env()
            assert config.cache_dir == "/env/cache"
        finally:
            del os.environ["WORKFLOW_RERANKER_CACHE_DIR"]
