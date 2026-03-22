# -*- coding: utf-8 -*-
"""
该模块实现检索能力，负责召回、融合、重排或结果标准化处理。

模块组成：
- EmbeddingRetriever: 向量检索器，基于语义相似度召回
- WeightedFusionRetriever: 加权融合检索器，基于 RRF 融合多路召回
- CrossEncoderReranker: Cross-Encoder 重排器，对召回结果精排
- model_cache: 模型单例缓存，避免重复加载 Embedding/Reranker 模型
"""
from __future__ import annotations

# 使用延迟导入避免循环依赖
# 当 retrievers 模块被 workflow 导入时，
# embedding_retriever 又导入了 workflow.common，导致循环

__all__ = [
    "EmbeddingRetriever",
    "EmbeddingRetrieverConfig",
    "WeightedFusionRetriever",
    "CrossEncoderReranker",
    "CrossEncoderRerankerConfig",
    # 模型缓存
    "get_embedding_model",
    "get_reranker_model",
    "get_cache_stats",
    "clear_cache",
]


def __getattr__(name: str):
    """延迟导入，避免循环依赖"""
    if name == "EmbeddingRetriever":
        from retrievers.embedding_retriever import EmbeddingRetriever
        return EmbeddingRetriever
    elif name == "EmbeddingRetrieverConfig":
        from retrievers.embedding_retriever import EmbeddingRetrieverConfig
        return EmbeddingRetrieverConfig
    elif name == "WeightedFusionRetriever":
        from retrievers.weighted_fusion import WeightedFusionRetriever
        return WeightedFusionRetriever
    elif name == "CrossEncoderReranker":
        from retrievers.cross_encoder_reranker import CrossEncoderReranker
        return CrossEncoderReranker
    elif name == "CrossEncoderRerankerConfig":
        from retrievers.cross_encoder_reranker import CrossEncoderRerankerConfig
        return CrossEncoderRerankerConfig
    # 模型缓存
    elif name == "get_embedding_model":
        from retrievers.model_cache import get_embedding_model
        return get_embedding_model
    elif name == "get_reranker_model":
        from retrievers.model_cache import get_reranker_model
        return get_reranker_model
    elif name == "get_cache_stats":
        from retrievers.model_cache import get_cache_stats
        return get_cache_stats
    elif name == "clear_cache":
        from retrievers.model_cache import clear_cache
        return clear_cache
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
