# -*- coding: utf-8 -*-
"""
该模块实现检索能力，负责召回、融合、重排或结果标准化处理。

模块组成：
- EmbeddingRetriever: 向量检索器，基于语义相似度召回
- WeightedFusionRetriever: 加权融合检索器，基于 RRF 融合多路召回
- CrossEncoderReranker: Cross-Encoder 重排器，对召回结果精排
"""
from workflow.retrievers.embedding_retriever import EmbeddingRetriever, EmbeddingRetrieverConfig
from workflow.retrievers.weighted_fusion import WeightedFusionRetriever
from workflow.retrievers.cross_encoder_reranker import (
    CrossEncoderReranker,
    CrossEncoderRerankerConfig,
)

__all__ = [
    "EmbeddingRetriever",
    "EmbeddingRetrieverConfig",
    "WeightedFusionRetriever",
    "CrossEncoderReranker",
    "CrossEncoderRerankerConfig",
]
