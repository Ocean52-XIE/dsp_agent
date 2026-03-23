# -*- coding: utf-8 -*-
"""Shared retriever components."""

from .cross_encoder_reranker import CrossEncoderReranker, CrossEncoderRerankerConfig
from .embedding_retriever import EmbeddingRetriever, EmbeddingRetrieverConfig
from .model_cache import clear_cache, get_cache_stats, get_embedding_model, get_reranker_model
from .weighted_fusion import WeightedFusionRetriever

__all__ = [
    "EmbeddingRetriever",
    "EmbeddingRetrieverConfig",
    "CrossEncoderReranker",
    "CrossEncoderRerankerConfig",
    "WeightedFusionRetriever",
    "get_embedding_model",
    "get_reranker_model",
    "get_cache_stats",
    "clear_cache",
]
