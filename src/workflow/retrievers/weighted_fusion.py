# -*- coding: utf-8 -*-
"""
该模块实现检索能力，负责召回、融合或结果标准化处理。
"""
from __future__ import annotations

from hashlib import md5
from typing import Any

from langchain_core.documents import Document


class WeightedFusionRetriever:
    """
    定义`WeightedFusionRetriever`，用于封装相关数据结构与处理行为。
    """

    def __init__(
        self,
        *,
        retrievers: list[Any],
        weights: list[float],
        c: int = 60,
        id_key: str | None = None,
    ) -> None:
        """
        内部辅助函数，负责` init  ` 相关处理。
        
        参数:
            self: 当前对象实例。
        
        返回:
            无返回值。
        """
        if not retrievers:
            raise ValueError("retrievers_required")
        if len(retrievers) != len(weights):
            raise ValueError("retriever_weight_size_mismatch")
        self.retrievers = retrievers
        self.weights = self._normalize_weights(weights)
        self.c = max(int(c), 1)
        self.id_key = id_key

    def invoke(self, input: str, config: Any | None = None, **kwargs: Any) -> list[Document]:
        """
        执行`invoke` 相关处理逻辑。
        
        参数:
            self: 当前对象实例。
            input: 输入参数，用于控制当前处理逻辑。
            config: 输入参数，用于控制当前处理逻辑。
            **kwargs: 列表参数，用于承载批量输入数据。
        
        返回:
            返回类型为 `list[Document]` 的处理结果。
        """
        scored: dict[str, float] = {}
        doc_by_key: dict[str, Document] = {}

        for retriever, weight in zip(self.retrievers, self.weights):
            docs = self._invoke_one(retriever=retriever, input=input, config=config, **kwargs)
            for rank, doc in enumerate(docs, start=1):
                key = self._doc_key(doc)
                doc_by_key.setdefault(key, doc)
                scored[key] = scored.get(key, 0.0) + weight * (1.0 / (self.c + rank))

        ordered_keys = sorted(scored.keys(), key=lambda key: scored[key], reverse=True)
        return [doc_by_key[key] for key in ordered_keys]

    def _invoke_one(self, *, retriever: Any, input: str, config: Any | None, **kwargs: Any) -> list[Document]:
        """
        内部辅助函数，负责`invoke one` 相关处理。
        
        参数:
            self: 当前对象实例。
            **kwargs: 列表参数，用于承载批量输入数据。
        
        返回:
            返回类型为 `list[Document]` 的处理结果。
        """
        try:
            result = retriever.invoke(input, config=config, **kwargs)
        except TypeError:
            result = retriever.invoke(input, **kwargs)
        if not isinstance(result, list):
            return []
        return [doc for doc in result if isinstance(doc, Document)]

    def _doc_key(self, doc: Document) -> str:
        """
        内部辅助函数，负责`doc key` 相关处理。
        
        参数:
            self: 当前对象实例。
            doc: 输入参数，用于控制当前处理逻辑。
        
        返回:
            返回类型为 `str` 的处理结果。
        """
        metadata = doc.metadata or {}
        if self.id_key:
            raw = metadata.get(self.id_key)
            if raw is not None:
                return f"id:{self.id_key}:{raw}"

        payload = f"{doc.page_content}|{sorted((str(k), str(v)) for k, v in metadata.items())}"
        digest = md5(payload.encode("utf-8")).hexdigest()
        return f"hash:{digest}"

    def _normalize_weights(self, weights: list[float]) -> list[float]:
        """
        内部辅助函数，负责`normalize weights` 相关处理。
        
        参数:
            self: 当前对象实例。
            weights: 列表参数，用于承载批量输入数据。
        
        返回:
            返回类型为 `list[float]` 的处理结果。
        """
        clipped = [max(float(weight), 0.0) for weight in weights]
        total = sum(clipped)
        if total <= 0:
            return [1.0 / len(clipped)] * len(clipped)
        return [value / total for value in clipped]
