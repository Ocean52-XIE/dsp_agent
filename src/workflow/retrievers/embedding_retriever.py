# -*- coding: utf-8 -*-
"""向量检索器模块，支持多种 Embedding 模型和向量存储。

该模块提供基于语义向量的检索能力，可与现有的 BM25/TFIDF 检索形成混合检索架构。
主要特性：
- 支持多种 Embedding 模型（BGE、Sentence-Transformers 等）
- 支持多种向量存储（Chroma、FAISS）
- 提供 LangChain Runnable 接口，便于与 LangGraph 集成
- 支持向量索引持久化，避免重复计算
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from workflow.common.func_utils import env_bool, env_int
from workflow.common.runtime_logging import get_file_logger
from workflow.common.domain_profile import EmbeddingProfile


# 向量检索默认配置
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-base-zh-v1.5"
DEFAULT_EMBEDDING_DEVICE = "cpu"
DEFAULT_TOP_K = 4


@dataclass
class EmbeddingRetrieverConfig:
    """向量检索器配置类，包含模型参数、存储参数和检索参数。

    属性:
        model_name: Embedding 模型名称，默认使用 BGE 中文模型
        device: 运行设备，cpu 或 cuda
        encode_kwargs: 编码额外参数，如 normalize_embeddings
        persist_directory: 向量存储持久化路径
        collection_name: Chroma collection 名称
        top_k: 默认返回结果数量
    """

    model_name: str = DEFAULT_EMBEDDING_MODEL
    device: str = DEFAULT_EMBEDDING_DEVICE
    encode_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"normalize_embeddings": True}
    )
    persist_directory: str | None = None
    collection_name: str = "default"
    top_k: int = DEFAULT_TOP_K
    # 配置文件来源标记
    _config_source: str = "default"

    @classmethod
    def from_profile(
        cls,
        profile: EmbeddingProfile,
        collection_name: str = "default",
        persist_root: str | None = None,
    ) -> "EmbeddingRetrieverConfig":
        """从 profile 配置创建

        支持的配置来源 (优先级从高到低):
            1. 环境变量 (WORKFLOW_EMBEDDING_*)
            2. profile 配置
            3. 默认值

        参数:
            profile: 领域配置中的 EmbeddingProfile
            collection_name: Collection 名称，用于区分不同知识库
            persist_root: 持久化根目录，默认从 profile 读取或使用 .vectorstore

        返回:
            配置实例
        """
        # 环境变量优先级最高
        model_name = os.getenv("WORKFLOW_EMBEDDING_MODEL", profile.model)
        device = os.getenv("WORKFLOW_EMBEDDING_DEVICE", profile.device)
        top_k = env_int("WORKFLOW_EMBEDDING_TOP_K", profile.top_k, minimum=1)

        # 确定持久化目录
        root = persist_root or profile.persist_root
        if root:
            root_path = Path(root)
            if not root_path.is_absolute():
                root_path = Path.cwd() / root_path
            persist_dir = str(root_path / collection_name)
        else:
            persist_dir = str(Path.cwd() / ".vectorstore" / collection_name)

        return cls(
            model_name=model_name,
            device=device,
            top_k=top_k,
            persist_directory=persist_dir,
            collection_name=collection_name,
            _config_source="profile",
        )

    @classmethod
    def from_env(cls, collection_name: str = "default", persist_root: str | None = None) -> "EmbeddingRetrieverConfig":
        """从环境变量加载配置

        支持的环境变量:
            WORKFLOW_EMBEDDING_MODEL: Embedding 模型名称
            WORKFLOW_EMBEDDING_DEVICE: 运行设备 (cpu/cuda)
            WORKFLOW_EMBEDDING_TOP_K: 默认返回结果数

        参数:
            collection_name: Collection 名称，用于区分不同知识库
            persist_root: 持久化根目录，默认 .vectorstore

        返回:
            配置实例
        """
        model_name = os.getenv("WORKFLOW_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        device = os.getenv("WORKFLOW_EMBEDDING_DEVICE", DEFAULT_EMBEDDING_DEVICE)
        top_k = env_int("WORKFLOW_EMBEDDING_TOP_K", DEFAULT_TOP_K, minimum=1)

        # 确定持久化目录
        if persist_root is None:
            root = Path.cwd()
        else:
            root = Path(persist_root)

        persist_dir = str(root / ".vectorstore" / collection_name)

        return cls(
            model_name=model_name,
            device=device,
            top_k=top_k,
            persist_directory=persist_dir,
            collection_name=collection_name,
            _config_source="env",
        )


class EmbeddingRetriever:
    """基于 Embedding 的向量检索器

    该类封装了向量检索的完整流程：
    1. 初始化 Embedding 模型
    2. 构建或加载向量索引
    3. 执行相似度检索

    支持 LangChain Runnable 接口，可直接用于 LangGraph 节点。

    示例:
        retriever = EmbeddingRetriever(project_root=Path("."))
        retriever.initialize(documents)
        results = retriever.search("查询内容", top_k=4)
    """

    def __init__(
        self,
        *,
        project_root: Path,
        config: EmbeddingRetrieverConfig | None = None,
    ) -> None:
        """初始化向量检索器

        参数:
            project_root: 项目根目录，用于日志和持久化路径
            config: 检索器配置，默认从环境变量加载
        """
        self.project_root = project_root
        self.config = config or EmbeddingRetrieverConfig.from_env(
            collection_name=self.__class__.__name__
        )
        self._logger = get_file_logger(project_root=project_root)

        self._embeddings: HuggingFaceEmbeddings | None = None
        self._vectorstore: Chroma | None = None
        self._initialized = False
        self._index_stats: dict[str, Any] = {}

    @property
    def is_initialized(self) -> bool:
        """检查是否已初始化向量索引"""
        return self._initialized

    @property
    def index_stats(self) -> dict[str, Any]:
        """获取索引统计信息"""
        return self._index_stats.copy()

    def initialize(self, documents: list[Document]) -> dict[str, Any]:
        """初始化向量存储并索引文档

        如果向量存储已存在，则增量添加新文档。

        参数:
            documents: LangChain Document 列表，每个 Document 包含 page_content 和 metadata

        返回:
            索引构建统计信息
        """
        if not documents:
            self._logger.warning("workflow.embedding_retriever.empty_docs", message="No documents to index")
            return {"doc_count": 0, "status": "skipped"}

        start_time = os.times().elapsed if hasattr(os.times(), 'elapsed') else 0

        # 初始化 Embedding 模型
        self._embeddings = HuggingFaceEmbeddings(
            model_name=self.config.model_name,
            model_kwargs={"device": self.config.device},
            encode_kwargs=self.config.encode_kwargs,
        )

        # 构建或加载向量存储
        collection_name = self.config.collection_name
        persist_dir = self.config.persist_directory

        if persist_dir and Path(persist_dir).exists():
            # 加载已有向量存储并增量添加
            self._logger.info(
                "workflow.embedding_retriever.loading_existing",
                persist_dir=persist_dir,
                collection=collection_name,
            )
            self._vectorstore = Chroma(
                client=None,
                collection_name=collection_name,
                embedding_function=self._embeddings,
                persist_directory=persist_dir,
            )
            # 增量添加文档
            self._vectorstore.add_documents(documents)
        else:
            # 创建新的向量存储
            self._logger.info(
                "workflow.embedding_retriever.creating_new",
                model=self.config.model_name,
                doc_count=len(documents),
                persist_dir=persist_dir or "memory",
            )
            self._vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self._embeddings,
                collection_name=collection_name,
                persist_directory=persist_dir,
            )

        # 持久化
        if persist_dir:
            self._vectorstore.persist()

        self._initialized = True
        self._index_stats = {
            "model": self.config.model_name,
            "device": self.config.device,
            "doc_count": len(documents),
            "collection": collection_name,
            "persist_dir": persist_dir,
        }

        self._logger.info(
            "workflow.embedding_retriever.initialized",
            **self._index_stats,
        )

        return self._index_stats

    def as_langchain_runnable(self) -> RunnableLambda:
        """转换为 LangChain Runnable 接口

        返回的 Runnable 可直接用于 LangGraph 节点或 LangChain LCEL 链。

        返回:
            封装检索逻辑的 RunnableLambda
        """
        def retrieve(input_data: str | dict[str, Any]) -> list[Document]:
            # 支持字符串或字典输入
            query = input_data if isinstance(input_data, str) else input_data.get("query", "")
            if not query:
                return []
            return self.search(query)

        return RunnableLambda(retrieve)

    def search(
        self,
        query: str,
        top_k: int | None = None,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[Document]:
        """执行向量相似度检索

        参数:
            query: 查询字符串
            top_k: 返回结果数量，默认使用配置值
            filter_metadata: 元数据过滤条件

        返回:
            按相似度排序的 Document 列表
        """
        if not self._initialized:
            self._logger.warning(
                "workflow.embedding_retriever.not_initialized",
                message="Retriever not initialized, returning empty results"
            )
            return []

        if not query or not query.strip():
            return []

        k = top_k or self.config.top_k
        filter_condition = None

        # 构建过滤条件（Chroma 格式）
        if filter_metadata:
            filter_condition = {
                f"$and": [
                    {"metadata.{key}": {"$eq": value}}
                    for key, value in filter_metadata.items()
                ]
            }

        try:
            results = self._vectorstore.similarity_search(
                query,
                k=k,
                filter=filter_condition,
            )
            return results
        except Exception as e:
            self._logger.error(
                "workflow.embedding_retriever.search_error",
                error=str(e),
                query=query[:100],
            )
            return []

    def search_with_scores(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[tuple[Document, float]]:
        """执行向量检索并返回相似度分数

        参数:
            query: 查询字符串
            top_k: 返回结果数量

        返回:
            (Document, score) 元组列表，按相似度降序排列
        """
        if not self._initialized:
            return []

        k = top_k or self.config.top_k

        try:
            results = self._vectorstore.similarity_search_with_score(query, k=k)
            return results
        except Exception as e:
            self._logger.error(
                "workflow.embedding_retriever.search_with_scores_error",
                error=str(e),
            )
            return []

    def get_relevant_documents(
        self,
        query: str,
        *,
        k: int | None = None,
    ) -> list[Document]:
        """LangChain BaseRetriever 兼容接口

        该方法实现了 langchain_core.retrievers.BaseRetriever 的核心接口，
        便于与 LangChain 的其他组件集成。

        参数:
            query: 查询字符串
            k: 返回结果数量

        返回:
            相关文档列表
        """
        return self.search(query, top_k=k or self.config.top_k)

    def reset(self) -> None:
        """重置检索器状态，清理内存中的向量索引

        注意：这不会删除持久化的向量存储。
        """
        self._vectorstore = None
        self._embeddings = None
        self._initialized = False
        self._index_stats = {}
        self._logger.info("workflow.embedding_retriever.reset")

    def delete_collection(self) -> bool:
        """删除持久化的向量存储

        返回:
            是否成功删除
        """
        if not self.config.persist_directory:
            return False

        persist_path = Path(self.config.persist_directory)
        if persist_path.exists():
            import shutil

            try:
                shutil.rmtree(persist_path)
                self._logger.info(
                    "workflow.embedding_retriever.collection_deleted",
                    path=str(persist_path),
                )
                self.reset()
                return True
            except Exception as e:
                self._logger.error(
                    "workflow.embedding_retriever.delete_error",
                    error=str(e),
                )
                return False
        return False
