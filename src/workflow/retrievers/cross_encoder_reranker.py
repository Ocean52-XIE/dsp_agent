# -*- coding: utf-8 -*-
"""Cross-Encoder 重排器模块，用于对检索候选集进行精排。

该模块基于 sentence-transformers 的 CrossEncoder 实现重排能力：
- 支持中英文模型，默认使用 BAAI/bge-reranker-base
- 支持批处理优化性能
- 支持配置候选集大小和最终输出大小
- 完全可选，默认关闭

Cross-Encoder 与 Bi-Encoder 的区别：
- Bi-Encoder（如 EmbeddingRetriever）分别编码 query 和 document，通过向量相似度检索
- Cross-Encoder 同时编码 query-document 对，能捕获更深层的语义交互，精度更高但速度较慢

典型使用场景：
1. 召回阶段：BM25 + Embedding + Lexical 召回候选集（candidate_top_k 条）
2. 重排阶段：Cross-Encoder 对候选集精排，返回 top_k 条结果
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

from langchain_core.documents import Document

from workflow.common.func_utils import env_bool, env_int
from workflow.common.runtime_logging import get_file_logger


# 重排器默认配置
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"
DEFAULT_RERANKER_DEVICE = "cpu"
DEFAULT_RERANKER_TOP_K = 4
DEFAULT_CANDIDATE_TOP_K = 20
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_LENGTH = 512


@dataclass
class CrossEncoderRerankerConfig:
    """Cross-Encoder 重排器配置类。

    属性:
        model_name: Cross-Encoder 模型名称，推荐使用 BGE 中文重排模型
        device: 运行设备，cpu 或 cuda
        top_k: 重排后返回的结果数量
        candidate_top_k: 参与重排的候选集大小
        batch_size: 批处理大小，影响重排性能
        max_length: 最大序列长度，超过会被截断
        _config_source: 配置来源标记，用于调试
    """

    model_name: str = DEFAULT_RERANKER_MODEL
    device: str = DEFAULT_RERANKER_DEVICE
    top_k: int = DEFAULT_RERANKER_TOP_K
    candidate_top_k: int = DEFAULT_CANDIDATE_TOP_K
    batch_size: int = DEFAULT_BATCH_SIZE
    max_length: int = DEFAULT_MAX_LENGTH
    _config_source: str = "default"

    @classmethod
    def from_profile(
        cls,
        profile: Any,  # RerankerProfile，使用 Any 避免循环导入
    ) -> "CrossEncoderRerankerConfig":
        """从 profile 配置创建，支持环境变量覆盖。

        配置优先级 (从高到低):
            1. 环境变量 (WORKFLOW_RERANKER_*)
            2. profile 配置
            3. 默认值

        参数:
            profile: 领域配置中的 RerankerProfile

        返回:
            CrossEncoderRerankerConfig 实例
        """
        # 环境变量优先级最高
        model_name = os.getenv("WORKFLOW_RERANKER_MODEL", profile.model)
        device = os.getenv("WORKFLOW_RERANKER_DEVICE", profile.device)
        top_k = env_int("WORKFLOW_RERANKER_TOP_K", profile.top_k, minimum=1)
        candidate_top_k = env_int("WORKFLOW_RERANKER_CANDIDATE_TOP_K", profile.candidate_top_k, minimum=1)
        batch_size = env_int("WORKFLOW_RERANKER_BATCH_SIZE", profile.batch_size, minimum=1)
        max_length = env_int("WORKFLOW_RERANKER_MAX_LENGTH", profile.max_length, minimum=64)

        return cls(
            model_name=model_name,
            device=device,
            top_k=top_k,
            candidate_top_k=candidate_top_k,
            batch_size=batch_size,
            max_length=max_length,
            _config_source="profile",
        )

    @classmethod
    def from_env(cls) -> "CrossEncoderRerankerConfig":
        """从环境变量加载配置。

        返回:
            CrossEncoderRerankerConfig 实例
        """
        return cls(
            model_name=os.getenv("WORKFLOW_RERANKER_MODEL", DEFAULT_RERANKER_MODEL),
            device=os.getenv("WORKFLOW_RERANKER_DEVICE", DEFAULT_RERANKER_DEVICE),
            top_k=env_int("WORKFLOW_RERANKER_TOP_K", DEFAULT_RERANKER_TOP_K, minimum=1),
            candidate_top_k=env_int("WORKFLOW_RERANKER_CANDIDATE_TOP_K", DEFAULT_CANDIDATE_TOP_K, minimum=1),
            batch_size=env_int("WORKFLOW_RERANKER_BATCH_SIZE", DEFAULT_BATCH_SIZE, minimum=1),
            max_length=env_int("WORKFLOW_RERANKER_MAX_LENGTH", DEFAULT_MAX_LENGTH, minimum=64),
            _config_source="env",
        )


class CrossEncoderReranker:
    """基于 Cross-Encoder 的重排器。

    该类封装了重排的完整流程：
    1. 初始化 Cross-Encoder 模型
    2. 构建查询-文档对
    3. 批量计算相关性分数
    4. 按分数重新排序

    使用示例:
        ```python
        reranker = CrossEncoderReranker(project_root=Path("."))
        reranker.initialize()
        results = reranker.rerank(query="查询内容", candidates=hits)
        ```

    属性:
        project_root: 项目根目录
        config: 重排器配置
        is_initialized: 模型是否已初始化
    """

    def __init__(
        self,
        *,
        project_root: Path,
        config: CrossEncoderRerankerConfig | None = None,
    ) -> None:
        """初始化重排器。

        参数:
            project_root: 项目根目录，用于日志记录
            config: 重排器配置，为 None 时从环境变量加载
        """
        self.project_root = project_root
        self.config = config or CrossEncoderRerankerConfig.from_env()
        self._logger = get_file_logger(project_root=project_root)

        self._model: Any = None  # CrossEncoder 实例，延迟加载
        self._initialized = False
        self._rerank_stats: dict[str, Any] = {}

    @property
    def is_initialized(self) -> bool:
        """检查模型是否已初始化。"""
        return self._initialized

    def initialize(self) -> dict[str, Any]:
        """初始化 Cross-Encoder 模型。

        该方法会加载模型到内存，首次调用可能需要下载模型文件。

        返回:
            初始化统计信息，包含模型名称、设备、延迟等

        异常:
            ImportError: 如果未安装 sentence-transformers
            Exception: 模型加载失败
        """
        if self._initialized:
            return self._rerank_stats

        start_time = perf_counter()

        try:
            # 延迟导入，避免未安装时启动失败
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                model_name=self.config.model_name,
                device=self.config.device,
                max_length=self.config.max_length,
            )
            self._initialized = True

            self._rerank_stats = {
                "model": self.config.model_name,
                "device": self.config.device,
                "top_k": self.config.top_k,
                "candidate_top_k": self.config.candidate_top_k,
                "batch_size": self.config.batch_size,
                "max_length": self.config.max_length,
                "init_latency_ms": round((perf_counter() - start_time) * 1000, 2),
            }

            self._logger.info(
                "workflow.cross_encoder_reranker.initialized",
                **self._rerank_stats,
            )

            # 控制台输出，方便验证模型加载状态
            print(f"[Reranker] 模型加载完成: {self.config.model_name}")
            print(f"[Reranker] 设备: {self.config.device}, 延迟: {self._rerank_stats['init_latency_ms']}ms")

            return self._rerank_stats

        except ImportError as e:
            self._logger.error(
                "workflow.cross_encoder_reranker.import_error",
                error=str(e),
                message="请安装 sentence-transformers: pip install sentence-transformers",
            )
            raise
        except Exception as e:
            self._logger.error(
                "workflow.cross_encoder_reranker.init_error",
                error=str(e),
                model=self.config.model_name,
            )
            raise

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int | None = None,
        content_key: str = "content",
    ) -> list[dict[str, Any]]:
        """对候选集进行重排。

        该方法会：
        1. 截取候选集到 candidate_top_k 大小
        2. 构建 query-document 对
        3. 批量计算相关性分数
        4. 按分数降序排列
        5. 返回 top_k 个结果

        参数:
            query: 查询字符串
            candidates: 候选结果列表，每个元素需包含 content_key 指定的字段
            top_k: 返回结果数量，默认使用配置值
            content_key: 候选结果中内容的字段名，默认为 "content"

        返回:
            重排后的候选列表，每个元素新增以下字段：
            - rerank_score: Cross-Encoder 计算的相关性分数
            - original_rank: 重排前的排名（从 1 开始）
        """
        if not self._initialized or not self._model:
            self._logger.warning(
                "workflow.cross_encoder_reranker.not_initialized",
                message="Reranker not initialized, returning original candidates",
            )
            return candidates[: top_k or self.config.top_k]

        if not query or not candidates:
            return []

        k = top_k or self.config.top_k
        candidate_k = min(len(candidates), self.config.candidate_top_k)

        # 截取候选集
        working_candidates = list(candidates[:candidate_k])

        # 构建查询-文档对
        pairs = [
            (query, str(c.get(content_key, ""))[: self.config.max_length])
            for c in working_candidates
        ]

        start_time = perf_counter()

        try:
            # 批量计算分数
            scores = self._model.predict(
                pairs,
                batch_size=self.config.batch_size,
                show_progress_bar=False,
            )

            # 添加重排分数并排序
            for i, score in enumerate(scores):
                working_candidates[i]["rerank_score"] = float(score)
                working_candidates[i]["original_rank"] = i + 1

            # 按重排分数降序排列
            reranked = sorted(
                working_candidates,
                key=lambda x: float(x.get("rerank_score", 0.0)),
                reverse=True,
            )

            latency_ms = round((perf_counter() - start_time) * 1000, 2)

            self._logger.info(
                "workflow.cross_encoder_reranker.rerank_complete",
                query_preview=query[:50],
                candidate_count=len(working_candidates),
                top_k=k,
                latency_ms=latency_ms,
            )

            # 控制台输出，方便验证重排执行
            print(f"[Reranker] 重排完成: {len(working_candidates)} -> {k} 条, 耗时 {latency_ms}ms")

            return reranked[:k]

        except Exception as e:
            self._logger.error(
                "workflow.cross_encoder_reranker.rerank_error",
                error=str(e),
                query=query[:100],
            )
            # 出错时返回原始候选集
            return working_candidates[:k]

    def rerank_documents(
        self,
        query: str,
        documents: list[Document],
        top_k: int | None = None,
    ) -> list[tuple[Document, float]]:
        """对 LangChain Document 列表进行重排。

        该方法是 rerank() 的 LangChain 兼容版本，用于与现有检索器集成。

        参数:
            query: 查询字符串
            documents: LangChain Document 列表
            top_k: 返回结果数量

        返回:
            (Document, score) 元组列表，按分数降序排列
        """
        if not self._initialized or not self._model:
            return [(doc, 0.0) for doc in documents[: top_k or self.config.top_k]]

        if not query or not documents:
            return []

        k = top_k or self.config.top_k
        candidate_k = min(len(documents), self.config.candidate_top_k)

        working_docs = list(documents[:candidate_k])

        pairs = [
            (query, doc.page_content[: self.config.max_length])
            for doc in working_docs
        ]

        try:
            scores = self._model.predict(
                pairs,
                batch_size=self.config.batch_size,
                show_progress_bar=False,
            )

            # 按分数排序
            scored_docs = list(zip(working_docs, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            return scored_docs[:k]

        except Exception as e:
            self._logger.error(
                "workflow.cross_encoder_reranker.rerank_documents_error",
                error=str(e),
            )
            return [(doc, 0.0) for doc in working_docs[:k]]

    def reset(self) -> None:
        """重置重排器状态，释放模型内存。

        调用此方法后，需要重新调用 initialize() 才能使用重排功能。
        """
        self._model = None
        self._initialized = False
        self._rerank_stats = {}
        self._logger.info("workflow.cross_encoder_reranker.reset")
