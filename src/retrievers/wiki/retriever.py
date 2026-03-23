# -*- coding: utf-8 -*-
"""
该模块实现工作流节点`retrieve_wiki` 的处理逻辑，负责读取状态并输出增量结果。
"""
from __future__ import annotations

"""Wiki retriever backed by LangChain retriever wrappers.

采用精简的三级召回策略：
- `BM25Retriever` 用于词项精确匹配
- `EmbeddingRetriever` 用于语义匹配（同义词理解）
- Lexical 词法覆盖率作为补充

这种设计兼顾了：
1. 精确匹配（关键词必须出现）
2. 语义理解（同义词、近义词）
3. 覆盖率补充
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
import json
import logging
import os
import re
from typing import Any

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from retrievers.core.embedding_retriever import EmbeddingRetriever, EmbeddingRetrieverConfig
from log import get_file_logger
from common.func_utils import env_bool, env_float, env_int
from domain_profile import EmbeddingProfile, RerankerProfile

# 模块级日志器
logger = logging.getLogger(__name__)


@dataclass
class WikiChunk:
    """Wiki 文档块结构

    Attributes:
        chunk_id: 块唯一标识
        source_path: 源文件路径
        title: 文档标题（H1）
        section: 当前章节名称
        chunk_type: 块类型（paragraph, code, table, list, steps, mixed）
        content: 块内容
        normalized_text: 归一化文本（用于检索）
        hierarchy: 标题层级链（如 ["广告引擎", "在线投放链路", "召回阶段"]）
        start_line: 起始行号
        end_line: 结束行号
        token_count: 近似 token 数量
    """
    chunk_id: int
    source_path: Path
    title: str
    section: str
    chunk_type: str
    content: str
    normalized_text: str
    hierarchy: list[str] = field(default_factory=list)
    start_line: int = 1
    end_line: int = 1
    token_count: int = 0


@dataclass
class HybridScoreWeights:
    """
    混合检索评分权重配置。

    采用 RRF (Reciprocal Rank Fusion) 归一化策略，将不同检索路径的分数
    统一映射到 [0, 1] 范围，确保加权求和的有效性。

    三级召回策略：
    - bm25: 词项精确匹配（关键词必须出现）
    - embedding: 语义匹配（同义词理解）
    - lexical: 查询词覆盖率补充

    RRF 公式: score = k / (rank + k)，其中 k=60 是业界验证的标准参数
    参考: Cormack, Clarke & Buettcher (2009) - Reciprocal Rank Fusion
    """
    bm25: float = 0.30  # 词项精确匹配权重
    embedding: float = 0.50  # 语义匹配权重（提高以增强语义理解）
    lexical: float = 0.20  # 词法覆盖率权重
    source: str = "default"  # 配置来源

    # RRF 常量：k=60 是信息检索领域验证的标准参数
    # 更大的 k 值使衰减更平缓，Top-K 结果间差距更小
    RRF_K: int = 60

    @classmethod
    def from_profile_and_env(cls, profile_dict: dict | None = None) -> "HybridScoreWeights":
        """从 profile 配置和环境变量加载权重配置

        优先级：环境变量 > profile 配置 > 默认值

        参数:
            profile_dict: profile 中的 retrieval.hybrid_weights 配置

        返回:
            归一化后的权重配置
        """
        weights = cls()

        # 1. 从 profile 读取配置
        if profile_dict:
            weights.bm25 = float(profile_dict.get("bm25", weights.bm25))
            weights.embedding = float(profile_dict.get("embedding", weights.embedding))
            weights.lexical = float(profile_dict.get("lexical", weights.lexical))
            weights.source = "profile"

        # 2. 从配置文件读取（覆盖 profile）
        path = os.getenv("AGENT_WIKI_HYBRID_WEIGHTS_PATH", "").strip()
        if path:
            file_path = Path(path)
            if file_path.exists():
                try:
                    payload = json.loads(file_path.read_text(encoding="utf-8"))
                    weights.bm25 = float(payload.get("bm25", weights.bm25))
                    weights.embedding = float(payload.get("embedding", weights.embedding))
                    weights.lexical = float(payload.get("lexical", weights.lexical))
                    weights.source = f"file:{file_path.as_posix()}"
                except Exception:
                    pass

        # 3. 环境变量优先级最高
        weights.bm25 = env_float("AGENT_WIKI_WEIGHT_BM25", weights.bm25)
        weights.embedding = env_float("AGENT_WIKI_WEIGHT_EMBEDDING", weights.embedding)
        weights.lexical = env_float("AGENT_WIKI_WEIGHT_LEXICAL", weights.lexical)
        return weights.normalized()

    def normalized(self) -> "HybridScoreWeights":
        """
        执行`normalized` 相关处理逻辑。

        参数:
            self: 当前对象实例。

        返回:
            返回类型为 `'HybridScoreWeights'` 的处理结果。
        """
        bm25 = max(self.bm25, 0.0)
        embedding = max(self.embedding, 0.0)
        lexical = max(self.lexical, 0.0)
        total = bm25 + embedding + lexical
        if total <= 0:
            return HybridScoreWeights(source="fallback")
        return HybridScoreWeights(
            bm25=bm25 / total,
            embedding=embedding / total,
            lexical=lexical / total,
            source=self.source,
        )


@dataclass
class WikiRetrieverRuntimeConfig:
    """Wiki 检索器运行时配置

    Attributes:
        default_top_k: 默认返回结果数
        max_chunks_per_doc: 每文档最大块数
        chunk_size: 分块大小（固定长度模式）
        chunk_overlap: 分块重叠（固定长度模式）
        excerpt_max_chars: 摘要最大字符数
        candidate_multiplier: 候选集倍数
        min_candidates: 最小候选数
        enable_embedding: 是否启用向量检索
        embedding_model: 向量模型名称
        embedding_top_k: 向量检索返回数量
        embedding_device: 向量模型运行设备
        embedding_persist_root: 向量索引持久化目录
        enable_semantic_chunking: 是否启用语义分块
        semantic_min_chunk_chars: 语义分块最小字符数
        semantic_max_chunk_chars: 语义分块最大字符数
        semantic_preserve_code_blocks: 语义分块是否保持代码块完整
        semantic_preserve_tables: 语义分块是否保持表格完整
        semantic_include_hierarchy: 语义分块是否包含层级信息
    """
    default_top_k: int = 4
    max_chunks_per_doc: int = 1
    chunk_size: int = 520
    chunk_overlap: int = 80
    excerpt_max_chars: int = 220
    candidate_multiplier: int = 6
    min_candidates: int = 12
    # 向量检索配置
    enable_embedding: bool = True
    embedding_model: str = "BAAI/bge-base-zh-v1.5"
    embedding_top_k: int = 4
    embedding_device: str = "cpu"
    embedding_persist_root: str = ".vectorstore"
    embedding_cache_dir: str | None = None  # 模型缓存目录，用于离线加载
    # 语义分块配置（新增）
    enable_semantic_chunking: bool = False
    semantic_min_chunk_chars: int = 200
    semantic_max_chunk_chars: int = 800
    semantic_preserve_code_blocks: bool = True
    semantic_preserve_tables: bool = True
    semantic_include_hierarchy: bool = True

    @classmethod
    def from_profile_and_env(
        cls,
        *,
        default_top_k: int,
        embedding_profile: EmbeddingProfile | None = None,
    ) -> "WikiRetrieverRuntimeConfig":
        """
        从 profile 配置和环境变量加载运行时配置。

        参数:
            cls: 当前类对象。
            default_top_k: 默认返回结果数
            embedding_profile: 领域配置中的 EmbeddingProfile

        返回:
            返回类型为 `'WikiRetrieverRuntimeConfig'` 的处理结果。
        """
        # 向量检索配置：环境变量优先级高于 profile 配置
        if embedding_profile:
            enable_embedding = env_bool("AGENT_WIKI_EMBEDDING_ENABLED", embedding_profile.enabled)
            embedding_model = os.getenv("AGENT_EMBEDDING_MODEL", embedding_profile.model)
            embedding_top_k = env_int("AGENT_WIKI_EMBEDDING_TOP_K", embedding_profile.top_k, minimum=1)
            embedding_device = os.getenv("AGENT_EMBEDDING_DEVICE", embedding_profile.device)
            embedding_persist_root = os.getenv("AGENT_EMBEDDING_PERSIST_ROOT", embedding_profile.persist_root)
            embedding_cache_dir = os.getenv("AGENT_EMBEDDING_CACHE_DIR", embedding_profile.cache_dir or "")
        else:
            enable_embedding = env_bool("AGENT_WIKI_EMBEDDING_ENABLED", True)
            embedding_model = os.getenv("AGENT_EMBEDDING_MODEL", "BAAI/bge-base-zh-v1.5")
            embedding_top_k = env_int("AGENT_WIKI_EMBEDDING_TOP_K", 4, minimum=1)
            embedding_device = os.getenv("AGENT_EMBEDDING_DEVICE", "cpu")
            embedding_persist_root = os.getenv("AGENT_EMBEDDING_PERSIST_ROOT", ".vectorstore")
            embedding_cache_dir = os.getenv("AGENT_EMBEDDING_CACHE_DIR", "")

        return cls(
            default_top_k=env_int("AGENT_WIKI_TOP_K", default_top_k, minimum=1),
            max_chunks_per_doc=env_int("AGENT_WIKI_MAX_CHUNKS_PER_DOC", 1, minimum=1),
            chunk_size=env_int("AGENT_WIKI_PARAGRAPH_MAX_CHARS", 520, minimum=120),
            chunk_overlap=env_int("AGENT_WIKI_PARAGRAPH_MIN_CHARS", 80, minimum=0),
            excerpt_max_chars=env_int("AGENT_WIKI_EXCERPT_MAX_CHARS", 220, minimum=60),
            candidate_multiplier=env_int("AGENT_WIKI_STAGE2_MULTIPLIER", 6, minimum=1),
            min_candidates=env_int("AGENT_WIKI_STAGE2_MIN_CANDIDATES", 12, minimum=1),
            enable_embedding=enable_embedding,
            embedding_model=embedding_model,
            embedding_top_k=embedding_top_k,
            embedding_device=embedding_device,
            embedding_persist_root=embedding_persist_root,
            embedding_cache_dir=embedding_cache_dir or None,
            # 语义分块配置
            enable_semantic_chunking=env_bool("AGENT_WIKI_SEMANTIC_CHUNKING_ENABLED", False),
            semantic_min_chunk_chars=env_int("AGENT_WIKI_SEMANTIC_MIN_CHARS", 200, minimum=50),
            semantic_max_chunk_chars=env_int("AGENT_WIKI_SEMANTIC_MAX_CHARS", 800, minimum=100),
            semantic_preserve_code_blocks=env_bool("AGENT_WIKI_SEMANTIC_PRESERVE_CODE", True),
            semantic_preserve_tables=env_bool("AGENT_WIKI_SEMANTIC_PRESERVE_TABLES", True),
            semantic_include_hierarchy=env_bool("AGENT_WIKI_SEMANTIC_INCLUDE_HIERARCHY", True),
        )


class MarkdownWikiRetriever:
    """
    Markdown Wiki 检索器。

    采用三级召回策略：
    1. BM25：词项精确匹配
    2. Embedding：语义向量匹配
    3. Lexical：词法覆盖率补充

    使用 RRF (Reciprocal Rank Fusion) 归一化不同检索路径的分数，
    确保加权求和的有效性。
    """
    GENERAL_DOC_PENALTY: float = 0.10  # 通用文档惩罚系数（RRF 范围下）

    STOP_WORDS = {
        "什么",
        "怎么",
        "如何",
        "这个",
        "那个",
        "我们",
        "你们",
        "他们",
        "以及",
        "并且",
        "或者",
        "是否",
        "可以",
        "需要",
        "相关",
        "问题",
        "方案",
        "模块",
        "系统",
    }

    def __init__(
        self,
        *,
        wiki_dir: Path,
        project_root: Path,
        default_top_k: int = 4,
        module_doc_hints: dict[str, tuple[str, ...]] | None = None,
        embedding_profile: EmbeddingProfile | None = None,
        hybrid_weights_profile: dict | None = None,
        reranker_profile: RerankerProfile | None = None,
    ) -> None:
        """
        初始化 Wiki 检索器。

        参数:
            wiki_dir: Wiki 文档目录
            project_root: 项目根目录
            default_top_k: 默认返回结果数
            module_doc_hints: 模块文档提示词
            embedding_profile: 领域配置中的 EmbeddingProfile
            hybrid_weights_profile: profile 中的 hybrid_weights 配置
            reranker_profile: Cross-Encoder 重排器配置
        """
        self.wiki_dir = wiki_dir
        self.project_root = project_root
        self.module_doc_hints = dict(module_doc_hints or {})
        self.runtime_config = WikiRetrieverRuntimeConfig.from_profile_and_env(
            default_top_k=default_top_k,
            embedding_profile=embedding_profile,
        )
        self.hybrid_weights = HybridScoreWeights.from_profile_and_env(hybrid_weights_profile)
        self._logger = get_file_logger(project_root=project_root)

        self._chunks: list[WikiChunk] = []
        self._documents: list[Document] = []
        self._chunk_by_id: dict[int, WikiChunk] = {}
        self._chunk_ids_by_path: defaultdict[str, list[int]] = defaultdict(list)
        self.last_search_profile: dict[str, Any] = {}

        self._bm25: BM25Retriever | None = None
        # 精简：移除 TFIDF 和 Ensemble，只保留 BM25 + Embedding

        # 向量检索器
        self._embedding_retriever: EmbeddingRetriever | None = None
        self._embedding_stats: dict[str, Any] = {}

        # Cross-Encoder 重排器（可选）
        self._reranker: Any = None  # CrossEncoderReranker 实例，延迟导入避免循环依赖
        self._reranker_profile = reranker_profile

        started = perf_counter()
        stats = self._build_index()

        # 初始化向量索引（在 BM25/TFIDF 索引构建完成后）
        if self.runtime_config.enable_embedding and self._documents:
            self._init_embedding_retriever()

        # 初始化 Cross-Encoder 重排器（可选，默认关闭）
        if self._reranker_profile and self._reranker_profile.enabled:
            self._init_reranker()

        self._logger.info(
            "workflow.wiki_index.built",
            wiki_dir=self._to_relative_path(self.wiki_dir),
            file_count=stats.get("file_count", 0),
            chunk_count=stats.get("chunk_count", 0),
            default_top_k=self.runtime_config.default_top_k,
            max_chunks_per_doc=self.runtime_config.max_chunks_per_doc,
            hybrid_weights={
                "bm25": round(self.hybrid_weights.bm25, 4),
                "embedding": round(self.hybrid_weights.embedding, 4),
                "lexical": round(self.hybrid_weights.lexical, 4),
                "source": self.hybrid_weights.source,
            },
            embedding_enabled=self.runtime_config.enable_embedding,
            embedding_model=self.runtime_config.embedding_model if self.runtime_config.enable_embedding else None,
            latency_ms=int((perf_counter() - started) * 1000),
        )

        # 记录重排器状态
        self._logger.info(
            "workflow.wiki_reranker.status",
            enabled=self._reranker is not None,
            model=self._reranker_profile.model if self._reranker_profile else None,
        )

    def _init_reranker(self) -> None:
        """初始化 Cross-Encoder 重排器。

        该方法延迟导入 CrossEncoderReranker，避免未安装 sentence-transformers 时启动失败。
        重排器初始化后会加载模型到内存，首次调用可能需要下载模型文件。
        """
        if not self._reranker_profile or not self._reranker_profile.enabled:
            return

        try:
            from retrievers.core.cross_encoder_reranker import (
                CrossEncoderReranker,
                CrossEncoderRerankerConfig,
            )

            config = CrossEncoderRerankerConfig.from_profile(self._reranker_profile)
            self._reranker = CrossEncoderReranker(
                project_root=self.project_root,
                config=config,
            )
            self._reranker.initialize()
            self._logger.info(
                "workflow.wiki_reranker.initialized",
                model=self._reranker_profile.model,
                top_k=self._reranker_profile.top_k,
                candidate_top_k=self._reranker_profile.candidate_top_k,
            )
        except ImportError as e:
            self._logger.warning(
                "workflow.wiki_reranker.import_error",
                error=str(e),
                message="CrossEncoderReranker 未安装，跳过重排器初始化",
            )
            self._reranker = None
        except Exception as e:
            self._logger.error(
                "workflow.wiki_reranker.init_error",
                error=str(e),
            )
            self._reranker = None

    def search(
        self,
        *,
        user_query: str,
        retrieval_queries: list[str] | None = None,
        module_name: str | None = None,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        执行`search` 相关处理逻辑。
        
        参数:
            self: 当前对象实例。
        
        返回:
            返回类型为 `list[dict[str, Any]]` 的处理结果。
        """
        if (
            not user_query.strip()
            or not self._documents
            or self._bm25 is None
        ):
            self.last_search_profile = {"latency_ms": 0.0, "hits": 0}
            return []

        started = perf_counter()
        final_top_k = int(top_k or self.runtime_config.default_top_k)
        candidate_k = max(
            final_top_k * self.runtime_config.candidate_multiplier,
            self.runtime_config.min_candidates,
        )

        expanded_queries = self._build_expanded_queries(
            user_query=user_query,
            retrieval_queries=retrieval_queries or [],
            module_name=module_name or "",
        )
        merged_query = " ".join(expanded_queries)

        # 精简后的三级召回：BM25 + Embedding + Lexical
        bm25_docs: list[Document] = []
        embedding_docs: list[Document] = []

        assert self._bm25 is not None
        self._bm25.k = candidate_k
        bm25_docs = self._bm25.invoke(merged_query)

        # 向量检索（与 BM25 串行执行）
        if self._embedding_retriever and self._embedding_retriever.is_initialized:
            embedding_results = self._embedding_retriever.search_with_scores(
                merged_query,
                top_k=candidate_k,
            )
            for doc, score in embedding_results:
                chunk_id = int(doc.metadata.get("chunk_id", -1))
                if chunk_id > 0:
                    embedding_docs.append(doc)

        bm25_ranks = self._rank_map(bm25_docs)
        embedding_ranks = self._rank_map(embedding_docs)

        query_terms = self._extract_terms(expanded_queries)

        candidate_ids: list[int] = []
        for docs in (bm25_docs, embedding_docs):
            for doc in docs:
                chunk_id = int(doc.metadata.get("chunk_id", -1))
                if chunk_id <= 0:
                    continue
                candidate_ids.append(chunk_id)

        dedup_candidate_ids = list(dict.fromkeys(candidate_ids))[:candidate_k]
        scored: list[dict[str, Any]] = []
        for chunk_id in dedup_candidate_ids:
            chunk = self._chunk_by_id.get(chunk_id)
            if chunk is None:
                continue

            bm25_rank = bm25_ranks.get(chunk_id)
            embedding_rank = embedding_ranks.get(chunk_id)

            bm25_score = self._rank_score(bm25_rank)
            embedding_score = self._rank_score(embedding_rank)
            lexical_score = self._lexical_match_score(chunk.normalized_text, query_terms)
            module_boost = self._module_prior_boost(chunk=chunk, module_name=module_name or "")

            # 计算通用文档惩罚：当查询匹配到具体模块时，对总体架构等通用文档降低分数
            general_doc_penalty = self._general_doc_penalty(
                chunk=chunk,
                module_name=module_name or "",
                module_boost=module_boost,
            )

            # 三级召回评分公式（RRF 归一化，范围 [0, 1]）
            # final_score = bm25*0.30 + embedding*0.50 + lexical*0.20 + module_boost - penalty
            score = (
                bm25_score * self.hybrid_weights.bm25
                + embedding_score * self.hybrid_weights.embedding
                + lexical_score * self.hybrid_weights.lexical
                + module_boost
                - general_doc_penalty
            )
            scored.append(
                {
                    "chunk": chunk,
                    "score": score,
                    "bm25_score": bm25_score,
                    "embedding_score": embedding_score,
                    "lexical_score": lexical_score,
                    "module_boost": module_boost,
                    "general_doc_penalty": general_doc_penalty,
                }
            )

        scored.sort(key=lambda item: item["score"], reverse=True)

        # ========== 优化后的重排流程 ==========
        # 核心改动：将重排移到多样性选择之前，扩大候选集
        # 1. 先从 scored 中取更多候选（使用 reranker.candidate_top_k）
        # 2. 对候选集执行 Cross-Encoder 重排
        # 3. 重排后再执行多样性选择，确保最终结果多样性

        rerank_enabled = self._reranker is not None and len(scored) > 0
        rerank_candidate_k = final_top_k  # 默认只取 final_top_k

        if rerank_enabled and self._reranker_profile:
            # 使用 profile 配置的 candidate_top_k，扩大重排候选集
            rerank_candidate_k = min(
                self._reranker_profile.candidate_top_k,
                len(scored)
            )

        # 准备重排候选集
        if rerank_enabled and rerank_candidate_k > final_top_k:
            # 优化场景：候选集足够大，执行重排
            rerank_start = perf_counter()
            rerank_candidates_raw = scored[:rerank_candidate_k]

            # 构建重排器需要的候选格式
            rerank_candidates = [
                {
                    "source_type": "wiki",
                    "title": item["chunk"].title,
                    "path": self._to_relative_path(item["chunk"].source_path),
                    "score": round(float(item["score"]), 4),
                    "section": item["chunk"].section,
                    "chunk_type": item["chunk"].chunk_type,
                    "excerpt": self._build_excerpt(item["chunk"].content),
                    "content": item["chunk"].content,
                    "rank": idx + 1,
                    "retrieval_debug": {
                        "bm25": round(float(item["bm25_score"]), 4),
                        "embedding": round(float(item["embedding_score"]), 4),
                        "lexical": round(float(item["lexical_score"]), 4),
                        "module_boost": round(float(item["module_boost"]), 4),
                        "general_doc_penalty": round(float(item.get("general_doc_penalty", 0.0)), 4),
                        "weights": {
                            "bm25": round(self.hybrid_weights.bm25, 4),
                            "embedding": round(self.hybrid_weights.embedding, 4),
                            "lexical": round(self.hybrid_weights.lexical, 4),
                        },
                    },
                    # 保留原始 chunk 引用，用于后续多样性选择
                    "_chunk_ref": item["chunk"],
                }
                for idx, item in enumerate(rerank_candidates_raw)
            ]

            # 执行重排：重排后返回 final_top_k * 2 条，再执行多样性选择
            rerank_output_k = min(final_top_k * 2, rerank_candidate_k)
            reranked_hits = self._reranker.rerank(
                query=user_query,
                candidates=rerank_candidates,
                top_k=rerank_output_k,
                content_key="content",
                source="wiki",
            )

            # 将重排结果转换回带 chunk 引用的格式，用于多样性选择
            reranked_with_chunks = []
            for rank, hit in enumerate(reranked_hits, start=1):
                rerank_score = hit.get("rerank_score", hit.get("score", 0.0))
                chunk_ref = hit.get("_chunk_ref")
                if chunk_ref is None:
                    continue
                reranked_with_chunks.append({
                    "chunk": chunk_ref,
                    "score": rerank_score,
                    "bm25_score": hit.get("retrieval_debug", {}).get("bm25", 0.0),
                    "embedding_score": hit.get("retrieval_debug", {}).get("embedding", 0.0),
                    "lexical_score": hit.get("retrieval_debug", {}).get("lexical", 0.0),
                    "module_boost": hit.get("retrieval_debug", {}).get("module_boost", 0.0),
                    "general_doc_penalty": hit.get("retrieval_debug", {}).get("general_doc_penalty", 0.0),
                    # 重排特有字段
                    "rerank_score": rerank_score,
                    "original_rank": hit.get("original_rank", rank),
                })

            # 对重排结果执行多样性选择
            selected = self._select_diverse(reranked_with_chunks, top_n=final_top_k)

            # 记录重排性能指标
            rerank_latency = round((perf_counter() - rerank_start) * 1000, 3)
            rerank_profile = {
                "enabled": True,
                "latency_ms": rerank_latency,
                "model": self._reranker_profile.model if self._reranker_profile else None,
                "candidate_k": rerank_candidate_k,
                "output_k": rerank_output_k,
            }
        else:
            # 无重排或候选集太小，直接执行多样性选择
            selected = self._select_diverse(scored, top_n=final_top_k)
            rerank_profile = {"enabled": False, "reason": "no_reranker_or_insufficient_candidates"}

        # 构建最终输出
        hits: list[dict[str, Any]] = []
        for rank, item in enumerate(selected, start=1):
            chunk = item["chunk"]
            hit = {
                "source_type": "wiki",
                "title": chunk.title,
                "path": self._to_relative_path(chunk.source_path),
                "score": round(float(item.get("rerank_score", item["score"])), 4),
                "section": chunk.section,
                "chunk_type": chunk.chunk_type,
                "excerpt": self._build_excerpt(chunk.content),
                "content": chunk.content,
                "rank": rank,
                "retrieval_debug": {
                    "bm25": round(float(item["bm25_score"]), 4),
                    "embedding": round(float(item["embedding_score"]), 4),
                    "lexical": round(float(item["lexical_score"]), 4),
                    "module_boost": round(float(item["module_boost"]), 4),
                    "general_doc_penalty": round(float(item.get("general_doc_penalty", 0.0)), 4),
                    "weights": {
                        "bm25": round(self.hybrid_weights.bm25, 4),
                        "embedding": round(self.hybrid_weights.embedding, 4),
                        "lexical": round(self.hybrid_weights.lexical, 4),
                    },
                },
            }
            # 如果是重排结果，添加重排相关信息
            if "rerank_score" in item:
                hit["score_source"] = "reranker"
                hit["retrieval_debug"]["rerank_score"] = round(float(item["rerank_score"]), 4)
                hit["retrieval_debug"]["original_rank"] = item.get("original_rank", rank)
            hits.append(hit)

        # 记录检索性能指标
        self.last_search_profile = {
            "latency_ms": round((perf_counter() - started) * 1000, 3),
            "candidate_k": candidate_k,
            "expanded_query_count": len(expanded_queries),
            "hits": len(hits),
            "embedding": {
                "enabled": bool(self._embedding_retriever and self._embedding_retriever.is_initialized),
                "hits": len(embedding_docs),
            },
            "rerank": rerank_profile if rerank_enabled else {"enabled": False},
        }

        return hits

    def _get_chunk_content_by_path(self, path: str, section: str) -> str:
        """根据路径和章节获取 chunk 内容，用于重排器。

        参数:
            path: 文档相对路径
            section: 章节名称

        返回:
            chunk 内容字符串
        """
        chunk_ids = self._chunk_ids_by_path.get(path, [])
        for chunk_id in chunk_ids:
            chunk = self._chunk_by_id.get(chunk_id)
            if chunk and chunk.section == section:
                return chunk.content
        # 如果找不到精确匹配，返回第一个 chunk 的内容
        if chunk_ids:
            chunk = self._chunk_by_id.get(chunk_ids[0])
            if chunk:
                return chunk.content
        return ""

    def get_chunk_snapshot(self) -> list[dict[str, Any]]:
        """
        执行`get chunk snapshot` 相关处理逻辑。
        
        参数:
            self: 当前对象实例。
        
        返回:
            返回类型为 `list[dict[str, Any]]` 的处理结果。
        """
        rows: list[dict[str, Any]] = []
        for index, chunk in enumerate(self._chunks, start=1):
            rows.append(
                {
                    "index": index,
                    "path": self._to_relative_path(chunk.source_path),
                    "title": chunk.title,
                    "section": chunk.section,
                    "chunk_type": chunk.chunk_type,
                    "content_length": len(chunk.content),
                    "content": chunk.content,
                }
            )
        return rows

    def _build_index(self) -> dict[str, int]:
        """构建文档索引

        支持两种分块策略：
        1. 语义分块（enable_semantic_chunking=True）：基于 Markdown 标题层级分块
        2. 固定长度分块（默认）：使用 RecursiveCharacterTextSplitter

        Returns:
            索引统计信息 {"file_count": N, "chunk_count": M}
        """
        stats = {"file_count": 0, "chunk_count": 0}
        if not self.wiki_dir.exists():
            return stats

        markdown_files = sorted(self.wiki_dir.rglob("*.md"))
        stats["file_count"] = len(markdown_files)

        # 根据配置选择分块策略
        if self.runtime_config.enable_semantic_chunking:
            # 使用语义分块器
            stats = self._build_index_semantic(markdown_files, stats)
        else:
            # 使用固定长度分块器（原有逻辑）
            stats = self._build_index_fixed(markdown_files, stats)

        stats["chunk_count"] = len(self._chunks)
        if not self._documents:
            return stats

        # 构建 BM25 索引
        self._bm25 = BM25Retriever.from_documents(self._documents)

        # 记录分块策略
        self._logger.info(
            "workflow.wiki_index.built",
            strategy="semantic" if self.runtime_config.enable_semantic_chunking else "fixed",
            file_count=stats["file_count"],
            chunk_count=stats["chunk_count"],
        )

        return stats

    def _build_index_semantic(self, markdown_files: list[Path], stats: dict[str, int]) -> dict[str, int]:
        """使用语义分块器构建索引

        Args:
            markdown_files: Markdown 文件列表
            stats: 统计信息字典

        Returns:
            更新后的统计信息
        """
        from retrievers.wiki.semantic_chunker import (
            SemanticMarkdownChunker,
            SemanticChunkerConfig,
        )

        # 创建语义分块器
        chunker_config = SemanticChunkerConfig(
            min_chunk_chars=self.runtime_config.semantic_min_chunk_chars,
            max_chunk_chars=self.runtime_config.semantic_max_chunk_chars,
            max_chunks_per_doc=self.runtime_config.max_chunks_per_doc,
            preserve_code_blocks=self.runtime_config.semantic_preserve_code_blocks,
            preserve_tables=self.runtime_config.semantic_preserve_tables,
            include_hierarchy=self.runtime_config.semantic_include_hierarchy,
        )
        chunker = SemanticMarkdownChunker(chunker_config)

        next_chunk_id = 1
        for md_file in markdown_files:
            try:
                raw = md_file.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                raw = md_file.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

            cleaned = raw.strip()
            if not cleaned:
                continue

            # 使用语义分块器
            semantic_chunks = chunker.chunk(cleaned, md_file)

            for sem_chunk in semantic_chunks:
                # 转换为 WikiChunk
                wiki_chunk = WikiChunk(
                    chunk_id=next_chunk_id,
                    source_path=md_file,
                    title=sem_chunk.title,
                    section=sem_chunk.section,
                    chunk_type=sem_chunk.chunk_type,
                    content=sem_chunk.content,
                    normalized_text=self._normalize(sem_chunk.content),
                    hierarchy=sem_chunk.hierarchy if self.runtime_config.semantic_include_hierarchy else [],
                    start_line=sem_chunk.start_line,
                    end_line=sem_chunk.end_line,
                    token_count=sem_chunk.token_count,
                )

                self._chunks.append(wiki_chunk)
                self._chunk_by_id[next_chunk_id] = wiki_chunk
                self._chunk_ids_by_path[self._to_relative_path(md_file)].append(next_chunk_id)

                # 构建 Document
                doc_metadata = {
                    "chunk_id": next_chunk_id,
                    "path": self._to_relative_path(md_file),
                    "title": sem_chunk.title,
                    "section": sem_chunk.section,
                    "chunk_type": sem_chunk.chunk_type,
                }
                if self.runtime_config.semantic_include_hierarchy:
                    doc_metadata["hierarchy"] = sem_chunk.hierarchy

                self._documents.append(
                    Document(
                        page_content=sem_chunk.content,
                        metadata=doc_metadata,
                    )
                )
                next_chunk_id += 1

        return stats

    def _build_index_fixed(self, markdown_files: list[Path], stats: dict[str, int]) -> dict[str, int]:
        """使用固定长度分块器构建索引（原有逻辑）

        Args:
            markdown_files: Markdown 文件列表
            stats: 统计信息字典

        Returns:
            更新后的统计信息
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.runtime_config.chunk_size,
            chunk_overlap=min(self.runtime_config.chunk_overlap, max(self.runtime_config.chunk_size // 2, 0)),
            separators=["\n## ", "\n### ", "\n\n", "\n", "。", "；", " ", ""],
            keep_separator=True,
        )

        next_chunk_id = 1
        for md_file in markdown_files:
            try:
                raw = md_file.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                raw = md_file.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

            cleaned = raw.strip()
            if not cleaned:
                continue

            title = self._extract_title(cleaned, md_file)
            pieces = splitter.split_text(cleaned)
            if not pieces:
                pieces = [cleaned]

            for piece in pieces:
                content = piece.strip()
                if not content:
                    continue
                section = self._extract_section(content, fallback=title)
                chunk_type = self._infer_chunk_type(content)
                chunk = WikiChunk(
                    chunk_id=next_chunk_id,
                    source_path=md_file,
                    title=title,
                    section=section,
                    chunk_type=chunk_type,
                    content=content,
                    normalized_text=self._normalize(content),
                    hierarchy=[],  # 固定分块不保留层级
                )
                self._chunks.append(chunk)
                self._chunk_by_id[next_chunk_id] = chunk
                self._chunk_ids_by_path[self._to_relative_path(md_file)].append(next_chunk_id)
                self._documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "chunk_id": next_chunk_id,
                            "path": self._to_relative_path(md_file),
                            "title": title,
                            "section": section,
                            "chunk_type": chunk_type,
                        },
                    )
                )
                next_chunk_id += 1

        return stats

    def _init_embedding_retriever(self) -> None:
        """初始化向量检索器

        在 BM25/TFIDF 索引构建完成后调用，用于构建向量索引。
        使用 profile 配置 + 环境变量覆盖。
        """
        if not self.runtime_config.enable_embedding:
            return

        try:
            # 使用 EmbeddingProfile 创建配置（支持环境变量覆盖）
            profile = EmbeddingProfile(
                enabled=self.runtime_config.enable_embedding,
                model=self.runtime_config.embedding_model,
                device=self.runtime_config.embedding_device,
                top_k=self.runtime_config.embedding_top_k,
                persist_root=self.runtime_config.embedding_persist_root,
                cache_dir=self.runtime_config.embedding_cache_dir,
            )
            embedding_config = EmbeddingRetrieverConfig.from_profile(
                profile=profile,
                collection_name=f"wiki_{self.wiki_dir.name}",
                persist_root=str(self.project_root),
            )

            self._embedding_retriever = EmbeddingRetriever(
                project_root=self.project_root,
                config=embedding_config,
            )

            # 初始化向量索引
            self._embedding_stats = self._embedding_retriever.initialize(self._documents)
            self._logger.info(
                "workflow.wiki_embedding_index.built",
                model=embedding_config.model_name,
                doc_count=self._embedding_stats.get("doc_count", 0),
                persist_dir=embedding_config.persist_directory,
            )
        except Exception as e:
            self._logger.warning(
                "workflow.wiki_embedding_index.failed",
                error=str(e),
                message="Falling back to BM25/TFIDF only",
            )
            self._embedding_retriever = None
            self._embedding_stats = {}

    def _build_expanded_queries(
        self,
        *,
        user_query: str,
        retrieval_queries: list[str],
        module_name: str,
    ) -> list[str]:
        """
        构建当前步骤所需的数据结构或文本内容。
        
        参数:
            self: 当前对象实例。
        
        返回:
            返回类型为 `list[str]` 的处理结果。
        """
        rows = [user_query.strip(), *[q.strip() for q in retrieval_queries if q.strip()]]
        module_hints = self.module_doc_hints.get(module_name.strip().lower(), ())
        rows.extend(module_hints)

        deduped: list[str] = []
        seen: set[str] = set()
        for row in rows:
            normalized = " ".join(row.split())
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(normalized)
        return deduped[:16]

    def _rank_map(self, docs: list[Document]) -> dict[int, int]:
        """
        内部辅助函数，负责`rank map` 相关处理。
        
        参数:
            self: 当前对象实例。
            docs: 列表参数，用于承载批量输入数据。
        
        返回:
            返回类型为 `dict[int, int]` 的处理结果。
        """
        ranks: dict[int, int] = {}
        for idx, doc in enumerate(docs, start=1):
            chunk_id = int(doc.metadata.get("chunk_id", -1))
            if chunk_id <= 0:
                continue
            ranks.setdefault(chunk_id, idx)
        return ranks

    def _rank_score(self, rank: int | None) -> float:
        """
        基于排名的分数归一化（Reciprocal Rank Fusion）。

        将检索排名转换为归一化分数，范围 [0, 1]。
        使用 RRF 公式: score = k / (rank + k)

        RRF 的优势：
        1. 消除不同检索器原始分数量级差异
        2. 对异常值鲁棒（只依赖排名顺序）
        3. k=60 是业界验证的标准参数，衰减平缓

        参数:
            rank: 检索排名（从 1 开始），None 表示未命中

        返回:
            归一化分数，范围 [0, 1]
            rank=1 → 0.984, rank=10 → 0.857, rank=50 → 0.545
        """
        if rank is None:
            return 0.0
        # RRF(k=60) - 信息检索领域验证的标准归一化方法
        k = HybridScoreWeights.RRF_K
        return k / (rank + k)

    def _extract_terms(self, query_phrases: list[str]) -> list[str]:
        """
        内部辅助函数，负责`extract terms` 相关处理。
        
        参数:
            self: 当前对象实例。
            query_phrases: 列表参数，用于承载批量输入数据。
        
        返回:
            返回类型为 `list[str]` 的处理结果。
        """
        merged = " ".join(query_phrases).lower()
        terms: list[str] = []
        terms.extend([token for token in re.findall(r"[a-z][a-z0-9_+-]{1,}", merged) if token not in self.STOP_WORDS])
        terms.extend([token for token in re.findall(r"[\u4e00-\u9fff]{2,10}", merged) if token not in self.STOP_WORDS])
        return list(dict.fromkeys(terms))[:48]

    def _lexical_match_score(self, text: str, terms: list[str]) -> float:
        """
        计算词法覆盖率分数。

        基于查询词在文本中的命中率计算分数，范围 [0, 1]。
        与 RRF 归一化保持一致，确保与 BM25/Embedding 分数可比。

        计算方式：
        - coverage = 命中词数 / 总词数
        - score = coverage（范围 0~1）

        参数:
            text: 待匹配的归一化文本
            terms: 查询词列表

        返回:
            词法覆盖率分数，范围 [0, 1]
        """
        if not terms:
            return 0.0
        hits = sum(1 for term in terms if term in text)
        if hits <= 0:
            return 0.0
        coverage = hits / len(terms)
        return coverage  # 范围 [0, 1]

    def _module_prior_boost(self, *, chunk: WikiChunk, module_name: str) -> float:
        """
        计算模块先验加成分数。

        当查询匹配到具体模块时，对与该模块相关的文档给予加成。
        加成分数范围 [0, 0.15]，确保不会过度影响主检索分数。

        匹配权重：
        - 路径匹配：+0.3（如 "04-" 匹配文件名前缀）
        - 标题匹配：+0.5（权重最高，标题更准确反映文档主题）
        - 内容匹配：+0.2（权重较低）

        参数:
            chunk: 文档块对象
            module_name: 模块名称

        返回:
            加成分数，范围 [0, 0.15]
        """
        normalized_module = module_name.strip().lower()
        if not normalized_module:
            return 0.0
        hints = self.module_doc_hints.get(normalized_module, ())
        if not hints:
            return 0.0

        path_key = self._to_relative_path(chunk.source_path).lower()
        title_lower = chunk.title.lower() if chunk.title else ""
        text_lower = chunk.normalized_text.lower() if chunk.normalized_text else ""

        # 计算匹配分数，三种匹配独立计算，可叠加
        score = 0.0
        for hint in hints:
            hint_lower = hint.lower()
            # 路径匹配
            if hint_lower in path_key:
                score += 0.3
            # 标题匹配（权重最高）
            if hint_lower in title_lower:
                score += 0.5
            # 文本内容匹配（权重较低）
            if hint_lower in text_lower:
                score += 0.2

        if score <= 0:
            return 0.0

        # 限制最大 boost 为 0.15，避免过度影响检索排序
        return min(score, 0.15)

    def _general_doc_penalty(self, *, chunk: WikiChunk, module_name: str, module_boost: float) -> float:
        """
        内部辅助函数，负责对通用文档（如总体架构）进行惩罚。

        当查询匹配到具体模块时，如果文档是通用文档（如 00-总体架构.md），
        且该文档不是当前模块的特异文档（路径不匹配模块前缀），则降低其分数。

        参数:
            self: 当前对象实例。
            chunk: 文档块对象
            module_name: 模块名称
            module_boost: 当前文档的模块 boost 值

        返回:
            返回类型为 `float` 的处理结果（惩罚值，从分数中减去）
        """
        normalized_module = module_name.strip().lower()
        if not normalized_module:
            return 0.0

        # 只有当模块有 wiki_hints 时才进行惩罚
        hints = self.module_doc_hints.get(normalized_module, ())
        if not hints:
            return 0.0

        path_key = self._to_relative_path(chunk.source_path).lower()

        # 检查是否是通用文档（总体架构文档）
        is_general_doc = "00-" in path_key or "总体架构" in path_key

        if not is_general_doc:
            return 0.0

        # 检查文档路径是否匹配当前模块的前缀 hints（如 "02-" 匹配 rate-prediction）
        # 如果匹配，说明这是模块的特异文档，不惩罚
        for hint in hints:
            hint_lower = hint.lower()
            # 只检查路径前缀匹配（如 "02-" 在文件名开头）
            if hint_lower in path_key and (hint_lower.endswith("-") or path_key.startswith(hint_lower)):
                return 0.0

        # 对通用文档进行惩罚，当查询明确指向具体模块时，通用文档的惩罚更大
        # 基础惩罚 + 额外惩罚（如果模块 boost 很低，说明查询非常特异）
        base_penalty = self.GENERAL_DOC_PENALTY
        extra_penalty = 0.0
        if module_boost < 0.1:
            # 如果模块 boost 很低，说明总体架构文档不是查询的主要目标
            extra_penalty = 0.05

        return min(base_penalty + extra_penalty, 0.15)  # 限制最大惩罚

    def _select_diverse(self, items: list[dict[str, Any]], *, top_n: int) -> list[dict[str, Any]]:
        """
        内部辅助函数，负责`select diverse` 相关处理。
        
        参数:
            self: 当前对象实例。
            items: 列表参数，用于承载批量输入数据。
        
        返回:
            返回类型为 `list[dict[str, Any]]` 的处理结果。
        """
        selected: list[dict[str, Any]] = []
        per_doc_counter: Counter[str] = Counter()
        for item in items:
            path_key = self._to_relative_path(item["chunk"].source_path)
            if per_doc_counter[path_key] >= self.runtime_config.max_chunks_per_doc:
                continue
            per_doc_counter[path_key] += 1
            selected.append(item)
            if len(selected) >= top_n:
                break
        return selected

    def _build_excerpt(self, text: str) -> str:
        """
        构建当前步骤所需的数据结构或文本内容。
        
        参数:
            self: 当前对象实例。
            text: 待处理的文本内容。
        
        返回:
            返回类型为 `str` 的处理结果。
        """
        merged = re.sub(r"\s+", " ", text.strip())
        if len(merged) <= self.runtime_config.excerpt_max_chars:
            return merged
        return merged[: self.runtime_config.excerpt_max_chars] + "..."

    def _extract_title(self, text: str, file_path: Path) -> str:
        """
        内部辅助函数，负责`extract title` 相关处理。
        
        参数:
            self: 当前对象实例。
            text: 待处理的文本内容。
            file_path: 路径参数，用于定位文件或目录。
        
        返回:
            返回类型为 `str` 的处理结果。
        """
        match = re.search(r"(?m)^#\s+(.+)$", text)
        if match:
            return match.group(1).strip()
        return file_path.stem

    def _extract_section(self, text: str, *, fallback: str) -> str:
        """
        内部辅助函数，负责`extract section` 相关处理。
        
        参数:
            self: 当前对象实例。
            text: 待处理的文本内容。
        
        返回:
            返回类型为 `str` 的处理结果。
        """
        match = re.search(r"(?m)^#{2,4}\s+(.+)$", text)
        if match:
            return match.group(1).strip()
        first_line = text.splitlines()[0].strip() if text.splitlines() else ""
        if first_line.startswith("#"):
            return first_line.lstrip("#").strip() or fallback
        return fallback

    def _infer_chunk_type(self, text: str) -> str:
        """
        内部辅助函数，负责`infer chunk type` 相关处理。
        
        参数:
            self: 当前对象实例。
            text: 待处理的文本内容。
        
        返回:
            返回类型为 `str` 的处理结果。
        """
        stripped = text.strip()
        if "```" in stripped:
            return "code"
        if re.search(r"(?m)^\s*[-*]\s+", stripped) or re.search(r"(?m)^\s*\d+\.\s+", stripped):
            return "list"
        if re.search(r"(?m)^\s*\|.+\|\s*$", stripped):
            return "table"
        normalized = self._normalize(stripped)
        if any(token in normalized for token in ("流程", "链路", "阶段", "步骤", "如下")):
            return "flow"
        return "paragraph"

    def _to_relative_path(self, path: Path) -> str:
        """
        内部辅助函数，负责`to relative path` 相关处理。
        
        参数:
            self: 当前对象实例。
            path: 文件或目录路径。
        
        返回:
            返回类型为 `str` 的处理结果。
        """
        try:
            return path.relative_to(self.project_root).as_posix()
        except ValueError:
            return path.as_posix()

    def _normalize(self, text: str) -> str:
        """
        内部辅助函数，负责`normalize` 相关处理。

        参数:
            self: 当前对象实例。
            text: 待处理的文本内容。

        返回:
            返回类型为 `str` 的处理结果。
        """
        return re.sub(r"\s+", " ", text.lower()).strip()


# ============================================================================
# 全局单例模式
# ============================================================================

# 全局 Wiki 检索器实例（单例）
_wiki_retriever_instance: MarkdownWikiRetriever | None = None


def get_wiki_retriever() -> MarkdownWikiRetriever | None:
    """获取全局 Wiki 检索器实例

    Returns:
        MarkdownWikiRetriever 实例，如果未初始化则返回 None
    """
    return _wiki_retriever_instance


def set_wiki_retriever(retriever: MarkdownWikiRetriever | None) -> None:
    """设置全局 Wiki 检索器实例

    Args:
        retriever: MarkdownWikiRetriever 实例
    """
    global _wiki_retriever_instance
    _wiki_retriever_instance = retriever
    if retriever:
        logger.info("[WikiRetriever] 全局单例已设置")
    else:
        logger.info("[WikiRetriever] 全局单例已清除")
