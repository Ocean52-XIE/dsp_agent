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
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import json
import os
import re
import shutil
import subprocess
from typing import Any

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from workflow.retrievers import EmbeddingRetriever, EmbeddingRetrieverConfig
from workflow.common.runtime_logging import get_file_logger
from workflow.common.func_utils import env_bool, env_float, env_int
from workflow.common.domain_profile import EmbeddingProfile, RerankerProfile


RG_STRATEGIES = {"rg_first", "rg_only", "no_rg"}


def _resolve_rg_strategy(raw_value: str | None, *, default: str) -> str:
    """
    内部辅助函数，负责`resolve rg strategy` 相关处理。
    
    参数:
        raw_value: 输入参数，用于控制当前处理逻辑。
    
    返回:
        返回类型为 `str` 的处理结果。
    """
    normalized = str(raw_value or "").strip().lower()
    if normalized in RG_STRATEGIES:
        return normalized
    if default in RG_STRATEGIES:
        return default
    return "rg_first"


@dataclass
class WikiChunk:
    """
    定义`WikiChunk`，用于封装相关数据结构与处理行为。
    """
    chunk_id: int
    source_path: Path
    title: str
    section: str
    chunk_type: str
    content: str
    normalized_text: str


@dataclass
class HybridScoreWeights:
    """
    定义`HybridScoreWeights`，用于封装相关数据结构与处理行为。

    精简后的三级召回策略：
    - bm25: 词项精确匹配（关键词必须出现）
    - embedding: 语义匹配（同义词理解）
    - lexical: 查询词覆盖率补充

    调整后：增加 embedding 权重，提升语义匹配能力，区分不同模块
    """
    bm25: float = 0.30  # 降低 BM25 权重，减少通用文档排名过高
    embedding: float = 0.50  # 增加 Embedding 权重，提升语义区分能力
    lexical: float = 0.20
    source: str = "default"

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
        path = os.getenv("WORKFLOW_WIKI_HYBRID_WEIGHTS_PATH", "").strip()
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
        weights.bm25 = env_float("WORKFLOW_WIKI_WEIGHT_BM25", weights.bm25)
        weights.embedding = env_float("WORKFLOW_WIKI_WEIGHT_EMBEDDING", weights.embedding)
        weights.lexical = env_float("WORKFLOW_WIKI_WEIGHT_LEXICAL", weights.lexical)
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
    """
    定义`WikiRetrieverRuntimeConfig`，用于封装相关数据结构与处理行为。
    """
    default_top_k: int = 4
    max_chunks_per_doc: int = 1
    chunk_size: int = 520
    chunk_overlap: int = 80
    excerpt_max_chars: int = 220
    candidate_multiplier: int = 6
    min_candidates: int = 12
    rg_strategy: str = "no_rg"
    rg_max_terms: int = 8
    rg_timeout_ms: int = 1200
    rg_path_boost: float = 0.55
    rg_max_matches_per_term: int = 80
    # 新增：向量检索配置
    enable_embedding: bool = True
    embedding_model: str = "BAAI/bge-base-zh-v1.5"
    embedding_top_k: int = 4
    embedding_device: str = "cpu"
    embedding_persist_root: str = ".vectorstore"

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
        legacy_flag = os.getenv("WORKFLOW_WIKI_RG_FIRST_ENABLED")
        if legacy_flag is None:
            default_strategy = "no_rg"
        else:
            default_strategy = "rg_first" if env_bool("WORKFLOW_WIKI_RG_FIRST_ENABLED", True) else "no_rg"
        strategy = _resolve_rg_strategy(
            os.getenv("WORKFLOW_WIKI_RG_STRATEGY"),
            default=default_strategy,
        )

        # 向量检索配置：环境变量优先级高于 profile 配置
        if embedding_profile:
            enable_embedding = env_bool("WORKFLOW_WIKI_EMBEDDING_ENABLED", embedding_profile.enabled)
            embedding_model = os.getenv("WORKFLOW_EMBEDDING_MODEL", embedding_profile.model)
            embedding_top_k = env_int("WORKFLOW_WIKI_EMBEDDING_TOP_K", embedding_profile.top_k, minimum=1)
            embedding_device = os.getenv("WORKFLOW_EMBEDDING_DEVICE", embedding_profile.device)
            embedding_persist_root = os.getenv("WORKFLOW_EMBEDDING_PERSIST_ROOT", embedding_profile.persist_root)
        else:
            enable_embedding = env_bool("WORKFLOW_WIKI_EMBEDDING_ENABLED", True)
            embedding_model = os.getenv("WORKFLOW_EMBEDDING_MODEL", "BAAI/bge-base-zh-v1.5")
            embedding_top_k = env_int("WORKFLOW_WIKI_EMBEDDING_TOP_K", 4, minimum=1)
            embedding_device = os.getenv("WORKFLOW_EMBEDDING_DEVICE", "cpu")
            embedding_persist_root = os.getenv("WORKFLOW_EMBEDDING_PERSIST_ROOT", ".vectorstore")

        return cls(
            default_top_k=env_int("WORKFLOW_WIKI_TOP_K", default_top_k, minimum=1),
            max_chunks_per_doc=env_int("WORKFLOW_WIKI_MAX_CHUNKS_PER_DOC", 1, minimum=1),
            chunk_size=env_int("WORKFLOW_WIKI_PARAGRAPH_MAX_CHARS", 520, minimum=120),
            chunk_overlap=env_int("WORKFLOW_WIKI_PARAGRAPH_MIN_CHARS", 80, minimum=0),
            excerpt_max_chars=env_int("WORKFLOW_WIKI_EXCERPT_MAX_CHARS", 220, minimum=60),
            candidate_multiplier=env_int("WORKFLOW_WIKI_STAGE2_MULTIPLIER", 6, minimum=1),
            min_candidates=env_int("WORKFLOW_WIKI_STAGE2_MIN_CANDIDATES", 12, minimum=1),
            rg_strategy=strategy,
            rg_max_terms=env_int("WORKFLOW_WIKI_RG_MAX_TERMS", 8, minimum=1),
            rg_timeout_ms=env_int("WORKFLOW_WIKI_RG_TIMEOUT_MS", 1200, minimum=100),
            rg_path_boost=env_float("WORKFLOW_WIKI_RG_PATH_BOOST", 0.55, minimum=0.0),
            rg_max_matches_per_term=env_int("WORKFLOW_WIKI_RG_MAX_MATCHES_PER_TERM", 80, minimum=1),
            enable_embedding=enable_embedding,
            embedding_model=embedding_model,
            embedding_top_k=embedding_top_k,
            embedding_device=embedding_device,
            embedding_persist_root=embedding_persist_root,
        )


class MarkdownWikiRetriever:
    """
    定义`MarkdownWikiRetriever`，用于封装相关数据结构与处理行为。
    """
    MODULE_DOC_PRIOR_BOOST: float = 0.45  # 进一步提升模块prior boost，增强模块特异性文档的排名
    GENERAL_DOC_PENALTY: float = 0.15  # 通用文档（如总体架构）的惩罚系数

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
        module_prior_boost: float = 0.25,
        reranker_profile: RerankerProfile | None = None,
    ) -> None:
        """
        初始化 Wiki 检索器。

        参数:
            self: 当前对象实例。
            wiki_dir: Wiki 文档目录
            project_root: 项目根目录
            default_top_k: 默认返回结果数
            module_doc_hints: 模块文档提示词
            embedding_profile: 领域配置中的 EmbeddingProfile
            hybrid_weights_profile: profile 中的 hybrid_weights 配置
            module_prior_boost: 模块 prior boost 权重
            reranker_profile: Cross-Encoder 重排器配置

        返回:
            无返回值。
        """
        self.wiki_dir = wiki_dir
        self.project_root = project_root
        self.module_doc_hints = dict(module_doc_hints or {})
        self.MODULE_DOC_PRIOR_BOOST = module_prior_boost  # 使用传入的配置
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
        self._rg_executable = shutil.which("rg") if self.runtime_config.rg_strategy != "no_rg" else None
        self._rg_unavailable_warned = False

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
        self._logger.info(
            "workflow.wiki_rg.status",
            strategy=self.runtime_config.rg_strategy,
            enabled=self.runtime_config.rg_strategy != "no_rg",
            available=bool(self._rg_executable),
            executable=self._rg_executable or "",
        )
        if self.runtime_config.rg_strategy in {"rg_first", "rg_only"} and self._rg_executable is None:
            self._warn_rg_unavailable("rg is unavailable; retrieve_wiki falls back to BM25/TFIDF only.")

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
            from workflow.retrievers.cross_encoder_reranker import (
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
        rg_strategy = self.runtime_config.rg_strategy
        use_rg = rg_strategy in {"rg_first", "rg_only"}
        use_semantic = rg_strategy in {"rg_first", "no_rg"}
        if (
            not user_query.strip()
            or not self._documents
            or (use_semantic and self._bm25 is None)
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

        if use_semantic:
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
        rg_terms = self._build_rg_terms(query_terms=query_terms, module_name=module_name or "") if use_rg else []
        rg_path_hits, rg_profile = self._run_rg_path_hits(terms=rg_terms)
        rg_boost_by_chunk: dict[int, float] = {}
        rg_candidate_ids: list[int] = []
        for path, match_count in rg_path_hits.items():
            chunk_ids = self._chunk_ids_by_path.get(path, [])
            if not chunk_ids:
                continue
            path_boost = min(
                self.runtime_config.rg_path_boost * (1.0 + 0.25 * max(match_count - 1, 0)),
                self.runtime_config.rg_path_boost * 2.5,
            )
            for chunk_id in chunk_ids:
                rg_candidate_ids.append(chunk_id)
                rg_boost_by_chunk[chunk_id] = max(rg_boost_by_chunk.get(chunk_id, 0.0), path_boost)

        candidate_ids: list[int] = []
        if rg_strategy in {"rg_first", "rg_only"}:
            candidate_ids.extend(rg_candidate_ids)
        if use_semantic:
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
            rg_boost = rg_boost_by_chunk.get(chunk_id, 0.0)
            path_key = self._to_relative_path(chunk.source_path)

            # 计算通用文档惩罚：当查询匹配到具体模块时，对总体架构等通用文档降低分数
            general_doc_penalty = self._general_doc_penalty(
                chunk=chunk,
                module_name=module_name or "",
                module_boost=module_boost,
            )

            # 精简后的三级召回评分公式：bm25*0.30 + embedding*0.50 + lexical*0.20 + module_boost - penalty
            score = 0.0
            if use_semantic:
                score = (
                    bm25_score * self.hybrid_weights.bm25
                    + embedding_score * self.hybrid_weights.embedding
                    + lexical_score * self.hybrid_weights.lexical
                    + module_boost
                    - general_doc_penalty
                )
            if rg_strategy == "rg_first":
                score += rg_boost
            elif rg_strategy == "rg_only":
                score = rg_boost * 4.0 + lexical_score * 0.8 + module_boost
            scored.append(
                {
                    "chunk": chunk,
                    "score": score,
                    "bm25_score": bm25_score,
                    "embedding_score": embedding_score,
                    "lexical_score": lexical_score,
                    "module_boost": module_boost,
                    "general_doc_penalty": general_doc_penalty,
                    "rg_boost": rg_boost,
                    "rg_path_hits": int(rg_path_hits.get(path_key, 0)),
                    "rg_strategy": rg_strategy,
                }
            )

        scored.sort(key=lambda item: item["score"], reverse=True)
        selected = self._select_diverse(scored, top_n=final_top_k)

        hits: list[dict[str, Any]] = []
        for rank, item in enumerate(selected, start=1):
            chunk = item["chunk"]
            hits.append(
                {
                    "source_type": "wiki",
                    "title": chunk.title,
                    "path": self._to_relative_path(chunk.source_path),
                    "score": round(float(item["score"]), 4),
                    "section": chunk.section,
                    "chunk_type": chunk.chunk_type,
                    "excerpt": self._build_excerpt(chunk.content),
                    "rank": rank,
                    "retrieval_debug": {
                        "bm25": round(float(item["bm25_score"]), 4),
                        "embedding": round(float(item["embedding_score"]), 4),
                        "lexical": round(float(item["lexical_score"]), 4),
                        "module_boost": round(float(item["module_boost"]), 4),
                        "general_doc_penalty": round(float(item.get("general_doc_penalty", 0.0)), 4),
                        "rg_boost": round(float(item["rg_boost"]), 4),
                        "rg_path_hits": int(item["rg_path_hits"]),
                        "rg_strategy": str(item["rg_strategy"]),
                        "weights": {
                            "bm25": round(self.hybrid_weights.bm25, 4),
                            "embedding": round(self.hybrid_weights.embedding, 4),
                            "lexical": round(self.hybrid_weights.lexical, 4),
                        },
                    },
                }
            )

        self.last_search_profile = {
            "latency_ms": round((perf_counter() - started) * 1000, 3),
            "candidate_k": candidate_k,
            "expanded_query_count": len(expanded_queries),
            "hits": len(hits),
            "rg_strategy": rg_strategy,
            "rg": {
                **rg_profile,
                "matched_paths": len(rg_path_hits),
                "matched_chunks": len(rg_boost_by_chunk),
            },
        }

        # Cross-Encoder 重排（可选）
        if self._reranker and hits:
            rerank_start = perf_counter()
            # 为重排器准备候选集，添加 content 字段
            rerank_candidates = [
                {
                    **hit,
                    "content": self._get_chunk_content_by_path(hit["path"], hit["section"]),
                }
                for hit in hits
            ]
            # 执行重排
            reranked_hits = self._reranker.rerank(
                query=user_query,
                candidates=rerank_candidates,
                top_k=final_top_k,
                content_key="content",
            )
            # 更新分数和排名
            hits = []
            for rank, hit in enumerate(reranked_hits, start=1):
                rerank_score = hit.get("rerank_score", hit.get("score", 0.0))
                hits.append({
                    **hit,
                    "score": round(float(rerank_score), 4),
                    "rank": rank,
                    "score_source": "reranker",
                    "retrieval_debug": {
                        **hit.get("retrieval_debug", {}),
                        "rerank_score": round(float(rerank_score), 4),
                        "original_rank": hit.get("original_rank", rank),
                    },
                })
            rerank_latency = round((perf_counter() - rerank_start) * 1000, 3)
            self.last_search_profile["rerank"] = {
                "enabled": True,
                "latency_ms": rerank_latency,
                "model": self._reranker_profile.model if self._reranker_profile else None,
            }
            self.last_search_profile["latency_ms"] = round((perf_counter() - started) * 1000, 3)

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
        """
        构建当前步骤所需的数据结构或文本内容。
        
        参数:
            self: 当前对象实例。
        
        返回:
            返回类型为 `dict[str, int]` 的处理结果。
        """
        stats = {"file_count": 0, "chunk_count": 0}
        if not self.wiki_dir.exists():
            return stats

        markdown_files = sorted(self.wiki_dir.rglob("*.md"))
        stats["file_count"] = len(markdown_files)

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

        stats["chunk_count"] = len(self._chunks)
        if not self._documents:
            return stats

        # 只保留 BM25 索引（精简方案）
        self._bm25 = BM25Retriever.from_documents(self._documents)
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
        内部辅助函数，负责`rank score` 相关处理。
        
        参数:
            self: 当前对象实例。
            rank: 输入参数，用于控制当前处理逻辑。
        
        返回:
            返回类型为 `float` 的处理结果。
        """
        if rank is None:
            return 0.0
        # Keep score range compatible with existing grade thresholds in node layer.
        return 12.0 / (rank + 1.0)

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

    def _build_rg_terms(self, *, query_terms: list[str], module_name: str) -> list[str]:
        """
        构建当前步骤所需的数据结构或文本内容。
        
        参数:
            self: 当前对象实例。
        
        返回:
            返回类型为 `list[str]` 的处理结果。
        """
        terms: list[str] = []
        terms.extend(query_terms)
        terms.extend([token for token in re.findall(r"[a-z0-9_+-]{2,}", module_name.lower()) if token])
        deduped = list(dict.fromkeys([term.strip() for term in terms if len(term.strip()) >= 2]))
        return deduped[: self.runtime_config.rg_max_terms]

    def _run_rg_path_hits(self, *, terms: list[str]) -> tuple[Counter[str], dict[str, Any]]:
        """
        内部辅助函数，负责`run rg path hits` 相关处理。
        
        参数:
            self: 当前对象实例。
        
        返回:
            返回类型为 `tuple[Counter[str], dict[str, Any]]` 的处理结果。
        """
        rg_enabled = self.runtime_config.rg_strategy in {"rg_first", "rg_only"}
        profile: dict[str, Any] = {
            "strategy": self.runtime_config.rg_strategy,
            "enabled": bool(rg_enabled),
            "available": bool(self._rg_executable),
            "term_count": len(terms),
            "timeout_ms": int(self.runtime_config.rg_timeout_ms),
        }
        if not rg_enabled:
            return Counter(), profile
        if self._rg_executable is None:
            warning = "rg is unavailable; retrieve_wiki falls back to BM25/TFIDF only."
            if self.runtime_config.rg_strategy == "rg_only":
                warning = "rg is unavailable; retrieve_wiki rg_only strategy cannot run."
            self._warn_rg_unavailable(warning)
            profile["warning"] = warning
            return Counter(), profile
        if not terms or not self.wiki_dir.exists():
            return Counter(), profile

        command: list[str] = [
            self._rg_executable,
            "--json",
            "--line-number",
            "--no-heading",
            "--smart-case",
            "--fixed-strings",
            "--max-count",
            str(self.runtime_config.rg_max_matches_per_term),
        ]
        for term in terms:
            command.extend(["-e", term])
        command.extend(["--glob", "*.md", str(self.wiki_dir)])

        started = perf_counter()
        try:
            result = subprocess.run(
                command,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                timeout=float(self.runtime_config.rg_timeout_ms) / 1000.0,
                check=False,
            )
        except subprocess.TimeoutExpired:
            warning = "rg timed out during wiki retrieval; fallback to BM25/TFIDF only."
            if self.runtime_config.rg_strategy == "rg_only":
                warning = "rg timed out during wiki retrieval; rg_only strategy cannot return results."
            self._logger.warning(
                "workflow.wiki_rg.timeout",
                timeout_ms=int(self.runtime_config.rg_timeout_ms),
                term_count=len(terms),
            )
            profile["warning"] = warning
            profile["timed_out"] = True
            return Counter(), profile
        except Exception as exc:
            warning = "rg failed during wiki retrieval; fallback to BM25/TFIDF only."
            if self.runtime_config.rg_strategy == "rg_only":
                warning = "rg failed during wiki retrieval; rg_only strategy cannot return results."
            self._logger.warning(
                "workflow.wiki_rg.error",
                reason=str(exc),
                term_count=len(terms),
            )
            profile["warning"] = warning
            return Counter(), profile

        path_hits: Counter[str] = Counter()
        for line in result.stdout.splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(payload.get("type", "")) != "match":
                continue
            path_raw = str(payload.get("data", {}).get("path", {}).get("text", "")).strip()
            if not path_raw:
                continue
            normalized_path = self._normalize_rg_path(path_raw)
            if not normalized_path:
                continue
            path_hits[normalized_path] += 1

        if result.returncode not in (0, 1):
            stderr = result.stderr.strip().splitlines()
            self._logger.warning(
                "workflow.wiki_rg.nonzero_exit",
                returncode=int(result.returncode),
                stderr=(stderr[0] if stderr else ""),
            )
            profile["warning"] = "rg exited with non-zero status during wiki retrieval; fallback results may degrade."
        profile["latency_ms"] = round((perf_counter() - started) * 1000, 3)
        profile["raw_match_count"] = sum(path_hits.values())
        return path_hits, profile

    def _lexical_match_score(self, text: str, terms: list[str]) -> float:
        """
        内部辅助函数，负责`lexical match score` 相关处理。
        
        参数:
            self: 当前对象实例。
            text: 待处理的文本内容。
            terms: 列表参数，用于承载批量输入数据。
        
        返回:
            返回类型为 `float` 的处理结果。
        """
        if not terms:
            return 0.0
        hits = sum(1 for term in terms if term in text)
        if hits <= 0:
            return 0.0
        coverage = hits / max(len(terms), 1)
        return min(2.0, coverage * 2.0)

    def _module_prior_boost(self, *, chunk: WikiChunk, module_name: str) -> float:
        """
        内部辅助函数，负责`module prior boost` 相关处理。

        优化点：
        1. 增加对文档标题的匹配权重（标题匹配权重更高）
        2. 提高最大 boost 限制，增强模块特异性文档的排名
        3. 路径、标题、文本三种匹配独立计算，可叠加

        参数:
            self: 当前对象实例。
            chunk: 文档块对象
            module_name: 模块名称

        返回:
            返回类型为 `float` 的处理结果。
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
            # 路径匹配（如 "04-" 匹配文件名前缀）
            if hint_lower in path_key:
                score += 1.0
            # 标题匹配（权重最高，因为标题更准确反映文档主题）
            if hint_lower in title_lower:
                score += 2.0
            # 文本内容匹配（权重较低）
            if hint_lower in text_lower:
                score += 0.5

        if score <= 0:
            return 0.0

        # 提高最大 boost 限制到 0.6，增强模块特异性文档的排名
        return min(self.MODULE_DOC_PRIOR_BOOST * score, 0.6)

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
        if module_boost < 0.3:
            # 如果模块 boost 很低，说明总体架构文档不是查询的主要目标
            extra_penalty = 0.1

        return base_penalty + extra_penalty

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

    def _normalize_rg_path(self, raw_path: str) -> str:
        """
        内部辅助函数，负责`normalize rg path` 相关处理。
        
        参数:
            self: 当前对象实例。
            raw_path: 路径参数，用于定位文件或目录。
        
        返回:
            返回类型为 `str` 的处理结果。
        """
        try:
            candidate = Path(raw_path)
            if not candidate.is_absolute():
                candidate = (self.project_root / candidate).resolve()
            return self._to_relative_path(candidate)
        except Exception:
            return ""

    def _warn_rg_unavailable(self, warning: str) -> None:
        """
        内部辅助函数，负责`warn rg unavailable` 相关处理。
        
        参数:
            self: 当前对象实例。
            warning: 输入参数，用于控制当前处理逻辑。
        
        返回:
            无返回值。
        """
        if self._rg_unavailable_warned:
            return
        self._rg_unavailable_warned = True
        self._logger.warning("workflow.wiki_rg.unavailable", warning=warning)

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
