# -*- coding: utf-8 -*-
"""
该模块实现工作流节点`retrieve_code` 的处理逻辑，负责读取状态并输出增量结果。
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
from typing import Any, Iterator

from langchain_community.retrievers import BM25Retriever, TFIDFRetriever
from langchain_core.documents import Document

from workflow.retrievers import WeightedFusionRetriever
from workflow.common.runtime_logging import get_file_logger
from workflow.common.func_utils import env_bool, env_float, env_int
from workflow.common.domain_profile import RerankerProfile


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
class CodeParentChunk:
    """
    定义`CodeParentChunk`，用于封装相关数据结构与处理行为。
    """
    parent_id: str
    source_path: Path
    language: str
    chunk_type: str
    symbol_name: str
    signature: str
    start_line: int
    end_line: int
    content: str
    normalized_text: str
    normalized_path: str
    normalized_symbol: str


@dataclass
class CodeChildChunk:
    """
    定义`CodeChildChunk`，用于封装相关数据结构与处理行为。
    """
    child_id: str
    parent_id: str
    source_path: Path
    language: str
    chunk_type: str
    symbol_name: str
    signature: str
    start_line: int
    end_line: int
    content: str
    normalized_text: str
    normalized_path: str
    normalized_symbol: str


@dataclass
class CodeRetrieverRuntimeConfig:
    """
    定义`CodeRetrieverRuntimeConfig`，用于封装相关数据结构与处理行为。
    """
    default_top_k: int = 4
    max_child_candidates: int = 64
    max_results_per_path: int = 2
    semantic_weight: float = 4.0
    pattern_weight: float = 1.6
    parent_best_pattern_weight: float = 0.9
    parent_avg_pattern_weight: float = 0.35
    min_final_score: float = 0.25
    grade_high_top1_threshold: float = 8.0
    grade_medium_top1_threshold: float = 4.0
    rg_strategy: str = "rg_first"
    rg_max_terms: int = 10
    rg_timeout_ms: int = 1200
    rg_path_boost: float = 1.0
    rg_line_boost: float = 2.2
    rg_max_matches_per_term: int = 120
    rg_max_lines_per_path: int = 40

    @classmethod
    def from_env(cls) -> "CodeRetrieverRuntimeConfig":
        """
        执行`from env` 相关处理逻辑。
        
        参数:
            cls: 当前类对象。
        
        返回:
            返回类型为 `'CodeRetrieverRuntimeConfig'` 的处理结果。
        """
        legacy_flag = os.getenv("WORKFLOW_CODE_RG_FIRST_ENABLED")
        if legacy_flag is None:
            default_strategy = "rg_first"
        else:
            default_strategy = "rg_first" if env_bool("WORKFLOW_CODE_RG_FIRST_ENABLED", True) else "no_rg"
        strategy = _resolve_rg_strategy(
            os.getenv("WORKFLOW_CODE_RG_STRATEGY"),
            default=default_strategy,
        )
        return cls(
            default_top_k=env_int("WORKFLOW_CODE_RETRIEVER_TOP_K", 4, minimum=1),
            max_child_candidates=env_int("WORKFLOW_CODE_RETRIEVER_MAX_CHILD_CANDIDATES", 64, minimum=8),
            max_results_per_path=env_int("WORKFLOW_CODE_RETRIEVER_MAX_PER_PATH", 2, minimum=1),
            semantic_weight=env_float("WORKFLOW_CODE_RETRIEVER_SEMANTIC_WEIGHT", 4.0, minimum=0.0),
            pattern_weight=env_float("WORKFLOW_CODE_RETRIEVER_PATTERN_WEIGHT", 1.6, minimum=0.0),
            parent_best_pattern_weight=env_float("WORKFLOW_CODE_RETRIEVER_PARENT_BEST_PATTERN_WEIGHT", 0.9, minimum=0.0),
            parent_avg_pattern_weight=env_float("WORKFLOW_CODE_RETRIEVER_PARENT_AVG_PATTERN_WEIGHT", 0.35, minimum=0.0),
            min_final_score=env_float("WORKFLOW_CODE_RETRIEVER_MIN_FINAL_SCORE", 0.25, minimum=0.0),
            grade_high_top1_threshold=env_float("WORKFLOW_CODE_RETRIEVER_GRADE_HIGH_TOP1_THRESHOLD", 8.0, minimum=0.0),
            grade_medium_top1_threshold=env_float("WORKFLOW_CODE_RETRIEVER_GRADE_MEDIUM_TOP1_THRESHOLD", 4.0, minimum=0.0),
            rg_strategy=strategy,
            rg_max_terms=env_int("WORKFLOW_CODE_RG_MAX_TERMS", 10, minimum=1),
            rg_timeout_ms=env_int("WORKFLOW_CODE_RG_TIMEOUT_MS", 1200, minimum=100),
            rg_path_boost=env_float("WORKFLOW_CODE_RG_PATH_BOOST", 1.0, minimum=0.0),
            rg_line_boost=env_float("WORKFLOW_CODE_RG_LINE_BOOST", 2.2, minimum=0.0),
            rg_max_matches_per_term=env_int("WORKFLOW_CODE_RG_MAX_MATCHES_PER_TERM", 120, minimum=1),
            rg_max_lines_per_path=env_int("WORKFLOW_CODE_RG_MAX_LINES_PER_PATH", 40, minimum=1),
        )


class LocalCodeRetriever:
    """
    定义`LocalCodeRetriever`，用于封装相关数据结构与处理行为。
    """
    SUPPORTED_EXTENSIONS = {".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".java", ".sql", ".yaml", ".yml", ".toml", ".ini", ".conf", ".sh"}
    IGNORED_DIR_NAMES = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", "node_modules", ".venv", "venv", "dist", "build", "target"}
    MAX_FILE_SIZE_BYTES = 256 * 1024
    CHILD_CHUNK_LINES = 36
    CHILD_CHUNK_OVERLAP = 8
    EXCERPT_MAX_LINES = 18
    EXCERPT_CONTEXT_RADIUS = 2

    def __init__(
        self,
        *,
        project_root: Path,
        code_dirs: list[Path] | None = None,
        default_top_k: int = 4,
        runtime_config: CodeRetrieverRuntimeConfig | None = None,
        reranker_profile: RerankerProfile | None = None,
    ) -> None:
        """
        初始化代码检索器。

        参数:
            project_root: 项目根目录
            code_dirs: 代码目录列表
            default_top_k: 默认返回结果数
            runtime_config: 运行时配置
            reranker_profile: Cross-Encoder 重排器配置
        """
        self.project_root = project_root
        self._logger = get_file_logger(project_root=project_root)
        self.runtime_config = runtime_config or CodeRetrieverRuntimeConfig.from_env()
        if default_top_k > 0:
            self.runtime_config.default_top_k = default_top_k
        self.default_top_k = self.runtime_config.default_top_k

        if code_dirs is None:
            default_codes_dir = project_root / "codes"
            self.code_dirs = [default_codes_dir] if default_codes_dir.exists() else []
        else:
            self.code_dirs = [d for d in code_dirs if d.exists()]

        self._parent_chunks: dict[str, CodeParentChunk] = {}
        self._child_chunks: list[CodeChildChunk] = []
        self._child_by_id: dict[str, CodeChildChunk] = {}
        self._child_docs: list[Document] = []
        self._children_by_path: defaultdict[str, list[CodeChildChunk]] = defaultdict(list)
        self._symbol_index: dict[str, set[str]] = defaultdict(set)
        self._path_token_index: dict[str, set[str]] = defaultdict(set)
        self.last_search_profile: dict[str, Any] = {}
        self._index_read_error_count = 0
        self._rg_executable = shutil.which("rg") if self.runtime_config.rg_strategy != "no_rg" else None
        self._rg_unavailable_warned = False

        self._bm25: BM25Retriever | None = None
        self._tfidf: TFIDFRetriever | None = None
        self._ensemble: WeightedFusionRetriever | None = None

        # Cross-Encoder 重排器（可选）
        self._reranker: Any = None
        self._reranker_profile = reranker_profile

        started_at = perf_counter()
        index_stats = self._build_index()
        self._logger.info(
            "workflow.code_index.built",
            code_dirs=[str(path) for path in self.code_dirs],
            scanned_file_count=index_stats.get("scanned_file_count", 0),
            indexed_file_count=index_stats.get("indexed_file_count", 0),
            parent_chunk_count=index_stats.get("parent_chunk_count", 0),
            child_chunk_count=index_stats.get("child_chunk_count", 0),
            read_error_count=self._index_read_error_count,
            latency_ms=int((perf_counter() - started_at) * 1000),
        )
        self._logger.info(
            "workflow.code_rg.status",
            strategy=self.runtime_config.rg_strategy,
            enabled=self.runtime_config.rg_strategy != "no_rg",
            available=bool(self._rg_executable),
            executable=self._rg_executable or "",
        )
        if self.runtime_config.rg_strategy in {"rg_first", "rg_only"} and self._rg_executable is None:
            self._warn_rg_unavailable("rg is unavailable; retrieve_code falls back to BM25/TFIDF only.")

        # 初始化 Cross-Encoder 重排器（可选）
        if self._reranker_profile and self._reranker_profile.enabled and not self._reranker:
            self._init_reranker()

        # 记录重排器状态
        self._logger.info(
            "workflow.code_reranker.status",
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
                "workflow.code_reranker.initialized",
                model=self._reranker_profile.model,
                top_k=self._reranker_profile.top_k,
                candidate_top_k=self._reranker_profile.candidate_top_k,
            )
        except ImportError as e:
            self._logger.warning(
                "workflow.code_reranker.import_error",
                error=str(e),
                message="CrossEncoderReranker 未安装，跳过重排器初始化",
            )
            self._reranker = None
        except Exception as e:
            self._logger.error(
                "workflow.code_reranker.init_error",
                error=str(e),
            )
            self._reranker = None

    def search(self, *, user_query: str, retrieval_queries: list[str] | None = None, module_name: str | None = None, top_k: int | None = None) -> list[dict[str, Any]]:
        """
        执行`search` 相关处理逻辑。
        
        参数:
            self: 当前对象实例。
        
        返回:
            返回类型为 `list[dict[str, Any]]` 的处理结果。
        """
        started = perf_counter()
        query = user_query.strip()
        rg_strategy = self.runtime_config.rg_strategy
        use_rg = rg_strategy in {"rg_first", "rg_only"}
        use_semantic = rg_strategy in {"rg_first", "no_rg"}
        if not query or not self._child_chunks or (use_semantic and (self._bm25 is None or self._tfidf is None or self._ensemble is None)):
            self.last_search_profile = {"latency_ms": round((perf_counter() - started) * 1000, 3)}
            return []

        query_phrases = [query, *(retrieval_queries or [])]
        merged_query = " ".join([*query_phrases, module_name or ""]).strip()
        patterns = self._extract_patterns(merged_query)
        module_tokens = {tok for tok in re.split(r"[._\-/\s]+", self._normalize(module_name or "")) if len(tok) >= 2}
        rg_terms = self._build_rg_terms(patterns=patterns, module_tokens=module_tokens) if use_rg else []
        rg_path_hits, rg_line_hits_by_path, rg_profile = self._run_rg_line_hits(terms=rg_terms)
        rg_path_boost_by_child: dict[str, float] = {}
        rg_line_boost_by_child: dict[str, float] = {}
        rg_candidate_child_ids: list[str] = []
        for path, hit_count in rg_path_hits.items():
            children = self._children_by_path.get(path, [])
            if not children:
                continue
            path_boost = min(
                self.runtime_config.rg_path_boost * (1.0 + 0.2 * max(hit_count - 1, 0)),
                self.runtime_config.rg_path_boost * 2.5,
            )
            path_lines = set(rg_line_hits_by_path.get(path, []))
            path_candidates = 0
            for child in children:
                rg_path_boost_by_child[child.child_id] = max(rg_path_boost_by_child.get(child.child_id, 0.0), path_boost)
                line_hit_count = 0
                for line_no in path_lines:
                    if child.start_line <= line_no <= child.end_line:
                        line_hit_count += 1
                if line_hit_count > 0:
                    line_boost = min(
                        self.runtime_config.rg_line_boost * line_hit_count,
                        self.runtime_config.rg_line_boost * 3.0,
                    )
                    rg_line_boost_by_child[child.child_id] = max(rg_line_boost_by_child.get(child.child_id, 0.0), line_boost)
                    rg_candidate_child_ids.append(child.child_id)
                    continue
                if path_candidates < 2:
                    rg_candidate_child_ids.append(child.child_id)
                    path_candidates += 1

        candidate_k = max(self.runtime_config.max_child_candidates, (top_k or self.runtime_config.default_top_k) * 8)
        bm25_docs: list[Document] = []
        tfidf_docs: list[Document] = []
        ens_docs: list[Document] = []
        if use_semantic:
            assert self._bm25 is not None and self._tfidf is not None and self._ensemble is not None
            self._bm25.k = candidate_k
            self._tfidf.k = candidate_k
            bm25_docs = self._bm25.invoke(merged_query)
            tfidf_docs = self._tfidf.invoke(merged_query)
            ens_docs = self._ensemble.invoke(merged_query)

        bm25_rank = self._rank_map(bm25_docs)
        tfidf_rank = self._rank_map(tfidf_docs)
        ens_rank = self._rank_map(ens_docs)

        candidate_child_ids = list(dict.fromkeys([
            *(rg_candidate_child_ids if rg_strategy in {"rg_first", "rg_only"} else []),
            *[str(doc.metadata.get("child_id", "")) for doc in ens_docs],
            *[str(doc.metadata.get("child_id", "")) for doc in bm25_docs],
            *[str(doc.metadata.get("child_id", "")) for doc in tfidf_docs],
        ]))[: self.runtime_config.max_child_candidates]

        scored_children: list[dict[str, Any]] = []
        for child_id in candidate_child_ids:
            child = self._child_by_id.get(child_id)
            if child is None:
                continue
            lexical, matched_terms = self._score_lexical(child, patterns)
            if module_tokens and any(tok in child.normalized_path for tok in module_tokens):
                lexical += 0.8
            pattern_score, matched_patterns = self._score_pattern(child, patterns)
            rg_path_boost = rg_path_boost_by_child.get(child_id, 0.0)
            rg_line_boost = rg_line_boost_by_child.get(child_id, 0.0)
            score = lexical
            if use_semantic:
                score += self._rank_score(bm25_rank.get(child_id)) * 0.6
                score += self._rank_score(max(tfidf_rank.get(child_id, 10**9), ens_rank.get(child_id, 10**9))) * self.runtime_config.semantic_weight
                score += pattern_score * self.runtime_config.pattern_weight
            if rg_strategy == "rg_first":
                score += rg_path_boost + rg_line_boost
            elif rg_strategy == "rg_only":
                score = rg_path_boost * 1.4 + rg_line_boost * 1.2 + lexical * 0.7 + pattern_score * 0.8
            if score < self.runtime_config.min_final_score:
                continue
            scored_children.append(
                {
                    "child": child,
                    "score": score,
                    "pattern_score": pattern_score,
                    "matched_terms": matched_terms,
                    "matched_patterns": matched_patterns,
                    "rg_path_boost": rg_path_boost,
                    "rg_line_boost": rg_line_boost,
                    "rg_strategy": rg_strategy,
                }
            )

        scored_children.sort(key=lambda item: (item["score"], item["pattern_score"]), reverse=True)
        parent_buckets: dict[str, dict[str, Any]] = {}
        for item in scored_children:
            child = item["child"]
            parent = self._parent_chunks.get(child.parent_id)
            if parent is None:
                continue
            bucket = parent_buckets.setdefault(
                parent.parent_id,
                {
                    "parent": parent,
                    "best_child": child,
                    "best_child_score": item["score"],
                    "best_pattern_score": item["pattern_score"],
                    "scores": [],
                    "pattern_scores": [],
                    "matched_terms": set(),
                    "matched_patterns": set(),
                    "hit_count": 0,
                    "best_rg_path_boost": item["rg_path_boost"],
                    "best_rg_line_boost": item["rg_line_boost"],
                },
            )
            bucket["scores"].append(item["score"])
            bucket["pattern_scores"].append(item["pattern_score"])
            bucket["matched_terms"].update(item["matched_terms"])
            bucket["matched_patterns"].update(item["matched_patterns"])
            bucket["hit_count"] += 1
            if item["score"] > bucket["best_child_score"]:
                bucket["best_child"] = child
                bucket["best_child_score"] = item["score"]
            if item["pattern_score"] > bucket["best_pattern_score"]:
                bucket["best_pattern_score"] = item["pattern_score"]
            bucket["best_rg_path_boost"] = max(bucket["best_rg_path_boost"], item["rg_path_boost"])
            bucket["best_rg_line_boost"] = max(bucket["best_rg_line_boost"], item["rg_line_boost"])

        parent_items: list[dict[str, Any]] = []
        for bucket in parent_buckets.values():
            top_scores = sorted(bucket["scores"], reverse=True)[:2]
            top_pattern = sorted(bucket["pattern_scores"], reverse=True)[:2]
            parent_path = self._to_relative_path(bucket["parent"].source_path)
            final_score = bucket["best_child_score"] + (sum(top_scores) / max(len(top_scores), 1)) * 0.25
            final_score += bucket["best_pattern_score"] * self.runtime_config.parent_best_pattern_weight
            final_score += (sum(top_pattern) / max(len(top_pattern), 1)) * self.runtime_config.parent_avg_pattern_weight
            parent_items.append(
                {
                    "parent": bucket["parent"],
                    "best_child": bucket["best_child"],
                    "best_child_score": bucket["best_child_score"],
                    "best_pattern_score": bucket["best_pattern_score"],
                    "final_score": final_score,
                    "matched_terms": sorted(bucket["matched_terms"], key=len, reverse=True),
                    "matched_patterns": sorted(bucket["matched_patterns"], key=len, reverse=True),
                    "hit_count": bucket["hit_count"],
                    "rg_path_boost": bucket["best_rg_path_boost"],
                    "rg_line_boost": bucket["best_rg_line_boost"],
                    "rg_path_hits": int(rg_path_hits.get(parent_path, 0)),
                    "rg_strategy": rg_strategy,
                }
            )
        parent_items.sort(key=lambda item: (item["final_score"], item["best_pattern_score"]), reverse=True)

        selected: list[dict[str, Any]] = []
        per_path: Counter[str] = Counter()
        for item in parent_items:
            path = self._to_relative_path(item["parent"].source_path)
            if per_path[path] >= self.runtime_config.max_results_per_path:
                continue
            per_path[path] += 1
            selected.append(item)
            if len(selected) >= (top_k or self.runtime_config.default_top_k):
                break

        hits: list[dict[str, Any]] = []
        for rank, item in enumerate(selected, start=1):
            parent = item["parent"]
            best_child = item["best_child"]
            excerpt = self._build_excerpt(best_child.content, best_child.start_line, item["matched_terms"], item["matched_patterns"])
            hits.append(
                {
                    "source_type": "code",
                    "title": parent.symbol_name or parent.source_path.name,
                    "path": self._to_relative_path(parent.source_path),
                    "score": round(float(item["final_score"]), 4),
                    "stage1_score": round(float(item["best_child_score"]), 4),
                    "excerpt": excerpt["excerpt_text"],
                    "excerpt_lines": excerpt["excerpt_lines"],
                    "highlight_lines": excerpt["highlight_lines"],
                    "section": f"{parent.chunk_type}:{parent.symbol_name or 'file'}",
                    "language": parent.language,
                    "chunk_type": parent.chunk_type,
                    "symbol_name": parent.symbol_name,
                    "signature": parent.signature,
                    "start_line": parent.start_line,
                    "end_line": parent.end_line,
                    "rank": rank,
                    "retrieval_debug": {
                        "matched_terms": item["matched_terms"][:8],
                        "matched_patterns": item["matched_patterns"][:8],
                        "parent_hit_count": item["hit_count"],
                        "pattern_score": round(float(item["best_pattern_score"]), 4),
                        "rg_path_boost": round(float(item["rg_path_boost"]), 4),
                        "rg_line_boost": round(float(item["rg_line_boost"]), 4),
                        "rg_path_hits": int(item["rg_path_hits"]),
                        "rg_strategy": str(item["rg_strategy"]),
                    },
                }
            )

        self.last_search_profile = {
            "latency_ms": round((perf_counter() - started) * 1000, 3),
            "child_candidates": len(scored_children),
            "parent_candidates": len(parent_items),
            "selected_count": len(hits),
            "rg_strategy": rg_strategy,
            "rg": {
                **rg_profile,
                "matched_paths": len(rg_path_hits),
                "matched_children": len(rg_candidate_child_ids),
            },
        }

        # Cross-Encoder 重排（可选）
        if self._reranker and hits:
            rerank_start = perf_counter()
            # 为重排器准备候选集，添加 content 字段
            rerank_candidates = [
                {
                    **hit,
                    "content": self._get_chunk_content_by_hit(
                        hit["path"],
                        hit["section"],
                        hit.get("excerpt_lines"),
                    ),
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
            for rank, hit in enumerate(reranked_hits, start= 1):
                hits.append({
                    **hit,
                    "score": round(float(hit.get("rerank_score", 0.0 or 0.0)), 4),
                    "rank": rank,
                    "score_source": "reranker",
                    "retrieval_debug": {
                        **hit.get("retrieval_debug", {}),
                        "rerank_score": round(float(hit.get("rerank_score", 0.0), 4)),
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

    def _get_chunk_content_by_hit(self, path: str, section: str) -> str:
        """根据路径和章节获取 chunk 内容，用于重排器。

        参数:
            path: 文档相对路径
            section: 章节名称

        返回:
            chunk 内容字符串
        """
        chunk_ids = self._children_by_path.get(path, [])
        for chunk_id in chunk_ids:
            chunk = self._child_by_id.get(chunk_id)
            if chunk and chunk.section == section:
                return chunk.content
        # 如果找不到精确匹配，返回第一个 chunk 的内容
        if chunk_ids:
            chunk = self._child_by_id.get(chunk_ids[0])
            if chunk:
                return chunk.content
        return ""

    def get_index_snapshot(self) -> list[dict[str, Any]]:
        """
        执行`get index snapshot` 相关处理逻辑。
        
        参数:
            self: 当前对象实例。
        
        返回:
            返回类型为 `list[dict[str, Any]]` 的处理结果。
        """
        return [
            {
                "path": self._to_relative_path(c.source_path),
                "language": c.language,
                "chunk_type": c.chunk_type,
                "symbol_name": c.symbol_name,
                "signature": c.signature,
                "start_line": c.start_line,
                "end_line": c.end_line,
                "content_length": len(c.content),
            }
            for c in sorted(self._parent_chunks.values(), key=lambda x: (self._to_relative_path(x.source_path), x.start_line))
        ]

    def _build_index(self) -> dict[str, int]:
        """
        构建当前步骤所需的数据结构或文本内容。
        
        参数:
            self: 当前对象实例。
        
        返回:
            返回类型为 `dict[str, int]` 的处理结果。
        """
        stats = {"scanned_file_count": 0, "indexed_file_count": 0, "parent_chunk_count": 0, "child_chunk_count": 0}
        for file_path in self._iter_code_files():
            stats["scanned_file_count"] += 1
            parents = self._parse_file_to_parent_chunks(file_path)
            if not parents:
                continue
            stats["indexed_file_count"] += 1
            for parent in parents:
                self._parent_chunks[parent.parent_id] = parent
                self._register_parent(parent)
                for child in self._split_parent_to_child(parent):
                    self._child_chunks.append(child)
                    self._child_by_id[child.child_id] = child
                    self._children_by_path[self._to_relative_path(child.source_path)].append(child)
                    self._child_docs.append(
                        Document(
                            page_content=child.content,
                            metadata={
                                "child_id": child.child_id,
                                "parent_id": child.parent_id,
                                "path": self._to_relative_path(child.source_path),
                                "symbol_name": child.symbol_name,
                                "chunk_type": child.chunk_type,
                            },
                        )
                    )

        stats["parent_chunk_count"] = len(self._parent_chunks)
        stats["child_chunk_count"] = len(self._child_chunks)

        if self._child_docs:
            self._bm25 = BM25Retriever.from_documents(self._child_docs)
            self._tfidf = TFIDFRetriever.from_documents(self._child_docs)
            self._ensemble = WeightedFusionRetriever(
                retrievers=[self._bm25, self._tfidf],
                weights=[0.6, 0.4],
                id_key="child_id",
            )
        return stats

    def _iter_code_files(self) -> Iterator[Path]:
        """
        内部辅助函数，负责`iter code files` 相关处理。
        
        参数:
            self: 当前对象实例。
        
        返回:
            返回类型为 `Iterator[Path]` 的处理结果。
        """
        for directory in self.code_dirs:
            for file_path in directory.rglob("*"):
                if not file_path.is_file():
                    continue
                if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                    continue
                if any(part in self.IGNORED_DIR_NAMES for part in file_path.parts):
                    continue
                try:
                    if file_path.stat().st_size <= self.MAX_FILE_SIZE_BYTES:
                        yield file_path
                except OSError:
                    continue

    def _parse_file_to_parent_chunks(self, file_path: Path) -> list[CodeParentChunk]:
        """
        解析输入内容并转换为结构化数据。
        
        参数:
            self: 当前对象实例。
            file_path: 路径参数，用于定位文件或目录。
        
        返回:
            返回类型为 `list[CodeParentChunk]` 的处理结果。
        """
        try:
            raw = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                raw = file_path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                self._index_read_error_count += 1
                return []
        except OSError:
            self._index_read_error_count += 1
            return []

        if not raw.strip():
            return []

        language = self._infer_language(file_path)
        lines = raw.splitlines()
        if language == "python":
            try:
                tree = ast.parse(raw)
            except SyntaxError:
                tree = None
            if tree is not None:
                parents: list[CodeParentChunk] = []
                for node in ast.walk(tree):
                    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        continue
                    start = int(getattr(node, "lineno", 1))
                    end = int(getattr(node, "end_lineno", start))
                    content = "\n".join(lines[start - 1 : min(end, len(lines))]).strip()
                    if not content:
                        continue
                    symbol = node.name
                    kind = "class" if isinstance(node, ast.ClassDef) else "function"
                    signature = f"class {node.name}" if kind == "class" else self._build_python_signature(node)
                    parents.append(self._make_parent(file_path, language, kind, symbol, signature, start, end, content))
                if parents:
                    parents.append(self._make_parent(file_path, language, "file", file_path.stem, file_path.name, 1, max(len(lines), 1), raw.strip()))
                    return sorted(parents, key=lambda item: (item.start_line, item.end_line))

        return [self._make_parent(file_path, language, "file", file_path.stem, file_path.name, 1, max(len(lines), 1), raw.strip())]

    def _make_parent(self, file_path: Path, language: str, chunk_type: str, symbol_name: str, signature: str, start_line: int, end_line: int, content: str) -> CodeParentChunk:
        """
        内部辅助函数，负责`make parent` 相关处理。
        
        参数:
            self: 当前对象实例。
            file_path: 路径参数，用于定位文件或目录。
            language: 输入参数，用于控制当前处理逻辑。
            chunk_type: 输入参数，用于控制当前处理逻辑。
            symbol_name: 输入参数，用于控制当前处理逻辑。
            signature: 输入参数，用于控制当前处理逻辑。
            start_line: 输入参数，用于控制当前处理逻辑。
            end_line: 输入参数，用于控制当前处理逻辑。
            content: 输入参数，用于控制当前处理逻辑。
        
        返回:
            返回类型为 `CodeParentChunk` 的处理结果。
        """
        key = f"{file_path.as_posix()}::{chunk_type}::{symbol_name}::{start_line}:{end_line}"
        return CodeParentChunk(
            parent_id=hashlib.md5(key.encode("utf-8")).hexdigest(),
            source_path=file_path,
            language=language,
            chunk_type=chunk_type,
            symbol_name=symbol_name,
            signature=signature,
            start_line=start_line,
            end_line=end_line,
            content=content,
            normalized_text=self._normalize(content),
            normalized_path=self._normalize(self._to_relative_path(file_path)),
            normalized_symbol=self._normalize(symbol_name),
        )

    def _register_parent(self, parent: CodeParentChunk) -> None:
        """
        内部辅助函数，负责`register parent` 相关处理。
        
        参数:
            self: 当前对象实例。
            parent: 输入参数，用于控制当前处理逻辑。
        
        返回:
            无返回值。
        """
        symbol_tokens = self._extract_tokens(parent.normalized_symbol)
        signature_tokens = self._extract_tokens(parent.signature)
        for token in [*symbol_tokens, *signature_tokens]:
            if len(token) >= 2:
                self._symbol_index[token].add(parent.parent_id)
        for token in self._extract_path_tokens(parent.source_path):
            if len(token) >= 2:
                self._path_token_index[token].add(parent.parent_id)

    def _extract_path_tokens(self, source_path: Path) -> list[str]:
        """
        内部辅助函数，负责`extract path tokens` 相关处理。
        
        参数:
            self: 当前对象实例。
            source_path: 路径参数，用于定位文件或目录。
        
        返回:
            返回类型为 `list[str]` 的处理结果。
        """
        return [tok for tok in re.split(r"[\\/._\-]+", self._to_relative_path(source_path).lower()) if len(tok) >= 2]

    def _split_parent_to_child(self, parent: CodeParentChunk) -> list[CodeChildChunk]:
        """
        内部辅助函数，负责`split parent to child` 相关处理。
        
        参数:
            self: 当前对象实例。
            parent: 输入参数，用于控制当前处理逻辑。
        
        返回:
            返回类型为 `list[CodeChildChunk]` 的处理结果。
        """
        lines = parent.content.splitlines()
        if not lines:
            return []
        step = max(self.CHILD_CHUNK_LINES - self.CHILD_CHUNK_OVERLAP, 8)
        chunks: list[CodeChildChunk] = []
        for idx, offset in enumerate(range(0, len(lines), step)):
            sub = lines[offset : offset + self.CHILD_CHUNK_LINES]
            content = "\n".join(sub).strip()
            if not content:
                continue
            start = parent.start_line + offset
            end = min(parent.end_line, start + len(sub) - 1)
            chunks.append(
                CodeChildChunk(
                    child_id=f"{parent.parent_id}#c{idx}",
                    parent_id=parent.parent_id,
                    source_path=parent.source_path,
                    language=parent.language,
                    chunk_type=parent.chunk_type,
                    symbol_name=parent.symbol_name,
                    signature=parent.signature,
                    start_line=start,
                    end_line=end,
                    content=content,
                    normalized_text=self._normalize(content),
                    normalized_path=parent.normalized_path,
                    normalized_symbol=parent.normalized_symbol,
                )
            )
            if end >= parent.end_line:
                break
        return chunks

    def _rank_map(self, docs: list[Document]) -> dict[str, int]:
        """
        内部辅助函数，负责`rank map` 相关处理。
        
        参数:
            self: 当前对象实例。
            docs: 列表参数，用于承载批量输入数据。
        
        返回:
            返回类型为 `dict[str, int]` 的处理结果。
        """
        ranks: dict[str, int] = {}
        for idx, doc in enumerate(docs, start=1):
            child_id = str(doc.metadata.get("child_id", "")).strip()
            if child_id and child_id not in ranks:
                ranks[child_id] = idx
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
        return 14.0 / (rank + 1.0)

    def _build_rg_terms(self, *, patterns: dict[str, Any], module_tokens: set[str]) -> list[str]:
        """
        构建当前步骤所需的数据结构或文本内容。
        
        参数:
            self: 当前对象实例。
        
        返回:
            返回类型为 `list[str]` 的处理结果。
        """
        terms: list[str] = []
        terms.extend(patterns.get("exact_identifiers", []))
        terms.extend(patterns.get("field_like_tokens", []))
        terms.extend(patterns.get("identifiers", [])[:24])
        terms.extend(sorted(module_tokens))
        deduped = list(dict.fromkeys([term.strip() for term in terms if len(term.strip()) >= 2]))
        return deduped[: self.runtime_config.rg_max_terms]

    def _run_rg_line_hits(self, *, terms: list[str]) -> tuple[Counter[str], dict[str, list[int]], dict[str, Any]]:
        """
        内部辅助函数，负责`run rg line hits` 相关处理。
        
        参数:
            self: 当前对象实例。
        
        返回:
            返回类型为 `tuple[Counter[str], dict[str, list[int]], dict[str, Any]]` 的处理结果。
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
            return Counter(), {}, profile
        if self._rg_executable is None:
            warning = "rg is unavailable; retrieve_code falls back to BM25/TFIDF only."
            if self.runtime_config.rg_strategy == "rg_only":
                warning = "rg is unavailable; retrieve_code rg_only strategy cannot run."
            self._warn_rg_unavailable(warning)
            profile["warning"] = warning
            return Counter(), {}, profile
        if not terms or not self.code_dirs:
            return Counter(), {}, profile

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
        for ext in sorted(self.SUPPORTED_EXTENSIONS):
            command.extend(["--glob", f"*{ext}"])
        command.extend([str(path) for path in self.code_dirs])

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
            warning = "rg timed out during code retrieval; fallback to BM25/TFIDF only."
            if self.runtime_config.rg_strategy == "rg_only":
                warning = "rg timed out during code retrieval; rg_only strategy cannot return results."
            self._logger.warning(
                "workflow.code_rg.timeout",
                timeout_ms=int(self.runtime_config.rg_timeout_ms),
                term_count=len(terms),
            )
            profile["warning"] = warning
            profile["timed_out"] = True
            return Counter(), {}, profile
        except Exception as exc:
            warning = "rg failed during code retrieval; fallback to BM25/TFIDF only."
            if self.runtime_config.rg_strategy == "rg_only":
                warning = "rg failed during code retrieval; rg_only strategy cannot return results."
            self._logger.warning(
                "workflow.code_rg.error",
                reason=str(exc),
                term_count=len(terms),
            )
            profile["warning"] = warning
            return Counter(), {}, profile

        path_hits: Counter[str] = Counter()
        line_hits_by_path: dict[str, list[int]] = defaultdict(list)
        for line in result.stdout.splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(payload.get("type", "")) != "match":
                continue
            data = payload.get("data", {})
            path_raw = str(data.get("path", {}).get("text", "")).strip()
            normalized_path = self._normalize_rg_path(path_raw)
            if not normalized_path:
                continue
            path_hits[normalized_path] += 1
            line_no = int(data.get("line_number", 0) or 0)
            if line_no <= 0:
                continue
            rows = line_hits_by_path.setdefault(normalized_path, [])
            if len(rows) >= self.runtime_config.rg_max_lines_per_path:
                continue
            rows.append(line_no)

        if result.returncode not in (0, 1):
            stderr = result.stderr.strip().splitlines()
            self._logger.warning(
                "workflow.code_rg.nonzero_exit",
                returncode=int(result.returncode),
                stderr=(stderr[0] if stderr else ""),
            )
            profile["warning"] = "rg exited with non-zero status during code retrieval; fallback results may degrade."
        profile["latency_ms"] = round((perf_counter() - started) * 1000, 3)
        profile["raw_match_count"] = sum(path_hits.values())
        return path_hits, line_hits_by_path, profile

    def _extract_patterns(self, merged_query: str) -> dict[str, Any]:
        """
        内部辅助函数，负责`extract patterns` 相关处理。
        
        参数:
            self: 当前对象实例。
            merged_query: 输入参数，用于控制当前处理逻辑。
        
        返回:
            返回类型为 `dict[str, Any]` 的处理结果。
        """
        normalized = self._normalize(merged_query)
        identifiers = [self._normalize(token) for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{1,}", merged_query)]
        identifiers = [tok for tok in identifiers if len(tok) >= 2]
        identifiers = list(dict.fromkeys(sorted(identifiers, key=len, reverse=True)))[:48]
        exact = [tok for tok in identifiers if "_" in tok or len(tok) >= 8][:24]
        field_like = [tok for tok in identifiers if tok.endswith(("_id", "_rate", "_score", "_cpa", "_cpc", "_ctr", "_cvr")) or tok in {"pctr", "pcvr", "trace_id", "request_id"}]
        is_location_query = any(
            k in normalized
            for k in ("在哪", "哪里", "定位", "定义", "实现", "哪个文件", "哪个函数", "入口函数", "where", "located", "find")
        )
        return {
            "identifiers": identifiers,
            "exact_identifiers": exact,
            "field_like_tokens": list(dict.fromkeys(field_like))[:24],
            "is_formula_query": any(k in normalized for k in ("公式", "计算", "怎么算", "formula", "calculate"))
            or any(op in merged_query for op in ("=", "*", "/", "+", "-")),
            "is_location_query": is_location_query,
        }

    def _score_lexical(self, child: CodeChildChunk, patterns: dict[str, Any]) -> tuple[float, list[str]]:
        """
        内部辅助函数，负责`score lexical` 相关处理。
        
        参数:
            self: 当前对象实例。
            child: 输入参数，用于控制当前处理逻辑。
            patterns: 列表参数，用于承载批量输入数据。
        
        返回:
            返回类型为 `tuple[float, list[str]]` 的处理结果。
        """
        score = 0.0
        matched: list[str] = []
        ids = patterns.get("identifiers", [])
        for token in ids:
            if len(token) < 2:
                continue
            cnt = child.normalized_text.count(token)
            if cnt > 0:
                matched.append(token)
                score += min(cnt, 3) * (1.8 if len(token) >= 4 else 1.0)
            if token in child.normalized_symbol:
                score += 1.4
            if token in child.normalized_path:
                score += 1.1
            if token in self._normalize(child.signature):
                score += 0.9
        return score, matched

    def _score_pattern(self, child: CodeChildChunk, patterns: dict[str, Any]) -> tuple[float, list[str]]:
        """
        内部辅助函数，负责`score pattern` 相关处理。
        
        参数:
            self: 当前对象实例。
            child: 输入参数，用于控制当前处理逻辑。
            patterns: 列表参数，用于承载批量输入数据。
        
        返回:
            返回类型为 `tuple[float, list[str]]` 的处理结果。
        """
        score = 0.0
        matched: list[str] = []
        is_location = bool(patterns.get("is_location_query"))
        for token in patterns.get("exact_identifiers", []):
            if token == child.normalized_symbol:
                score += 6.0
                matched.append(f"exact_symbol:{token}")
            elif token in child.normalized_symbol:
                score += 3.6 if is_location else 2.2
                matched.append(f"symbol_like:{token}")
            elif token in child.normalized_path:
                score += 2.8 if is_location else 1.6
                matched.append(f"path_like:{token}")
        for token in patterns.get("field_like_tokens", []):
            if token in child.normalized_text:
                score += 0.8
                matched.append(token)
        if patterns.get("is_formula_query") and "=" in child.content:
            score += 1.2
            matched.append("formula_expression")
        if is_location:
            score += 0.6
            matched.append("location_intent")
        return score, sorted(set(matched), key=len, reverse=True)

    def _build_excerpt(self, content: str, content_start_line: int, matched_terms: list[str], matched_patterns: list[str]) -> dict[str, Any]:
        """
        构建当前步骤所需的数据结构或文本内容。
        
        参数:
            self: 当前对象实例。
            content: 输入参数，用于控制当前处理逻辑。
            content_start_line: 输入参数，用于控制当前处理逻辑。
            matched_terms: 列表参数，用于承载批量输入数据。
            matched_patterns: 列表参数，用于承载批量输入数据。
        
        返回:
            返回类型为 `dict[str, Any]` 的处理结果。
        """
        lines = content.splitlines()
        if not lines:
            return {"excerpt_text": "", "excerpt_lines": [], "highlight_lines": []}
        keywords = sorted({self._normalize(k) for k in [*matched_terms, *matched_patterns] if k and len(self._normalize(k)) >= 2}, key=len, reverse=True)
        hit_indexes = [i for i, line in enumerate(lines) if any(k in self._normalize(line) for k in keywords)]
        if not hit_indexes:
            hit_indexes = [next((i for i, line in enumerate(lines) if line.strip()), 0)]
        start = max(0, hit_indexes[0] - self.EXCERPT_CONTEXT_RADIUS)
        end = min(len(lines), start + self.EXCERPT_MAX_LINES)
        hit_set = set(hit_indexes)
        excerpt_lines: list[dict[str, Any]] = []
        display: list[str] = []
        highlight_lines: list[int] = []
        for i in range(start, end):
            line_no = content_start_line + i
            raw = lines[i].rstrip()
            is_hit = i in hit_set and bool(raw.strip())
            excerpt_lines.append({"line_no": line_no, "content": raw, "is_hit": is_hit})
            display.append(f"{'>>' if is_hit else '  '} {line_no:>4}: {raw}")
            if is_hit:
                highlight_lines.append(line_no)
        if start > 0:
            display.insert(0, "...")
        if end < len(lines):
            display.append("...")
        return {"excerpt_text": "\n".join(display).strip(), "excerpt_lines": excerpt_lines, "highlight_lines": highlight_lines}

    def _extract_tokens(self, text: str) -> list[str]:
        """
        内部辅助函数，负责`extract tokens` 相关处理。
        
        参数:
            self: 当前对象实例。
            text: 待处理的文本内容。
        
        返回:
            返回类型为 `list[str]` 的处理结果。
        """
        normalized = self._normalize(text)
        tokens: list[str] = []
        tokens.extend([tok for tok in re.findall(r"[a-z_][a-z0-9_]{1,}", normalized)])
        tokens.extend([tok for tok in re.findall(r"[\u4e00-\u9fff]{2,8}", normalized)])
        return tokens

    def _infer_language(self, file_path: Path) -> str:
        """
        内部辅助函数，负责`infer language` 相关处理。
        
        参数:
            self: 当前对象实例。
            file_path: 路径参数，用于定位文件或目录。
        
        返回:
            返回类型为 `str` 的处理结果。
        """
        mapping = {
            ".py": "python", ".js": "javascript", ".jsx": "javascript", ".ts": "typescript", ".tsx": "typescript", ".go": "go", ".java": "java", ".sql": "sql", ".yaml": "yaml", ".yml": "yaml", ".json": "json", ".toml": "toml", ".ini": "ini", ".conf": "config", ".sh": "shell",
        }
        return mapping.get(file_path.suffix.lower(), "text")

    def _build_python_signature(self, node: ast.AST) -> str:
        """
        构建当前步骤所需的数据结构或文本内容。
        
        参数:
            self: 当前对象实例。
            node: 输入参数，用于控制当前处理逻辑。
        
        返回:
            返回类型为 `str` 的处理结果。
        """
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return ""
        args: list[str] = [argument.arg for argument in node.args.args]
        if node.args.vararg is not None:
            args.append(f"*{node.args.vararg.arg}")
        args.extend(argument.arg for argument in node.args.kwonlyargs)
        if node.args.kwarg is not None:
            args.append(f"**{node.args.kwarg.arg}")
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        return f"{prefix} {node.name}({', '.join(args)})"

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
        self._logger.warning("workflow.code_rg.unavailable", warning=warning)

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


def parse_code_dirs_from_env(*, project_root: Path) -> list[Path] | None:
    """
    执行`parse code dirs from env` 相关处理逻辑。
    
    返回:
        返回类型为 `list[Path] | None` 的处理结果。
    """
    raw = os.getenv("WORKFLOW_CODE_RETRIEVER_DIRS", "").strip()
    if not raw:
        default_codes_dir = (project_root / "codes").resolve()
        if default_codes_dir.exists():
            return [default_codes_dir]
        return None
    dirs: list[Path] = []
    for part in [value.strip() for value in re.split(r"[;,]", raw) if value.strip()]:
        path = Path(part)
        if not path.is_absolute():
            path = (project_root / path).resolve()
        if path.exists():
            dirs.append(path)
    return dirs or None
