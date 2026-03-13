# -*- coding: utf-8 -*-
"""
该模块实现工作流节点`retrieve_wiki` 的处理逻辑，负责读取状态并输出增量结果。
"""
from __future__ import annotations

"""Wiki retriever backed by LangChain retriever wrappers.

Compared with the previous hand-written retriever, this implementation uses:
- `RecursiveCharacterTextSplitter` for chunking
- `BM25Retriever` for sparse retrieval
- `TFIDFRetriever` for lightweight vector-like retrieval
- project-local weighted fusion retriever for rank fusion
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

from langchain_community.retrievers import BM25Retriever, TFIDFRetriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from workflow.retrievers import WeightedFusionRetriever
from workflow.runtime_logging import get_file_logger
from workflow.utils import env_bool, env_float, env_int


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
    """
    bm25: float = 0.50
    vector: float = 0.35
    lexical: float = 0.15
    source: str = "default"

    @classmethod
    def from_env(cls) -> "HybridScoreWeights":
        """
        执行`from env` 相关处理逻辑。
        
        参数:
            cls: 当前类对象。
        
        返回:
            返回类型为 `'HybridScoreWeights'` 的处理结果。
        """
        weights = cls()
        path = os.getenv("WORKFLOW_WIKI_HYBRID_WEIGHTS_PATH", "").strip()
        if path:
            file_path = Path(path)
            if file_path.exists():
                try:
                    payload = json.loads(file_path.read_text(encoding="utf-8"))
                    weights.bm25 = float(payload.get("bm25", weights.bm25))
                    weights.vector = float(payload.get("vector", weights.vector))
                    weights.lexical = float(payload.get("lexical", weights.lexical))
                    weights.source = f"file:{file_path.as_posix()}"
                except Exception:
                    pass

        weights.bm25 = env_float("WORKFLOW_WIKI_WEIGHT_BM25", weights.bm25)
        weights.vector = env_float("WORKFLOW_WIKI_WEIGHT_VECTOR", weights.vector)
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
        vector = max(self.vector, 0.0)
        lexical = max(self.lexical, 0.0)
        total = bm25 + vector + lexical
        if total <= 0:
            return HybridScoreWeights(source="fallback")
        return HybridScoreWeights(
            bm25=bm25 / total,
            vector=vector / total,
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

    @classmethod
    def from_env(cls, *, default_top_k: int) -> "WikiRetrieverRuntimeConfig":
        """
        执行`from env` 相关处理逻辑。
        
        参数:
            cls: 当前类对象。
        
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
        )


class MarkdownWikiRetriever:
    """
    定义`MarkdownWikiRetriever`，用于封装相关数据结构与处理行为。
    """
    MODULE_DOC_PRIOR_BOOST: float = 0.08

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
    ) -> None:
        """
        内部辅助函数，负责` init  ` 相关处理。
        
        参数:
            self: 当前对象实例。
        
        返回:
            无返回值。
        """
        self.wiki_dir = wiki_dir
        self.project_root = project_root
        self.module_doc_hints = dict(module_doc_hints or {})
        self.runtime_config = WikiRetrieverRuntimeConfig.from_env(default_top_k=default_top_k)
        self.hybrid_weights = HybridScoreWeights.from_env()
        self._logger = get_file_logger(project_root=project_root)

        self._chunks: list[WikiChunk] = []
        self._documents: list[Document] = []
        self._chunk_by_id: dict[int, WikiChunk] = {}
        self._chunk_ids_by_path: defaultdict[str, list[int]] = defaultdict(list)
        self.last_search_profile: dict[str, Any] = {}
        self._rg_executable = shutil.which("rg") if self.runtime_config.rg_strategy != "no_rg" else None
        self._rg_unavailable_warned = False

        self._bm25: BM25Retriever | None = None
        self._tfidf: TFIDFRetriever | None = None
        self._ensemble: WeightedFusionRetriever | None = None

        started = perf_counter()
        stats = self._build_index()
        self._logger.info(
            "workflow.wiki_index.built",
            wiki_dir=self._to_relative_path(self.wiki_dir),
            file_count=stats.get("file_count", 0),
            chunk_count=stats.get("chunk_count", 0),
            default_top_k=self.runtime_config.default_top_k,
            max_chunks_per_doc=self.runtime_config.max_chunks_per_doc,
            hybrid_weights={
                "bm25": round(self.hybrid_weights.bm25, 4),
                "vector": round(self.hybrid_weights.vector, 4),
                "lexical": round(self.hybrid_weights.lexical, 4),
                "source": self.hybrid_weights.source,
            },
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
            or (use_semantic and (self._bm25 is None or self._tfidf is None or self._ensemble is None))
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

        bm25_docs: list[Document] = []
        tfidf_docs: list[Document] = []
        ensemble_docs: list[Document] = []
        if use_semantic:
            assert self._bm25 is not None and self._tfidf is not None and self._ensemble is not None
            self._bm25.k = candidate_k
            self._tfidf.k = candidate_k

            bm25_docs = self._bm25.invoke(merged_query)
            tfidf_docs = self._tfidf.invoke(merged_query)
            ensemble_docs = self._ensemble.invoke(merged_query)

        bm25_ranks = self._rank_map(bm25_docs)
        tfidf_ranks = self._rank_map(tfidf_docs)
        ensemble_ranks = self._rank_map(ensemble_docs)

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
            for docs in (ensemble_docs, bm25_docs, tfidf_docs):
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
            tfidf_rank = tfidf_ranks.get(chunk_id)
            ensemble_rank = ensemble_ranks.get(chunk_id)

            bm25_score = self._rank_score(bm25_rank)
            tfidf_score = self._rank_score(tfidf_rank)
            lexical_score = self._lexical_match_score(chunk.normalized_text, query_terms)
            ensemble_score = self._rank_score(ensemble_rank)
            module_boost = self._module_prior_boost(chunk=chunk, module_name=module_name or "")
            rg_boost = rg_boost_by_chunk.get(chunk_id, 0.0)
            path_key = self._to_relative_path(chunk.source_path)

            score = 0.0
            if use_semantic:
                score = (
                    bm25_score * self.hybrid_weights.bm25
                    + tfidf_score * self.hybrid_weights.vector
                    + lexical_score * self.hybrid_weights.lexical
                    + ensemble_score * 0.25
                    + module_boost
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
                    "vector_score": tfidf_score,
                    "lexical_score": lexical_score,
                    "ensemble_score": ensemble_score,
                    "module_boost": module_boost,
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
                        "vector": round(float(item["vector_score"]), 4),
                        "lexical": round(float(item["lexical_score"]), 4),
                        "ensemble": round(float(item["ensemble_score"]), 4),
                        "module_boost": round(float(item["module_boost"]), 4),
                        "rg_boost": round(float(item["rg_boost"]), 4),
                        "rg_path_hits": int(item["rg_path_hits"]),
                        "rg_strategy": str(item["rg_strategy"]),
                        "weights": {
                            "bm25": round(self.hybrid_weights.bm25, 4),
                            "vector": round(self.hybrid_weights.vector, 4),
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
        return hits

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

        # Keep high candidate size internally; query-time k is adjusted in `search`.
        self._bm25 = BM25Retriever.from_documents(self._documents)
        self._tfidf = TFIDFRetriever.from_documents(self._documents)
        self._ensemble = WeightedFusionRetriever(
            retrievers=[self._bm25, self._tfidf],
            weights=[self.hybrid_weights.bm25, self.hybrid_weights.vector],
            id_key="chunk_id",
        )
        return stats

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
        
        参数:
            self: 当前对象实例。
        
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
        hit_count = sum(1 for hint in hints if hint.lower() in path_key or hint.lower() in chunk.normalized_text)
        if hit_count <= 0:
            return 0.0
        return min(self.MODULE_DOC_PRIOR_BOOST * hit_count, 0.3)

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
