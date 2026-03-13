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
from typing import Any

from langchain_community.retrievers import BM25Retriever, TFIDFRetriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from workflow.retrievers import WeightedFusionRetriever
from workflow.runtime_logging import get_file_logger
from workflow.utils import env_float, env_int


@dataclass
class WikiChunk:
    chunk_id: int
    source_path: Path
    title: str
    section: str
    chunk_type: str
    content: str
    normalized_text: str


@dataclass
class HybridScoreWeights:
    bm25: float = 0.50
    vector: float = 0.35
    lexical: float = 0.15
    source: str = "default"

    @classmethod
    def from_env(cls) -> "HybridScoreWeights":
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
    default_top_k: int = 4
    max_chunks_per_doc: int = 1
    chunk_size: int = 520
    chunk_overlap: int = 80
    excerpt_max_chars: int = 220
    candidate_multiplier: int = 6
    min_candidates: int = 12

    @classmethod
    def from_env(cls, *, default_top_k: int) -> "WikiRetrieverRuntimeConfig":
        return cls(
            default_top_k=env_int("WORKFLOW_WIKI_TOP_K", default_top_k, minimum=1),
            max_chunks_per_doc=env_int("WORKFLOW_WIKI_MAX_CHUNKS_PER_DOC", 1, minimum=1),
            chunk_size=env_int("WORKFLOW_WIKI_PARAGRAPH_MAX_CHARS", 520, minimum=120),
            chunk_overlap=env_int("WORKFLOW_WIKI_PARAGRAPH_MIN_CHARS", 80, minimum=0),
            excerpt_max_chars=env_int("WORKFLOW_WIKI_EXCERPT_MAX_CHARS", 220, minimum=60),
            candidate_multiplier=env_int("WORKFLOW_WIKI_STAGE2_MULTIPLIER", 6, minimum=1),
            min_candidates=env_int("WORKFLOW_WIKI_STAGE2_MIN_CANDIDATES", 12, minimum=1),
        )


class MarkdownWikiRetriever:
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
        self.wiki_dir = wiki_dir
        self.project_root = project_root
        self.module_doc_hints = dict(module_doc_hints or {})
        self.runtime_config = WikiRetrieverRuntimeConfig.from_env(default_top_k=default_top_k)
        self.hybrid_weights = HybridScoreWeights.from_env()
        self._logger = get_file_logger(project_root=project_root)

        self._chunks: list[WikiChunk] = []
        self._documents: list[Document] = []
        self._chunk_by_id: dict[int, WikiChunk] = {}
        self.last_search_profile: dict[str, Any] = {}

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

    def search(
        self,
        *,
        user_query: str,
        retrieval_queries: list[str] | None = None,
        module_name: str | None = None,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        if not user_query.strip() or not self._documents or self._bm25 is None or self._tfidf is None or self._ensemble is None:
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

        self._bm25.k = candidate_k
        self._tfidf.k = candidate_k

        bm25_docs = self._bm25.invoke(merged_query)
        tfidf_docs = self._tfidf.invoke(merged_query)
        ensemble_docs = self._ensemble.invoke(merged_query)

        bm25_ranks = self._rank_map(bm25_docs)
        tfidf_ranks = self._rank_map(tfidf_docs)
        ensemble_ranks = self._rank_map(ensemble_docs)

        query_terms = self._extract_terms(expanded_queries)

        candidate_ids: list[int] = []
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

            score = (
                bm25_score * self.hybrid_weights.bm25
                + tfidf_score * self.hybrid_weights.vector
                + lexical_score * self.hybrid_weights.lexical
                + ensemble_score * 0.25
                + module_boost
            )
            scored.append(
                {
                    "chunk": chunk,
                    "score": score,
                    "bm25_score": bm25_score,
                    "vector_score": tfidf_score,
                    "lexical_score": lexical_score,
                    "ensemble_score": ensemble_score,
                    "module_boost": module_boost,
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
        }
        return hits

    def get_chunk_snapshot(self) -> list[dict[str, Any]]:
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
        ranks: dict[int, int] = {}
        for idx, doc in enumerate(docs, start=1):
            chunk_id = int(doc.metadata.get("chunk_id", -1))
            if chunk_id <= 0:
                continue
            ranks.setdefault(chunk_id, idx)
        return ranks

    def _rank_score(self, rank: int | None) -> float:
        if rank is None:
            return 0.0
        # Keep score range compatible with existing grade thresholds in node layer.
        return 12.0 / (rank + 1.0)

    def _extract_terms(self, query_phrases: list[str]) -> list[str]:
        merged = " ".join(query_phrases).lower()
        terms: list[str] = []
        terms.extend([token for token in re.findall(r"[a-z][a-z0-9_+-]{1,}", merged) if token not in self.STOP_WORDS])
        terms.extend([token for token in re.findall(r"[\u4e00-\u9fff]{2,10}", merged) if token not in self.STOP_WORDS])
        return list(dict.fromkeys(terms))[:48]

    def _lexical_match_score(self, text: str, terms: list[str]) -> float:
        if not terms:
            return 0.0
        hits = sum(1 for term in terms if term in text)
        if hits <= 0:
            return 0.0
        coverage = hits / max(len(terms), 1)
        return min(2.0, coverage * 2.0)

    def _module_prior_boost(self, *, chunk: WikiChunk, module_name: str) -> float:
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
        merged = re.sub(r"\s+", " ", text.strip())
        if len(merged) <= self.runtime_config.excerpt_max_chars:
            return merged
        return merged[: self.runtime_config.excerpt_max_chars] + "..."

    def _extract_title(self, text: str, file_path: Path) -> str:
        match = re.search(r"(?m)^#\s+(.+)$", text)
        if match:
            return match.group(1).strip()
        return file_path.stem

    def _extract_section(self, text: str, *, fallback: str) -> str:
        match = re.search(r"(?m)^#{2,4}\s+(.+)$", text)
        if match:
            return match.group(1).strip()
        first_line = text.splitlines()[0].strip() if text.splitlines() else ""
        if first_line.startswith("#"):
            return first_line.lstrip("#").strip() or fallback
        return fallback

    def _infer_chunk_type(self, text: str) -> str:
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
        try:
            return path.relative_to(self.project_root).as_posix()
        except ValueError:
            return path.as_posix()

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.lower()).strip()
