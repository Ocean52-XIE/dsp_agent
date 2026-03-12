from __future__ import annotations

"""本地代码检索器（工程化收口版）。

实现目标：
1. 保持“可直接落地”：不依赖外部向量库；
2. 保持“可扩展”：参数可配置、结构可观测；
3. 保持“可解释”：返回符号、路径、行号、高亮片段。
"""

from collections import defaultdict
from dataclasses import dataclass
from math import log, sqrt
from pathlib import Path
from time import perf_counter
import ast
import hashlib
import os
import re
from typing import Any, Iterator
from workflow.runtime_logging import get_file_logger


@dataclass
class CodeParentChunk:
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
    """retrieve_code 运行参数。

    这些参数用于线上灰度调优，避免每次改权重都要改代码。
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

    @classmethod
    def from_env(cls) -> "CodeRetrieverRuntimeConfig":
        def env_int(name: str, default: int, minimum: int) -> int:
            raw = os.getenv(name, "").strip()
            if not raw:
                return default
            try:
                return max(int(raw), minimum)
            except ValueError:
                return default

        def env_float(name: str, default: float, minimum: float) -> float:
            raw = os.getenv(name, "").strip()
            if not raw:
                return default
            try:
                return max(float(raw), minimum)
            except ValueError:
                return default

        return cls(
            default_top_k=env_int("WORKFLOW_CODE_RETRIEVER_TOP_K", 4, 1),
            max_child_candidates=env_int("WORKFLOW_CODE_RETRIEVER_MAX_CHILD_CANDIDATES", 64, 8),
            max_results_per_path=env_int("WORKFLOW_CODE_RETRIEVER_MAX_PER_PATH", 2, 1),
            semantic_weight=env_float("WORKFLOW_CODE_RETRIEVER_SEMANTIC_WEIGHT", 4.0, 0.0),
            pattern_weight=env_float("WORKFLOW_CODE_RETRIEVER_PATTERN_WEIGHT", 1.6, 0.0),
            parent_best_pattern_weight=env_float("WORKFLOW_CODE_RETRIEVER_PARENT_BEST_PATTERN_WEIGHT", 0.9, 0.0),
            parent_avg_pattern_weight=env_float("WORKFLOW_CODE_RETRIEVER_PARENT_AVG_PATTERN_WEIGHT", 0.35, 0.0),
            min_final_score=env_float("WORKFLOW_CODE_RETRIEVER_MIN_FINAL_SCORE", 0.25, 0.0),
            grade_high_top1_threshold=env_float("WORKFLOW_CODE_RETRIEVER_GRADE_HIGH_TOP1_THRESHOLD", 8.0, 0.0),
            grade_medium_top1_threshold=env_float("WORKFLOW_CODE_RETRIEVER_GRADE_MEDIUM_TOP1_THRESHOLD", 4.0, 0.0),
        )


class LocalCodeRetriever:
    SUPPORTED_EXTENSIONS = {
        ".py",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".go",
        ".java",
        ".sql",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".conf",
        ".sh",
    }
    IGNORED_DIR_NAMES = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", "node_modules", ".venv", "venv", "dist", "build", "target"}
    MAX_FILE_SIZE_BYTES = 256 * 1024
    CHILD_CHUNK_LINES = 36
    CHILD_CHUNK_OVERLAP = 8
    EXCERPT_MAX_LINES = 18
    EXCERPT_CONTEXT_RADIUS = 2
    STOP_WORDS = {"the", "is", "are", "and", "or", "for", "with", "from", "this", "that", "what", "when", "where", "which", "how", "to", "in", "on", "什么", "怎么", "如何", "这个", "那个", "问题", "相关", "模块", "代码", "实现", "逻辑"}

    def __init__(
        self,
        *,
        project_root: Path,
        code_dirs: list[Path] | None = None,
        default_top_k: int = 4,
        runtime_config: CodeRetrieverRuntimeConfig | None = None,
    ) -> None:
        self.project_root = project_root
        self._logger = get_file_logger(project_root=project_root)
        self.runtime_config = runtime_config or CodeRetrieverRuntimeConfig.from_env()
        if default_top_k > 0:
            self.runtime_config.default_top_k = default_top_k
        self.default_top_k = self.runtime_config.default_top_k
        # 目录收口策略：
        # 1. 若外部显式传入 code_dirs（或由环境变量解析得到），按传入目录索引；
        # 2. 若未传入，则默认只索引仓库根目录下的 `codes/`；
        # 3. 不再默认回退到整个 project_root，避免把工具脚本/评测工程文件误纳入检索语料。
        if code_dirs is None:
            default_codes_dir = project_root / "codes"
            fallback_dirs = [default_codes_dir] if default_codes_dir.exists() else []
            self.code_dirs = fallback_dirs
        else:
            self.code_dirs = [d for d in code_dirs if d.exists()]

        self._parent_chunks: dict[str, CodeParentChunk] = {}
        self._child_chunks: list[CodeChildChunk] = []
        self._child_vectors: dict[str, dict[str, float]] = {}
        self._child_vector_norms: dict[str, float] = {}
        self._idf: dict[str, float] = {}
        # 符号级索引：symbol token -> parent_id 集合。
        # 目标是增强“函数/类/字段名定位”类查询。
        self._symbol_index: dict[str, set[str]] = defaultdict(set)
        # 路径 token 索引：path token -> parent_id 集合。
        # 目标是增强“模块/目录定位”类查询。
        self._path_token_index: dict[str, set[str]] = defaultdict(set)
        self.last_search_profile: dict[str, Any] = {}
        self._index_read_error_count = 0
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
        if not self._child_chunks:
            self._logger.warning(
                "workflow.code_index.empty",
                code_dirs=[str(path) for path in self.code_dirs],
            )

    def search(self, *, user_query: str, retrieval_queries: list[str] | None = None, module_name: str | None = None, top_k: int | None = None) -> list[dict[str, Any]]:
        """检索代码证据。"""
        started = perf_counter()
        query = user_query.strip()
        if not query or not self._child_chunks:
            self.last_search_profile = {"latency_ms": round((perf_counter() - started) * 1000, 3)}
            return []

        query_phrases = [query, *(retrieval_queries or [])]
        query_terms = self._extract_query_terms(query_phrases, module_name=module_name or "")
        query_patterns = self._extract_query_patterns(query_phrases, module_name=module_name or "")
        module_tokens = self._build_module_tokens(module_name or "")
        prior_parent_ids = self._lookup_prior_parent_ids(
            # P1：优先使用“先验标识符”集合（函数定位问题下会更偏向精确符号），
            # 避免把通用 token（如 module/recall/rate）过度注入先验候选。
            identifiers=query_patterns.get("prior_identifiers", query_patterns.get("identifiers", [])),
            module_tokens=module_tokens,
        )
        query_vector = self._build_query_tfidf_vector(query_terms)
        child_scored = self._score_child_chunks(
            query_phrases=query_phrases,
            query_terms=query_terms,
            query_patterns=query_patterns,
            query_vector=query_vector,
            module_name=(module_name or "").lower().strip(),
            module_tokens=module_tokens,
            prior_parent_ids=prior_parent_ids,
        )
        parent_scored = self._aggregate_to_parent(
            child_scored=child_scored[: self.runtime_config.max_child_candidates],
            query_phrases=query_phrases,
            query_patterns=query_patterns,
            module_name=(module_name or "").lower().strip(),
            module_tokens=module_tokens,
            prior_parent_ids=prior_parent_ids,
        )
        selected = self._select_diverse_parent_items(parent_items=parent_scored, top_n=(top_k or self.runtime_config.default_top_k), max_per_path=self.runtime_config.max_results_per_path)

        hits: list[dict[str, Any]] = []
        for rank, item in enumerate(selected, start=1):
            parent = item["parent"]
            best_child = item["best_child"]
            excerpt = self._build_line_level_excerpt(content=best_child.content, content_start_line=best_child.start_line, matched_terms=item["matched_terms"], matched_patterns=item["matched_patterns"])
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
                    "retrieval_debug": {"matched_terms": item["matched_terms"][:8], "matched_patterns": item["matched_patterns"][:8], "parent_hit_count": item["hit_count"], "pattern_score": round(float(item["best_pattern_score"]), 4)},
                }
            )

        self.last_search_profile = {
            "latency_ms": round((perf_counter() - started) * 1000, 3),
            "child_candidates": len(child_scored),
            "parent_candidates": len(parent_scored),
            "selected_count": len(hits),
            "prior_parent_candidates": len(prior_parent_ids),
        }
        return hits

    def get_index_snapshot(self) -> list[dict[str, Any]]:
        return [{"path": self._to_relative_path(c.source_path), "language": c.language, "chunk_type": c.chunk_type, "symbol_name": c.symbol_name, "signature": c.signature, "start_line": c.start_line, "end_line": c.end_line, "content_length": len(c.content)} for c in sorted(self._parent_chunks.values(), key=lambda x: (self._to_relative_path(x.source_path), x.start_line))]

    def _build_index(self) -> dict[str, int]:
        stats = {
            "scanned_file_count": 0,
            "indexed_file_count": 0,
            "parent_chunk_count": 0,
            "child_chunk_count": 0,
        }
        for file_path in self._iter_code_files():
            stats["scanned_file_count"] += 1
            parents = self._parse_file_to_parent_chunks(file_path)
            if not parents:
                continue
            stats["indexed_file_count"] += 1
            for parent in parents:
                self._parent_chunks[parent.parent_id] = parent
                self._register_parent_to_indexes(parent)
                self._child_chunks.extend(self._split_parent_to_child_chunks(parent))
        self._build_tfidf_index()
        stats["parent_chunk_count"] = len(self._parent_chunks)
        stats["child_chunk_count"] = len(self._child_chunks)
        return stats

    def _iter_code_files(self) -> Iterator[Path]:
        for directory in self.code_dirs:
            for file_path in directory.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS and not any(part in self.IGNORED_DIR_NAMES for part in file_path.parts):
                    try:
                        if file_path.stat().st_size <= self.MAX_FILE_SIZE_BYTES:
                            yield file_path
                    except OSError:
                        continue

    def _parse_file_to_parent_chunks(self, file_path: Path) -> list[CodeParentChunk]:
        try:
            raw = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                raw = file_path.read_text(encoding="utf-8", errors="ignore")
            except OSError as exc:
                self._index_read_error_count += 1
                self._logger.warning(
                    "workflow.code_index.read_failed",
                    path=self._to_relative_path(file_path),
                    error_type=type(exc).__name__,
                )
                return []
        except OSError as exc:
            self._index_read_error_count += 1
            self._logger.warning(
                "workflow.code_index.read_failed",
                path=self._to_relative_path(file_path),
                error_type=type(exc).__name__,
            )
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
                    parents.append(self._build_parent_chunk(file_path=file_path, language=language, chunk_type=kind, symbol_name=symbol, signature=signature, start_line=start, end_line=end, content=content))
                if parents:
                    parents.append(self._build_parent_chunk(file_path=file_path, language=language, chunk_type="file", symbol_name=file_path.stem, signature=file_path.name, start_line=1, end_line=max(len(lines), 1), content=raw.strip()))
                    return sorted(parents, key=lambda x: (x.start_line, x.end_line))
        # 非 python 或 python 解析失败，走文件级 fallback
        return [self._build_parent_chunk(file_path=file_path, language=language, chunk_type="file", symbol_name=file_path.stem, signature=file_path.name, start_line=1, end_line=max(len(lines), 1), content=raw.strip())]

    def _build_parent_chunk(self, *, file_path: Path, language: str, chunk_type: str, symbol_name: str, signature: str, start_line: int, end_line: int, content: str) -> CodeParentChunk:
        key = f"{file_path.as_posix()}::{chunk_type}::{symbol_name}::{start_line}:{end_line}"
        return CodeParentChunk(parent_id=hashlib.md5(key.encode("utf-8")).hexdigest(), source_path=file_path, language=language, chunk_type=chunk_type, symbol_name=symbol_name, signature=signature, start_line=start_line, end_line=end_line, content=content, normalized_text=self._normalize(content), normalized_path=self._normalize(self._to_relative_path(file_path)), normalized_symbol=self._normalize(symbol_name))

    def _register_parent_to_indexes(self, parent: CodeParentChunk) -> None:
        """把 parent 注册到符号索引与路径索引。"""
        symbol_tokens = self._extract_tokens(parent.normalized_symbol)
        signature_tokens = self._extract_tokens(parent.signature)
        for token in [*symbol_tokens, *signature_tokens]:
            if len(token) < 2:
                continue
            self._symbol_index[token].add(parent.parent_id)

        path_tokens = self._extract_path_tokens(parent.source_path)
        for token in path_tokens:
            if len(token) < 2:
                continue
            self._path_token_index[token].add(parent.parent_id)

    def _extract_path_tokens(self, source_path: Path) -> list[str]:
        """从相对路径中提取可检索 token。"""
        relative_path = self._to_relative_path(source_path)
        lowered = relative_path.lower()
        parts = re.split(r"[\\/._\-]+", lowered)
        return [token for token in parts if len(token) >= 2]

    def _build_module_tokens(self, module_name: str) -> set[str]:
        """把模块名拆成路径先验 token。"""
        normalized = self._normalize(module_name)
        tokens = re.split(r"[._\-/\s]+", normalized)
        return {token for token in tokens if len(token) >= 2}

    def _lookup_prior_parent_ids(self, *, identifiers: list[str], module_tokens: set[str]) -> set[str]:
        """根据 symbol/path 索引查找先验 parent 集合。"""
        parent_ids: set[str] = set()
        for token in identifiers:
            normalized = self._normalize(token)
            if len(normalized) < 2:
                continue
            parent_ids.update(self._symbol_index.get(normalized, set()))
        for token in module_tokens:
            parent_ids.update(self._path_token_index.get(token, set()))
        return parent_ids

    def _split_parent_to_child_chunks(self, parent: CodeParentChunk) -> list[CodeChildChunk]:
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
            chunks.append(CodeChildChunk(child_id=f"{parent.parent_id}#c{idx}", parent_id=parent.parent_id, source_path=parent.source_path, language=parent.language, chunk_type=parent.chunk_type, symbol_name=parent.symbol_name, signature=parent.signature, start_line=start, end_line=end, content=content, normalized_text=self._normalize(content), normalized_path=parent.normalized_path, normalized_symbol=parent.normalized_symbol))
            if end >= parent.end_line:
                break
        return chunks

    def _build_tfidf_index(self) -> None:
        token_freqs: dict[str, dict[str, int]] = {}
        df: dict[str, int] = defaultdict(int)
        for child in self._child_chunks:
            tf = self._count_tokens(child.normalized_text)
            token_freqs[child.child_id] = tf
            for token in tf:
                df[token] += 1
        total_docs = max(len(self._child_chunks), 1)
        self._idf = {token: log((total_docs + 1) / (freq + 1)) + 1.0 for token, freq in df.items()}
        for child in self._child_chunks:
            vec = self._build_tfidf_vector(token_freqs.get(child.child_id, {}))
            self._child_vectors[child.child_id] = vec
            self._child_vector_norms[child.child_id] = self._vector_norm(vec)

    def _score_child_chunks(
        self,
        *,
        query_phrases: list[str],
        query_terms: list[str],
        query_patterns: dict[str, Any],
        query_vector: dict[str, float],
        module_name: str,
        module_tokens: set[str],
        prior_parent_ids: set[str],
    ) -> list[dict[str, Any]]:
        scored: list[dict[str, Any]] = []
        for child in self._child_chunks:
            lexical, matched_terms = self._score_child_lexical(
                child=child,
                query_phrases=query_phrases,
                query_terms=query_terms,
                module_name=module_name,
                module_tokens=module_tokens,
            )
            # 命中“符号索引/路径索引”先验的 child，给予额外加分。
            if child.parent_id in prior_parent_ids:
                lexical += 1.6
            semantic = self._cosine_similarity(query_vector, self._child_vectors.get(child.child_id, {}), self._child_vector_norms.get(child.child_id, 0.0))
            pattern, matched_patterns = self._score_child_pattern(child=child, query_patterns=query_patterns)
            final = lexical + semantic * self.runtime_config.semantic_weight + pattern * self.runtime_config.pattern_weight
            if final < self.runtime_config.min_final_score:
                continue
            scored.append({"child": child, "final_score": final, "lexical_score": lexical, "semantic_score": semantic, "pattern_score": pattern, "matched_terms": matched_terms, "matched_patterns": matched_patterns})
        return sorted(scored, key=lambda x: (x["final_score"], x["pattern_score"], x["lexical_score"]), reverse=True)

    def _score_child_lexical(
        self,
        *,
        child: CodeChildChunk,
        query_phrases: list[str],
        query_terms: list[str],
        module_name: str,
        module_tokens: set[str],
    ) -> tuple[float, list[str]]:
        score = 0.0
        matched: list[str] = []
        signature = self._normalize(child.signature)
        for term in query_terms:
            if len(term) < 2:
                continue
            cnt = child.normalized_text.count(term)
            if cnt <= 0:
                continue
            matched.append(term)
            score += min(cnt, 3) * (1.8 if len(term) >= 4 else 1.0)
            if term in child.normalized_symbol:
                score += 1.4
            if term in child.normalized_path:
                score += 1.1
            if term in signature:
                score += 0.9
        for phrase in query_phrases:
            p = self._normalize(phrase)
            if len(p) >= 4 and p in child.normalized_text:
                score += 2.2
        if module_name and module_name in child.normalized_path:
            score += 1.0
        # 路径先验：模块 token 命中路径时，给稳定加分。
        if module_tokens and any(token in child.normalized_path for token in module_tokens):
            score += 0.9
        return score, matched

    def _score_child_pattern(self, *, child: CodeChildChunk, query_patterns: dict[str, Any]) -> tuple[float, list[str]]:
        score = 0.0
        matched: list[str] = []
        is_location_query = bool(query_patterns.get("is_location_query"))
        # P1：函数定位类 query 的“符号精确命中”奖励要显著高于语义匹配。
        # 说明：
        # - exact_identifier 来自 query 中显式函数/类标识符；
        # - 对 child.normalized_symbol 的全等命中给最高加分；
        # - 该分值会再经过 pattern_weight 进入总分，因此这里设置为明显高于通用 token 匹配。
        for token in query_patterns.get("exact_identifiers", []):
            if len(token) < 2:
                continue
            if token == child.normalized_symbol:
                score += 6.0
                matched.append(f"exact_symbol:{token}")
                continue
            if token in child.normalized_symbol:
                score += 3.6 if is_location_query else 2.2
                matched.append(f"symbol_like:{token}")
                continue
            if token in child.normalized_path:
                score += 2.8 if is_location_query else 1.6
                matched.append(f"path_like:{token}")
                continue

        for token in query_patterns.get("identifiers", []):
            if len(token) < 2:
                continue
            if token in child.normalized_symbol:
                score += 2.8 if not is_location_query else 2.2
                matched.append(token)
            elif token in child.normalized_path:
                score += 1.8 if not is_location_query else 1.5
                matched.append(token)
            elif token in child.normalized_text:
                score += 1.0
                matched.append(token)
        for token in query_patterns.get("field_like_tokens", []):
            if len(token) >= 2 and token in child.normalized_text:
                score += 0.8
                matched.append(token)
        if query_patterns.get("is_formula_query") and "=" in child.content:
            score += 1.2
            matched.append("formula_expression")
        if is_location_query:
            score += 0.6
            matched.append("location_intent")
        return score, sorted(set(matched), key=len, reverse=True)

    def _aggregate_to_parent(
        self,
        *,
        child_scored: list[dict[str, Any]],
        query_phrases: list[str],
        query_patterns: dict[str, Any],
        module_name: str,
        module_tokens: set[str],
        prior_parent_ids: set[str],
    ) -> list[dict[str, Any]]:
        grouped: dict[str, dict[str, Any]] = {}
        for item in child_scored:
            child = item["child"]
            parent = self._parent_chunks.get(child.parent_id)
            if parent is None:
                continue
            b = grouped.setdefault(parent.parent_id, {"parent": parent, "best_child": child, "best_child_score": item["final_score"], "best_pattern_score": item["pattern_score"], "scores": [], "pattern_scores": [], "matched_terms": set(), "matched_patterns": set(), "hit_count": 0})
            b["scores"].append(item["final_score"])
            b["pattern_scores"].append(item["pattern_score"])
            b["matched_terms"].update(item["matched_terms"])
            b["matched_patterns"].update(item["matched_patterns"])
            b["hit_count"] += 1
            if item["final_score"] > b["best_child_score"]:
                b["best_child_score"] = item["final_score"]
                b["best_child"] = child
            if item["pattern_score"] > b["best_pattern_score"]:
                b["best_pattern_score"] = item["pattern_score"]
        items: list[dict[str, Any]] = []
        for b in grouped.values():
            top = sorted(b["scores"], reverse=True)[:2]
            top_pattern = sorted(b["pattern_scores"], reverse=True)[:2]
            final = b["best_child_score"] + (sum(top) / max(len(top), 1)) * 0.25
            final += b["best_pattern_score"] * self.runtime_config.parent_best_pattern_weight
            final += (sum(top_pattern) / max(len(top_pattern), 1)) * self.runtime_config.parent_avg_pattern_weight
            parent = b["parent"]
            # Parent 级别路径先验：目录命中模块 token 时进一步稳住排序。
            if module_tokens and any(token in parent.normalized_path for token in module_tokens):
                final += 0.8
            # Parent 级别符号索引先验：命中先验候选时再加轻量分。
            if parent.parent_id in prior_parent_ids:
                final += 0.6
            # P1：若 parent 符号名与查询中的精确标识符全等，给予额外 parent 级奖励，
            # 进一步稳定“函数定位”题型的 Top1 命中率。
            exact_identifiers = query_patterns.get("exact_identifiers", [])
            if parent.normalized_symbol and parent.normalized_symbol in exact_identifiers:
                final += 1.8
            items.append({"parent": b["parent"], "best_child": b["best_child"], "best_child_score": b["best_child_score"], "best_pattern_score": b["best_pattern_score"], "final_score": final, "matched_terms": sorted(b["matched_terms"], key=len, reverse=True), "matched_patterns": sorted(b["matched_patterns"], key=len, reverse=True), "hit_count": b["hit_count"]})
        return sorted(items, key=lambda x: (x["final_score"], x["best_pattern_score"], x["best_child_score"], x["hit_count"]), reverse=True)

    def _select_diverse_parent_items(self, *, parent_items: list[dict[str, Any]], top_n: int, max_per_path: int) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        counts: dict[str, int] = defaultdict(int)
        backup: list[dict[str, Any]] = []
        for item in parent_items:
            path = self._to_relative_path(item["parent"].source_path)
            if counts[path] >= max_per_path:
                backup.append(item)
                continue
            counts[path] += 1
            selected.append(item)
            if len(selected) >= top_n:
                return selected
        for item in backup:
            selected.append(item)
            if len(selected) >= top_n:
                break
        return selected

    def _extract_query_terms(self, query_phrases: list[str], *, module_name: str) -> list[str]:
        merged = " ".join(query_phrases).strip().lower()
        if module_name:
            merged = f"{merged} {module_name}"
        freq = self._count_tokens(merged)
        return sorted(freq.keys(), key=lambda x: (len(x), freq[x]), reverse=True)[:64]

    def _extract_query_patterns(self, query_phrases: list[str], *, module_name: str) -> dict[str, Any]:
        merged = " ".join(query_phrases).strip()
        if module_name:
            merged = f"{merged} {module_name}"
        normalized = self._normalize(merged)
        identifiers = [self._normalize(t) for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]{1,}", merged)]
        identifiers = [t for t in identifiers if len(t) >= 2]
        identifiers = list(dict.fromkeys(sorted(identifiers, key=len, reverse=True)))[:48]
        field_like = [
            t
            for t in identifiers
            if t.endswith(("_id", "_rate", "_score", "_cpa", "_cpc", "_ctr", "_cvr"))
            or t in {"pctr", "pcvr", "trace_id", "request_id"}
        ]
        is_location_query = any(
            k in normalized
            for k in ("在哪", "哪里", "定位", "定义", "实现", "哪个文件", "哪个函数", "入口函数", "where", "located", "find")
        )
        # 精确标识符候选：
        # 仅保留下划线风格或长度较长 token，尽量贴近真实函数/类名。
        exact_identifiers = [
            token
            for token in identifiers
            if "_" in token or len(token) >= 8
        ][:24]
        # 函数定位问法下，先验集合优先采用 exact_identifiers，降低噪声 token 干扰；
        # 非定位问法保持原有广召回能力。
        prior_identifiers = exact_identifiers if (is_location_query and exact_identifiers) else identifiers
        return {
            "identifiers": identifiers,
            "exact_identifiers": exact_identifiers,
            "prior_identifiers": prior_identifiers[:32],
            "field_like_tokens": list(dict.fromkeys(field_like))[:24],
            "is_formula_query": any(k in normalized for k in ("公式", "计算", "怎么算", "formula", "calculate", "target_cpa", "ocpc", "pcvr", "pctr"))
            or any(op in merged for op in ("=", "*", "/", "+", "-")),
            "is_location_query": is_location_query,
        }

    def _build_query_tfidf_vector(self, query_terms: list[str]) -> dict[str, float]:
        freq: dict[str, int] = defaultdict(int)
        for token in query_terms:
            freq[token] += 1
        return self._build_tfidf_vector(freq)

    def _build_tfidf_vector(self, token_freq: dict[str, int]) -> dict[str, float]:
        total = sum(token_freq.values())
        if total <= 0:
            return {}
        return {token: (freq / total) * self._idf.get(token, 1.0) for token, freq in token_freq.items() if freq > 0}

    def _cosine_similarity(self, query_vector: dict[str, float], doc_vector: dict[str, float], doc_norm: float) -> float:
        if not query_vector or not doc_vector or doc_norm <= 0:
            return 0.0
        q_norm = self._vector_norm(query_vector)
        if q_norm <= 0:
            return 0.0
        dot = sum(qv * doc_vector.get(token, 0.0) for token, qv in query_vector.items())
        return dot / (q_norm * doc_norm) if dot > 0 else 0.0

    def _vector_norm(self, vector: dict[str, float]) -> float:
        return sqrt(sum(v * v for v in vector.values())) if vector else 0.0

    def _count_tokens(self, text: str) -> dict[str, int]:
        freq: dict[str, int] = defaultdict(int)
        for token in self._extract_tokens(text):
            freq[token] += 1
        return freq

    def _extract_tokens(self, text: str) -> list[str]:
        normalized = self._normalize(text)
        tokens: list[str] = []
        tokens.extend([t for t in re.findall(r"[a-z_][a-z0-9_]{1,}", normalized) if t not in self.STOP_WORDS])
        tokens.extend([t for t in re.findall(r"[\u4e00-\u9fff]{2,8}", normalized) if t not in self.STOP_WORDS])
        return tokens

    def _infer_language(self, file_path: Path) -> str:
        mapping = {".py": "python", ".js": "javascript", ".jsx": "javascript", ".ts": "typescript", ".tsx": "typescript", ".go": "go", ".java": "java", ".sql": "sql", ".yaml": "yaml", ".yml": "yaml", ".json": "json", ".toml": "toml", ".ini": "ini", ".conf": "config", ".sh": "shell"}
        return mapping.get(file_path.suffix.lower(), "text")

    def _build_python_signature(self, node: ast.AST) -> str:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return ""
        args: list[str] = [a.arg for a in node.args.args]
        if node.args.vararg is not None:
            args.append(f"*{node.args.vararg.arg}")
        args.extend(a.arg for a in node.args.kwonlyargs)
        if node.args.kwarg is not None:
            args.append(f"**{node.args.kwarg.arg}")
        return f"{'async def' if isinstance(node, ast.AsyncFunctionDef) else 'def'} {node.name}({', '.join(args)})"

    def _build_line_level_excerpt(self, *, content: str, content_start_line: int, matched_terms: list[str], matched_patterns: list[str]) -> dict[str, Any]:
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

    def _to_relative_path(self, path: Path) -> str:
        try:
            return path.relative_to(self.project_root).as_posix()
        except ValueError:
            return path.as_posix()

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.lower()).strip()


def parse_code_dirs_from_env(*, project_root: Path) -> list[Path] | None:
    """解析代码索引目录。

    环境变量：
    - WORKFLOW_CODE_RETRIEVER_DIRS（逗号/分号分隔）
    """
    raw = os.getenv("WORKFLOW_CODE_RETRIEVER_DIRS", "").strip()
    if not raw:
        # 默认目录收口到仓库根目录 `codes/`，避免未配置时扫描整个工程目录。
        default_codes_dir = (project_root / "codes").resolve()
        if default_codes_dir.exists():
            return [default_codes_dir]
        return None
    dirs: list[Path] = []
    for part in [p.strip() for p in re.split(r"[;,]", raw) if p.strip()]:
        path = Path(part)
        if not path.is_absolute():
            path = (project_root / path).resolve()
        if path.exists():
            dirs.append(path)
    return dirs or None
