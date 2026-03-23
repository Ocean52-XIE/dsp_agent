# -*- coding: utf-8 -*-
"""
该模块实现工作流节点`retrieve_code` 的处理逻辑，负责读取状态并输出增量结果。

简化版检索策略：
    - BM25：词项精确匹配（RRF 归一化）
    - Embedding：语义向量匹配（RRF 归一化）
    - Pattern：精确标识符匹配

已移除的冗余路径：
    - TFIDF：与 BM25 功能重复
    - Ensemble：BM25+TFIDF 的融合，已移除
    - RG (ripgrep)：外部依赖，已移除
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import ast
import hashlib
import logging
import os
import re
from typing import Any, Iterator

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from log import get_file_logger
from workflow.common.func_utils import env_bool, env_float, env_int
from domain_profile import RerankerProfile, EmbeddingProfile

# 模块级日志器
logger = logging.getLogger(__name__)


# ==============================================================================
# 数据类定义
# ==============================================================================

@dataclass
class CodeParentChunk:
    """
    代码父块（符号级别）

    父块代表一个完整的符号（函数、类、方法等），用于提供符号级别的上下文，
    在最终结果展示时使用。

    Attributes:
        parent_id: 父块唯一标识（基于路径和符号名生成的哈希）
        source_path: 源文件路径
        language: 编程语言
        chunk_type: 类型（function/class/method/file）
        symbol_name: 符号名称（函数名/类名）
        signature: 函数签名（包含参数和返回类型）
        start_line: 起始行号
        end_line: 结束行号
        content: 完整内容（符号的全部代码）
        normalized_text: 归一化文本（用于检索）
        normalized_path: 归一化路径（小写、统一分隔符）
        normalized_symbol: 归一化符号名（小写）
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
    代码子块（滑动窗口）

    子块是父块的细分，使用滑动窗口方式生成，提供细粒度检索，
    提高召回精度。每个子块约 36 行，重叠 8 行。

    Attributes:
        child_id: 子块唯一标识（基于父块 ID 和行号范围生成）
        parent_id: 关联的父块 ID
        source_path: 源文件路径
        language: 编程语言
        chunk_type: 类型
        symbol_name: 符号名称
        signature: 函数签名
        start_line: 起始行号
        end_line: 结束行号
        content: 内容（约 36 行）
        normalized_text: 归一化文本（用于检索）
        normalized_path: 归一化路径
        normalized_symbol: 归一化符号名
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


# ==============================================================================
# 配置类定义
# ==============================================================================

@dataclass
class CodeRetrieverRuntimeConfig:
    """
    Code 检索器运行时配置（简化版）

    简化后的检索策略：BM25 + Embedding + Pattern
    移除了 TFIDF、Ensemble、RG 等冗余路径

    分数归一化：
        所有检索路径统一使用 RRF(k=60) 归一化，确保分数在 [0, 1] 范围内。

    Attributes:
        default_top_k: 默认返回结果数
        max_child_candidates: 最大子块候选数
        max_results_per_path: 每个路径最大结果数
        bm25_weight: BM25 检索权重（RRF 归一化后）
        pattern_weight: 模式匹配权重
        parent_best_pattern_weight: 父块最佳模式权重
        parent_avg_pattern_weight: 父块平均模式权重
        min_final_score: 最低最终分数阈值
        grade_high_top1_threshold: 高质量评级 Top1 阈值
        grade_medium_top1_threshold: 中等质量评级 Top1 阈值
        enable_embedding: 是否启用向量检索
        embedding_model: 向量模型名称
        embedding_device: 向量模型运行设备
        embedding_top_k: 向量检索返回数量
        embedding_persist_root: 向量索引持久化目录
        embedding_weight: 向量检索权重
    """
    default_top_k: int = 4
    max_child_candidates: int = 64
    max_results_per_path: int = 2

    # BM25 检索权重（RRF 分数 [0,1]）
    bm25_weight: float = 0.35

    # 模式匹配权重
    pattern_weight: float = 0.20
    parent_best_pattern_weight: float = 0.15
    parent_avg_pattern_weight: float = 0.08

    # 质量评级阈值
    min_final_score: float = 0.30
    grade_high_top1_threshold: float = 0.85
    grade_medium_top1_threshold: float = 0.55

    # 向量检索配置
    enable_embedding: bool = True
    embedding_model: str = "BAAI/bge-base-zh-v1.5"
    embedding_device: str = "cpu"
    embedding_top_k: int = 4
    embedding_persist_root: str = ".vectorstore_code"
    embedding_weight: float = 0.40

    # RRF 常量：k=60 是信息检索领域验证的标准参数
    RRF_K: int = 60

    @classmethod
    def from_env(cls) -> "CodeRetrieverRuntimeConfig":
        """从环境变量加载配置"""
        return cls(
            default_top_k=env_int("WORKFLOW_CODE_RETRIEVER_TOP_K", 4, minimum=1),
            max_child_candidates=env_int("WORKFLOW_CODE_RETRIEVER_MAX_CHILD_CANDIDATES", 64, minimum=8),
            max_results_per_path=env_int("WORKFLOW_CODE_RETRIEVER_MAX_PER_PATH", 2, minimum=1),
            # RRF 归一化后的权重（范围 [0, 1]）
            bm25_weight=env_float("WORKFLOW_CODE_RETRIEVER_BM25_WEIGHT", 0.35, minimum=0.0),
            pattern_weight=env_float("WORKFLOW_CODE_RETRIEVER_PATTERN_WEIGHT", 0.20, minimum=0.0),
            parent_best_pattern_weight=env_float("WORKFLOW_CODE_RETRIEVER_PARENT_BEST_PATTERN_WEIGHT", 0.15, minimum=0.0),
            parent_avg_pattern_weight=env_float("WORKFLOW_CODE_RETRIEVER_PARENT_AVG_PATTERN_WEIGHT", 0.08, minimum=0.0),
            min_final_score=env_float("WORKFLOW_CODE_RETRIEVER_MIN_FINAL_SCORE", 0.30, minimum=0.0),
            grade_high_top1_threshold=env_float("WORKFLOW_CODE_RETRIEVER_GRADE_HIGH_TOP1_THRESHOLD", 0.85, minimum=0.0),
            grade_medium_top1_threshold=env_float("WORKFLOW_CODE_RETRIEVER_GRADE_MEDIUM_TOP1_THRESHOLD", 0.55, minimum=0.0),
            # 向量检索配置
            enable_embedding=env_bool("WORKFLOW_CODE_EMBEDDING_ENABLED", True),
            embedding_model=os.getenv("WORKFLOW_CODE_EMBEDDING_MODEL", "BAAI/bge-base-zh-v1.5"),
            embedding_device=os.getenv("WORKFLOW_CODE_EMBEDDING_DEVICE", "cpu"),
            embedding_top_k=env_int("WORKFLOW_CODE_EMBEDDING_TOP_K", 4, minimum=1),
            embedding_persist_root=os.getenv("WORKFLOW_CODE_EMBEDDING_PERSIST_ROOT", ".vectorstore_code"),
            embedding_weight=env_float("WORKFLOW_CODE_EMBEDDING_WEIGHT", 0.40, minimum=0.0),
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
        embedding_profile: "EmbeddingProfile | None" = None,
    ) -> None:
        """
        初始化代码检索器。

        参数:
            project_root: 项目根目录
            code_dirs: 代码目录列表
            default_top_k: 默认返回结果数
            runtime_config: 运行时配置
            reranker_profile: Cross-Encoder 重排器配置
            embedding_profile: Embedding 向量检索配置（可选）
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
        self._semantic_docs: list[Document] = []  # 语义化文档（用于 Embedding 检索）
        self._children_by_path: defaultdict[str, list[CodeChildChunk]] = defaultdict(list)
        self._symbol_index: dict[str, set[str]] = defaultdict(set)
        self._path_token_index: dict[str, set[str]] = defaultdict(set)
        self.last_search_profile: dict[str, Any] = {}
        self._index_read_error_count = 0

        self._bm25: BM25Retriever | None = None
        # 移除 TFIDF、Ensemble（与 BM25 功能重复）和 RG（外部依赖）

        # Embedding 向量检索器（可选）
        self._embedding_retriever: Any = None
        self._embedding_profile = embedding_profile

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

        # 初始化 Cross-Encoder 重排器（可选）
        if self._reranker_profile and self._reranker_profile.enabled and not self._reranker:
            self._init_reranker()

        # 记录重排器状态
        self._logger.info(
            "workflow.code_reranker.status",
            enabled=self._reranker is not None,
            model=self._reranker_profile.model if self._reranker_profile else None,
        )

        # 初始化 Embedding 向量检索器（可选）
        if self._embedding_profile and self._embedding_profile.enabled and not self._embedding_retriever:
            self._init_embedding_retriever()

        # 记录 Embedding 检索器状态
        self._logger.info(
            "workflow.code_embedding.status",
            enabled=self._embedding_retriever is not None,
            model=self._embedding_profile.model if self._embedding_profile else None,
        )

    def _init_reranker(self) -> None:
        """初始化 Cross-Encoder 重排器。

        该方法延迟导入 CrossEncoderReranker，避免未安装 sentence-transformers 时启动失败。
        重排器初始化后会加载模型到内存，首次调用可能需要下载模型文件。
        """
        if not self._reranker_profile or not self._reranker_profile.enabled:
            return

        try:
            from retrievers.cross_encoder_reranker import (
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

    def _init_embedding_retriever(self) -> None:
        """初始化 Embedding 向量检索器。

        该方法为代码块生成语义化文本，并构建向量索引。
        语义化文本包含：符号名、签名、注释摘要、关键代码片段。
        """
        if not self._embedding_profile or not self._embedding_profile.enabled:
            return

        if not self._child_chunks:
            self._logger.warning(
                "workflow.code_embedding.no_chunks",
                message="没有代码块，跳过 Embedding 检索器初始化",
            )
            return

        try:
            from retrievers import EmbeddingRetriever, EmbeddingRetrieverConfig

            # 为每个子块生成语义化文本
            self._semantic_docs = []
            for child in self._child_chunks:
                semantic_text = self._build_semantic_text(child)
                if not semantic_text.strip():
                    continue
                self._semantic_docs.append(
                    Document(
                        page_content=semantic_text,
                        metadata={
                            "child_id": child.child_id,
                            "parent_id": child.parent_id,
                            "path": self._to_relative_path(child.source_path),
                            "symbol_name": child.symbol_name,
                            "chunk_type": child.chunk_type,
                            "language": child.language,
                        },
                    )
                )

            if not self._semantic_docs:
                self._logger.warning(
                    "workflow.code_embedding.no_semantic_docs",
                    message="没有生成语义化文档，跳过 Embedding 检索器初始化",
                )
                return

            # 创建 Embedding 检索器配置
            config = EmbeddingRetrieverConfig.from_profile(
                profile=self._embedding_profile,
                collection_name="code_semantic",
                persist_root=self._embedding_profile.persist_root,
            )

            # 初始化 Embedding 检索器
            self._embedding_retriever = EmbeddingRetriever(
                project_root=self.project_root,
                config=config,
            )
            index_stats = self._embedding_retriever.initialize(self._semantic_docs)

            self._logger.info(
                "workflow.code_embedding.initialized",
                model=self._embedding_profile.model,
                doc_count=index_stats.get("doc_count", 0),
                persist_dir=index_stats.get("persist_dir", "memory"),
            )

            # 控制台输出，方便验证
            print(f"[CodeEmbedding] 向量检索器初始化完成: {len(self._semantic_docs)} 个代码块")

        except ImportError as e:
            self._logger.warning(
                "workflow.code_embedding.import_error",
                error=str(e),
                message="EmbeddingRetriever 未安装，跳过向量检索器初始化",
            )
            self._embedding_retriever = None
        except Exception as e:
            self._logger.error(
                "workflow.code_embedding.init_error",
                error=str(e),
            )
            self._embedding_retriever = None

    def _build_semantic_text(self, child: CodeChildChunk) -> str:
        """为代码块生成语义化文本，用于向量检索。

        语义化文本包含多个维度的信息，以便于语义匹配：
        1. 符号类型和名称（如 "函数 get_user_info"）
        2. 签名信息（如 "def get_user_info(db: Session, user_id: int) -> User"）
        3. 提取的注释摘要
        4. 关键代码片段（去除重复的样板代码）

        参数:
            child: 代码子块

        返回:
            语义化文本字符串
        """
        parts: list[str] = []

        # 1. 符号类型和名称
        type_map = {
            "function": "函数",
            "class": "类",
            "method": "方法",
            "file": "文件",
        }
        symbol_type = type_map.get(child.chunk_type, child.chunk_type)
        parts.append(f"{symbol_type} {child.symbol_name}")

        # 2. 签名信息
        if child.signature and child.signature != child.symbol_name:
            parts.append(child.signature)

        # 3. 提取注释摘要
        comments = self._extract_comments(child.content, child.language)
        if comments:
            parts.append(comments)

        # 4. 关键代码片段（前 300 字符，去除过长行）
        content_preview = self._extract_key_code(child.content)
        if content_preview:
            parts.append(content_preview)

        # 5. 添加路径信息（帮助理解上下文）
        relative_path = self._to_relative_path(child.source_path)
        parts.append(f"文件路径: {relative_path}")

        return " ".join(parts)

    def _extract_comments(self, content: str, language: str) -> str:
        """从代码中提取注释摘要。

        参数:
            content: 代码内容
            language: 编程语言

        返回:
            注释摘要字符串
        """
        comments: list[str] = []

        if language == "python":
            # Python: 提取 docstring 和 # 注释
            # 提取三引号 docstring
            docstring_patterns = [
                r'"""([\s\S]*?)"""',
                r"'''([\s\S]*?)'''",
            ]
            for pattern in docstring_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    cleaned = match.strip()
                    if cleaned and len(cleaned) > 10:
                        comments.append(cleaned[:200])

            # 提取 # 注释
            for line in content.split("\n"):
                if "#" in line:
                    comment = line.split("#", 1)[1].strip()
                    if comment and len(comment) > 5:
                        comments.append(comment[:100])

        else:
            # 通用: 提取 // 和 /* */ 注释
            # 单行注释
            for line in content.split("\n"):
                if "//" in line:
                    comment = line.split("//", 1)[1].strip()
                    if comment and len(comment) > 5:
                        comments.append(comment[:100])

            # 多行注释
            block_comments = re.findall(r"/\*([\s\S]*?)\*/", content)
            for match in block_comments:
                cleaned = match.strip()
                if cleaned and len(cleaned) > 10:
                    comments.append(cleaned[:200])

        # 合并注释，限制总长度
        if not comments:
            return ""

        merged = " ".join(comments[:5])  # 最多取前 5 条注释
        return merged[:500] if len(merged) > 500 else merged

    def _extract_key_code(self, content: str) -> str:
        """提取关键代码片段。

        去除过长的行，保留核心代码逻辑。

        参数:
            content: 代码内容

        返回:
            关键代码片段
        """
        lines = content.split("\n")
        key_lines: list[str] = []

        for line in lines:
            # 去除空行和过长的行
            stripped = line.strip()
            if not stripped:
                continue
            if len(stripped) > 120:
                # 过长的行截断
                stripped = stripped[:120] + "..."

            # 保留有意义的代码行
            # 跳过纯 import 语句、纯 pass、纯 ...
            if stripped.startswith(("import ", "from ")) and len(key_lines) > 0:
                continue
            if stripped in ("pass", "...", "break", "continue"):
                continue

            key_lines.append(stripped)

            # 限制行数
            if len(key_lines) >= 15:
                break

        return " ".join(key_lines)[:300]

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

        # 简化后只使用 BM25 + Embedding（移除 TFIDF/Ensemble/RG）
        if not query or not self._child_chunks or self._bm25 is None:
            self.last_search_profile = {"latency_ms": round((perf_counter() - started) * 1000, 3)}
            return []

        query_phrases = [query, *(retrieval_queries or [])]
        merged_query = " ".join([*query_phrases, module_name or ""]).strip()
        patterns = self._extract_patterns(merged_query)
        module_tokens = {tok for tok in re.split(r"[._\-/\s]+", self._normalize(module_name or "")) if len(tok) >= 2}

        candidate_k = max(self.runtime_config.max_child_candidates, (top_k or self.runtime_config.default_top_k) * 8)
        bm25_docs: list[Document] = []
        embedding_docs: list[Document] = []

        # BM25 检索
        self._bm25.k = candidate_k
        bm25_docs = self._bm25.invoke(merged_query)

        # Embedding 向量检索
        embedding_start = perf_counter()
        if self._embedding_retriever and self._embedding_retriever.is_initialized:
            try:
                embedding_results = self._embedding_retriever.search_with_scores(
                    merged_query,
                    top_k=candidate_k,
                )
                for doc, score in embedding_results:
                    child_id = str(doc.metadata.get("child_id", ""))
                    if child_id:
                        embedding_docs.append(doc)
                self._logger.debug(
                    "workflow.code_embedding.search",
                    query_preview=merged_query[:50],
                    hits=len(embedding_docs),
                    latency_ms=round((perf_counter() - embedding_start) * 1000, 2),
                )
            except Exception as e:
                self._logger.warning(
                    "workflow.code_embedding.search_error",
                    error=str(e),
                    query_preview=merged_query[:50],
                )

        bm25_rank = self._rank_map(bm25_docs)
        embedding_rank = self._rank_map(embedding_docs)

        candidate_child_ids = list(dict.fromkeys([
            *[str(doc.metadata.get("child_id", "")) for doc in bm25_docs],
            *[str(doc.metadata.get("child_id", "")) for doc in embedding_docs],
        ]))[: self.runtime_config.max_child_candidates]

        scored_children: list[dict[str, Any]] = []
        for child_id in candidate_child_ids:
            child = self._child_by_id.get(child_id)
            if child is None:
                continue
            lexical, matched_terms = self._score_lexical(child, patterns)
            if module_tokens and any(tok in child.normalized_path for tok in module_tokens):
                lexical += 0.15
            pattern_score, matched_patterns = self._score_pattern(child, patterns)

            # 计算各路检索分数（RRF 归一化，范围 [0, 1]）
            bm25_score = self._rank_score(bm25_rank.get(child_id))
            embedding_score = self._rank_score(embedding_rank.get(child_id))

            # 简化后的混合评分公式
            # lexical [0,1] + bm25 [0,1] + embedding [0,1] + pattern [0,1]
            score = lexical * 0.20
            if embedding_rank:
                # 有 Embedding 时
                score += bm25_score * 0.35
                score += embedding_score * self.runtime_config.embedding_weight
            else:
                # 无 Embedding 时
                score += bm25_score * 0.70
            score += pattern_score * self.runtime_config.pattern_weight

            if score < self.runtime_config.min_final_score:
                continue
            scored_children.append(
                {
                    "child": child,
                    "score": score,
                    "pattern_score": pattern_score,
                    "matched_terms": matched_terms,
                    "matched_patterns": matched_patterns,
                    # 各路检索分数（用于调试）
                    "bm25_score": bm25_score,
                    "embedding_score": embedding_score,
                    "lexical_score": lexical,
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

        parent_items: list[dict[str, Any]] = []
        for bucket in parent_buckets.values():
            top_scores = sorted(bucket["scores"], reverse=True)[:2]
            top_pattern = sorted(bucket["pattern_scores"], reverse=True)[:2]
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
                    "content": best_child.content,  # 始终包含完整内容，供下游节点使用
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
                    },
                }
            )

        self.last_search_profile = {
            "latency_ms": round((perf_counter() - started) * 1000, 3),
            "child_candidates": len(scored_children),
            "parent_candidates": len(parent_items),
            "selected_count": len(hits),
            # BM25 检索状态
            "bm25": {
                "hits": len(bm25_docs),
                "weight": self.runtime_config.bm25_weight,
            },
            # Embedding 检索状态
            "embedding": {
                "enabled": self._embedding_retriever is not None and self._embedding_retriever.is_initialized,
                "hits": len(embedding_docs),
                "weight": self.runtime_config.embedding_weight,
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
                top_k=top_k or self.runtime_config.default_top_k,
                content_key="content",
                source="code",
            )
            # 用重排结果替换原来的 hits，更新分数和排名
            reranked_results: list[dict[str, Any]] = []
            for rank, hit in enumerate(reranked_hits, start=1):
                reranked_results.append({
                    **hit,
                    "score": round(float(hit.get("rerank_score", 0.0)), 4),
                    "rank": rank,
                    "score_source": "reranker",
                    "retrieval_debug": {
                        **hit.get("retrieval_debug", {}),
                        "rerank_score": round(float(hit.get("rerank_score", 0.0)), 4),
                        "original_rank": hit.get("original_rank", rank),
                    },
                })
            hits = reranked_results  # 用重排结果替换
            rerank_latency = round((perf_counter() - rerank_start) * 1000, 3)
            self.last_search_profile["rerank"] = {
                "enabled": True,
                "latency_ms": rerank_latency,
                "model": self._reranker_profile.model if self._reranker_profile else None,
            }
            self.last_search_profile["latency_ms"] = round((perf_counter() - started) * 1000, 3)

        return hits

    def _get_chunk_content_by_hit(self, path: str, section: str, excerpt_lines: Any = None) -> str:
        """根据路径和章节获取 chunk 内容，用于重排器。

        参数:
            path: 文档相对路径
            section: 章节名称，格式为 "{chunk_type}:{symbol_name}"
            excerpt_lines: 可选的行范围信息（预留参数，暂未使用）

        返回:
            chunk 内容字符串
        """
        # _children_by_path 存储的是 CodeChildChunk 对象列表，直接遍历使用
        chunks = self._children_by_path.get(path, [])
        for chunk in chunks:
            # section 格式: "{chunk_type}:{symbol_name}"
            chunk_section = f"{chunk.chunk_type}:{chunk.symbol_name or 'file'}"
            if chunk_section == section:
                return chunk.content
        # 如果找不到精确匹配，返回第一个 chunk 的内容
        if chunks:
            return chunks[0].content
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

        # 只保留 BM25 索引（移除 TFIDF 和 Ensemble）
        if self._child_docs:
            self._bm25 = BM25Retriever.from_documents(self._child_docs)
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
        基于排名的分数归一化（Reciprocal Rank Fusion）。

        将检索排名转换为归一化分数，范围 [0, 1]。
        使用 RRF 公式: score = k / (rank + k)，k=60 是业界标准参数。

        RRF 的优势：
        1. 消除不同检索器原始分数量级差异
        2. 对异常值鲁棒（只依赖排名顺序）
        3. k=60 使衰减平缓，Top-K 结果间差距合理

        参数:
            rank: 检索排名（从 1 开始），None 表示未命中

        返回:
            归一化分数，范围 [0, 1]
            rank=1 → 0.984, rank=10 → 0.857, rank=50 → 0.545
        """
        if rank is None:
            return 0.0
        # RRF(k=60) - 信息检索领域验证的标准归一化方法
        k = 60
        return k / (rank + k)

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
        计算词法匹配分数。

        基于查询词在代码块中的匹配情况计算分数，范围 [0, 1]。
        与 RRF 归一化保持一致，确保与其他检索分数可比。

        评分规则：
        - 文本匹配：+0.15（长词 +0.25）
        - 符号匹配：+0.20
        - 路径匹配：+0.15
        - 签名匹配：+0.10

        参数:
            child: 代码子块
            patterns: 查询模式（包含 identifiers）

        返回:
            (分数, 匹配的词列表)
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
                # 文本匹配：限制最大贡献，长词权重更高
                score += min(cnt, 3) * (0.08 if len(token) >= 4 else 0.05)
            if token in child.normalized_symbol:
                score += 0.20  # 符号匹配权重高
            if token in child.normalized_path:
                score += 0.15  # 路径匹配
            if token in self._normalize(child.signature):
                score += 0.10  # 签名匹配

        # 限制最大分数为 1.0
        return min(score, 1.0), matched

    def _score_pattern(self, child: CodeChildChunk, patterns: dict[str, Any]) -> tuple[float, list[str]]:
        """
        计算模式匹配分数。

        基于精确标识符匹配计算分数，范围 [0, 1]。
        与 RRF 归一化保持一致，确保与其他检索分数可比。

        评分规则：
        - 精确符号匹配：+0.50
        - 部分符号匹配：+0.25（位置查询 +0.35）
        - 路径匹配：+0.20（位置查询 +0.30）
        - 字段匹配：+0.08
        - 公式表达式：+0.10
        - 位置意图：+0.05

        参数:
            child: 代码子块
            patterns: 查询模式

        返回:
            (分数, 匹配的模式列表)
        """
        score = 0.0
        matched: list[str] = []
        is_location = bool(patterns.get("is_location_query"))
        for token in patterns.get("exact_identifiers", []):
            if token == child.normalized_symbol:
                score += 0.50  # 精确符号匹配，权重最高
                matched.append(f"exact_symbol:{token}")
            elif token in child.normalized_symbol:
                score += 0.35 if is_location else 0.25
                matched.append(f"symbol_like:{token}")
            elif token in child.normalized_path:
                score += 0.30 if is_location else 0.20
                matched.append(f"path_like:{token}")
        for token in patterns.get("field_like_tokens", []):
            if token in child.normalized_text:
                score += 0.08
                matched.append(token)
        if patterns.get("is_formula_query") and "=" in child.content:
            score += 0.10
            matched.append("formula_expression")
        if is_location:
            score += 0.05
            matched.append("location_intent")

        # 限制最大分数为 1.0
        return min(score, 1.0), sorted(set(matched), key=len, reverse=True)

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


# ============================================================================
# 全局单例模式
# ============================================================================

# 全局 Code 检索器实例（单例）
_code_retriever_instance: LocalCodeRetriever | None = None


def get_code_retriever() -> LocalCodeRetriever | None:
    """获取全局 Code 检索器实例

    Returns:
        LocalCodeRetriever 实例，如果未初始化则返回 None
    """
    return _code_retriever_instance


def set_code_retriever(retriever: LocalCodeRetriever | None) -> None:
    """设置全局 Code 检索器实例

    Args:
        retriever: LocalCodeRetriever 实例
    """
    global _code_retriever_instance
    _code_retriever_instance = retriever
    if retriever:
        logger.info("[CodeRetriever] 全局单例已设置")
    else:
        logger.info("[CodeRetriever] 全局单例已清除")
