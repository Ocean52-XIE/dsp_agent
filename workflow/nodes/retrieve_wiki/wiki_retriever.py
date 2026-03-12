from __future__ import annotations

"""本地 Markdown Wiki 检索器（P0 + P1.1）。

P0：
1. 切片优化：标题层级感知 + 列表/表格/代码块保护 + 邻接块回拼。
2. 召回优化：同义词扩展 + 指标词典 + 缩写归一。

P1.1：
1. Hybrid 检索：BM25 + 向量（TF-IDF）+ 规则分融合。
2. 支持加载“学习到的权重文件”，避免固定单路分数。
"""

from collections import defaultdict
from dataclasses import dataclass
from math import log, sqrt
from pathlib import Path
from time import perf_counter
import json
import os
import re
from typing import Any
from workflow.runtime_logging import get_file_logger


@dataclass
class WikiChunk:
    """单个可检索切片。"""

    chunk_id: int
    source_path: Path
    title: str
    section: str
    chunk_type: str
    content: str
    normalized_text: str
    normalized_title: str
    normalized_section: str
    prev_chunk_id: int | None = None
    next_chunk_id: int | None = None


@dataclass
class HybridScoreWeights:
    """Hybrid 融合权重。"""

    # 默认值来自当前评测集上的一轮离线调优结果，
    # 目标是优先提升 Top1 质量与可解释性。
    bm25: float = 0.25
    vector: float = 0.15
    lexical: float = 0.60
    source: str = "default"

    @classmethod
    def from_env(cls) -> "HybridScoreWeights":
        """从权重文件或环境变量加载融合权重。"""
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
                    # 配置异常时不抛错，降级到默认权重。
                    pass

        def env_float(name: str, default: float) -> float:
            raw = os.getenv(name, "").strip()
            if not raw:
                return default
            try:
                return float(raw)
            except ValueError:
                return default

        weights.bm25 = env_float("WORKFLOW_WIKI_WEIGHT_BM25", weights.bm25)
        weights.vector = env_float("WORKFLOW_WIKI_WEIGHT_VECTOR", weights.vector)
        weights.lexical = env_float("WORKFLOW_WIKI_WEIGHT_LEXICAL", weights.lexical)
        return weights.normalized()

    def normalized(self) -> "HybridScoreWeights":
        """归一化权重，保证三路权重和为 1。"""
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
    """Wiki 检索运行参数。"""

    default_top_k: int = 4
    # 默认每个文档仅保留 1 个切片，优先保障路径级多样性，减少同文档“刷屏”。
    max_chunks_per_doc: int = 1
    stage2_candidate_multiplier: int = 6
    stage2_min_candidates: int = 12
    paragraph_chunk_max_chars: int = 520
    paragraph_chunk_min_chars: int = 160
    excerpt_max_chars: int = 220
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    min_retrieval_score: float = 0.1
    # 总览型文档（如“总体架构”）在非架构问题下的惩罚系数。
    architecture_doc_penalty: float = 1.2
    # P1：函数/文件定位问法下，对 00-总体架构文档施加更强惩罚，
    # 避免“泛化架构描述”干扰代码定位类问题。
    architecture_doc_location_penalty: float = 2.0
    # 排障类问题命中排障手册时的额外加分，帮助把手册顶到更靠前位置。
    troubleshoot_doc_boost: float = 0.2

    @classmethod
    def from_env(cls, *, default_top_k: int) -> "WikiRetrieverRuntimeConfig":
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
            default_top_k=env_int("WORKFLOW_WIKI_TOP_K", default_top_k, 1),
            max_chunks_per_doc=env_int("WORKFLOW_WIKI_MAX_CHUNKS_PER_DOC", 1, 1),
            stage2_candidate_multiplier=env_int("WORKFLOW_WIKI_STAGE2_MULTIPLIER", 6, 1),
            stage2_min_candidates=env_int("WORKFLOW_WIKI_STAGE2_MIN_CANDIDATES", 12, 1),
            paragraph_chunk_max_chars=env_int("WORKFLOW_WIKI_PARAGRAPH_MAX_CHARS", 520, 120),
            paragraph_chunk_min_chars=env_int("WORKFLOW_WIKI_PARAGRAPH_MIN_CHARS", 160, 40),
            excerpt_max_chars=env_int("WORKFLOW_WIKI_EXCERPT_MAX_CHARS", 220, 60),
            bm25_k1=env_float("WORKFLOW_WIKI_BM25_K1", 1.2, 0.1),
            bm25_b=env_float("WORKFLOW_WIKI_BM25_B", 0.75, 0.0),
            min_retrieval_score=env_float("WORKFLOW_WIKI_MIN_RETRIEVAL_SCORE", 0.0, 0.0),
            architecture_doc_penalty=env_float("WORKFLOW_WIKI_ARCH_DOC_PENALTY", 1.2, 0.0),
            architecture_doc_location_penalty=env_float(
                "WORKFLOW_WIKI_ARCH_DOC_LOCATION_PENALTY",
                2.0,
                0.0,
            ),
            troubleshoot_doc_boost=env_float("WORKFLOW_WIKI_TROUBLESHOOT_DOC_BOOST", 0.2, 0.0),
        )


class MarkdownWikiRetriever:
    """基于本地 Markdown 文档的 Hybrid 检索器。"""

    # 模块到主文档的轻量先验映射（用于 Stage2 重排加分）。
    # 说明：该映射只做“软引导”而非硬过滤，避免影响跨模块问题的召回覆盖。
    MODULE_DOC_HINTS: dict[str, tuple[str, ...]] = {
        "ad-recall": ("01-", "在线召回"),
        "rate-prediction": ("02-", "两率预估"),
        "bid-optimizer": ("03-", "出价策略"),
        "rerank-engine": ("04-", "精排策略"),
    }
    MODULE_DOC_PRIOR_BOOST: float = 1.4

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

    DOMAIN_TERMS = (
        "广告引擎",
        "在线投放",
        "召回",
        "候选",
        "两率预估",
        "点击率",
        "转化率",
        "pctr",
        "pcvr",
        "出价",
        "ocpc",
        "tcpa",
        "troas",
        "pacing",
        "精排",
        "重排",
        "ecpm",
        "频控",
        "计费",
        "日志回传",
        "链路",
        "流程",
        "指标",
        "口径",
        "trace_id",
        "request_id",
    )

    ABBR_NORMALIZATION = {
        "ctr": "点击率",
        "cvr": "转化率",
        "pctr": "预估点击率",
        "pcvr": "预估转化率",
        "ecpm": "千次展示收益",
        "rpm": "千次请求收益",
        "cpc": "单次点击成本",
        "cpa": "单次转化成本",
        "roas": "广告投入产出比",
    }

    SYNONYM_MAP: dict[str, tuple[str, ...]] = {
        "在线投放": ("广告投放", "实时投放", "在线链路"),
        "核心链路": ("核心流程", "主链路", "主流程"),
        "指标": ("监控指标", "关键指标", "业务指标"),
        "口径": ("定义", "统计口径", "计算口径"),
        "召回": ("候选召回", "候选生成"),
        "两率预估": ("ctr cvr 预估", "点击率转化率预估"),
        "出价": ("竞价", "bid", "出价策略"),
        "精排": ("排序", "重排", "rerank"),
        "排障": ("排查", "故障定位", "问题定位"),
    }

    RERANK_INTENT_KEYWORDS: dict[str, tuple[str, ...]] = {
        "metric": ("指标", "口径", "监控", "ctr", "cvr", "pctr", "pcvr", "ecpm", "roi", "winrate"),
        "pipeline": ("链路", "流程", "阶段", "环节", "步骤", "先后"),
        "troubleshoot": ("排障", "排查", "故障", "异常", "无量", "抖动", "定位"),
        "formula": ("公式", "计算", "怎么算", "target_cpa", "target_roas", "bid", "="),
        "architecture": ("架构", "总体", "全景", "模块关系", "基架"),
        # P1：代码定位意图，供 Stage2 对 wiki 结果做额外去泛化处理。
        "code_location": (
            "哪个函数",
            "什么函数",
            "函数名",
            "入口函数",
            "哪个文件",
            "文件路径",
            "路径在哪",
            "哪一行",
            "行号",
            "实现位置",
            "where",
            "which function",
        ),
    }

    def __init__(self, *, wiki_dir: Path, project_root: Path, default_top_k: int = 4) -> None:
        self.wiki_dir = wiki_dir
        self.project_root = project_root
        self.runtime_config = WikiRetrieverRuntimeConfig.from_env(default_top_k=default_top_k)
        self.hybrid_weights = HybridScoreWeights.from_env()
        self._logger = get_file_logger(project_root=project_root)

        self._chunks: list[WikiChunk] = []
        self._chunk_by_id: dict[int, WikiChunk] = {}
        self._token_freq_by_chunk: dict[int, dict[str, int]] = {}
        self._doc_len_by_chunk: dict[int, int] = {}
        self._df: dict[str, int] = defaultdict(int)
        self._idf: dict[str, float] = {}
        self._vector_by_chunk: dict[int, dict[str, float]] = {}
        self._vector_norm_by_chunk: dict[int, float] = {}
        self._avg_doc_len: float = 1.0

        started_at = perf_counter()
        index_stats = self._build_index()
        self._logger.info(
            "workflow.wiki_index.built",
            wiki_dir=str(self.wiki_dir),
            file_count=index_stats.get("file_count", 0),
            indexed_file_count=index_stats.get("indexed_file_count", 0),
            chunk_count=index_stats.get("chunk_count", 0),
            read_error_count=index_stats.get("read_error_count", 0),
            hybrid_weight_source=self.hybrid_weights.source,
            default_top_k=self.runtime_config.default_top_k,
            latency_ms=int((perf_counter() - started_at) * 1000),
        )
        if not self._chunks:
            self._logger.warning(
                "workflow.wiki_index.empty",
                wiki_dir=str(self.wiki_dir),
                file_count=index_stats.get("file_count", 0),
            )

    def search(
        self,
        *,
        user_query: str,
        retrieval_queries: list[str] | None = None,
        module_name: str | None = None,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """执行 Wiki 检索并返回结构化结果。"""
        if not user_query.strip() or not self._chunks:
            return []

        query_phrases = [user_query.strip(), *(retrieval_queries or [])]
        expanded_phrases = self._expand_query_phrases(query_phrases)
        terms = self._extract_query_terms(expanded_phrases)
        if not terms:
            terms = self._extract_fallback_ngrams(expanded_phrases)
        query_vector = self._build_query_vector(terms)
        query_tf = self._count_tokens(" ".join(terms))

        recalled = self._stage1_hybrid_recall(
            query_phrases=expanded_phrases,
            terms=terms,
            query_tf=query_tf,
            query_vector=query_vector,
            module_name=module_name or "",
        )
        if not recalled:
            return []

        stage2_candidates = self._build_stage2_candidates(recalled, top_n=top_k or self.runtime_config.default_top_k)
        reranked = self._stage2_rerank(
            stage2_candidates,
            query_phrases=expanded_phrases,
            terms=terms,
            module_name=module_name or "",
        )
        selected = self._select_diverse(
            reranked,
            top_n=top_k or self.runtime_config.default_top_k,
            max_per_doc=self.runtime_config.max_chunks_per_doc,
        )

        result_items: list[dict[str, Any]] = []
        for rank, item in enumerate(selected, start=1):
            chunk = item["chunk"]
            result_items.append(
                {
                    "source_type": "wiki",
                    "title": chunk.title,
                    "path": self._to_relative_path(chunk.source_path),
                    "score": round(float(item["final_score"]), 4),
                    "stage1_score": round(float(item["stage1_score"]), 4),
                    "section": chunk.section,
                    "chunk_type": chunk.chunk_type,
                    "rank": rank,
                    "excerpt": self._build_excerpt_with_neighbors(
                        chunk=chunk,
                        matched_terms=item["matched_terms"],
                        matched_phrases=item["matched_phrases"],
                    ),
                    "rerank_features": {
                        "hybrid_weights": {
                            "bm25": round(self.hybrid_weights.bm25, 4),
                            "vector": round(self.hybrid_weights.vector, 4),
                            "lexical": round(self.hybrid_weights.lexical, 4),
                            "source": self.hybrid_weights.source,
                        },
                        "stage1_bm25": round(float(item["bm25_score"]), 4),
                        "stage1_vector": round(float(item["vector_score"]), 4),
                        "stage1_lexical": round(float(item["lexical_score"]), 4),
                        "stage2_intent": round(float(item["intent_score"]), 4),
                        "stage2_title_section": round(float(item["title_section_score"]), 4),
                        "stage2_type": round(float(item["chunk_type_score"]), 4),
                        "stage2_doc_boost": round(float(item["doc_boost"]), 4),
                        "stage2_penalty": round(float(item["doc_penalty"]), 4),
                    },
                }
            )
        return result_items

    def get_chunk_snapshot(self) -> list[dict[str, Any]]:
        """导出切片快照，用于验证切片质量。"""
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
        """扫描 Markdown 并构建索引。"""
        stats = {
            "file_count": 0,
            "indexed_file_count": 0,
            "chunk_count": 0,
            "read_error_count": 0,
        }
        if not self.wiki_dir.exists():
            return stats
        markdown_files = sorted(self.wiki_dir.rglob("*.md"))
        stats["file_count"] = len(markdown_files)
        next_chunk_id = 1
        for md_file in markdown_files:
            try:
                raw = md_file.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                try:
                    raw = md_file.read_text(encoding="utf-8", errors="ignore")
                except OSError as exc:
                    stats["read_error_count"] += 1
                    self._logger.warning(
                        "workflow.wiki_index.read_failed",
                        path=self._to_relative_path(md_file),
                        error_type=type(exc).__name__,
                    )
                    continue
            except OSError as exc:
                stats["read_error_count"] += 1
                self._logger.warning(
                    "workflow.wiki_index.read_failed",
                    path=self._to_relative_path(md_file),
                    error_type=type(exc).__name__,
                )
                continue
            file_chunks = self._split_markdown(md_file, raw, start_chunk_id=next_chunk_id)
            if not file_chunks:
                continue
            stats["indexed_file_count"] += 1
            next_chunk_id = file_chunks[-1].chunk_id + 1
            self._chunks.extend(file_chunks)
            for chunk in file_chunks:
                self._chunk_by_id[chunk.chunk_id] = chunk
        self._build_token_indexes()
        stats["chunk_count"] = len(self._chunks)
        return stats

    def _split_markdown(self, source_path: Path, raw_text: str, *, start_chunk_id: int) -> list[WikiChunk]:
        """按标题层级切片，并保护列表/表格/代码块。"""
        lines = raw_text.splitlines()
        if not lines:
            return []

        chunks: list[WikiChunk] = []
        heading_stack: list[str] = []
        document_title = source_path.stem
        section_buffer: list[str] = []
        in_code_fence = False
        chunk_id = start_chunk_id

        def current_section() -> str:
            return " / ".join(heading_stack) if heading_stack else document_title

        def flush_section_buffer() -> None:
            nonlocal chunk_id
            section_text = "\n".join(section_buffer).strip()
            section_buffer.clear()
            if not section_text:
                return
            blocks = self._split_section_to_blocks(section_text)
            blocks = self._merge_contextual_blocks(blocks)
            for block_type, block_text in blocks:
                # 长段落做句级切分；列表/表格/代码块尽量保留完整。
                if block_type == "paragraph":
                    pieces = self._split_long_paragraph(block_text)
                else:
                    pieces = [block_text.strip()]
                for piece in pieces:
                    content = piece.strip()
                    if not content:
                        continue
                    section_name = current_section()
                    chunk = WikiChunk(
                        chunk_id=chunk_id,
                        source_path=source_path,
                        title=document_title,
                        section=section_name,
                        chunk_type=block_type,
                        content=content,
                        normalized_text=self._normalize(f"{document_title} {section_name} {content}"),
                        normalized_title=self._normalize(document_title),
                        normalized_section=self._normalize(section_name),
                    )
                    chunks.append(chunk)
                    chunk_id += 1

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```"):
                in_code_fence = not in_code_fence

            # 仅在非代码块状态下处理标题，避免误切。
            if not in_code_fence and re.match(r"^#{1,6}\s+", stripped):
                flush_section_buffer()
                level = len(stripped) - len(stripped.lstrip("#"))
                heading_text = stripped[level:].strip()
                if not heading_text:
                    continue
                if level == 1:
                    document_title = heading_text
                while len(heading_stack) >= level:
                    heading_stack.pop()
                heading_stack.append(heading_text)
                continue

            section_buffer.append(line)

        flush_section_buffer()
        self._link_neighbors(chunks)
        return chunks

    def _split_section_to_blocks(self, section_text: str) -> list[tuple[str, str]]:
        """将 section 文本拆成语义块。"""
        lines = section_text.splitlines()
        blocks: list[tuple[str, str]] = []
        buffer: list[str] = []
        current_type = "paragraph"
        in_code_fence = False

        def flush() -> None:
            nonlocal buffer, current_type
            text = "\n".join(buffer).strip()
            buffer = []
            if text:
                blocks.append((current_type, text))
            current_type = "paragraph"

        for line in lines:
            stripped = line.rstrip()
            if stripped.strip().startswith("```"):
                if not in_code_fence and buffer and current_type != "code":
                    flush()
                in_code_fence = not in_code_fence
                if not buffer:
                    current_type = "code"
                buffer.append(line)
                continue

            if in_code_fence:
                buffer.append(line)
                continue

            line_type = self._infer_line_type(stripped)
            if not stripped.strip():
                flush()
                continue

            if not buffer:
                current_type = line_type
                buffer.append(line)
                continue

            if current_type == line_type:
                buffer.append(line)
                continue

            # 仅允许段落自然续写，其它块类型隔离。
            if current_type == "paragraph" and line_type == "paragraph":
                buffer.append(line)
            else:
                flush()
                current_type = line_type
                buffer.append(line)

        flush()
        return blocks

    def _merge_contextual_blocks(self, blocks: list[tuple[str, str]]) -> list[tuple[str, str]]:
        """回拼强语义依赖块，减少证据断裂。"""
        if not blocks:
            return []

        merged: list[tuple[str, str]] = []
        index = 0
        while index < len(blocks):
            current_type, current_text = blocks[index]
            if index + 1 >= len(blocks):
                merged.append((current_type, current_text))
                break
            next_type, next_text = blocks[index + 1]
            if self._should_merge_blocks(
                current_type=current_type,
                current_text=current_text,
                next_type=next_type,
                next_text=next_text,
            ):
                merged_type = current_type if current_type != "paragraph" else next_type
                merged.append((merged_type, f"{current_text}\n{next_text}".strip()))
                index += 2
                continue
            merged.append((current_type, current_text))
            index += 1
        return merged

    def _should_merge_blocks(
        self,
        *,
        current_type: str,
        current_text: str,
        next_type: str,
        next_text: str,
    ) -> bool:
        """判断相邻块是否应回拼。"""
        current_norm = current_text.strip()
        next_norm = next_text.strip()
        if not current_norm or not next_norm:
            return False

        lead_markers = ("如下：", "如下", "包括：", "包括", "例如：", "例如", "如下所示：")
        if any(current_norm.endswith(marker) for marker in lead_markers):
            return True

        if current_type == "paragraph" and next_type in {"list", "table", "flow", "code"}:
            if any(token in current_norm for token in ("流程", "链路", "阶段", "指标", "步骤", "如下")):
                return True

        if next_type == "paragraph" and len(next_norm) <= 48:
            if any(token in current_norm for token in ("说明", "定义", "公式", "口径")):
                return True
        return False

    def _split_long_paragraph(self, text: str) -> list[str]:
        """长段落按句子边界切分。"""
        clean = text.strip()
        if len(clean) <= self.runtime_config.paragraph_chunk_max_chars:
            return [clean]

        sentences = re.split(r"(?<=[。！？!?；;])\s*", clean)
        pieces: list[str] = []
        current = ""
        for sentence in sentences:
            if not sentence:
                continue
            if not current:
                current = sentence
                continue
            merged = f"{current}{sentence}"
            if len(merged) <= self.runtime_config.paragraph_chunk_max_chars:
                current = merged
                continue
            if len(current) >= self.runtime_config.paragraph_chunk_min_chars:
                pieces.append(current.strip())
                current = sentence
            else:
                pieces.append(merged.strip())
                current = ""
        if current.strip():
            pieces.append(current.strip())
        return pieces or [clean]

    def _build_token_indexes(self) -> None:
        """构建 BM25 与向量索引。"""
        for chunk in self._chunks:
            tf = self._count_tokens(chunk.normalized_text)
            self._token_freq_by_chunk[chunk.chunk_id] = tf
            doc_len = sum(tf.values())
            self._doc_len_by_chunk[chunk.chunk_id] = max(doc_len, 1)
            for token in tf:
                self._df[token] += 1

        total_docs = max(len(self._chunks), 1)
        self._avg_doc_len = sum(self._doc_len_by_chunk.values()) / max(len(self._doc_len_by_chunk), 1)
        for token, df in self._df.items():
            self._idf[token] = log((total_docs - df + 0.5) / (df + 0.5) + 1.0)

        for chunk in self._chunks:
            tf = self._token_freq_by_chunk.get(chunk.chunk_id, {})
            vector = self._build_tfidf_vector(tf)
            self._vector_by_chunk[chunk.chunk_id] = vector
            self._vector_norm_by_chunk[chunk.chunk_id] = self._vector_norm(vector)

    def _stage1_hybrid_recall(
        self,
        *,
        query_phrases: list[str],
        terms: list[str],
        query_tf: dict[str, int],
        query_vector: dict[str, float],
        module_name: str,
    ) -> list[dict[str, Any]]:
        """Stage1：BM25 + 向量 + 规则分融合召回。"""
        scored: list[dict[str, Any]] = []
        module_norm = self._normalize(module_name)
        for chunk in self._chunks:
            bm25_score = self._score_bm25(chunk.chunk_id, query_tf)
            vector_score = self._cosine_similarity(
                query_vector,
                self._vector_by_chunk.get(chunk.chunk_id, {}),
                self._vector_norm_by_chunk.get(chunk.chunk_id, 0.0),
            )
            lexical_score, matched_terms, matched_phrases = self._score_lexical(
                chunk=chunk,
                query_phrases=query_phrases,
                terms=terms,
                module_norm=module_norm,
            )

            bm25_norm = bm25_score / (bm25_score + 8.0) if bm25_score > 0 else 0.0
            lexical_norm = lexical_score / (lexical_score + 12.0) if lexical_score > 0 else 0.0
            hybrid = (
                bm25_norm * self.hybrid_weights.bm25
                + vector_score * self.hybrid_weights.vector
                + lexical_norm * self.hybrid_weights.lexical
            )
            stage1_score = hybrid * 10.0
            if stage1_score < self.runtime_config.min_retrieval_score:
                continue
            scored.append(
                {
                    "chunk": chunk,
                    "stage1_score": stage1_score,
                    "bm25_score": bm25_score,
                    "vector_score": vector_score,
                    "lexical_score": lexical_score,
                    "matched_terms": matched_terms,
                    "matched_phrases": matched_phrases,
                }
            )
        scored.sort(key=lambda item: item["stage1_score"], reverse=True)
        return scored

    def _build_stage2_candidates(self, scored: list[dict[str, Any]], *, top_n: int) -> list[dict[str, Any]]:
        if not scored:
            return []
        window = max(
            top_n * self.runtime_config.stage2_candidate_multiplier,
            self.runtime_config.stage2_min_candidates,
        )
        return scored[:window]

    def _stage2_rerank(
        self,
        candidates: list[dict[str, Any]],
        *,
        query_phrases: list[str],
        terms: list[str],
        module_name: str = "",
    ) -> list[dict[str, Any]]:
        """Stage2：意图与结构特征重排。"""
        intents = self._infer_query_intents(query_phrases, terms)
        priority_terms = self._build_rerank_terms(query_phrases, terms)
        reranked: list[dict[str, Any]] = []
        for item in candidates:
            chunk = item["chunk"]
            intent_score = self._compute_intent_score(chunk, intents)
            title_section_score = self._compute_title_section_score(chunk, priority_terms)
            chunk_type_score = self._compute_chunk_type_score(chunk, intents)
            doc_boost = self._compute_doc_boost(chunk, intents, module_name=module_name)
            doc_penalty = self._compute_doc_penalty(chunk, intents)
            # 最终分数 = Stage1 召回分 + 正向特征（意图/标题章节/块类型/文档先验）- 惩罚项
            final_score = (
                item["stage1_score"] + intent_score + title_section_score + chunk_type_score + doc_boost - doc_penalty
            )
            if final_score < 0:
                final_score = 0.0
            merged = dict(item)
            merged.update(
                {
                    "intent_score": intent_score,
                    "title_section_score": title_section_score,
                    "chunk_type_score": chunk_type_score,
                    "doc_boost": doc_boost,
                    "doc_penalty": doc_penalty,
                    "final_score": final_score,
                }
            )
            reranked.append(merged)
        reranked.sort(key=lambda item: (item["final_score"], item["stage1_score"]), reverse=True)
        return reranked

    def _select_diverse(self, scored: list[dict[str, Any]], *, top_n: int, max_per_doc: int) -> list[dict[str, Any]]:
        """按文档配额裁剪结果，避免单文档刷屏。"""
        selected: list[dict[str, Any]] = []
        backup: list[dict[str, Any]] = []
        counts: dict[str, int] = defaultdict(int)
        selected_paths: set[str] = set()
        for item in scored:
            path_key = self._to_relative_path(item["chunk"].source_path)
            if counts[path_key] >= max_per_doc:
                backup.append(item)
                continue
            counts[path_key] += 1
            selected_paths.add(path_key)
            selected.append(item)
            if len(selected) >= top_n:
                return selected

        # 第二轮：优先补“新文档”而非重复文档，减少同一路径重复占位。
        deferred_duplicates: list[dict[str, Any]] = []
        for item in backup:
            path_key = self._to_relative_path(item["chunk"].source_path)
            if path_key in selected_paths:
                deferred_duplicates.append(item)
                continue
            selected_paths.add(path_key)
            selected.append(item)
            if len(selected) >= top_n:
                return selected

        # 第三轮：如果仍未达到 top_n，再放宽限制补充重复文档，保证结果数量稳定。
        for item in deferred_duplicates:
            selected.append(item)
            if len(selected) >= top_n:
                break
        return selected

    def _expand_query_phrases(self, query_phrases: list[str]) -> list[str]:
        """查询扩展：缩写归一 + 同义词扩展 + 指标词补充。"""
        expanded: list[str] = []
        seen: set[str] = set()

        def append_phrase(text: str) -> None:
            normalized = re.sub(r"\s+", " ", text.strip())
            if not normalized:
                return
            key = normalized.lower()
            if key in seen:
                return
            seen.add(key)
            expanded.append(normalized)

        merged_query = " ".join(query_phrases).lower()
        # 仅使用“原始用户问题”判断是否触发指标词扩展，
        # 避免被 query_rewriter 的附加检索语句误触发（导致过度偏向指标文档）。
        original_query = (query_phrases[0] if query_phrases else "").lower()
        for phrase in query_phrases:
            append_phrase(phrase)

        for abbr, canonical in self.ABBR_NORMALIZATION.items():
            if abbr in merged_query:
                append_phrase(canonical)

        for anchor, aliases in self.SYNONYM_MAP.items():
            trigger_terms = (anchor, *aliases)
            if any(term in original_query for term in trigger_terms):
                append_phrase(anchor)
                for alias in aliases:
                    append_phrase(alias)

        if any(token in original_query for token in ("指标", "口径", "监控", "评估")):
            for term in ("ctr", "cvr", "pctr", "pcvr", "ecpm", "cpc", "cpa", "roas"):
                append_phrase(term)
                if term in self.ABBR_NORMALIZATION:
                    append_phrase(self.ABBR_NORMALIZATION[term])
        return expanded[:28]

    def _extract_query_terms(self, query_phrases: list[str]) -> list[str]:
        merged_query = " ".join(query_phrases).lower()
        terms: set[str] = set()

        for term in self.DOMAIN_TERMS:
            if term.lower() in merged_query:
                terms.add(term.lower())
        for token in re.findall(r"[a-z][a-z0-9_+-]{1,}", merged_query):
            if token not in self.STOP_WORDS:
                terms.add(token)
        for token in re.findall(r"[\u4e00-\u9fff]{2,10}", merged_query):
            if token not in self.STOP_WORDS:
                terms.add(token)
        return sorted(terms, key=len, reverse=True)[:64]

    def _extract_fallback_ngrams(self, query_phrases: list[str]) -> list[str]:
        ngrams: set[str] = set()
        merged_query = " ".join(query_phrases)
        chinese_parts = re.findall(r"[\u4e00-\u9fff]{2,}", merged_query)
        for part in chinese_parts:
            cleaned = part.strip()
            for n in (3, 2):
                if len(cleaned) < n:
                    continue
                for start in range(0, len(cleaned) - n + 1):
                    ngram = cleaned[start : start + n]
                    if ngram in self.STOP_WORDS:
                        continue
                    ngrams.add(ngram)
        return sorted(ngrams, key=len, reverse=True)[:64]

    def _score_bm25(self, chunk_id: int, query_tf: dict[str, int]) -> float:
        tf = self._token_freq_by_chunk.get(chunk_id, {})
        doc_len = self._doc_len_by_chunk.get(chunk_id, 1)
        k1 = self.runtime_config.bm25_k1
        b = self.runtime_config.bm25_b
        score = 0.0
        for token, q_freq in query_tf.items():
            token_tf = tf.get(token, 0)
            if token_tf <= 0:
                continue
            idf = self._idf.get(token, 0.0)
            numerator = token_tf * (k1 + 1.0)
            denominator = token_tf + k1 * (1.0 - b + b * doc_len / max(self._avg_doc_len, 1.0))
            score += idf * (numerator / max(denominator, 1e-9)) * (1.0 + min(q_freq, 3) * 0.15)
        return score

    def _score_lexical(
        self,
        *,
        chunk: WikiChunk,
        query_phrases: list[str],
        terms: list[str],
        module_norm: str,
    ) -> tuple[float, list[str], list[str]]:
        score = 0.0
        matched_terms: list[str] = []
        matched_phrases: list[str] = []

        for phrase in query_phrases:
            normalized_phrase = self._normalize(phrase)
            if len(normalized_phrase) < 2:
                continue
            if normalized_phrase in chunk.normalized_text:
                score += 2.8
                matched_phrases.append(normalized_phrase)
            if normalized_phrase in chunk.normalized_title:
                score += 1.6
            if normalized_phrase in chunk.normalized_section:
                score += 1.8

        for term in terms:
            if len(term) < 2:
                continue
            count = chunk.normalized_text.count(term)
            if count <= 0:
                continue
            matched_terms.append(term)
            score += min(count, 4) * (1.3 if len(term) >= 4 else 0.8)
            if term in chunk.normalized_title:
                score += 0.9
            if term in chunk.normalized_section:
                score += 1.0

        if module_norm and module_norm in chunk.normalized_text:
            score += 1.2
        if chunk.chunk_type in {"list", "table"} and len(matched_terms) >= 2:
            score += 0.6
        return score, sorted(set(matched_terms), key=len, reverse=True), sorted(set(matched_phrases), key=len, reverse=True)

    def _infer_query_intents(self, query_phrases: list[str], terms: list[str]) -> set[str]:
        merged = self._normalize(" ".join([*query_phrases, *terms]))
        intents: set[str] = set()
        for intent, keywords in self.RERANK_INTENT_KEYWORDS.items():
            if any(keyword in merged for keyword in keywords):
                intents.add(intent)
        if not intents:
            intents.add("general")
        return intents

    def _build_rerank_terms(self, query_phrases: list[str], terms: list[str]) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for token in [*terms, *self._extract_fallback_ngrams(query_phrases)[:24]]:
            normalized = self._normalize(token)
            if len(normalized) < 2:
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            merged.append(normalized)
        return merged

    def _compute_intent_score(self, chunk: WikiChunk, intents: set[str]) -> float:
        searchable = f"{chunk.normalized_title} {chunk.normalized_section} {chunk.normalized_text}"
        score = 0.0
        for intent in intents:
            keywords = self.RERANK_INTENT_KEYWORDS.get(intent, ())
            hit = sum(1 for keyword in keywords if keyword in searchable)
            if hit <= 0:
                continue
            if intent == "metric":
                score += min(hit, 4) * 0.65
            elif intent == "pipeline":
                score += min(hit, 4) * 0.6
            elif intent == "troubleshoot":
                score += min(hit, 4) * 0.55
            elif intent == "formula":
                score += min(hit, 4) * 0.55
            elif intent == "architecture":
                score += min(hit, 4) * 0.5
        return score

    def _compute_title_section_score(self, chunk: WikiChunk, terms: list[str]) -> float:
        score = 0.0
        for term in terms:
            if len(term) < 2:
                continue
            if term in chunk.normalized_title:
                score += 0.45 if len(term) >= 3 else 0.25
            if term in chunk.normalized_section:
                score += 0.65 if len(term) >= 3 else 0.35
        return min(score, 4.0)

    def _compute_chunk_type_score(self, chunk: WikiChunk, intents: set[str]) -> float:
        score = 0.0
        if "metric" in intents and chunk.chunk_type in {"list", "table"}:
            score += 1.1
        if "pipeline" in intents and chunk.chunk_type == "flow":
            score += 1.2
        if "formula" in intents and chunk.chunk_type in {"code", "table"}:
            score += 0.9
        if "troubleshoot" in intents and chunk.chunk_type in {"list", "paragraph"}:
            score += 0.5
        if "architecture" in intents and chunk.chunk_type in {"flow", "paragraph"}:
            score += 0.45
        return score

    def _compute_doc_boost(self, chunk: WikiChunk, intents: set[str], *, module_name: str = "") -> float:
        """对高意图匹配文档做正向先验加分。"""
        boost = 0.0
        relative = self._to_relative_path(chunk.source_path).lower()
        if "troubleshoot" in intents and self._is_troubleshoot_doc(relative):
            boost += self.runtime_config.troubleshoot_doc_boost
        if "troubleshoot" in intents and self._is_module_detail_doc(relative):
            boost += 0.35
        if "pipeline" in intents and self._is_module_detail_doc(relative):
            boost += 0.45
        if "architecture" in intents and self._is_architecture_doc(relative):
            boost += 0.8
        if "metric" in intents and chunk.chunk_type in {"list", "table"} and self._is_architecture_doc(relative):
            boost += 0.4
        # 模块主文档先验：当上游已推断出明确模块时，给主文档一个轻量加分。
        if module_name:
            hints = self.MODULE_DOC_HINTS.get(module_name, ())
            if hints and any(hint.lower() in relative for hint in hints):
                boost += self.MODULE_DOC_PRIOR_BOOST
        return boost

    def _compute_doc_penalty(self, chunk: WikiChunk, intents: set[str]) -> float:
        penalty = 0.0
        relative = self._to_relative_path(chunk.source_path).lower()
        if relative.endswith("/readme.md") or relative.endswith("readme.md"):
            penalty += 1.2
        if "目录" in chunk.normalized_section:
            penalty += 0.6
        if "metric" in intents and "pipeline" not in intents and chunk.chunk_type == "flow":
            penalty += 0.8
        # 非“架构/链路”问题下，适度降低“总体架构”文档权重，避免总览文档长期占据 Top1。
        if (
            self._is_architecture_doc(relative)
            and "architecture" not in intents
            and "pipeline" not in intents
            and "metric" not in intents
        ):
            penalty += self.runtime_config.architecture_doc_penalty
        # 排障问题进一步压低总览文档，促使“联调排障手册”进入头部结果。
        if "troubleshoot" in intents and self._is_architecture_doc(relative):
            penalty += self.runtime_config.architecture_doc_penalty * 0.4
        # P1：函数定位类 query 下继续压低 00-总体架构文档，仅作用于定位场景，
        # 不影响正常“架构/流程”类问答。
        if "code_location" in intents and self._is_architecture_doc(relative):
            penalty += self.runtime_config.architecture_doc_location_penalty
        if len(chunk.content.strip()) < 20:
            penalty += 0.4
        return penalty

    def _is_architecture_doc(self, relative_path: str) -> bool:
        """判断是否为总体架构类文档（按文件编号 00-* 识别）。"""
        return bool(re.search(r"/00-[^/]+\.md$", relative_path))

    def _is_troubleshoot_doc(self, relative_path: str) -> bool:
        """判断是否为联调排障类文档（按文件编号 05-* 识别）。"""
        return bool(re.search(r"/05-[^/]+\.md$", relative_path))

    def _is_module_detail_doc(self, relative_path: str) -> bool:
        """判断是否为模块细分文档（01~04）。"""
        return bool(re.search(r"/0[1-4]-[^/]+\.md$", relative_path))

    def _build_excerpt_with_neighbors(
        self,
        *,
        chunk: WikiChunk,
        matched_terms: list[str],
        matched_phrases: list[str],
    ) -> str:
        """构建摘要片段，并在必要时拼接邻接块。"""
        parts: list[str] = []
        prev_chunk = self._chunk_by_id.get(chunk.prev_chunk_id or -1)
        next_chunk = self._chunk_by_id.get(chunk.next_chunk_id or -1)
        lead_markers = ("如下：", "如下", "包括：", "包括", "例如：", "例如", "如下所示：")

        if prev_chunk and chunk.chunk_type in {"list", "table", "flow", "code"}:
            if any(prev_chunk.content.strip().endswith(marker) for marker in lead_markers):
                parts.append(prev_chunk.content.strip())
        parts.append(chunk.content.strip())

        if next_chunk and any(chunk.content.strip().endswith(marker) for marker in lead_markers):
            if next_chunk.chunk_type in {"list", "table", "flow"}:
                parts.append(next_chunk.content.strip())

        merged = "\n".join(part for part in parts if part).strip()
        if len(merged) <= self.runtime_config.excerpt_max_chars:
            return merged

        hit_terms = [*matched_terms, *matched_phrases]
        normalized_merged = self._normalize(merged)
        for term in hit_terms:
            term_norm = self._normalize(term)
            if len(term_norm) < 2:
                continue
            index = normalized_merged.find(term_norm)
            if index < 0:
                continue
            start = max(0, index - 30)
            end = min(len(merged), start + self.runtime_config.excerpt_max_chars)
            snippet = merged[start:end].strip()
            if start > 0:
                snippet = f"...{snippet}"
            if end < len(merged):
                snippet = f"{snippet}..."
            return snippet
        return merged[: self.runtime_config.excerpt_max_chars] + "..."

    def _link_neighbors(self, chunks: list[WikiChunk]) -> None:
        for i, chunk in enumerate(chunks):
            chunk.prev_chunk_id = chunks[i - 1].chunk_id if i - 1 >= 0 else None
            chunk.next_chunk_id = chunks[i + 1].chunk_id if i + 1 < len(chunks) else None

    def _infer_line_type(self, line: str) -> str:
        stripped = line.strip()
        if not stripped:
            return "paragraph"
        if stripped.startswith("```"):
            return "code"
        if re.match(r"^(\-|\*|\+|\d+\.)\s+", stripped):
            return "list"
        if "|" in stripped and re.search(r"\|.*\|", stripped):
            return "table"
        if "->" in stripped or "=>" in stripped:
            return "flow"
        return "paragraph"

    def _build_query_vector(self, terms: list[str]) -> dict[str, float]:
        tf = self._count_tokens(" ".join(terms))
        return self._build_tfidf_vector(tf)

    def _build_tfidf_vector(self, token_freq: dict[str, int]) -> dict[str, float]:
        total = sum(token_freq.values())
        if total <= 0:
            return {}
        vector: dict[str, float] = {}
        for token, freq in token_freq.items():
            if freq <= 0:
                continue
            vector[token] = (freq / total) * self._idf.get(token, 1.0)
        return vector

    def _cosine_similarity(
        self,
        query_vector: dict[str, float],
        doc_vector: dict[str, float],
        doc_norm: float,
    ) -> float:
        if not query_vector or not doc_vector or doc_norm <= 0:
            return 0.0
        query_norm = self._vector_norm(query_vector)
        if query_norm <= 0:
            return 0.0
        dot = sum(value * doc_vector.get(token, 0.0) for token, value in query_vector.items())
        if dot <= 0:
            return 0.0
        return dot / (query_norm * doc_norm)

    def _vector_norm(self, vector: dict[str, float]) -> float:
        return sqrt(sum(value * value for value in vector.values())) if vector else 0.0

    def _count_tokens(self, text: str) -> dict[str, int]:
        freq: dict[str, int] = defaultdict(int)
        for token in self._extract_tokens(text):
            freq[token] += 1
        return freq

    def _extract_tokens(self, text: str) -> list[str]:
        normalized = self._normalize(text)
        tokens: list[str] = []
        tokens.extend(
            [
                token
                for token in re.findall(r"[a-z_][a-z0-9_]{1,}", normalized)
                if token not in self.STOP_WORDS
            ]
        )
        tokens.extend(
            [
                token
                for token in re.findall(r"[\u4e00-\u9fff]{2,10}", normalized)
                if token not in self.STOP_WORDS
            ]
        )
        return tokens

    def _to_relative_path(self, path: Path) -> str:
        try:
            return path.relative_to(self.project_root).as_posix()
        except ValueError:
            return path.as_posix()

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.lower()).strip()
