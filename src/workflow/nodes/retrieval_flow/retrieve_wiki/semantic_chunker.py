# -*- coding: utf-8 -*-
"""语义化 Markdown 分块器模块

该模块实现基于语义边界的 Markdown 文档分块策略：
1. 按标题层级分块，保持文档结构
2. 识别并保持特殊块类型完整（代码块、表格、列表）
3. 动态合并过短块，拆分过长块
4. 保留层级上下文信息

相比固定长度分块，语义分块的优势：
- 保持语义完整性，不会在句子中间切断
- 保留层级结构，便于上下文理解
- 特殊块类型（代码、表格）保持完整
- 长文档可以生成多个块，提高覆盖率
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SemanticChunk:
    """语义分块结构

    Attributes:
        chunk_id: 块唯一标识
        source_path: 源文件路径
        title: 文档标题（H1）
        section: 当前章节名称
        hierarchy: 标题层级链，如 ["广告引擎总体架构", "在线投放链路", "召回阶段"]
        content: 块内容
        chunk_type: 块类型（paragraph, code, table, list, steps, mixed）
        token_count: 近似 token 数量
        parent_chunk_id: 父块 ID（用于层级关联，可选）
        start_line: 起始行号
        end_line: 结束行号
    """
    chunk_id: int
    source_path: Path
    title: str
    section: str
    hierarchy: list[str] = field(default_factory=list)
    content: str = ""
    chunk_type: str = "paragraph"
    token_count: int = 0
    parent_chunk_id: int | None = None
    start_line: int = 1
    end_line: int = 1


@dataclass
class SemanticChunkerConfig:
    """语义分块器配置

    Attributes:
        min_chunk_chars: 最小块大小（字符数）
        max_chunk_chars: 最大块大小（字符数）
        chunk_overlap_chars: 块重叠字符数
        max_chunks_per_doc: 每文档最大块数
        preserve_code_blocks: 是否保持代码块完整
        preserve_tables: 是否保持表格完整
        include_hierarchy: 是否包含层级信息
        merge_short_sections: 是否合并短章节
    """
    min_chunk_chars: int = 200
    max_chunk_chars: int = 800
    chunk_overlap_chars: int = 50
    max_chunks_per_doc: int = 3
    preserve_code_blocks: bool = True
    preserve_tables: bool = True
    include_hierarchy: bool = True
    merge_short_sections: bool = True


class SemanticMarkdownChunker:
    """语义化 Markdown 分块器

    核心功能：
    1. 按标题层级解析 Markdown 结构
    2. 识别特殊块类型（代码、表格、列表、步骤）
    3. 动态合并/拆分块
    4. 保留层级上下文

    使用示例:
        chunker = SemanticMarkdownChunker(config)
        chunks = chunker.chunk(markdown_content, file_path)
    """

    # 标题正则
    HEADER_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

    # 特殊块模式
    CODE_BLOCK_PATTERN = re.compile(r'```[\s\S]*?```', re.MULTILINE)
    TABLE_PATTERN = re.compile(r'^\|.*\|\s*$', re.MULTILINE)
    LIST_PATTERN = re.compile(r'^(\s*[-*+]|\s*\d+\.)\s+', re.MULTILINE)
    STEPS_PATTERN = re.compile(r'^\s*(?:步骤|Step|\d+\.|[①②③④⑤⑥⑦⑧⑨⑩])', re.MULTILINE)

    def __init__(self, config: SemanticChunkerConfig | None = None) -> None:
        """初始化分块器

        Args:
            config: 分块器配置，为 None 时使用默认配置
        """
        self.config = config or SemanticChunkerConfig()

    def chunk(self, content: str, source_path: Path) -> list[SemanticChunk]:
        """对 Markdown 内容进行语义分块

        Args:
            content: Markdown 文档内容
            source_path: 源文件路径

        Returns:
            语义分块列表
        """
        if not content or not content.strip():
            return []

        # 1. 提取文档标题
        title = self._extract_title(content, source_path)

        # 2. 解析 Markdown 结构
        sections = self._parse_markdown_structure(content, title)

        if not sections:
            # 没有章节结构，按特殊块分块
            return self._chunk_by_special_blocks(content, source_path, title)

        # 3. 为每个章节创建块
        raw_chunks = self._create_chunks_from_sections(sections, source_path, title)

        # 4. 动态合并/拆分
        final_chunks = self._merge_and_split_chunks(raw_chunks)

        # 5. 限制每文档最大块数
        if len(final_chunks) > self.config.max_chunks_per_doc:
            # 按内容长度排序，保留最重要的块
            final_chunks = sorted(final_chunks, key=lambda c: len(c.content), reverse=True)
            final_chunks = final_chunks[:self.config.max_chunks_per_doc]
            # 重新按 chunk_id 排序
            final_chunks = sorted(final_chunks, key=lambda c: c.chunk_id)

        return final_chunks

    def _extract_title(self, content: str, source_path: Path) -> str:
        """提取文档标题

        优先使用 H1 标题，如果没有则使用文件名

        Args:
            content: 文档内容
            source_path: 源文件路径

        Returns:
            文档标题
        """
        match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if match:
            return match.group(1).strip()
        return source_path.stem

    def _parse_markdown_structure(self, content: str, doc_title: str) -> list[dict[str, Any]]:
        """解析 Markdown 文档结构

        将文档解析为层级结构，每个节点包含：
        - level: 标题级别（1-6）
        - title: 标题文本
        - content: 该标题下的内容
        - children: 子章节列表

        Args:
            content: 文档内容
            doc_title: 文档标题

        Returns:
            解析后的结构列表
        """
        lines = content.split('\n')
        sections: list[dict[str, Any]] = []

        # 当前层级栈
        stack: list[dict[str, Any]] = []
        current_section: dict[str, Any] = {
            "level": 0,
            "title": doc_title,
            "content": [],
            "start_line": 1,
        }

        for line_no, line in enumerate(lines, start=1):
            header_match = self.HEADER_PATTERN.match(line)

            if header_match:
                # 保存当前章节
                if current_section["content"]:
                    current_section["end_line"] = line_no - 1
                    sections.append(current_section.copy())

                # 开始新章节
                level = len(header_match.group(1))
                title_text = header_match.group(2).strip()

                current_section = {
                    "level": level,
                    "title": title_text,
                    "content": [],
                    "start_line": line_no,
                    "hierarchy": [s["title"] for s in stack] + [title_text],
                }

                # 更新层级栈
                while stack and stack[-1]["level"] >= level:
                    stack.pop()
                stack.append(current_section.copy())
            else:
                # 添加内容到当前章节
                current_section["content"].append(line)

        # 保存最后一个章节
        if current_section["content"]:
            current_section["end_line"] = len(lines)
            sections.append(current_section)

        return sections

    def _create_chunks_from_sections(
        self,
        sections: list[dict[str, Any]],
        source_path: Path,
        doc_title: str,
    ) -> list[SemanticChunk]:
        """从解析的章节结构创建分块

        Args:
            sections: 解析后的章节列表
            source_path: 源文件路径
            doc_title: 文档标题

        Returns:
            分块列表
        """
        chunks: list[SemanticChunk] = []
        chunk_id = 1

        for section in sections:
            content_lines = section.get("content", [])
            if not content_lines:
                continue

            content = '\n'.join(content_lines).strip()
            if not content:
                continue

            # 识别块类型
            chunk_type = self._identify_chunk_type(content)

            # 构建层级信息
            hierarchy = section.get("hierarchy", [doc_title, section.get("title", "")])
            if self.config.include_hierarchy:
                hierarchy = [h for h in hierarchy if h]
            else:
                hierarchy = []

            # 计算 token 数量（近似：中文按字符，英文按单词）
            token_count = self._estimate_tokens(content)

            chunk = SemanticChunk(
                chunk_id=chunk_id,
                source_path=source_path,
                title=doc_title,
                section=section.get("title", ""),
                hierarchy=hierarchy,
                content=content,
                chunk_type=chunk_type,
                token_count=token_count,
                start_line=section.get("start_line", 1),
                end_line=section.get("end_line", 1),
            )

            chunks.append(chunk)
            chunk_id += 1

        return chunks

    def _chunk_by_special_blocks(
        self,
        content: str,
        source_path: Path,
        doc_title: str,
    ) -> list[SemanticChunk]:
        """当没有章节结构时，按特殊块分块

        识别代码块、表格等特殊块，作为分块边界

        Args:
            content: 文档内容
            source_path: 源文件路径
            doc_title: 文档标题

        Returns:
            分块列表
        """
        chunks: list[SemanticChunk] = []

        # 找到所有特殊块的位置
        special_blocks = []

        # 找代码块
        for match in self.CODE_BLOCK_PATTERN.finditer(content):
            special_blocks.append((match.start(), match.end(), "code"))

        # 找表格
        for match in self.TABLE_PATTERN.finditer(content):
            special_blocks.append((match.start(), match.end(), "table"))

        # 按位置排序
        special_blocks.sort(key=lambda x: x[0])

        # 根据特殊块分割内容
        if not special_blocks:
            # 没有特殊块，按长度分割
            return self._split_by_length(content, source_path, doc_title)

        # 创建分块
        chunk_id = 1
        last_end = 0

        for start, end, block_type in special_blocks:
            # 特殊块之前的内容
            if start > last_end:
                prefix = content[last_end:start].strip()
                if prefix and len(prefix) >= self.config.min_chunk_chars:
                    chunks.append(SemanticChunk(
                        chunk_id=chunk_id,
                        source_path=source_path,
                        title=doc_title,
                        section="",
                        hierarchy=[doc_title],
                        content=prefix,
                        chunk_type="paragraph",
                        token_count=self._estimate_tokens(prefix),
                    ))
                    chunk_id += 1

            # 特殊块本身
            block_content = content[start:end].strip()
            if block_content:
                chunks.append(SemanticChunk(
                    chunk_id=chunk_id,
                    source_path=source_path,
                    title=doc_title,
                    section="",
                    hierarchy=[doc_title],
                    content=block_content,
                    chunk_type=block_type,
                    token_count=self._estimate_tokens(block_content),
                ))
                chunk_id += 1

            last_end = end

        # 最后剩余的内容
        if last_end < len(content):
            suffix = content[last_end:].strip()
            if suffix and len(suffix) >= self.config.min_chunk_chars:
                chunks.append(SemanticChunk(
                    chunk_id=chunk_id,
                    source_path=source_path,
                    title=doc_title,
                    section="",
                    hierarchy=[doc_title],
                    content=suffix,
                    chunk_type="paragraph",
                    token_count=self._estimate_tokens(suffix),
                ))

        return chunks

    def _split_by_length(
        self,
        content: str,
        source_path: Path,
        doc_title: str,
    ) -> list[SemanticChunk]:
        """按长度分割内容

        当没有结构也没有特殊块时使用

        Args:
            content: 文档内容
            source_path: 源文件路径
            doc_title: 文档标题

        Returns:
            分块列表
        """
        chunks: list[SemanticChunk] = []

        if len(content) <= self.config.max_chunk_chars:
            # 内容不长，不分割
            chunks.append(SemanticChunk(
                chunk_id=1,
                source_path=source_path,
                title=doc_title,
                section="",
                hierarchy=[doc_title],
                content=content,
                chunk_type="paragraph",
                token_count=self._estimate_tokens(content),
            ))
            return chunks

        # 按段落分割
        paragraphs = re.split(r'\n\s*\n', content)
        current_content = ""
        chunk_id = 1

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_content) + len(para) + 2 <= self.config.max_chunk_chars:
                current_content += "\n\n" + para if current_content else para
            else:
                # 保存当前块
                if current_content:
                    chunks.append(SemanticChunk(
                        chunk_id=chunk_id,
                        source_path=source_path,
                        title=doc_title,
                        section="",
                        hierarchy=[doc_title],
                        content=current_content,
                        chunk_type="paragraph",
                        token_count=self._estimate_tokens(current_content),
                    ))
                    chunk_id += 1

                # 如果单个段落就超长，直接作为一个块
                if len(para) > self.config.max_chunk_chars:
                    chunks.append(SemanticChunk(
                        chunk_id=chunk_id,
                        source_path=source_path,
                        title=doc_title,
                        section="",
                        hierarchy=[doc_title],
                        content=para[:self.config.max_chunk_chars],
                        chunk_type="paragraph",
                        token_count=self._estimate_tokens(para[:self.config.max_chunk_chars]),
                    ))
                    chunk_id += 1
                    current_content = ""
                else:
                    current_content = para

        # 保存最后的内容
        if current_content:
            chunks.append(SemanticChunk(
                chunk_id=chunk_id,
                source_path=source_path,
                title=doc_title,
                section="",
                hierarchy=[doc_title],
                content=current_content,
                chunk_type="paragraph",
                token_count=self._estimate_tokens(current_content),
            ))

        return chunks

    def _identify_chunk_type(self, content: str) -> str:
        """识别块类型

        Args:
            content: 块内容

        Returns:
            块类型：paragraph, code, table, list, steps, mixed
        """
        has_code = bool(self.CODE_BLOCK_PATTERN.search(content))
        has_table = bool(self.TABLE_PATTERN.search(content))
        has_list = bool(self.LIST_PATTERN.search(content))
        has_steps = bool(self.STEPS_PATTERN.search(content))

        # 计算类型数量
        type_count = sum([has_code, has_table, has_list, has_steps])

        if type_count == 0:
            return "paragraph"
        elif type_count == 1:
            if has_code:
                return "code"
            elif has_table:
                return "table"
            elif has_list:
                return "list"
            elif has_steps:
                return "steps"
        else:
            return "mixed"

    def _merge_and_split_chunks(self, chunks: list[SemanticChunk]) -> list[SemanticChunk]:
        """动态合并和拆分块

        规则：
        1. 短块（< min_chunk_chars）：与相邻块合并
        2. 正常块（min_chunk_chars ~ max_chunk_chars）：保持独立
        3. 超长块（> max_chunk_chars）：按段落/句子拆分

        Args:
            chunks: 原始块列表

        Returns:
            处理后的块列表
        """
        if not chunks:
            return []

        result: list[SemanticChunk] = []
        merged: list[SemanticChunk] = []

        for chunk in chunks:
            content_len = len(chunk.content)

            if content_len < self.config.min_chunk_chars:
                # 短块：尝试合并
                if merged and len(merged[-1].content) + content_len <= self.config.max_chunk_chars:
                    # 与前一个块合并
                    prev = merged[-1]
                    prev.content += "\n\n" + chunk.content
                    prev.token_count += chunk.token_count
                    prev.end_line = chunk.end_line
                    # 更新块类型
                    if prev.chunk_type != chunk.chunk_type:
                        prev.chunk_type = "mixed"
                else:
                    # 无法合并，单独保留
                    merged.append(chunk)
            elif content_len > self.config.max_chunk_chars:
                # 超长块：拆分
                split_chunks = self._split_long_chunk(chunk)
                merged.extend(split_chunks)
            else:
                # 正常块
                merged.append(chunk)

        # 重新分配 chunk_id
        for i, chunk in enumerate(merged, start=1):
            chunk.chunk_id = i
            result.append(chunk)

        return result

    def _split_long_chunk(self, chunk: SemanticChunk) -> list[SemanticChunk]:
        """拆分超长块

        优先按段落拆分，保持语义完整性

        Args:
            chunk: 待拆分的块

        Returns:
            拆分后的块列表
        """
        content = chunk.content

        # 如果是代码块或表格，不拆分
        if chunk.chunk_type in ("code", "table") and self.config.preserve_code_blocks:
            return [chunk]

        # 按段落拆分
        paragraphs = re.split(r'\n\s*\n', content)
        sub_chunks: list[SemanticChunk] = []
        current_content = ""
        sub_id = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_content) + len(para) + 2 <= self.config.max_chunk_chars:
                current_content += "\n\n" + para if current_content else para
            else:
                # 保存当前内容
                if current_content:
                    sub_chunks.append(SemanticChunk(
                        chunk_id=chunk.chunk_id * 100 + sub_id,
                        source_path=chunk.source_path,
                        title=chunk.title,
                        section=chunk.section,
                        hierarchy=chunk.hierarchy.copy(),
                        content=current_content,
                        chunk_type=chunk.chunk_type,
                        token_count=self._estimate_tokens(current_content),
                    ))
                    sub_id += 1

                current_content = para

        # 保存最后的内容
        if current_content:
            sub_chunks.append(SemanticChunk(
                chunk_id=chunk.chunk_id * 100 + sub_id,
                source_path=chunk.source_path,
                title=chunk.title,
                section=chunk.section,
                hierarchy=chunk.hierarchy.copy(),
                content=current_content,
                chunk_type=chunk.chunk_type,
                token_count=self._estimate_tokens(current_content),
            ))

        return sub_chunks if sub_chunks else [chunk]

    def _estimate_tokens(self, text: str) -> int:
        """估算文本的 token 数量

        简单估算：
        - 中文字符：约 1 token/字符
        - 英文单词：约 1 token/单词
        - 其他字符：约 0.5 token/字符

        Args:
            text: 输入文本

        Returns:
            估算的 token 数量
        """
        if not text:
            return 0

        # 统计中文字符
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))

        # 统计英文单词
        english_words = len(re.findall(r'[a-zA-Z]+', text))

        # 其他字符
        other_chars = len(text) - chinese_chars - sum(len(w) for w in re.findall(r'[a-zA-Z]+', text))

        return chinese_chars + english_words + int(other_chars * 0.5)


def create_semantic_chunker_from_config(config_dict: dict[str, Any] | None = None) -> SemanticMarkdownChunker:
    """从配置字典创建语义分块器

    Args:
        config_dict: 配置字典，来自 profile.json 的 retrieval.chunking 配置

    Returns:
        配置好的分块器实例
    """
    if not config_dict:
        return SemanticMarkdownChunker()

    config = SemanticChunkerConfig(
        min_chunk_chars=config_dict.get("min_chunk_chars", 200),
        max_chunk_chars=config_dict.get("max_chunk_chars", 800),
        chunk_overlap_chars=config_dict.get("chunk_overlap_chars", 50),
        max_chunks_per_doc=config_dict.get("max_chunks_per_doc", 3),
        preserve_code_blocks=config_dict.get("preserve_code_blocks", True),
        preserve_tables=config_dict.get("preserve_tables", True),
        include_hierarchy=config_dict.get("include_hierarchy", True),
        merge_short_sections=config_dict.get("merge_short_sections", True),
    )

    return SemanticMarkdownChunker(config)
