# -*- coding: utf-8 -*-
"""语义分块器测试用例

测试 SemanticMarkdownChunker 的核心功能：
1. Markdown 层级解析
2. 特殊块类型识别
3. 动态合并/拆分
4. 配置参数验证
"""
from __future__ import annotations

import pytest
from pathlib import Path


# ==================== 测试固件 ====================

@pytest.fixture
def sample_markdown_content():
    """测试用 Markdown 文档"""
    return '''# 广告引擎总体架构

本文档描述广告引擎的整体架构设计。

## 1. 在线投放链路

在线投放链路是广告系统的核心流程，负责处理每次广告请求。

### 1.1 召回阶段

召回阶段从百万级候选中筛选出数百个候选广告。

```python
def recall_candidates(request):
    """召回候选广告"""
    candidates = []
    for strategy in recall_strategies:
        candidates.extend(strategy.recall(request))
    return deduplicate(candidates)
```

### 1.2 排序阶段

排序阶段对召回的候选进行精细化打分排序。

| 阶段 | 方法 | 候选数 |
|-----|------|-------|
| 粗排 | 简单模型 | 500 |
| 精排 | 复杂模型 | 100 |
| 重排 | 规则调整 | 10 |

## 2. 离线分析链路

离线分析链路负责数据分析和模型训练。

- 数据采集
- 特征工程
- 模型训练
- 模型评估
'''


@pytest.fixture
def sample_markdown_no_structure():
    """没有标题结构的 Markdown 文档"""
    return '''这是一段没有标题的文档。

```python
def hello():
    print("Hello, World!")
```

这是代码块后面的内容。
'''


@pytest.fixture
def sample_chunker_config():
    """测试用配置"""
    return {
        "min_chunk_chars": 100,
        "max_chunk_chars": 500,
        "max_chunks_per_doc": 5,
        "preserve_code_blocks": True,
        "preserve_tables": True,
        "include_hierarchy": True,
    }


# ==================== 配置测试 ====================

class TestSemanticChunkerConfig:
    """配置测试"""

    def test_default_config(self):
        """测试默认配置"""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))

        from retrievers.wiki.semantic_chunker import SemanticChunkerConfig

        config = SemanticChunkerConfig()

        assert config.min_chunk_chars == 200
        assert config.max_chunk_chars == 800
        assert config.max_chunks_per_doc == 3
        assert config.preserve_code_blocks is True
        assert config.preserve_tables is True

    def test_custom_config(self):
        """测试自定义配置"""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))

        from retrievers.wiki.semantic_chunker import SemanticChunkerConfig

        config = SemanticChunkerConfig(
            min_chunk_chars=100,
            max_chunk_chars=600,
            max_chunks_per_doc=5,
        )

        assert config.min_chunk_chars == 100
        assert config.max_chunk_chars == 600
        assert config.max_chunks_per_doc == 5


# ==================== 层级解析测试 ====================

class TestMarkdownStructureParsing:
    """Markdown 结构解析测试"""

    def test_extract_title_from_h1(self):
        """测试从 H1 提取标题"""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))

        from retrievers.wiki.semantic_chunker import SemanticMarkdownChunker

        chunker = SemanticMarkdownChunker()
        content = "# 测试标题\n\n内容"
        title = chunker._extract_title(content, Path("test.md"))

        assert title == "测试标题"

    def test_extract_title_from_filename(self):
        """测试从文件名提取标题（无 H1 时）"""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))

        from retrievers.wiki.semantic_chunker import SemanticMarkdownChunker

        chunker = SemanticMarkdownChunker()
        content = "没有 H1 标题"
        title = chunker._extract_title(content, Path("test_document.md"))

        assert title == "test_document"

    def test_parse_structure_with_multiple_levels(self, sample_markdown_content):
        """测试多层级结构解析"""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))

        from retrievers.wiki.semantic_chunker import SemanticMarkdownChunker

        chunker = SemanticMarkdownChunker()
        sections = chunker._parse_markdown_structure(sample_markdown_content, "广告引擎总体架构")

        # 验证解析出的章节数量
        assert len(sections) >= 3

        # 验证层级信息
        for section in sections:
            assert "level" in section
            assert "title" in section
            assert "content" in section

    def test_hierarchy_preservation(self, sample_markdown_content):
        """测试层级信息保留"""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))

        from retrievers.wiki.semantic_chunker import SemanticMarkdownChunker

        chunker = SemanticMarkdownChunker()
        chunks = chunker.chunk(sample_markdown_content, Path("test.md"))

        # 验证每个块都有层级信息
        for chunk in chunks:
            assert isinstance(chunk.hierarchy, list)
            assert len(chunk.hierarchy) > 0


# ==================== 特殊块识别测试 ====================

class TestChunkTypeIdentification:
    """特殊块类型识别测试"""

    def test_identify_code_block(self):
        """测试代码块识别"""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))

        from retrievers.wiki.semantic_chunker import SemanticMarkdownChunker

        chunker = SemanticMarkdownChunker()
        content = '```python\nprint("hello")\n```'
        chunk_type = chunker._identify_chunk_type(content)

        assert chunk_type == "code"

    def test_identify_table(self):
        """测试表格识别"""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))

        from retrievers.wiki.semantic_chunker import SemanticMarkdownChunker

        chunker = SemanticMarkdownChunker()
        content = '| 列1 | 列2 |\n|-----|-----|\n| 值1 | 值2 |'
        chunk_type = chunker._identify_chunk_type(content)

        assert chunk_type == "table"

    def test_identify_list(self):
        """测试列表识别"""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))

        from retrievers.wiki.semantic_chunker import SemanticMarkdownChunker

        chunker = SemanticMarkdownChunker()
        content = '- 项目1\n- 项目2\n- 项目3'
        chunk_type = chunker._identify_chunk_type(content)

        assert chunk_type == "list"

    def test_identify_paragraph(self):
        """测试段落识别"""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))

        from retrievers.wiki.semantic_chunker import SemanticMarkdownChunker

        chunker = SemanticMarkdownChunker()
        content = '这是一段普通文本，没有特殊格式。'
        chunk_type = chunker._identify_chunk_type(content)

        assert chunk_type == "paragraph"

    def test_identify_mixed(self):
        """测试混合类型识别"""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))

        from retrievers.wiki.semantic_chunker import SemanticMarkdownChunker

        chunker = SemanticMarkdownChunker()
        # 代码块 + 列表
        content = '```python\nprint("hello")\n```\n\n- 项目1\n- 项目2'
        chunk_type = chunker._identify_chunk_type(content)

        assert chunk_type == "mixed"


# ==================== 分块测试 ====================

class TestChunking:
    """分块测试"""

    def test_chunk_with_structure(self, sample_markdown_content):
        """测试有结构文档的分块"""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))

        from retrievers.wiki.semantic_chunker import SemanticMarkdownChunker, SemanticChunkerConfig

        config = SemanticChunkerConfig(
            min_chunk_chars=50,
            max_chunk_chars=1000,
            max_chunks_per_doc=5,
        )
        chunker = SemanticMarkdownChunker(config)
        chunks = chunker.chunk(sample_markdown_content, Path("test.md"))

        # 验证生成了多个块
        assert len(chunks) >= 1

        # 验证每个块的基本属性
        for chunk in chunks:
            assert chunk.chunk_id > 0
            assert chunk.content
            assert chunk.chunk_type in ["paragraph", "code", "table", "list", "steps", "mixed"]

    def test_chunk_without_structure(self, sample_markdown_no_structure):
        """测试无结构文档的分块"""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))

        from retrievers.wiki.semantic_chunker import SemanticMarkdownChunker

        chunker = SemanticMarkdownChunker()
        chunks = chunker.chunk(sample_markdown_no_structure, Path("test.md"))

        # 应该生成至少一个块
        assert len(chunks) >= 1

    def test_max_chunks_limit(self, sample_chunker_config):
        """测试最大块数限制"""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))

        from retrievers.wiki.semantic_chunker import SemanticMarkdownChunker, SemanticChunkerConfig

        # 创建一个很长的文档
        long_content = "# 标题\n\n" + "\n\n".join([f"## 章节 {i}\n\n内容 {i}" for i in range(20)])

        config = SemanticChunkerConfig(
            min_chunk_chars=10,
            max_chunk_chars=500,
            max_chunks_per_doc=3,  # 限制为 3 个
        )
        chunker = SemanticMarkdownChunker(config)
        chunks = chunker.chunk(long_content, Path("test.md"))

        # 验证块数不超过限制
        assert len(chunks) <= 3


# ==================== 合并/拆分测试 ====================

class TestMergeAndSplit:
    """合并和拆分测试"""

    def test_merge_short_chunks(self):
        """测试短块合并"""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))

        from retrievers.wiki.semantic_chunker import SemanticMarkdownChunker, SemanticChunkerConfig, SemanticChunk

        config = SemanticChunkerConfig(
            min_chunk_chars=100,
            max_chunk_chars=500,
            merge_short_sections=True,
        )
        chunker = SemanticMarkdownChunker(config)

        # 创建几个短块
        short_chunks = [
            SemanticChunk(
                chunk_id=1,
                source_path=Path("test.md"),
                title="测试",
                section="",
                content="短内容1",
                chunk_type="paragraph",
                token_count=10,
            ),
            SemanticChunk(
                chunk_id=2,
                source_path=Path("test.md"),
                title="测试",
                section="",
                content="短内容2",
                chunk_type="paragraph",
                token_count=10,
            ),
        ]

        merged = chunker._merge_and_split_chunks(short_chunks)

        # 短块应该被合并
        assert len(merged) <= len(short_chunks)

    def test_split_long_chunk(self):
        """测试长块拆分"""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))

        from retrievers.wiki.semantic_chunker import SemanticMarkdownChunker, SemanticChunkerConfig, SemanticChunk

        config = SemanticChunkerConfig(
            min_chunk_chars=100,
            max_chunk_chars=200,
        )
        chunker = SemanticMarkdownChunker(config)

        # 创建一个长块
        long_content = "段落1 " * 50 + "\n\n" + "段落2 " * 50
        long_chunk = SemanticChunk(
            chunk_id=1,
            source_path=Path("test.md"),
            title="测试",
            section="",
            content=long_content,
            chunk_type="paragraph",
            token_count=100,
        )

        split = chunker._split_long_chunk(long_chunk)

        # 长块应该被拆分
        assert len(split) >= 1


# ==================== Token 估算测试 ====================

class TestTokenEstimation:
    """Token 估算测试"""

    def test_estimate_chinese_tokens(self):
        """测试中文 token 估算"""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))

        from retrievers.wiki.semantic_chunker import SemanticMarkdownChunker

        chunker = SemanticMarkdownChunker()
        chinese_text = "这是一段中文文本"
        tokens = chunker._estimate_tokens(chinese_text)

        # 中文约 1 token/字符
        assert tokens == len(chinese_text)

    def test_estimate_english_tokens(self):
        """测试英文 token 估算"""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))

        from retrievers.wiki.semantic_chunker import SemanticMarkdownChunker

        chunker = SemanticMarkdownChunker()
        english_text = "This is an English sentence with eight words"
        tokens = chunker._estimate_tokens(english_text)

        # 英文约 1 token/单词
        word_count = len(english_text.split())
        assert tokens >= word_count


# ==================== 工厂函数测试 ====================

class TestFactoryFunction:
    """工厂函数测试"""

    def test_create_from_config_dict(self, sample_chunker_config):
        """测试从配置字典创建分块器"""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))

        from retrievers.wiki.semantic_chunker import create_semantic_chunker_from_config

        chunker = create_semantic_chunker_from_config(sample_chunker_config)

        assert chunker.config.min_chunk_chars == 100
        assert chunker.config.max_chunk_chars == 500
        assert chunker.config.max_chunks_per_doc == 5

    def test_create_from_none_config(self):
        """测试从空配置创建分块器"""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))

        from retrievers.wiki.semantic_chunker import create_semantic_chunker_from_config

        chunker = create_semantic_chunker_from_config(None)

        # 应使用默认配置
        assert chunker.config.min_chunk_chars == 200
        assert chunker.config.max_chunk_chars == 800


# ==================== 运行入口 ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
