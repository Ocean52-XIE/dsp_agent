# -*- coding: utf-8 -*-
"""检索器初始化器

负责初始化检索系统：
- Wiki 检索器：Markdown 文档检索
- Code 检索器：代码检索
- Case 检索器：案例检索（可选）

设计原则：
1. 在程序启动时完成索引加载/构建
2. 支持向量索引和 BM25 索引
3. 避免首次请求时的延迟初始化
4. 初始化后设置全局单例
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ============================================================================
# Wiki 检索器初始化
# ============================================================================

def init_wiki_retriever(
    domain_profile: Any,
    project_root: Path,
) -> Any | None:
    """初始化 Wiki 检索器

    加载 Markdown 文档并构建/加载索引，并设置全局单例。

    Args:
        domain_profile: 私域配置
        project_root: 项目根目录

    Returns:
        MarkdownWikiRetriever 实例
    """
    try:
        from workflow.nodes.retrieval_flow.retrieve_wiki.wiki_retriever import (
            MarkdownWikiRetriever,
            set_wiki_retriever,
        )

        wiki_dir = domain_profile.resolve_wiki_dir(project_root)

        retriever = MarkdownWikiRetriever(
            wiki_dir=wiki_dir,
            project_root=project_root,
            default_top_k=4,
            module_doc_hints=domain_profile.module_doc_hints(),
            embedding_profile=domain_profile.retrieval.embedding,
            hybrid_weights_profile=domain_profile.retrieval.hybrid_weights,
            reranker_profile=domain_profile.retrieval.reranker,
        )

        # 设置全局单例
        set_wiki_retriever(retriever)

        logger.info(
            f"[RetrieverInit] Wiki 检索器初始化完成: "
            f"wiki_dir={wiki_dir}"
        )
        return retriever

    except Exception as e:
        logger.warning(f"[RetrieverInit] Wiki 检索器初始化失败: {e}")
        return None


# ============================================================================
# Code 检索器初始化
# ============================================================================

def init_code_retriever(
    domain_profile: Any,
    project_root: Path,
) -> Any | None:
    """初始化代码检索器

    加载代码文件并构建/加载索引，并设置全局单例。

    Args:
        domain_profile: 私域配置
        project_root: 项目根目录

    Returns:
        LocalCodeRetriever 实例
    """
    import os

    try:
        from workflow.nodes.retrieval_flow.retrieve_code.code_retriever import (
            LocalCodeRetriever,
            parse_code_dirs_from_env,
            set_code_retriever,
        )

        # 优先从环境变量读取代码目录
        env_code_dirs = os.getenv("WORKFLOW_CODE_RETRIEVER_DIRS", "").strip()
        if env_code_dirs:
            code_dirs = parse_code_dirs_from_env(project_root=project_root)
        else:
            code_dirs = domain_profile.resolve_code_roots(project_root)

        retriever = LocalCodeRetriever(
            project_root=project_root,
            code_dirs=code_dirs,
            default_top_k=4,
            embedding_profile=domain_profile.retrieval.embedding,
            reranker_profile=domain_profile.retrieval.reranker,
        )

        # 设置全局单例
        set_code_retriever(retriever)

        logger.info(
            f"[RetrieverInit] Code 检索器初始化完成: "
            f"code_dirs={[str(d) for d in code_dirs]}"
        )
        return retriever

    except Exception as e:
        logger.warning(f"[RetrieverInit] Code 检索器初始化失败: {e}")
        return None


# ============================================================================
# Case 检索器初始化（可选）
# ============================================================================

def init_case_retriever(
    domain_profile: Any,
    project_root: Path,
) -> Any | None:
    """初始化案例检索器（可选）

    加载案例数据并构建索引，并设置全局单例。

    Args:
        domain_profile: 私域配置
        project_root: 项目根目录

    Returns:
        案例检索器实例（如果有）
    """
    # 检查是否启用案例检索
    if not domain_profile.retrieval.enable_cases:
        logger.info("[RetrieverInit] 案例检索未启用")
        return None

    try:
        # 尝试导入案例检索器
        from workflow.nodes.retrieval_flow.retrieve_cases import (
            CaseRetriever,
            set_case_retriever,
        )

        case_dir = domain_profile.resolve_eval_path("cases_dir", project_root)
        if not case_dir or not case_dir.exists():
            logger.info("[RetrieverInit] 案例目录不存在，跳过案例检索器初始化")
            return None

        retriever = CaseRetriever(
            case_dir=case_dir,
            project_root=project_root,
            default_top_k=2,
        )

        # 设置全局单例
        set_case_retriever(retriever)

        logger.info(f"[RetrieverInit] Case 检索器初始化完成: case_dir={case_dir}")
        return retriever

    except ImportError:
        logger.info("[RetrieverInit] Case 检索器模块不存在，跳过")
        return None
    except Exception as e:
        logger.warning(f"[RetrieverInit] Case 检索器初始化失败: {e}")
        return None
