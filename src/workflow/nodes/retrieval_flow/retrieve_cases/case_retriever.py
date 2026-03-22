# -*- coding: utf-8 -*-
"""案例检索器

提供案例检索功能（当前为可扩展占位实现）。

设计原则：
1. 支持从指定目录加载案例数据
2. 支持关键词和语义检索
3. 全局单例模式，避免重复初始化
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class CaseRetriever:
    """案例检索器

    从指定目录加载案例数据并提供检索功能。
    当前为占位实现，可扩展支持向量检索、BM25 等。
    """

    def __init__(
        self,
        case_dir: Path,
        project_root: Path,
        default_top_k: int = 2,
    ):
        """初始化案例检索器

        Args:
            case_dir: 案例目录路径
            project_root: 项目根目录
            default_top_k: 默认返回数量
        """
        self.case_dir = case_dir
        self.project_root = project_root
        self.default_top_k = default_top_k

        # 加载案例数据
        self._cases: list[dict[str, Any]] = []
        self._load_cases()

        logger.info(
            f"[CaseRetriever] 初始化完成: "
            f"case_dir={case_dir}, cases={len(self._cases)}"
        )

    def _load_cases(self) -> None:
        """加载案例数据"""
        if not self.case_dir.exists():
            logger.warning(f"[CaseRetriever] 案例目录不存在: {self.case_dir}")
            return

        # 加载所有 JSON 文件
        for json_file in self.case_dir.glob("**/*.json"):
            try:
                import json
                with open(json_file, "r", encoding="utf-8") as f:
                    case_data = json.load(f)
                    case_data["_source_file"] = str(json_file)
                    self._cases.append(case_data)
            except Exception as e:
                logger.warning(f"[CaseRetriever] 加载案例失败: {json_file}, error={e}")

    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """搜索案例

        Args:
            query: 查询字符串
            top_k: 返回数量，默认使用 default_top_k

        Returns:
            案例列表，每个案例包含 title、content、score 等字段
        """
        if not self._cases:
            return []

        top_k = top_k or self.default_top_k

        # 简单的关键词匹配（占位实现）
        results: list[dict[str, Any]] = []
        query_lower = query.lower()

        for case in self._cases:
            # 计算匹配分数
            title = str(case.get("title", "")).lower()
            content = str(case.get("content", "")).lower()
            keywords = case.get("keywords", [])

            score = 0.0
            if query_lower in title:
                score += 3.0
            if query_lower in content:
                score += 1.0
            if any(query_lower in str(kw).lower() for kw in keywords):
                score += 2.0

            if score > 0:
                results.append({
                    **case,
                    "score": score,
                })

        # 按分数排序并返回 top_k
        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return results[:top_k]

    @property
    def case_count(self) -> int:
        """案例数量"""
        return len(self._cases)


# ============================================================================
# 全局单例模式
# ============================================================================

# 全局案例检索器实例（单例）
_case_retriever_instance: CaseRetriever | None = None


def get_case_retriever() -> CaseRetriever | None:
    """获取全局案例检索器实例

    Returns:
        CaseRetriever 实例，如果未初始化则返回 None
    """
    return _case_retriever_instance


def set_case_retriever(retriever: CaseRetriever | None) -> None:
    """设置全局案例检索器实例

    Args:
        retriever: CaseRetriever 实例
    """
    global _case_retriever_instance
    _case_retriever_instance = retriever
    if retriever:
        logger.info("[CaseRetriever] 全局单例已设置")
    else:
        logger.info("[CaseRetriever] 全局单例已清除")
