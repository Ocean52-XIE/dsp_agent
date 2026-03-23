# -*- coding: utf-8 -*-
"""模型缓存模块

提供 Embedding 和 Reranker 模型的单例缓存，避免重复加载。

设计原则：
1. 同一个模型只加载一次，多个检索器共享
2. 按模型名称缓存，支持不同配置使用不同模型
3. 线程安全，支持并发初始化
4. 提供缓存统计和清理能力
5. 支持本地模型路径和离线加载

内存优化：
- Embedding 模型 (bge-base-zh-v1.5): ~400MB
- Reranker 模型 (bge-reranker-base): ~280MB
- 单例共享后可节省 ~680MB 内存

本地模型支持：
- 通过 cache_dir 参数指定本地模型目录
- 支持 Hub ID (如 BAAI/bge-base-zh-v1.5) 或本地路径
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 全局模型缓存
_embedding_cache: dict[str, Any] = {}
_reranker_cache: dict[str, Any] = {}
_cache_lock = threading.Lock()

# 缓存统计
_cache_stats = {
    "embedding_hits": 0,
    "embedding_misses": 0,
    "reranker_hits": 0,
    "reranker_misses": 0,
}


def _resolve_model_path(model_name: str, cache_dir: str | None = None) -> str:
    """解析模型路径，支持本地路径和 Hub ID

    参数:
        model_name: 模型名称（Hub ID 或本地路径）
        cache_dir: 模型缓存目录

    返回:
        解析后的模型路径
    """
    # 如果 model_name 已经是绝对路径，直接使用
    if Path(model_name).is_absolute() and Path(model_name).exists():
        logger.debug(f"[ModelCache] 使用绝对路径模型: {model_name}")
        return model_name

    # 如果指定了 cache_dir，检查本地是否存在模型
    if cache_dir:
        cache_path = Path(cache_dir)

        # 处理相对路径
        if not cache_path.is_absolute():
            # 相对于当前工作目录
            cache_path = Path.cwd() / cache_path

        # 尝试多种可能的模型目录结构
        # 1. cache_dir/model_name (直接使用 Hub ID 作为目录名)
        # 2. cache_dir/models--org--model-name (HuggingFace 缓存格式)
        # 3. cache_dir/model_name/snapshots/xxx (HuggingFace 缓存格式)

        # 方式1: 直接使用模型名
        direct_path = cache_path / model_name
        if direct_path.exists():
            logger.info(f"[ModelCache] 使用本地模型: {direct_path}")
            return str(direct_path)

        # 方式2: HuggingFace 缓存格式 (models--org--model)
        if "/" in model_name:
            hf_cache_name = "models--" + model_name.replace("/", "--")
            hf_cache_path = cache_path / hf_cache_name
            if hf_cache_path.exists():
                # 查找 snapshots 目录下的最新版本
                snapshots_path = hf_cache_path / "snapshots"
                if snapshots_path.exists():
                    snapshots = sorted(
                        [d for d in snapshots_path.iterdir() if d.is_dir()],
                        key=lambda x: x.stat().st_mtime,
                        reverse=True,
                    )
                    if snapshots:
                        logger.info(f"[ModelCache] 使用 HuggingFace 缓存模型: {snapshots[0]}")
                        return str(snapshots[0])

        # 方式3: 只使用模型名的最后一部分（去掉组织名）
        if "/" in model_name:
            model_basename = model_name.split("/")[-1]
            basename_path = cache_path / model_basename
            if basename_path.exists():
                logger.info(f"[ModelCache] 使用本地模型 (basename): {basename_path}")
                return str(basename_path)

    # 未找到本地模型，返回原始名称（将从 Hub 下载）
    if cache_dir:
        logger.warning(
            f"[ModelCache] 本地模型未找到: model={model_name}, cache_dir={cache_dir}, "
            "将尝试从 HuggingFace Hub 下载"
        )
    return model_name


def get_embedding_model(
    model_name: str,
    device: str = "cpu",
    encode_kwargs: dict | None = None,
    cache_dir: str | None = None,
) -> Any:
    """获取 Embedding 模型实例（单例缓存）

    相同模型名称和配置返回同一个实例，避免重复加载。
    支持从本地目录加载模型，实现离线部署。

    Args:
        model_name: 模型名称（Hub ID 如 "BAAI/bge-base-zh-v1.5" 或本地路径）
        device: 运行设备，cpu 或 cuda
        encode_kwargs: 编码参数
        cache_dir: 模型缓存目录，用于离线加载
            - 为 None 时使用 HuggingFace 默认缓存
            - 指定路径时，优先从该目录查找并加载模型

    Returns:
        HuggingFaceEmbeddings 实例
    """
    # 构建缓存键（包含 cache_dir 以区分不同来源）
    cache_key = f"{model_name}|{device}|{cache_dir or 'default'}"

    with _cache_lock:
        if cache_key in _embedding_cache:
            _cache_stats["embedding_hits"] += 1
            logger.debug(
                f"[ModelCache] Embedding 缓存命中: {model_name} (device={device})"
            )
            return _embedding_cache[cache_key]

        _cache_stats["embedding_misses"] += 1

        # 解析模型路径（支持本地路径）
        resolved_model_path = _resolve_model_path(model_name, cache_dir)

        logger.info(
            f"[ModelCache] Embedding 模型加载: {model_name} -> {resolved_model_path} "
            f"(device={device})"
        )

        from langchain_huggingface import HuggingFaceEmbeddings

        # 构建模型参数
        model_kwargs: dict[str, Any] = {"device": device}

        # 准备 HuggingFaceEmbeddings 的 cache_folder 参数
        # 注意：cache_folder 必须作为顶层参数传递，不能放在 model_kwargs 中
        hf_cache_folder: str | None = None
        if cache_dir:
            cache_path = Path(cache_dir)
            if not cache_path.is_absolute():
                cache_path = Path.cwd() / cache_path
            hf_cache_folder = str(cache_path)

        model = HuggingFaceEmbeddings(
            model_name=resolved_model_path,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs or {"normalize_embeddings": True},
            cache_folder=hf_cache_folder,
        )

        _embedding_cache[cache_key] = model
        logger.info(
            f"[ModelCache] Embedding 模型已缓存: {model_name} "
            f"(当前缓存数: {len(_embedding_cache)})"
        )
        return model


def get_reranker_model(
    model_name: str,
    device: str = "cpu",
    max_length: int = 512,
    cache_dir: str | None = None,
) -> Any:
    """获取 Reranker 模型实例（单例缓存）

    相同模型名称和配置返回同一个实例，避免重复加载。
    支持从本地目录加载模型，实现离线部署。

    Args:
        model_name: 模型名称（Hub ID 如 "BAAI/bge-reranker-base" 或本地路径）
        device: 运行设备，cpu 或 cuda
        max_length: 最大序列长度
        cache_dir: 模型缓存目录，用于离线加载
            - 为 None 时使用 HuggingFace 默认缓存
            - 指定路径时，优先从该目录查找并加载模型

    Returns:
        CrossEncoder 实例
    """
    # 构建缓存键（包含 cache_dir 以区分不同来源）
    cache_key = f"{model_name}|{device}|{max_length}|{cache_dir or 'default'}"

    with _cache_lock:
        if cache_key in _reranker_cache:
            _cache_stats["reranker_hits"] += 1
            logger.debug(
                f"[ModelCache] Reranker 缓存命中: {model_name} (device={device})"
            )
            return _reranker_cache[cache_key]

        _cache_stats["reranker_misses"] += 1

        # 解析模型路径（支持本地路径）
        resolved_model_path = _resolve_model_path(model_name, cache_dir)

        logger.info(
            f"[ModelCache] Reranker 模型加载: {model_name} -> {resolved_model_path} "
            f"(device={device})"
        )

        from sentence_transformers import CrossEncoder

        # CrossEncoder 支持直接传入本地路径
        # 如果指定了 cache_dir 且需要从 Hub 下载，设置环境变量
        if cache_dir and resolved_model_path == model_name:
            cache_path = Path(cache_dir)
            if not cache_path.is_absolute():
                cache_path = Path.cwd() / cache_path
            # sentence-transformers 使用 SENTENCE_TRANSFORMERS_HOME 环境变量
            # 但在运行时设置可能不生效，所以主要通过 _resolve_model_path 处理
            os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(cache_path))

        model = CrossEncoder(
            resolved_model_path,
            max_length=max_length,
            device=device,
        )

        _reranker_cache[cache_key] = model
        logger.info(
            f"[ModelCache] Reranker 模型已缓存: {model_name} "
            f"(当前缓存数: {len(_reranker_cache)})"
        )
        return model


def get_cache_stats() -> dict[str, Any]:
    """获取缓存统计信息

    Returns:
        包含缓存命中/未命中次数和当前缓存大小的字典
    """
    with _cache_lock:
        return {
            "embedding": {
                "cache_size": len(_embedding_cache),
                "hits": _cache_stats["embedding_hits"],
                "misses": _cache_stats["embedding_misses"],
                "cached_models": list(_embedding_cache.keys()),
            },
            "reranker": {
                "cache_size": len(_reranker_cache),
                "hits": _cache_stats["reranker_hits"],
                "misses": _cache_stats["reranker_misses"],
                "cached_models": list(_reranker_cache.keys()),
            },
        }


def clear_cache() -> None:
    """清空所有模型缓存并重置统计

    警告：调用此方法后，所有已缓存的模型将被释放，
    后续调用会重新加载模型。
    """
    global _embedding_cache, _reranker_cache, _cache_stats

    with _cache_lock:
        logger.info(
            f"[ModelCache] 清空缓存: "
            f"embedding={len(_embedding_cache)}, reranker={len(_reranker_cache)}"
        )
        _embedding_cache.clear()
        _reranker_cache.clear()
        # 重置统计
        _cache_stats = {
            "embedding_hits": 0,
            "embedding_misses": 0,
            "reranker_hits": 0,
            "reranker_misses": 0,
        }


def warmup_embedding_model(
    model_name: str,
    device: str = "cpu",
    cache_dir: str | None = None,
) -> None:
    """预热 Embedding 模型

    提前加载模型到内存，避免首次检索时的延迟。

    Args:
        model_name: 模型名称
        device: 运行设备
        cache_dir: 模型缓存目录
    """
    logger.info(f"[ModelCache] 预热 Embedding 模型: {model_name}")
    model = get_embedding_model(model_name, device, cache_dir=cache_dir)
    # 执行一次简单的编码来预热模型
    _ = model.embed_query("warmup")
    logger.info(f"[ModelCache] Embedding 模型预热完成: {model_name}")


def warmup_reranker_model(
    model_name: str,
    device: str = "cpu",
    cache_dir: str | None = None,
) -> None:
    """预热 Reranker 模型

    提前加载模型到内存，避免首次检索时的延迟。

    Args:
        model_name: 模型名称
        device: 运行设备
        cache_dir: 模型缓存目录
    """
    logger.info(f"[ModelCache] 预热 Reranker 模型: {model_name}")
    model = get_reranker_model(model_name, device, cache_dir=cache_dir)
    # 执行一次简单的重排来预热模型
    _ = model.predict([("warmup query", "warmup document")])
    logger.info(f"[ModelCache] Reranker 模型预热完成: {model_name}")
