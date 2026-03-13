# -*- coding: utf-8 -*-
"""
该模块实现评测流程，负责样本执行、指标统计与结果输出。
"""
from __future__ import annotations

"""代码检索评测脚本（retrieve_code 第一版本）。

评测目标：
1. 路径级召回能力：Recall@K、MRR、Top1 命中率；
2. 符号级命中能力：symbol_hit_rate@K；
3. 关键片段命中能力：pattern_hit_rate@1、pattern_hit_rate@K；
4. 高亮可解释性：highlight_hit_rate@K（高亮行是否覆盖期望 pattern）。

说明：
- 本脚本直接评估 `LocalCodeRetriever`，不依赖完整 LangGraph 链路；
- 默认使用仓库根目录 `codes` 作为稳定评测语料；
- 你也可以把 `code_roots` 指向真实业务代码目录做离线回归。
"""

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
import sys
from time import perf_counter
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from workflow.nodes.retrieval_flow.retrieve_code.code_retriever import LocalCodeRetriever  # noqa: E402


@dataclass
class CodeEvalCase:
    """
    定义`CodeEvalCase`，用于封装相关数据结构与处理行为。
    """

    case_id: str
    query: str
    retrieval_queries: list[str]
    gold_paths: list[str]
    gold_symbols: list[str]
    expected_patterns: list[str]
    intent: str
    note: str


def _load_json(path: Path) -> dict[str, Any]:
    """
    内部辅助函数，负责`load json` 相关处理。
    
    参数:
        path: 文件或目录路径。
    
    返回:
        返回类型为 `dict[str, Any]` 的处理结果。
    """
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _display_path(path: Path) -> str:
    """
    内部辅助函数，负责`display path` 相关处理。
    
    参数:
        path: 文件或目录路径。
    
    返回:
        返回类型为 `str` 的处理结果。
    """
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _normalize_text(text: str) -> str:
    """
    内部辅助函数，负责`normalize text` 相关处理。
    
    参数:
        text: 待处理的文本内容。
    
    返回:
        返回类型为 `str` 的处理结果。
    """
    lowered = text.lower()
    no_spaces = re.sub(r"\s+", "", lowered)
    return re.sub(r"[`~!@#$%^&*()\-_=+\[\]{}\\|;:'\",.<>/?，。！？；：、“”‘’（）【】《》]", "", no_spaces)


def _contains_text(haystack: str, needle: str) -> bool:
    """
    内部辅助函数，负责`contains text` 相关处理。
    
    参数:
        haystack: 输入参数，用于控制当前处理逻辑。
        needle: 输入参数，用于控制当前处理逻辑。
    
    返回:
        返回类型为 `bool` 的处理结果。
    """
    normalized_haystack = _normalize_text(haystack)
    normalized_needle = _normalize_text(needle)
    if not normalized_needle:
        return False
    return normalized_needle in normalized_haystack


def _load_dataset(path: Path) -> list[CodeEvalCase]:
    """
    内部辅助函数，负责`load dataset` 相关处理。
    
    参数:
        path: 文件或目录路径。
    
    返回:
        返回类型为 `list[CodeEvalCase]` 的处理结果。
    """
    items: list[CodeEvalCase] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            query = str(record.get("query", "")).strip()
            if not query:
                raise ValueError(f"dataset line {line_number} missing query")
            gold_paths = [str(item).strip() for item in record.get("gold_paths", []) if str(item).strip()]
            if not gold_paths:
                raise ValueError(f"dataset line {line_number} missing gold_paths")
            items.append(
                CodeEvalCase(
                    case_id=str(record.get("id", f"case_{line_number:04d}")),
                    query=query,
                    retrieval_queries=[str(item).strip() for item in record.get("retrieval_queries", []) if str(item).strip()],
                    gold_paths=gold_paths,
                    gold_symbols=[str(item).strip() for item in record.get("gold_symbols", []) if str(item).strip()],
                    expected_patterns=[
                        str(item).strip()
                        for item in record.get("expected_patterns", [])
                        if str(item).strip()
                    ],
                    intent=str(record.get("intent", "issue_analysis")).strip() or "issue_analysis",
                    note=str(record.get("note", "")).strip(),
                )
            )
    return items


def _compute_recall_at_k(retrieved_paths: list[str], gold_paths: set[str], k: int) -> float:
    """
    内部辅助函数，负责`compute recall at k` 相关处理。
    
    参数:
        retrieved_paths: 列表参数，用于承载批量输入数据。
        gold_paths: 列表参数，用于承载批量输入数据。
        k: 输入参数，用于控制当前处理逻辑。
    
    返回:
        返回类型为 `float` 的处理结果。
    """
    top_k_paths = retrieved_paths[:k]
    return 1.0 if any(path in gold_paths for path in top_k_paths) else 0.0


def _reciprocal_rank(retrieved_paths: list[str], gold_paths: set[str]) -> float:
    """
    内部辅助函数，负责`reciprocal rank` 相关处理。
    
    参数:
        retrieved_paths: 列表参数，用于承载批量输入数据。
        gold_paths: 列表参数，用于承载批量输入数据。
    
    返回:
        返回类型为 `float` 的处理结果。
    """
    for index, path in enumerate(retrieved_paths, start=1):
        if path in gold_paths:
            return 1.0 / index
    return 0.0


def _compute_symbol_hit_at_k(hits: list[dict[str, Any]], gold_symbols: list[str], k: int) -> float:
    """
    内部辅助函数，负责`compute symbol hit at k` 相关处理。
    
    参数:
        hits: 列表参数，用于承载批量输入数据。
        gold_symbols: 列表参数，用于承载批量输入数据。
        k: 输入参数，用于控制当前处理逻辑。
    
    返回:
        返回类型为 `float` 的处理结果。
    """
    if not gold_symbols:
        return 1.0
    top_hits = hits[:k]
    for hit in top_hits:
        symbol_name = str(hit.get("symbol_name", ""))
        signature = str(hit.get("signature", ""))
        search_space = f"{symbol_name} {signature}"
        if any(_contains_text(search_space, symbol) for symbol in gold_symbols):
            return 1.0
    return 0.0


def _compute_pattern_hit_at_k(hits: list[dict[str, Any]], expected_patterns: list[str], k: int) -> float:
    """
    内部辅助函数，负责`compute pattern hit at k` 相关处理。
    
    参数:
        hits: 列表参数，用于承载批量输入数据。
        expected_patterns: 列表参数，用于承载批量输入数据。
        k: 输入参数，用于控制当前处理逻辑。
    
    返回:
        返回类型为 `float` 的处理结果。
    """
    if not expected_patterns:
        return 1.0
    top_hits = hits[:k]
    for hit in top_hits:
        text = " ".join(
            [
                str(hit.get("title", "")),
                str(hit.get("symbol_name", "")),
                str(hit.get("signature", "")),
                str(hit.get("excerpt", "")),
            ]
        )
        if any(_contains_text(text, pattern) for pattern in expected_patterns):
            return 1.0
    return 0.0


def _compute_highlight_hit_at_k(hits: list[dict[str, Any]], expected_patterns: list[str], k: int) -> float:
    """
    内部辅助函数，负责`compute highlight hit at k` 相关处理。
    
    参数:
        hits: 列表参数，用于承载批量输入数据。
        expected_patterns: 列表参数，用于承载批量输入数据。
        k: 输入参数，用于控制当前处理逻辑。
    
    返回:
        返回类型为 `float` 的处理结果。
    """
    if not expected_patterns:
        return 1.0

    for hit in hits[:k]:
        excerpt_lines = hit.get("excerpt_lines", [])
        if isinstance(excerpt_lines, list) and excerpt_lines:
            for line_item in excerpt_lines:
                if not isinstance(line_item, dict):
                    continue
                if not bool(line_item.get("is_hit", False)):
                    continue
                line_text = str(line_item.get("content", ""))
                if any(_contains_text(line_text, pattern) for pattern in expected_patterns):
                    return 1.0
            continue

        excerpt_text = str(hit.get("excerpt", ""))
        for line in excerpt_text.splitlines():
            if not line.strip().startswith(">>"):
                continue
            if any(_contains_text(line, pattern) for pattern in expected_patterns):
                return 1.0
    return 0.0


def run_eval(config_path: Path) -> dict[str, Any]:
    """
    执行对应子流程并返回执行结果。
    
    参数:
        config_path: 路径参数，用于定位文件或目录。
    
    返回:
        返回类型为 `dict[str, Any]` 的处理结果。
    """
    config = _load_json(config_path)
    dataset_path = (PROJECT_ROOT / config["dataset_path"]).resolve()
    output_path = (PROJECT_ROOT / config["output_path"]).resolve()
    code_root_paths = [
        (PROJECT_ROOT / path).resolve() if not Path(path).is_absolute() else Path(path)
        for path in config.get("code_roots", ["codes"])
    ]
    top_ks = sorted({int(item) for item in config.get("top_ks", [1, 3, 5])})
    max_candidates = int(config.get("max_candidates", max(top_ks)))
    symbol_eval_k = int(config.get("symbol_eval_k", 3))
    pattern_eval_k = int(config.get("pattern_eval_k", 3))
    highlight_eval_k = int(config.get("highlight_eval_k", pattern_eval_k))
    save_per_case = bool(config.get("save_per_case", True))

    # 使用 code_roots 里第一个有效目录初始化检索器。
    effective_roots = [path for path in code_root_paths if path.exists()]
    if not effective_roots:
        raise FileNotFoundError("No valid code_roots found in config")
    retriever = LocalCodeRetriever(
        project_root=PROJECT_ROOT,
        code_dirs=effective_roots,
        default_top_k=max_candidates,
    )
    cases = _load_dataset(dataset_path)

    recall_sums = {k: 0.0 for k in top_ks}
    mrr_sum = 0.0
    top1_path_accuracy_sum = 0.0
    symbol_hit_sum = 0.0
    pattern_hit_sum = 0.0
    pattern_hit_at1_sum = 0.0
    highlight_hit_sum = 0.0
    latency_ms_sum = 0.0
    per_case_results: list[dict[str, Any]] = []

    for case in cases:
        started_at = perf_counter()
        hits = retriever.search(
            user_query=case.query,
            retrieval_queries=case.retrieval_queries,
            module_name="",
            top_k=max_candidates,
        )
        latency_ms = (perf_counter() - started_at) * 1000
        latency_ms_sum += latency_ms

        retrieved_paths = [str(hit.get("path", "")) for hit in hits if str(hit.get("path", "")).strip()]
        gold_paths_set = set(case.gold_paths)

        case_recalls: dict[str, float] = {}
        for k in top_ks:
            case_recall = _compute_recall_at_k(retrieved_paths, gold_paths_set, k)
            recall_sums[k] += case_recall
            case_recalls[f"recall@{k}"] = case_recall

        case_mrr = _reciprocal_rank(retrieved_paths, gold_paths_set)
        mrr_sum += case_mrr

        top1_hit = 1.0 if retrieved_paths and retrieved_paths[0] in gold_paths_set else 0.0
        top1_path_accuracy_sum += top1_hit

        symbol_hit = _compute_symbol_hit_at_k(hits, case.gold_symbols, symbol_eval_k)
        pattern_hit_at1 = _compute_pattern_hit_at_k(hits, case.expected_patterns, 1)
        pattern_hit = _compute_pattern_hit_at_k(hits, case.expected_patterns, pattern_eval_k)
        highlight_hit = _compute_highlight_hit_at_k(hits, case.expected_patterns, highlight_eval_k)
        symbol_hit_sum += symbol_hit
        pattern_hit_at1_sum += pattern_hit_at1
        pattern_hit_sum += pattern_hit
        highlight_hit_sum += highlight_hit

        if save_per_case:
            per_case_results.append(
                {
                    "id": case.case_id,
                    "query": case.query,
                    "retrieval_queries": case.retrieval_queries,
                    "gold_paths": case.gold_paths,
                    "gold_symbols": case.gold_symbols,
                    "expected_patterns": case.expected_patterns,
                    "retrieved_paths": retrieved_paths,
                    "top_hits": hits[:5],
                    "mrr": round(case_mrr, 6),
                    "top1_path_hit": top1_hit,
                    f"symbol_hit@{symbol_eval_k}": symbol_hit,
                    "pattern_hit@1": pattern_hit_at1,
                    f"pattern_hit@{pattern_eval_k}": pattern_hit,
                    f"highlight_hit@{highlight_eval_k}": highlight_hit,
                    "latency_ms": round(latency_ms, 2),
                    "intent": case.intent,
                    "note": case.note,
                    **case_recalls,
                }
            )

    total_cases = len(cases)
    metrics: dict[str, Any] = {}
    for k in top_ks:
        metrics[f"recall@{k}"] = round(recall_sums[k] / total_cases, 6)
    metrics["mrr"] = round(mrr_sum / total_cases, 6)
    metrics["top1_path_accuracy"] = round(top1_path_accuracy_sum / total_cases, 6)
    metrics[f"symbol_hit_rate@{symbol_eval_k}"] = round(symbol_hit_sum / total_cases, 6)
    metrics["pattern_hit_rate@1"] = round(pattern_hit_at1_sum / total_cases, 6)
    metrics[f"pattern_hit_rate@{pattern_eval_k}"] = round(pattern_hit_sum / total_cases, 6)
    metrics[f"highlight_hit_rate@{highlight_eval_k}"] = round(highlight_hit_sum / total_cases, 6)
    metrics["avg_latency_ms"] = round(latency_ms_sum / total_cases, 3)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": _display_path(config_path),
        "dataset_path": _display_path(dataset_path),
        "output_path": _display_path(output_path),
        "code_roots": [_display_path(path) for path in effective_roots],
        "total_cases": total_cases,
        "top_ks": top_ks,
        "max_candidates": max_candidates,
        "symbol_eval_k": symbol_eval_k,
        "pattern_eval_k": pattern_eval_k,
        "highlight_eval_k": highlight_eval_k,
        "metrics": metrics,
        "per_case": per_case_results if save_per_case else [],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    return report


def main() -> int:
    """
    执行`main` 相关处理逻辑。
    
    返回:
        返回类型为 `int` 的处理结果。
    """
    parser = argparse.ArgumentParser(description="Run code retrieval evaluation.")
    parser.add_argument(
        "--config",
        default="src/workflow/eval/config.code.template.json",
        help="Path to code eval config JSON (relative to project root or absolute).",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")

    report = run_eval(config_path)
    metrics = report["metrics"]
    print("==== Code Retrieval Evaluation Summary ====")
    print(f"total_cases: {report['total_cases']}")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value}")
    print("report_path:", report["output_path"])
    print("===========================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
