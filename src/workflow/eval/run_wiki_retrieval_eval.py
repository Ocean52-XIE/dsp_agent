# -*- coding: utf-8 -*-
"""
该模块实现评测流程，负责样本执行、指标统计与结果输出。
"""
from __future__ import annotations

"""检索评测脚本。

功能目标：
1. 基于评测集计算 Wiki 检索 Recall@K、MRR；
2. 通过 WorkflowService 端到端跑一次问答，统计“引用命中率”；
3. 输出结构化评测报告，便于后续做基线对比与回归门禁。

说明：
- 脚本默认读取 `src/workflow/eval/config.template.json`。
- 评测集格式为 JSONL，每行一条样本。
"""

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any


# 将项目根目录加入 sys.path，保证在任意目录执行脚本时都能正确 import workflow。
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from workflow.engine import WorkflowService  # noqa: E402
from workflow.common.module_inference import infer_module  # noqa: E402


@dataclass
class EvalCase:
    """
    定义`EvalCase`，用于封装相关数据结构与处理行为。
    """

    case_id: str
    query: str
    gold_paths: list[str]
    gold_sections: list[str]
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


def _load_dataset(path: Path) -> list[EvalCase]:
    """
    内部辅助函数，负责`load dataset` 相关处理。
    
    参数:
        path: 文件或目录路径。
    
    返回:
        返回类型为 `list[EvalCase]` 的处理结果。
    """
    items: list[EvalCase] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            case_id = str(record.get("id", f"case_{line_number:04d}"))
            query = str(record.get("query", "")).strip()
            if not query:
                raise ValueError(f"dataset line {line_number} missing query")
            gold_paths = [str(item) for item in record.get("gold_paths", []) if str(item).strip()]
            if not gold_paths:
                raise ValueError(f"dataset line {line_number} missing gold_paths")
            gold_sections = [str(item) for item in record.get("gold_sections", []) if str(item).strip()]
            items.append(
                EvalCase(
                    case_id=case_id,
                    query=query,
                    gold_paths=gold_paths,
                    gold_sections=gold_sections,
                    intent=str(record.get("intent", "knowledge_qa")),
                    note=str(record.get("note", "")),
                )
            )
    return items


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
    top_k = retrieved_paths[:k]
    return 1.0 if any(path in gold_paths for path in top_k) else 0.0


def _build_retrieval_queries(service: WorkflowService, query: str) -> tuple[str, str, list[str]]:
    """
    构建当前步骤所需的数据结构或文本内容。
    
    参数:
        service: 工作流服务对象，提供检索、路由、日志与配置能力。
        query: 输入参数，用于控制当前处理逻辑。
    
    返回:
        返回类型为 `tuple[str, str, list[str]]` 的处理结果。
    """
    module_name, module_hint = infer_module(query, domain_profile=service.domain_profile)
    rewritten = service._query_rewriter(  # 这里直接调用节点方法，保持与线上链路一致。
        {
            "user_query": query,
            "module_name": module_name,
            "route": "knowledge_qa",
        }
    )
    retrieval_queries = rewritten.get("retrieval_queries", [])
    return module_name, module_hint, retrieval_queries


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
    top_ks = sorted({int(item) for item in config.get("top_ks", [1, 3, 5])})
    max_candidates = int(config.get("max_candidates", max(top_ks)))
    if max_candidates < max(top_ks):
        max_candidates = max(top_ks)

    enable_agent_citation_eval = bool(config.get("enable_agent_citation_eval", True))
    save_per_case = bool(config.get("save_per_case", True))
    session_id_prefix = str(config.get("session_id_prefix", "eval_sess"))
    trace_id_prefix = str(config.get("trace_id_prefix", "eval_trace"))

    cases = _load_dataset(dataset_path)
    service = WorkflowService()

    recall_sums = {k: 0.0 for k in top_ks}
    mrr_sum = 0.0
    citation_hit_sum = 0.0
    citation_total = 0
    per_case_results: list[dict[str, Any]] = []

    for case_index, case in enumerate(cases, start=1):
        gold_paths_set = set(case.gold_paths)
        module_name, module_hint, retrieval_queries = _build_retrieval_queries(service, case.query)
        wiki_hits = service._wiki_retriever.search(
            user_query=case.query,
            retrieval_queries=retrieval_queries,
            module_name=module_name,
            top_k=max_candidates,
        )
        retrieved_paths = [str(item.get("path", "")) for item in wiki_hits if str(item.get("path", "")).strip()]

        case_recalls: dict[str, float] = {}
        for top_k in top_ks:
            case_recall = _compute_recall_at_k(retrieved_paths, gold_paths_set, top_k)
            recall_sums[top_k] += case_recall
            case_recalls[f"recall@{top_k}"] = case_recall

        case_mrr = _reciprocal_rank(retrieved_paths, gold_paths_set)
        mrr_sum += case_mrr

        citation_hit = None
        citation_paths: list[str] = []
        if enable_agent_citation_eval:
            # 使用端到端调用验证最终引用是否命中 gold。
            response = service.run_user_message(
                session_id=f"{session_id_prefix}_{case.case_id}",
                trace_id=f"{trace_id_prefix}_{case_index:04d}",
                user_query=case.query,
                history=[],
            )
            citations = response.get("citations", [])
            citation_paths = [
                str(item.get("path", ""))
                for item in citations
                if item.get("source_type") == "wiki" and str(item.get("path", "")).strip()
            ]
            citation_hit = 1.0 if any(path in gold_paths_set for path in citation_paths) else 0.0
            citation_hit_sum += citation_hit
            citation_total += 1

        if save_per_case:
            per_case_results.append(
                {
                    "id": case.case_id,
                    "query": case.query,
                    "gold_paths": case.gold_paths,
                    "module_name": module_name,
                    "module_hint": module_hint,
                    "retrieval_queries": retrieval_queries,
                    "retrieved_paths": retrieved_paths,
                    "top_hits": wiki_hits[:5],
                    "mrr": case_mrr,
                    **case_recalls,
                    "citation_hit": citation_hit,
                    "citation_paths": citation_paths,
                    "intent": case.intent,
                    "note": case.note,
                }
            )

    total_cases = len(cases)
    metrics: dict[str, Any] = {}
    for top_k in top_ks:
        metrics[f"recall@{top_k}"] = round(recall_sums[top_k] / total_cases, 6)
    metrics["mrr"] = round(mrr_sum / total_cases, 6)
    if citation_total > 0:
        metrics["citation_hit_rate"] = round(citation_hit_sum / citation_total, 6)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": _display_path(config_path),
        "dataset_path": _display_path(dataset_path),
        "output_path": _display_path(output_path),
        "total_cases": total_cases,
        "top_ks": top_ks,
        "max_candidates": max_candidates,
        "enable_agent_citation_eval": enable_agent_citation_eval,
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
    parser = argparse.ArgumentParser(description="Run retrieval evaluation (Recall@K, MRR, citation hit rate).")
    parser.add_argument(
        "--config",
        default="src/workflow/eval/config.template.json",
        help="Path to evaluation config JSON (relative to project root or absolute).",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")

    report = run_eval(config_path)
    metrics = report["metrics"]
    print("==== Retrieval Evaluation Summary ====")
    print(f"total_cases: {report['total_cases']}")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value}")
    print("report_path:", report["output_path"])
    print("=====================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
