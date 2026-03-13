# -*- coding: utf-8 -*-
"""
该模块实现评测流程，负责样本执行、指标统计与结果输出。
"""
from __future__ import annotations

"""Issue-analysis routing evaluation script.

Evaluates whether queries expected to trigger issue-analysis flow are routed to:
- route: issue_analysis
- kind: issue_analysis
- status: confirm_code
- module: expected_module (optional)
"""

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from workflow.engine import WorkflowService  # noqa: E402


@dataclass
class IssueEvalCase:
    """
    定义`IssueEvalCase`，用于封装相关数据结构与处理行为。
    """
    case_id: str
    query: str
    expected_route: str
    expected_kind: str
    expected_status: str
    expected_module: str
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


def _load_dataset(path: Path) -> list[IssueEvalCase]:
    """
    内部辅助函数，负责`load dataset` 相关处理。
    
    参数:
        path: 文件或目录路径。
    
    返回:
        返回类型为 `list[IssueEvalCase]` 的处理结果。
    """
    items: list[IssueEvalCase] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            query = str(record.get("query", "")).strip()
            if not query:
                raise ValueError(f"dataset line {line_number} missing query")

            items.append(
                IssueEvalCase(
                    case_id=str(record.get("id", f"issue_case_{line_number:04d}")),
                    query=query,
                    expected_route=str(record.get("expected_route", "issue_analysis")).strip() or "issue_analysis",
                    expected_kind=str(record.get("expected_kind", "issue_analysis")).strip() or "issue_analysis",
                    expected_status=str(record.get("expected_status", "confirm_code")).strip() or "confirm_code",
                    expected_module=str(record.get("expected_module", "")).strip(),
                    intent=str(record.get("intent", "issue_analysis")).strip() or "issue_analysis",
                    note=str(record.get("note", "")).strip(),
                )
            )
    return items


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
    save_per_case = bool(config.get("save_per_case", True))
    session_id_prefix = str(config.get("session_id_prefix", "issue_eval_sess"))
    trace_id_prefix = str(config.get("trace_id_prefix", "issue_eval_trace"))

    cases = _load_dataset(dataset_path)
    service = WorkflowService()

    route_hit_sum = 0.0
    kind_hit_sum = 0.0
    status_hit_sum = 0.0
    module_hit_sum = 0.0
    module_eval_count = 0
    strict_pass_sum = 0.0
    per_case_results: list[dict[str, Any]] = []

    for case_index, case in enumerate(cases, start=1):
        response = service.run_user_message(
            session_id=f"{session_id_prefix}_{case.case_id}",
            trace_id=f"{trace_id_prefix}_{case_index:04d}",
            user_query=case.query,
            history=[],
        )
        actual_route = str(response.get("intent", "")).strip()
        actual_kind = str(response.get("kind", "")).strip()
        actual_status = str(response.get("status", "")).strip()
        actual_module = str((response.get("analysis") or {}).get("module", "")).strip()

        route_hit = 1.0 if actual_route == case.expected_route else 0.0
        kind_hit = 1.0 if actual_kind == case.expected_kind else 0.0
        status_hit = 1.0 if actual_status == case.expected_status else 0.0
        route_hit_sum += route_hit
        kind_hit_sum += kind_hit
        status_hit_sum += status_hit

        module_hit: float | None = None
        if case.expected_module:
            module_eval_count += 1
            module_hit = 1.0 if actual_module == case.expected_module else 0.0
            module_hit_sum += module_hit

        strict_pass = route_hit == 1.0 and kind_hit == 1.0 and status_hit == 1.0
        if module_hit is not None:
            strict_pass = strict_pass and module_hit == 1.0
        strict_pass_sum += 1.0 if strict_pass else 0.0

        if save_per_case:
            per_case_results.append(
                {
                    "id": case.case_id,
                    "query": case.query,
                    "intent": case.intent,
                    "note": case.note,
                    "expected": {
                        "route": case.expected_route,
                        "kind": case.expected_kind,
                        "status": case.expected_status,
                        "module": case.expected_module,
                    },
                    "actual": {
                        "route": actual_route,
                        "kind": actual_kind,
                        "status": actual_status,
                        "module": actual_module,
                    },
                    "hits": {
                        "route_hit": route_hit,
                        "kind_hit": kind_hit,
                        "status_hit": status_hit,
                        "module_hit": module_hit,
                    },
                    "strict_pass": strict_pass,
                }
            )

    total_cases = len(cases)
    metrics: dict[str, Any] = {
        "route_accuracy": round(route_hit_sum / total_cases, 6),
        "kind_accuracy": round(kind_hit_sum / total_cases, 6),
        "status_accuracy": round(status_hit_sum / total_cases, 6),
        "strict_pass_rate": round(strict_pass_sum / total_cases, 6),
    }
    if module_eval_count > 0:
        metrics["module_accuracy"] = round(module_hit_sum / module_eval_count, 6)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": _display_path(config_path),
        "dataset_path": _display_path(dataset_path),
        "output_path": _display_path(output_path),
        "total_cases": total_cases,
        "module_eval_count": module_eval_count,
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
    parser = argparse.ArgumentParser(description="Run issue-analysis routing evaluation.")
    parser.add_argument(
        "--config",
        default="src/workflow/eval/config.issue_analysis.template.json",
        help="Path to issue-analysis eval config JSON (relative to project root or absolute).",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")

    report = run_eval(config_path)
    metrics = report["metrics"]
    print("==== Issue Analysis Eval Summary ====")
    print(f"total_cases: {report['total_cases']}")
    print(f"route_accuracy: {metrics['route_accuracy']}")
    print(f"kind_accuracy: {metrics['kind_accuracy']}")
    print(f"status_accuracy: {metrics['status_accuracy']}")
    if "module_accuracy" in metrics:
        print(f"module_accuracy: {metrics['module_accuracy']}")
    print(f"strict_pass_rate: {metrics['strict_pass_rate']}")
    print("report_path:", report["output_path"])
    print("=====================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
