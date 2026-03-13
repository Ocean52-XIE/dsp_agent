# -*- coding: utf-8 -*-
"""
该模块实现评测流程，负责样本执行、指标统计与结果输出。
"""
from __future__ import annotations

"""Minimal regression checks for session-context continuity and state isolation."""

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from workflow.engine import WorkflowService  # noqa: E402


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


def _build_user_message(content: str, index: int) -> dict[str, Any]:
    """
    构建当前步骤所需的数据结构或文本内容。
    
    参数:
        content: 输入参数，用于控制当前处理逻辑。
        index: 输入参数，用于控制当前处理逻辑。
    
    返回:
        返回类型为 `dict[str, Any]` 的处理结果。
    """
    return {
        "id": f"user_{index}",
        "role": "user",
        "kind": "user_input",
        "intent": None,
        "status": "submitted",
        "content": content,
        "created_at": "",
        "trace_id": None,
        "citations": [],
        "analysis": None,
        "actions": [],
        "debug": {},
    }


@dataclass
class RegressionCase:
    """
    定义`RegressionCase`，用于封装相关数据结构与处理行为。
    """
    case_id: str
    description: str
    turns: list[str]
    verify: Callable[[list[dict[str, Any]]], list[str]]


def _verify_context_followup(turn_outputs: list[dict[str, Any]]) -> list[str]:
    """
    内部辅助函数，负责`verify context followup` 相关处理。
    
    参数:
        turn_outputs: 列表参数，用于承载批量输入数据。
    
    返回:
        返回类型为 `list[str]` 的处理结果。
    """
    failures: list[str] = []
    if len(turn_outputs) != 2:
        return ["expected 2 turns"]

    first = turn_outputs[0]
    second = turn_outputs[1]
    first_module = str((first.get("analysis") or {}).get("module", ""))
    second_module = str((second.get("analysis") or {}).get("module", ""))
    second_kind = str(second.get("kind", ""))

    if first_module != "bid-optimizer":
        failures.append(f"turn1 module should be bid-optimizer, got: {first_module or '<empty>'}")
    if second_kind != "knowledge_qa":
        failures.append(f"turn2 kind should be knowledge_qa, got: {second_kind or '<empty>'}")
    if second_module != first_module:
        failures.append(f"turn2 module should keep turn1 module ({first_module}), got: {second_module or '<empty>'}")

    citation_paths = [str(item.get("path", "")) for item in (second.get("citations") or [])]
    if not any("bid/bid_optimizer.py" in path.replace("\\", "/") for path in citation_paths):
        failures.append("turn2 citations should include bid/bid_optimizer.py")

    return failures


def _verify_out_of_scope_no_leak(turn_outputs: list[dict[str, Any]]) -> list[str]:
    """
    内部辅助函数，负责`verify out of scope no leak` 相关处理。
    
    参数:
        turn_outputs: 列表参数，用于承载批量输入数据。
    
    返回:
        返回类型为 `list[str]` 的处理结果。
    """
    failures: list[str] = []
    if len(turn_outputs) != 2:
        return ["expected 2 turns"]

    second = turn_outputs[1]
    second_kind = str(second.get("kind", ""))
    second_status = str(second.get("status", ""))
    second_citations = list(second.get("citations", []) or [])

    if second_kind != "out_of_scope":
        failures.append(f"turn2 kind should be out_of_scope, got: {second_kind or '<empty>'}")
    if second_status != "out_of_scope":
        failures.append(f"turn2 status should be out_of_scope, got: {second_status or '<empty>'}")
    if second_citations:
        failures.append(f"turn2 citations should be empty, got {len(second_citations)}")

    debug = second.get("debug") or {}
    verbose = debug.get("verbose") or {}
    retrieval_queries = list(verbose.get("retrieval_queries", []) or [])
    evidence_count = verbose.get("evidence_count")

    if retrieval_queries:
        failures.append(f"turn2 retrieval_queries should be empty, got {len(retrieval_queries)}")
    if evidence_count not in (0, None):
        failures.append(f"turn2 evidence_count should be 0/None, got {evidence_count}")

    return failures


def _verify_generic_troubleshoot_to_knowledge(turn_outputs: list[dict[str, Any]]) -> list[str]:
    """
    内部辅助函数，负责`verify generic troubleshoot to knowledge` 相关处理。
    
    参数:
        turn_outputs: 列表参数，用于承载批量输入数据。
    
    返回:
        返回类型为 `list[str]` 的处理结果。
    """
    failures: list[str] = []
    if len(turn_outputs) != 1:
        return ["expected 1 turn"]

    first = turn_outputs[0]
    kind = str(first.get("kind", ""))
    status = str(first.get("status", ""))
    if kind != "knowledge_qa":
        failures.append(f"generic troubleshoot query should route to knowledge_qa, got: {kind or '<empty>'}")
    if status == "out_of_scope":
        failures.append("generic troubleshoot query should stay in-domain, got out_of_scope")
    return failures


def _build_cases() -> list[RegressionCase]:
    """
    构建当前步骤所需的数据结构或文本内容。
    
    返回:
        返回类型为 `list[RegressionCase]` 的处理结果。
    """
    return [
        RegressionCase(
            case_id="context_followup_code_location",
            description="The short follow-up 'code where' should keep previous module/topic.",
            turns=[
                "target_cpa \u548c pCVR \u7684\u51fa\u4ef7\u516c\u5f0f\u662f\u4ec0\u4e48",
                "\u4ee3\u7801\u5728\u54ea\u91cc",
            ],
            verify=_verify_context_followup,
        ),
        RegressionCase(
            case_id="out_of_scope_no_citation_leak",
            description="Out-of-scope response should not leak previous turn citations/evidence.",
            turns=[
                "target_cpa \u548c pCVR \u7684\u51fa\u4ef7\u516c\u5f0f\u662f\u4ec0\u4e48",
                "\u4eca\u5929\u5929\u6c14\u600e\u4e48\u6837",
            ],
            verify=_verify_out_of_scope_no_leak,
        ),
        RegressionCase(
            case_id="generic_troubleshoot_to_knowledge",
            description="Generic troubleshoot method query without concrete incident details should go to knowledge QA.",
            turns=[
                "\u53ec\u56de\u961f\u5217\u4e0b\u964d\u600e\u4e48\u6392\u67e5",
            ],
            verify=_verify_generic_troubleshoot_to_knowledge,
        ),
    ]


def _run_single_case(service: WorkflowService, case: RegressionCase, *, session_id: str) -> dict[str, Any]:
    """
    内部辅助函数，负责`run single case` 相关处理。
    
    参数:
        service: 工作流服务对象，提供检索、路由、日志与配置能力。
        case: 输入参数，用于控制当前处理逻辑。
    
    返回:
        返回类型为 `dict[str, Any]` 的处理结果。
    """
    history: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    turn_snapshots: list[dict[str, Any]] = []

    for index, query in enumerate(case.turns, start=1):
        user_message = _build_user_message(query, index=index)
        history.append(user_message)
        response = service.run_user_message(
            session_id=session_id,
            trace_id=f"reg_trace_{case.case_id}_{index}",
            user_query=query,
            history=history,
        )
        response = dict(response)
        response["id"] = f"assistant_{index}"
        response["created_at"] = ""
        history.append(response)
        outputs.append(response)

        analysis = response.get("analysis") or {}
        turn_snapshots.append(
            {
                "query": query,
                "kind": response.get("kind"),
                "status": response.get("status"),
                "module": analysis.get("module"),
                "citation_count": len(response.get("citations", []) or []),
                "first_citation_path": ((response.get("citations") or [{}])[0]).get("path"),
            }
        )

    failures = case.verify(outputs)
    return {
        "case_id": case.case_id,
        "description": case.description,
        "passed": len(failures) == 0,
        "failures": failures,
        "turns": turn_snapshots,
    }


def run_regression(*, output_path: Path, debug_verbose: bool) -> dict[str, Any]:
    """
    执行对应子流程并返回执行结果。
    
    返回:
        返回类型为 `dict[str, Any]` 的处理结果。
    """
    previous_debug_env = os.getenv("WORKFLOW_DEBUG_VERBOSE")
    os.environ["WORKFLOW_DEBUG_VERBOSE"] = "1" if debug_verbose else "0"
    try:
        service = WorkflowService()
        cases = _build_cases()
        case_results: list[dict[str, Any]] = []
        for index, case in enumerate(cases, start=1):
            session_id = f"context_regression_case_{index}_{case.case_id}"
            case_results.append(_run_single_case(service, case, session_id=session_id))
    finally:
        if previous_debug_env is None:
            os.environ.pop("WORKFLOW_DEBUG_VERBOSE", None)
        else:
            os.environ["WORKFLOW_DEBUG_VERBOSE"] = previous_debug_env

    total = len(case_results)
    passed = sum(1 for item in case_results if item.get("passed"))
    failed = total - passed

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total_cases": total,
        "passed_cases": passed,
        "failed_cases": failed,
        "all_passed": failed == 0,
        "debug_verbose_enabled": debug_verbose,
        "cases": case_results,
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
    parser = argparse.ArgumentParser(description="Run minimal context-regression checks.")
    parser.add_argument(
        "--output",
        default="src/workflow/eval/results/latest_context_regression_report.json",
        help="Path to output JSON report (relative to project root or absolute).",
    )
    parser.add_argument(
        "--debug-verbose",
        action="store_true",
        help="Enable WORKFLOW_DEBUG_VERBOSE for richer assertions in report.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = (PROJECT_ROOT / output_path).resolve()

    report = run_regression(output_path=output_path, debug_verbose=bool(args.debug_verbose))
    print("==== Context Regression Summary ====")
    print(f"total_cases: {report['total_cases']}")
    print(f"passed_cases: {report['passed_cases']}")
    print(f"failed_cases: {report['failed_cases']}")
    print("report_path:", _display_path(output_path))
    print("===================================")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
