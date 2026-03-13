from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from workflow.eval.run_code_retrieval_eval import run_eval as run_code_eval  # noqa: E402
from workflow.eval.run_retrieval_eval import run_eval as run_wiki_eval  # noqa: E402


STRATEGIES = ("rg_first", "rg_only", "no_rg")


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_compare(
    *,
    wiki_dataset: str,
    code_dataset: str,
    results_dir: Path,
    max_candidates: int,
) -> dict[str, Any]:
    config_dir = results_dir / "rg_strategy_compare_configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "datasets": {"wiki": wiki_dataset, "code": code_dataset},
        "wiki": {},
        "code": {},
    }

    for strategy in STRATEGIES:
        os.environ["WORKFLOW_WIKI_RG_STRATEGY"] = strategy
        wiki_output = _display_path(results_dir / f"rg_compare_wiki_{strategy}.json")
        wiki_config = {
            "dataset_path": wiki_dataset,
            "top_ks": [1, 3, 5],
            "max_candidates": int(max_candidates),
            "enable_agent_citation_eval": False,
            "session_id_prefix": f"eval_wiki_{strategy}",
            "trace_id_prefix": f"eval_wiki_{strategy}",
            "output_path": wiki_output,
            "save_per_case": True,
        }
        wiki_config_path = config_dir / f"wiki_{strategy}.json"
        _write_json(wiki_config_path, wiki_config)
        wiki_report = run_wiki_eval(wiki_config_path)
        summary["wiki"][strategy] = {
            "output_path": wiki_report["output_path"],
            "total_cases": wiki_report["total_cases"],
            "metrics": wiki_report["metrics"],
        }

    for strategy in STRATEGIES:
        os.environ["WORKFLOW_CODE_RG_STRATEGY"] = strategy
        code_output = _display_path(results_dir / f"rg_compare_code_{strategy}.json")
        code_config = {
            "dataset_path": code_dataset,
            "code_roots": ["domain/ad_engine/codes"],
            "top_ks": [1, 3, 5],
            "max_candidates": int(max_candidates),
            "symbol_eval_k": 3,
            "pattern_eval_k": 3,
            "highlight_eval_k": 3,
            "output_path": code_output,
            "save_per_case": True,
        }
        code_config_path = config_dir / f"code_{strategy}.json"
        _write_json(code_config_path, code_config)
        code_report = run_code_eval(code_config_path)
        summary["code"][strategy] = {
            "output_path": code_report["output_path"],
            "total_cases": code_report["total_cases"],
            "metrics": code_report["metrics"],
        }

    return summary


def _build_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# RG Strategy Compare (30 Cases)")
    lines.append("")
    lines.append(f"- Generated: {summary['generated_at']}")
    lines.append(f"- Wiki dataset: `{summary['datasets']['wiki']}`")
    lines.append(f"- Code dataset: `{summary['datasets']['code']}`")
    lines.append("")

    wiki_keys = ["recall@1", "recall@3", "recall@5", "mrr"]
    lines.append("## Wiki Metrics")
    lines.append("| strategy | " + " | ".join(wiki_keys) + " |")
    lines.append("|" + "---|" * (len(wiki_keys) + 1))
    for strategy in STRATEGIES:
        metrics = summary["wiki"][strategy]["metrics"]
        lines.append("| " + strategy + " | " + " | ".join(str(metrics.get(key, "")) for key in wiki_keys) + " |")
    lines.append("")

    code_keys = [
        "recall@1",
        "recall@3",
        "recall@5",
        "mrr",
        "top1_path_accuracy",
        "symbol_hit_rate@3",
        "pattern_hit_rate@1",
        "pattern_hit_rate@3",
        "highlight_hit_rate@3",
        "avg_latency_ms",
    ]
    lines.append("## Code Metrics")
    lines.append("| strategy | " + " | ".join(code_keys) + " |")
    lines.append("|" + "---|" * (len(code_keys) + 1))
    for strategy in STRATEGIES:
        metrics = summary["code"][strategy]["metrics"]
        lines.append("| " + strategy + " | " + " | ".join(str(metrics.get(key, "")) for key in code_keys) + " |")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run wiki/code retrieval eval across rg strategies and compare metrics.")
    parser.add_argument(
        "--wiki-dataset",
        default="domain/ad_engine/eval/datasets/ad_engine_retrieval_eval_30.jsonl",
        help="Wiki retrieval dataset path (relative to project root or absolute).",
    )
    parser.add_argument(
        "--code-dataset",
        default="domain/ad_engine/eval/datasets/ad_engine_code_retrieval_eval_30.jsonl",
        help="Code retrieval dataset path (relative to project root or absolute).",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=8,
        help="Max candidates passed into both eval runners.",
    )
    parser.add_argument(
        "--results-dir",
        default="src/workflow/eval/results",
        help="Directory for generated reports.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = (PROJECT_ROOT / results_dir).resolve()

    summary = run_compare(
        wiki_dataset=str(args.wiki_dataset),
        code_dataset=str(args.code_dataset),
        results_dir=results_dir,
        max_candidates=int(args.max_candidates),
    )

    summary_path = results_dir / "rg_strategy_compare_30.json"
    markdown_path = results_dir / "rg_strategy_compare_30.md"
    _write_json(summary_path, summary)
    markdown_path.write_text(_build_markdown(summary), encoding="utf-8")

    print("==== RG Strategy Compare Summary ====")
    print("summary_json:", _display_path(summary_path))
    print("summary_md  :", _display_path(markdown_path))
    for strategy in STRATEGIES:
        print(f"wiki[{strategy}]:", summary["wiki"][strategy]["metrics"])
    for strategy in STRATEGIES:
        print(f"code[{strategy}]:", summary["code"][strategy]["metrics"])
    print("====================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
