from __future__ import annotations

"""Wiki 切片导出工具。

目标：
1. 读取当前 MarkdownWikiRetriever 的切片结果；
2. 按文档聚合导出 chunk 明细；
3. 输出 JSON 与 Markdown 两种格式，便于人工检查与回归对比。
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from workflow.nodes.retrieve_wiki.wiki_retriever import MarkdownWikiRetriever  # noqa: E402


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _build_export_payload(
    *,
    wiki_dir: Path,
    default_top_k: int,
    max_content_chars: int,
) -> dict[str, Any]:
    retriever = MarkdownWikiRetriever(
        wiki_dir=wiki_dir,
        project_root=PROJECT_ROOT,
        default_top_k=default_top_k,
    )
    rows = retriever.get_chunk_snapshot()

    docs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        docs[row["path"]].append(row)

    documents: list[dict[str, Any]] = []
    for path_key in sorted(docs.keys()):
        doc_rows = docs[path_key]
        section_counter = Counter(item["section"] for item in doc_rows)
        type_counter = Counter(item["chunk_type"] for item in doc_rows)

        chunk_items: list[dict[str, Any]] = []
        for local_index, item in enumerate(doc_rows, start=1):
            chunk_items.append(
                {
                    "doc_chunk_index": local_index,
                    "global_chunk_index": item["index"],
                    "section": item["section"],
                    "chunk_type": item["chunk_type"],
                    "content_length": item["content_length"],
                    "content_preview": item["content"][:max_content_chars],
                }
            )

        documents.append(
            {
                "path": path_key,
                "title": doc_rows[0]["title"] if doc_rows else "",
                "chunk_count": len(doc_rows),
                "sections": dict(section_counter),
                "chunk_types": dict(type_counter),
                "chunks": chunk_items,
            }
        )

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "wiki_dir": _display_path(wiki_dir),
        "total_docs": len(documents),
        "total_chunks": len(rows),
        "documents": documents,
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Wiki 切片导出结果")
    lines.append("")
    lines.append(f"- generated_at: `{payload['generated_at']}`")
    lines.append(f"- wiki_dir: `{payload['wiki_dir']}`")
    lines.append(f"- total_docs: `{payload['total_docs']}`")
    lines.append(f"- total_chunks: `{payload['total_chunks']}`")
    lines.append("")

    for doc in payload["documents"]:
        lines.append(f"## {doc['path']}")
        lines.append("")
        lines.append(f"- title: `{doc['title']}`")
        lines.append(f"- chunk_count: `{doc['chunk_count']}`")
        lines.append(f"- sections: `{doc['sections']}`")
        lines.append(f"- chunk_types: `{doc['chunk_types']}`")
        lines.append("")
        lines.append("| idx | section | type | len | preview |")
        lines.append("| --- | --- | --- | --- | --- |")
        for chunk in doc["chunks"]:
            preview = chunk["content_preview"].replace("\n", " | ").replace("|", "\\|")
            lines.append(
                f"| {chunk['doc_chunk_index']} | {chunk['section']} | {chunk['chunk_type']} | "
                f"{chunk['content_length']} | {preview} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export wiki chunk snapshot per document.")
    parser.add_argument(
        "--wiki-dir",
        default="domain/ad_engine/wiki",
        help="Wiki directory path (relative to project root or absolute).",
    )
    parser.add_argument(
        "--output-json",
        default="src/workflow/eval/results/wiki_chunk_export.json",
        help="Output JSON path (relative to project root or absolute).",
    )
    parser.add_argument(
        "--output-md",
        default="src/workflow/eval/results/wiki_chunk_export.md",
        help="Output Markdown path (relative to project root or absolute).",
    )
    parser.add_argument(
        "--max-content-chars",
        type=int,
        default=180,
        help="Max preview length for each chunk.",
    )
    args = parser.parse_args()

    wiki_dir = Path(args.wiki_dir)
    if not wiki_dir.is_absolute():
        wiki_dir = (PROJECT_ROOT / wiki_dir).resolve()

    output_json = Path(args.output_json)
    if not output_json.is_absolute():
        output_json = (PROJECT_ROOT / output_json).resolve()

    output_md = Path(args.output_md)
    if not output_md.is_absolute():
        output_md = (PROJECT_ROOT / output_md).resolve()

    payload = _build_export_payload(
        wiki_dir=wiki_dir,
        default_top_k=6,
        max_content_chars=max(40, args.max_content_chars),
    )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    markdown_text = _render_markdown(payload)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(markdown_text, encoding="utf-8")

    print("==== Wiki Chunk Export ====")
    print("wiki_dir:", _display_path(wiki_dir))
    print("total_docs:", payload["total_docs"])
    print("total_chunks:", payload["total_chunks"])
    print("json:", _display_path(output_json))
    print("md  :", _display_path(output_md))
    print("===========================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
