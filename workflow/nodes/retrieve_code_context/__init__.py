from __future__ import annotations

from typing import Any


def run(service: Any, state: dict[str, Any]) -> dict[str, Any]:
    """模拟代码生成前的额外代码上下文检索。"""
    module_name = state["module_name"]
    items = [
        {
            "source_type": "code",
            "title": f"{module_name} 目标文件",
            "path": f"services/ad_engine/{module_name}/handler.py",
            "score": 0.91,
            "excerpt": "建议在入口阶段增加特征完整性校验与预算节奏保护。",
        },
        {
            "source_type": "code",
            "title": f"{module_name} 测试文件",
            "path": f"tests/ad_engine/{module_name}/test_handler.py",
            "score": 0.84,
            "excerpt": "建议新增冷启动、频控边界和回传延迟三类测试。",
        },
    ]
    return {
        "code_hits": items,
        "citations": items,
        "node_trace": service._trace(state, "retrieve_code_context", f"hits={len(items)}"),
    }

