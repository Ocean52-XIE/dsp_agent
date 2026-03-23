"""追踪日志 mock 代码语料。"""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class TraceRecord:
    """请求级 trace 记录。"""

    trace_id: str
    module_name: str
    event: str
    timestamp: str


def build_trace_record(*, trace_id: str, module_name: str, event: str) -> TraceRecord:
    """构造 trace 记录。"""
    return TraceRecord(
        trace_id=trace_id,
        module_name=module_name,
        event=event,
        timestamp=datetime.utcnow().isoformat(timespec="seconds"),
    )


def format_trace_key(trace_id: str, module_name: str) -> str:
    """trace key 格式化函数。"""
    return f"{module_name}:{trace_id}"

