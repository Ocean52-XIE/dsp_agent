from __future__ import annotations

"""FastAPI 接入层。

这个文件只负责三类事情：
1. 提供前端需要的 HTTP 接口；
2. 管理会话内存态、消息 ID、trace ID；
3. 把真正的智能编排委托给 WorkflowService。

也就是说，业务路由、检索、问题分析、代码生成都不应该继续往这里堆，
后续都应该下沉到 workflow 层。
"""

from datetime import datetime
from itertools import count
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from workflow.engine import WorkflowService


# 统一计算静态资源目录，方便 FastAPI 既提供页面，又提供 API。
BASE_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = BASE_DIR / "web"
ASSETS_DIR = WEB_DIR / "assets"

app = FastAPI(
    title="Engine Smart Agent Workflow API",
    version="0.2.0",
    description="LangGraph-driven orchestration and routing demo.",
)
app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")

# WorkflowService 持有 LangGraph 编译后的图，API 层只调用它，不直接参与节点细节。
WORKFLOW = WorkflowService()
# 这里仍然使用内存态存储会话，便于快速演示；后续可替换为数据库。
SESSIONS: dict[str, dict[str, Any]] = {}
TRACE_REFERENCES: dict[str, list[dict[str, Any]]] = {}
SESSION_SEQ = count(1)
MESSAGE_SEQ = count(1)
TRACE_SEQ = count(1)


class SessionCreateRequest(BaseModel):
    """创建会话请求。"""

    title: str | None = None


class MessageCreateRequest(BaseModel):
    """发送消息请求。"""

    session_id: str
    content: str = Field(min_length=1, max_length=4000)


class CodeConfirmRequest(BaseModel):
    """代码实现确认请求。"""

    approved: bool = True


def now_iso() -> str:
    """统一生成秒级时间戳字符串，便于前端直接展示。"""
    return datetime.now().isoformat(timespec="seconds")


def next_id(prefix: str, seq: count) -> str:
    """生成带前缀的递增 ID。"""
    return f"{prefix}_{next(seq):04d}"


def build_welcome_message() -> dict[str, Any]:
    """新会话默认插入一条欢迎消息，说明当前系统已经切到工作流模式。"""
    return {
        "id": next_id("msg", MESSAGE_SEQ),
        "role": "assistant",
        "kind": "welcome",
        "intent": "system",
        "status": "completed",
        "content": (
            "这里已经切换成 LangGraph 工作流入口。你现在发起的每条消息都会经过领域判定、"
            "意图路由和后续节点编排；检索、分析和代码生成节点目前仍然是 mock 实现。"
        ),
        "created_at": now_iso(),
        "trace_id": None,
        "citations": [],
        "analysis": None,
        "actions": [],
        "debug": {
            "domain_relevance": 1.0,
            "latency_ms": 0,
            "route": "welcome",
            "next_action": "await_user_input",
            "graph_backend": WORKFLOW.backend_name,
            "graph_path": ["welcome"],
        },
    }


def create_session_record(title: str | None = None) -> dict[str, Any]:
    """创建内存态会话对象。"""
    session_id = next_id("sess", SESSION_SEQ)
    created_at = now_iso()
    session = {
        "id": session_id,
        "title": title or f"新会话 {session_id.split('_')[-1]}",
        "created_at": created_at,
        "updated_at": created_at,
        "status": "idle",
        "messages": [build_welcome_message()],
    }
    SESSIONS[session_id] = session
    return session


def ensure_session(session_id: str) -> dict[str, Any]:
    """按 ID 读取会话，不存在时直接返回 404。"""
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


def summarize_session(session: dict[str, Any]) -> dict[str, Any]:
    """生成会话摘要，供左侧列表展示。"""
    preview = ""
    for message in reversed(session["messages"]):
        if message["role"] == "user":
            preview = message["content"][:72]
            break
    return {
        "id": session["id"],
        "title": session["title"],
        "updated_at": session["updated_at"],
        "status": session["status"],
        "last_user_preview": preview,
        "message_count": len(session["messages"]),
    }


def serialize_session(session: dict[str, Any]) -> dict[str, Any]:
    """把内部会话对象整理成可返回给前端的结构。"""
    return {
        "id": session["id"],
        "title": session["title"],
        "created_at": session["created_at"],
        "updated_at": session["updated_at"],
        "status": session["status"],
        "messages": session["messages"],
    }


def build_user_message(content: str) -> dict[str, Any]:
    """把前端输入包装成统一的用户消息结构。"""
    return {
        "id": next_id("msg", MESSAGE_SEQ),
        "role": "user",
        "kind": "user_input",
        "intent": None,
        "status": "submitted",
        "content": content.strip(),
        "created_at": now_iso(),
        "trace_id": None,
        "citations": [],
        "analysis": None,
        "actions": [],
        "debug": {},
    }


def materialize_assistant_message(payload: dict[str, Any]) -> dict[str, Any]:
    """给工作流产物补齐消息 ID 和时间戳。"""
    message = dict(payload)
    message["id"] = next_id("msg", MESSAGE_SEQ)
    message["created_at"] = now_iso()
    return message


def close_open_code_confirmation(session: dict[str, Any]) -> None:
    """关闭会话里最近一个仍处于等待确认状态的代码生成动作。

    多轮阶段式工作流下，用户可能不会点击按钮，而是直接继续发文字消息，例如：

    - “先不用代码”
    - “给我代码实现”
    - “那我们换个问题”

    一旦新一轮响应已经产出，就不应该让旧的确认按钮继续悬挂，否则前端会出现过期动作。
    这里选择只关闭最近一个等待确认的助手消息，保持行为可解释且范围最小。
    """
    for message in reversed(session["messages"]):
        if message.get("role") == "assistant" and message.get("status") == "confirm_code":
            message["status"] = "completed"
            message["actions"] = []
            return


def find_message(message_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """在所有会话中查找指定消息，同时返回所属会话。"""
    for session in SESSIONS.values():
        for message in session["messages"]:
            if message["id"] == message_id:
                return session, message
    raise HTTPException(status_code=404, detail="Message not found")


@app.get("/")
def root() -> FileResponse:
    """返回前端页面入口。"""
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    """健康检查，同时暴露当前工作流后端类型。"""
    return {"status": "ok", "workflow_backend": WORKFLOW.backend_name}


@app.get("/api/sessions")
def list_sessions() -> dict[str, list[dict[str, Any]]]:
    """返回会话列表，供左侧会话区渲染。"""
    items = sorted(
        (summarize_session(session) for session in SESSIONS.values()),
        key=lambda item: item["updated_at"],
        reverse=True,
    )
    return {"items": items}


@app.post("/api/sessions")
def create_session(request: SessionCreateRequest) -> dict[str, Any]:
    """创建新会话。"""
    session = create_session_record(request.title)
    return {"session": serialize_session(session), "summary": summarize_session(session)}


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    """获取指定会话的完整消息列表。"""
    session = ensure_session(session_id)
    return {"session": serialize_session(session)}


@app.post("/api/messages")
def create_message(request: MessageCreateRequest) -> dict[str, Any]:
    """处理用户输入的主入口。

    接口层只负责：
    1. 把用户消息入会话；
    2. 生成 trace_id；
    3. 调用 WorkflowService；
    4. 把工作流结果落回会话。
    """
    session = ensure_session(request.session_id)
    if len([msg for msg in session["messages"] if msg["role"] == "user"]) == 0:
        # 首轮用户消息默认作为会话标题，方便左侧列表识别。
        session["title"] = request.content[:24]

    user_message = build_user_message(request.content)
    session["messages"].append(user_message)

    # 一次前端发送对应一次工作流执行，因此 trace_id 在这里生成。
    trace_id = next_id("trace", TRACE_SEQ)
    workflow_payload = WORKFLOW.run_user_message(
        session_id=session["id"],
        trace_id=trace_id,
        user_query=request.content,
        history=session["messages"],
    )
    assistant_message = materialize_assistant_message(workflow_payload)
    # 引用证据按 trace_id 索引，方便右侧详情或单独接口查询。
    TRACE_REFERENCES[trace_id] = assistant_message["citations"]

    # 新一轮助手响应已经产生后，上一条等待确认的消息不应再继续保留按钮。
    # 无论本轮是继续分析、切换主题还是直接进入代码生成，都统一把旧动作收口。
    close_open_code_confirmation(session)
    session["messages"].append(assistant_message)
    session["status"] = assistant_message["status"]
    session["updated_at"] = now_iso()

    return {
        "session": serialize_session(session),
        "summary": summarize_session(session),
        "assistant_message_id": assistant_message["id"],
    }


@app.post("/api/messages/{message_id}/confirm-code")
def confirm_code_generation(message_id: str, request: CodeConfirmRequest) -> dict[str, Any]:
    """处理“是否需要代码实现”的确认动作。"""
    session, source_message = find_message(message_id)
    if source_message["status"] != "confirm_code":
        raise HTTPException(status_code=400, detail="Message is not waiting for code confirmation")

    # 一旦点击过确认按钮，就先把原消息的按钮撤掉，避免前端重复提交。
    source_message["status"] = "completed"
    source_message["actions"] = []

    if not request.approved:
        # 用户选择不继续时，直接结束本轮链路。
        session["status"] = "completed"
        session["updated_at"] = now_iso()
        return {"session": serialize_session(session), "summary": summarize_session(session)}

    # 用户确认继续后，进入工作流的代码生成路径。
    trace_id = next_id("trace", TRACE_SEQ)
    workflow_payload = WORKFLOW.run_code_generation(
        session_id=session["id"],
        trace_id=trace_id,
        source_message=source_message,
        history=session["messages"],
    )
    assistant_message = materialize_assistant_message(workflow_payload)
    TRACE_REFERENCES[trace_id] = assistant_message["citations"]

    session["messages"].append(assistant_message)
    session["status"] = assistant_message["status"]
    session["updated_at"] = now_iso()

    return {
        "session": serialize_session(session),
        "summary": summarize_session(session),
        "assistant_message_id": assistant_message["id"],
    }


@app.get("/api/references/{trace_id}")
def get_references(trace_id: str) -> dict[str, Any]:
    """根据 trace_id 查询本轮工作流实际返回的引用证据。"""
    references = TRACE_REFERENCES.get(trace_id)
    if references is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return {"trace_id": trace_id, "items": references}
