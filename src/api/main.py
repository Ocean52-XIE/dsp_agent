# -*- coding: utf-8 -*-
"""
该模块实现 API 层逻辑，负责请求处理、参数校验与响应组装。
"""
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
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from workflow.engine import WorkflowService
from workflow.observability import PostgresObservabilityStore
from workflow.runtime_logging import get_file_logger
from workflow.session import PostgresSessionStore


# 统一计算静态资源目录，方便 FastAPI 既提供页面，又提供 API。
BASE_DIR = Path(__file__).resolve().parents[2]
SOURCE_DIR = BASE_DIR / "src"
WEB_DIR = SOURCE_DIR / "web"
if not WEB_DIR.exists():
    WEB_DIR = BASE_DIR / "web"
ASSETS_DIR = WEB_DIR / "assets"
APP_LOGGER = get_file_logger(project_root=BASE_DIR)

app = FastAPI(
    title="Engine Smart Agent Workflow API",
    version="0.2.0",
    description="LangGraph-driven orchestration and routing demo.",
)
app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")

# WorkflowService 持有 LangGraph 编译后的图，API 层只调用它，不直接参与节点细节。
WORKFLOW = WorkflowService()
OBS_STORE = PostgresObservabilityStore.from_env()
SESSION_STORE = PostgresSessionStore.from_env()
# 这里仍然使用内存态存储会话，便于快速演示；后续可替换为数据库。
SESSIONS: dict[str, dict[str, Any]] = {}
TRACE_REFERENCES: dict[str, list[dict[str, Any]]] = {}

APP_LOGGER.info(
    "api.service.initialized",
    workflow_backend=WORKFLOW.backend_name,
    session_store=SESSION_STORE.status(),
    observability=OBS_STORE.status(),
    runtime_logging=APP_LOGGER.status(),
)


class SessionCreateRequest(BaseModel):
    """
    定义`SessionCreateRequest`，用于封装相关数据结构与处理行为。
    """

    title: str | None = None


class MessageCreateRequest(BaseModel):
    """
    定义`MessageCreateRequest`，用于封装相关数据结构与处理行为。
    """

    session_id: str
    content: str = Field(min_length=1, max_length=4000)


class CodeConfirmRequest(BaseModel):
    """
    定义`CodeConfirmRequest`，用于封装相关数据结构与处理行为。
    """

    approved: bool = True


class MessageFeedbackRequest(BaseModel):
    """
    定义`MessageFeedbackRequest`，用于封装相关数据结构与处理行为。
    """

    helpful: bool
    reason_tag: str = Field(default="", max_length=64)
    rating: int | None = Field(default=None, ge=1, le=5)
    comment: str = Field(default="", max_length=2000)


def now_iso() -> str:
    """
    执行`now iso` 相关处理逻辑。
    
    返回:
        返回类型为 `str` 的处理结果。
    """
    return datetime.now().isoformat(timespec="seconds")


def text_preview(value: Any, *, max_chars: int = 120) -> str:
    """
    执行`text preview` 相关处理逻辑。
    
    参数:
        value: 输入参数，用于控制当前处理逻辑。
    
    返回:
        返回类型为 `str` 的处理结果。
    """
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}..."


def next_id(prefix: str) -> str:
    """
    执行`next id` 相关处理逻辑。
    
    参数:
        prefix: 输入参数，用于控制当前处理逻辑。
    
    返回:
        返回类型为 `str` 的处理结果。
    """
    return f"{prefix}_{uuid4().hex}"


def build_welcome_message() -> dict[str, Any]:
    """
    执行`build welcome message` 相关处理逻辑。
    
    返回:
        返回类型为 `dict[str, Any]` 的处理结果。
    """
    return {
        "id": next_id("msg"),
        "role": "assistant",
        "kind": "welcome",
        "intent": "system",
        "status": "completed",
        "content": WORKFLOW.domain_profile.welcome_message(),
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
    """
    执行`create session record` 相关处理逻辑。
    
    参数:
        title: 输入参数，用于控制当前处理逻辑。
    
    返回:
        返回类型为 `dict[str, Any]` 的处理结果。
    """
    session_id = next_id("sess")
    created_at = now_iso()
    session = {
        "id": session_id,
        "title": title or f"新会话 {session_id.split('_')[-1]}",
        "created_at": created_at,
        "updated_at": created_at,
        "status": "idle",
        "messages": [build_welcome_message()],
    }
    persist_session_record(session)
    APP_LOGGER.info(
        "api.session.created",
        session_id=session_id,
        title=text_preview(session["title"], max_chars=80),
    )
    return session


def ensure_session(session_id: str) -> dict[str, Any]:
    """
    执行`ensure session` 相关处理逻辑。
    
    参数:
        session_id: 会话标识。
    
    返回:
        返回类型为 `dict[str, Any]` 的处理结果。
    """
    session: dict[str, Any] | None = None
    if SESSION_STORE.is_active:
        session = SESSION_STORE.get_session(session_id)
    else:
        session = SESSIONS.get(session_id)
    if session is None:
        APP_LOGGER.warning("api.session.not_found", session_id=session_id)
        raise HTTPException(status_code=404, detail="Session not found")
    return session


def list_session_records(limit: int = 20) -> list[dict[str, Any]]:
    """
    执行`list session records` 相关处理逻辑。
    
    参数:
        limit: 输入参数，用于控制当前处理逻辑。
    
    返回:
        返回类型为 `list[dict[str, Any]]` 的处理结果。
    """
    safe_limit = max(1, min(int(limit), 200))
    if SESSION_STORE.is_active:
        return SESSION_STORE.list_sessions(limit=safe_limit)
    return list(SESSIONS.values())


def persist_session_record(session: dict[str, Any]) -> None:
    """
    执行`persist session record` 相关处理逻辑。
    
    参数:
        session: 输入参数，用于控制当前处理逻辑。
    
    返回:
        无返回值。
    """
    if SESSION_STORE.is_active:
        SESSION_STORE.save_session(session)
        return
    SESSIONS[str(session["id"])] = session


def summarize_session(session: dict[str, Any]) -> dict[str, Any]:
    """
    执行`summarize session` 相关处理逻辑。
    
    参数:
        session: 输入参数，用于控制当前处理逻辑。
    
    返回:
        返回类型为 `dict[str, Any]` 的处理结果。
    """
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
    """
    执行`serialize session` 相关处理逻辑。
    
    参数:
        session: 输入参数，用于控制当前处理逻辑。
    
    返回:
        返回类型为 `dict[str, Any]` 的处理结果。
    """
    return {
        "id": session["id"],
        "title": session["title"],
        "created_at": session["created_at"],
        "updated_at": session["updated_at"],
        "status": session["status"],
        "messages": session["messages"],
    }


def build_user_message(content: str) -> dict[str, Any]:
    """
    执行`build user message` 相关处理逻辑。
    
    参数:
        content: 输入参数，用于控制当前处理逻辑。
    
    返回:
        返回类型为 `dict[str, Any]` 的处理结果。
    """
    return {
        "id": next_id("msg"),
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
    """
    执行`materialize assistant message` 相关处理逻辑。
    
    参数:
        payload: 输入参数，用于控制当前处理逻辑。
    
    返回:
        返回类型为 `dict[str, Any]` 的处理结果。
    """
    message = dict(payload)
    message["id"] = next_id("msg")
    message["created_at"] = now_iso()
    return message


def persist_observability_turn(
    *,
    turn_type: str,
    session: dict[str, Any],
    user_query: str,
    assistant_message: dict[str, Any],
) -> None:
    """
    执行`persist observability turn` 相关处理逻辑。
    
    返回:
        无返回值。
    """
    try:
        OBS_STORE.record_turn(
            turn_type=turn_type,
            session_id=session["id"],
            trace_id=str(assistant_message.get("trace_id", "") or ""),
            message_id=str(assistant_message.get("id", "") or ""),
            user_query=user_query,
            assistant_message=assistant_message,
        )
    except Exception as exc:
        APP_LOGGER.warning(
            "api.observability.record_turn_failed",
            turn_type=turn_type,
            session_id=session.get("id", ""),
            trace_id=str(assistant_message.get("trace_id", "") or ""),
            error_type=type(exc).__name__,
        )
        return


def close_open_code_confirmation(session: dict[str, Any]) -> None:
    """
    执行`close open code confirmation` 相关处理逻辑。
    
    参数:
        session: 输入参数，用于控制当前处理逻辑。
    
    返回:
        无返回值。
    """
    for message in reversed(session["messages"]):
        if message.get("role") == "assistant" and message.get("status") == "confirm_code":
            message["status"] = "completed"
            message["actions"] = []
            return


def find_message(message_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    执行`find message` 相关处理逻辑。
    
    参数:
        message_id: 标识参数，用于定位上下文对象。
    
    返回:
        返回类型为 `tuple[dict[str, Any], dict[str, Any]]` 的处理结果。
    """
    if SESSION_STORE.is_active:
        result = SESSION_STORE.find_message(message_id)
        if result is not None:
            return result
    else:
        for session in SESSIONS.values():
            for message in session["messages"]:
                if message["id"] == message_id:
                    return session, message
    raise HTTPException(status_code=404, detail="Message not found")


@app.get("/")
def root() -> FileResponse:
    """
    执行`root` 相关处理逻辑。
    
    返回:
        返回类型为 `FileResponse` 的处理结果。
    """
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    """
    执行`health` 相关处理逻辑。
    
    返回:
        返回类型为 `dict[str, Any]` 的处理结果。
    """
    APP_LOGGER.debug("api.health.checked")
    return {
        "status": "ok",
        "workflow_backend": WORKFLOW.backend_name,
        "debug_verbose_enabled": bool(getattr(WORKFLOW, "debug_verbose_enabled", False)),
        "runtime_logging": WORKFLOW.runtime_log_status(),
        "observability": OBS_STORE.status(),
        "session_store": SESSION_STORE.status(),
    }


@app.get("/api/sessions")
def list_sessions(limit: int = 20) -> dict[str, list[dict[str, Any]]]:
    """
    执行`list sessions` 相关处理逻辑。
    
    参数:
        limit: 输入参数，用于控制当前处理逻辑。
    
    返回:
        返回类型为 `dict[str, list[dict[str, Any]]]` 的处理结果。
    """
    safe_limit = max(1, min(int(limit), 200))
    items = sorted(
        (summarize_session(session) for session in list_session_records(limit=safe_limit)),
        key=lambda item: item["updated_at"],
        reverse=True,
    )[:safe_limit]
    APP_LOGGER.debug("api.session.list", limit=safe_limit, returned=len(items))
    return {"items": items}


@app.post("/api/sessions")
def create_session(request: SessionCreateRequest) -> dict[str, Any]:
    """
    执行`create session` 相关处理逻辑。
    
    参数:
        request: 输入参数，用于控制当前处理逻辑。
    
    返回:
        返回类型为 `dict[str, Any]` 的处理结果。
    """
    APP_LOGGER.info("api.session.create.requested", title=text_preview(request.title or "", max_chars=80))
    session = create_session_record(request.title)
    return {"session": serialize_session(session), "summary": summarize_session(session)}


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    """
    执行`get session` 相关处理逻辑。
    
    参数:
        session_id: 会话标识。
    
    返回:
        返回类型为 `dict[str, Any]` 的处理结果。
    """
    APP_LOGGER.debug("api.session.get", session_id=session_id)
    session = ensure_session(session_id)
    return {"session": serialize_session(session)}


@app.post("/api/messages")
def create_message(request: MessageCreateRequest) -> dict[str, Any]:
    """
    执行`create message` 相关处理逻辑。
    
    参数:
        request: 输入参数，用于控制当前处理逻辑。
    
    返回:
        返回类型为 `dict[str, Any]` 的处理结果。
    """
    APP_LOGGER.info(
        "api.message.create.requested",
        session_id=request.session_id,
        content_preview=text_preview(request.content, max_chars=120),
    )
    session = ensure_session(request.session_id)
    if len([msg for msg in session["messages"] if msg["role"] == "user"]) == 0:
        # 首轮用户消息默认作为会话标题，方便左侧列表识别。
        session["title"] = request.content[:24]

    user_message = build_user_message(request.content)
    session["messages"].append(user_message)

    # 一次前端发送对应一次工作流执行，因此 trace_id 在这里生成。
    trace_id = next_id("trace")
    try:
        workflow_payload = WORKFLOW.run_user_message(
            session_id=session["id"],
            trace_id=trace_id,
            user_query=request.content,
            history=session["messages"],
        )
    except Exception as exc:
        APP_LOGGER.exception(
            "api.message.create.failed",
            session_id=session.get("id", ""),
            trace_id=trace_id,
            error_type=type(exc).__name__,
        )
        raise
    assistant_message = materialize_assistant_message(workflow_payload)
    # 引用证据按 trace_id 索引，方便右侧详情或单独接口查询。
    TRACE_REFERENCES[trace_id] = assistant_message["citations"]

    # 新一轮助手响应已经产生后，上一条等待确认的消息不应再继续保留按钮。
    # 无论本轮是继续分析、切换主题还是直接进入代码生成，都统一把旧动作收口。
    close_open_code_confirmation(session)
    session["messages"].append(assistant_message)
    session["status"] = assistant_message["status"]
    session["updated_at"] = now_iso()
    persist_session_record(session)
    persist_observability_turn(
        turn_type="message",
        session=session,
        user_query=request.content,
        assistant_message=assistant_message,
    )
    APP_LOGGER.info(
        "api.message.create.completed",
        session_id=session["id"],
        trace_id=trace_id,
        assistant_kind=assistant_message.get("kind", "unknown"),
        assistant_status=assistant_message.get("status", "unknown"),
        citation_count=len(assistant_message.get("citations", []) or []),
        session_message_count=len(session.get("messages", []) or []),
    )

    return {
        "session": serialize_session(session),
        "summary": summarize_session(session),
        "assistant_message_id": assistant_message["id"],
    }


@app.post("/api/messages/{message_id}/confirm-code")
def confirm_code_generation(message_id: str, request: CodeConfirmRequest) -> dict[str, Any]:
    """
    执行`confirm code generation` 相关处理逻辑。
    
    参数:
        message_id: 标识参数，用于定位上下文对象。
        request: 输入参数，用于控制当前处理逻辑。
    
    返回:
        返回类型为 `dict[str, Any]` 的处理结果。
    """
    APP_LOGGER.info(
        "api.confirm_code.requested",
        message_id=message_id,
        approved=bool(request.approved),
    )
    session, source_message = find_message(message_id)
    if source_message["status"] != "confirm_code":
        APP_LOGGER.warning(
            "api.confirm_code.invalid_state",
            message_id=message_id,
            current_status=source_message.get("status", ""),
        )
        raise HTTPException(status_code=400, detail="Message is not waiting for code confirmation")

    # 一旦点击过确认按钮，就先把原消息的按钮撤掉，避免前端重复提交。
    source_message["status"] = "completed"
    source_message["actions"] = []

    if not request.approved:
        # 用户选择不继续时，直接结束本轮链路。
        session["status"] = "completed"
        session["updated_at"] = now_iso()
        persist_session_record(session)
        APP_LOGGER.info(
            "api.confirm_code.declined",
            session_id=session.get("id", ""),
            message_id=message_id,
        )
        return {"session": serialize_session(session), "summary": summarize_session(session)}

    # 用户确认继续后，进入工作流的代码生成路径。
    trace_id = next_id("trace")
    try:
        workflow_payload = WORKFLOW.run_code_generation(
            session_id=session["id"],
            trace_id=trace_id,
            source_message=source_message,
            history=session["messages"],
        )
    except Exception as exc:
        APP_LOGGER.exception(
            "api.confirm_code.failed",
            session_id=session.get("id", ""),
            trace_id=trace_id,
            message_id=message_id,
            error_type=type(exc).__name__,
        )
        raise
    assistant_message = materialize_assistant_message(workflow_payload)
    TRACE_REFERENCES[trace_id] = assistant_message["citations"]

    session["messages"].append(assistant_message)
    session["status"] = assistant_message["status"]
    session["updated_at"] = now_iso()
    persist_session_record(session)
    persist_observability_turn(
        turn_type="confirm_code",
        session=session,
        user_query=str(source_message.get("content", "") or ""),
        assistant_message=assistant_message,
    )
    APP_LOGGER.info(
        "api.confirm_code.completed",
        session_id=session.get("id", ""),
        trace_id=trace_id,
        message_id=message_id,
        assistant_kind=assistant_message.get("kind", "unknown"),
        assistant_status=assistant_message.get("status", "unknown"),
        citation_count=len(assistant_message.get("citations", []) or []),
    )

    return {
        "session": serialize_session(session),
        "summary": summarize_session(session),
        "assistant_message_id": assistant_message["id"],
    }


@app.get("/api/references/{trace_id}")
def get_references(trace_id: str) -> dict[str, Any]:
    """
    执行`get references` 相关处理逻辑。
    
    参数:
        trace_id: 请求追踪标识。
    
    返回:
        返回类型为 `dict[str, Any]` 的处理结果。
    """
    references = TRACE_REFERENCES.get(trace_id)
    if references is None:
        APP_LOGGER.warning("api.references.not_found", trace_id=trace_id)
        raise HTTPException(status_code=404, detail="Trace not found")
    APP_LOGGER.debug("api.references.fetched", trace_id=trace_id, count=len(references))
    return {"trace_id": trace_id, "items": references}


@app.post("/api/messages/{message_id}/feedback")
def create_message_feedback(message_id: str, request: MessageFeedbackRequest) -> dict[str, Any]:
    """
    执行`create message feedback` 相关处理逻辑。
    
    参数:
        message_id: 标识参数，用于定位上下文对象。
        request: 输入参数，用于控制当前处理逻辑。
    
    返回:
        返回类型为 `dict[str, Any]` 的处理结果。
    """
    APP_LOGGER.info(
        "api.feedback.requested",
        message_id=message_id,
        helpful=bool(request.helpful),
        reason_tag=text_preview(request.reason_tag, max_chars=48),
        rating=request.rating,
    )
    session, message = find_message(message_id)
    if message.get("role") != "assistant":
        APP_LOGGER.warning("api.feedback.invalid_role", message_id=message_id, role=message.get("role", ""))
        raise HTTPException(status_code=400, detail="Only assistant message can receive feedback")

    trace_id = str(message.get("trace_id", "") or "")
    OBS_STORE.record_feedback(
        session_id=session["id"],
        trace_id=trace_id,
        message_id=message_id,
        helpful=bool(request.helpful),
        reason_tag=request.reason_tag.strip(),
        rating=request.rating,
        comment=request.comment.strip(),
        payload={
            "kind": message.get("kind"),
            "intent": message.get("intent"),
            "status": message.get("status"),
        },
    )
    message["feedback"] = {
        "helpful": bool(request.helpful),
        "reason_tag": request.reason_tag.strip(),
        "rating": request.rating,
        "comment": request.comment.strip(),
        "updated_at": now_iso(),
    }
    session["updated_at"] = now_iso()
    persist_session_record(session)
    APP_LOGGER.info(
        "api.feedback.completed",
        message_id=message_id,
        session_id=session.get("id", ""),
    )
    return {"ok": True, "message_id": message_id}


@app.get("/api/observability/summary")
def get_observability_summary(window_minutes: int = 60) -> dict[str, Any]:
    """
    执行`get observability summary` 相关处理逻辑。
    
    参数:
        window_minutes: 列表参数，用于承载批量输入数据。
    
    返回:
        返回类型为 `dict[str, Any]` 的处理结果。
    """
    APP_LOGGER.debug("api.observability.summary.requested", window_minutes=max(1, int(window_minutes)))
    summary = OBS_STORE.get_summary(window_minutes=max(1, int(window_minutes)))
    return {
        "observability": OBS_STORE.status(),
        "summary": summary,
    }


@app.get("/api/observability/alerts")
def get_observability_alerts(limit: int = 50) -> dict[str, Any]:
    """
    执行`get observability alerts` 相关处理逻辑。
    
    参数:
        limit: 输入参数，用于控制当前处理逻辑。
    
    返回:
        返回类型为 `dict[str, Any]` 的处理结果。
    """
    APP_LOGGER.debug("api.observability.alerts.requested", limit=max(1, min(int(limit), 200)))
    return {
        "observability": OBS_STORE.status(),
        "items": OBS_STORE.list_alerts(limit=max(1, min(int(limit), 200))),
    }
