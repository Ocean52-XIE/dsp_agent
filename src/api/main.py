# -*- coding: utf-8 -*-
"""FastAPI entrypoint for the new deep-agent stack."""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, AsyncGenerator
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent import DeepAgentService
from api.message_mapper import to_assistant_message
from observability import PostgresObservabilityStore
from log import get_file_logger, setup_global_logging
from session import (
    ConversationSummarizer,
    PostgresSessionStore,
    build_conversation_memory,
    default_conversation_memory,
    merge_summary_memory_updates,
)

BASE_DIR = Path(__file__).resolve().parents[2]
SOURCE_DIR = BASE_DIR / 'src'
WEB_DIR = SOURCE_DIR / 'web'
if not WEB_DIR.exists():
    WEB_DIR = BASE_DIR / 'web'
ASSETS_DIR = WEB_DIR / 'assets'

# 设置全局日志配置
setup_global_logging(BASE_DIR)
APP_LOGGER = get_file_logger(project_root=BASE_DIR)

AGENT_SERVICE: DeepAgentService
OBS_STORE: PostgresObservabilityStore
SESSION_STORE: PostgresSessionStore
SESSION_SUMMARIZER: ConversationSummarizer
SESSIONS: dict[str, dict[str, Any]] = {}
TRACE_REFERENCES: dict[str, list[dict[str, Any]]] = {}

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI 生命周期管理器

    使用 async context manager 管理应用生命周期：
    1. 启动时：初始化检索器、MCP、领域配置
    2. 创建 DeepAgentService
    3. 关闭时：清理资源
    """
    global AGENT_SERVICE, OBS_STORE, SESSION_STORE, SESSION_SUMMARIZER

    # 启动时初始化
    startup_started_at = perf_counter()
    runtime_log_status = APP_LOGGER.status()
    APP_LOGGER.info(
        'init.startup.begin',
        project_root=str(BASE_DIR),
        log_path=runtime_log_status.get('path'),
        log_level=runtime_log_status.get('level'),
    )

    # 使用异步初始化（确保 MCP 等异步组件在正确的上下文中初始化）
    from init import initialize_async
    init_summary = await initialize_async(
        project_root=BASE_DIR,
        enable_mcp=True,
        enable_retrievers=True,
    )

    AGENT_SERVICE = await DeepAgentService.create_async(project_root=BASE_DIR)
    agent_summary = AGENT_SERVICE.startup_summary()
    SESSION_SUMMARIZER = ConversationSummarizer.create(project_root=BASE_DIR)

    # 初始化存储
    APP_LOGGER.info('init.observability.begin')
    observability_started_at = perf_counter()
    OBS_STORE = await PostgresObservabilityStore.create_from_env()
    observability_status = OBS_STORE.status()
    APP_LOGGER.info(
        'init.observability.completed',
        active=observability_status.get('active', False),
        schema=observability_status.get('schema'),
        dsn_configured=observability_status.get('dsn_configured', False),
        reason=observability_status.get('reason'),
        init_error=observability_status.get('init_error'),
        latency_ms=int((perf_counter() - observability_started_at) * 1000),
    )
    APP_LOGGER.info('init.session_store.begin')
    session_store_started_at = perf_counter()
    SESSION_STORE = await PostgresSessionStore.create_from_env()
    session_store_status = SESSION_STORE.status()
    APP_LOGGER.info(
        'init.session_store.completed',
        active=session_store_status.get('active', False),
        schema=session_store_status.get('schema'),
        dsn_configured=session_store_status.get('dsn_configured', False),
        reason=session_store_status.get('reason'),
        init_error=session_store_status.get('init_error'),
        latency_ms=int((perf_counter() - session_store_started_at) * 1000),
    )
    domain_summary = dict(init_summary.get('domain', {}) or {})
    mcp_summary = dict(init_summary.get('mcp', {}) or {})
    retriever_summary = dict(init_summary.get('retrievers', {}) or {})
    wiki_summary = dict(retriever_summary.get('wiki', {}) or {})
    code_summary = dict(retriever_summary.get('code', {}) or {})
    APP_LOGGER.info(
        'init.startup.completed',
        domain=domain_summary.get('domain', ''),
        profile_id=domain_summary.get('profile_id', ''),
        display_name=domain_summary.get('display_name', ''),
        profile_path=domain_summary.get('profile_path', ''),
        skills=agent_summary.get('skills', []) or domain_summary.get('skills', []),
        mcp_servers=mcp_summary.get('servers', []),
        mcp_tools=mcp_summary.get('tools', []),
        wiki_dir=wiki_summary.get('wiki_dir', ''),
        wiki_file_count=wiki_summary.get('file_count', 0),
        wiki_chunk_count=wiki_summary.get('chunk_count', 0),
        wiki_embedding_model=wiki_summary.get('embedding_model'),
        wiki_reranker_model=wiki_summary.get('reranker_model'),
        code_dirs=code_summary.get('code_dirs', []),
        code_indexed_file_count=code_summary.get('indexed_file_count', 0),
        code_parent_chunk_count=code_summary.get('parent_chunk_count', 0),
        code_child_chunk_count=code_summary.get('child_chunk_count', 0),
        code_embedding_model=code_summary.get('embedding_model'),
        code_reranker_model=code_summary.get('reranker_model'),
        llm_model=agent_summary.get('llm_model', ''),
        agent_tools=agent_summary.get('agent_tools', []),
        checkpointer_backend=agent_summary.get('checkpointer_backend', ''),
        checkpointer_status=agent_summary.get('checkpointer_status', ''),
        checkpointer_reason=agent_summary.get('checkpointer_reason'),
        checkpointer_fallback=agent_summary.get('checkpointer_fallback', False),
        checkpointer_fallback_from=agent_summary.get('checkpointer_fallback_from'),
        checkpointer_fallback_to=agent_summary.get('checkpointer_fallback_to'),
        session_store_active=session_store_status.get('active', False),
        session_store_schema=session_store_status.get('schema'),
        session_store_reason=session_store_status.get('reason'),
        observability_active=observability_status.get('active', False),
        observability_schema=observability_status.get('schema'),
        observability_reason=observability_status.get('reason'),
        log_path=runtime_log_status.get('path'),
        log_level=runtime_log_status.get('level'),
        startup_total_ms=int((perf_counter() - startup_started_at) * 1000),
    )

    # Yield 控制权给应用
    yield

    # 关闭时清理资源
    APP_LOGGER.info('api.lifespan.shutdown.begin')

    # 关闭 DeepAgentService 资源
    try:
        await AGENT_SERVICE.aclose()
        APP_LOGGER.info('api.lifespan.shutdown.agent.closed')
    except Exception as exc:
        APP_LOGGER.warning(
            'api.lifespan.shutdown.agent.close_failed',
            error_type=type(exc).__name__,
        )
    try:
        await OBS_STORE.aclose()
        APP_LOGGER.info('api.lifespan.shutdown.observability.closed')
    except Exception as exc:
        APP_LOGGER.warning(
            'api.lifespan.shutdown.observability.close_failed',
            error_type=type(exc).__name__,
        )
    try:
        await SESSION_STORE.aclose()
        APP_LOGGER.info('api.lifespan.shutdown.session_store.closed')
    except Exception as exc:
        APP_LOGGER.warning(
            'api.lifespan.shutdown.session_store.close_failed',
            error_type=type(exc).__name__,
        )

    APP_LOGGER.info('api.lifespan.shutdown.complete')


# 创建 FastAPI 应用（使用 lifespan）
app = FastAPI(
    title='Engine Smart Agent Workflow API',
    version='0.2.0',
    description='Deep Agents-driven orchestration demo.',
    lifespan=lifespan,
)
app.mount('/assets', StaticFiles(directory=ASSETS_DIR), name='assets')


class SessionCreateRequest(BaseModel):
    title: str | None = None


class MessageCreateRequest(BaseModel):
    session_id: str
    content: str = Field(min_length=1, max_length=4000)


class MessageFeedbackRequest(BaseModel):
    helpful: bool
    reason_tag: str = Field(default='', max_length=64)
    rating: int | None = Field(default=None, ge=1, le=5)
    comment: str = Field(default='', max_length=2000)


def now_iso() -> str:
    return datetime.now().isoformat(timespec='seconds')


def text_preview(value: Any, *, max_chars: int = 120) -> str:
    text = str(value or '').strip()
    if len(text) <= max_chars:
        return text
    return f'{text[:max_chars]}...'


def next_id(prefix: str) -> str:
    return f'{prefix}_{uuid4().hex}'


async def create_session_record(title: str | None = None) -> dict[str, Any]:
    session_id = next_id('sess')
    created_at = now_iso()
    session = {
        'id': session_id,
        'title': title or f"New Session {session_id.split('_')[-1]}",
        'created_at': created_at,
        'updated_at': created_at,
        'status': 'idle',
        'messages': [],
        'conversation_summary': '',
        'conversation_summary_updated_at': None,
        'conversation_memory': default_conversation_memory(),
        'conversation_memory_updated_at': None,
    }
    await persist_session_record(session)
    APP_LOGGER.info('api.session.created', session_id=session_id, title=text_preview(session['title'], max_chars=80))
    return session


async def ensure_session(session_id: str) -> dict[str, Any]:
    session: dict[str, Any] | None = None
    if SESSION_STORE.is_active:
        session = await SESSION_STORE.get_session(session_id)
    else:
        session = SESSIONS.get(session_id)
    if session is None:
        APP_LOGGER.warning('api.session.not_found', session_id=session_id)
        raise HTTPException(status_code=404, detail='Session not found')
    return session


async def list_session_records(limit: int = 20) -> list[dict[str, Any]]:
    safe_limit = max(1, min(int(limit), 200))
    if SESSION_STORE.is_active:
        return await SESSION_STORE.list_sessions(limit=safe_limit)
    return list(SESSIONS.values())


async def persist_session_record(session: dict[str, Any]) -> None:
    if SESSION_STORE.is_active:
        await SESSION_STORE.save_session(session)
        return
    SESSIONS[str(session['id'])] = session


def summarize_session(session: dict[str, Any]) -> dict[str, Any]:
    preview = ''
    for message in reversed(session['messages']):
        if message['role'] == 'user':
            preview = message['content'][:72]
            break
    conversation_memory = session.get('conversation_memory')
    has_conversation_memory = False
    if isinstance(conversation_memory, dict):
        has_conversation_memory = any(
            bool(value)
            for value in conversation_memory.values()
        )
    return {
        'id': session['id'],
        'title': session['title'],
        'updated_at': session['updated_at'],
        'status': session['status'],
        'last_user_preview': preview,
        'message_count': len(session['messages']),
        'has_conversation_summary': bool(str(session.get('conversation_summary', '') or '').strip()),
        'has_conversation_memory': has_conversation_memory,
    }


def serialize_session(session: dict[str, Any]) -> dict[str, Any]:
    return {
        'id': session['id'],
        'title': session['title'],
        'created_at': session['created_at'],
        'updated_at': session['updated_at'],
        'status': session['status'],
        'messages': session['messages'],
        'conversation_summary': str(session.get('conversation_summary', '') or ''),
        'conversation_summary_updated_at': session.get('conversation_summary_updated_at'),
        'conversation_memory': session.get('conversation_memory') or default_conversation_memory(),
        'conversation_memory_updated_at': session.get('conversation_memory_updated_at'),
    }


def build_user_message(content: str) -> dict[str, Any]:
    return {
        'id': next_id('msg'),
        'role': 'user',
        'kind': 'user_input',
        'intent': None,
        'status': 'submitted',
        'content': content.strip(),
        'created_at': now_iso(),
        'trace_id': None,
        'citations': [],
        'analysis': None,
        'actions': [],
        'debug': {}
    }


def materialize_assistant_message(payload: dict[str, Any]) -> dict[str, Any]:
    message = dict(payload)
    message['id'] = next_id('msg')
    message['created_at'] = now_iso()
    return message


async def persist_observability_turn(
    *,
    turn_type: str,
    session: dict[str, Any],
    user_query: str,
    assistant_message: dict[str, Any]
) -> None:
    try:
        await OBS_STORE.record_turn(
            turn_type=turn_type,
            session_id=session['id'],
            trace_id=str(assistant_message.get('trace_id', '') or ''),
            message_id=str(assistant_message.get('id', '') or ''),
            user_query=user_query,
            assistant_message=assistant_message
        )
    except Exception as exc:
        APP_LOGGER.warning(
            'api.observability.record_turn_failed',
            turn_type=turn_type,
            session_id=session.get('id', ''),
            trace_id=str(assistant_message.get('trace_id', '') or ''),
            error_type=type(exc).__name__
        )
        return


async def refresh_conversation_summary(session: dict[str, Any]) -> None:
    previous_summary = str(session.get('conversation_summary', '') or '').strip()
    summary_update = await SESSION_SUMMARIZER.refresh_summary(
        messages=session.get('messages', []) or [],
        previous_summary=previous_summary,
        conversation_memory=session.get('conversation_memory'),
    )
    next_summary = summary_update.summary
    if next_summary == previous_summary:
        summary_changed = False
    else:
        session['conversation_summary'] = next_summary
        session['conversation_summary_updated_at'] = now_iso()
        summary_changed = True

    previous_memory = session.get('conversation_memory')
    next_memory = merge_summary_memory_updates(previous_memory, summary_update.memory_updates)
    memory_changed = next_memory != previous_memory
    if memory_changed:
        session['conversation_memory'] = next_memory
        session['conversation_memory_updated_at'] = now_iso()

    if summary_changed or memory_changed:
        APP_LOGGER.info(
            'api.session.summary.updated',
            session_id=session.get('id', ''),
            summary_chars=len(next_summary),
            confirmed_fact_count=len(next_memory.get('confirmed_facts', []) or []),
            open_question_count=len(next_memory.get('open_questions', []) or []),
        )


def refresh_conversation_memory(
    *,
    session: dict[str, Any],
    user_query: str,
    assistant_message: dict[str, Any],
) -> None:
    previous_memory = session.get('conversation_memory')
    next_memory = build_conversation_memory(
        previous_memory=previous_memory if isinstance(previous_memory, dict) else None,
        user_query=user_query,
        assistant_message=assistant_message,
    )
    if next_memory == previous_memory:
        return
    session['conversation_memory'] = next_memory
    session['conversation_memory_updated_at'] = now_iso()
    APP_LOGGER.info(
        'api.session.memory.updated',
        session_id=session.get('id', ''),
        module_name=next_memory.get('module_name', ''),
        entity_count=len(next_memory.get('entities', []) or []),
    )


def attach_memory_debug_snapshot(
    *,
    assistant_message: dict[str, Any],
    session: dict[str, Any],
) -> None:
    debug = dict(assistant_message.get('debug') or {})
    memory = session.get('conversation_memory')
    if not isinstance(memory, dict):
        assistant_message['debug'] = debug
        return
    debug['current_topic'] = str(memory.get('current_topic', '') or '').strip()
    debug['active_issue'] = str(memory.get('active_issue', '') or '').strip()
    debug['confirmed_facts'] = list(memory.get('confirmed_facts', []) or [])
    debug['open_questions'] = list(memory.get('open_questions', []) or [])
    assistant_message['debug'] = debug


async def find_message(message_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if SESSION_STORE.is_active:
        result = await SESSION_STORE.find_message(message_id)
        if result is not None:
            return result
    else:
        for session in SESSIONS.values():
            for message in session['messages']:
                if message['id'] == message_id:
                    return (session, message)
    raise HTTPException(status_code=404, detail='Message not found')


@app.get('/')
def root() -> FileResponse:
    return FileResponse(WEB_DIR / 'index.html')


@app.get('/api/health')
def health() -> dict[str, Any]:
    """健康检查端点"""
    APP_LOGGER.debug('api.health.checked')
    return {
        'status': 'ok',
        'checkpointer': AGENT_SERVICE.checkpointer_status(),
        'debug_verbose_enabled': False,
        'runtime_logging': AGENT_SERVICE.runtime_log_status(),
        'observability': OBS_STORE.status(),
        'session_store': SESSION_STORE.status(),
        'conversation_summary': SESSION_SUMMARIZER.status(),
    }


@app.get('/api/sessions')
async def list_sessions(limit: int = 20) -> dict[str, list[dict[str, Any]]]:
    safe_limit = max(1, min(int(limit), 200))
    items = sorted(
        (summarize_session(session) for session in await list_session_records(limit=safe_limit)),
        key=lambda item: item['updated_at'],
        reverse=True
    )[:safe_limit]
    APP_LOGGER.debug('api.session.list', limit=safe_limit, returned=len(items))
    return {'items': items}


@app.post('/api/sessions')
async def create_session(request: SessionCreateRequest) -> dict[str, Any]:
    APP_LOGGER.info('api.session.create.requested', title=text_preview(request.title or '', max_chars=80))
    session = await create_session_record(request.title)
    return {'session': serialize_session(session), 'summary': summarize_session(session)}


@app.get('/api/sessions/{session_id}')
async def get_session(session_id: str) -> dict[str, Any]:
    APP_LOGGER.debug('api.session.get', session_id=session_id)
    session = await ensure_session(session_id)
    return {'session': serialize_session(session)}


@app.post('/api/messages')
async def create_message(request: MessageCreateRequest) -> dict[str, Any]:
    """创建消息（同步版本）

    使用 DeepAgentService 处理用户消息。
    """
    request_started_at = perf_counter()
    APP_LOGGER.info(
        'api.message.create.requested',
        session_id=request.session_id,
        content_preview=text_preview(request.content, max_chars=120),
    )

    session = await ensure_session(request.session_id)

    # 首条消息时，用内容设置会话标题
    if len([msg for msg in session['messages'] if msg['role'] == 'user']) == 0:
        session['title'] = request.content[:24]

    user_message = build_user_message(request.content)
    session['messages'].append(user_message)
    trace_id = next_id('trace')

    # 使用 Deep Agent 后端处理消息（同步调用）
    APP_LOGGER.info(
        'api.message.processing',
        session_id=session['id'],
        trace_id=trace_id,
    )

    try:
        turn_result = await AGENT_SERVICE.run_user_message_async(
            session_id=session['id'],
            trace_id=trace_id,
            user_query=request.content,
            history=session['messages'],
            conversation_summary=str(session.get('conversation_summary', '') or ''),
            conversation_memory=session.get('conversation_memory'),
        )
    except Exception as exc:
        APP_LOGGER.exception(
            'api.message.create.failed',
            session_id=session.get('id', ''),
            trace_id=trace_id,
            error_type=type(exc).__name__,
            latency_ms=int((perf_counter() - request_started_at) * 1000),
        )
        raise

    assistant_message = materialize_assistant_message(to_assistant_message(turn_result))
    TRACE_REFERENCES[trace_id] = assistant_message['citations']
    session['messages'].append(assistant_message)
    session['status'] = assistant_message['status']
    session['updated_at'] = now_iso()
    refresh_conversation_memory(
        session=session,
        user_query=request.content,
        assistant_message=assistant_message,
    )
    await refresh_conversation_summary(session)
    attach_memory_debug_snapshot(
        assistant_message=assistant_message,
        session=session,
    )
    await persist_session_record(session)
    await persist_observability_turn(
        turn_type='message',
        session=session,
        user_query=request.content,
        assistant_message=assistant_message,
    )
    APP_LOGGER.info(
        'api.message.create.completed',
        session_id=session['id'],
        trace_id=trace_id,
        assistant_kind=assistant_message.get('kind', 'unknown'),
        assistant_status=assistant_message.get('status', 'unknown'),
        citation_count=len(assistant_message.get('citations', []) or []),
        session_message_count=len(session.get('messages', []) or []),
        latency_ms=int((perf_counter() - request_started_at) * 1000),
    )
    return {
        'session': serialize_session(session),
        'summary': summarize_session(session),
        'assistant_message_id': assistant_message['id']
    }


@app.get('/api/references/{trace_id}')
def get_references(trace_id: str) -> dict[str, Any]:
    references = TRACE_REFERENCES.get(trace_id)
    if references is None:
        APP_LOGGER.warning('api.references.not_found', trace_id=trace_id)
        raise HTTPException(status_code=404, detail='Trace not found')
    APP_LOGGER.debug('api.references.fetched', trace_id=trace_id, count=len(references))
    return {'trace_id': trace_id, 'items': references}


@app.post('/api/messages/{message_id}/feedback')
async def create_message_feedback(message_id: str, request: MessageFeedbackRequest) -> dict[str, Any]:
    APP_LOGGER.info(
        'api.feedback.requested',
        message_id=message_id,
        helpful=bool(request.helpful),
        reason_tag=text_preview(request.reason_tag, max_chars=48),
        rating=request.rating
    )
    session, message = await find_message(message_id)
    if message.get('role') != 'assistant':
        APP_LOGGER.warning('api.feedback.invalid_role', message_id=message_id, role=message.get('role', ''))
        raise HTTPException(status_code=400, detail='Only assistant message can receive feedback')
    trace_id = str(message.get('trace_id', '') or '')
    await OBS_STORE.record_feedback(
        session_id=session['id'],
        trace_id=trace_id,
        message_id=message_id,
        helpful=bool(request.helpful),
        reason_tag=request.reason_tag.strip(),
        rating=request.rating,
        comment=request.comment.strip(),
        payload={
            'kind': message.get('kind'),
            'intent': message.get('intent'),
            'status': message.get('status')
        }
    )
    message['feedback'] = {
        'helpful': bool(request.helpful),
        'reason_tag': request.reason_tag.strip(),
        'rating': request.rating,
        'comment': request.comment.strip(),
        'updated_at': now_iso()
    }
    session['updated_at'] = now_iso()
    await persist_session_record(session)
    APP_LOGGER.info('api.feedback.completed', message_id=message_id, session_id=session.get('id', ''))
    return {'ok': True, 'message_id': message_id}


@app.get('/api/config')
def get_api_config_info() -> dict[str, Any]:
    """获取 API 配置信息"""
    return {
        'backend': 'deepagents',
        'checkpointer': AGENT_SERVICE.checkpointer_status(),
        'session_store': SESSION_STORE.status(),
        'conversation_summary': SESSION_SUMMARIZER.status(),
        'observability': OBS_STORE.status(),
    }


@app.get('/api/observability/summary')
async def get_observability_summary(window_minutes: int = 60) -> dict[str, Any]:
    APP_LOGGER.debug('api.observability.summary.requested', window_minutes=max(1, int(window_minutes)))
    summary = await OBS_STORE.get_summary(window_minutes=max(1, int(window_minutes)))
    return {'observability': OBS_STORE.status(), 'summary': summary}


@app.get('/api/observability/alerts')
async def get_observability_alerts(limit: int = 50) -> dict[str, Any]:
    APP_LOGGER.debug('api.observability.alerts.requested', limit=max(1, min(int(limit), 200)))
    return {
        'observability': OBS_STORE.status(),
        'items': await OBS_STORE.list_alerts(limit=max(1, min(int(limit), 200))),
    }
