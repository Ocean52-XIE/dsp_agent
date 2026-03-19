# -*- coding: utf-8 -*-
"""FastAPI API entrypoint for the DSP agent."""
from __future__ import annotations
from contextlib import asynccontextmanager
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
from workflow.common.runtime_logging import get_file_logger
from workflow.session import PostgresSessionStore
from api.config import get_api_config, BackendVersion

BASE_DIR = Path(__file__).resolve().parents[2]
SOURCE_DIR = BASE_DIR / 'src'
WEB_DIR = SOURCE_DIR / 'web'
if not WEB_DIR.exists():
    WEB_DIR = BASE_DIR / 'web'
ASSETS_DIR = WEB_DIR / 'assets'
APP_LOGGER = get_file_logger(project_root=BASE_DIR)

# 加载 API 配置
API_CONFIG = get_api_config()

# 初始化 v1 后端 (Workflow) - 同步初始化
WORKFLOW = WorkflowService()

# v2 后端 (Agent) - 在 lifespan 中异步初始化
_AGENT_SERVICE_V2 = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 生命周期管理

    在服务启动时完成异步初始化（包括 MCP），
    在服务关闭时释放资源。
    """
    global _AGENT_SERVICE_V2

    # =========== 启动时初始化 ===========
    APP_LOGGER.info('api.lifespan.startup.begin')

    if API_CONFIG.use_v2:
        try:
            APP_LOGGER.info('api.lifespan.startup.v2_init')
            from agent.service import AgentService
            _AGENT_SERVICE_V2 = AgentService.from_env()
            await _AGENT_SERVICE_V2.ainitialize()  # 异步初始化 MCP
            APP_LOGGER.info('api.lifespan.startup.v2_ready')
        except Exception as e:
            APP_LOGGER.error('api.lifespan.startup.v2_failed', error=str(e))
            # 如果 v2 初始化失败，根据配置决定是否回退到 v1
            if not API_CONFIG.v2_fallback_to_v1:
                raise

    APP_LOGGER.info('api.lifespan.startup.complete',
                    workflow_backend=WORKFLOW.backend_name,
                    checkpointer=WORKFLOW.checkpointer_status(),
                    api_backend_version=API_CONFIG.backend_version.value)

    yield  # =========== 服务运行中 ===========

    # =========== 关闭时清理 ===========
    APP_LOGGER.info('api.lifespan.shutdown.begin')

    if _AGENT_SERVICE_V2:
        try:
            await _AGENT_SERVICE_V2.ashutdown()
            APP_LOGGER.info('api.lifespan.shutdown.v2_done')
        except Exception as e:
            APP_LOGGER.warning('api.lifespan.shutdown.v2_error', error=str(e))

    APP_LOGGER.info('api.lifespan.shutdown.complete')


# 创建 FastAPI 应用，使用 lifespan 管理生命周期
app = FastAPI(
    title='Engine Smart Agent Workflow API',
    version='0.2.0',
    description='LangGraph-driven orchestration and routing demo.',
    lifespan=lifespan,
)
app.mount('/assets', StaticFiles(directory=ASSETS_DIR), name='assets')


def _get_agent_service_v2():
    """获取 v2 Agent 服务

    注意：此函数假设 lifespan 已经完成初始化。
    如果未初始化，将抛出 RuntimeError。

    Returns:
        AgentService 实例
    """
    if _AGENT_SERVICE_V2 is None:
        raise RuntimeError(
            "AgentService 未初始化。请检查：\n"
            "1. API_BACKEND_VERSION 配置是否为 'v2' 或 'hybrid'\n"
            "2. FastAPI lifespan 是否正确配置"
        )
    return _AGENT_SERVICE_V2


# 初始化存储
OBS_STORE = PostgresObservabilityStore.from_env()
SESSION_STORE = PostgresSessionStore.from_env()
SESSIONS: dict[str, dict[str, Any]] = {}
TRACE_REFERENCES: dict[str, list[dict[str, Any]]] = {}
APP_LOGGER.info('api.service.initialized', workflow_backend=WORKFLOW.backend_name, checkpointer=WORKFLOW.checkpointer_status(), session_store=SESSION_STORE.status(), observability=OBS_STORE.status(), runtime_logging=APP_LOGGER.status(), api_backend_version=API_CONFIG.backend_version.value)

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

def text_preview(value: Any, *, max_chars: int=120) -> str:
    text = str(value or '').strip()
    if len(text) <= max_chars:
        return text
    return f'{text[:max_chars]}...'

def next_id(prefix: str) -> str:
    return f'{prefix}_{uuid4().hex}'

def create_session_record(title: str | None=None) -> dict[str, Any]:
    session_id = next_id('sess')
    created_at = now_iso()
    session = {'id': session_id, 'title': title or f"New Session {session_id.split('_')[-1]}", 'created_at': created_at, 'updated_at': created_at, 'status': 'idle', 'messages': []}
    persist_session_record(session)
    APP_LOGGER.info('api.session.created', session_id=session_id, title=text_preview(session['title'], max_chars=80))
    return session

def ensure_session(session_id: str) -> dict[str, Any]:
    session: dict[str, Any] | None = None
    if SESSION_STORE.is_active:
        session = SESSION_STORE.get_session(session_id)
    else:
        session = SESSIONS.get(session_id)
    if session is None:
        APP_LOGGER.warning('api.session.not_found', session_id=session_id)
        raise HTTPException(status_code=404, detail='Session not found')
    return session

def list_session_records(limit: int=20) -> list[dict[str, Any]]:
    safe_limit = max(1, min(int(limit), 200))
    if SESSION_STORE.is_active:
        return SESSION_STORE.list_sessions(limit=safe_limit)
    return list(SESSIONS.values())

def persist_session_record(session: dict[str, Any]) -> None:
    if SESSION_STORE.is_active:
        SESSION_STORE.save_session(session)
        return
    SESSIONS[str(session['id'])] = session

def summarize_session(session: dict[str, Any]) -> dict[str, Any]:
    preview = ''
    for message in reversed(session['messages']):
        if message['role'] == 'user':
            preview = message['content'][:72]
            break
    return {'id': session['id'], 'title': session['title'], 'updated_at': session['updated_at'], 'status': session['status'], 'last_user_preview': preview, 'message_count': len(session['messages'])}

def serialize_session(session: dict[str, Any]) -> dict[str, Any]:
    return {'id': session['id'], 'title': session['title'], 'created_at': session['created_at'], 'updated_at': session['updated_at'], 'status': session['status'], 'messages': session['messages']}

def build_user_message(content: str) -> dict[str, Any]:
    return {'id': next_id('msg'), 'role': 'user', 'kind': 'user_input', 'intent': None, 'status': 'submitted', 'content': content.strip(), 'created_at': now_iso(), 'trace_id': None, 'citations': [], 'analysis': None, 'actions': [], 'debug': {}}

def materialize_assistant_message(payload: dict[str, Any]) -> dict[str, Any]:
    message = dict(payload)
    message['id'] = next_id('msg')
    message['created_at'] = now_iso()
    return message

def persist_observability_turn(*, turn_type: str, session: dict[str, Any], user_query: str, assistant_message: dict[str, Any]) -> None:
    try:
        OBS_STORE.record_turn(turn_type=turn_type, session_id=session['id'], trace_id=str(assistant_message.get('trace_id', '') or ''), message_id=str(assistant_message.get('id', '') or ''), user_query=user_query, assistant_message=assistant_message)
    except Exception as exc:
        APP_LOGGER.warning('api.observability.record_turn_failed', turn_type=turn_type, session_id=session.get('id', ''), trace_id=str(assistant_message.get('trace_id', '') or ''), error_type=type(exc).__name__)
        return

def find_message(message_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if SESSION_STORE.is_active:
        result = SESSION_STORE.find_message(message_id)
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
    APP_LOGGER.debug('api.health.checked')
    health_info = {'status': 'ok', 'workflow_backend': WORKFLOW.backend_name, 'checkpointer': WORKFLOW.checkpointer_status(), 'debug_verbose_enabled': bool(getattr(WORKFLOW, 'debug_verbose_enabled', False)), 'runtime_logging': WORKFLOW.runtime_log_status(), 'observability': OBS_STORE.status(), 'session_store': SESSION_STORE.status(), 'api_backend_version': API_CONFIG.backend_version.value, 'api_debug_verbose': API_CONFIG.debug_verbose}
    if API_CONFIG.use_v2:
        try:
            agent_service = _get_agent_service_v2()
            health_info['agent_service'] = agent_service.get_stats() if agent_service else None
        except Exception as e:
            health_info['agent_service_error'] = str(e)
    return health_info

@app.get('/api/sessions')
def list_sessions(limit: int=20) -> dict[str, list[dict[str, Any]]]:
    safe_limit = max(1, min(int(limit), 200))
    items = sorted((summarize_session(session) for session in list_session_records(limit=safe_limit)), key=lambda item: item['updated_at'], reverse=True)[:safe_limit]
    APP_LOGGER.debug('api.session.list', limit=safe_limit, returned=len(items))
    return {'items': items}

@app.post('/api/sessions')
def create_session(request: SessionCreateRequest) -> dict[str, Any]:
    APP_LOGGER.info('api.session.create.requested', title=text_preview(request.title or '', max_chars=80))
    session = create_session_record(request.title)
    return {'session': serialize_session(session), 'summary': summarize_session(session)}

@app.get('/api/sessions/{session_id}')
def get_session(session_id: str) -> dict[str, Any]:
    APP_LOGGER.debug('api.session.get', session_id=session_id)
    session = ensure_session(session_id)
    return {'session': serialize_session(session)}

@app.post('/api/messages')
async def create_message(request: MessageCreateRequest) -> dict[str, Any]:
    """创建消息

    根据 API_BACKEND_VERSION 配置选择后端：
    - v1: 使用 Workflow (WORKFLOW.run_user_message)
    - v2: 使用 Agent (AgentService.arun)
    """
    APP_LOGGER.info('api.message.create.requested', session_id=request.session_id, content_preview=text_preview(request.content, max_chars=120))
    session = ensure_session(request.session_id)
    if len([msg for msg in session['messages'] if msg['role'] == 'user']) == 0:
        session['title'] = request.content[:24]
    user_message = build_user_message(request.content)
    session['messages'].append(user_message)
    trace_id = next_id('trace')

    # 根据配置选择后端
    if API_CONFIG.use_v2:
        # 使用 v2 Agent 后端
        APP_LOGGER.info('api.message.using_v2_backend', session_id=session['id'], trace_id=trace_id)
        try:
            agent_service = _get_agent_service_v2()
            response = await agent_service.arun(
                user_query=request.content,
                session_id=session['id'],
                trace_id=trace_id,
                history=session['messages'],
            )
            workflow_payload = {
                'role': 'assistant',
                'kind': response.kind or 'agent_v2',
                'intent': response.intent or 'agent_v2',
                'status': response.status or 'completed',
                'content': response.content,
                'trace_id': trace_id,
                'citations': response.citations or [],
                'analysis': response.analysis or {},
                'actions': [],
                'debug': response.debug or {},
            }
        except Exception as exc:
            APP_LOGGER.exception('api.message.v2_failed', session_id=session.get('id', ''), trace_id=trace_id, error_type=type(exc).__name__)
            # 如果配置了回退到 v1
            if API_CONFIG.v2_fallback_to_v1:
                APP_LOGGER.info('api.message.fallback_to_v1', session_id=session.get('id', ''), trace_id=trace_id)
                workflow_payload = WORKFLOW.run_user_message(session_id=session['id'], trace_id=trace_id, user_query=request.content, history=session['messages'])
            else:
                raise
    else:
        # 使用 v1 Workflow 后端
        APP_LOGGER.info('api.message.using_v1_backend', session_id=session['id'], trace_id=trace_id)
        try:
            workflow_payload = WORKFLOW.run_user_message(session_id=session['id'], trace_id=trace_id, user_query=request.content, history=session['messages'])
        except Exception as exc:
            APP_LOGGER.exception('api.message.create.failed', session_id=session.get('id', ''), trace_id=trace_id, error_type=type(exc).__name__)
            raise

    assistant_message = materialize_assistant_message(workflow_payload)
    TRACE_REFERENCES[trace_id] = assistant_message['citations']
    session['messages'].append(assistant_message)
    session['status'] = assistant_message['status']
    session['updated_at'] = now_iso()
    persist_session_record(session)
    turn_type = 'v2_message' if API_CONFIG.use_v2 else 'message'
    persist_observability_turn(turn_type=turn_type, session=session, user_query=request.content, assistant_message=assistant_message)
    APP_LOGGER.info('api.message.create.completed', session_id=session['id'], trace_id=trace_id, assistant_kind=assistant_message.get('kind', 'unknown'), assistant_status=assistant_message.get('status', 'unknown'), citation_count=len(assistant_message.get('citations', []) or []), session_message_count=len(session.get('messages', []) or []), backend_version=API_CONFIG.backend_version.value)
    return {'session': serialize_session(session), 'summary': summarize_session(session), 'assistant_message_id': assistant_message['id']}

@app.get('/api/references/{trace_id}')
def get_references(trace_id: str) -> dict[str, Any]:
    references = TRACE_REFERENCES.get(trace_id)
    if references is None:
        APP_LOGGER.warning('api.references.not_found', trace_id=trace_id)
        raise HTTPException(status_code=404, detail='Trace not found')
    APP_LOGGER.debug('api.references.fetched', trace_id=trace_id, count=len(references))
    return {'trace_id': trace_id, 'items': references}

@app.post('/api/messages/{message_id}/feedback')
def create_message_feedback(message_id: str, request: MessageFeedbackRequest) -> dict[str, Any]:
    APP_LOGGER.info('api.feedback.requested', message_id=message_id, helpful=bool(request.helpful), reason_tag=text_preview(request.reason_tag, max_chars=48), rating=request.rating)
    session, message = find_message(message_id)
    if message.get('role') != 'assistant':
        APP_LOGGER.warning('api.feedback.invalid_role', message_id=message_id, role=message.get('role', ''))
        raise HTTPException(status_code=400, detail='Only assistant message can receive feedback')
    trace_id = str(message.get('trace_id', '') or '')
    OBS_STORE.record_feedback(session_id=session['id'], trace_id=trace_id, message_id=message_id, helpful=bool(request.helpful), reason_tag=request.reason_tag.strip(), rating=request.rating, comment=request.comment.strip(), payload={'kind': message.get('kind'), 'intent': message.get('intent'), 'status': message.get('status')})
    message['feedback'] = {'helpful': bool(request.helpful), 'reason_tag': request.reason_tag.strip(), 'rating': request.rating, 'comment': request.comment.strip(), 'updated_at': now_iso()}
    session['updated_at'] = now_iso()
    persist_session_record(session)
    APP_LOGGER.info('api.feedback.completed', message_id=message_id, session_id=session.get('id', ''))
    return {'ok': True, 'message_id': message_id}

# =============================================================================
# V2 API Endpoints (Agent 架构)
# =============================================================================

class V2MessageCreateRequest(BaseModel):
    """V2 消息创建请求"""
    session_id: str
    content: str = Field(min_length=1, max_length=4000)

@app.post('/v2/messages')
async def create_v2_message(request: V2MessageCreateRequest) -> dict[str, Any]:
    """创建消息 (使用 V2 Agent 架构)

    使用新的 Agent 架构处理消息：
    - 动态工具调用循环
    - Skill 按需加载
    - MCP 工具集成
    """
    APP_LOGGER.info(
        'api.v2.message.create.requested',
        session_id=request.session_id,
        content_preview=text_preview(request.content, max_chars=120),
    )

    session = ensure_session(request.session_id)

    # 首次消息时设置标题
    if len([msg for msg in session['messages'] if msg['role'] == 'user']) == 0:
        session['title'] = request.content[:24]

    # 构建用户消息
    user_message = build_user_message(request.content)
    session['messages'].append(user_message)

    trace_id = next_id('trace')

    try:
        # 使用 V2 Agent 服务处理
        agent_service = _get_agent_service_v2()
        response = await agent_service.arun(
            user_query=request.content,
            session_id=session['id'],
            trace_id=trace_id,
            history=session['messages'],
        )

        # 将 V2 响应转换为兼容格式
        workflow_payload = {
            'role': 'assistant',
            'kind': response.kind or 'agent_v2',
            'intent': response.intent or 'agent_v2',
            'status': response.status or 'completed',
            'content': response.content,
            'trace_id': trace_id,
            'citations': response.citations or [],
            'analysis': response.analysis or {},
            'actions': [],
            'debug': response.debug or {},
        }

    except Exception as exc:
        APP_LOGGER.exception(
            'api.v2.message.create.failed',
            session_id=session.get('id', ''),
            trace_id=trace_id,
            error_type=type(exc).__name__,
        )

        # 如果配置了回退到 v1
        if API_CONFIG.v2_fallback_to_v1:
            APP_LOGGER.info(
                'api.v2.message.fallback_to_v1',
                session_id=session.get('id', ''),
                trace_id=trace_id,
            )
            workflow_payload = WORKFLOW.run_user_message(
                session_id=session['id'],
                trace_id=trace_id,
                user_query=request.content,
                history=session['messages'],
            )
        else:
            raise

    assistant_message = materialize_assistant_message(workflow_payload)
    TRACE_REFERENCES[trace_id] = assistant_message['citations']
    session['messages'].append(assistant_message)
    session['status'] = assistant_message['status']
    session['updated_at'] = now_iso()
    persist_session_record(session)
    persist_observability_turn(
        turn_type='v2_message',
        session=session,
        user_query=request.content,
        assistant_message=assistant_message,
    )

    APP_LOGGER.info(
        'api.v2.message.create.completed',
        session_id=session['id'],
        trace_id=trace_id,
        assistant_kind=assistant_message.get('kind', 'unknown'),
        assistant_status=assistant_message.get('status', 'unknown'),
        citation_count=len(assistant_message.get('citations', []) or []),
        session_message_count=len(session.get('messages', []) or []),
    )

    return {
        'session': serialize_session(session),
        'summary': summarize_session(session),
        'assistant_message_id': assistant_message['id'],
    }

@app.get('/api/config')
def get_api_config_info() -> dict[str, Any]:
    """获取 API 配置信息"""
    return {
        'backend_version': API_CONFIG.backend_version.value,
        'debug_verbose': API_CONFIG.debug_verbose,
        'v2_fallback_to_v1': API_CONFIG.v2_fallback_to_v1,
        'endpoints': {
            'v1': '/api/messages',
            'v2': '/v2/messages',
        },
    }

@app.get('/api/observability/summary')
def get_observability_summary(window_minutes: int=60) -> dict[str, Any]:
    APP_LOGGER.debug('api.observability.summary.requested', window_minutes=max(1, int(window_minutes)))
    summary = OBS_STORE.get_summary(window_minutes=max(1, int(window_minutes)))
    return {'observability': OBS_STORE.status(), 'summary': summary}

@app.get('/api/observability/alerts')
def get_observability_alerts(limit: int=50) -> dict[str, Any]:
    APP_LOGGER.debug('api.observability.alerts.requested', limit=max(1, min(int(limit), 200)))
    return {'observability': OBS_STORE.status(), 'items': OBS_STORE.list_alerts(limit=max(1, min(int(limit), 200)))}
