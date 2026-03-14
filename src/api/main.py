# -*- coding: utf-8 -*-
"""FastAPI API entrypoint for the DSP agent."""
from __future__ import annotations
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
BASE_DIR = Path(__file__).resolve().parents[2]
SOURCE_DIR = BASE_DIR / 'src'
WEB_DIR = SOURCE_DIR / 'web'
if not WEB_DIR.exists():
    WEB_DIR = BASE_DIR / 'web'
ASSETS_DIR = WEB_DIR / 'assets'
APP_LOGGER = get_file_logger(project_root=BASE_DIR)
app = FastAPI(title='Engine Smart Agent Workflow API', version='0.2.0', description='LangGraph-driven orchestration and routing demo.')
app.mount('/assets', StaticFiles(directory=ASSETS_DIR), name='assets')
WORKFLOW = WorkflowService()
OBS_STORE = PostgresObservabilityStore.from_env()
SESSION_STORE = PostgresSessionStore.from_env()
SESSIONS: dict[str, dict[str, Any]] = {}
TRACE_REFERENCES: dict[str, list[dict[str, Any]]] = {}
APP_LOGGER.info('api.service.initialized', workflow_backend=WORKFLOW.backend_name, checkpointer=WORKFLOW.checkpointer_status(), session_store=SESSION_STORE.status(), observability=OBS_STORE.status(), runtime_logging=APP_LOGGER.status())

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
    return {'status': 'ok', 'workflow_backend': WORKFLOW.backend_name, 'checkpointer': WORKFLOW.checkpointer_status(), 'debug_verbose_enabled': bool(getattr(WORKFLOW, 'debug_verbose_enabled', False)), 'runtime_logging': WORKFLOW.runtime_log_status(), 'observability': OBS_STORE.status(), 'session_store': SESSION_STORE.status()}

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
def create_message(request: MessageCreateRequest) -> dict[str, Any]:
    APP_LOGGER.info('api.message.create.requested', session_id=request.session_id, content_preview=text_preview(request.content, max_chars=120))
    session = ensure_session(request.session_id)
    if len([msg for msg in session['messages'] if msg['role'] == 'user']) == 0:
        session['title'] = request.content[:24]
    user_message = build_user_message(request.content)
    session['messages'].append(user_message)
    trace_id = next_id('trace')
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
    persist_observability_turn(turn_type='message', session=session, user_query=request.content, assistant_message=assistant_message)
    APP_LOGGER.info('api.message.create.completed', session_id=session['id'], trace_id=trace_id, assistant_kind=assistant_message.get('kind', 'unknown'), assistant_status=assistant_message.get('status', 'unknown'), citation_count=len(assistant_message.get('citations', []) or []), session_message_count=len(session.get('messages', []) or []))
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

@app.get('/api/observability/summary')
def get_observability_summary(window_minutes: int=60) -> dict[str, Any]:
    APP_LOGGER.debug('api.observability.summary.requested', window_minutes=max(1, int(window_minutes)))
    summary = OBS_STORE.get_summary(window_minutes=max(1, int(window_minutes)))
    return {'observability': OBS_STORE.status(), 'summary': summary}

@app.get('/api/observability/alerts')
def get_observability_alerts(limit: int=50) -> dict[str, Any]:
    APP_LOGGER.debug('api.observability.alerts.requested', limit=max(1, min(int(limit), 200)))
    return {'observability': OBS_STORE.status(), 'items': OBS_STORE.list_alerts(limit=max(1, min(int(limit), 200)))}
