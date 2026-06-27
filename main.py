from __future__ import annotations

import asyncio
import mimetypes
import time
import uuid
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from src.agents.main_agent import MainAgent
from src.bootstrap import bootstrap_registries
from src.config import settings
from src.core.backend import resolve_user_context
from src.core.context import RuntimeContext
from src.core.oss import upload_fileobj_to_oss
from src.core.registry import agent_registry, tool_registry
from src.repositories.agent_memory import agent_memory_repository
from src.security.chat_buffer import chat_redis_buffer
from src.security.sensitive_matcher import sensitive_matcher
from src.security.sensitive_mq import ChatAuditMessage, chat_audit_mq_service
from src.tools.frontend import FrontendActionTool

app = FastAPI(title="YAAI Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=".*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConnectionState:
    def __init__(self, websocket: WebSocket, context: RuntimeContext) -> None:
        self.websocket = websocket
        self.context = context
        self.last_pong_at = time.monotonic()
        self.closed = False
        self.send_lock = asyncio.Lock()
        self.action_results: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self.interrupted_request_ids: set[str] = set()
        self.message_tasks: set[asyncio.Task[None]] = set()


connections: dict[str, ConnectionState] = {}
chat_audit_consumer_task: asyncio.Task | None = None


def _cookie_token(websocket: WebSocket) -> str | None:
    return websocket.cookies.get("yaai")


async def _http_user_context(request: Request) -> dict[str, Any]:
    token = request.cookies.get("yaai")
    user_context = await resolve_user_context(token)
    if not user_context.get("authenticated"):
        raise HTTPException(status_code=401, detail="请先登录后使用 YAAI 助手")
    return user_context


async def _send(state: ConnectionState, event_type: str, payload: dict[str, Any] | None = None) -> None:
    async with state.send_lock:
        await state.websocket.send_json({"eventType": event_type, "payload": payload or {}})


async def _publish_chat_audit(message: ChatAuditMessage) -> None:
    try:
        await chat_audit_mq_service.publish(message)
    except Exception:
        # MQ is a safety side channel. Do not break the user request if RabbitMQ
        # is temporarily unavailable; production logging can be attached later.
        pass


async def _insert_memory_safe(**kwargs: Any) -> int | None:
    try:
        return await agent_memory_repository.insert_memory(**kwargs)
    except Exception:
        return None


async def _ensure_session_safe(state: ConnectionState, content: str) -> None:
    if state.context.session_id is not None:
        return
    try:
        title = content.strip().replace("\n", " ")[:200] or "新会话"
        state.context.session_id = await agent_memory_repository.create_session(
            user_id=state.context.user_id,
            title=title,
        )
        await _send(state, "session.created", {"sessionId": state.context.session_id, "title": title})
    except Exception:
        state.context.session_id = None


def _bind_page_context(context: RuntimeContext, message: dict[str, Any]) -> None:
    page = message.get("page")
    if not isinstance(page, dict):
        page = {}
    context.page = page
    context.current_page = (
        str(page.get("currentPage") or page.get("current_page") or page.get("routeName") or page.get("name") or "").strip()
        or None
    )
    context.page_type = str(page.get("pageType") or page.get("page_type") or page.get("type") or "").strip() or None
    context.page_description = str(page.get("description") or "").strip() or None
    if not context.current_page:
        path = str(page.get("path") or "").strip("/")
        context.current_page = path.replace("/", "_") if path else "home"
    if not context.page_type:
        context.page_type = context.current_page


async def _handle_sensitive_reject(state: ConnectionState, request_id: str) -> None:
    """同步拒绝命中敏感词的消息，统一回复固定文案。"""
    reply = "对不起，这个问题我暂时无法回答"
    await _send(state, "assistant.message.start", {"requestId": request_id})
    for ch in reply:
        await _send(state, "assistant.message.delta", {"requestId": request_id, "delta": ch})
    await _send(state, "assistant.message.done", {"requestId": request_id})


async def _handle_sensitive_hit(message: ChatAuditMessage, word: str) -> None:
    state = connections.get(message.connection_id)
    if state is None:
        return
    state.interrupted_request_ids.add(message.request_id)
    await _send(
        state,
        "message.revoke",
        {
            "requestId": message.request_id,
            "messageId": message.assistant_message_id or message.message_id,
            "role": "assistant",
            "reason": "命中敏感词，消息已撤回",
        },
    )


async def _flush_request_to_mysql(request_id: str) -> None:
    if await chat_redis_buffer.is_rejected(request_id):
        await chat_redis_buffer.clear(request_id)
        return
    events = await chat_redis_buffer.load_events(request_id)
    if not events:
        return

    session_id = events[0].get("session_id")
    user_id = events[0].get("user_id")
    if session_id is None:
        title = next((event.get("content", "") for event in events if event.get("stage") == "user_input"), "新会话")
        try:
            session_id = await agent_memory_repository.create_session(user_id=user_id, title=str(title).strip().replace("\n", " ")[:200] or "新会话")
        except Exception:
            session_id = None

    for event in events:
        stage = event.get("stage")
        if stage == "user_input":
            role = "user"
        elif stage == "final_output":
            role = "assistant"
        else:
            role = "sub_agent"
        await _insert_memory_safe(
            session_id=session_id,
            user_id=user_id,
            role=role,
            content=str(event.get("content") or ""),
            attachments=event.get("attachments") or None,
        )
    await chat_redis_buffer.clear(request_id)


async def _run_chat_audit_consumer() -> None:
    while True:
        try:
            await chat_audit_mq_service.consume(_handle_sensitive_hit, _flush_request_to_mysql)
        except asyncio.CancelledError:
            raise
        except Exception:
            await asyncio.sleep(5)


async def _send_frontend_action(context: RuntimeContext, action: str, payload: dict[str, Any]) -> dict[str, Any]:
    state = connections.get(context.connection_id)
    if state is None:
        return {"success": False, "error": "connection not found"}

    action_id = f"act_{uuid.uuid4().hex}"
    loop = asyncio.get_running_loop()
    future: asyncio.Future[dict[str, Any]] = loop.create_future()
    state.action_results[action_id] = future
    action_payload = dict(payload)
    action_payload.setdefault("requiresConfirm", True)
    action_payload.setdefault("platform", context.platform)
    await _send(state, "frontend.action", {"actionId": action_id, "action": action, "payload": action_payload})

    try:
        return await asyncio.wait_for(future, timeout=60)
    except asyncio.TimeoutError:
        state.action_results.pop(action_id, None)
        return {"success": False, "error": "frontend action timeout"}


def _bind_frontend_tools() -> None:
    for name in ["frontend.navigate", "frontend.fill", "frontend.highlight"]:
        try:
            tool = tool_registry.get(name).handler
        except KeyError:
            continue
        if isinstance(tool, FrontendActionTool):
            tool.bind_sender(_send_frontend_action)


bootstrap_registries()
_bind_frontend_tools()


async def _ping_loop(state: ConnectionState) -> None:
    while not state.closed:
        await asyncio.sleep(settings.ping_interval_seconds)
        if state.closed:
            return
        if time.monotonic() - state.last_pong_at > settings.pong_timeout_seconds:
            await state.websocket.close(code=4000, reason="pong timeout")
            state.closed = True
            return
        await _send(state, "ping", {"ts": int(time.time() * 1000)})


async def _handle_user_message(state: ConnectionState, message: dict[str, Any]) -> None:
    content = str(message.get("content") or "").strip()
    attachments = message.get("attachments")
    state.context.attachments = attachments if isinstance(attachments, list) else []
    _bind_page_context(state.context, message)
    if not content and not state.context.attachments:
        return

    request_id = str(message.get("requestId") or uuid.uuid4())
    user_message_id = str(message.get("userMessageId") or f"user_{uuid.uuid4().hex}")
    assistant_message_id = str(message.get("assistantMessageId") or f"assistant_{uuid.uuid4().hex}")
    state.context.request_id = request_id
    state.context.assistant_message_id = assistant_message_id

    # 同步敏感词检测 —— 用户输入命中直接拒绝
    if sensitive_matcher.find_first(content):
        await _handle_sensitive_reject(state, request_id)
        return

    await _publish_chat_audit(
        ChatAuditMessage(
            check_id=f"chk_{uuid.uuid4().hex}",
            connection_id=state.context.connection_id,
            request_id=request_id,
            message_id=user_message_id,
            role="user",
            stage="user_input",
            content=content,
            session_id=state.context.session_id,
            user_id=state.context.user_id,
            attachments=state.context.attachments,
            assistant_message_id=assistant_message_id,
        )
    )

    await _send(state, "assistant.message.start", {"requestId": request_id})
    agent = agent_registry.get("main").handler
    if not isinstance(agent, MainAgent):
        await _send(state, "error", {"requestId": request_id, "message": "main agent unavailable"})
        return

    assistant_content_parts: list[str] = []
    async for delta in agent.stream_reply(state.context, content):
        if request_id in state.interrupted_request_ids:
            await _send(state, "assistant.message.done", {"requestId": request_id})
            await chat_redis_buffer.clear(request_id)
            return
        assistant_content_parts.append(delta)
    assistant_content = "".join(assistant_content_parts)
    if request_id in state.interrupted_request_ids:
        await _send(state, "assistant.message.done", {"requestId": request_id})
        await chat_redis_buffer.clear(request_id)
        return

    # 同步敏感词检测 —— Agent 输出命中替换为固定回复，但仍写入 MySQL
    clean_output = assistant_content
    if sensitive_matcher.find_first(assistant_content):
        clean_output = "对不起，这个问题我暂时无法回答"

    # 流式输出（已替换过的安全内容）
    for delta in clean_output:
        await _send(state, "assistant.message.delta", {"requestId": request_id, "delta": delta})
    await _publish_chat_audit(
        ChatAuditMessage(
            check_id=f"chk_{uuid.uuid4().hex}",
            connection_id=state.context.connection_id,
            request_id=request_id,
            message_id=assistant_message_id,
            role="assistant",
            stage="final_output",
            content=clean_output,
            session_id=state.context.session_id,
            user_id=state.context.user_id,
            assistant_message_id=assistant_message_id,
        )
    )
    await _send(state, "assistant.message.done", {"requestId": request_id, "messageId": assistant_message_id})


@app.on_event("startup")
async def startup() -> None:
    global chat_audit_consumer_task
    chat_audit_consumer_task = asyncio.create_task(_run_chat_audit_consumer())


@app.on_event("shutdown")
async def shutdown() -> None:
    if chat_audit_consumer_task is not None:
        chat_audit_consumer_task.cancel()
    await chat_audit_mq_service.close()
    await chat_redis_buffer.close()
    await agent_memory_repository.close()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/agent/sessions")
async def list_agent_sessions(request: Request, cursor: str | None = None, limit: int = 20) -> dict[str, Any]:
    user_context = await _http_user_context(request)
    try:
        return await agent_memory_repository.list_sessions(
            user_id=user_context.get("userId"),
            cursor=cursor,
            limit=limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/agent/sessions")
async def create_agent_session(request: Request) -> dict[str, Any]:
    user_context = await _http_user_context(request)
    body = await request.json()
    title = str(body.get("title") or "新会话").strip()[:200] or "新会话"
    session_id = await agent_memory_repository.create_session(user_id=user_context.get("userId"), title=title)
    return {"session": {"id": session_id, "title": title}}


@app.get("/api/agent/sessions/{session_id}/messages")
async def list_agent_session_messages(
    session_id: int,
    request: Request,
    cursor: str | None = None,
    limit: int = 30,
) -> dict[str, Any]:
    user_context = await _http_user_context(request)
    try:
        return await agent_memory_repository.list_messages(
            session_id=session_id,
            user_id=user_context.get("userId"),
            cursor=cursor,
            limit=limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.patch("/api/agent/sessions/{session_id}")
async def rename_agent_session(session_id: int, request: Request) -> dict[str, Any]:
    body = await request.json()
    title = str(body.get("title") or "新会话").strip()[:200] or "新会话"
    await agent_memory_repository.update_session_title(session_id=session_id, title=title)
    return {"session": {"id": session_id, "title": title}}


@app.delete("/api/agent/sessions/{session_id}")
async def delete_agent_session(session_id: int, request: Request) -> dict[str, Any]:
    await agent_memory_repository.delete_session(session_id=session_id)
    return {"ok": True}


def _safe_filename(filename: str) -> str:
    stem = Path(filename).stem.strip().replace(" ", "_")[:80] or "file"
    suffix = Path(filename).suffix[:20]
    safe_stem = "".join(ch for ch in stem if ch.isascii() and (ch.isalnum() or ch in {"-", "_", "."})) or "file"
    safe_suffix = "".join(ch for ch in suffix if ch.isascii() and (ch.isalnum() or ch in {"."}))
    return f"{uuid.uuid4().hex}_{safe_stem}{safe_suffix}"


@app.post("/api/agent/upload")
async def upload_agent_file(file: UploadFile = File(...)) -> dict[str, Any]:
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    original_name = file.filename or "file"
    saved_name = _safe_filename(original_name)
    mime = file.content_type or mimetypes.guess_type(original_name)[0] or "application/octet-stream"
    attachment_type = "image" if mime.startswith("image/") else "file"

    try:
        file.file.seek(0, 2)
        size = file.file.tell()
        file.file.seek(0)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"读取上传文件失败：{exc}") from exc

    if size > max_bytes:
        raise HTTPException(status_code=413, detail=f"文件不能超过 {settings.max_upload_size_mb}MB")

    try:
        url = await upload_fileobj_to_oss(saved_name, file.file, mime)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"上传到 OSS 失败：{exc}") from exc

    return {
        "url": url,
        "name": original_name,
        "type": attachment_type,
        "size": size,
        "mime": mime,
    }


@app.websocket("/ws/agent")
async def agent_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    connection_id = str(uuid.uuid4())
    token = _cookie_token(websocket)
    user_context = await resolve_user_context(token)
    if not user_context.get("authenticated"):
        await websocket.send_json({"eventType": "auth.required", "payload": {"message": "请先登录后使用 YAAI 助手"}})
        await websocket.close(code=4401, reason="login required")
        return

    platform = websocket.query_params.get("platform") or "frontend"
    context = RuntimeContext(
        connection_id=connection_id,
        platform=platform,
        session_id=None,
        user_id=user_context.get("userId"),
        role=user_context.get("role"),
        roles=user_context.get("roles") or [],
        authenticated=True,
    )
    state = ConnectionState(websocket, context)
    connections[connection_id] = state
    ping_task = asyncio.create_task(_ping_loop(state))

    try:
        await _send(
            state,
            "connected",
            {
                "connectionId": connection_id,
                "platform": platform,
                "userId": context.user_id,
                "role": context.role,
            },
        )
        while True:
            message = await websocket.receive_json()
            event_type = message.get("eventType")
            if event_type == "pong":
                state.last_pong_at = time.monotonic()
                continue
            if event_type == "ping":
                await _send(state, "pong", {"ts": message.get("ts")})
                continue
            if event_type == "session.resume":
                session_id = message.get("sessionId")
                state.context.session_id = int(session_id) if session_id else None
                await _send(state, "session.resumed", {"sessionId": state.context.session_id})
                continue
            if event_type == "frontend.action.result":
                result = message.get("payload") if isinstance(message.get("payload"), dict) else message
                action_id = str(result.get("actionId") or "")
                future = state.action_results.pop(action_id, None)
                if future is not None and not future.done():
                    future.set_result(result)
                continue
            if event_type == "user.message":
                task = asyncio.create_task(_handle_user_message(state, message))
                state.message_tasks.add(task)
                task.add_done_callback(state.message_tasks.discard)
                continue
            await _send(state, "error", {"message": f"unsupported eventType: {event_type}"})
    except WebSocketDisconnect:
        pass
    finally:
        state.closed = True
        ping_task.cancel()
        for task in state.message_tasks:
            task.cancel()
        for future in state.action_results.values():
            if not future.done():
                future.set_result({"success": False, "error": "connection closed"})
        connections.pop(connection_id, None)


if __name__ == "__main__":
    uvicorn.run("main:app", host=settings.agent_host, port=settings.agent_port, reload=False)
