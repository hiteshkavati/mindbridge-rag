from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .auth import AuthError, UserStore, issue_token, read_token
from .bootstrap import build_pipeline
from .config import Settings, get_settings
from .pipeline import Pipeline
from .sessions import SessionStore

log = logging.getLogger("mindbridge.api")
STATIC_DIR = Path(__file__).parent / "static"


class RateLimiter:
    """Sliding-window limiter keyed by user or client address. Per process, in memory."""

    def __init__(self, limit: int, window_s: float = 60.0):
        self.limit = limit
        self.window = window_s
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > self.window:
                q.popleft()
            if len(q) >= self.limit:
                raise HTTPException(429, "too many requests, slow down a little")
            q.append(now)


class Credentials(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=1, max_length=128)


class ChatIn(BaseModel):
    message: str = Field(min_length=1)
    session_id: str | None = Field(default=None, max_length=64)


class SourceOut(BaseModel):
    id: str
    title: str
    section: str
    snippet: str
    score: float


class SafetyOut(BaseModel):
    level: str
    about_other: bool


class ChatOut(BaseModel):
    session_id: str
    reply: str
    mode: str
    safety: SafetyOut
    sources: list[SourceOut]
    timings_ms: dict[str, float]


def _client_key(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def create_app(settings: Settings | None = None, pipeline: Pipeline | None = None) -> FastAPI:
    cfg = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.pipeline = pipeline or build_pipeline(cfg)
        yield

    app = FastAPI(title="MindBridge", version=__version__, lifespan=lifespan)
    app.state.pipeline = pipeline  # tests inject one; lifespan fills it otherwise
    sessions = SessionStore(cfg.session_ttl_s, max_turns=2 * cfg.history_turns)
    users = UserStore(cfg.db_path) if cfg.require_auth else None
    chat_limit = RateLimiter(cfg.rate_limit_per_min)
    auth_limit = RateLimiter(10)

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data:; frame-ancestors 'none'"
        )
        response.headers["Cache-Control"] = "no-store"
        return response

    def current_user(request: Request) -> str:
        if not cfg.require_auth:
            return "anonymous"
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise HTTPException(401, "missing bearer token")
        try:
            return read_token(token, cfg.jwt_secret)
        except AuthError as exc:
            raise HTTPException(401, str(exc)) from exc

    # ----------------------------------------------------------------- meta

    @app.get("/api/health")
    def health(request: Request) -> dict:
        pipe: Pipeline = request.app.state.pipeline
        return {
            "status": "ok",
            "version": __version__,
            "auth_required": cfg.require_auth,
            "llm": pipe.llm.name,
            "embedding": pipe.retriever.embedder.name,
            "chunks": len(pipe.retriever.chunks),
        }

    @app.get("/api/helplines")
    def helplines(request: Request) -> dict:
        h = request.app.state.pipeline.helplines
        return {
            "emergency": h.emergency_number,
            "lines": [
                {"name": x.name, "numbers": list(x.numbers), "hours": x.hours} for x in h.lines
            ],
        }

    # ----------------------------------------------------------------- auth

    if users is not None:

        @app.post("/api/auth/register", status_code=201)
        def register(body: Credentials, request: Request) -> dict:
            auth_limit.check(_client_key(request))
            try:
                return {"username": users.register(body.username, body.password)}
            except AuthError as exc:
                raise HTTPException(400, str(exc)) from exc

        @app.post("/api/auth/login")
        def login(body: Credentials, request: Request) -> dict:
            auth_limit.check(_client_key(request))
            try:
                name = users.authenticate(body.username, body.password)
            except AuthError as exc:
                raise HTTPException(401, str(exc)) from exc
            return {
                "access_token": issue_token(name, cfg.jwt_secret, cfg.jwt_ttl_minutes),
                "token_type": "bearer",
                "expires_in": cfg.jwt_ttl_minutes * 60,
            }

    # ----------------------------------------------------------------- chat

    @app.post("/api/chat", response_model=ChatOut)
    def chat(body: ChatIn, request: Request, user: str = Depends(current_user)) -> ChatOut:
        chat_limit.check(user if cfg.require_auth else _client_key(request))
        message = body.message.strip()
        if not message:
            raise HTTPException(422, "message is empty")
        if len(message) > cfg.max_message_chars:
            raise HTTPException(422, f"message is longer than {cfg.max_message_chars} characters")

        sid = body.session_id or sessions.new_id()
        history = sessions.history(sid, user)
        reply = request.app.state.pipeline.answer(message, history)
        try:
            sessions.add(sid, user, message, reply.text)
        except PermissionError as exc:
            raise HTTPException(403, "that session belongs to someone else") from exc

        # message text is deliberately never logged
        log.info(
            "chat session=%s mode=%s level=%s total_ms=%s",
            hashlib.sha256(sid.encode()).hexdigest()[:8],
            reply.mode,
            reply.level.name.lower(),
            reply.timings_ms.get("total"),
        )
        return ChatOut(
            session_id=sid,
            reply=reply.text,
            mode=reply.mode,
            safety=SafetyOut(level=reply.level.name.lower(), about_other=reply.about_other),
            sources=[
                SourceOut(
                    id=s.doc_id, title=s.title, section=s.section, snippet=s.snippet, score=s.score
                )
                for s in reply.sources
            ],
            timings_ms=reply.timings_ms,
        )

    @app.delete("/api/sessions/{session_id}", status_code=204)
    def delete_session(session_id: str, user: str = Depends(current_user)) -> Response:
        sessions.delete(session_id, user)
        return Response(status_code=204)

    # ------------------------------------------------------------------ ui

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app
