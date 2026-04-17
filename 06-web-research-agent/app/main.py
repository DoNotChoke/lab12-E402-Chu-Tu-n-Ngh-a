from __future__ import annotations

import logging
import os
import signal
import socket
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from app.agent import ResearchAgent
from app.auth import AuthenticatedUser, verify_api_key
from app.config import settings
from app.cost_guard import (
    BudgetStatus,
    check_budget,
    ensure_budget_available,
    estimate_llm_cost,
    record_cost,
)
from app.logging_utils import configure_logging
from app.models import AskRequest, AskResponse, SessionHistoryResponse, UsageSummary
from app.rate_limiter import RateLimitStatus, check_rate_limit
from app.redis_client import get_redis
from app.session_store import session_store


configure_logging(settings.log_level)
logger = logging.getLogger(__name__)

START_TIME = time.time()
INSTANCE_ID = os.getenv("INSTANCE_ID", socket.gethostname())
_is_ready = False
_in_flight_requests = 0
agent = ResearchAgent()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _is_ready
    redis_client = get_redis()
    redis_client.ping()
    _is_ready = True
    logger.info("agent started", extra={"event": "startup", "instance_id": INSTANCE_ID})
    yield
    _is_ready = False
    logger.info("agent shutting down", extra={"event": "shutdown", "instance_id": INSTANCE_ID})


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs" if settings.environment != "production" else None,
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "X-API-Key"],
)


@app.middleware("http")
async def request_tracking_middleware(request: Request, call_next):
    global _in_flight_requests
    start = time.time()
    _in_flight_requests += 1
    try:
        response: Response = await call_next(request)
    finally:
        _in_flight_requests -= 1

    duration_ms = round((time.time() - start) * 1000, 2)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["X-Instance-Id"] = INSTANCE_ID
    if "server" in response.headers:
        del response.headers["server"]

    logger.info(
        "request complete",
        extra={
            "event": "request",
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
            "instance_id": INSTANCE_ID,
        },
    )
    return response


@app.get("/")
def root():
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "instance_id": INSTANCE_ID,
        "endpoints": {
            "ask": "POST /ask",
            "history": "GET /sessions/{session_id}",
            "health": "GET /health",
            "ready": "GET /ready",
        },
    }


@app.get("/health")
def health():
    redis_ok = False
    try:
        get_redis().ping()
        redis_ok = True
    except Exception:  # noqa: BLE001
        redis_ok = False

    return {
        "status": "ok" if redis_ok else "degraded",
        "instance_id": INSTANCE_ID,
        "version": settings.app_version,
        "environment": settings.environment,
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "redis_connected": redis_ok,
        "openai_configured": bool(settings.openai_api_key),
        "serper_configured": bool(settings.serper_api_key),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/ready")
def ready():
    if not _is_ready:
        raise HTTPException(status_code=503, detail="Application is not ready yet.")

    try:
        get_redis().ping()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Redis not ready: {exc}") from exc

    return {
        "ready": True,
        "instance_id": INSTANCE_ID,
        "in_flight_requests": _in_flight_requests,
    }


@app.post("/ask", response_model=AskResponse)
async def ask_agent(
    body: AskRequest,
    user: AuthenticatedUser = Depends(verify_api_key),
    rate_limit: RateLimitStatus = Depends(check_rate_limit),
    _budget_status: BudgetStatus = Depends(check_budget),
):
    if not settings.openai_api_key:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY is not configured.")
    if not settings.serper_api_key:
        raise HTTPException(status_code=503, detail="SERPER_API_KEY is not configured.")

    session = session_store.get_or_create_session(user_id=user.user_id, session_id=body.session_id)
    history = session_store.history_for_llm(user_id=user.user_id, session_id=session["session_id"])

    projected_cost = settings.search_tool_cost_usd + settings.fetch_tool_cost_usd + 0.01
    ensure_budget_available(user.user_id, projected_cost_usd=projected_cost)

    result = await agent.answer(question=body.question, history=history)

    llm_cost = estimate_llm_cost(
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )
    total_cost = llm_cost + result.tool_cost_usd
    updated_budget = record_cost(user.user_id, total_cost)

    session_store.append_message(
        session_id=session["session_id"],
        user_id=user.user_id,
        role="user",
        content=body.question,
    )
    session_store.append_message(
        session_id=session["session_id"],
        user_id=user.user_id,
        role="assistant",
        content=result.answer,
        metadata={
            "citations": [citation.model_dump() for citation in result.citations],
            "tools_used": [trace.model_dump() for trace in result.tool_traces],
        },
    )

    return AskResponse(
        session_id=session["session_id"],
        question=body.question,
        answer=result.answer,
        model=settings.llm_model,
        citations=result.citations,
        tools_used=result.tool_traces,
        usage=UsageSummary(
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            total_cost_usd=round(total_cost, 6),
            requests_remaining=rate_limit.remaining,
            budget_remaining_usd=round(updated_budget.remaining_usd, 6),
        ),
    )


@app.get("/sessions/{session_id}", response_model=SessionHistoryResponse)
def get_session_history(
    session_id: str,
    user: AuthenticatedUser = Depends(verify_api_key),
):
    session = session_store.get_session(user_id=user.user_id, session_id=session_id)
    return SessionHistoryResponse(
        session_id=session["session_id"],
        user_id=session["user_id"],
        messages=session["messages"],
    )


@app.delete("/sessions/{session_id}")
def delete_session(
    session_id: str,
    user: AuthenticatedUser = Depends(verify_api_key),
):
    session_store.delete_session(user_id=user.user_id, session_id=session_id)
    return {"deleted": session_id}


def _handle_signal(signum: int, _frame) -> None:
    logger.info("signal received", extra={"event": "signal", "signum": signum})


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        timeout_graceful_shutdown=30,
    )
