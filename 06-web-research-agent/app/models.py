from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, HttpUrl


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, description="Reuse a session to continue a conversation")


class Citation(BaseModel):
    title: str | None = None
    url: HttpUrl | str
    source_type: str


class ToolTrace(BaseModel):
    name: str
    input: dict[str, Any]
    success: bool
    summary: str


class UsageSummary(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_cost_usd: float = 0.0
    requests_remaining: int
    budget_remaining_usd: float


class AskResponse(BaseModel):
    session_id: str
    question: str
    answer: str
    model: str
    citations: list[Citation]
    tools_used: list[ToolTrace]
    usage: UsageSummary


class SessionMessage(BaseModel):
    role: str
    content: str
    timestamp: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionHistoryResponse(BaseModel):
    session_id: str
    user_id: str
    messages: list[SessionMessage]
