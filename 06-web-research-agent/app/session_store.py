from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import HTTPException

from app.config import settings
from app.redis_client import get_redis


class SessionStore:
    def _key(self, session_id: str) -> str:
        return f"session:{session_id}"

    def get_or_create_session(self, user_id: str, session_id: str | None = None) -> dict:
        redis_client = get_redis()
        resolved_session_id = session_id or str(uuid4())
        key = self._key(resolved_session_id)
        raw = redis_client.get(key)

        if raw:
            session = json.loads(raw)
            if session["user_id"] != user_id:
                raise HTTPException(status_code=403, detail="Session does not belong to the authenticated user.")
            return session

        now = datetime.now(timezone.utc).isoformat()
        session = {
            "session_id": resolved_session_id,
            "user_id": user_id,
            "created_at": now,
            "updated_at": now,
            "messages": [],
        }
        self._save(session)
        return session

    def get_session(self, user_id: str, session_id: str) -> dict:
        redis_client = get_redis()
        raw = redis_client.get(self._key(session_id))
        if not raw:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")

        session = json.loads(raw)
        if session["user_id"] != user_id:
            raise HTTPException(status_code=403, detail="Session does not belong to the authenticated user.")
        return session

    def append_message(
        self,
        session_id: str,
        user_id: str,
        role: str,
        content: str,
        metadata: dict | None = None,
    ) -> dict:
        session = self.get_or_create_session(user_id=user_id, session_id=session_id)
        session["messages"].append(
            {
                "role": role,
                "content": content,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": metadata or {},
            }
        )
        if len(session["messages"]) > settings.max_history_messages:
            session["messages"] = session["messages"][-settings.max_history_messages :]
        session["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._save(session)
        return session

    def history_for_llm(self, user_id: str, session_id: str) -> list[dict]:
        session = self.get_session(user_id=user_id, session_id=session_id)
        return session["messages"][-settings.max_history_messages :]

    def delete_session(self, user_id: str, session_id: str) -> None:
        self.get_session(user_id=user_id, session_id=session_id)
        get_redis().delete(self._key(session_id))

    def _save(self, session: dict) -> None:
        get_redis().setex(
            self._key(session["session_id"]),
            settings.session_ttl_seconds,
            json.dumps(session, ensure_ascii=True),
        )


session_store = SessionStore()
