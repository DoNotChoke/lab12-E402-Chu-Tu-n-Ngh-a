from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from fastapi import HTTPException, Security
from fastapi.security.api_key import APIKeyHeader

from app.config import settings


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


@dataclass(frozen=True)
class AuthenticatedUser:
    user_id: str
    api_key_fingerprint: str


def verify_api_key(api_key: str = Security(api_key_header)) -> AuthenticatedUser:
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Include header: X-API-Key: <your-key>",
        )

    if api_key != settings.agent_api_key:
        raise HTTPException(status_code=401, detail="Invalid API key.")

    fingerprint = sha256(api_key.encode("utf-8")).hexdigest()[:16]
    return AuthenticatedUser(
        user_id=f"user-{fingerprint}",
        api_key_fingerprint=fingerprint,
    )
