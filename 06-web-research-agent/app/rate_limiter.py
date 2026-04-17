from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

from fastapi import Depends, HTTPException

from app.auth import AuthenticatedUser, verify_api_key
from app.config import settings
from app.redis_client import get_redis


@dataclass(frozen=True)
class RateLimitStatus:
    limit: int
    remaining: int
    retry_after_seconds: int


def _key_for_user(user_id: str) -> str:
    return f"rate_limit:{user_id}"


def check_rate_limit(
    user: AuthenticatedUser = Depends(verify_api_key),
) -> RateLimitStatus:
    redis_client = get_redis()
    now_ms = int(time.time() * 1000)
    window_start_ms = now_ms - 60_000
    key = _key_for_user(user.user_id)

    redis_client.zremrangebyscore(key, 0, window_start_ms)
    current_count = redis_client.zcard(key)

    if current_count >= settings.rate_limit_per_minute:
        oldest = redis_client.zrange(key, 0, 0, withscores=True)
        retry_after_seconds = 60
        if oldest:
            retry_after_ms = max(0, int(oldest[0][1]) + 60_000 - now_ms)
            retry_after_seconds = max(1, retry_after_ms // 1000)

        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                "limit": settings.rate_limit_per_minute,
                "window_seconds": 60,
                "retry_after_seconds": retry_after_seconds,
            },
            headers={"Retry-After": str(retry_after_seconds)},
        )

    redis_client.zadd(key, {f"{now_ms}-{uuid.uuid4().hex}": now_ms})
    redis_client.expire(key, 65)

    return RateLimitStatus(
        limit=settings.rate_limit_per_minute,
        remaining=max(settings.rate_limit_per_minute - current_count - 1, 0),
        retry_after_seconds=0,
    )
