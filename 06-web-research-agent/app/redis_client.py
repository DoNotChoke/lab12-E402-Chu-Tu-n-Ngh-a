from __future__ import annotations

from functools import lru_cache

import redis

from app.config import settings


@lru_cache
def get_redis() -> redis.Redis:
    return redis.Redis.from_url(
        settings.redis_url,
        decode_responses=True,
        health_check_interval=30,
        socket_connect_timeout=5,
        socket_timeout=5,
    )
