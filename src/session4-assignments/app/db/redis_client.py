from functools import lru_cache

import redis

from app.core.config import settings


@lru_cache
def get_redis() -> redis.Redis:
    """FastAPI dependency: one shared Redis client built from REDIS_URL (connects lazily, short timeouts)."""
    return redis.Redis.from_url(
        settings.REDIS_URL.get_secret_value(),
        decode_responses=True,
        socket_connect_timeout=1,
        socket_timeout=1,
    )