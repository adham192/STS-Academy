"""Redis answer cache. Redis is a cache only: every failure is logged and ignored."""

import json
import logging
import threading

import redis

import settings

logger = logging.getLogger("backend.cache")

KEY_PREFIX = "qa:"
_client = None
_lock = threading.Lock()


def _redis():
    global _client
    if not settings.REDIS_URL:
        return None
    with _lock:
        if _client is None:
            # Short timeouts: a dead Redis must not make every request wait.
            _client = redis.Redis.from_url(
                settings.REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=1,
                socket_timeout=1,
                health_check_interval=30,
            )
        return _client


def get(q_hash: str):
    """Return {'answer', 'documents'} or None (miss, expired, or Redis unavailable)."""
    try:
        client = _redis()
        raw = client.get(KEY_PREFIX + q_hash) if client else None
        return json.loads(raw) if raw else None
    except (redis.RedisError, ValueError) as error:
        logger.warning("Redis read failed, falling back: %s", error)
        return None


def put(q_hash: str, answer: str, documents) -> None:
    try:
        client = _redis()
        if client:
            payload = json.dumps({"answer": answer, "documents": list(documents)}, ensure_ascii=False)
            client.set(KEY_PREFIX + q_hash, payload, ex=settings.CACHE_TTL_SECONDS)
    except redis.RedisError as error:
        logger.warning("Redis write failed (answer not cached): %s", error)


def delete(q_hashes) -> None:
    try:
        client = _redis()
        if client and q_hashes:
            client.delete(*[KEY_PREFIX + h for h in q_hashes])
    except redis.RedisError as error:
        logger.warning("Redis delete failed: %s", error)


def flush_answers() -> None:
    """Delete every cached answer (used when PostgreSQL cannot tell us which keys are stale)."""
    try:
        client = _redis()
        if client:
            for key in client.scan_iter(match=KEY_PREFIX + "*", count=500):
                client.delete(key)
    except redis.RedisError as error:
        logger.warning("Redis flush failed: %s", error)


def ping() -> bool:
    try:
        client = _redis()
        return bool(client and client.ping())
    except redis.RedisError:
        return False
