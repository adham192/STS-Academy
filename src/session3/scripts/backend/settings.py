"""Backend configuration. Everything comes from environment variables (.env locally)."""

import logging
import os

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("backend.settings")

# Application connection strings. The POSTGRES_USER / POSTGRES_PASSWORD / POSTGRES_DB
# variables are only for the PostgreSQL *container*; the app never reads them.
DATABASE_URL = os.getenv("DATABASE_URL")
REDIS_URL = os.getenv("REDIS_URL")


def _positive_int(name, default):
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
        if value > 0:
            return value
    except ValueError:
        pass
    logger.warning("%s=%r is not a positive integer; using %d", name, raw, default)
    return default


CACHE_TTL_SECONDS = _positive_int("CACHE_TTL_SECONDS", 3600)
