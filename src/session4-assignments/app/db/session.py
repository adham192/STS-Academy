import logging
from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings
from app.db.base import Base

logger = logging.getLogger(__name__)

# pool_pre_ping drops dead connections (e.g. after the database container restarts)
engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db() -> Iterator[Session]:
    """FastAPI dependency: one database session per request."""
    with SessionLocal() as db:
        yield db


def init_db() -> None:
    """Create tables that do not exist yet (simple alternative to migrations)."""
    import app.models  # noqa: F401  (registers the models on Base.metadata)

    try:
        Base.metadata.create_all(engine)
    except SQLAlchemyError:
        logger.exception("Could not initialise the database (%s)", engine.url.render_as_string(hide_password=True))
        raise
    logger.info("Database ready (%s)", engine.url.render_as_string(hide_password=True))