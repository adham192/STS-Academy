"""PostgreSQL storage (SQLAlchemy 2.0 + psycopg 3) for questions and answers.

Every public function swallows database errors (logging them) and returns a neutral
value, so a database outage degrades the app to "no persistence" instead of crashing it.
Tables are created lazily and retried on the next call if PostgreSQL was not ready yet.
"""

import logging
import threading
import uuid
from datetime import datetime

from sqlalchemy import JSON, DateTime, String, Text, Uuid, create_engine, delete, func, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

import settings

logger = logging.getLogger("backend.database")


class Base(DeclarativeBase):
    pass


class QARecord(Base):
    __tablename__ = "qa_records"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    question: Mapped[str] = mapped_column(Text)                  # as the user typed it
    normalized_question: Mapped[str] = mapped_column(Text)       # for duplicate detection
    scope: Mapped[str] = mapped_column(String(255), default="", server_default="")
    # sha256(normalized_question + scope): fixed-size unique key (a btree index on a long
    # TEXT column can overflow), also the Redis key suffix.
    question_hash: Mapped[str] = mapped_column(String(64), unique=True)
    answer: Mapped[str] = mapped_column(Text)
    documents: Mapped[list] = mapped_column(JSON, default=list)  # PDFs the answer used
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


_engine = None
_tables_ready = False
_lock = threading.Lock()


def _get_engine():
    global _engine
    if not settings.DATABASE_URL:
        raise SQLAlchemyError("DATABASE_URL is not set")
    with _lock:
        if _engine is None:
            _engine = create_engine(
                settings.DATABASE_URL,
                pool_pre_ping=True,
                pool_size=5,
                max_overflow=5,
                connect_args={"connect_timeout": 5},
            )
        return _engine


def _ready_engine():
    """Return the engine after making sure the tables exist."""
    global _tables_ready
    engine = _get_engine()
    if not _tables_ready:
        with _lock:
            if not _tables_ready:
                Base.metadata.create_all(engine)
                _tables_ready = True
                logger.info("Database tables are ready")
    return engine


def init_db() -> bool:
    """Try to create the tables now. Returns False (and logs) if PostgreSQL is unreachable."""
    try:
        _ready_engine()
        return True
    except SQLAlchemyError as error:
        logger.error("PostgreSQL is not available yet: %s", error)
        return False


def ping() -> bool:
    try:
        with Session(_ready_engine()) as session:
            session.execute(select(1))
        return True
    except SQLAlchemyError:
        return False


def find_answer(q_hash: str):
    """Return {'answer', 'documents'} for a stored question, or None (also on errors)."""
    try:
        with Session(_ready_engine()) as session:
            row = session.execute(
                select(QARecord.answer, QARecord.documents).where(QARecord.question_hash == q_hash)
            ).first()
        return {"answer": row.answer, "documents": list(row.documents or [])} if row else None
    except SQLAlchemyError as error:
        logger.error("PostgreSQL lookup failed: %s", error)
        return None


def save_answer(question, normalized, scope, q_hash, answer, documents) -> bool:
    """Insert a question/answer pair. A duplicate (same hash) is silently ignored."""
    try:
        with Session(_ready_engine()) as session:
            session.add(QARecord(
                question=question, normalized_question=normalized, scope=scope,
                question_hash=q_hash, answer=answer, documents=list(documents),
            ))
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
                logger.debug("Duplicate question ignored (%s)", q_hash[:12])
        return True
    except SQLAlchemyError as error:
        logger.error("Could not save the answer to PostgreSQL: %s", error)
        return False


def delete_by_scope(scopes):
    """Delete stored answers whose scope is in `scopes`. Returns their hashes, or None on error."""
    try:
        with Session(_ready_engine()) as session:
            hashes = list(session.scalars(
                select(QARecord.question_hash).where(QARecord.scope.in_(scopes))
            ))
            session.execute(delete(QARecord).where(QARecord.scope.in_(scopes)))
            session.commit()
        return hashes
    except SQLAlchemyError as error:
        logger.error("Could not delete stale answers from PostgreSQL: %s", error)
        return None
