import logging
import time

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.models.chat_message import ChatMessage
from app.services.llm_clients import LLMClient, LLMError

logger = logging.getLogger(__name__)


class ChatService:
    """Generates chatbot answers and manages persisted conversations in PostgreSQL."""

    def __init__(self, db: Session, llm: LLMClient) -> None:
        self._db = db
        self._llm = llm

    def ask(self, session_id: str, question: str) -> ChatMessage:
        """CREATE: get an answer from the LLM and store the exchange."""
        logger.info("Question received (session=%s, chars=%d)", session_id, len(question))
        logger.debug("Question text (session=%s): %s", session_id, question)

        start = time.perf_counter()
        answer = self._llm.generate(question)
        elapsed = time.perf_counter() - start
        if not answer:
            logger.error("LLM returned an empty answer (session=%s)", session_id)
            raise LLMError("Empty answer from LLM")
        logger.info("Answer generated (session=%s, chars=%d, %.2fs)", session_id, len(answer), elapsed)

        record = ChatMessage(session_id=session_id, question=question, answer=answer)
        self._db.add(record)
        self._db.commit()
        self._db.refresh(record)
        logger.info("Saved exchange id=%s (session=%s)", record.id, session_id)
        return record

    def list_sessions(self, limit: int, offset: int) -> list[dict]:
        """READ: list conversations with message counts, newest activity first."""
        latest_at = func.max(ChatMessage.created_at).label("latest_message_at")
        stmt = (
            select(
                ChatMessage.session_id,
                func.count(ChatMessage.id).label("message_count"),
                latest_at,
            )
            .group_by(ChatMessage.session_id)
            .order_by(latest_at.desc(), ChatMessage.session_id)
            .offset(offset)
            .limit(limit)
        )
        rows = self._db.execute(stmt).all()
        logger.info("Conversation list read (returned=%d)", len(rows))
        return [
            {
                "session_id": row.session_id,
                "message_count": row.message_count,
                "latest_message_at": row.latest_message_at,
            }
            for row in rows
        ]

    def history(self, session_id: str, limit: int, offset: int) -> list[ChatMessage]:
        """READ: stored exchanges of one session, oldest first."""
        stmt = (
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.id)
            .offset(offset)
            .limit(limit)
        )
        messages = list(self._db.scalars(stmt))
        logger.info("History read (session=%s, returned=%d)", session_id, len(messages))
        return messages

    def clear(self, session_id: str) -> int:
        """DELETE: remove a session's history and return how many rows were deleted."""
        result = self._db.execute(delete(ChatMessage).where(ChatMessage.session_id == session_id))
        self._db.commit()
        logger.info("History deleted (session=%s, rows=%d)", session_id, result.rowcount)
        return result.rowcount