import logging
import time
from functools import lru_cache

from app.core.config import settings
from app.services.llm_clients import LLMClient, LLMError, build_llm_client

logger = logging.getLogger(__name__)


class ChatService:
    """Takes a question, asks the configured LLM, returns the answer."""

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    async def reply(self, session_id: str, message: str) -> str:
        logger.info("Question received (session=%s, chars=%d)", session_id, len(message))
        logger.debug("Question text (session=%s): %s", session_id, message)

        start = time.perf_counter()
        answer = await self._client.generate(message)
        elapsed = time.perf_counter() - start

        if not answer:
            logger.error("LLM returned an empty answer (session=%s)", session_id)
            raise LLMError("Empty answer from LLM")

        logger.info("Answer generated (session=%s, chars=%d, %.2fs)", session_id, len(answer), elapsed)
        logger.debug("Answer text (session=%s): %s", session_id, answer)
        return answer


@lru_cache
def get_chat_service() -> ChatService:
    return ChatService(build_llm_client(settings))