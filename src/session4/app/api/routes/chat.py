import logging

from fastapi import APIRouter, Depends, HTTPException

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import ChatService, get_chat_service
from app.services.llm_clients import LLMError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(payload: ChatRequest, service: ChatService = Depends(get_chat_service)) -> ChatResponse:
    try:
        answer = await service.reply(payload.session_id, payload.message)
    except LLMError as exc:
        raise HTTPException(status_code=502, detail="The language model is unavailable.") from exc
    return ChatResponse(session_id=payload.session_id, answer=answer)