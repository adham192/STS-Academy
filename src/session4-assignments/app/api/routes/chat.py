from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Response

from app.api.deps import get_chat_service
from app.schemas.chat import (
    SESSION_ID_PATTERN,
    ChatHistoryResponse,
    ChatRequest,
    ChatResponse,
    ChatSessionSummary,
)
from app.services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["chat"])

SessionId = Annotated[
    str,
    Path(min_length=1, max_length=64, pattern=SESSION_ID_PATTERN, description="Letters, digits, _ or -"),
]


@router.post("", response_model=ChatResponse, status_code=201)
def ask(payload: ChatRequest, service: ChatService = Depends(get_chat_service)):
    """Create a chat exchange: ask the LLM and save the question and answer."""
    return service.ask(payload.session_id, payload.message)


@router.get("/sessions", response_model=list[ChatSessionSummary])
def list_sessions(
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    service: ChatService = Depends(get_chat_service),
):
    """Read a paginated list of conversations stored in the database."""
    return service.list_sessions(limit, offset)


@router.get("/{session_id}/history", response_model=ChatHistoryResponse)
def history(
    session_id: SessionId,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    service: ChatService = Depends(get_chat_service),
):
    """Read the stored conversation of one session, oldest first."""
    messages = service.history(session_id, limit, offset)
    return {"session_id": session_id, "count": len(messages), "messages": messages}


@router.delete("/{session_id}", status_code=204)
def delete_history(session_id: SessionId, service: ChatService = Depends(get_chat_service)):
    """Delete every stored message belonging to a conversation."""
    if service.clear(session_id) == 0:
        raise HTTPException(status_code=404, detail="No history found for this session.")
    return Response(status_code=204)