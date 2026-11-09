from fastapi import Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.services.chat_service import ChatService
from app.services.llm_clients import LLMClient, get_llm_client


def get_chat_service(
    db: Session = Depends(get_db), llm: LLMClient = Depends(get_llm_client)
) -> ChatService:
    return ChatService(db, llm)