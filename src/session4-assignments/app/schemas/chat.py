from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

# Shared by the request body and the URL path so both enforce the same rule.
SESSION_ID_PATTERN = r"^[A-Za-z0-9_-]+$"


class ChatRequest(BaseModel):
    """Input validation for POST /chat."""

    model_config = ConfigDict(str_strip_whitespace=True)

    session_id: str = Field(
        min_length=1, max_length=64, pattern=SESSION_ID_PATTERN, examples=["user-1"]
    )
    message: str = Field(min_length=1, max_length=1000, examples=["What is Docker?"])


class ChatResponse(BaseModel):
    """Output validation: one stored question/answer pair."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    session_id: str = Field(min_length=1, max_length=64)
    question: str
    answer: str = Field(min_length=1, max_length=10000)
    created_at: datetime


class ChatHistoryResponse(BaseModel):
    session_id: str
    count: int
    messages: list[ChatResponse]


class ChatSessionSummary(BaseModel):
    session_id: str
    message_count: int
    latest_message_at: datetime