from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    session_id: str = Field(
        min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$", examples=["user-1"]
    )
    message: str = Field(min_length=1, max_length=1000, examples=["What is FastAPI?"])


class ChatResponse(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    answer: str = Field(min_length=1, max_length=10000)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))