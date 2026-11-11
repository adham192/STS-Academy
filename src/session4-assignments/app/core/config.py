from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL, make_url


# Root directory of STS-Academy
# This is where your existing .env file is located.
PROJECT_ROOT = Path(__file__).resolve().parents[4]

# The session4-assignments folder itself (where logs/ lives)
PROJECT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    # Application settings
    APP_NAME: str = "STS Chatbot API"
    LOG_LEVEL: str = "INFO"
    LOG_DIR: Path = PROJECT_DIR / "logs"

    # PostgreSQL: one connection URL, e.g. postgresql+psycopg://user:pass@localhost:5432/chatbot
    DATABASE_URL: SecretStr

    # Redis cache: one connection URL, e.g. redis://localhost:6379/0
    REDIS_URL: SecretStr = SecretStr("redis://localhost:6379/0")
    CACHE_TTL_SECONDS: int = 3600

    # LLM provider
    LLM_PROVIDER: Literal["gemini", "ollama"] = "gemini"

    # Gemini settings
    GEMINI_API_KEY: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices("GEMINI_API_KEY", "api-key"),
    )

    GEMINI_MODEL: str = Field(
        default="gemini-2.5-flash",
        validation_alias=AliasChoices("GEMINI_MODEL", "model"),
    )
    LLM_TIMEOUT: int = 60

    # Ollama settings
    OLLAMA_HOST: str = "http://localhost:11434"
    LOCAL_MODEL: str = "qwen2.5-coder:latest"


    @property
    def database_url(self) -> URL:
        """SQLAlchemy URL parsed from DATABASE_URL.

        A plain postgresql:// (or postgres://) URL is switched to the psycopg v3 driver.
        """
        url = make_url(self.DATABASE_URL.get_secret_value())
        if url.drivername in ("postgresql", "postgres"):
            url = url.set(drivername="postgresql+psycopg")
        return url


settings = Settings()