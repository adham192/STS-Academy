from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]  # STS-Academy/src/session4
PROJECT_ROOT = BASE_DIR.parents[1]              # STS-Academy (shared .env lives here)


class Settings(BaseSettings):
    APP_NAME: str = "STS Chatbot"
    LOG_LEVEL: str = "INFO"
    LOG_DIR: Path = BASE_DIR / "logs"

    # Which LLM answers: "gemini" (cloud) or "ollama" (local)
    LLM_PROVIDER: Literal["gemini", "ollama"] = "gemini"
    LLM_TIMEOUT: float = 60.0

    # Cloud (Gemini). Accepts your existing .env names `api-key` / `model`,
    # or the clearer GEMINI_API_KEY / GEMINI_MODEL.
    GEMINI_API_KEY: SecretStr = Field(
        default=SecretStr(""), validation_alias=AliasChoices("GEMINI_API_KEY", "api-key")
    )
    GEMINI_MODEL: str = Field(
        default="gemini-2.5-flash", validation_alias=AliasChoices("GEMINI_MODEL", "model")
    )

    # Local (Ollama)
    OLLAMA_HOST: str = "http://localhost:11434"
    LOCAL_MODEL: str = ""

    # Later files win, so a session4/.env can override the shared root .env
    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env", BASE_DIR / ".env"), extra="ignore"
    )


settings = Settings()