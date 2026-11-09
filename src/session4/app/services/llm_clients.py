import logging
import time
from typing import Protocol

import httpx

from app.core.config import Settings

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when the LLM call fails or returns something unusable."""


class LLMClient(Protocol):
    async def generate(self, prompt: str) -> str: ...


async def _post_json(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as exc:
        logger.error("LLM returned HTTP %s: %s", exc.response.status_code, exc.response.text[:300])
        raise LLMError("LLM returned an error status") from exc
    except (httpx.HTTPError, ValueError) as exc:
        logger.error("LLM request failed: %r", exc)
        raise LLMError("LLM request failed") from exc


class GeminiClient:
    def __init__(self, api_key: str, model: str, timeout: float) -> None:
        self._api_key, self._model, self._timeout = api_key, model, timeout

    async def generate(self, prompt: str) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self._model}:generateContent"
        payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
        data = await _post_json(url, payload, {"x-goog-api-key": self._api_key}, self._timeout)
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            logger.error("Unexpected Gemini response shape: %s", str(data)[:300])
            raise LLMError("Unexpected Gemini response") from exc


class OllamaClient:
    def __init__(self, host: str, model: str, timeout: float) -> None:
        self._host, self._model, self._timeout = host.rstrip("/"), model, timeout

    async def generate(self, prompt: str) -> str:
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        data = await _post_json(f"{self._host}/api/chat", payload, {}, self._timeout)
        try:
            return data["message"]["content"].strip()
        except (KeyError, TypeError) as exc:
            logger.error("Unexpected Ollama response shape: %s", str(data)[:300])
            raise LLMError("Unexpected Ollama response") from exc


def build_llm_client(settings: Settings) -> LLMClient:
    """Pick the client from settings. Fails fast if the config is incomplete."""
    if settings.LLM_PROVIDER == "gemini":
        key = settings.GEMINI_API_KEY.get_secret_value()
        if not key:
            raise RuntimeError("LLM_PROVIDER=gemini but no API key (api-key / GEMINI_API_KEY) is set")
        logger.info("LLM provider: gemini (model=%s)", settings.GEMINI_MODEL)
        return GeminiClient(key, settings.GEMINI_MODEL, settings.LLM_TIMEOUT)

    if not settings.LOCAL_MODEL:
        raise RuntimeError("LLM_PROVIDER=ollama but LOCAL_MODEL is not set")
    logger.info("LLM provider: ollama (model=%s, host=%s)", settings.LOCAL_MODEL, settings.OLLAMA_HOST)
    return OllamaClient(settings.OLLAMA_HOST, settings.LOCAL_MODEL, settings.LLM_TIMEOUT)