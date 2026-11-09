import logging
from functools import lru_cache
from typing import Protocol

import httpx

from app.core.config import Settings, settings

logger = logging.getLogger(__name__)

MAX_OUTPUT_TOKENS = 4096


class LLMError(Exception):
    """Raised when the LLM call fails or returns something unusable."""


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str: ...


def _post_json(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    try:
        response = httpx.post(url, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        logger.error("LLM returned HTTP %s: %s", exc.response.status_code, exc.response.text[:300])
        raise LLMError("LLM returned an error status") from exc
    except (httpx.HTTPError, ValueError) as exc:
        logger.error("LLM request failed: %r", exc)
        raise LLMError("LLM request failed") from exc


class GeminiClient:
    """Cloud model: Google Gemini over its REST API."""

    def __init__(self, api_key: str, model: str, timeout: float) -> None:
        self._api_key, self._model, self._timeout = api_key, model, timeout

    def generate(self, prompt: str) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self._model}:generateContent"
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": MAX_OUTPUT_TOKENS},
        }
        data = _post_json(url, payload, {"x-goog-api-key": self._api_key}, self._timeout)
        try:
            candidate = data["candidates"][0]
            # The answer can be split across several parts; join them all.
            parts = candidate["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
            reason = candidate.get("finishReason")
            if reason and reason != "STOP":
                logger.warning("Gemini finishReason=%s (answer may be cut off)", reason)
            return text
        except (KeyError, IndexError, TypeError) as exc:
            logger.error("Unexpected Gemini response shape: %s", str(data)[:300])
            raise LLMError("Unexpected Gemini response") from exc


class OllamaClient:
    """Local model: an Ollama server."""

    def __init__(self, host: str, model: str, timeout: float) -> None:
        self._host, self._model, self._timeout = host.rstrip("/"), model, timeout

    def generate(self, prompt: str) -> str:
        payload = {"model": self._model, "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"num_predict": MAX_OUTPUT_TOKENS},
        }
        data = _post_json(f"{self._host}/api/chat", payload, {}, self._timeout)
        try:
            if data.get("done_reason") not in (None, "stop"):
                logger.warning("Ollama done_reason=%s (answer may be cut off)", data.get("done_reason"))
            return data["message"]["content"].strip()
        except (KeyError, TypeError) as exc:
            logger.error("Unexpected Ollama response shape: %s", str(data)[:300])
            raise LLMError("Unexpected Ollama response") from exc


def build_llm_client(cfg: Settings) -> LLMClient:
    """Pick the client from settings. Fails fast if the configuration is incomplete."""
    if cfg.LLM_PROVIDER == "gemini":
        key = cfg.GEMINI_API_KEY.get_secret_value()
        if not key:
            raise RuntimeError("LLM_PROVIDER=gemini but GEMINI_API_KEY is not set")
        logger.info("LLM provider: gemini (model=%s)", cfg.GEMINI_MODEL)
        return GeminiClient(key, cfg.GEMINI_MODEL, cfg.LLM_TIMEOUT)

    if not cfg.LOCAL_MODEL:
        raise RuntimeError("LLM_PROVIDER=ollama but LOCAL_MODEL is not set")
    logger.info("LLM provider: ollama (model=%s, host=%s)", cfg.LOCAL_MODEL, cfg.OLLAMA_HOST)
    return OllamaClient(cfg.OLLAMA_HOST, cfg.LOCAL_MODEL, cfg.LLM_TIMEOUT)


@lru_cache
def get_llm_client() -> LLMClient:
    """FastAPI dependency: one shared client built from the settings."""
    return build_llm_client(settings)