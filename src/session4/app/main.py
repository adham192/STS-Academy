import logging

from fastapi import FastAPI

from app.api.routes import chat, health
from app.core.config import settings
from app.core.logging_config import setup_logging
from app.middleware.request_logging import RequestLoggingMiddleware
from app.services.chat_service import get_chat_service


def create_app() -> FastAPI:
    setup_logging()  # first thing, so every later log is captured
    logger = logging.getLogger(__name__)

    get_chat_service()  # build the LLM client now: bad config fails at startup, not on first request

    app = FastAPI(title=settings.APP_NAME)
    app.add_middleware(RequestLoggingMiddleware)
    app.include_router(health.router)
    app.include_router(chat.router, prefix="/api/v1")

    logger.info("%s started", settings.APP_NAME)
    return app


app = create_app()