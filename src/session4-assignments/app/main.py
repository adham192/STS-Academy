import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.errors import register_exception_handlers
from app.api.routes import chat, health
from app.core.config import settings
from app.core.logging_config import setup_logging
from app.db.session import init_db
from app.middleware.request_logging import RequestLoggingMiddleware
from app.services.llm_clients import get_llm_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()  # create tables on startup
    yield


def create_app() -> FastAPI:
    setup_logging()  # first thing, so every later log is captured
    logger = logging.getLogger(__name__)

    get_llm_client()  # build the LLM client now: bad config fails at startup, not on the first request

    app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)
    app.add_middleware(RequestLoggingMiddleware)
    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(chat.router, prefix="/api/v1")

    logger.info("%s started", settings.APP_NAME)
    return app


app = create_app()