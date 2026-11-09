from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.services.llm_clients import LLMError


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(LLMError)
    async def llm_error_handler(request: Request, exc: LLMError) -> JSONResponse:
        # The client already logged the real cause to error.log
        return JSONResponse(status_code=502, content={"detail": "The language model is unavailable."})