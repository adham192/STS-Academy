import logging
import logging.config
from contextvars import ContextVar

from app.core.config import settings

# Holds the current request's ID so every log line can include it.
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get()
        return True


def setup_logging() -> None:
    settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_dir = settings.LOG_DIR

    def file_handler(filename: str, level: str) -> dict:
        return {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(log_dir / filename),
            "maxBytes": 5_000_000,
            "backupCount": 5,
            "encoding": "utf-8",
            "level": level,
            "formatter": "standard",
            "filters": ["request_id"],
        }

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {"request_id": {"()": RequestIdFilter}},
        "formatters": {
            "standard": {
                "format": "%(asctime)s | %(levelname)-8s | %(request_id)s | %(name)s:%(lineno)d | %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "DEBUG",
                "formatter": "standard",
                "filters": ["request_id"],
            },
            "app_file": file_handler("app.log", "DEBUG"),      # everything
            "error_file": file_handler("error.log", "ERROR"),  # errors only
            "access_file": file_handler("access.log", "INFO"), # HTTP requests
        },
        "loggers": {
            # Your application code: logging.getLogger(__name__) under "app.*"
            "app": {
                "level": settings.LOG_LEVEL,
                "handlers": ["console", "app_file", "error_file"],
                "propagate": False,
            },
            # One line per HTTP request (written by the middleware)
            "app.access": {
                "level": "INFO",
                "handlers": ["console", "access_file"],
                "propagate": False,
            },
            # Route uvicorn's own logs into the same files
            "uvicorn": {
                "level": "INFO",
                "handlers": ["console", "app_file", "error_file"],
                "propagate": False,
            },
            # Silenced: our middleware already logs requests
            "uvicorn.access": {"handlers": [], "propagate": False},
        },
        "root": {
            "level": "WARNING",
            "handlers": ["console", "app_file", "error_file"],
        },
    }
    logging.config.dictConfig(config)