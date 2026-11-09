import logging.config
from contextvars import ContextVar

from app.core.config import settings

# Holds the current request's ID so every log line written during it can include it.
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(request_id)s | %(name)s:%(lineno)d | %(message)s"


class RequestIdFilter(logging.Filter):
    """Stamps each record with the current request ID."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get()
        return True


def _file_handler(filename: str, level: str) -> dict:
    """Rotating file handler: 5 MB per file, 5 backups kept."""
    return {
        "class": "logging.handlers.RotatingFileHandler",
        "filename": str(settings.LOG_DIR / filename),
        "maxBytes": 5_000_000,
        "backupCount": 5,
        "encoding": "utf-8",
        "level": level,
        "formatter": "standard",
        "filters": ["request_id"],
    }


def setup_logging() -> None:
    """Configure logging for the whole application (call once at startup)."""
    settings.LOG_DIR.mkdir(parents=True, exist_ok=True)

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {"request_id": {"()": RequestIdFilter}},
        "formatters": {"standard": {"format": LOG_FORMAT}},
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "DEBUG",
                "formatter": "standard",
                "filters": ["request_id"],
            },
            "app_file": _file_handler("app.log", "DEBUG"),       # everything
            "error_file": _file_handler("error.log", "ERROR"),   # errors only
            "access_file": _file_handler("access.log", "INFO"),  # one line per HTTP request
        },
        "loggers": {
            # Application code: logging.getLogger(__name__) under "app.*"
            "app": {
                "level": settings.LOG_LEVEL,
                "handlers": ["console", "app_file", "error_file"],
                "propagate": False,
            },
            # HTTP request lines (written by the request-logging middleware)
            "app.access": {
                "level": "INFO",
                "handlers": ["console", "access_file"],
                "propagate": False,
            },
            # Route uvicorn's own messages into the same files
            "uvicorn": {
                "level": "INFO",
                "handlers": ["console", "app_file", "error_file"],
                "propagate": False,
            },
            # Silenced: our middleware already logs every request
            "uvicorn.access": {"handlers": [], "propagate": False},
            # SQL statements: only visible when LOG_LEVEL=DEBUG
            "sqlalchemy.engine": {
                "level": "INFO" if settings.LOG_LEVEL == "DEBUG" else "WARNING",
                "handlers": ["app_file"],
                "propagate": False,
            },
        },
        # Third-party libraries only speak up for warnings and above
        "root": {"level": "WARNING", "handlers": ["console", "app_file", "error_file"]},
    }
    logging.config.dictConfig(config)