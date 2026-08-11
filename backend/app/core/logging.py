"""Structured logging for the QuickBite backend.

Emits one JSON object per line so hosting platforms that capture stdout (Render,
Fly, Cloud Run) produce logs that can be filtered and queried rather than
grepped. Timestamps are UTC and ISO 8601 so entries sort correctly regardless of
where the container runs.

Typical use::

    from app.core.logging import get_logger

    logger = get_logger(__name__)
    logger.info("database ready", extra={"rows": 20000})
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Final

from app.config import settings

# Attributes present on every LogRecord. Anything outside this set was supplied
# by the caller through `extra=` and is merged into the JSON payload.
_RESERVED_ATTRS: Final[frozenset[str]] = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)

# Set once so repeated imports do not stack duplicate handlers on the root
# logger, which would print every line more than once.
_configured: bool = False


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        """Render one record.

        Args:
            record: The record to format.

        Returns:
            A single-line JSON object carrying the timestamp, level, logger
            name, message, any ``extra`` fields and the exception text when one
            is attached.
        """
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str | None = None) -> None:
    """Install the JSON formatter on the root logger.

    Safe to call more than once; the handler is only attached on the first call.

    Args:
        level: Log level name. Defaults to ``settings.LOG_LEVEL``.
    """
    global _configured
    if _configured:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel((level or settings.LOG_LEVEL).upper())

    # uvicorn installs its own handlers; clearing them and letting records
    # propagate to root keeps every line in one format.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger.

    Args:
        name: Logger name, conventionally the calling module's ``__name__``.

    Returns:
        A logger writing single-line JSON to stdout.
    """
    configure_logging()
    return logging.getLogger(name)
