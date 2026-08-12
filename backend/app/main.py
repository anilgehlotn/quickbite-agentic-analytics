"""FastAPI application entrypoint for QuickBite Agentic Analytics.

This is the deployment skeleton: application metadata, CORS, structured
logging, a health probe and a schema introspection endpoint. Analytical
endpoints are added in later steps.

The health probe deliberately runs a real query against ``fact_orders`` rather
than only checking that the database file exists. A committed SQLite file can be
present but truncated, corrupted by a bad build, or unreadable because of
filesystem permissions on the host; only a query proves the data layer actually
works in the deployed environment.

Run locally with::

    uvicorn app.main:app --reload

or, honouring the configured port::

    python -m app.main
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Awaitable, Callable

from fastapi import FastAPI, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import (
    ErrorResponse,
    RateLimitExceeded,
    RETRY_AFTER_HEADER,
    REQUEST_ID_HEADER,
    new_request_id,
    router,
)
from app.config import settings
from app.core.cache import get_cache
from app.core.llm import get_llm_client
from app.core.logging import configure_logging, get_logger
from app.semantic.schema import METRIC_DEFINITIONS, TABLE_ALLOWLIST, get_compact_schema

configure_logging()
logger = get_logger(__name__)

# Health status values.
STATUS_OK = "ok"
STATUS_DEGRADED = "degraded"


def count_orders() -> tuple[int | None, str | None]:
    """Count the rows in ``fact_orders`` to prove the database is readable.

    Returns:
        The row count and ``None`` on success, or ``None`` and an error message
        when the database is missing or the query fails.
    """
    if not settings.DB_PATH.exists():
        return None, f"database file not found at {settings.DB_PATH}"
    try:
        connection = sqlite3.connect(f"file:{settings.DB_PATH}?mode=ro", uri=True)
        try:
            count = connection.execute("SELECT COUNT(*) FROM fact_orders").fetchone()[0]
            return int(count), None
        finally:
            connection.close()
    except sqlite3.Error as error:
        return None, f"{type(error).__name__}: {error}"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Log the resolved runtime configuration on startup.

    Deployments fail quietly when an environment variable is missing or a file
    did not ship. Logging the resolved state once at startup makes the first few
    lines of the deploy log enough to diagnose that.

    Args:
        app: The application being started.

    Yields:
        Control back to the server for the lifetime of the application.
    """
    row_count, error = count_orders()
    # Loading the cache at startup rather than on first request means a cold
    # start pays the file read once, and the startup log states plainly how
    # many questions are answerable without a provider.
    cached = len(get_cache())
    logger.info(
        "startup",
        extra={
            "app": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT,
            "port": settings.PORT,
            "database_path": str(settings.DB_PATH),
            "database_ready": error is None,
            "fact_orders_rows": row_count,
            "database_error": error,
            "providers_configured": settings.available_providers(),
            "cors_origins": settings.CORS_ORIGINS,
            "data_asof": settings.DATA_ASOF_DATE.isoformat(),
            "cached_answers": cached,
        },
    )
    if error:
        logger.warning("database not readable; /health will report degraded")
    if not settings.available_providers():
        logger.warning(
            "no LLM provider keys configured; only cached answers are available",
            extra={"cached_answers": cached},
        )

    # Probe the providers in the background. Deliberately not awaited: a
    # provider that is slow to answer must not delay the port opening, because
    # the platform's health check has its own deadline and the cached answers
    # are servable before any provider is known to work. The task is kept
    # referenced so it cannot be garbage collected mid-flight.
    probe_task: asyncio.Task[Any] | None = None
    if settings.PROVIDER_PROBE_ON_STARTUP and settings.available_providers():
        probe_task = asyncio.create_task(_probe_providers_quietly())

    yield

    if probe_task is not None and not probe_task.done():
        probe_task.cancel()
    logger.info("shutdown")


async def _probe_providers_quietly() -> None:
    """Run the startup provider probe, swallowing every failure.

    A probe exists to make the *first user request* fast by discovering dead
    providers in advance. It is an optimisation, so nothing it does may ever
    prevent the application running.
    """
    try:
        await get_llm_client().probe_providers(force=True)
    except asyncio.CancelledError:
        raise
    except Exception as error:  # noqa: BLE001 - a probe must never break startup
        logger.warning("provider_probe_error", extra={"error": str(error)[:200]})


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Log every request with its outcome and duration.

    The request id is generated here when the route did not supply one, and is
    echoed in a header on every response, so a user reporting "it failed" can
    quote an id that appears verbatim in the server logs.

    Args:
        request: The incoming request.
        call_next: The rest of the middleware and routing stack.

    Returns:
        The response, with the correlation header attached.
    """
    request_id = request.headers.get(REQUEST_ID_HEADER) or new_request_id()
    request.state.request_id = request_id
    started = time.perf_counter()

    response = await call_next(request)

    duration_ms = (time.perf_counter() - started) * 1000
    response.headers.setdefault(REQUEST_ID_HEADER, request_id)
    logger.info(
        "request",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "duration_ms": round(duration_ms, 1),
        },
    )
    return response


def error_payload(
    request: Request, code: str, message: str, detail: Any | None = None
) -> ErrorResponse:
    """Build a structured error carrying the request's correlation id.

    Args:
        request: The request being answered.
        code: Short machine-readable error code.
        message: Plain-language explanation.
        detail: Optional structured context.

    Returns:
        The error model.
    """
    return ErrorResponse(
        error=code,
        message=message,
        request_id=getattr(request.state, "request_id", "unknown"),
        detail=detail,
    )


@app.exception_handler(RateLimitExceeded)
async def handle_rate_limit(
    request: Request, exception: RateLimitExceeded
) -> JSONResponse:
    """Return 429 with a retry delay the client can act on.

    Args:
        request: The refused request.
        exception: The limit that was hit.

    Returns:
        A structured 429 carrying Retry-After.
    """
    payload = error_payload(
        request,
        "rate_limited",
        exception.message,
        {"retry_after": exception.retry_after},
    )
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content=payload.model_dump(),
        headers={
            RETRY_AFTER_HEADER: str(exception.retry_after),
            REQUEST_ID_HEADER: payload.request_id,
        },
    )


@app.exception_handler(RequestValidationError)
async def handle_validation_error(
    request: Request, exception: RequestValidationError
) -> JSONResponse:
    """Return 422 explaining what was wrong with the input.

    FastAPI's default body is a bare list of pydantic errors. Wrapping it keeps
    one response shape across the API and names the constraint in a sentence a
    user can act on, while the raw errors stay available under ``detail``.

    Args:
        request: The rejected request.
        exception: The validation failure.

    Returns:
        A structured 422.
    """
    payload = error_payload(
        request,
        "invalid_request",
        (
            f"The question must be between {settings.MIN_QUESTION_LENGTH} and "
            f"{settings.MAX_QUESTION_LENGTH} characters and cannot be blank."
        ),
        exception.errors(),
    )
    logger.warning(
        "request_invalid",
        extra={"request_id": payload.request_id, "path": request.url.path},
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=json_safe(payload.model_dump()),
        headers={REQUEST_ID_HEADER: payload.request_id},
    )


@app.exception_handler(Exception)
async def handle_unexpected(request: Request, exception: Exception) -> JSONResponse:
    """Return a structured 500 instead of a traceback.

    A stack trace in an HTTP response is both a poor user experience and an
    information leak. The trace goes to the logs under the request id; the
    caller gets that id and a sentence.

    Args:
        request: The failed request.
        exception: The unhandled error.

    Returns:
        A structured 500.
    """
    payload = error_payload(
        request,
        "internal_error",
        (
            "The request could not be completed because of an unexpected "
            "error. Quote the request id if you report this."
        ),
    )
    logger.error(
        "unhandled_exception",
        extra={
            "request_id": payload.request_id,
            "path": request.url.path,
            "error": str(exception),
        },
        exc_info=True,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=payload.model_dump(),
        headers={REQUEST_ID_HEADER: payload.request_id},
    )


def json_safe(value: Any) -> Any:
    """Coerce a payload into something ``json.dumps`` accepts.

    Pydantic validation errors can carry exception instances and byte strings
    under ``ctx``, which are not JSON-serializable; stringifying them keeps the
    detail useful without failing the response.

    Args:
        value: The payload to coerce.

    Returns:
        The same structure with unserializable leaves stringified.
    """
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


app.include_router(router)


@app.get("/")
def read_root() -> dict[str, Any]:
    """Identify the service.

    Returns:
        The application name, version and a running indicator.
    """
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
    }


@app.get("/health")
def read_health() -> dict[str, Any]:
    """Report service readiness, including a live database query.

    Always responds 200 so the service stays in rotation and an operator can
    read the reason for a problem; the ``status`` field carries the verdict.
    ``degraded`` means the process is up but the data layer is not usable.

    Returns:
        Overall status, database readiness and row count, the dataset as-of
        date, configured providers and the environment name.
    """
    row_count, error = count_orders()
    payload: dict[str, Any] = {
        "status": STATUS_OK if error is None else STATUS_DEGRADED,
        "database_ready": error is None,
        "fact_orders_rows": row_count,
        "data_asof": settings.DATA_ASOF_DATE.isoformat(),
        "providers_configured": settings.available_providers(),
        "environment": settings.ENVIRONMENT,
        "version": settings.APP_VERSION,
    }
    if error:
        payload["database_error"] = error
    return payload


@app.get("/api/providers")
def read_provider_health() -> dict[str, Any]:
    """Report which providers are configured, healthy or in cooldown.

    Split out from ``/api/health`` because it is an operator's view rather than
    a platform probe: the platform only needs to know whether to keep the
    service in rotation, while a human debugging a slow answer needs to know
    which provider is being skipped and for how long.

    No credential material appears in the output, not even a masked prefix.

    Returns:
        The client's health view, including per-provider breaker state and the
        last probe result.
    """
    return get_llm_client().health()


@app.post("/api/providers/probe")
async def probe_providers() -> dict[str, Any]:
    """Re-probe every configured provider and return the outcome.

    Exposed so a dead provider can be brought back without a redeploy: fix the
    key, call this, and the breaker closes on the next success rather than
    after the cooldown.

    Returns:
        Per-provider probe results and the resulting health view.
    """
    outcome = await get_llm_client().probe_providers(force=True)
    return {"probe": outcome, "health": get_llm_client().health()}


@app.get("/api/schema")
def read_schema() -> dict[str, Any]:
    """Expose the semantic layer's compact schema.

    Useful for confirming after a deploy that the semantic layer imported and
    that the agents will see the schema they expect.

    Returns:
        The compact schema block, the metric names, the table allowlist and the
        time anchor.
    """
    return {
        "compact_schema": get_compact_schema(),
        "metrics": list(METRIC_DEFINITIONS),
        "tables": TABLE_ALLOWLIST,
        "data_asof": settings.DATA_ASOF_DATE.isoformat(),
        "data_start": settings.DATA_START_DATE.isoformat(),
        "revenue_metric": settings.REVENUE_METRIC,
    }


def main() -> None:
    """Run the development server on the configured port."""
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.PORT,
        log_config=None,
    )


if __name__ == "__main__":
    main()
