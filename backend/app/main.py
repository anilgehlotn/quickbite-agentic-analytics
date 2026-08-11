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

import sqlite3
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
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
        },
    )
    if error:
        logger.warning("database not readable; /health will report degraded")
    if not settings.available_providers():
        logger.warning("no LLM provider keys configured")
    yield
    logger.info("shutdown")


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
