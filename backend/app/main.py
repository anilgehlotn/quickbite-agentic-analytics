"""FastAPI application entrypoint for QuickBite Agentic Analytics.

This is a deployment skeleton: it wires up the application metadata, CORS and a
health probe only. Analytical endpoints are added in later steps.

Run locally with::

    uvicorn app.main:app --reload
"""

from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
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
    """Report service readiness.

    Surfaces whether the SQLite database has been built, which LLM providers
    have credentials, and the dataset's fixed as-of date so clients can confirm
    the time anchor they are querying against.

    Returns:
        Health details: overall status, database readiness, the dataset as-of
        date in ISO format, configured providers and the environment name.
    """
    return {
        "status": "ok",
        "database_ready": settings.DB_PATH.exists(),
        "data_asof": settings.DATA_ASOF_DATE.isoformat(),
        "providers_configured": settings.available_providers(),
        "environment": settings.ENVIRONMENT,
    }
