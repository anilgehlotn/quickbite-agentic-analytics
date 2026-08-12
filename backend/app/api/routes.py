"""HTTP endpoints.

Every response on this router is a typed pydantic model, including the errors.
That is worth the extra classes: a frontend that can rely on one response shape
never has to branch on whether the payload is an answer or an exception, and a
client written against these models cannot be surprised by a stray dict.

The ordering inside ``/api/ask`` is the interesting part, and it follows the
cost of each step. Validation is free, so it runs first. The cache is free, so
it runs second - and because a cache hit costs nothing, it deliberately does
not consume anyone's rate-limit quota. Only work that will actually reach a
provider is charged against the limit. The consequence is that the eight
evaluation questions stay available to a visitor who has exhausted their quota
on their own questions, which is the behaviour you want when the cached
answers are the ones being evaluated.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Final

from fastapi import APIRouter, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.agents.contracts import AnalysisResponse
from app.agents.orchestrator import CANONICAL_QUESTIONS, Orchestrator
from app.config import settings
from app.core.cache import get_cache, get_rate_limiter, normalise_question
from app.core.llm import get_llm_client
from app.core.logging import get_logger
from app.etl.quality_checks import run_quality_checks
from app.semantic.schema import (
    METRIC_DEFINITIONS,
    TABLE_ALLOWLIST,
    get_compact_schema,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["analytics"])

# Header carrying the retry delay on a 429, per RFC 9110.
RETRY_AFTER_HEADER: Final[str] = "Retry-After"

# Header echoing the request id, so a user can quote it from the browser's
# network tab without opening the response body.
REQUEST_ID_HEADER: Final[str] = "X-Request-ID"

STATUS_OK: Final[str] = "ok"
STATUS_DEGRADED: Final[str] = "degraded"


def new_request_id() -> str:
    """Generate a short correlation id.

    Returns:
        Twelve hex characters, long enough to be unique within a deployment's
        logs and short enough for a person to read aloud.
    """
    return uuid.uuid4().hex[:12]


def client_ip(request: Request) -> str:
    """Identify the caller for rate limiting.

    Behind a platform proxy the socket address is the proxy, so the first
    entry of ``X-Forwarded-For`` is used when present. It is client-supplied
    and therefore spoofable; that is acceptable for a cost guard, and is the
    reason this is not treated as a security control.

    Args:
        request: The incoming request.

    Returns:
        The client's IP address, or "unknown".
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# ---------------------------------------------------------------------------
# Request and response models
# ---------------------------------------------------------------------------


class AskRequest(BaseModel):
    """A question to analyse."""

    question: str = Field(
        min_length=settings.MIN_QUESTION_LENGTH,
        max_length=settings.MAX_QUESTION_LENGTH,
        description=(
            "The business question in natural language, "
            f"{settings.MIN_QUESTION_LENGTH} to "
            f"{settings.MAX_QUESTION_LENGTH} characters."
        ),
    )
    use_cache: bool = Field(
        default=True,
        description=(
            "Whether a previously computed answer may be served. Set false to "
            "force a fresh run, which consumes rate-limit quota."
        ),
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": (
                    "What was our total revenue, order count and AOV in the "
                    "last 3 months?"
                ),
                "use_cache": True,
            }
        }
    )

    @field_validator("question")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        """Reject a question that is only whitespace.

        ``min_length`` counts characters, so three spaces satisfies it while
        carrying no question at all. Checked after stripping.

        Args:
            value: The submitted question.

        Returns:
            The question, unchanged.

        Raises:
            ValueError: If the question is blank once stripped.
        """
        if len(value.strip()) < settings.MIN_QUESTION_LENGTH:
            raise ValueError(
                f"question must contain at least "
                f"{settings.MIN_QUESTION_LENGTH} non-whitespace characters"
            )
        return value


class QuestionSuggestion(BaseModel):
    """One suggested question for the frontend's chips."""

    id: str = Field(description="Stable identifier, matching the golden answers.")
    question: str = Field(description="The question text to submit verbatim.")
    label: str = Field(description="Short label for the chip.")
    cached: bool = Field(
        description=(
            "Whether this question already has a cached answer, and will "
            "therefore respond instantly and without a provider."
        )
    )


class QuestionsResponse(BaseModel):
    """The canonical evaluation questions."""

    questions: list[QuestionSuggestion] = Field(
        description="The suggested questions, in evaluation order."
    )
    count: int = Field(description="How many questions are offered.")


class ProviderHealth(BaseModel):
    """Configuration state of one LLM provider."""

    name: str = Field(description="Provider name.")
    configured: bool = Field(
        description="Whether an API key is present. Key values are never shown."
    )
    model: str = Field(
        description=(
            "The model id this provider would use. Exposed because a stale "
            "model id is a silent provider outage, and reading it here is "
            "faster than reading a failover error."
        )
    )


class HealthResponse(BaseModel):
    """Service readiness."""

    status: str = Field(description="'ok' or 'degraded'.")
    database_ready: bool = Field(description="Whether a live query succeeded.")
    fact_orders_rows: int | None = Field(
        default=None, description="Row count proving the data layer works."
    )
    orchestrator_ready: bool = Field(
        description=(
            "Whether the pipeline can answer a new question, which requires "
            "both a readable database and at least one configured provider."
        )
    )
    providers: list[ProviderHealth] = Field(
        description="Every known provider and whether it is configured."
    )
    providers_configured: list[str] = Field(
        description="Configured providers in failover order."
    )
    cached_answers: int = Field(
        description=(
            "How many answers are cached. Non-zero means the evaluation "
            "questions are answerable even with no provider available."
        )
    )
    data_asof: str = Field(description="The dataset's fixed as-of date.")
    environment: str = Field(description="Deployment environment name.")
    version: str = Field(description="Application version.")
    database_error: str | None = Field(
        default=None, description="Why the database is unusable, when it is."
    )


class SchemaResponse(BaseModel):
    """The semantic layer, as the agents see it."""

    compact_schema: str = Field(description="The compact schema block.")
    metrics: list[str] = Field(description="Canonical metric names.")
    tables: list[str] = Field(description="Tables an agent may query.")
    data_asof: str = Field(description="The dataset's fixed as-of date.")
    data_start: str = Field(description="First date covered by the dataset.")
    revenue_metric: str = Field(description="The canonical revenue column.")


class VerifyResponse(BaseModel):
    """The data quality gate's report."""

    passed: bool = Field(description="Whether the gate passed with no errors.")
    total_checks: int = Field(description="How many checks ran.")
    error_count: int = Field(description="Failing error-severity checks.")
    warning_count: int = Field(description="Failing warning-severity checks.")
    info_count: int = Field(description="Informational observations.")
    categories: dict[str, list[dict[str, Any]]] = Field(
        description="Checks grouped by category, in execution order."
    )
    checks: list[dict[str, Any]] = Field(description="Every check, in order.")


class ErrorResponse(BaseModel):
    """A structured failure. Never a traceback."""

    error: str = Field(description="Short machine-readable error code.")
    message: str = Field(description="What went wrong, in plain language.")
    request_id: str = Field(
        description="Correlation id, quotable when reporting the problem."
    )
    detail: Any | None = Field(
        default=None, description="Optional structured context."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "rate_limited",
                "message": "Rate limit reached: 10 live analyses per minute.",
                "request_id": "9f2c1a4b7e08",
                "detail": {"retry_after": 42},
            }
        }
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

class RateLimitExceeded(Exception):
    """A caller exceeded their request quota.

    Raised rather than returned so the global handler owns the status code and
    the Retry-After header in one place.

    Attributes:
        message: Why the request was refused.
        retry_after: Seconds until a retry could succeed.
        request_id: Correlation id for the logs.
    """

    def __init__(self, message: str, retry_after: int, request_id: str) -> None:
        """Initialise the error.

        Args:
            message: Why the request was refused.
            retry_after: Seconds until a retry could succeed.
            request_id: Correlation id.
        """
        super().__init__(message)
        self.message = message
        self.retry_after = retry_after
        self.request_id = request_id


_orchestrator: Orchestrator | None = None


def get_orchestrator() -> Orchestrator:
    """Return the shared orchestrator, building it on first use.

    Constructing it lazily keeps import of this module free of side effects,
    which matters because the agents build large prompts at construction.

    Returns:
        The process-wide orchestrator.
    """
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator


def unavailable_response(question: str, request_id: str) -> AnalysisResponse:
    """Explain that live analysis cannot run, and offer what can.

    Reached when a question is not cached and no provider is configured or
    reachable. Returning a bare error would be accurate and useless; naming the
    questions that are answerable from cache turns a dead end into a next step.

    Args:
        question: The question that could not be run.
        request_id: Correlation id for the logs.

    Returns:
        An unanswered response carrying the explanation.
    """
    cached = get_cache().questions()
    suggestions = cached or [entry["question"] for entry in CANONICAL_QUESTIONS]
    return AnalysisResponse.unanswered(
        question=question,
        error=(
            "Live analysis is temporarily unavailable: no language model "
            "provider is currently reachable, so a new question cannot be "
            "planned or queried. Previously computed answers are still "
            "served, including these: "
            + "; ".join(suggestions[:8])
        ),
    )


@router.post(
    "/ask",
    response_model=AnalysisResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Invalid question."},
        429: {"model": ErrorResponse, "description": "Rate limit reached."},
    },
)
async def ask(
    payload: AskRequest, request: Request, response: Response
) -> AnalysisResponse:
    """Answer one business question.

    Args:
        payload: The question and cache preference.
        request: The incoming request, for the client address.
        response: The outgoing response, for headers.

    Returns:
        The full analysis, including the agent trace. A failure is expressed as
        ``answered=False`` with an explanation rather than as an HTTP error,
        because the run producing no answer is still a valid outcome with a
        trace worth showing.
    """
    request_id = new_request_id()
    response.headers[REQUEST_ID_HEADER] = request_id
    question = payload.question.strip()
    started = time.perf_counter()

    logger.info(
        "ask_received",
        extra={
            "request_id": request_id,
            "question": question,
            "use_cache": payload.use_cache,
            "client": client_ip(request),
        },
    )

    cache = get_cache()
    if payload.use_cache and settings.CACHE_ENABLED:
        cached = cache.get(question)
        if cached is not None:
            logger.info(
                "ask_served_from_cache",
                extra={"request_id": request_id, "question": question},
            )
            cached.request_id = request_id
            return cached

    limiter = get_rate_limiter()
    decision = limiter.check(client_ip(request))
    if not decision.allowed:
        logger.warning(
            "ask_rate_limited",
            extra={"request_id": request_id, "client": client_ip(request)},
        )
        raise RateLimitExceeded(decision.reason, decision.retry_after, request_id)

    if not settings.available_providers():
        logger.warning("ask_no_provider", extra={"request_id": request_id})
        unavailable = unavailable_response(question, request_id)
        unavailable.request_id = request_id
        return unavailable

    result = await get_orchestrator().run(question, request_id=request_id)
    result.request_id = request_id

    if result.answered and settings.CACHE_ENABLED:
        cache.put(question, result)

    logger.info(
        "ask_completed",
        extra={
            "request_id": request_id,
            "answered": result.answered,
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            "tokens": result.trace.total_tokens,
        },
    )
    return result


@router.get("/questions", response_model=QuestionsResponse)
def list_questions() -> QuestionsResponse:
    """List the canonical evaluation questions.

    Returns:
        The eight questions, each flagged with whether it is already cached so
        the frontend can show which will answer instantly.
    """
    cached_keys = set(get_cache().keys())
    return QuestionsResponse(
        questions=[
            QuestionSuggestion(
                id=entry["id"],
                question=entry["question"],
                label=entry["label"],
                cached=normalise_question(entry["question"]) in cached_keys,
            )
            for entry in CANONICAL_QUESTIONS
        ],
        count=len(CANONICAL_QUESTIONS),
    )


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report readiness of the data layer, the providers and the pipeline.

    Always 200 so the service stays in rotation and an operator can read the
    reason; ``status`` carries the verdict.

    Returns:
        Database, provider, orchestrator and cache readiness.
    """
    from app.main import count_orders

    row_count, error = count_orders()
    provider_health = get_llm_client().health()
    configured = settings.available_providers()
    ready = error is None and bool(configured)

    return HealthResponse(
        status=STATUS_OK if error is None else STATUS_DEGRADED,
        database_ready=error is None,
        fact_orders_rows=row_count,
        orchestrator_ready=ready,
        providers=[
            ProviderHealth(name=name, configured=name in configured, model=model)
            for name, model in provider_health.get("models", {}).items()
        ],
        providers_configured=configured,
        cached_answers=len(get_cache()),
        data_asof=settings.DATA_ASOF_DATE.isoformat(),
        environment=settings.ENVIRONMENT,
        version=settings.APP_VERSION,
        database_error=error,
    )


@router.get("/schema", response_model=SchemaResponse)
def read_schema() -> SchemaResponse:
    """Expose the semantic layer the agents are given.

    Returns:
        The compact schema, the metric catalogue and the table allowlist.
    """
    return SchemaResponse(
        compact_schema=get_compact_schema(),
        metrics=list(METRIC_DEFINITIONS),
        tables=list(TABLE_ALLOWLIST),
        data_asof=settings.DATA_ASOF_DATE.isoformat(),
        data_start=settings.DATA_START_DATE.isoformat(),
        revenue_metric=settings.REVENUE_METRIC,
    )


@router.get("/verify", response_model=VerifyResponse)
def verify() -> VerifyResponse:
    """Run the data quality gate and return its report.

    Exposed as a first-class endpoint rather than left as a build-time script,
    because "the numbers are right" is a claim, and this is the evidence for
    it. It also proves the gate passes against the database that actually
    shipped, not the one on the author's machine.

    Returns:
        The full report: the verdict, the counts and every individual check.
    """
    report = run_quality_checks()
    payload = report.to_dict()
    return VerifyResponse(
        passed=payload["passed"],
        total_checks=payload["total_checks"],
        error_count=payload["error_count"],
        warning_count=payload["warning_count"],
        info_count=payload["info_count"],
        categories=payload["categories"],
        checks=payload["checks"],
    )
