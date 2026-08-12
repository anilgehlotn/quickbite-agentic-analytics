"""Contract tests for the HTTP layer.

The orchestrator is replaced by a stub throughout: these tests are about the
API's promises - status codes, response shapes, caching, rate limiting and the
guarantee that no traceback ever reaches a client - not about answer quality,
which is what ``test_golden_e2e.py`` is for.

Every test gets its own cache file and a fresh rate limiter, because both are
process-wide singletons and a test that inherited another's state would pass or
fail depending on ordering.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pytest
from fastapi.testclient import TestClient

from app.agents.contracts import (
    AgentStatus,
    AgentStep,
    AgentTrace,
    AnalysisPlan,
    AnalysisResponse,
    ChartSpec,
    ChartType,
    Insight,
    QueryIntent,
    QueryResult,
    SubQuery,
    TimeWindow,
    VerificationReport,
    VerificationStatus,
)
from app.config import settings
from app.core import cache as cache_module

QUESTION = "What was our total revenue in the last 3 months?"


def make_response(question: str = QUESTION) -> AnalysisResponse:
    """Build a complete, successful analysis for the stub to return.

    Args:
        question: The question it answers.

    Returns:
        A valid response with a plan, results, verification, insight and trace.
    """
    plan = AnalysisPlan(
        question=question,
        intent=QueryIntent.AGGREGATE,
        time_window=TimeWindow(
            start_date=settings.LAST_3M_START,
            end_date=settings.LAST_3M_END,
            label="last 3 months",
        ),
        metrics=["revenue"],
        dimensions=[],
        sub_queries=[SubQuery(id="totals", purpose="Total revenue.")],
        requires_diagnostics=False,
        reasoning="One aggregate answers it.",
        confidence=0.95,
    )
    return AnalysisResponse(
        question=question,
        answered=True,
        plan=plan,
        query_results=[
            QueryResult(
                sub_query_id="totals",
                sql="SELECT SUM(net_before_tax) AS revenue_inr FROM fact_orders",
                columns=["revenue_inr"],
                rows=[{"revenue_inr": 3197076.5}],
                row_count=1,
                execution_ms=3.0,
            )
        ],
        verification=VerificationReport(
            status=VerificationStatus.PASSED, checks=[], summary="All checks passed."
        ),
        insight=Insight(
            headline="Revenue was 3,197,076.50 INR.",
            narrative="Trading was steady.",
            key_findings=["Revenue was 3,197,076.50 INR."],
            caveats=["Revenue is tax-exclusive."],
            recommended_actions=[],
            confidence=0.9,
        ),
        chart=ChartSpec(
            chart_type=ChartType.NONE, x_field="", y_fields=[], title="Revenue"
        ),
        trace=AgentTrace(
            steps=[
                AgentStep(
                    agent_name="planner",
                    status=AgentStatus.SUCCEEDED,
                    started_at=datetime.now(timezone.utc),
                    duration_ms=10.0,
                    summary="Planned one query.",
                    llm_provider="gemini",
                    tokens=100,
                )
            ],
            total_duration_ms=10.0,
            total_tokens=100,
            providers_used=["gemini"],
        ),
        data_asof=settings.DATA_ASOF_DATE,
    )


class StubOrchestrator:
    """Stands in for the real pipeline, counting the runs it was asked for."""

    def __init__(self, response: AnalysisResponse | None = None) -> None:
        """Initialise the stub.

        Args:
            response: What to return. Defaults to a successful analysis.
        """
        self.response = response
        self.calls: list[str] = []

    async def run(
        self, question: str, request_id: str | None = None
    ) -> AnalysisResponse:
        """Return the canned response.

        Args:
            question: The question asked, recorded for assertions.
            request_id: Correlation id, ignored.

        Returns:
            The canned response, or a fresh successful one.
        """
        self.calls.append(question)
        return self.response or make_response(question)

    @property
    def call_count(self) -> int:
        """How many analyses were requested.

        Returns:
            The number of runs.
        """
        return len(self.calls)


@pytest.fixture
def isolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[StubOrchestrator]:
    """Give one test its own cache file, limiter and orchestrator stub.

    Args:
        tmp_path: Pytest's per-test temporary directory.
        monkeypatch: Pytest's patcher.

    Yields:
        The stub the API will call.
    """
    monkeypatch.setattr(settings, "CACHE_PATH", tmp_path / "answer_cache.json")
    cache_module.reset_cache_and_limiter()

    from app.api import routes

    stub = StubOrchestrator()
    monkeypatch.setattr(routes, "_orchestrator", stub)
    monkeypatch.setattr(routes, "get_orchestrator", lambda: stub)
    yield stub

    cache_module.reset_cache_and_limiter()


@pytest.fixture
def client(isolated: StubOrchestrator) -> Iterator[TestClient]:
    """A test client wired to the isolated state.

    Server exceptions are not re-raised, so the global handler's response can
    be asserted rather than the exception escaping into the test.

    Args:
        isolated: The isolation fixture, applied first.

    Yields:
        The client.
    """
    from app.main import app

    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_a_valid_question_returns_the_full_analysis(
    client: TestClient, isolated: StubOrchestrator
) -> None:
    """A good question returns 200 and the documented shape."""
    response = client.post("/api/ask", json={"question": QUESTION})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answered"] is True
    assert payload["question"] == QUESTION
    assert payload["plan"]["intent"] == "aggregate"
    assert payload["insight"]["headline"]
    assert payload["verification"]["status"] == "passed"
    assert payload["trace"]["steps"]
    assert payload["data_asof"] == settings.DATA_ASOF_DATE.isoformat()
    assert payload["request_id"]
    assert isolated.call_count == 1


def test_the_request_id_appears_in_the_body_and_the_header(
    client: TestClient,
) -> None:
    """One id links what the user sees to what the logs recorded."""
    response = client.post("/api/ask", json={"question": QUESTION})

    assert response.headers["X-Request-ID"]
    assert response.json()["request_id"]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    ["", "a", "ab", "   ", "\n\t "],
    ids=["empty", "one-char", "two-chars", "spaces", "whitespace"],
)
def test_short_or_blank_questions_are_rejected(
    client: TestClient, question: str, isolated: StubOrchestrator
) -> None:
    """Too short or blank is 422, and never reaches the orchestrator."""
    response = client.post("/api/ask", json={"question": question})

    assert response.status_code == 422
    payload = response.json()
    assert payload["error"] == "invalid_request"
    assert str(settings.MIN_QUESTION_LENGTH) in payload["message"]
    assert payload["request_id"]
    assert isolated.call_count == 0


def test_an_overlong_question_is_rejected(
    client: TestClient, isolated: StubOrchestrator
) -> None:
    """A pasted document is not a question."""
    response = client.post(
        "/api/ask", json={"question": "x" * (settings.MAX_QUESTION_LENGTH + 1)}
    )

    assert response.status_code == 422
    assert response.json()["error"] == "invalid_request"
    assert isolated.call_count == 0


def test_a_question_at_the_length_limit_is_accepted(client: TestClient) -> None:
    """The boundary itself is valid, not off by one."""
    response = client.post(
        "/api/ask", json={"question": "x" * settings.MAX_QUESTION_LENGTH}
    )

    assert response.status_code == 200


def test_a_missing_question_field_is_a_structured_error(
    client: TestClient,
) -> None:
    """A malformed body gets the same shape as every other error."""
    response = client.post("/api/ask", json={"use_cache": True})

    assert response.status_code == 422
    assert response.json()["error"] == "invalid_request"


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def test_a_repeated_question_is_served_from_cache(
    client: TestClient, isolated: StubOrchestrator
) -> None:
    """The second ask costs nothing and says so."""
    first = client.post("/api/ask", json={"question": QUESTION})
    second = client.post("/api/ask", json={"question": QUESTION})

    assert first.json()["from_cache"] is False
    assert second.json()["from_cache"] is True
    assert isolated.call_count == 1


def test_the_cache_ignores_case_punctuation_and_spacing(
    client: TestClient, isolated: StubOrchestrator
) -> None:
    """One question typed three ways is one cache entry."""
    client.post("/api/ask", json={"question": QUESTION})
    variant = client.post(
        "/api/ask", json={"question": f"  {QUESTION.upper().rstrip('?')}  "}
    )

    assert variant.json()["from_cache"] is True
    assert isolated.call_count == 1


def test_a_cached_answer_keeps_its_original_trace(
    client: TestClient,
) -> None:
    """A cached answer still shows which agents produced it."""
    client.post("/api/ask", json={"question": QUESTION})
    cached = client.post("/api/ask", json={"question": QUESTION}).json()

    assert cached["from_cache"] is True
    assert cached["trace"]["steps"]
    assert cached["trace"]["steps"][0]["agent_name"] == "planner"
    assert cached["trace"]["total_tokens"] == 100


def test_use_cache_false_forces_a_fresh_run(
    client: TestClient, isolated: StubOrchestrator
) -> None:
    """A caller can always demand a new analysis."""
    client.post("/api/ask", json={"question": QUESTION})
    fresh = client.post("/api/ask", json={"question": QUESTION, "use_cache": False})

    assert fresh.json()["from_cache"] is False
    assert isolated.call_count == 2


def test_a_failed_analysis_is_not_cached(
    client: TestClient, isolated: StubOrchestrator
) -> None:
    """Caching a failure would serve it long after the cause had cleared."""
    isolated.response = AnalysisResponse.unanswered(
        question=QUESTION, error="every provider failed"
    )

    client.post("/api/ask", json={"question": QUESTION})
    second = client.post("/api/ask", json={"question": QUESTION})

    assert second.json()["from_cache"] is False
    assert isolated.call_count == 2


def test_the_cache_survives_a_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cold start must not lose the warmed answers."""
    monkeypatch.setattr(settings, "CACHE_PATH", tmp_path / "answer_cache.json")
    cache_module.reset_cache_and_limiter()

    first = cache_module.AnswerCache()
    first.put(QUESTION, make_response())

    cache_module.reset_cache_and_limiter()
    second = cache_module.AnswerCache()

    assert len(second) == 1
    restored = second.get(QUESTION)
    assert restored is not None
    assert restored.from_cache is True
    assert restored.insight is not None


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


def test_the_rate_limit_returns_429_with_a_retry_after(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Past the threshold a caller is refused, and told when to return."""
    limiter = cache_module.get_rate_limiter()
    monkeypatch.setattr(limiter, "per_minute", 3)

    for index in range(3):
        allowed = client.post(
            "/api/ask", json={"question": f"unique question number {index}"}
        )
        assert allowed.status_code == 200

    refused = client.post("/api/ask", json={"question": "one question too many"})

    assert refused.status_code == 429
    payload = refused.json()
    assert payload["error"] == "rate_limited"
    assert "per minute" in payload["message"]
    assert payload["request_id"]
    assert int(refused.headers["Retry-After"]) > 0
    assert payload["detail"]["retry_after"] > 0


def test_cached_answers_do_not_consume_the_rate_limit(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cached answer costs nothing, so it must not cost quota.

    This is what keeps the evaluation questions available to a visitor who has
    spent their allowance on their own questions.
    """
    limiter = cache_module.get_rate_limiter()
    monkeypatch.setattr(limiter, "per_minute", 2)

    assert client.post("/api/ask", json={"question": QUESTION}).status_code == 200
    for _ in range(10):
        repeat = client.post("/api/ask", json={"question": QUESTION})
        assert repeat.status_code == 200
        assert repeat.json()["from_cache"] is True


def test_the_daily_limit_is_enforced_separately(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A slow trickle of requests still hits a daily ceiling."""
    limiter = cache_module.get_rate_limiter()
    monkeypatch.setattr(limiter, "per_minute", 100)
    monkeypatch.setattr(limiter, "per_day", 2)

    client.post("/api/ask", json={"question": "first distinct question"})
    client.post("/api/ask", json={"question": "second distinct question"})
    refused = client.post("/api/ask", json={"question": "third distinct question"})

    assert refused.status_code == 429
    assert "per day" in refused.json()["message"]


def test_rate_limits_are_tracked_per_client(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One heavy user must not exhaust everyone else's quota."""
    limiter = cache_module.get_rate_limiter()
    monkeypatch.setattr(limiter, "per_minute", 1)

    first = client.post(
        "/api/ask",
        json={"question": "a question from one client"},
        headers={"X-Forwarded-For": "10.0.0.1"},
    )
    blocked = client.post(
        "/api/ask",
        json={"question": "another question from the same client"},
        headers={"X-Forwarded-For": "10.0.0.1"},
    )
    other = client.post(
        "/api/ask",
        json={"question": "a question from a different client"},
        headers={"X-Forwarded-For": "10.0.0.2"},
    )

    assert first.status_code == 200
    assert blocked.status_code == 429
    assert other.status_code == 200


# ---------------------------------------------------------------------------
# Other endpoints
# ---------------------------------------------------------------------------


def test_the_questions_endpoint_returns_eight_suggestions(
    client: TestClient,
) -> None:
    """The frontend's chips come from one source shared with the tests."""
    response = client.get("/api/questions")

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 8
    assert len(payload["questions"]) == 8
    for entry in payload["questions"]:
        assert entry["id"]
        assert entry["question"].endswith("?")
        assert entry["label"]
        assert entry["cached"] is False


def test_the_questions_endpoint_reports_what_is_cached(
    client: TestClient,
) -> None:
    """A chip can be shown as instant when its answer is already on disk."""
    from app.agents.orchestrator import CANONICAL_QUESTIONS

    first = CANONICAL_QUESTIONS[0]["question"]
    client.post("/api/ask", json={"question": first})

    payload = client.get("/api/questions").json()
    cached = [entry for entry in payload["questions"] if entry["cached"]]

    assert [entry["id"] for entry in cached] == ["q1"]


def test_health_reports_readiness_without_leaking_keys(
    client: TestClient,
) -> None:
    """Health is useful to an operator and useless to an attacker."""
    response = client.get("/api/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"ok", "degraded"}
    assert payload["database_ready"] is True
    assert payload["fact_orders_rows"] > 0
    assert isinstance(payload["orchestrator_ready"], bool)
    assert payload["providers"]
    body = response.text
    for key in (
        settings.ANTHROPIC_API_KEY,
        settings.OPENAI_API_KEY,
        settings.GEMINI_API_KEY,
        settings.GROK_API_KEY,
    ):
        if key:
            assert key not in body


def test_the_root_and_legacy_health_endpoints_still_work(
    client: TestClient,
) -> None:
    """Mounting the router must not break what was already deployed."""
    assert client.get("/").status_code == 200
    assert client.get("/health").status_code == 200
    assert client.get("/health").json()["status"] in {"ok", "degraded"}


def test_the_schema_endpoint_exposes_the_semantic_layer(
    client: TestClient,
) -> None:
    """What the agents are told is inspectable after a deploy."""
    payload = client.get("/api/schema").json()

    assert payload["data_asof"] == settings.DATA_ASOF_DATE.isoformat()
    assert payload["revenue_metric"] == settings.REVENUE_METRIC
    assert "fact_orders" in payload["tables"]
    assert "revenue" in payload["metrics"]


def test_the_verify_endpoint_returns_the_quality_report(
    client: TestClient,
) -> None:
    """The data validation work is a feature, not a buried script."""
    response = client.get("/api/verify")

    assert response.status_code == 200
    payload = response.json()
    assert payload["passed"] is True
    assert payload["total_checks"] > 30
    assert payload["error_count"] == 0
    assert payload["checks"]
    assert all("severity" in check for check in payload["checks"])


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


def test_an_unhandled_error_returns_a_structured_payload(
    client: TestClient, isolated: StubOrchestrator, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bug must never reach a client as a traceback."""

    async def explode(question: str, request_id: str | None = None) -> Any:
        """Fail the way an unexpected bug would.

        Args:
            question: Ignored.
            request_id: Ignored.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError("a genuinely unexpected failure")

    monkeypatch.setattr(isolated, "run", explode)

    response = client.post("/api/ask", json={"question": QUESTION})

    assert response.status_code == 500
    payload = response.json()
    assert payload["error"] == "internal_error"
    assert payload["request_id"]
    assert "Traceback" not in response.text
    assert "genuinely unexpected failure" not in response.text


def test_an_unanswerable_question_is_a_200_with_an_explanation(
    client: TestClient, isolated: StubOrchestrator
) -> None:
    """A refusal is a valid outcome with a trace, not an HTTP error."""
    isolated.response = AnalysisResponse.unanswered(
        question=QUESTION, error="This question cannot be answered from this data."
    )

    response = client.post("/api/ask", json={"question": QUESTION})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answered"] is False
    assert "cannot be answered" in payload["error"]
    assert payload["trace"] is not None


def test_no_provider_available_still_offers_the_cached_questions(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dead-key case names what can still be answered.

    This is the state the deployment will most likely be in when a reviewer
    opens it, so "unavailable" is not an acceptable whole answer.
    """
    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", None)
    monkeypatch.setattr(settings, "OPENAI_API_KEY", None)
    monkeypatch.setattr(settings, "GEMINI_API_KEY", None)
    monkeypatch.setattr(settings, "GROK_API_KEY", None)

    response = client.post("/api/ask", json={"question": "an uncached question"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answered"] is False
    assert "temporarily unavailable" in payload["error"]
    assert "revenue" in payload["error"].lower()
