"""Tests for the orchestrator.

One scripted LLM stands in for every provider call the pipeline makes - the
planner's JSON, the analyst's SQL, the verifier's escalation and the insight's
narrative - dispatched on the system prompt. SQL still runs against the real
database, because a fake executor would make the self-healing test vacuous:
the point of that test is that a genuine verification failure produces
different SQL on the second attempt.

The emphasis throughout is on the degradation paths. A pipeline that works when
everything works is not interesting; this one has to produce a valid response
when the planner refuses, when every query fails, when half of them fail, when
the numbers do not verify, and when an agent hangs.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

import pytest

from app.agents.analyst import SQLAnalystAgent
from app.agents.base import AgentError
from app.agents.contracts import AgentStatus, AnalysisPlan, QueryIntent
from app.agents.insight import InsightAgent
from app.agents.orchestrator import CANONICAL_QUESTIONS, Orchestrator
from app.agents.planner import PlannerAgent
from app.agents.verifier import VerifierAgent
from app.config import settings
from app.core.llm import LLMResponse

GOOD_SQL = (
    "SELECT ROUND(SUM(net_before_tax), 2) AS revenue_inr, "
    "COUNT(DISTINCT order_id) AS orders, "
    "ROUND(SUM(net_before_tax) / COUNT(DISTINCT order_id), 2) AS aov_inr "
    "FROM fact_orders WHERE order_date BETWEEN '2026-05-01' AND '2026-07-31'"
)
BROKEN_SQL = "SELECT nonexistent_column FROM fact_orders"
NEGATIVE_SQL = "SELECT -1.0 AS revenue_inr FROM fact_orders"


def plan_payload(**overrides: Any) -> dict[str, Any]:
    """Build a valid plan payload for the scripted planner.

    Args:
        **overrides: Plan fields to replace.

    Returns:
        The payload.
    """
    payload: dict[str, Any] = {
        "question": "What was revenue in the last 3 months?",
        "intent": "aggregate",
        "time_window": {
            "start_date": settings.LAST_3M_START.isoformat(),
            "end_date": settings.LAST_3M_END.isoformat(),
            "label": "last 3 months",
            "comparison_start": None,
            "comparison_end": None,
        },
        "metrics": ["revenue", "orders", "aov"],
        "dimensions": [],
        "sub_queries": [
            {
                "id": "totals",
                "purpose": "Total revenue, orders and AOV for the window.",
                "tables": ["fact_orders"],
                "metrics": ["revenue", "orders", "aov"],
                "dimensions": [],
                "filters": {},
            }
        ],
        "requires_diagnostics": False,
        "reasoning": "A single aggregate answers the question.",
        "confidence": 0.95,
    }
    payload.update(overrides)
    return payload


INSIGHT_PAYLOAD: dict[str, Any] = {
    "insight": {
        "headline": "Trading held steady across the last three months.",
        "narrative": "Order volume and basket size were both flat.",
        "key_findings": ["Average order value was flat across the window."],
        "caveats": ["Revenue is tax-exclusive."],
        "recommended_actions": [],
        "confidence": 0.9,
    },
    "chart": {
        "chart_type": "none",
        "x_field": "",
        "y_fields": [],
        "title": "Revenue, orders and AOV",
        "series_field": None,
    },
}


class ScriptedLLM:
    """One fake client serving every agent in the pipeline.

    Dispatches on the system prompt, which is how the real client would see
    the difference too: each agent has its own persona sentence.
    """

    def __init__(
        self,
        plan: Any = None,
        sql: str | Callable[[str], str] = GOOD_SQL,
        insight: Any = None,
        escalation: Any = None,
    ) -> None:
        """Initialise the fake.

        Args:
            plan: Payload the planner receives. Defaults to a valid plan.
            sql: SQL text, or a function of the analyst's user prompt.
            insight: Payload the insight agent receives.
            escalation: Payload the verifier's escalation receives.
        """
        self.plan = plan if plan is not None else plan_payload()
        self.sql = sql
        self.insight = insight if insight is not None else INSIGHT_PAYLOAD
        self.escalation = escalation or {"plausible": True, "reason": "consistent"}
        self.sql_prompts: list[str] = []
        self.json_systems: list[str] = []

    async def complete(
        self, system: str, user: str, **kwargs: Any
    ) -> LLMResponse:
        """Serve the analyst's SQL request.

        Args:
            system: The system prompt.
            user: The user prompt, recorded for assertions.
            **kwargs: Ignored generation settings.

        Returns:
            A completion whose text is the scripted SQL.
        """
        self.sql_prompts.append(user)
        text = self.sql(user) if callable(self.sql) else self.sql
        return LLMResponse(
            text=text,
            provider="gemini",
            model="gemini-flash-latest",
            input_tokens=100,
            output_tokens=50,
            latency_ms=10.0,
            attempts=["gemini"],
        )

    async def complete_json_with_response(
        self, system: str, user: str, **kwargs: Any
    ) -> tuple[Any, LLMResponse]:
        """Serve the planner, verifier escalation or insight request.

        Args:
            system: The system prompt, used to decide which agent is calling.
            user: The user prompt.
            **kwargs: Ignored generation settings.

        Returns:
            The scripted payload and a synthetic completion.

        Raises:
            AssertionError: If the system prompt matches no known agent.
        """
        self.json_systems.append(system)
        if "planning agent" in system:
            payload = self.plan
        elif "checking layer" in system:
            payload = self.escalation
        elif "senior retail analyst" in system:
            payload = self.insight
        else:  # pragma: no cover - a new agent would need a branch here
            raise AssertionError("unrecognised system prompt")
        return payload, LLMResponse(
            text="{}",
            provider="gemini",
            model="gemini-flash-latest",
            input_tokens=200,
            output_tokens=100,
            latency_ms=20.0,
            attempts=["gemini"],
        )

    @property
    def sql_calls(self) -> int:
        """How many times SQL was requested.

        Returns:
            The number of analyst completions.
        """
        return len(self.sql_prompts)


def build(llm: ScriptedLLM) -> Orchestrator:
    """Wire an orchestrator whose agents all share one scripted client.

    Args:
        llm: The scripted client.

    Returns:
        The orchestrator.
    """
    return Orchestrator(
        planner=PlannerAgent(llm=llm),  # type: ignore[arg-type]
        analyst=SQLAnalystAgent(llm=llm),
        verifier=VerifierAgent(llm=llm),  # type: ignore[arg-type]
        insight=InsightAgent(llm=llm),  # type: ignore[arg-type]
    )


def step_names(response: Any) -> list[str]:
    """List the agent names in a trace, in order.

    Args:
        response: The analysis response.

    Returns:
        The step names.
    """
    return [step.agent_name for step in response.trace.steps]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_happy_path_answers_with_a_complete_trace() -> None:
    """A working run produces an answer and shows all four agents."""
    llm = ScriptedLLM()
    response = await build(llm).run("What was revenue in the last 3 months?")

    assert response.answered is True
    assert response.error is None
    assert response.plan is not None
    assert response.insight is not None
    assert response.verification is not None
    assert step_names(response) == ["planner", "sql_analyst", "verifier", "insight"]
    assert all(
        step.status is AgentStatus.SUCCEEDED for step in response.trace.steps
    )
    assert response.query_results[0].rows[0]["orders"] > 0


@pytest.mark.asyncio
async def test_the_response_carries_the_fixed_as_of_date() -> None:
    """Clients are told what "today" meant for this answer."""
    response = await build(ScriptedLLM()).run("What was revenue?")

    assert response.data_asof == settings.DATA_ASOF_DATE


@pytest.mark.asyncio
async def test_the_trace_aggregates_duration_and_tokens() -> None:
    """Totals are sums of the steps, not estimates."""
    response = await build(ScriptedLLM()).run("What was revenue?")
    trace = response.trace

    assert trace.total_tokens == sum(step.tokens or 0 for step in trace.steps)
    assert trace.total_tokens > 0
    assert trace.total_duration_ms >= max(step.duration_ms for step in trace.steps)
    assert trace.providers_used == ["gemini"]


# ---------------------------------------------------------------------------
# Unsupported questions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unsupported_intent_short_circuits_before_any_sql() -> None:
    """A question the data cannot answer must not become a query."""
    llm = ScriptedLLM(
        plan=plan_payload(
            intent="unsupported",
            confidence=0.1,
            reasoning="The dataset holds no competitor information.",
        )
    )
    response = await build(llm).run("How do we compare with McDonald's?")

    assert response.answered is False
    assert llm.sql_calls == 0
    assert response.query_results == []
    assert "competitors" in (response.error or "")


@pytest.mark.asyncio
async def test_unsupported_questions_still_explain_the_scope() -> None:
    """A refusal names what the system does hold, not just what it lacks."""
    llm = ScriptedLLM(plan=plan_payload(intent="unsupported", confidence=0.1))
    response = await build(llm).run("What is the weather forecast?")

    assert response.insight is not None
    assert settings.DATA_ASOF_DATE.isoformat() in response.insight.narrative
    skipped = [
        step for step in response.trace.steps if step.status is AgentStatus.SKIPPED
    ]
    assert {step.agent_name for step in skipped} == {
        "sql_analyst",
        "verifier",
        "insight",
    }


# ---------------------------------------------------------------------------
# Query failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_every_sub_query_failing_yields_a_useful_failure() -> None:
    """A total failure is still a well-formed response, not an exception."""
    llm = ScriptedLLM(sql=BROKEN_SQL)
    response = await build(llm).run("What was revenue?")

    assert response.answered is False
    assert "nonexistent_column" in (response.error or "")
    assert response.plan is not None
    assert response.query_results and response.query_results[0].error
    assert "verifier" in step_names(response)


@pytest.mark.asyncio
async def test_partial_failure_still_answers_and_shows_the_gap() -> None:
    """One broken sub-query must not discard the ones that worked."""
    plan = plan_payload(
        sub_queries=[
            {
                "id": "totals",
                "purpose": "Total revenue, orders and AOV for the window.",
                "tables": ["fact_orders"],
                "metrics": ["revenue", "orders", "aov"],
                "dimensions": [],
                "filters": {},
            },
            {
                "id": "by_channel",
                "purpose": "Revenue split by channel.",
                "tables": ["fact_orders"],
                "metrics": ["revenue"],
                "dimensions": ["channel"],
                "filters": {},
            },
        ]
    )
    llm = ScriptedLLM(
        plan=plan,
        sql=lambda prompt: BROKEN_SQL if "by channel" in prompt else GOOD_SQL,
    )
    response = await build(llm).run("What was revenue?")

    assert response.answered is True
    failed = [result for result in response.query_results if result.error]
    assert [result.sub_query_id for result in failed] == ["by_channel"]

    analyst_step = next(
        step for step in response.trace.steps if step.agent_name == "sql_analyst"
    )
    assert "by_channel" in analyst_step.summary
    assert analyst_step.status is AgentStatus.SUCCEEDED


# ---------------------------------------------------------------------------
# Self-healing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verification_failure_triggers_exactly_one_retry() -> None:
    """A failed check is fed back once, and the trace shows both attempts."""
    attempts: list[str] = []

    def sql_for(prompt: str) -> str:
        """Return bad SQL first and good SQL once verification complains.

        Args:
            prompt: The analyst's user prompt.

        Returns:
            The SQL to hand back.
        """
        attempts.append(prompt)
        return GOOD_SQL if "failed automated verification" in prompt else NEGATIVE_SQL

    llm = ScriptedLLM(sql=sql_for)
    response = await build(llm).run("What was revenue?")

    names = step_names(response)
    assert names == [
        "planner",
        "sql_analyst",
        "verifier",
        "sql_analyst_retry",
        "verifier",
        "insight",
    ]
    assert sum(1 for name in names if name.startswith("sql_analyst")) == 2
    assert response.verification is not None
    assert response.verification.passed is True
    assert response.answered is True
    assert response.query_results[0].rows[0]["revenue_inr"] > 0


@pytest.mark.asyncio
async def test_a_retry_that_does_not_help_still_answers_and_stays_marked() -> None:
    """Two failures do not become an infinite loop, and the user is warned."""
    llm = ScriptedLLM(sql=NEGATIVE_SQL)
    response = await build(llm).run("What was revenue?")

    assert step_names(response).count("sql_analyst_retry") == 1
    assert response.verification is not None
    assert response.verification.passed is False
    assert response.insight is not None
    assert "failed automated consistency checks" in response.insight.caveats[0]


# ---------------------------------------------------------------------------
# Timeouts
# ---------------------------------------------------------------------------


class HangingPlanner(PlannerAgent):
    """A planner that never returns."""

    async def execute(self, question: str) -> AnalysisPlan:
        """Sleep past any plausible deadline.

        Args:
            question: Ignored.

        Returns:
            Never returns.

        Raises:
            AgentError: Never; the sleep is interrupted by the timeout.
        """
        await asyncio.sleep(30)
        raise AgentError(self.name, "unreachable")


@pytest.mark.asyncio
async def test_a_hanging_agent_produces_a_failed_step_not_a_hang(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stuck agent must cost one timeout, not the whole request."""
    monkeypatch.setattr(settings, "PLANNER_TIMEOUT_SECONDS", 0.05)
    llm = ScriptedLLM()
    orchestrator = build(llm)
    orchestrator.planner = HangingPlanner(llm=llm)  # type: ignore[arg-type]

    response = await asyncio.wait_for(orchestrator.run("What was revenue?"), timeout=5)

    planner_step = response.trace.steps[0]
    assert planner_step.status is AgentStatus.FAILED
    assert "timed out" in (planner_step.error or "")
    assert response.answered is False
    assert llm.sql_calls == 0
    assert {step.status for step in response.trace.steps[1:]} == {AgentStatus.SKIPPED}


class HangingInsight(InsightAgent):
    """An insight agent that never returns."""

    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Sleep past any plausible deadline.

        Args:
            *args: Ignored.
            **kwargs: Ignored.

        Returns:
            Never returns.
        """
        await asyncio.sleep(30)


@pytest.mark.asyncio
async def test_an_insight_timeout_still_returns_the_numbers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Losing the narrative layer must not lose the verified figures."""
    monkeypatch.setattr(settings, "INSIGHT_TIMEOUT_SECONDS", 0.05)
    llm = ScriptedLLM()
    orchestrator = build(llm)
    orchestrator.insight = HangingInsight(llm=llm)  # type: ignore[arg-type]

    response = await asyncio.wait_for(orchestrator.run("What was revenue?"), timeout=5)

    insight_step = next(
        step for step in response.trace.steps if step.agent_name == "insight"
    )
    assert insight_step.status is AgentStatus.FAILED
    assert response.answered is True
    assert response.insight is not None
    assert response.insight.key_findings


# ---------------------------------------------------------------------------
# Never raises
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_unexpected_crash_becomes_a_response() -> None:
    """Nothing reaches the caller as an exception, including a bug."""

    class ExplodingVerifier(VerifierAgent):
        """A verifier that raises outside the agent's own error handling."""

        async def run(self, *args: Any, **kwargs: Any) -> Any:
            """Raise where the base class cannot catch it.

            Args:
                *args: Ignored.
                **kwargs: Ignored.

            Raises:
                ZeroDivisionError: Always.
            """
            raise ZeroDivisionError("boom")

    llm = ScriptedLLM()
    orchestrator = build(llm)
    orchestrator.verifier = ExplodingVerifier(llm=llm)  # type: ignore[arg-type]

    response = await orchestrator.run("What was revenue?")

    assert response.answered is False
    assert "boom" in (response.error or "")
    assert response.trace.steps


@pytest.mark.asyncio
async def test_a_planner_that_cannot_produce_a_plan_is_reported_clearly() -> None:
    """An unusable plan is a stated failure, not a stack trace."""
    llm = ScriptedLLM(plan={"not": "a plan"})
    response = await build(llm).run("What was revenue?")

    assert response.answered is False
    assert "analysis plan" in (response.error or "")
    assert llm.sql_calls == 0


# ---------------------------------------------------------------------------
# Canonical questions
# ---------------------------------------------------------------------------


def test_canonical_questions_are_the_eight_evaluation_questions() -> None:
    """The frontend's chips and the golden answers share one source."""
    assert len(CANONICAL_QUESTIONS) == 8
    assert [entry["id"] for entry in CANONICAL_QUESTIONS] == [
        f"q{index}" for index in range(1, 9)
    ]
    for entry in CANONICAL_QUESTIONS:
        assert entry["question"].endswith("?")
        assert entry["label"]
        assert len(entry["label"]) <= 40


def test_canonical_questions_never_name_a_relative_date() -> None:
    """Chips must resolve against the anchor, not the reader's calendar."""
    for entry in CANONICAL_QUESTIONS:
        assert "today" not in entry["question"].lower()
        assert "this year" not in entry["question"].lower()


@pytest.mark.asyncio
async def test_the_intent_enum_covers_the_short_circuit() -> None:
    """The unsupported branch keys off the contract, not a string."""
    llm = ScriptedLLM(plan=plan_payload(intent=QueryIntent.UNSUPPORTED.value))
    response = await build(llm).run("Anything?")

    assert response.plan is not None
    assert response.plan.intent is QueryIntent.UNSUPPORTED
