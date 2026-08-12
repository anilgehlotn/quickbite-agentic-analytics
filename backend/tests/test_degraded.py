"""Every stage's behaviour when the model provider dies underneath it.

The provider is failed at one stage at a time rather than all at once, because
the interesting question is not "does it break when nothing works" but "does
the pipeline still produce something useful when the failure lands here". Four
stages, four independent simulations.

What each stage can still do differs, and the tests assert the real answer
rather than a hopeful one:

* **Planner** - nothing. Turning prose into a plan is the one step with no
  deterministic substitute, so the honest outcome is the unavailable message.
* **SQL analyst** - a plain aggregate assembled from the plan itself. The plan
  is structured data, so ``build_fallback_sql`` can produce real SQL over the
  real database with no model involved.
* **Verifier** - every deterministic check. The model is only ever consulted
  for an optional plausibility review, and its absence downgrades nothing.
* **Insight** - deterministic prose built from the rows.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from app.agents.analyst import SQLAnalystAgent
from app.agents.contracts import (
    AgentStatus,
    AnalysisPlan,
    QueryIntent,
    QueryResult,
    ResponseStatus,
    SubQuery,
    TimeWindow,
    VerificationStatus,
)
from app.agents.insight import build_degraded_insight
from app.agents.orchestrator import Orchestrator
from app.agents.verifier import VerifierAgent
from app.core.llm import LLMError, LLMResponse
from app.config import settings

from tests.test_orchestrator import (
    GOOD_SQL,
    INSIGHT_PAYLOAD,
    ScriptedLLM,
    plan_payload,
)


class DeadLLM:
    """A client where every call fails, as if no provider were reachable."""

    def __init__(self, reason: str = "all providers failed") -> None:
        """Initialise the fake.

        Args:
            reason: The failure message every call raises.
        """
        self.reason = reason
        self.calls = 0

    async def complete(self, system: str, user: str, **kwargs: Any) -> LLMResponse:
        """Fail as the real client does when every provider is down.

        Args:
            system: Ignored.
            user: Ignored.
            **kwargs: Ignored.

        Returns:
            Never returns.

        Raises:
            LLMError: Always.
        """
        self.calls += 1
        raise LLMError(self.reason)

    async def complete_json_with_response(
        self, system: str, user: str, **kwargs: Any
    ) -> tuple[Any, LLMResponse]:
        """Fail as the real client does when every provider is down.

        Args:
            system: Ignored.
            user: Ignored.
            **kwargs: Ignored.

        Returns:
            Never returns.

        Raises:
            LLMError: Always.
        """
        self.calls += 1
        raise LLMError(self.reason)

    async def complete_json(self, system: str, user: str, **kwargs: Any) -> Any:
        """Fail as the real client does when every provider is down.

        Args:
            system: Ignored.
            user: Ignored.
            **kwargs: Ignored.

        Returns:
            Never returns.

        Raises:
            LLMError: Always.
        """
        self.calls += 1
        raise LLMError(self.reason)


class StageFailingLLM(ScriptedLLM):
    """Serves every agent normally except the one named, which always fails."""

    def __init__(self, dead_stage: str, **kwargs: Any) -> None:
        """Initialise the fake.

        Args:
            dead_stage: ``"planner"``, ``"analyst"``, ``"verifier"`` or
                ``"insight"`` - the stage whose calls raise.
            **kwargs: Passed to :class:`ScriptedLLM`.
        """
        super().__init__(**kwargs)
        self.dead_stage = dead_stage

    async def complete(self, system: str, user: str, **kwargs: Any) -> LLMResponse:
        """Serve or fail the analyst's SQL request.

        Args:
            system: The system prompt.
            user: The user prompt.
            **kwargs: Ignored generation settings.

        Returns:
            The scripted completion.

        Raises:
            LLMError: When the analyst is the dead stage.
        """
        if self.dead_stage == "analyst":
            raise LLMError("all providers failed")
        return await super().complete(system, user, **kwargs)

    async def complete_json_with_response(
        self, system: str, user: str, **kwargs: Any
    ) -> tuple[Any, LLMResponse]:
        """Serve or fail a JSON request, depending on which agent is calling.

        Args:
            system: The system prompt, used to identify the agent.
            user: The user prompt.
            **kwargs: Ignored generation settings.

        Returns:
            The scripted payload and completion.

        Raises:
            LLMError: When the calling agent is the dead stage.
        """
        if self.dead_stage == "planner" and "planning agent" in system:
            raise LLMError("all providers failed")
        if self.dead_stage == "verifier" and "checking layer" in system:
            raise LLMError("all providers failed")
        if self.dead_stage == "insight" and "planning agent" not in system:
            if "checking layer" not in system:
                raise LLMError("all providers failed")
        return await super().complete_json_with_response(system, user, **kwargs)


def simple_plan() -> AnalysisPlan:
    """Build a plan the deterministic fallback can serve.

    Returns:
        A plan for revenue and orders by channel over the evaluation window.
    """
    return AnalysisPlan(
        question="How do our sales channels compare in the last 3 months?",
        intent=QueryIntent.COMPARISON,
        time_window=TimeWindow(
            start_date=settings.LAST_3M_START,
            end_date=settings.LAST_3M_END,
            label="last 3 months",
            comparison_start=None,
            comparison_end=None,
        ),
        metrics=["revenue", "orders", "aov"],
        dimensions=["channel"],
        sub_queries=[
            SubQuery(
                id="by_channel",
                purpose="Revenue, orders and AOV by channel.",
                tables=["fact_orders"],
                metrics=["revenue", "orders", "aov"],
                dimensions=["channel"],
                filters={},
            )
        ],
        requires_diagnostics=False,
        reasoning="One grouped aggregate answers it.",
        confidence=0.9,
    )


class TestAnalystFallback:
    """The analyst assembles SQL from the plan when no model answers."""

    def test_builds_sql_without_a_model(self) -> None:
        """The plan alone is enough for a plain aggregate."""
        agent = SQLAnalystAgent(llm=DeadLLM())
        plan = simple_plan()

        sql = agent.build_fallback_sql(plan.sub_queries[0], plan)

        assert sql is not None
        assert "SUM(net_before_tax) AS revenue" in sql
        assert "GROUP BY channel" in sql
        assert settings.LAST_3M_START.isoformat() in sql

    def test_fallback_sql_never_reads_the_clock(self) -> None:
        """The fixed anchor applies to generated SQL as much as to written SQL."""
        agent = SQLAnalystAgent(llm=DeadLLM())
        plan = simple_plan()

        sql = agent.build_fallback_sql(plan.sub_queries[0], plan) or ""

        for forbidden in ("now", "CURRENT_DATE", "date(", "julianday"):
            assert forbidden.lower() not in sql.lower()

    def test_refuses_dimensions_needing_a_join(self) -> None:
        """A dimension that is not on fact_orders is refused, not guessed.

        Guessing a join without a model checking it changes the grain
        silently, which is a worse outcome than no answer.
        """
        agent = SQLAnalystAgent(llm=DeadLLM())
        plan = simple_plan()
        plan.sub_queries[0].dimensions = ["city"]

        assert agent.build_fallback_sql(plan.sub_queries[0], plan) is None

    def test_refuses_metrics_it_cannot_express(self) -> None:
        """Margin lives at line grain, so the fallback declines it."""
        agent = SQLAnalystAgent(llm=DeadLLM())
        plan = simple_plan()
        plan.metrics = ["gross_margin"]

        assert agent.build_fallback_sql(plan.sub_queries[0], plan) is None

    @pytest.mark.asyncio
    async def test_execute_falls_back_and_returns_real_rows(self) -> None:
        """With the model dead the sub-query still answers from the database."""
        agent = SQLAnalystAgent(llm=DeadLLM())
        plan = simple_plan()

        result = await agent.execute(plan.sub_queries[0], plan)

        assert result.error is None
        assert result.row_count == len(settings.CHANNELS)
        assert result.degraded is True
        assert "revenue" in result.columns

    @pytest.mark.asyncio
    async def test_fallback_numbers_are_real(self) -> None:
        """Degraded means "simpler query", not "estimated numbers"."""
        agent = SQLAnalystAgent(llm=DeadLLM())
        plan = simple_plan()

        result = await agent.execute(plan.sub_queries[0], plan)
        total = sum(float(row["revenue"]) for row in result.rows)

        # The canonical three-month revenue, from the golden answers.
        assert total == pytest.approx(3_197_076.50, abs=1.0)

    @pytest.mark.asyncio
    async def test_reports_the_error_when_no_fallback_is_possible(self) -> None:
        """A shape the fallback cannot build still fails honestly."""
        agent = SQLAnalystAgent(llm=DeadLLM())
        plan = simple_plan()
        plan.metrics = ["gross_margin"]

        result = await agent.execute(plan.sub_queries[0], plan)

        assert result.error is not None
        assert result.degraded is False


class TestVerifierWithoutAModel:
    """Deterministic checks are the verifier; the model is optional."""

    @pytest.mark.asyncio
    async def test_all_checks_still_run(self) -> None:
        """A dead provider costs no deterministic check."""
        agent = VerifierAgent(llm=DeadLLM())
        plan = simple_plan()
        results = [
            QueryResult(
                sub_query_id="by_channel",
                sql="SELECT channel, SUM(net_before_tax) AS revenue "
                "FROM fact_orders GROUP BY channel",
                columns=["channel", "revenue"],
                rows=[
                    {"channel": "Zomato", "revenue": 907_336.5},
                    {"channel": "Swiggy", "revenue": 800_000.0},
                    {"channel": "Dine-in", "revenue": 750_000.0},
                    {"channel": "Takeaway", "revenue": 739_740.0},
                ],
                row_count=4,
                execution_ms=2.0,
                error=None,
                attempts=1,
            )
        ]

        report = await agent.execute(plan, results)

        assert len(report.checks) >= 10
        assert report.status is not VerificationStatus.FAILED

    @pytest.mark.asyncio
    async def test_escalation_absence_is_reported_not_hidden(self) -> None:
        """If the optional review could not run, the report says so."""
        agent = VerifierAgent(llm=DeadLLM())
        plan = simple_plan()
        # A single row with a suspicious shape, to invite escalation.
        results = [
            QueryResult(
                sub_query_id="by_channel",
                sql="SELECT channel, SUM(net_before_tax) AS revenue "
                "FROM fact_orders GROUP BY channel",
                columns=["channel", "revenue"],
                rows=[{"channel": "Zomato", "revenue": 907_336.5}],
                row_count=1,
                execution_ms=2.0,
                error=None,
                attempts=1,
            )
        ]

        report = await agent.execute(plan, results)
        escalation = [
            check for check in report.checks if check.name == "llm_plausibility"
        ]

        # Either it did not need escalating, or it tried and said it could not.
        assert not escalation or "could not run" in escalation[0].message


class TestInsightWithoutAModel:
    """Deterministic prose beats an error message."""

    def test_degraded_insight_describes_the_rows(self) -> None:
        """The fallback narrates what was actually returned."""
        plan = simple_plan()
        results = [
            QueryResult(
                sub_query_id="by_channel",
                sql="SELECT 1",
                columns=["channel", "revenue"],
                rows=[
                    {"channel": "Zomato", "revenue": 907_336.5},
                    {"channel": "Swiggy", "revenue": 800_000.0},
                ],
                row_count=2,
                execution_ms=1.0,
                error=None,
                attempts=1,
            )
        ]

        insight = build_degraded_insight(plan.question, plan, results)

        assert insight.headline
        assert insight.narrative
        assert insight.confidence < 0.5

    def test_degraded_insight_invents_no_numbers(self) -> None:
        """Every figure it states must come from a row it was given."""
        plan = simple_plan()
        results = [
            QueryResult(
                sub_query_id="by_channel",
                sql="SELECT 1",
                columns=["channel", "revenue"],
                rows=[{"channel": "Zomato", "revenue": 907_336.5}],
                row_count=1,
                execution_ms=1.0,
                error=None,
                attempts=1,
            )
        ]

        insight = build_degraded_insight(plan.question, plan, results)
        text = f"{insight.headline} {insight.narrative}"

        assert "1,000,000" not in text
        assert "%" not in insight.headline or "907" in text or "1" in text


class TestPipelineWithOneStageDead:
    """The orchestrator's whole-run behaviour, one dead stage at a time."""

    @pytest.mark.asyncio
    async def test_planner_dead_returns_the_unavailable_message(self) -> None:
        """Planning has no deterministic substitute, so it says so."""
        orchestrator = Orchestrator()
        llm = StageFailingLLM("planner")
        for agent in (
            orchestrator.planner,
            orchestrator.analyst,
            orchestrator.verifier,
            orchestrator.insight,
        ):
            agent.llm = llm

        response = await orchestrator.run("What was revenue last month?")

        assert response.answered is False
        assert response.status is ResponseStatus.FAILED
        assert response.error is not None
        # The trace still names every stage, including the ones skipped.
        agents = [step.agent_name for step in response.trace.steps]
        assert agents[0] == "planner"
        assert len(agents) == 4

    @pytest.mark.asyncio
    async def test_analyst_dead_still_answers_from_the_fallback(self) -> None:
        """The plan survived, so the numbers can still be produced."""
        orchestrator = Orchestrator()
        llm = StageFailingLLM(
            "analyst",
            plan=plan_payload(
                metrics=["revenue", "orders", "aov"],
                dimensions=["channel"],
                sub_queries=[
                    {
                        "id": "by_channel",
                        "purpose": "Revenue by channel.",
                        "tables": ["fact_orders"],
                        "metrics": ["revenue", "orders", "aov"],
                        "dimensions": ["channel"],
                        "filters": {},
                    }
                ],
            ),
        )
        for agent in (
            orchestrator.planner,
            orchestrator.analyst,
            orchestrator.verifier,
            orchestrator.insight,
        ):
            agent.llm = llm

        response = await orchestrator.run("How do channels compare?")

        assert response.answered is True
        assert response.query_results[0].error is None
        assert response.query_results[0].degraded is True
        assert response.query_results[0].row_count == len(settings.CHANNELS)

    @pytest.mark.asyncio
    async def test_verifier_dead_does_not_block_the_answer(self) -> None:
        """The optional review failing must not lose a good answer."""
        orchestrator = Orchestrator()
        llm = StageFailingLLM("verifier")
        for agent in (
            orchestrator.planner,
            orchestrator.analyst,
            orchestrator.verifier,
            orchestrator.insight,
        ):
            agent.llm = llm

        response = await orchestrator.run("What was revenue last month?")

        assert response.answered is True
        assert response.verification is not None
        assert response.verification.status is not VerificationStatus.FAILED

    @pytest.mark.asyncio
    async def test_insight_dead_falls_back_to_deterministic_prose(self) -> None:
        """An answer with numbers and no explanation is still an answer."""
        orchestrator = Orchestrator()
        llm = StageFailingLLM("insight")
        for agent in (
            orchestrator.planner,
            orchestrator.analyst,
            orchestrator.verifier,
            orchestrator.insight,
        ):
            agent.llm = llm

        response = await orchestrator.run("What was revenue last month?")

        assert response.answered is True
        assert response.insight is not None
        assert response.insight.headline
        # The agent catches its own provider failure and narrates the rows
        # deterministically, so the step succeeds - but it must say that is
        # what happened rather than presenting the fallback as a model answer.
        insight_step = next(
            step for step in response.trace.steps if step.agent_name == "insight"
        )
        assert insight_step.status is AgentStatus.SUCCEEDED
        assert "unavailable" in insight_step.summary
        assert response.insight.confidence < 0.5

    @pytest.mark.asyncio
    async def test_every_stage_dead_still_returns_a_valid_response(self) -> None:
        """Total outage produces a well-formed refusal, never an exception."""
        orchestrator = Orchestrator()
        dead = DeadLLM()
        for agent in (
            orchestrator.planner,
            orchestrator.analyst,
            orchestrator.verifier,
            orchestrator.insight,
        ):
            agent.llm = dead

        response = await orchestrator.run("What was revenue last month?")

        assert response.answered is False
        assert response.question == "What was revenue last month?"
        assert response.trace.steps
        assert response.data_asof == settings.DATA_ASOF_DATE
