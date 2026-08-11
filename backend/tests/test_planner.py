"""Tests for the planner agent.

The LLM is replaced by a scripted fake, so these tests assert on the planner's
own logic — schema validation, the retry-once-then-fail policy, prompt assembly
and trace production — rather than on model quality, which no unit test can
pin down. No network call is made.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from app.agents.base import AgentError
from app.agents.contracts import AgentStatus, QueryIntent
from app.agents.planner import MAX_PLANNING_ATTEMPTS, PlannerAgent
from app.config import settings
from app.core.llm import LLMResponse


class FakeLLM:
    """A scripted stand-in for :class:`LLMClient`.

    Returns queued payloads in order and records the prompts it was given, so
    a test can assert on what the planner actually asked for.
    """

    def __init__(self, *payloads: Any) -> None:
        """Initialise the fake.

        Args:
            *payloads: Values to return from successive calls. The last one
                repeats once exhausted.
        """
        self.payloads = list(payloads)
        self.calls: list[dict[str, Any]] = []

    async def complete_json_with_response(
        self, system: str, user: str, **kwargs: Any
    ) -> tuple[Any, LLMResponse]:
        """Return the next scripted payload.

        Args:
            system: The system prompt, recorded for assertions.
            user: The user prompt, recorded for assertions.
            **kwargs: Ignored generation settings.

        Returns:
            The payload and a synthetic completion carrying token counts.
        """
        self.calls.append({"system": system, "user": user, **kwargs})
        index = min(len(self.calls) - 1, len(self.payloads) - 1)
        return self.payloads[index], LLMResponse(
            text="{}",
            provider="anthropic",
            model="claude-opus-5",
            input_tokens=100,
            output_tokens=50,
            latency_ms=12.0,
            attempts=["anthropic"],
        )

    @property
    def call_count(self) -> int:
        """How many completions were requested.

        Returns:
            The number of calls made.
        """
        return len(self.calls)


def plan_payload(**overrides: Any) -> dict[str, Any]:
    """Build a valid plan payload for a fake to return.

    Args:
        **overrides: Fields to replace.

    Returns:
        A dict that validates as an AnalysisPlan.
    """
    payload: dict[str, Any] = {
        "question": "placeholder",
        "intent": "aggregate",
        "time_window": {
            "start_date": "2026-05-01",
            "end_date": "2026-07-31",
            "label": "last 3 months",
        },
        "metrics": ["revenue", "orders", "aov"],
        "dimensions": [],
        "sub_queries": [
            {"id": "headline", "purpose": "Total revenue, orders and AOV."}
        ],
        "requires_diagnostics": False,
        "reasoning": "One aggregate answers the question.",
        "confidence": 0.95,
    }
    payload.update(overrides)
    return payload


class TestSimplePlanning:
    """A single-part question produces a single-query plan."""

    @pytest.mark.asyncio
    async def test_one_sub_query_for_a_simple_question(self) -> None:
        """Revenue, orders and AOV together are one query, not three."""
        llm = FakeLLM(plan_payload())
        agent = PlannerAgent(llm=llm)

        plan = await agent.execute(
            "What were total revenue, orders and AOV for the last 3 months?"
        )

        assert len(plan.sub_queries) == 1
        assert plan.intent is QueryIntent.AGGREGATE
        assert set(plan.metrics) == {"revenue", "orders", "aov"}

    @pytest.mark.asyncio
    async def test_question_is_taken_from_the_caller_not_the_model(self) -> None:
        """A model that paraphrases cannot change what the answer is about."""
        llm = FakeLLM(plan_payload(question="something the model made up"))
        agent = PlannerAgent(llm=llm)

        plan = await agent.execute("What was revenue last month?")

        assert plan.question == "What was revenue last month?"


class TestDecomposition:
    """Multi-part questions become multiple sub-queries."""

    @pytest.mark.asyncio
    async def test_top_and_bottom_is_two_sub_queries(self) -> None:
        """Different orderings cannot share one query."""
        llm = FakeLLM(
            plan_payload(
                intent="ranking",
                dimensions=["store_id"],
                sub_queries=[
                    {"id": "top_5", "purpose": "Five highest-revenue stores."},
                    {"id": "bottom_5", "purpose": "Five lowest-revenue stores."},
                ],
            )
        )
        agent = PlannerAgent(llm=llm)

        plan = await agent.execute("Which 5 stores are best and which 5 are worst?")

        assert len(plan.sub_queries) == 2
        assert {sub.id for sub in plan.sub_queries} == {"top_5", "bottom_5"}


class TestDiagnosticPlanning:
    """A 'why' question must decompose the change, not just measure it."""

    @pytest.mark.asyncio
    async def test_why_question_sets_diagnostics_and_adds_breakdowns(self) -> None:
        """Channel, monthly and prior-period sub-queries are all present."""
        llm = FakeLLM(
            plan_payload(
                intent="diagnostic",
                requires_diagnostics=True,
                dimensions=["store_id", "month_key"],
                time_window={
                    "start_date": "2026-05-01",
                    "end_date": "2026-07-31",
                    "label": "last 3 months",
                    "comparison_start": "2026-02-01",
                    "comparison_end": "2026-04-30",
                },
                sub_queries=[
                    {
                        "id": "monthly_revenue",
                        "purpose": "Revenue per store per month.",
                        "dimensions": ["store_id", "month_key"],
                    },
                    {
                        "id": "channel_breakdown",
                        "purpose": "Revenue by channel for declining stores.",
                        "dimensions": ["store_id", "channel"],
                    },
                    {
                        "id": "orders_vs_aov",
                        "purpose": "Order count and AOV per month.",
                        "metrics": ["orders", "aov"],
                    },
                    {
                        "id": "prior_period",
                        "purpose": "Revenue in the prior three months.",
                    },
                ],
                reasoning="Explaining a decline requires decomposing it.",
            )
        )
        agent = PlannerAgent(llm=llm)

        plan = await agent.execute("Why did store ST039 decline?")

        assert plan.requires_diagnostics is True
        assert plan.intent is QueryIntent.DIAGNOSTIC
        ids = {sub.id for sub in plan.sub_queries}
        assert "channel_breakdown" in ids
        assert "monthly_revenue" in ids
        assert plan.time_window.comparison_start == date(2026, 2, 1)
        assert plan.time_window.comparison_end == date(2026, 4, 30)

    def test_prompt_demands_decomposition_for_why_questions(self) -> None:
        """The rule is actually in the prompt, not just in the docstring."""
        prompt = PlannerAgent(llm=FakeLLM()).build_system_prompt()

        assert "requires_diagnostics=true" in prompt
        assert "by channel" in prompt
        assert "order count versus AOV" in prompt

    def test_prompt_forbids_endpoint_only_trends(self) -> None:
        """Monthly values, never just first and last."""
        prompt = PlannerAgent(llm=FakeLLM()).build_system_prompt()

        assert "individual monthly values" in prompt
        assert "only the first and last" in prompt


class TestUnsupportedQuestions:
    """Questions the data cannot answer must be refused, not invented."""

    @pytest.mark.asyncio
    async def test_out_of_scope_question_is_unsupported(self) -> None:
        """A competitor question yields UNSUPPORTED with low confidence."""
        llm = FakeLLM(
            plan_payload(
                intent="unsupported",
                metrics=[],
                sub_queries=[
                    {
                        "id": "none",
                        "purpose": "No query; the data has no competitor information.",
                    }
                ],
                reasoning="This dataset contains only QuickBite's own orders.",
                confidence=0.05,
            )
        )
        agent = PlannerAgent(llm=llm)

        plan = await agent.execute("How do we compare to McDonald's?")

        assert plan.intent is QueryIntent.UNSUPPORTED
        assert plan.confidence < 0.3

    def test_prompt_lists_out_of_scope_topics(self) -> None:
        """The prompt names what the data cannot answer."""
        prompt = PlannerAgent(llm=FakeLLM()).build_system_prompt()

        for topic in ("competitors", "weather", "staff"):
            assert topic in prompt


class TestTimeResolution:
    """Relative expressions resolve against the fixed anchor."""

    @pytest.mark.asyncio
    async def test_last_three_months_resolves_to_the_configured_window(self) -> None:
        """The window matches the configured last-3-months range."""
        llm = FakeLLM(plan_payload())
        agent = PlannerAgent(llm=llm)

        plan = await agent.execute("Revenue for the last 3 months?")

        assert plan.time_window.start_date == settings.LAST_3M_START
        assert plan.time_window.end_date == settings.LAST_3M_END
        assert plan.time_window.start_date == date(2026, 5, 1)
        assert plan.time_window.end_date == date(2026, 7, 31)

    def test_prompt_states_the_anchor_and_forbids_the_clock(self) -> None:
        """The anchor is in the prompt and the real calendar is ruled out."""
        prompt = PlannerAgent(llm=FakeLLM()).build_system_prompt()

        assert settings.DATA_ASOF_DATE.isoformat() in prompt
        assert "never against the real calendar" in prompt

    def test_prompt_lists_only_valid_metrics(self) -> None:
        """The metric allowlist reaches the model."""
        from app.semantic.schema import METRIC_DEFINITIONS

        prompt = PlannerAgent(llm=FakeLLM()).build_system_prompt()

        for metric in METRIC_DEFINITIONS:
            assert metric in prompt


class TestValidationRetry:
    """An invalid plan gets exactly one corrective retry."""

    @pytest.mark.asyncio
    async def test_invalid_metric_triggers_one_retry_then_succeeds(self) -> None:
        """The second attempt, with the error fed back, is accepted."""
        llm = FakeLLM(
            plan_payload(metrics=["total_sales"]),  # not a real metric
            plan_payload(metrics=["revenue"]),
        )
        agent = PlannerAgent(llm=llm)

        plan = await agent.execute("What was revenue?")

        assert llm.call_count == 2
        assert plan.metrics == ["revenue"]

    @pytest.mark.asyncio
    async def test_retry_prompt_contains_the_validation_error(self) -> None:
        """The model is told exactly what was wrong."""
        llm = FakeLLM(
            plan_payload(metrics=["total_sales"]), plan_payload(metrics=["revenue"])
        )
        agent = PlannerAgent(llm=llm)

        await agent.execute("What was revenue?")

        retry_prompt = llm.calls[1]["user"]
        assert "rejected by schema validation" in retry_prompt
        assert "total_sales" in retry_prompt

    @pytest.mark.asyncio
    async def test_two_invalid_plans_raise(self) -> None:
        """After the retry fails, the planner gives up rather than looping."""
        llm = FakeLLM(plan_payload(metrics=["nonsense_metric"]))
        agent = PlannerAgent(llm=llm)

        with pytest.raises(AgentError, match="could not produce a valid plan"):
            await agent.execute("What was revenue?")

        assert llm.call_count == MAX_PLANNING_ATTEMPTS

    @pytest.mark.asyncio
    async def test_missing_sub_queries_is_a_validation_failure(self) -> None:
        """A plan with no queries cannot answer anything."""
        llm = FakeLLM(plan_payload(sub_queries=[]))
        agent = PlannerAgent(llm=llm)

        with pytest.raises(AgentError):
            await agent.execute("What was revenue?")

    @pytest.mark.asyncio
    async def test_non_dict_payload_is_a_validation_failure(self) -> None:
        """A model returning a list instead of an object fails cleanly."""
        llm = FakeLLM(["not", "a", "plan"])
        agent = PlannerAgent(llm=llm)

        with pytest.raises(AgentError):
            await agent.execute("What was revenue?")


class TestInstrumentation:
    """The base class produces a trace step for every run."""

    @pytest.mark.asyncio
    async def test_successful_run_produces_a_succeeded_step(self) -> None:
        """Timing, provider and tokens all reach the trace."""
        agent = PlannerAgent(llm=FakeLLM(plan_payload()))

        outcome = await agent.run("What was revenue?")

        assert outcome.succeeded is True
        assert outcome.step.agent_name == "planner"
        assert outcome.step.status is AgentStatus.SUCCEEDED
        assert outcome.step.llm_provider == "anthropic"
        assert outcome.step.tokens == 150
        assert outcome.step.duration_ms >= 0
        assert outcome.result is not None

    @pytest.mark.asyncio
    async def test_failed_run_produces_a_failed_step_not_an_exception(self) -> None:
        """run() never raises; the orchestrator decides how to degrade."""
        agent = PlannerAgent(llm=FakeLLM(plan_payload(metrics=["bogus"])))

        outcome = await agent.run("What was revenue?")

        assert outcome.succeeded is False
        assert outcome.result is None
        assert outcome.step.status is AgentStatus.FAILED
        assert outcome.step.error is not None

    @pytest.mark.asyncio
    async def test_retry_tokens_accumulate(self) -> None:
        """Two attempts report the sum, not the last call."""
        agent = PlannerAgent(
            llm=FakeLLM(plan_payload(metrics=["bad"]), plan_payload())
        )

        outcome = await agent.run("What was revenue?")

        assert outcome.step.tokens == 300

    @pytest.mark.asyncio
    async def test_summary_describes_the_plan(self) -> None:
        """The trace summary is written for a user watching the run."""
        agent = PlannerAgent(
            llm=FakeLLM(
                plan_payload(
                    intent="diagnostic",
                    requires_diagnostics=True,
                    sub_queries=[
                        {"id": "a", "purpose": "First."},
                        {"id": "b", "purpose": "Second."},
                    ],
                )
            )
        )

        outcome = await agent.run("Why did revenue fall?")

        assert "diagnostic" in outcome.step.summary
        assert "2 queries" in outcome.step.summary

    @pytest.mark.asyncio
    async def test_unsupported_summary_says_so_plainly(self) -> None:
        """An unanswerable question is reported as such in the trace."""
        agent = PlannerAgent(llm=FakeLLM(plan_payload(intent="unsupported")))

        outcome = await agent.run("What is the weather?")

        assert "cannot be answered" in outcome.step.summary
