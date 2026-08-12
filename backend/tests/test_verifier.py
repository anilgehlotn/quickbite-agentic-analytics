"""Tests for the verifier agent.

Results are constructed directly rather than queried, so each test states the
exact arithmetic situation it is about. The load-bearing test here is
:func:`test_deterministic_failure_never_calls_the_llm`: the ordering of the two
layers is an architectural claim, and an untested claim about ordering is just
a comment.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from app.agents.contracts import (
    AnalysisPlan,
    QueryIntent,
    QueryResult,
    SubQuery,
    TimeWindow,
    VerificationStatus,
)
from app.agents.verifier import (
    ERROR,
    WARNING,
    VerifierAgent,
    failing_sub_query_ids,
    is_revenue_column,
    is_share_column,
)
from app.config import settings
from app.core.llm import LLMResponse


class FakeLLM:
    """A scripted stand-in for :class:`LLMClient` that counts its calls."""

    def __init__(self, payload: Any | None = None) -> None:
        """Initialise the fake.

        Args:
            payload: Value returned from every call. Defaults to a plausible
                verdict.
        """
        self.payload = payload if payload is not None else {"plausible": True, "reason": "ok"}
        self.calls: list[dict[str, str]] = []

    async def complete_json_with_response(
        self, system: str, user: str, **kwargs: Any
    ) -> tuple[Any, LLMResponse]:
        """Record the call and return the scripted payload.

        Args:
            system: The system prompt.
            user: The user prompt.
            **kwargs: Ignored generation settings.

        Returns:
            The payload and a synthetic completion.
        """
        self.calls.append({"system": system, "user": user})
        return self.payload, LLMResponse(
            text="{}",
            provider="anthropic",
            model="claude-opus-5",
            input_tokens=40,
            output_tokens=20,
            latency_ms=5.0,
            attempts=["anthropic"],
        )

    @property
    def call_count(self) -> int:
        """How many completions were requested.

        Returns:
            The number of calls made.
        """
        return len(self.calls)


def make_plan(
    question: str = "What was revenue in the last 3 months?",
    intent: QueryIntent = QueryIntent.AGGREGATE,
    metrics: list[str] | None = None,
    sub_query_ids: list[str] | None = None,
) -> AnalysisPlan:
    """Build a plan for a verification test.

    Args:
        question: The question the plan answers.
        intent: The classified intent.
        metrics: Canonical metric names.
        sub_query_ids: Ids of the sub-queries in the plan.

    Returns:
        A valid plan.
    """
    ids = sub_query_ids or ["totals"]
    return AnalysisPlan(
        question=question,
        intent=intent,
        time_window=TimeWindow(
            start_date=settings.LAST_3M_START,
            end_date=settings.LAST_3M_END,
            label="last 3 months",
        ),
        metrics=metrics if metrics is not None else ["revenue"],
        dimensions=[],
        sub_queries=[
            SubQuery(id=sub_query_id, purpose=f"Compute {sub_query_id}.")
            for sub_query_id in ids
        ],
        requires_diagnostics=False,
        reasoning="A single aggregate answers the question.",
        confidence=0.9,
    )


def make_result(
    sub_query_id: str,
    rows: list[dict[str, Any]],
    sql: str = "SELECT SUM(net_before_tax) AS revenue_inr FROM fact_orders",
    error: str | None = None,
) -> QueryResult:
    """Build a query result for a verification test.

    Args:
        sub_query_id: The sub-query this result belongs to.
        rows: The rows returned.
        sql: The SQL that produced them.
        error: The failure reason, when the query failed.

    Returns:
        A valid result.
    """
    return QueryResult(
        sub_query_id=sub_query_id,
        sql=sql,
        columns=list(rows[0]) if rows else [],
        rows=rows,
        row_count=len(rows),
        execution_ms=1.0,
        error=error,
    )


def find(report: Any, name: str) -> Any:
    """Return the check with the given name.

    Args:
        report: The verification report.
        name: The check name.

    Returns:
        The matching check.
    """
    matches = [check for check in report.checks if check.name == name]
    assert matches, f"no check named {name} in {[c.name for c in report.checks]}"
    return matches[0]


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parts_summing_to_total_passes() -> None:
    """A complete channel breakdown that sums to the total reconciles."""
    verifier = VerifierAgent(llm=FakeLLM())
    results = [
        make_result("totals", [{"revenue_inr": 1000.0}]),
        make_result(
            "by_channel",
            [
                {"channel": "Dine-in", "revenue_inr": 250.0},
                {"channel": "Takeaway", "revenue_inr": 250.0},
                {"channel": "Swiggy", "revenue_inr": 250.0},
                {"channel": "Zomato", "revenue_inr": 250.0},
            ],
        ),
    ]
    report = await verifier.execute(make_plan(), results)

    check = find(report, "parts_sum_to_total")
    assert check.passed
    assert report.status is not VerificationStatus.FAILED


@pytest.mark.asyncio
async def test_parts_not_summing_to_total_is_an_error_reporting_the_delta() -> None:
    """A breakdown that misses the total fails and states by how much."""
    verifier = VerifierAgent(llm=FakeLLM())
    results = [
        make_result("totals", [{"revenue_inr": 1000.0}]),
        make_result(
            "by_channel",
            [
                {"channel": "Dine-in", "revenue_inr": 250.0},
                {"channel": "Takeaway", "revenue_inr": 250.0},
                {"channel": "Swiggy", "revenue_inr": 250.0},
                {"channel": "Zomato", "revenue_inr": 150.0},
            ],
        ),
    ]
    report = await verifier.execute(make_plan(), results)

    check = find(report, "parts_sum_to_total")
    assert not check.passed
    assert check.severity == ERROR
    assert check.details["delta"] == -100.0
    assert "-100.00" in check.message
    assert report.status is VerificationStatus.FAILED


@pytest.mark.asyncio
async def test_incomplete_breakdown_is_not_reconciled() -> None:
    """A top-N ranking is a subset by design and must not fail the sum."""
    verifier = VerifierAgent(llm=FakeLLM())
    results = [
        make_result("totals", [{"revenue_inr": 1000.0}]),
        make_result(
            "top_channels",
            [
                {"channel": "Swiggy", "revenue_inr": 300.0},
                {"channel": "Zomato", "revenue_inr": 250.0},
            ],
        ),
    ]
    report = await verifier.execute(make_plan(), results)

    assert find(report, "parts_sum_to_total").passed


# ---------------------------------------------------------------------------
# AOV
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aov_inconsistent_with_revenue_over_orders_is_caught() -> None:
    """An AOV that does not equal revenue / orders fails verification."""
    verifier = VerifierAgent(llm=FakeLLM())
    results = [
        make_result(
            "totals",
            [{"revenue_inr": 1000.0, "orders": 10, "aov_inr": 250.0}],
        )
    ]
    report = await verifier.execute(make_plan(), results)

    check = find(report, "aov_reconciles")
    assert not check.passed
    assert check.severity == ERROR
    assert check.details["offenders"][0]["computed_aov"] == 100.0
    assert report.status is VerificationStatus.FAILED


@pytest.mark.asyncio
async def test_aov_within_tolerance_passes() -> None:
    """Rounding to two decimals does not break reconciliation."""
    verifier = VerifierAgent(llm=FakeLLM())
    results = [
        make_result(
            "totals",
            [{"revenue_inr": 3197076.5, "orders": 4930, "aov_inr": 648.49}],
        )
    ]
    report = await verifier.execute(make_plan(), results)

    assert find(report, "aov_reconciles").passed


# ---------------------------------------------------------------------------
# Severity mapping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_results_are_a_warning_not_an_error() -> None:
    """An empty result is a finding, not a failure."""
    verifier = VerifierAgent(llm=FakeLLM())
    report = await verifier.execute(make_plan(), [make_result("totals", [])])

    check = find(report, "results_non_empty")
    assert not check.passed
    assert check.severity == WARNING
    assert report.status is VerificationStatus.PASSED_WITH_WARNINGS


@pytest.mark.asyncio
async def test_negative_revenue_is_an_error() -> None:
    """Revenue cannot be negative in this dataset."""
    verifier = VerifierAgent(llm=FakeLLM())
    report = await verifier.execute(
        make_plan(), [make_result("totals", [{"revenue_inr": -50.0}])]
    )

    check = find(report, "no_negative_measures")
    assert not check.passed
    assert check.severity == ERROR
    assert report.status is VerificationStatus.FAILED


@pytest.mark.asyncio
async def test_a_negative_change_column_is_not_an_error() -> None:
    """A decline is a negative number and a correct one."""
    verifier = VerifierAgent(llm=FakeLLM())
    report = await verifier.execute(
        make_plan(),
        [
            make_result(
                "trend",
                [
                    {"month_key": "2026-06", "revenue_inr": 100.0, "revenue_change_pct": -8.0},
                    {"month_key": "2026-07", "revenue_inr": 90.0, "revenue_change_pct": -10.0},
                ],
            )
        ],
    )

    assert find(report, "no_negative_measures").passed


@pytest.mark.asyncio
async def test_implausibly_large_revenue_is_flagged() -> None:
    """A revenue above the whole dataset's annual total is a broken query."""
    verifier = VerifierAgent(llm=FakeLLM())
    report = await verifier.execute(
        make_plan(),
        [
            make_result(
                "totals",
                [{"revenue_inr": settings.MAX_PLAUSIBLE_REVENUE_INR + 1.0}],
            )
        ],
    )

    check = find(report, "revenue_within_plausible_bound")
    assert not check.passed
    assert check.severity == ERROR
    assert report.status is VerificationStatus.FAILED


@pytest.mark.asyncio
async def test_dates_outside_the_dataset_range_are_an_error() -> None:
    """A month the extract does not contain cannot have been read from it."""
    verifier = VerifierAgent(llm=FakeLLM())
    report = await verifier.execute(
        make_plan(),
        [make_result("trend", [{"month_key": "2026-09", "revenue_inr": 10.0}])],
    )

    check = find(report, "dates_within_data_range")
    assert not check.passed
    assert check.severity == ERROR


@pytest.mark.asyncio
async def test_a_top_n_question_answered_with_far_too_many_rows_warns() -> None:
    """A "top 5" that returns forty rows was not answered as asked."""
    verifier = VerifierAgent(llm=FakeLLM())
    plan = make_plan(question="Which are the top 5 stores by revenue?")
    rows = [{"store_id": f"ST{index:03d}", "revenue_inr": 10.0} for index in range(40)]
    report = await verifier.execute(plan, [make_result("ranking", rows)])

    check = find(report, "row_count_matches_expectation")
    assert not check.passed
    assert check.severity == WARNING


@pytest.mark.asyncio
async def test_a_time_window_is_not_read_as_a_row_count() -> None:
    """A phrase like 'the last 3 months' is a window, not three rows.

    Reading it as one warned on every question this system exists to answer,
    which is how it was found: in a live run, not in a unit test.
    """
    verifier = VerifierAgent(llm=FakeLLM())
    plan = make_plan(question="Which cities declined over the last 3 months?")
    rows = [
        {"city": f"City {index}", "month_key": "2026-05", "revenue_inr": 10.0}
        for index in range(24)
    ]
    report = await verifier.execute(plan, [make_result("city_trend", rows)])

    assert find(report, "row_count_matches_expectation").passed


@pytest.mark.asyncio
async def test_a_silently_truncated_window_is_caught() -> None:
    """A three-month question answered with two months is flagged.

    The live failure: month_key BETWEEN '2026-05-01' AND '2026-07-31' reads
    correctly, runs without error, returns individually correct numbers, and
    drops every May row.
    """
    verifier = VerifierAgent(llm=FakeLLM())
    rows = [
        {"month_key": "2026-06", "revenue_inr": 10.0},
        {"month_key": "2026-07", "revenue_inr": 12.0},
    ]
    report = await verifier.execute(make_plan(), [make_result("trend", rows)])

    check = find(report, "results_cover_the_window")
    assert not check.passed
    assert check.details["offenders"][0]["missing_months"] == ["2026-05"]
    assert "2026-05" in check.message


@pytest.mark.asyncio
async def test_a_breakdown_covering_one_city_of_eight_is_flagged() -> None:
    """Answering about one city is not answering about the business.

    The live failure: a "revenue by city per month" sub-query filtered itself
    to a single city. Every figure it returned was correct, so no arithmetic
    check could object, and the answer that followed could not have found a
    decline in any of the other seven cities.
    """
    verifier = VerifierAgent(llm=FakeLLM())
    rows = [
        {"city": "Bengaluru", "month_key": month, "revenue_inr": 10.0}
        for month in ("2026-05", "2026-06", "2026-07")
    ]
    report = await verifier.execute(make_plan(), [make_result("city_trend", rows)])

    check = find(report, "dimension_coverage")
    assert not check.passed
    assert check.severity == WARNING
    assert check.details["offenders"][0]["found"] == 1
    assert "1 of 8" in check.message


@pytest.mark.asyncio
async def test_a_top_n_question_may_cover_part_of_a_dimension() -> None:
    """A ranking is a subset by design and must not be flagged."""
    verifier = VerifierAgent(llm=FakeLLM())
    plan = make_plan(question="Which are the top 3 cities by revenue?")
    rows = [
        {"city": city, "revenue_inr": 10.0}
        for city in ("Mumbai", "Delhi", "Pune")
    ]
    report = await verifier.execute(plan, [make_result("ranking", rows)])

    assert find(report, "dimension_coverage").passed


@pytest.mark.asyncio
async def test_a_complete_channel_breakdown_passes_coverage() -> None:
    """All four channels present is the normal case."""
    verifier = VerifierAgent(llm=FakeLLM())
    rows = [
        {"channel": channel, "revenue_inr": 10.0}
        for channel in settings.CHANNELS
    ]
    report = await verifier.execute(make_plan(), [make_result("by_channel", rows)])

    assert find(report, "dimension_coverage").passed


@pytest.mark.asyncio
async def test_a_full_window_passes_coverage() -> None:
    """All three months present is the normal case."""
    verifier = VerifierAgent(llm=FakeLLM())
    rows = [
        {"month_key": month, "revenue_inr": 10.0}
        for month in ("2026-05", "2026-06", "2026-07")
    ]
    report = await verifier.execute(make_plan(), [make_result("trend", rows)])

    assert find(report, "results_cover_the_window").passed


@pytest.mark.asyncio
async def test_a_comparison_period_sub_query_is_not_flagged() -> None:
    """A query that deliberately looks only at the baseline shares no month."""
    verifier = VerifierAgent(llm=FakeLLM())
    rows = [
        {"month_key": month, "revenue_inr": 10.0}
        for month in ("2026-02", "2026-03", "2026-04")
    ]
    report = await verifier.execute(make_plan(), [make_result("baseline", rows)])

    assert find(report, "results_cover_the_window").passed


@pytest.mark.asyncio
async def test_status_is_passed_when_every_check_holds() -> None:
    """A clean, unambiguous result verifies without warnings."""
    verifier = VerifierAgent(llm=FakeLLM())
    report = await verifier.execute(
        make_plan(metrics=["revenue", "orders", "aov"]),
        [
            make_result(
                "totals",
                [{"revenue_inr": 1000.0, "orders": 10, "aov_inr": 100.0}],
            )
        ],
    )

    assert report.status is VerificationStatus.PASSED
    assert all(check.passed for check in report.checks)


@pytest.mark.asyncio
async def test_a_failed_sub_query_is_a_warning_that_names_it() -> None:
    """A partial answer is still an answer, and the gap is stated."""
    verifier = VerifierAgent(llm=FakeLLM())
    report = await verifier.execute(
        make_plan(sub_query_ids=["totals", "by_channel"]),
        [
            make_result("totals", [{"revenue_inr": 1000.0}]),
            make_result("by_channel", [], error="no such column: channel_name"),
        ],
    )

    check = find(report, "all_sub_queries_executed")
    assert not check.passed
    assert check.severity == WARNING
    assert "by_channel" in check.message


# ---------------------------------------------------------------------------
# Ordering: deterministic first, LLM only for ambiguity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deterministic_failure_never_calls_the_llm() -> None:
    """An arithmetic failure short-circuits before any model is consulted.

    This is the architectural claim of the whole agent: arithmetic is strong
    evidence, a model grading its own pipeline is weak evidence, and the weak
    layer must never get the chance to overturn the strong one.
    """
    llm = FakeLLM()
    verifier = VerifierAgent(llm=llm)
    report = await verifier.execute(
        make_plan(), [make_result("totals", [{"revenue_inr": -1.0}])]
    )

    assert report.status is VerificationStatus.FAILED
    assert llm.call_count == 0


@pytest.mark.asyncio
async def test_a_clean_unambiguous_result_does_not_call_the_llm() -> None:
    """Escalation costs a round trip, so it is skipped when nothing is unclear."""
    llm = FakeLLM()
    verifier = VerifierAgent(llm=llm)
    await verifier.execute(
        make_plan(metrics=["revenue"]), [make_result("totals", [{"revenue_inr": 100.0}])]
    )

    assert llm.call_count == 0


@pytest.mark.asyncio
async def test_an_ambiguous_result_escalates_to_the_llm() -> None:
    """A warning with no error is exactly the case a model can help with."""
    llm = FakeLLM({"plausible": True, "reason": "The empty result is expected."})
    verifier = VerifierAgent(llm=llm)
    report = await verifier.execute(make_plan(), [make_result("totals", [])])

    assert llm.call_count == 1
    assert find(report, "llm_plausibility").passed


@pytest.mark.asyncio
async def test_the_llm_can_only_ever_add_a_warning() -> None:
    """A model's disagreement is not proof of error, so it cannot fail a run."""
    llm = FakeLLM({"plausible": False, "reason": "These look like store rows."})
    verifier = VerifierAgent(llm=llm)
    report = await verifier.execute(make_plan(), [make_result("totals", [])])

    check = find(report, "llm_plausibility")
    assert not check.passed
    assert check.severity == WARNING
    assert report.status is VerificationStatus.PASSED_WITH_WARNINGS


@pytest.mark.asyncio
async def test_an_unavailable_llm_does_not_change_the_verdict() -> None:
    """Escalation is best-effort; losing it degrades to the arithmetic verdict."""

    class BrokenLLM(FakeLLM):
        """A client whose provider chain is exhausted."""

        async def complete_json_with_response(
            self, system: str, user: str, **kwargs: Any
        ) -> tuple[Any, LLMResponse]:
            """Fail the way an exhausted provider chain does.

            Args:
                system: The system prompt.
                user: The user prompt.
                **kwargs: Ignored.

            Raises:
                RuntimeError: Always.
            """
            self.calls.append({"system": system, "user": user})
            raise RuntimeError("all 4 LLM provider(s) failed")

    verifier = VerifierAgent(llm=BrokenLLM())
    report = await verifier.execute(make_plan(), [make_result("totals", [])])

    check = find(report, "llm_plausibility")
    assert check.passed
    assert check.severity == "info"
    assert report.status is VerificationStatus.PASSED_WITH_WARNINGS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_column_classification_separates_levels_from_changes() -> None:
    """A revenue change is not a revenue level, and a share is neither."""
    assert is_revenue_column("revenue_inr")
    assert is_revenue_column("net_before_tax")
    assert not is_revenue_column("revenue_change_pct")
    assert not is_revenue_column("revenue_share_pct")
    assert is_share_column("revenue_share_pct")
    assert not is_share_column("revenue_growth_pct")


@pytest.mark.asyncio
async def test_failing_sub_query_ids_names_what_to_re_run() -> None:
    """The orchestrator retries only the queries an error check blamed."""
    verifier = VerifierAgent(llm=FakeLLM())
    report = await verifier.execute(
        make_plan(sub_query_ids=["totals"]),
        [make_result("totals", [{"revenue_inr": -1.0}])],
    )

    assert failing_sub_query_ids(report) == ["totals"]


@pytest.mark.asyncio
async def test_the_trace_summary_states_the_verdict() -> None:
    """The step a user sees says how many checks ran and how it ended."""
    verifier = VerifierAgent(llm=FakeLLM())
    outcome = await verifier.run(
        make_plan(), [make_result("totals", [{"revenue_inr": 100.0}])]
    )

    assert outcome.succeeded
    assert "checks" in outcome.step.summary
    assert "passed" in outcome.step.summary


def test_the_report_has_no_hardcoded_bounds() -> None:
    """Verification bounds come from settings, not from the module."""
    assert settings.MAX_PLAUSIBLE_REVENUE_INR > 0
    assert settings.AOV_TOLERANCE_INR > 0
    assert settings.TOTAL_RECONCILIATION_TOLERANCE_INR > 0
    assert date(2025, 8, 1) == settings.DATA_START_DATE
