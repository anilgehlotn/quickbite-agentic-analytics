"""Tests for the insight agent.

The model is scripted, so these tests assert on the agent's own guarantees
rather than on narrative quality: that a well-formed reply validates, that a
figure the data does not contain is caught, that a dead model still yields
correct numbers, and that the chart matches the shape of the result.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.agents.contracts import (
    AnalysisPlan,
    ChartType,
    Insight,
    QueryIntent,
    QueryResult,
    SubQuery,
    TimeWindow,
    VerificationCheck,
    VerificationReport,
    VerificationStatus,
)
from app.agents.insight import (
    DEGRADED_CONFIDENCE,
    InsightAgent,
    build_degraded_insight,
    choose_chart,
    find_unsupported_identifiers,
    find_unsupported_numbers,
    flag_summary,
    schema_without_examples,
    seasonality_context,
)
from app.config import settings
from app.core.llm import LLMResponse


class FakeLLM:
    """A scripted stand-in for :class:`LLMClient`."""

    def __init__(self, payload: Any) -> None:
        """Initialise the fake.

        Args:
            payload: The value every call returns.
        """
        self.payload = payload
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
            provider="gemini",
            model="gemini-flash-latest",
            input_tokens=800,
            output_tokens=400,
            latency_ms=900.0,
            attempts=["gemini"],
        )


class BrokenLLM:
    """A client whose provider chain is exhausted."""

    def __init__(self) -> None:
        """Initialise the fake."""
        self.calls = 0

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
        self.calls += 1
        raise RuntimeError("all 4 LLM provider(s) failed")


def make_plan(
    question: str = "What was revenue in the last 3 months?",
    intent: QueryIntent = QueryIntent.AGGREGATE,
) -> AnalysisPlan:
    """Build a plan for an insight test.

    Args:
        question: The question the plan answers.
        intent: The classified intent.

    Returns:
        A valid plan.
    """
    return AnalysisPlan(
        question=question,
        intent=intent,
        time_window=TimeWindow(
            start_date=settings.LAST_3M_START,
            end_date=settings.LAST_3M_END,
            label="last 3 months",
        ),
        metrics=["revenue", "orders", "aov"],
        dimensions=[],
        sub_queries=[SubQuery(id="totals", purpose="Compute the headline figures.")],
        requires_diagnostics=False,
        reasoning="A single aggregate answers the question.",
        confidence=0.95,
    )


def make_result(
    rows: list[dict[str, Any]], sub_query_id: str = "totals"
) -> QueryResult:
    """Build a query result for an insight test.

    Args:
        rows: The rows returned.
        sub_query_id: The sub-query this result belongs to.

    Returns:
        A valid result.
    """
    return QueryResult(
        sub_query_id=sub_query_id,
        sql="SELECT SUM(net_before_tax) AS revenue_inr FROM fact_orders",
        columns=list(rows[0]) if rows else [],
        rows=rows,
        row_count=len(rows),
        execution_ms=2.0,
    )


HEADLINE_ROWS: list[dict[str, Any]] = [
    {"revenue_inr": 3197076.5, "orders": 4930, "aov_inr": 648.49}
]


def good_payload(**overrides: Any) -> dict[str, Any]:
    """Build a well-formed model reply.

    Args:
        **overrides: Fields to replace inside the insight object.

    Returns:
        A payload with an insight and a chart.
    """
    insight = {
        "headline": (
            "Revenue was 3,197,076.5 INR across 4,930 orders in the last three "
            "months."
        ),
        "narrative": (
            "Trading was steady. Average order value held at 648.49 INR, so "
            "the result is driven by order volume rather than basket size."
        ),
        "key_findings": ["Average order value was 648.49 INR."],
        "caveats": ["Revenue is tax-exclusive."],
        "recommended_actions": [],
        "confidence": 0.9,
    }
    insight.update(overrides)
    return {
        "insight": insight,
        "chart": {
            "chart_type": "none",
            "x_field": "",
            "y_fields": [],
            "title": "Revenue, orders and AOV",
            "series_field": None,
        },
    }


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_well_formed_reply_validates_into_an_insight() -> None:
    """The model's JSON becomes a validated contract object."""
    agent = InsightAgent(llm=FakeLLM(good_payload()))
    bundle = await agent.execute(make_plan(), [make_result(HEADLINE_ROWS)])

    assert isinstance(bundle.insight, Insight)
    assert bundle.insight.headline.startswith("Revenue was 3,197,076.5")
    assert bundle.degraded is False
    assert bundle.unsupported_numbers == []


@pytest.mark.asyncio
async def test_the_prompt_carries_the_analytical_rules_and_seasonality() -> None:
    """The rules that make this a consulting layer are actually sent."""
    llm = FakeLLM(good_payload())
    agent = InsightAgent(llm=llm)
    await agent.execute(make_plan(), [make_result(HEADLINE_ROWS)])

    system = llm.calls[0]["system"]
    assert "NEVER INVENT A NUMBER" in system
    assert "DECOMPOSE EVERY CHANGE" in system
    assert "RETURN TO NORMAL" in system.upper()
    assert '"NONE" IS AN ANSWER' in system
    assert settings.SEASONAL_PEAK_MONTH in system
    assert settings.SEASONAL_TROUGH_MONTH in system


@pytest.mark.asyncio
async def test_the_prompt_never_shows_the_contract_example() -> None:
    """The Insight example contains findings about this dataset.

    A live run reproduced them verbatim - naming stores and figures no query
    in that run had returned - because model_json_schema() embeds the example.
    The schema teaches the shape; the rows must supply the content.
    """
    llm = FakeLLM(good_payload())
    agent = InsightAgent(llm=llm)
    await agent.execute(make_plan(), [make_result(HEADLINE_ROWS)])

    system = llm.calls[0]["system"]
    example = Insight.model_config["json_schema_extra"]["example"]
    assert example["headline"] not in system
    assert "48,860" not in system
    assert "ST039" not in system
    # The shape must survive the stripping.
    assert "recommended_actions" in system
    assert "key_findings" in system


def test_stripping_examples_leaves_the_schema_intact() -> None:
    """Removing examples must not remove the field definitions."""
    schema = schema_without_examples(Insight)

    assert "example" not in schema
    assert set(schema["properties"]) == set(Insight.model_fields)


@pytest.mark.asyncio
async def test_the_prompt_carries_the_rows_and_the_verification_verdict() -> None:
    """The model reasons over the data, not over a summary of it."""
    llm = FakeLLM(good_payload())
    agent = InsightAgent(llm=llm)
    report = VerificationReport(
        status=VerificationStatus.PASSED,
        checks=[
            VerificationCheck(
                name="aov_reconciles",
                passed=True,
                severity="error",
                message="AOV reconciles.",
            )
        ],
        summary="All checks passed.",
    )
    await agent.execute(make_plan(), [make_result(HEADLINE_ROWS)], report)

    user = llm.calls[0]["user"]
    assert "3197076.5" in user
    assert "4930" in user
    assert "passed" in user


@pytest.mark.asyncio
async def test_usage_is_recorded_for_the_trace() -> None:
    """Token spend reaches the trace rather than being estimated."""
    agent = InsightAgent(llm=FakeLLM(good_payload()))
    outcome = await agent.run(make_plan(), [make_result(HEADLINE_ROWS)])

    assert outcome.succeeded
    assert outcome.step.tokens == 1200
    assert outcome.step.llm_provider == "gemini"


# ---------------------------------------------------------------------------
# Fabricated numbers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_number_absent_from_the_results_is_flagged() -> None:
    """The worst failure mode is a plausible invented figure, so it is caught."""
    payload = good_payload(
        headline="Revenue was 5,412,900 INR across 4,930 orders.",
    )
    agent = InsightAgent(llm=FakeLLM(payload))
    bundle = await agent.execute(make_plan(), [make_result(HEADLINE_ROWS)])

    assert "5,412,900" in bundle.unsupported_numbers
    assert any("could not be traced" in caveat for caveat in bundle.insight.caveats)
    assert bundle.insight.confidence < 0.9


def test_rounding_and_scaling_do_not_count_as_fabrication() -> None:
    """A figure written as 3.2M or rounded to whole rupees is still supported."""
    results = [make_result(HEADLINE_ROWS)]

    assert find_unsupported_numbers("Revenue was 3,197,076.5 INR.", results) == []
    assert find_unsupported_numbers("Revenue was 3,197,077 INR.", results) == []
    assert find_unsupported_numbers("Revenue was 3.2M INR.", results) == []
    assert find_unsupported_numbers("Revenue was 31.97 lakh INR.", results) == []


def test_percentages_and_small_counts_are_not_treated_as_figures() -> None:
    """A derived growth rate or a count of rows is arithmetic, not fabrication."""
    results = [make_result(HEADLINE_ROWS)]

    assert find_unsupported_numbers("Revenue fell 8.4% over the period.", results) == []
    assert find_unsupported_numbers("9 of the 50 stores declined.", results) == []
    assert find_unsupported_numbers("The window ends 2026-07-31.", results) == []


def test_a_period_over_period_difference_is_supported() -> None:
    """Subtracting two months of one column is analysis, not invention.

    Found in a live run, where correct decline figures were being flagged and
    the caveat was telling the user to distrust the right answer.
    """
    trend = make_result(
        [
            {"month_key": "2026-05", "revenue_inr": 129210.0},
            {"month_key": "2026-06", "revenue_inr": 130530.0},
            {"month_key": "2026-07", "revenue_inr": 124005.0},
        ]
    )

    assert find_unsupported_numbers("Hyderabad fell by 5,205 INR.", [trend]) == []
    assert find_unsupported_numbers("It rose 1,320 INR in June.", [trend]) == []


def test_a_difference_across_unrelated_columns_is_not_supported() -> None:
    """The allowance is per column, or it would excuse almost any number."""
    result = make_result([{"revenue_inr": 900.0, "orders": 100}])

    assert find_unsupported_numbers("The gap was 800 INR.", [result]) == ["800"]


def test_the_datasets_own_years_are_not_treated_as_figures() -> None:
    """A year written in prose, as in 'July 2026', is a date not a figure."""
    results = [make_result(HEADLINE_ROWS)]

    assert find_unsupported_numbers("Revenue ran from May to July 2026.", results) == []


def test_a_fabricated_magnitude_is_caught() -> None:
    """A figure of the class the data contains must come from the data."""
    results = [make_result(HEADLINE_ROWS)]

    assert find_unsupported_numbers("AOV was 812.40 INR.", results) == ["812.40"]


# ---------------------------------------------------------------------------
# Degradation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_failure_produces_a_degraded_insight_not_an_exception() -> None:
    """Correct numbers without a story beat an error page."""
    agent = InsightAgent(llm=BrokenLLM())
    bundle = await agent.execute(make_plan(), [make_result(HEADLINE_ROWS)])

    assert bundle.degraded is True
    assert bundle.insight.confidence == DEGRADED_CONFIDENCE
    assert "3,197,076.50" in bundle.insight.key_findings[0]
    assert any("unavailable" in caveat for caveat in bundle.insight.caveats)


@pytest.mark.asyncio
async def test_a_malformed_reply_degrades_rather_than_raising() -> None:
    """A model that ignores the schema must not take the request down."""
    agent = InsightAgent(llm=FakeLLM({"insight": {"headline": "no other fields"}}))
    bundle = await agent.execute(make_plan(), [make_result(HEADLINE_ROWS)])

    assert bundle.degraded is True


@pytest.mark.asyncio
async def test_the_degraded_insight_never_invents_a_number() -> None:
    """The deterministic path is only allowed to restate the rows."""
    agent = InsightAgent(llm=BrokenLLM())
    bundle = await agent.execute(make_plan(), [make_result(HEADLINE_ROWS)])

    text = " ".join([bundle.insight.headline, *bundle.insight.key_findings])
    assert find_unsupported_numbers(text, [make_result(HEADLINE_ROWS)]) == []


def test_the_degraded_headline_does_not_present_one_row_as_the_answer() -> None:
    """With 300 store-months there is no single headline figure to state."""
    rows = [
        {"store_id": f"ST{index:03d}", "month_key": "2026-05", "orders": index}
        for index in range(1, 20)
    ]
    insight = build_degraded_insight("Which stores declined?", make_plan(), [make_result(rows)])

    assert "19 rows" in insight.headline
    assert "is 1." not in insight.headline


def test_the_degraded_insight_handles_having_no_rows_at_all() -> None:
    """The floor of the system must not itself fail."""
    insight = build_degraded_insight("Anything?", make_plan(), [])

    assert insight.confidence == 0.0
    assert insight.key_findings == []


@pytest.mark.asyncio
async def test_failed_verification_is_stated_in_the_caveats() -> None:
    """The user is told when the numbers did not pass their own checks."""
    report = VerificationReport(
        status=VerificationStatus.FAILED,
        checks=[
            VerificationCheck(
                name="parts_sum_to_total",
                passed=False,
                severity="error",
                message="The breakdown misses the total by 100.00 INR.",
            )
        ],
        summary="1 consistency check failed.",
    )
    agent = InsightAgent(llm=FakeLLM(good_payload()))
    bundle = await agent.execute(make_plan(), [make_result(HEADLINE_ROWS)], report)

    assert "failed automated consistency checks" in bundle.insight.caveats[0]
    assert bundle.insight.confidence <= 0.4


# ---------------------------------------------------------------------------
# Chart selection
# ---------------------------------------------------------------------------


def test_a_time_series_yields_a_line_chart() -> None:
    """A metric over months is a line."""
    result = make_result(
        [
            {"month_key": "2026-05", "revenue_inr": 1052233.5},
            {"month_key": "2026-06", "revenue_inr": 1041607.5},
            {"month_key": "2026-07", "revenue_inr": 1103235.5},
        ]
    )
    chart = choose_chart(make_plan(intent=QueryIntent.TREND), [result])

    assert chart.chart_type is ChartType.LINE
    assert chart.x_field == "month_key"
    assert chart.y_fields == ["revenue_inr"]


def test_a_ranking_yields_a_bar_chart() -> None:
    """Categories compared against each other are bars."""
    result = make_result(
        [
            {"store_name": "Kolkata 39", "revenue_inr": 25150.0},
            {"store_name": "Pune 12", "revenue_inr": 24010.0},
        ]
    )
    chart = choose_chart(make_plan(intent=QueryIntent.RANKING), [result])

    assert chart.chart_type is ChartType.BAR
    assert chart.x_field == "store_name"


def test_a_single_value_yields_no_chart() -> None:
    """One number is already the headline; a single bar adds nothing."""
    chart = choose_chart(make_plan(), [make_result(HEADLINE_ROWS)])

    assert chart.chart_type is ChartType.NONE


def test_a_dimension_across_time_yields_a_grouped_bar() -> None:
    """Two dimensions at once need a series, not a single line."""
    result = make_result(
        [
            {"month_key": "2026-05", "channel": "Swiggy", "revenue_inr": 1.0},
            {"month_key": "2026-06", "channel": "Swiggy", "revenue_inr": 2.0},
            {"month_key": "2026-05", "channel": "Zomato", "revenue_inr": 3.0},
        ]
    )
    chart = choose_chart(make_plan(intent=QueryIntent.COMPARISON), [result])

    assert chart.chart_type is ChartType.GROUPED_BAR
    assert chart.series_field == "channel"


@pytest.mark.asyncio
async def test_a_chart_naming_columns_that_do_not_exist_is_replaced() -> None:
    """A chart pointing at a missing column renders as a broken product."""
    payload = good_payload()
    payload["chart"] = {
        "chart_type": "line",
        "x_field": "quarter",
        "y_fields": ["profit"],
        "title": "Invented columns",
        "series_field": None,
    }
    agent = InsightAgent(llm=FakeLLM(payload))
    bundle = await agent.execute(make_plan(), [make_result(HEADLINE_ROWS)])

    assert bundle.chart.chart_type is ChartType.NONE


@pytest.mark.asyncio
async def test_a_valid_model_chart_is_kept() -> None:
    """The model's own choice wins when it references real columns."""
    payload = good_payload()
    payload["chart"] = {
        "chart_type": "bar",
        "x_field": "orders",
        "y_fields": ["revenue_inr"],
        "title": "Revenue by order count",
        "series_field": None,
    }
    agent = InsightAgent(llm=FakeLLM(payload))
    bundle = await agent.execute(make_plan(), [make_result(HEADLINE_ROWS)])

    assert bundle.chart.chart_type is ChartType.BAR
    assert bundle.chart.title == "Revenue by order count"


def test_seasonality_context_comes_from_settings() -> None:
    """The seasonal shape is configuration, not a sentence in a prompt."""
    text = seasonality_context()

    assert settings.SEASONAL_PEAK_MONTH in text
    assert settings.SEASONAL_TROUGH_MONTH in text
    assert f"{settings.SEASONAL_SPREAD:.2f}x" in text


# ---------------------------------------------------------------------------
# Identifier traceability
# ---------------------------------------------------------------------------


STORE_ROWS: list[dict[str, Any]] = [
    {"store_id": "ST007", "store_name": "QuickBite Kolkata 07", "revenue_inr": 44151.0},
    {"store_id": "ST042", "store_name": "QuickBite Mumbai 42", "revenue_inr": 62345.0},
]


def test_a_mistyped_store_id_is_caught() -> None:
    """ST07 is not ST007, and nothing numeric would notice.

    From a live answer: the findings list had ST007, the headline had ST07.
    """
    results = [make_result(STORE_ROWS)]

    assert find_unsupported_identifiers("Nine stores including ST07 declined.", results) == [
        "ST07"
    ]


def test_correct_identifiers_are_not_flagged() -> None:
    """Ids copied from the results pass unremarked."""
    results = [make_result(STORE_ROWS)]

    assert (
        find_unsupported_identifiers("ST007 and ST042 both declined.", results) == []
    )


def test_an_identifier_embedded_in_a_name_is_not_flagged() -> None:
    """A code that only appears inside a longer label still counts."""
    results = [
        make_result([{"store_name": "QuickBite ST099 Annexe", "revenue_inr": 1.0}])
    ]

    assert find_unsupported_identifiers("ST099 is the outlier.", results) == []


def test_ordinary_prose_is_not_mistaken_for_an_identifier() -> None:
    """The pattern must not fire on words, years or short codes."""
    results = [make_result(STORE_ROWS)]

    assert find_unsupported_identifiers("Revenue in 2026 rose in Q3 by 12%.", results) == []


@pytest.mark.asyncio
async def test_a_mistyped_identifier_becomes_a_caveat() -> None:
    """The reader is told which id could not be found in the data."""
    payload = good_payload(
        headline="Store ST07 declined the most.",
        key_findings=["Store ST07 declined the most."],
    )
    agent = InsightAgent(llm=FakeLLM(payload))
    bundle = await agent.execute(make_plan(), [make_result(STORE_ROWS)])

    assert bundle.unsupported_identifiers == ["ST07"]
    assert any("ST07" in caveat for caveat in bundle.insight.caveats)


@pytest.mark.asyncio
async def test_the_prompt_forbids_reformatting_identifiers() -> None:
    """The rule is stated, not only checked after the fact."""
    llm = FakeLLM(good_payload())
    agent = InsightAgent(llm=llm)
    await agent.execute(make_plan(), [make_result(HEADLINE_ROWS)])

    system = llm.calls[0]["system"]
    assert "IDENTIFIERS ARE COPIED" in system
    assert "ST007 is not ST07" in system


# ---------------------------------------------------------------------------
# Empty qualifying sets
# ---------------------------------------------------------------------------


def test_an_empty_qualifying_group_is_stated_explicitly() -> None:
    """"None qualified" must be visible, not inferred from a missing key."""
    result = make_result(
        [
            {"city": "Bengaluru", "is_strictly_declining": 0},
            {"city": "Chennai", "is_strictly_declining": 0},
        ]
    )

    summary = flag_summary(result)

    assert summary["is_strictly_declining"]["1"] == []
    assert set(summary["is_strictly_declining"]["0"]) == {"Bengaluru", "Chennai"}


def test_a_populated_qualifying_group_lists_its_members() -> None:
    """The membership list replaces the row-by-row comparison."""
    result = make_result(
        [
            {"store_id": "ST007", "is_strictly_declining": 1},
            {"store_id": "ST002", "is_strictly_declining": 1},
            {"store_id": "ST004", "is_strictly_declining": 0},
        ]
    )

    summary = flag_summary(result)

    assert summary["is_strictly_declining"]["1"] == ["ST007", "ST002"]
    assert summary["is_strictly_declining"]["0"] == ["ST004"]


@pytest.mark.asyncio
async def test_the_prompt_requires_leading_with_none_when_nothing_qualifies(
) -> None:
    """The rule that stops a weaker test being substituted for the answer."""
    llm = FakeLLM(good_payload())
    agent = InsightAgent(llm=llm)
    await agent.execute(make_plan(), [make_result(HEADLINE_ROWS)])

    system = llm.calls[0]["system"]
    assert "FIRST SENTENCE must state that none met the criterion" in system
    assert "Never lead with the weaker test" in system
