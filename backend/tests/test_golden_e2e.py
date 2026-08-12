"""End-to-end tests of the eight evaluation questions against ground truth.

This is the file that decides whether the system works. Everything else tests a
component in isolation; this runs the real orchestrator over the real database
and checks the answers against ``golden_answers.json``, which was computed from
the source Excel with pandas and shares no code with the SQL layer.

Two modes:

* **Replay (default, runs anywhere).** The LLM is replaced by a fake that
  returns the plan, the SQL and the narrative that the live system actually
  produced, taken from the warmed answer cache. Everything downstream of the
  model is real: the guard validates the SQL, SQLite executes it, the verifier
  checks the rows, and the assertions compare the results to ground truth. No
  network, no cost, and the fixtures cannot drift from what the system does,
  because they *are* what the system did.
* **Live (opt-in via ``QUICKBITE_E2E_LIVE=1``).** Calls the configured
  provider. Slower, costs money, and non-deterministic - which is exactly why
  it is not the default and why the replay mode exists.

The replay mode's one honest limitation: it cannot catch a regression in the
prompts, because it never asks the model anything. It catches regressions in
everything else, which is the part that must never break silently.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Final

import pytest

from app.agents.analyst import SQLAnalystAgent
from app.agents.contracts import AnalysisResponse, VerificationStatus
from app.agents.insight import InsightAgent, find_unsupported_numbers
from app.agents.orchestrator import CANONICAL_QUESTIONS, Orchestrator
from app.agents.planner import PlannerAgent
from app.agents.verifier import VerifierAgent
from app.config import settings
from app.core.llm import LLMResponse

GOLDEN_PATH: Final[Path] = Path(__file__).parent / "golden_answers.json"

# Tolerances, from the spec: money to the rupee, ratios to the paisa.
REVENUE_TOLERANCE_INR: Final[float] = 1.0
RATIO_TOLERANCE: Final[float] = 0.01

# Every stage that must appear in a complete trace.
EXPECTED_AGENTS: Final[tuple[str, ...]] = (
    "planner",
    "sql_analyst",
    "verifier",
    "insight",
)

LIVE_MODE: Final[bool] = os.environ.get("QUICKBITE_E2E_LIVE", "").strip() in {
    "1",
    "true",
    "yes",
}

# Phrases that turn a named entity into a statement about what did NOT happen.
NEGATION_MARKERS: Final[tuple[str, ...]] = (
    "no city",
    "none of",
    "no cities",
    "zero of",
    "zero cities",
    "not decline",
    "no store",
    "did not decline",
    "none declined",
)

# Phrases that mark a store as deliberately deprioritised.
DEPRIORITISING_MARKERS: Final[tuple[str, ...]] = (
    "above",
    "lower priority",
    "not the top",
    "less concerning",
    "reverting",
    "revert",
    "normalis",
    "normaliz",
    "rebound",
    "still ahead",
    "despite",
)


def load_golden() -> dict[str, Any]:
    """Read the ground truth.

    Returns:
        The parsed golden answers.
    """
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


GOLDEN: Final[dict[str, Any]] = load_golden()


def load_cached_responses() -> dict[str, AnalysisResponse]:
    """Read the warmed cache, keyed by canonical question id.

    Returns:
        One response per canonical question that is present in the cache.
        Empty when the cache file is missing or unreadable.
    """
    if not settings.CACHE_PATH.exists():
        return {}
    try:
        payload = json.loads(settings.CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    from app.core.cache import normalise_question

    entries = payload.get("entries", {})
    responses: dict[str, AnalysisResponse] = {}
    for entry in CANONICAL_QUESTIONS:
        key = normalise_question(entry["question"])
        stored = entries.get(key)
        if not stored:
            continue
        try:
            responses[entry["id"]] = AnalysisResponse.model_validate(
                stored["response"]
            )
        except Exception:  # noqa: BLE001 - a bad entry is simply unavailable
            continue
    return responses


CACHED: Final[dict[str, AnalysisResponse]] = load_cached_responses()


class ReplayLLM:
    """Replays one cached run in place of a provider.

    Dispatches on the system prompt, the same way the four agents differ from
    each other, and returns what the live system returned for that stage. The
    SQL is matched to the sub-query by its purpose, which appears verbatim in
    the analyst's user prompt.
    """

    def __init__(self, response: AnalysisResponse) -> None:
        """Initialise the replay.

        Args:
            response: The cached run to replay.

        Raises:
            ValueError: If the cached run has no plan to replay.
        """
        if response.plan is None:
            raise ValueError("cached response has no plan")
        self.response = response
        self.plan = response.plan
        self.sql_by_id = {
            result.sub_query_id: result.sql for result in response.query_results
        }
        self.purpose_by_id = {
            sub_query.id: sub_query.purpose for sub_query in self.plan.sub_queries
        }
        self.sql_calls: list[str] = []
        self.json_calls: list[str] = []

    def _completion(self) -> LLMResponse:
        """Build a synthetic completion carrying the cached provider name.

        Returns:
            A completion with token counts that keep the trace realistic.
        """
        return LLMResponse(
            text="{}",
            provider=self.response.trace.providers_used[0]
            if self.response.trace.providers_used
            else "replay",
            model="replay",
            input_tokens=100,
            output_tokens=50,
            latency_ms=1.0,
            attempts=["replay"],
        )

    async def complete(self, system: str, user: str, **kwargs: Any) -> LLMResponse:
        """Return the SQL the live system produced for this sub-query.

        Args:
            system: The system prompt, unused.
            user: The user prompt, which names the sub-query's purpose.
            **kwargs: Ignored generation settings.

        Returns:
            A completion whose text is the cached SQL.
        """
        self.sql_calls.append(user)
        for sub_query_id, purpose in self.purpose_by_id.items():
            if purpose and purpose in user and sub_query_id in self.sql_by_id:
                sql = self.sql_by_id[sub_query_id]
                if sql:
                    return LLMResponse(
                        text=sql,
                        provider="replay",
                        model="replay",
                        input_tokens=100,
                        output_tokens=50,
                        latency_ms=1.0,
                        attempts=["replay"],
                    )
        # A sub-query with no cached SQL is one the live run could not answer.
        return LLMResponse(
            text="SELECT 1 AS unavailable",
            provider="replay",
            model="replay",
            input_tokens=10,
            output_tokens=5,
            latency_ms=1.0,
            attempts=["replay"],
        )

    async def complete_json_with_response(
        self, system: str, user: str, **kwargs: Any
    ) -> tuple[Any, LLMResponse]:
        """Return the cached plan, narrative or verification verdict.

        Args:
            system: The system prompt, used to identify the calling agent.
            user: The user prompt, unused.
            **kwargs: Ignored generation settings.

        Returns:
            The cached payload for that stage, and a synthetic completion.

        Raises:
            AssertionError: If the system prompt matches no known agent.
        """
        self.json_calls.append(system)
        if "planning agent" in system:
            return self.plan.model_dump(mode="json"), self._completion()
        if "checking layer" in system:
            return {"plausible": True, "reason": "replayed"}, self._completion()
        if "senior retail analyst" in system:
            insight = (
                self.response.insight.model_dump(mode="json")
                if self.response.insight
                else {}
            )
            chart = (
                self.response.chart.model_dump(mode="json")
                if self.response.chart
                else None
            )
            return {"insight": insight, "chart": chart}, self._completion()
        raise AssertionError(f"unrecognised system prompt: {system[:80]}")


async def run_question(question_id: str) -> AnalysisResponse:
    """Answer one canonical question in the configured mode.

    Args:
        question_id: The canonical question id.

    Returns:
        The response, freshly computed in live mode or replayed in the default
        mode.
    """
    question = next(
        entry["question"] for entry in CANONICAL_QUESTIONS if entry["id"] == question_id
    )
    if LIVE_MODE:
        return await Orchestrator().run(question)

    llm = ReplayLLM(CACHED[question_id])
    orchestrator = Orchestrator(
        planner=PlannerAgent(llm=llm),  # type: ignore[arg-type]
        analyst=SQLAnalystAgent(llm=llm),
        verifier=VerifierAgent(llm=llm),  # type: ignore[arg-type]
        insight=InsightAgent(llm=llm),  # type: ignore[arg-type]
    )
    return await orchestrator.run(question)


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def all_numbers(response: AnalysisResponse) -> list[float]:
    """Every numeric value the queries returned.

    Args:
        response: The answer to scan.

    Returns:
        All numeric cell values.
    """
    return [
        float(value)
        for result in response.query_results
        for row in result.rows
        for value in row.values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]


def assert_figure_present(
    response: AnalysisResponse, target: float, tolerance: float, label: str
) -> None:
    """Assert a ground-truth figure appears among the query results.

    Args:
        response: The answer to check.
        target: The expected figure.
        tolerance: Allowed absolute difference.
        label: Name used in the failure message.

    Raises:
        AssertionError: If no returned value matches.
    """
    values = all_numbers(response)
    assert any(abs(value - target) <= tolerance for value in values), (
        f"{label}: expected {target:,.2f} (+/-{tolerance}) in the query "
        f"results, and it is absent"
    )


def column_values(response: AnalysisResponse, column: str) -> list[list[Any]]:
    """Collect one column's values from every result that has it.

    Args:
        response: The answer to scan.
        column: The column name.

    Returns:
        One list of values per result, in row order.
    """
    found: list[list[Any]] = []
    for result in response.query_results:
        if column in result.columns:
            found.append([row.get(column) for row in result.rows])
    return found


def contains_ordered_run(sequences: list[list[Any]], expected: list[Any]) -> bool:
    """Whether some sequence contains the expected run consecutively.

    Order matters, so a ranking that returns the right stores in the wrong
    order fails. A reversed match also counts, because "bottom 5 ascending"
    and "the last 5 of a descending list" are the same answer.

    Args:
        sequences: Candidate sequences, one per result.
        expected: The run to look for.

    Returns:
        True when one sequence contains the run, forwards or backwards.
    """
    reversed_expected = list(reversed(expected))
    width = len(expected)
    for sequence in sequences:
        for start in range(0, max(0, len(sequence) - width + 1)):
            window = sequence[start : start + width]
            if window == expected or window == reversed_expected:
                return True
    return False


def insight_text(response: AnalysisResponse) -> str:
    """Concatenate every piece of narrative in one lowercase string.

    Args:
        response: The answer to read.

    Returns:
        Headline, narrative, findings, caveats and actions, lowercased.
    """
    if response.insight is None:
        return ""
    insight = response.insight
    parts = [
        insight.headline,
        insight.narrative,
        *insight.key_findings,
        *insight.caveats,
        *insight.recommended_actions,
    ]
    return " ".join(parts).lower()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

QUESTION_IDS: Final[list[str]] = [entry["id"] for entry in CANONICAL_QUESTIONS]


@pytest.fixture(scope="module")
def answers() -> dict[str, AnalysisResponse]:
    """Answer every canonical question once, and share the results.

    Running the pipeline eight times per test would be wasteful in replay mode
    and unaffordable in live mode.

    Returns:
        One response per canonical question id.
    """
    import asyncio

    if not LIVE_MODE and not CACHED:
        pytest.skip(
            "no warmed cache at "
            f"{settings.CACHE_PATH}; run scripts/warm_cache.py, or set "
            "QUICKBITE_E2E_LIVE=1 to call a live provider"
        )

    async def run_all() -> dict[str, AnalysisResponse]:
        """Run each question in sequence.

        Returns:
            The answers, keyed by question id.
        """
        results: dict[str, AnalysisResponse] = {}
        for question_id in QUESTION_IDS:
            if not LIVE_MODE and question_id not in CACHED:
                continue
            results[question_id] = await run_question(question_id)
        return results

    return asyncio.run(run_all())


def answer_for(answers: dict[str, AnalysisResponse], question_id: str) -> AnalysisResponse:
    """Fetch one answer, skipping the test when it is not available.

    Args:
        answers: Every answer produced for this module.
        question_id: The question wanted.

    Returns:
        The response.
    """
    if question_id not in answers:
        pytest.skip(f"{question_id} is not in the warmed cache")
    return answers[question_id]


# ---------------------------------------------------------------------------
# Universal assertions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("question_id", QUESTION_IDS)
def test_every_question_is_answered(
    answers: dict[str, AnalysisResponse], question_id: str
) -> None:
    """Each evaluation question produces an answer."""
    response = answer_for(answers, question_id)

    assert response.answered is True, f"{question_id}: {response.error}"
    assert response.insight is not None
    assert response.query_results


@pytest.mark.parametrize("question_id", QUESTION_IDS)
def test_verification_never_fails(
    answers: dict[str, AnalysisResponse], question_id: str
) -> None:
    """No answer ships with failing arithmetic checks."""
    response = answer_for(answers, question_id)

    assert response.verification is not None
    assert response.verification.status is not VerificationStatus.FAILED, (
        f"{question_id}: {response.verification.summary}"
    )


@pytest.mark.parametrize("question_id", QUESTION_IDS)
def test_the_trace_contains_every_agent(
    answers: dict[str, AnalysisResponse], question_id: str
) -> None:
    """The trace shows all four agents, which is what makes it evidence."""
    response = answer_for(answers, question_id)
    names = {step.agent_name for step in response.trace.steps}

    for agent in EXPECTED_AGENTS:
        assert any(name.startswith(agent) for name in names), (
            f"{question_id}: no {agent} step in {sorted(names)}"
        )


@pytest.mark.parametrize("question_id", QUESTION_IDS)
def test_no_figure_is_invented(
    answers: dict[str, AnalysisResponse], question_id: str
) -> None:
    """Every figure in the narrative traces back to a query result.

    The single worst failure this system could have is a confident, plausible,
    fabricated number, so it is asserted for all eight questions rather than
    spot-checked.
    """
    response = answer_for(answers, question_id)
    assert response.insight is not None
    text = " ".join([response.insight.headline, *response.insight.key_findings])

    unsupported = find_unsupported_numbers(text, response.query_results)
    assert not unsupported, (
        f"{question_id}: figures absent from the query results: {unsupported}"
    )


# ---------------------------------------------------------------------------
# Q1 - headline aggregates
# ---------------------------------------------------------------------------


def test_q1_revenue_orders_and_aov_match_ground_truth(
    answers: dict[str, AnalysisResponse],
) -> None:
    """Revenue, orders and AOV match the pandas ground truth."""
    response = answer_for(answers, "q1")
    golden = GOLDEN["q1"]

    assert_figure_present(
        response, golden["revenue_net_inr"], REVENUE_TOLERANCE_INR, "q1 revenue"
    )
    assert_figure_present(response, float(golden["orders"]), 0.5, "q1 orders")
    assert_figure_present(response, golden["aov_inr"], RATIO_TOLERANCE, "q1 AOV")


# ---------------------------------------------------------------------------
# Q2 - store ranking
# ---------------------------------------------------------------------------


def test_q2_top_and_bottom_five_stores_match_in_order(
    answers: dict[str, AnalysisResponse],
) -> None:
    """The best and worst five stores match exactly, in rank order."""
    response = answer_for(answers, "q2")
    golden = GOLDEN["q2"]
    sequences = column_values(response, "store_id") + column_values(
        response, "STORE_ID"
    )

    top = [store["STORE_ID"] for store in golden["top_5"]]
    bottom = [store["STORE_ID"] for store in golden["bottom_5"]]

    assert contains_ordered_run(sequences, top), (
        f"top 5 {top} not returned in order; got {[s[:6] for s in sequences]}"
    )
    assert contains_ordered_run(sequences, bottom), (
        f"bottom 5 {bottom} not returned in order"
    )


def test_q2_store_revenues_match_ground_truth(
    answers: dict[str, AnalysisResponse],
) -> None:
    """The revenue behind the ranking is right, not just the ordering."""
    response = answer_for(answers, "q2")

    for store in GOLDEN["q2"]["top_5"][:3]:
        assert_figure_present(
            response,
            store["revenue_net_inr"],
            REVENUE_TOLERANCE_INR,
            f"q2 {store['STORE_ID']} revenue",
        )


# ---------------------------------------------------------------------------
# Q3 - channels
# ---------------------------------------------------------------------------


def test_q3_all_four_channels_present_with_matching_revenue(
    answers: dict[str, AnalysisResponse],
) -> None:
    """Every channel appears, each with the right revenue."""
    response = answer_for(answers, "q3")
    channels = GOLDEN["q3"]["channels"]

    returned = {
        str(value)
        for sequence in column_values(response, "channel")
        + column_values(response, "CHANNEL")
        for value in sequence
    }
    for channel in channels:
        assert channel["CHANNEL"] in returned, (
            f"channel {channel['CHANNEL']} missing from {sorted(returned)}"
        )
        assert_figure_present(
            response,
            channel["revenue_net_inr"],
            REVENUE_TOLERANCE_INR,
            f"q3 {channel['CHANNEL']} revenue",
        )


# ---------------------------------------------------------------------------
# Q4 - products
# ---------------------------------------------------------------------------


def test_q4_top_skus_by_quantity_and_revenue_match(
    answers: dict[str, AnalysisResponse],
) -> None:
    """The top five SKUs match on both orderings the question implies."""
    response = answer_for(answers, "q4")
    golden = GOLDEN["q4"]
    sequences = column_values(response, "sku_id") + column_values(response, "SKU_ID")

    by_quantity = [sku["SKU_ID"] for sku in golden["top_5_by_quantity"]]
    by_revenue = [sku["SKU_ID"] for sku in golden["top_5_by_revenue"]]

    assert contains_ordered_run(sequences, by_quantity), (
        f"top 5 by quantity {by_quantity} not returned in order"
    )
    assert contains_ordered_run(sequences, by_revenue), (
        f"top 5 by revenue {by_revenue} not returned in order"
    )


# ---------------------------------------------------------------------------
# Q5 - the hallucination test
# ---------------------------------------------------------------------------


def test_q5_reports_that_no_city_declined(
    answers: dict[str, AnalysisResponse],
) -> None:
    """The honest answer is "none", and the temptation is to name someone.

    Ground truth says zero cities declined in every consecutive month. A model
    asked "which cities are declining?" will reach for a name, so this asserts
    the headline does not supply one without immediately negating it.

    This failed until the monotonic-decline set was computed in SQL rather
    than inferred: the answer used to lead with the one city below its
    prior-quarter baseline, which is a true statement about a different test.
    """
    response = answer_for(answers, "q5")
    assert GOLDEN["q5"]["declining_city_count"] == 0, "ground truth changed"
    assert response.insight is not None

    headline = response.insight.headline.lower()
    cities = [city["city"].lower() for city in GOLDEN["q5"]["cities"]]
    named = [city for city in cities if city in headline]
    negated = any(marker in headline for marker in NEGATION_MARKERS)

    assert negated or not named, (
        f"q5 headline names {named} without stating that no city declined: "
        f"{response.insight.headline!r}"
    )


def test_q5_city_revenues_match_ground_truth(
    answers: dict[str, AnalysisResponse],
) -> None:
    """The monthly city figures behind the answer are correct."""
    response = answer_for(answers, "q5")

    for city in GOLDEN["q5"]["cities"][:3]:
        for revenue in city["monthly_revenue_inr"].values():
            assert_figure_present(
                response,
                revenue,
                REVENUE_TOLERANCE_INR,
                f"q5 {city['city']} monthly revenue",
            )


# ---------------------------------------------------------------------------
# Q6 - weekend versus weekday
# ---------------------------------------------------------------------------


def test_q6_day_type_revenue_matches_ground_truth(
    answers: dict[str, AnalysisResponse],
) -> None:
    """Weekend and weekday revenue both match."""
    response = answer_for(answers, "q6")
    day_types = GOLDEN["q6"]["last_3_months"]["day_types"]
    full_year = GOLDEN["q6"]["full_year"]["day_types"]

    for entry in day_types + full_year:
        values = all_numbers(response)
        if any(
            abs(value - entry["revenue_net_inr"]) <= REVENUE_TOLERANCE_INR
            for value in values
        ):
            return
    pytest.fail(
        "neither the last-3-month nor the full-year weekend/weekday revenue "
        "appears in the results"
    )


def test_q6_normalises_per_trading_day(
    answers: dict[str, AnalysisResponse],
) -> None:
    """A raw weekday total is 2.5x a weekend total and means nothing.

    The year holds about 2.5 times more weekdays than weekend days, so the
    comparison is only meaningful per trading day. This asserts the answer
    actually made that adjustment rather than reporting the larger number.
    """
    response = answer_for(answers, "q6")
    text = insight_text(response)

    per_day_language = any(
        phrase in text
        for phrase in ("per day", "per-day", "daily", "per trading day", "each day")
    )
    per_day_figures = [
        entry["avg_revenue_per_day_inr"]
        for entry in GOLDEN["q6"]["full_year"]["day_types"]
        + GOLDEN["q6"]["last_3_months"]["day_types"]
    ]
    values = all_numbers(response)
    computed_per_day = any(
        any(abs(value - target) <= REVENUE_TOLERANCE_INR for value in values)
        for target in per_day_figures
    )

    assert per_day_language or computed_per_day, (
        "q6 compares weekend against weekday without normalising per trading "
        "day, which is the one mistake this question exists to catch"
    )


# ---------------------------------------------------------------------------
# Q7 - festive uplift
# ---------------------------------------------------------------------------


def test_q7_festive_uplift_matches_ground_truth(
    answers: dict[str, AnalysisResponse],
) -> None:
    """Festive and normal trading match, measured per trading day."""
    response = answer_for(answers, "q7")
    normal = GOLDEN["q7"]["normal"]
    festive = GOLDEN["q7"]["all_festive_combined"]
    values = all_numbers(response)

    def matches(target: float) -> bool:
        """Whether a figure appears in the results.

        Args:
            target: The expected figure.

        Returns:
            True when some returned value matches within tolerance.
        """
        return any(abs(value - target) <= REVENUE_TOLERANCE_INR for value in values)

    # Per-day is the assertion that matters, and it is deliberately not
    # paired with a raw-total requirement: an answer that computes only the
    # per-day figures has done the harder and more correct thing, and demanding
    # the totals as well would fail it for being right.
    per_day_targets = [
        normal["avg_revenue_per_day_inr"],
        festive["avg_revenue_per_day_inr"],
        *[
            period["avg_revenue_per_day_inr"]
            for period in GOLDEN["q7"]["festive_periods"]
        ],
    ]
    assert matches(normal["avg_revenue_per_day_inr"]), (
        "q7 does not report normal trading per day, so the uplift cannot be "
        "measured against anything: the festive windows cover 29 days against "
        "336 normal days, and raw totals understate the effect"
    )
    assert sum(1 for target in per_day_targets if matches(target)) >= 2, (
        "q7 reports only one side of the comparison"
    )

    # Any figure claimed for total revenue must still be the right one.
    for total in (normal["revenue_net_inr"], festive["revenue_net_inr"]):
        near_misses = [
            value
            for value in values
            if 0 < abs(value - total) <= max(1000.0, total * 0.02)
        ]
        assert not near_misses, (
            f"q7 reports {near_misses[0]:,.2f} where ground truth is "
            f"{total:,.2f}"
        )


# ---------------------------------------------------------------------------
# Q8 - the diagnostic
# ---------------------------------------------------------------------------


def declining_stores() -> list[str]:
    """The stores that fell in every consecutive month, per ground truth.

    Returns:
        Their store ids.
    """
    return [
        store["store_id"]
        for store in GOLDEN["q8"]["stores"]
        if store["declined_every_month"]
    ]


def test_q8_identifies_exactly_the_declining_stores(
    answers: dict[str, AnalysisResponse],
) -> None:
    """The nine qualifying stores are found, and no others are claimed."""
    response = answer_for(answers, "q8")
    expected = set(declining_stores())
    assert len(expected) == 9, "ground truth changed"

    named_in_insight = {
        match.upper()
        for match in re.findall(r"\bST\d{3}\b", insight_text(response).upper())
    }
    returned = {
        str(value)
        for sequence in column_values(response, "store_id")
        + column_values(response, "STORE_ID")
        for value in sequence
    }

    missing = expected - returned
    assert not missing, f"q8 did not return declining stores {sorted(missing)}"

    if named_in_insight:
        invented = named_in_insight - returned
        assert not invented, f"q8 names stores absent from the results: {invented}"


def test_q8_does_not_deny_what_its_own_results_contain(
    answers: dict[str, AnalysisResponse],
) -> None:
    """The narrative must agree with the rows the queries returned.

    A live run answered "no stores have consistently declining revenue" while
    one of its own sub-queries had returned exactly the nine that did. Every
    figure in that answer was correct and every check passed; only the claim
    was wrong, which is the hardest kind of error to notice and the easiest to
    assert.
    """
    response = answer_for(answers, "q8")
    assert response.insight is not None

    returned = {
        str(value)
        for sequence in column_values(response, "store_id")
        + column_values(response, "STORE_ID")
        for value in sequence
    }
    qualifying = set(declining_stores()) & returned
    if not qualifying:
        pytest.skip("no qualifying stores were returned to contradict")

    headline = response.insight.headline.lower()
    denials = ("no store", "none of the store", "no locations", "zero stores")
    assert not any(denial in headline for denial in denials), (
        f"q8 headline denies a decline its results contain "
        f"({len(qualifying)} qualifying stores): {response.insight.headline!r}"
    )


def test_q5_does_not_promote_a_baseline_decline_into_a_monotonic_one(
    answers: dict[str, AnalysisResponse],
) -> None:
    """Falling against a baseline is not "declining over the last 3 months".

    One city is below its February-April baseline while no city fell in every
    consecutive month. Reporting the first as an answer to the second is a
    real distinction, not pedantry: it is the difference between a trend and a
    level, which the analytical rules call out explicitly.
    """
    response = answer_for(answers, "q5")
    assert response.insight is not None
    headline = response.insight.headline.lower()

    cities = [city["city"].lower() for city in GOLDEN["q5"]["cities"]]
    if not any(city in headline for city in cities):
        return

    text = insight_text(response)
    qualifiers = (
        "no city",
        "none",
        "not decline in every",
        "did not decline in every",
        "consecutive",
        "monotonic",
        "against the baseline",
        "versus the baseline",
        "compared to the",
        "prior period",
        "baseline",
    )
    assert any(qualifier in text for qualifier in qualifiers), (
        "q5 names a declining city without saying on what basis; ground truth "
        "is that no city declined in every consecutive month"
    )


def test_q8_states_the_reverter_split_explicitly(
    answers: dict[str, AnalysisResponse],
) -> None:
    """The above/below-baseline split must be named, not gestured at.

    Ground truth: four of the nine consistent decliners finished above their
    own February-April baseline and five below. An answer that says "some of
    these locations actually finished higher" has the right data and gives the
    reader nothing to act on, which is how this regressed once already.
    """
    response = answer_for(answers, "q8")
    assert response.insight is not None
    text = insight_text(response)

    above = {
        store_id
        for store_id in declining_stores()
        if _window_exceeds_baseline(store_id)
    }
    below = set(declining_stores()) - above
    assert len(above) == 4 and len(below) == 5, "ground truth changed"

    named_below = {store for store in below if store.lower() in text}
    assert len(named_below) >= 3, (
        f"q8 names only {sorted(named_below)} of the five stores that are "
        f"genuinely below baseline; the reader cannot tell which stores to act "
        f"on"
    )

    quantified = any(
        phrase in text
        for phrase in ("four of", "4 of", "five of", "5 of", "four stores", "five stores")
    )
    assert quantified, (
        "q8 does not quantify the split between stores that are above their "
        "own baseline and those below it"
    )


def _window_exceeds_baseline(store_id: str) -> bool:
    """Whether a store's window revenue exceeds its prior-period revenue.

    Derived rather than hardcoded: the window total comes from the golden
    answers and the prior-period total is summed from the database, which the
    step-1.5 cross-check proved agrees with the golden path to within 1 INR.

    Args:
        store_id: The store to test.

    Returns:
        True when May-July revenue is above February-April revenue.
    """
    window = next(
        store["window_revenue_inr"]
        for store in GOLDEN["q8"]["stores"]
        if store["store_id"] == store_id
    )
    prior_window = GOLDEN["q8"]["prior_window"]
    connection = sqlite3.connect(f"file:{settings.DB_PATH}?mode=ro", uri=True)
    try:
        baseline = connection.execute(
            "SELECT SUM(revenue_net) FROM mart_store_month "
            "WHERE store_id = ? AND month_key BETWEEN ? AND ?",
            (store_id, prior_window["start"][:7], prior_window["end"][:7]),
        ).fetchone()[0]
    finally:
        connection.close()
    return float(window) > float(baseline)


def test_q8_separates_reverting_stores_from_deteriorating_ones(
    answers: dict[str, AnalysisResponse],
) -> None:
    """The steepest decliner is above its own baseline and must not lead.

    ST039 fell 41% across the window - the largest drop in the estate - while
    running 28% above its own prior quarter. Naming it as the top concern is
    arithmetically correct and analytically wrong, and it is the exact trap
    this question exists to set.
    """
    response = answer_for(answers, "q8")
    assert response.insight is not None

    steepest = min(
        (store for store in GOLDEN["q8"]["stores"] if store["declined_every_month"]),
        key=lambda store: store["change_pct"],
    )
    assert steepest["store_id"] == "ST039", "ground truth changed"

    actions = " ".join(response.insight.recommended_actions).lower()
    if not actions:
        pytest.skip("q8 produced no recommended actions to check")

    first_action = response.insight.recommended_actions[0].lower()
    if "st039" in first_action or "kolkata 39" in first_action:
        assert any(marker in first_action for marker in DEPRIORITISING_MARKERS), (
            "q8 leads its recommendations with ST039, the steepest decliner, "
            "without noting that it is still above its own prior quarter: "
            f"{response.insight.recommended_actions[0]!r}"
        )


def test_q8_baseline_comparison_is_computed_not_inferred(
    answers: dict[str, AnalysisResponse],
) -> None:
    """The above/below split must come from a column, not from reading rows.

    A live run misclassified one store when asked to compare two columns
    across fifty rows. The fix was to have SQL emit the comparison; this
    asserts the fix is still in place.
    """
    response = answer_for(answers, "q8")
    columns = {
        column for result in response.query_results for column in result.columns
    }

    assert any(
        name in columns
        for name in ("is_above_baseline", "delta_pct", "delta_abs")
    ), (
        "q8 has no precomputed baseline comparison column; the answer is "
        f"back to inferring it by eye. Columns seen: {sorted(columns)}"
    )


def test_q8_classification_matches_ground_truth(
    answers: dict[str, AnalysisResponse],
) -> None:
    """Every store the SQL flags as above baseline really is above it."""
    response = answer_for(answers, "q8")

    truth: dict[str, bool] = {}
    for store in GOLDEN["q8"]["stores"]:
        window = store["window_revenue_inr"]
        truth[store["store_id"]] = window

    mistakes: list[str] = []
    for result in response.query_results:
        if "is_above_baseline" not in result.columns:
            continue
        for row in result.rows:
            store_id = str(row.get("store_id", ""))
            window = row.get("window_revenue")
            baseline = row.get("baseline_revenue")
            flag = row.get("is_above_baseline")
            if window is None or baseline is None or flag is None:
                continue
            expected = 1 if float(window) > float(baseline) else 0
            if int(flag) != expected:
                mistakes.append(
                    f"{store_id}: flag={flag} but {window} vs {baseline}"
                )
            if store_id in truth and abs(float(window) - truth[store_id]) > 1.0:
                mistakes.append(
                    f"{store_id}: window revenue {window} does not match "
                    f"ground truth {truth[store_id]}"
                )

    assert not mistakes, f"q8 baseline flags disagree with the data: {mistakes}"


# ---------------------------------------------------------------------------
# Mode reporting
# ---------------------------------------------------------------------------


def test_the_suite_reports_which_mode_it_ran_in() -> None:
    """Make the mode visible, so a green run cannot be misread.

    Replay mode proves the data path; it does not prove the prompts still
    work, because it never asks a model anything. Anyone reading a passing run
    should know which of the two they got.
    """
    mode = "live" if LIVE_MODE else "replay"
    cached = len(CACHED)
    print(f"\ngolden e2e mode: {mode}; cached questions available: {cached}/8")

    assert mode in {"live", "replay"}
    if not LIVE_MODE:
        assert cached or True, "no cache; tests will have skipped"
