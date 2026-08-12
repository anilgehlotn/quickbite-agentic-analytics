"""Insight agent: turns verified numbers into the answer a business asks for.

Everything upstream of this file establishes that the numbers are right. This
file is where they become useful, and it is the part of the system that
separates an analytics tool from a query engine. "Revenue fell 8%" is a query
result. "Revenue fell 8% because Zomato orders halved at three stores while
Swiggy grew, and the same three stores are still above their own prior quarter"
is an answer.

The prompt therefore encodes analytical judgement rather than formatting rules.
Three of those rules do most of the work:

* **Decompose the change.** Revenue is orders times average order value. Any
  movement in revenue is a movement in one of those two terms, and the two call
  for entirely different responses.
* **A decline is not automatically a deterioration.** A store falling from an
  unusually strong month may still be running above its own historical rate.
  Naming it as the top concern is arithmetically correct and analytically
  wrong, and that mistake is invisible unless a baseline is checked.
* **"None" is a valid answer.** If nothing meets the criterion the honest reply
  is to say so, not to relax the criterion until the list is non-empty.

Two safety properties are enforced in code rather than trusted to the prompt.
Every figure in the narrative is checked against the query results, because a
fabricated number is the worst failure this system can produce; and if the
model is unavailable the agent still returns an :class:`Insight` built
deterministically from the rows, so the user always gets correct numbers even
when the narrative layer is gone.
"""

from __future__ import annotations

import json
import re
from bisect import bisect_left
from dataclasses import dataclass, field
from typing import Any, Final

from pydantic import BaseModel, ValidationError

from app.agents.base import Agent
from app.agents.contracts import (
    AnalysisPlan,
    ChartSpec,
    ChartType,
    Insight,
    QueryResult,
    VerificationReport,
    VerificationStatus,
)
from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Rows sent to the model per sub-query. Enough for it to see the shape and the
# extremes of a ranking; beyond this the prompt grows faster than the insight
# improves.
MAX_ROWS_IN_PROMPT: Final[int] = 60

# Entity labels listed per flag group. Enough to name every store in the
# estate; beyond this the group is a population, not a list.
MAX_FLAG_LABELS: Final[int] = 50

# Confidence assigned to a deterministically built insight. The numbers are
# exact but nothing has interpreted them, so it must not look authoritative.
DEGRADED_CONFIDENCE: Final[float] = 0.3

# Confidence ceiling when verification did not pass cleanly.
UNVERIFIED_CONFIDENCE_CEILING: Final[float] = 0.4

# Confidence penalty applied when a figure in the narrative cannot be traced
# back to the query results. Not zero, because a flagged figure is more often a
# second-order derivation - a total of two differences, say - than an
# invention; and not severe enough to hide, because the one case it must catch
# is a fabricated headline number.
UNSUPPORTED_NUMBER_PENALTY: Final[float] = 0.7

# Numbers below this are not checked against the results. Counts, ranks and
# month numbers ("3 of the 9 stores") are legitimate derivations that appear
# nowhere in a result row, whereas every fabricated *figure* this check exists
# to catch - a revenue, an order count, an AOV - is far above it.
MIN_CHECKED_MAGNITUDE: Final[float] = 100.0

# Relative tolerance when matching a narrative figure to a result value, which
# absorbs rounding and unit scaling ("3.2M" for 3,197,076.5).
NUMBER_MATCH_TOLERANCE: Final[float] = 0.01

# Multipliers a writer may apply to a raw figure: thousands, lakh, millions,
# crore. Indian-format scaling is included because this is an INR dataset.
_SCALE_FACTORS: Final[tuple[float, ...]] = (
    1.0,
    1_000.0,
    100_000.0,
    1_000_000.0,
    10_000_000.0,
)

# Numeric tokens in prose. Grouped form first so "2,789.50" is one token, and
# neither alternative swallows a trailing comma from the sentence around it.
_NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")

# Dates and month keys, whose digits are not figures to verify.
_DATE_LIKE = re.compile(r"\d{4}-\d{2}(?:-\d{2})?")

# Identifier shape: ST007, SKU005, and nothing else. Two or more digits keeps
# it away from ordinary prose and from figures like "Q3".
_IDENTIFIER = re.compile(r"\b[A-Z]{2,5}\d{2,6}\b")

# Years the dataset spans, written bare in prose ("July 2026"). Only these two
# are exempt, so a genuine figure that happens to be four digits is still
# checked.
_DATA_YEARS: Final[frozenset[str]] = frozenset(
    str(year)
    for year in range(settings.DATA_START_DATE.year, settings.DATA_ASOF_DATE.year + 1)
)

# Column names that carry a time axis.
_TIME_COLUMNS: Final[tuple[str, ...]] = (
    "month_key",
    "month",
    "order_date",
    "date",
    "day",
    "week",
    "period",
)

# Column names that are labels rather than measures.
_LABEL_HINTS: Final[tuple[str, ...]] = (
    "id",
    "name",
    "city",
    "channel",
    "region",
    "store",
    "product",
    "sku",
    "category",
    "segment",
    "format",
    "type",
    "period",
    "day",
)

ANALYTICAL_RULES: Final[str] = """
HOW TO ANALYSE

1. LEAD WITH THE ANSWER. The headline is one sentence that answers exactly what
   was asked, with the number in it. Never open with methodology, never open
   with "based on the data" or "the analysis shows".

2. NEVER INVENT A NUMBER. Every figure you write must appear in the query
   results given to you. If a figure is not there, say it is not available.
   A plausible fabricated number is the worst possible failure of this system,
   worse than an incomplete answer, because the reader cannot tell.

3. DECOMPOSE EVERY CHANGE. Revenue = orders x average order value. When a
   metric moved, say which of the two terms moved. Fewer customers and smaller
   baskets are different problems with different fixes, and "revenue fell" says
   neither.

4. LEVEL IS NOT TREND. A store can be small and improving, or large and
   deteriorating. These call for opposite actions. Say which one you are
   describing.

5. DO NOT COMPARE ONLY THE ENDPOINTS. When you have the intermediate months,
   check whether the change was monotonic. A metric that fell, recovered, then
   fell again is a different situation from a steady slide, and a single
   start-to-end percentage hides the difference.

6. ASK WHETHER A DECLINE IS A RETURN TO NORMAL. Before calling anything a
   deterioration, compare it against its own prior-period baseline if one is in
   the results. A store falling sharply from an unusually strong month may
   still be running above its own historical rate. Naming that store as the top
   concern is arithmetically correct and analytically wrong. If a baseline is
   present, you MUST separate genuine decliners from reverters.

   READ THE COMPARISON, DO NOT REDO IT. When a result already carries the
   comparison as a column - is_above_baseline, delta_abs, delta_pct, or any
   precomputed flag - that column IS the answer. Take it literally, row by
   row. Never re-derive it by eyeballing two other columns, never override it
   with your own impression, and never generalise it into "all of them" or
   "none of them" without checking every row. Your headline must agree with
   that column, and with your own narrative.

   COUNT BOTH SIDES AND NAME THEM. When a result carries both a qualifying
   flag and a baseline flag, report the split as numbers and names: "four of
   the nine are still above their own prior quarter (A, B, C, D); the other
   five are genuinely down (E, F, G, H, I)". "Several" and "some" are not
   answers - the reader has to know which stores to act on and which to leave
   alone, and that is the entire point of the comparison.

   Where a result carries a "flag_summary", that grouping was computed from
   the rows in code and is authoritative. Use its membership lists exactly as
   given: every name in the "1" group is above baseline, every name in the "0"
   group is below it. Do not move a name between groups, and make sure any
   count you state matches the size of the list you were handed.

7. SEPARATE SPECIFIC FROM MARKET-WIDE. If the surrounding city or the business
   as a whole moved the same way, the cause is not that store. Say so.

8. "NONE" IS AN ANSWER. If no entity meets the stated criterion, say plainly
   that none does. Do not loosen the criterion to produce a non-empty list, and
   do not name the closest case as though it qualified.

   Specifically: when a result carries a qualifying flag column and NO row has
   the flag set, your FIRST SENTENCE must state that none met the criterion,
   naming the criterion. That is a complete and correct answer; an empty set
   is a finding, not a failure to find something. You may then add a weaker
   comparison - a prior-period baseline, an endpoint change - but only AFTER
   that sentence and only labelled as a different test, in the form "no city
   declined in every consecutive month; on the weaker test of the full window
   against the prior quarter, one city is down".
   Never lead with the weaker test. Never let it stand in as the answer to the
   question that was actually asked.

11. IDENTIFIERS ARE COPIED, NEVER RETYPED. Store ids, SKU ids, store names and
    every other identifier must be reproduced exactly as they appear in the
    query results - ST007 is not ST07, SKU005 is not SKU5. Copy the characters
    across; do not normalise, abbreviate, pad or shorten them. A mistyped
    identifier sends someone to the wrong store.

9. STATE CAVEATS. Seasonality, small samples, known data limitations, and the
   fact that revenue is tax-exclusive. A careful analyst states the limits of
   their own answer unprompted.

10. RECOMMEND FROM THE FINDING. Actions must follow from this specific result.
    "Investigate why Zomato orders fell at ST014 while Swiggy rose there" is
    useful. "Improve marketing" is not an insight, it is filler.
"""


@dataclass
class InsightBundle:
    """The insight agent's output: the narrative and how to chart it.

    Attributes:
        insight: The business explanation.
        chart: How to visualise the primary result.
        degraded: True when the narrative was built in code because the model
            was unavailable.
        unsupported_numbers: Figures in the narrative that could not be traced
            back to any query result.
        unsupported_identifiers: Store, SKU or other ids in the narrative that
            appear in no query result, and are therefore probably mistyped.
    """

    insight: Insight
    chart: ChartSpec
    degraded: bool = False
    unsupported_numbers: list[str] = field(default_factory=list)
    unsupported_identifiers: list[str] = field(default_factory=list)


def schema_without_examples(model: type[BaseModel]) -> dict[str, Any]:
    """Render a model's JSON schema with every worked example removed.

    ``json_schema_extra`` examples are excellent few-shot material for
    *structural* output like a plan, and actively dangerous here. The
    :class:`Insight` example contains findings about this very dataset, and a
    model shown it will reproduce those findings verbatim rather than analyse
    the rows it was given - which was observed in a live run, where the answer
    named stores and figures that no query in that run had returned. The
    example's numbers happened to be right, which made the copy nearly
    undetectable.

    Args:
        model: The contract to render.

    Returns:
        The schema with 'example' and 'examples' keys stripped at every level.
    """

    def strip(node: Any) -> Any:
        """Remove example keys from one node of the schema.

        Args:
            node: A schema fragment.

        Returns:
            The fragment without example keys.
        """
        if isinstance(node, dict):
            return {
                key: strip(value)
                for key, value in node.items()
                if key not in ("example", "examples")
            }
        if isinstance(node, list):
            return [strip(item) for item in node]
        return node

    return strip(model.model_json_schema())


def seasonality_context() -> str:
    """Describe the dataset's annual shape for the prompt.

    Returns:
        A short paragraph naming the peak, the trough and the spread.
    """
    return (
        f"SEASONALITY. Revenue is not flat across the year. It peaks in "
        f"{settings.SEASONAL_PEAK_MONTH} during the festive period "
        f"({', '.join(settings.FESTIVE_PERIODS)}) and troughs in "
        f"{settings.SEASONAL_TROUGH_MONTH}, a spread of about "
        f"{settings.SEASONAL_SPREAD:.2f}x between the strongest and weakest "
        f"month. Any month-to-month comparison must account for this: a fall "
        f"into a seasonally weak month is not by itself evidence of a problem."
    )


def _is_time_column(column: str) -> bool:
    """Whether a column carries a time axis.

    Args:
        column: The column name.

    Returns:
        True when the column looks like a date or month.
    """
    lowered = column.lower()
    return any(token in lowered for token in _TIME_COLUMNS)


def _is_label_column(column: str, rows: list[dict[str, Any]]) -> bool:
    """Whether a column is a label rather than a measure.

    Judged from the values first - anything non-numeric is a label - and from
    the name only when the values are inconclusive.

    Args:
        column: The column name.
        rows: The rows to inspect.

    Returns:
        True when the column identifies rather than measures.
    """
    values = [row.get(column) for row in rows if row.get(column) is not None]
    if values and any(not isinstance(value, (int, float)) for value in values):
        return True
    if values and all(isinstance(value, bool) for value in values):
        return True
    lowered = column.lower()
    return any(lowered == hint or lowered.endswith(f"_{hint}") for hint in _LABEL_HINTS)


def _measure_columns(result: QueryResult) -> list[str]:
    """Columns holding plottable measures.

    Args:
        result: The result to inspect.

    Returns:
        Numeric, non-label column names in result order.
    """
    return [
        column
        for column in result.columns
        if not _is_time_column(column) and not _is_label_column(column, result.rows)
    ]


def choose_chart(plan: AnalysisPlan, results: list[QueryResult]) -> ChartSpec:
    """Pick a visualisation for the primary result, deterministically.

    Used as the fallback when the model declines to choose or names columns
    that do not exist, and as the sole source of a chart on the degraded path.

    Args:
        plan: The plan, for the title.
        results: Every sub-query result.

    Returns:
        A chart spec, possibly of type NONE.
    """
    primary = next(
        (
            result
            for result in results
            if result.error is None and result.row_count > 0
        ),
        None,
    )
    none_chart = ChartSpec(
        chart_type=ChartType.NONE, x_field="", y_fields=[], title=plan.question
    )
    if primary is None:
        return none_chart

    measures = _measure_columns(primary)
    if not measures or primary.row_count <= 1:
        # A single number is already the headline; a chart of one bar adds
        # decoration, not information.
        return none_chart

    time_column = next(
        (column for column in primary.columns if _is_time_column(column)), None
    )
    labels = [
        column
        for column in primary.columns
        if column != time_column and _is_label_column(column, primary.rows)
    ]

    if time_column is not None:
        if labels:
            return ChartSpec(
                chart_type=ChartType.GROUPED_BAR,
                x_field=time_column,
                y_fields=measures[:1],
                title=f"{measures[0]} by {labels[0]} over {plan.time_window.label}",
                series_field=labels[0],
            )
        return ChartSpec(
            chart_type=ChartType.LINE,
            x_field=time_column,
            y_fields=measures[:2],
            title=f"{measures[0]} over {plan.time_window.label}",
        )

    if labels:
        return ChartSpec(
            chart_type=ChartType.BAR,
            x_field=labels[0],
            y_fields=measures[:1],
            title=f"{measures[0]} by {labels[0]}, {plan.time_window.label}",
        )
    return none_chart


def flag_summary(result: QueryResult) -> dict[str, dict[str, list[str]]]:
    """Group entity labels by the value of each boolean flag column.

    The last mile of the baseline problem. Even with ``is_above_baseline``
    computed in SQL and stated as a rule, a live run still put a store with
    ``is_above_baseline = 0`` into the "above baseline" list, because the model
    was reading nine rows and assigning them by hand. Grouping the rows here
    removes the step: what reaches the prompt is the membership itself, not the
    evidence for it.

    A flag column is one whose name reads as a predicate and whose values are
    exactly two-valued booleans or 0/1.

    Args:
        result: The result to summarise.

    Returns:
        A mapping of flag column to value to the entity labels carrying it.
        Empty when the result has no flag column or nothing to label rows by.
    """
    if not result.rows:
        return {}
    label_column = next(
        (
            column
            for column in result.columns
            if _is_label_column(column, result.rows)
        ),
        None,
    )
    if label_column is None:
        return {}

    summary: dict[str, dict[str, list[str]]] = {}
    for column in result.columns:
        lowered = column.lower()
        if not (
            lowered.startswith(("is_", "has_", "was_"))
            or lowered.endswith("_flag")
        ):
            continue
        values = {row.get(column) for row in result.rows}
        if not values <= {0, 1, True, False, None} or len(values - {None}) < 1:
            continue
        # Both groups are always present, even when one is empty. An absent
        # "1" key reads as "no information about which members qualify"; an
        # explicit empty list reads as "none of them do", which is the answer.
        groups: dict[str, list[str]] = {"1": [], "0": []}
        for row in result.rows:
            value = row.get(column)
            key = str(int(value)) if isinstance(value, (int, bool)) else "null"
            groups.setdefault(key, []).append(str(row.get(label_column)))
        summary[column] = {
            key: labels[:MAX_FLAG_LABELS] for key, labels in groups.items()
        }
    return summary


def _result_numbers(results: list[QueryResult]) -> list[float]:
    """Every numeric value present in the results.

    Args:
        results: The results to scan.

    Returns:
        All numeric cell values, booleans excluded.
    """
    numbers: list[float] = []
    for result in results:
        for row in result.rows:
            for value in row.values():
                if isinstance(value, bool) or value is None:
                    continue
                if isinstance(value, (int, float)):
                    numbers.append(float(value))
    return numbers


def _sorted_columns(results: list[QueryResult]) -> list[list[float]]:
    """Numeric values grouped by column and sorted.

    Grouping matters: a period-over-period change is the difference of two
    values from the *same* column, and allowing differences across unrelated
    columns would excuse almost any number.

    Args:
        results: The results to scan.

    Returns:
        One sorted list of values per numeric column.
    """
    columns: list[list[float]] = []
    for result in results:
        for column in result.columns:
            values = [
                float(row[column])
                for row in result.rows
                if isinstance(row.get(column), (int, float))
                and not isinstance(row.get(column), bool)
            ]
            if len(values) > 1:
                columns.append(sorted(values))
    return columns


def _is_supported(
    candidate: float, numbers: list[float], columns: list[list[float]]
) -> bool:
    """Whether a figure in prose traces back to the results.

    Accepts three things. A direct match, allowing rounding. A unit rescaling,
    so 3.2 supports 3,197,076.5 written as "3.2M" and 31.97 supports it written
    as "31.97 lakh". And the difference between two values of one column, which
    is what every "fell by 5,205 INR" in this domain actually is - the writer
    is entitled to subtract, and calling that a fabrication would train a
    reader to ignore the warning.

    Args:
        candidate: The figure written in the narrative.
        numbers: Every numeric value in the results.
        columns: Numeric values grouped by column and sorted.

    Returns:
        True when the results explain the figure.
    """
    for scale in _SCALE_FACTORS:
        scaled = candidate * scale
        for value in numbers:
            tolerance = max(abs(value) * NUMBER_MATCH_TOLERANCE, 0.5)
            if abs(scaled - value) <= tolerance:
                return True
    return _is_column_difference(candidate, columns)


def _is_column_difference(candidate: float, columns: list[list[float]]) -> bool:
    """Whether a figure is the gap between two values of one column.

    Args:
        candidate: The figure written in the narrative.
        columns: Numeric values grouped by column and sorted.

    Returns:
        True when some column contains two values that differ by the figure.
    """
    tolerance = max(abs(candidate) * NUMBER_MATCH_TOLERANCE, 0.5)
    for values in columns:
        for value in values:
            target = value + candidate
            index = bisect_left(values, target - tolerance)
            if index < len(values) and values[index] <= target + tolerance:
                return True
    return False


def find_unsupported_numbers(text: str, results: list[QueryResult]) -> list[str]:
    """Find figures in prose that no query result supports.

    Deliberately narrow, because a check that cries wolf gets ignored. Dates
    and the dataset's own years are skipped, percentages are skipped - a growth
    rate is arithmetic the writer is entitled to do - and so is anything below
    :data:`MIN_CHECKED_MAGNITUDE`, which covers counts and ranks. What remains
    is the class of figure that must have come from the data: revenue, order
    counts and average order values.

    Args:
        text: The narrative to check.
        results: Every sub-query result.

    Returns:
        The unsupported figures as they were written, in order.
    """
    numbers = _result_numbers(results)
    columns = _sorted_columns(results)
    masked = _DATE_LIKE.sub(" ", text)
    unsupported: list[str] = []
    for match in _NUMBER.finditer(masked):
        token = match.group(0)
        trailing = masked[match.end() : match.end() + 1]
        if trailing == "%":
            continue
        if token in _DATA_YEARS:
            continue
        try:
            candidate = float(token.replace(",", ""))
        except ValueError:
            continue
        if candidate < MIN_CHECKED_MAGNITUDE:
            continue
        if (
            not _is_supported(candidate, numbers, columns)
            and token not in unsupported
        ):
            unsupported.append(token)
    return unsupported


def find_unsupported_identifiers(
    text: str, results: list[QueryResult]
) -> list[str]:
    """Find identifier-shaped tokens that no query result contains.

    The same pattern as :func:`find_unsupported_numbers`, for the same reason:
    a figure or an id that is nearly right is worse than one that is obviously
    wrong, because nobody checks it. A live answer rendered ST007 as "ST07" -
    correct in the findings list, wrong in the headline, and invisible to every
    numeric check.

    Matching is deliberately narrow: two to five capital letters followed by
    two or more digits, which covers ST007 and SKU005 without catching ordinary
    words or years.

    Args:
        text: The narrative to check.
        results: Every sub-query result.

    Returns:
        The unsupported identifiers as written, in order of appearance.
    """
    known = {
        str(value)
        for result in results
        for row in result.rows
        for value in row.values()
        if isinstance(value, str)
    }
    unsupported: list[str] = []
    for match in _IDENTIFIER.finditer(text):
        token = match.group(0)
        if token in known or token in unsupported:
            continue
        # An id may legitimately appear inside a longer label, such as a store
        # name that embeds its own code.
        if any(token in value for value in known):
            continue
        unsupported.append(token)
    return unsupported


def _format_value(value: Any) -> str:
    """Render a cell for a deterministically written sentence.

    Args:
        value: The cell value.

    Returns:
        A readable string; large numbers get thousands separators.
    """
    if isinstance(value, bool) or value is None:
        return str(value)
    if isinstance(value, float):
        return f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def build_degraded_insight(
    question: str, plan: AnalysisPlan | None, results: list[QueryResult]
) -> Insight:
    """Build an insight from the rows alone, with no model involved.

    This is the floor of the system's behaviour: when the narrative layer is
    unavailable the user still receives the correct figures, clearly labelled
    as uninterpreted, rather than an error page.

    Args:
        question: The user's question.
        plan: The plan, when one was produced.
        results: Every sub-query result.

    Returns:
        An insight containing only figures taken directly from the results.
    """
    successful = [
        result for result in results if result.error is None and result.rows
    ]
    window = plan.time_window.label if plan else "the requested period"

    if not successful:
        return Insight(
            headline=f"No data could be retrieved for: {question}",
            narrative=(
                "The queries for this question did not return any rows, and "
                "the explanation layer was unavailable, so there is nothing to "
                "report."
            ),
            key_findings=[],
            caveats=["The narrative layer was unavailable for this answer."],
            recommended_actions=[],
            confidence=0.0,
        )

    primary = successful[0]
    first_row = primary.rows[0]
    measures = _measure_columns(primary)
    if measures and primary.row_count == 1:
        # A single row has one obvious primary figure, so state it.
        headline = (
            f"{measures[0].replace('_', ' ')} for {window} is "
            f"{_format_value(first_row.get(measures[0]))}."
        )
    else:
        # With many rows there is no single headline figure, and picking the
        # first row's value would present one store's number as the answer.
        headline = (
            f"{primary.row_count} rows of "
            f"{', '.join(column.replace('_', ' ') for column in primary.columns)} "
            f"were returned for {window}; the first are listed below."
        )

    findings = [
        ", ".join(
            f"{column.replace('_', ' ')} {_format_value(row.get(column))}"
            for column in primary.columns
        )
        for row in primary.rows[:5]
    ]

    return Insight(
        headline=headline,
        narrative=(
            f"These are the figures returned for {window}, presented without "
            f"interpretation: the explanation layer was unavailable, so no "
            f"analysis of drivers, trends or causes has been applied. The "
            f"numbers themselves come straight from the verified query "
            f"results."
        ),
        key_findings=findings,
        caveats=[
            "The narrative layer was unavailable, so these figures have not "
            "been interpreted.",
            f"Revenue figures exclude the {settings.TAX_RATE:.0%} tax.",
        ],
        recommended_actions=[],
        confidence=DEGRADED_CONFIDENCE,
    )


class InsightAgent(Agent[InsightBundle]):
    """Explains verified results the way a senior analyst would."""

    name = "insight"

    def build_system_prompt(self) -> str:
        """Assemble the insight agent's system prompt.

        Returns:
            The analyst persona, the analytical rules, the seasonality context
            and the output schema.
        """
        insight_schema = schema_without_examples(Insight)
        chart_schema = schema_without_examples(ChartSpec)
        return (
            "You are a senior retail analyst presenting to the leadership of a "
            "quick-service restaurant chain. You are not a calculator and not "
            "a report generator: your job is to say what the numbers mean, "
            "what is driving them, and what should be done about it.\n\n"
            f"{ANALYTICAL_RULES}\n\n"
            f"{seasonality_context()}\n\n"
            f"Revenue in this business is tax-exclusive; the underlying data "
            f"applies a {settings.TAX_RATE:.0%} tax on top. All amounts are "
            f"INR. The dataset ends on "
            f"{settings.DATA_ASOF_DATE.isoformat()}.\n\n"
            "OUTPUT\n"
            "Return a single JSON object with two keys, 'insight' and "
            "'chart'.\n"
            f"'insight' must match this schema:\n"
            f"{json.dumps(insight_schema, separators=(',', ':'))}\n"
            f"'chart' must match this schema:\n"
            f"{json.dumps(chart_schema, separators=(',', ':'))}\n"
            "Choose the chart from the data you were given: LINE for a metric "
            "over time, BAR for a ranking or a categorical comparison, "
            "GROUPED_BAR for a dimension across time, and NONE when the answer "
            "is a single number that a chart would not improve. x_field and "
            "y_fields must be column names that actually appear in the query "
            "results.\n"
        )

    def build_user_prompt(
        self,
        plan: AnalysisPlan,
        results: list[QueryResult],
        verification: VerificationReport | None,
    ) -> str:
        """Assemble the evidence the model reasons over.

        Args:
            plan: The plan that was executed.
            results: Every sub-query result.
            verification: The verifier's report, when it ran.

        Returns:
            The user message as compact JSON with a short instruction.
        """
        payload: dict[str, Any] = {
            "question": plan.question,
            "intent": plan.intent.value,
            "time_window": {
                "start": plan.time_window.start_date.isoformat(),
                "end": plan.time_window.end_date.isoformat(),
                "label": plan.time_window.label,
                "comparison_start": (
                    plan.time_window.comparison_start.isoformat()
                    if plan.time_window.comparison_start
                    else None
                ),
                "comparison_end": (
                    plan.time_window.comparison_end.isoformat()
                    if plan.time_window.comparison_end
                    else None
                ),
            },
            "plan_reasoning": plan.reasoning,
            "results": [
                {
                    "sub_query_id": result.sub_query_id,
                    "sql": result.sql,
                    "columns": result.columns,
                    "row_count": result.row_count,
                    "rows": result.rows[:MAX_ROWS_IN_PROMPT],
                    "truncated": result.row_count > MAX_ROWS_IN_PROMPT,
                    "error": result.error,
                    **(
                        {"flag_summary": summary}
                        if (summary := flag_summary(result))
                        else {}
                    ),
                }
                for result in results
            ],
        }
        if verification is not None:
            payload["verification"] = {
                "status": verification.status.value,
                "summary": verification.summary,
                "failed_checks": [
                    {"name": check.name, "message": check.message}
                    for check in verification.checks
                    if not check.passed
                ],
            }
        return (
            f"{json.dumps(payload, default=str)}\n\n"
            f"Answer the question above using only these figures."
        )

    async def execute(
        self,
        plan: AnalysisPlan,
        results: list[QueryResult],
        verification: VerificationReport | None = None,
    ) -> InsightBundle:
        """Explain the results.

        Never raises for a model failure: an unavailable narrative layer
        produces a deterministic insight instead, because correct numbers
        without a story beat an error page.

        Args:
            plan: The plan that was executed.
            results: Every sub-query result.
            verification: The verifier's report, when it ran.

        Returns:
            The insight, the chart, and any figures that could not be traced
            back to the results.
        """
        try:
            payload, response = await self.llm.complete_json_with_response(
                system=self.build_system_prompt(),
                user=self.build_user_prompt(plan, results, verification),
                max_tokens=settings.INSIGHT_MAX_TOKENS,
                temperature=0.0,
            )
            self.record_usage(
                response.provider, response.input_tokens + response.output_tokens
            )
            insight, chart = self._parse(payload, plan, results)
        except Exception as error:  # noqa: BLE001 - degrade, never fail the run
            logger.error(
                "insight_generation_failed",
                extra={"error": str(error), "question": plan.question},
            )
            return InsightBundle(
                insight=build_degraded_insight(plan.question, plan, results),
                chart=choose_chart(plan, results),
                degraded=True,
            )

        checked_text = " ".join(
            [insight.headline, insight.narrative, *insight.key_findings]
        )
        bad_identifiers = find_unsupported_identifiers(checked_text, results)
        if bad_identifiers:
            logger.warning(
                "insight_identifiers_unsupported",
                extra={"identifiers": bad_identifiers, "question": plan.question},
            )
            insight = insight.model_copy(
                update={
                    "caveats": [
                        *insight.caveats,
                        (
                            f"The identifier(s) {', '.join(bad_identifiers)} do "
                            f"not appear in the query results and may be "
                            f"mistyped; check them against the data table."
                        ),
                    ]
                }
            )

        unsupported = find_unsupported_numbers(
            " ".join([insight.headline, *insight.key_findings]), results
        )
        if unsupported:
            logger.warning(
                "insight_numbers_unsupported",
                extra={"figures": unsupported, "question": plan.question},
            )
            insight = insight.model_copy(
                update={
                    "caveats": [
                        *insight.caveats,
                        (
                            f"The figure(s) {', '.join(unsupported)} could not "
                            f"be traced back to the query results and should "
                            f"be treated as unverified."
                        ),
                    ],
                    "confidence": round(
                        insight.confidence * UNSUPPORTED_NUMBER_PENALTY, 2
                    ),
                }
            )

        if verification is not None and verification.status is VerificationStatus.FAILED:
            insight = insight.model_copy(
                update={
                    "caveats": [
                        f"These numbers failed automated consistency checks: "
                        f"{verification.summary}",
                        *insight.caveats,
                    ],
                    "confidence": min(
                        insight.confidence, UNVERIFIED_CONFIDENCE_CEILING
                    ),
                }
            )

        return InsightBundle(
            insight=insight,
            chart=chart,
            unsupported_numbers=unsupported,
            unsupported_identifiers=bad_identifiers,
        )

    def _parse(
        self,
        payload: Any,
        plan: AnalysisPlan,
        results: list[QueryResult],
    ) -> tuple[Insight, ChartSpec]:
        """Validate the model's reply into an insight and a chart.

        Args:
            payload: The decoded model output.
            plan: The plan, for the deterministic chart fallback.
            results: Every sub-query result.

        Returns:
            The insight and the chart.

        Raises:
            ValidationError: If the insight does not satisfy its contract. The
                caller converts this into the degraded path.
        """
        if not isinstance(payload, dict):
            raise ValueError(f"expected a JSON object, got {type(payload).__name__}")

        insight_data = payload.get("insight", payload)
        insight = Insight.model_validate(insight_data)

        chart_data = payload.get("chart")
        if chart_data is None:
            return insight, choose_chart(plan, results)
        try:
            chart = ChartSpec.model_validate(chart_data)
        except ValidationError as error:
            logger.warning("chart_spec_invalid", extra={"error": str(error)})
            return insight, choose_chart(plan, results)

        if not self._chart_is_renderable(chart, results):
            logger.warning(
                "chart_spec_references_missing_columns",
                extra={"x_field": chart.x_field, "y_fields": chart.y_fields},
            )
            return insight, choose_chart(plan, results)
        return insight, chart

    @staticmethod
    def _chart_is_renderable(chart: ChartSpec, results: list[QueryResult]) -> bool:
        """Check that a chart names columns the results actually contain.

        A chart pointing at a column that does not exist renders as an empty
        box, which reads as a broken product rather than a missing chart.

        Args:
            chart: The model's chart choice.
            results: Every sub-query result.

        Returns:
            True when every referenced column exists in some result.
        """
        if chart.chart_type is ChartType.NONE:
            return True
        available = {
            column
            for result in results
            if result.error is None
            for column in result.columns
        }
        referenced = [chart.x_field, *chart.y_fields]
        if chart.series_field:
            referenced.append(chart.series_field)
        return all(column in available for column in referenced if column)

    def summarize(self, result: InsightBundle) -> str:
        """Describe the insight for the trace.

        Args:
            result: The bundle produced.

        Returns:
            A one-sentence summary.
        """
        if result.degraded:
            return (
                "The narrative layer was unavailable, so the figures were "
                "reported directly from the query results."
            )
        chart = (
            "no chart"
            if result.chart.chart_type is ChartType.NONE
            else f"a {result.chart.chart_type.value} chart"
        )
        flagged = (
            f", flagging {len(result.unsupported_numbers)} untraceable figure(s)"
            if result.unsupported_numbers
            else ""
        )
        return (
            f"Explained the result in {len(result.insight.key_findings)} "
            f"findings with {chart}{flagged}."
        )
