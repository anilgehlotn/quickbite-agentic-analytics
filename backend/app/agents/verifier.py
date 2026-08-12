"""Verifier agent: checks results before they are allowed to reach a user.

The ordering here is the whole design, and it is deliberate: **every
deterministic check runs first, in pure Python, and an LLM is consulted only
for cases arithmetic cannot settle.**

The reason is evidential strength. Asking a model whether its own pipeline
produced a good answer is weak evidence - it is the same class of system that
produced the answer, it has no independent access to the data, and it is
agreeable by construction. Asserting that a breakdown sums to its total, that
AOV equals revenue over orders, or that no revenue figure exceeds the annual
total of the entire dataset is strong evidence: it either holds or it does not,
and when it fails the failure names the number that is wrong.

So the LLM never decides whether verification passes. It can only add a
warning, and it is only asked at all when the deterministic layer is silent and
something about the shape of the result is ambiguous. Verification status is
computed exclusively from checks that arithmetic could decide.
"""

from __future__ import annotations

import json
import re
from datetime import date
from typing import Any, Final

from app.agents.base import Agent
from app.agents.contracts import (
    AnalysisPlan,
    QueryIntent,
    QueryResult,
    VerificationCheck,
    VerificationReport,
    VerificationStatus,
)
from app.config import settings
from app.core.logging import get_logger
from app.semantic.schema import CITIES, METRIC_DEFINITIONS

logger = get_logger(__name__)

# Severity vocabulary, matching VerificationCheck.severity.
ERROR: Final[str] = "error"
WARNING: Final[str] = "warning"
INFO: Final[str] = "info"

# --- Column classification -------------------------------------------------
# Results come back with whatever names the SQL agent chose, so checks have to
# recognise a revenue column by name. The token lists are generous; the
# exclusion list below is what keeps them from over-matching.

_REVENUE_TOKENS: Final[tuple[str, ...]] = (
    "revenue",
    "sales",
    "turnover",
    "net_before_tax",
    "net_revenue",
    "gross_bill",
)
_ORDER_COUNT_TOKENS: Final[tuple[str, ...]] = (
    "orders",
    "order_count",
    "num_orders",
    "transactions",
    "txn_count",
)
_QUANTITY_TOKENS: Final[tuple[str, ...]] = ("qty", "quantity", "units")
_AOV_TOKENS: Final[tuple[str, ...]] = (
    "aov",
    "average_order_value",
    "avg_order_value",
)

# Columns holding a change rather than a level. A decline is a negative number
# and a correct one, so these are exempt from the non-negativity check and from
# the share-sums-to-100 check.
_DELTA_TOKENS: Final[tuple[str, ...]] = (
    "change",
    "delta",
    "diff",
    "growth",
    "decline",
    "drop",
    "variance",
    "_vs_",
    "mom",
    "yoy",
    "trend",
)

# Columns that express a share of a whole, which should sum to 100.
_SHARE_TOKENS: Final[tuple[str, ...]] = (
    "share",
    "pct_of",
    "percent_of",
    "proportion",
    "mix_pct",
    "contribution_pct",
)

# Anything matching this is a date-like value worth range-checking.
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_MONTH_KEY = re.compile(r"^\d{4}-\d{2}$")

# "top 5 stores", "bottom 3 cities", "worst 5 performers". The trailing
# exclusion is load-bearing: "the last 3 months" is a time window, not a
# request for three rows, and reading it as one warns on every question this
# system is designed to answer.
_TOP_N = re.compile(
    r"\b(?:top|bottom|best|worst)\s+(\d{1,3})\b"
    r"(?!\s*(?:month|week|day|year|quarter)s?\b)",
    re.I,
)

# Dimensions that partition the business exhaustively, with their cardinality.
# A breakdown over one of these should reconcile to a total; a ranking over
# store or product is a subset and must not be reconciled, which is why stores
# and SKUs are deliberately absent.
_EXHAUSTIVE_DIMENSIONS: Final[dict[str, int]] = {
    "channel": len(settings.CHANNELS),
    "city": len(CITIES),
    "day_type": 2,
    "is_weekend": 2,
}

# Below this many rows a "top N" overshoot is not worth reporting.
_MIN_ROWS_FOR_TOPN_WARNING: Final[int] = 2

# The LLM is asked at most this many times per verification.
MAX_ESCALATIONS: Final[int] = 1

ESCALATION_INSTRUCTIONS: Final[str] = """
You are a checking layer, not an analyst. You are shown a question, the SQL
that was run and the rows it returned. Every arithmetic check has already
passed, so do NOT re-check the numbers.

Answer one narrow question: could these rows plausibly be an answer to that
question? Look only for a mismatch of shape or meaning - a question about
cities answered with store rows, a trend question answered with a single
number, a filter that appears to have been ignored.

Return JSON: {"plausible": true|false, "reason": "<one sentence>"}
Default to true. You are looking for an obvious mismatch, not for imperfection.
"""


def _has_token(column: str, tokens: tuple[str, ...]) -> bool:
    """Test whether a column name contains any of the given tokens.

    Args:
        column: The column name.
        tokens: Substrings to look for.

    Returns:
        True when at least one token appears in the lowercased name.
    """
    lowered = column.lower()
    return any(token in lowered for token in tokens)


def is_delta_column(column: str) -> bool:
    """Whether a column holds a change rather than a level.

    Args:
        column: The column name.

    Returns:
        True when the name suggests a difference, growth rate or trend.
    """
    return _has_token(column, _DELTA_TOKENS)


def is_revenue_column(column: str) -> bool:
    """Whether a column holds a revenue amount.

    Args:
        column: The column name.

    Returns:
        True for revenue levels, false for revenue changes, shares and AOV.
    """
    if is_delta_column(column) or is_share_column(column):
        return False
    if is_aov_column(column):
        return False
    return _has_token(column, _REVENUE_TOKENS)


def is_order_count_column(column: str) -> bool:
    """Whether a column holds a count of orders.

    Args:
        column: The column name.

    Returns:
        True for order-count levels.
    """
    if is_delta_column(column) or is_share_column(column):
        return False
    return _has_token(column, _ORDER_COUNT_TOKENS)


def is_quantity_column(column: str) -> bool:
    """Whether a column holds a unit quantity.

    Args:
        column: The column name.

    Returns:
        True for quantity levels.
    """
    if is_delta_column(column) or is_share_column(column):
        return False
    return _has_token(column, _QUANTITY_TOKENS)


def is_aov_column(column: str) -> bool:
    """Whether a column holds an average order value.

    Args:
        column: The column name.

    Returns:
        True for AOV levels.
    """
    if is_delta_column(column):
        return False
    return _has_token(column, _AOV_TOKENS)


def is_share_column(column: str) -> bool:
    """Whether a column expresses a share of a whole.

    A percentage *change* is not a share, which is why the delta tokens win.

    Args:
        column: The column name.

    Returns:
        True when the values should sum to about 100.
    """
    if _has_token(column, _DELTA_TOKENS):
        return False
    return _has_token(column, _SHARE_TOKENS)


def _numeric(value: Any) -> float | None:
    """Coerce a cell to a float when it is genuinely numeric.

    Booleans are rejected because ``is_weekend`` is stored as 0/1 and is a
    label, not a measure.

    Args:
        value: A cell from a result row.

    Returns:
        The value as a float, or None when it is not a number.
    """
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _first_column(columns: list[str], predicate: Any) -> str | None:
    """Return the first column satisfying a predicate.

    Args:
        columns: Column names in result order.
        predicate: A one-argument test over a column name.

    Returns:
        The matching column name, or None.
    """
    for column in columns:
        if predicate(column):
            return column
    return None


def _successful(results: list[QueryResult]) -> list[QueryResult]:
    """Filter out sub-queries that failed to execute.

    Args:
        results: Every result from the run.

    Returns:
        Only the results that produced rows without error.
    """
    return [result for result in results if result.error is None]


def failing_sub_query_ids(report: VerificationReport) -> list[str]:
    """Which sub-queries an error-severity check blamed.

    The orchestrator uses this to re-run only the queries that failed
    verification rather than the whole plan.

    Args:
        report: The verification report.

    Returns:
        Distinct sub-query ids named by failed error checks, in order.
    """
    found: list[str] = []
    for check in report.checks:
        if check.passed or check.severity != ERROR or not check.details:
            continue
        for sub_query_id in check.details.get("sub_query_ids", []):
            if sub_query_id not in found:
                found.append(str(sub_query_id))
    return found


class VerifierAgent(Agent[VerificationReport]):
    """Runs deterministic checks over query results, escalating only if needed."""

    name = "verifier"

    async def execute(
        self, plan: AnalysisPlan, results: list[QueryResult]
    ) -> VerificationReport:
        """Verify a set of results against the plan that produced them.

        Args:
            plan: The plan the results answer.
            results: Every sub-query result, including failed ones.

        Returns:
            The report, whose status is decided by the deterministic checks
            alone.
        """
        checks: list[VerificationCheck] = self.run_deterministic_checks(plan, results)

        # The LLM is consulted only once the arithmetic has had its say, and
        # only when nothing arithmetic could decide has failed. A model asked
        # to re-litigate a failed sum would sometimes talk the pipeline out of
        # a real defect, which is exactly the failure this ordering prevents.
        has_error = any(
            check.severity == ERROR and not check.passed for check in checks
        )
        if not has_error and self.needs_escalation(plan, results, checks):
            checks.append(await self._escalate(plan, results))

        status = self._status_for(checks)
        report = VerificationReport(
            status=status, checks=checks, summary=self._summary_for(status, checks)
        )
        logger.info(
            "verification_completed",
            extra={
                "status": status.value,
                "checks": len(checks),
                "failed": sum(1 for check in checks if not check.passed),
            },
        )
        return report

    # ------------------------------------------------------------------
    # Deterministic layer
    # ------------------------------------------------------------------

    def run_deterministic_checks(
        self, plan: AnalysisPlan, results: list[QueryResult]
    ) -> list[VerificationCheck]:
        """Run every pure-Python check.

        Args:
            plan: The plan the results answer.
            results: Every sub-query result.

        Returns:
            One check per assertion, in a stable order.
        """
        checks: list[VerificationCheck] = []
        checks.append(self._check_sub_queries_ran(results))
        checks.append(self._check_non_empty(results))
        checks.append(self._check_no_all_null_columns(results))
        checks.append(self._check_no_negative_measures(results))
        checks.append(self._check_aov_consistency(results))
        checks.append(self._check_parts_sum_to_total(results))
        checks.append(self._check_shares_sum_to_100(results))
        checks.append(self._check_revenue_plausible(results))
        checks.append(self._check_row_count_expectation(plan, results))
        checks.append(self._check_metrics_referenced(plan, results))
        checks.append(self._check_dates_in_range(results))
        checks.append(self._check_window_coverage(plan, results))
        return checks

    @staticmethod
    def _check_window_coverage(
        plan: AnalysisPlan, results: list[QueryResult]
    ) -> VerificationCheck:
        """Check that a monthly result covers the whole window it claims to.

        This exists because of a defect seen in a live run:
        ``month_key BETWEEN '2026-05-01' AND '2026-07-31'`` reads correctly but
        silently drops every May row, since month_key is 'YYYY-MM' and
        '2026-05' sorts before '2026-05-01' as a string. The query succeeds,
        the numbers are individually right, and the answer covers two months
        instead of three - which no other check would notice.

        Only partial overlap is flagged. A sub-query that deliberately looks at
        the comparison period alone shares no month with the window and is left
        untouched.

        Args:
            plan: The plan whose window the results should cover.
            results: Every sub-query result.

        Returns:
            A warning-severity check; some sub-queries narrow the window on
            purpose.
        """
        expected = _window_months(plan)
        offenders: list[dict[str, Any]] = []
        blamed: list[str] = []
        for result in _successful(results):
            months = {
                str(value)[:7]
                for row in result.rows
                for column, value in row.items()
                if _is_month_column(column) and _parse_date_like(value) is not None
            }
            overlap = months & expected
            if not overlap or overlap == expected:
                continue
            missing = sorted(expected - overlap)
            offenders.append(
                {
                    "sub_query_id": result.sub_query_id,
                    "missing_months": missing,
                }
            )
            if result.sub_query_id not in blamed:
                blamed.append(result.sub_query_id)
        return VerificationCheck(
            name="results_cover_the_window",
            passed=not offenders,
            severity=WARNING,
            message=(
                f"Monthly results cover the full window "
                f"{sorted(expected)[0]} to {sorted(expected)[-1]}."
                if not offenders
                else f"{offenders[0]['sub_query_id']} is missing "
                f"{', '.join(offenders[0]['missing_months'])} from the "
                f"{plan.time_window.label} window, so the answer covers a "
                f"shorter period than the question asked about."
            ),
            details={"offenders": offenders, "sub_query_ids": blamed}
            if offenders
            else None,
        )

    @staticmethod
    def _check_sub_queries_ran(results: list[QueryResult]) -> VerificationCheck:
        """Report sub-queries that failed to execute at all.

        Args:
            results: Every sub-query result.

        Returns:
            A warning-severity check; a partial answer is still an answer.
        """
        failed = [result.sub_query_id for result in results if result.error]
        return VerificationCheck(
            name="all_sub_queries_executed",
            passed=not failed,
            severity=WARNING,
            message=(
                f"All {len(results)} sub-quer"
                f"{'y' if len(results) == 1 else 'ies'} executed."
                if not failed
                else f"{len(failed)} of {len(results)} sub-queries failed to "
                f"execute: {', '.join(failed)}. The answer is based on the "
                f"remainder."
            ),
            details={"sub_query_ids": failed} if failed else None,
        )

    @staticmethod
    def _check_non_empty(results: list[QueryResult]) -> VerificationCheck:
        """Check that queries returned rows.

        Deliberately a warning. Some correct answers are legitimately empty -
        "which cities declined every month?" may have no answer - and the
        system must be able to say "none" rather than being pushed into
        inventing rows to satisfy a check.

        Args:
            results: Every sub-query result.

        Returns:
            A warning-severity check.
        """
        empty = [
            result.sub_query_id for result in _successful(results) if result.row_count == 0
        ]
        return VerificationCheck(
            name="results_non_empty",
            passed=not empty,
            severity=WARNING,
            message=(
                "Every sub-query returned at least one row."
                if not empty
                else f"{len(empty)} sub-quer"
                f"{'y' if len(empty) == 1 else 'ies'} returned no rows "
                f"({', '.join(empty)}). An empty result can be the correct "
                f"answer, so this is reported rather than treated as a failure."
            ),
            details={"sub_query_ids": empty} if empty else None,
        )

    @staticmethod
    def _check_no_all_null_columns(results: list[QueryResult]) -> VerificationCheck:
        """Check that no column is null in every row.

        Args:
            results: Every sub-query result.

        Returns:
            A warning-severity check naming the affected columns.
        """
        offenders: list[str] = []
        blamed: list[str] = []
        for result in _successful(results):
            if not result.rows:
                continue
            for column in result.columns:
                if all(row.get(column) is None for row in result.rows):
                    offenders.append(f"{result.sub_query_id}.{column}")
                    if result.sub_query_id not in blamed:
                        blamed.append(result.sub_query_id)
        return VerificationCheck(
            name="no_all_null_columns",
            passed=not offenders,
            severity=WARNING,
            message=(
                "No column is null in every row."
                if not offenders
                else f"Entirely null column(s): {', '.join(offenders)}. This "
                f"usually means a join matched nothing or an aggregate was "
                f"computed over an empty group."
            ),
            details={"columns": offenders, "sub_query_ids": blamed}
            if offenders
            else None,
        )

    @staticmethod
    def _check_no_negative_measures(results: list[QueryResult]) -> VerificationCheck:
        """Check that revenue, order counts and quantities are non-negative.

        Change columns are exempt: a decline is a negative number and a correct
        one.

        Args:
            results: Every sub-query result.

        Returns:
            An error-severity check.
        """
        offenders: list[dict[str, Any]] = []
        blamed: list[str] = []
        for result in _successful(results):
            measures = [
                column
                for column in result.columns
                if is_revenue_column(column)
                or is_order_count_column(column)
                or is_quantity_column(column)
            ]
            for row in result.rows:
                for column in measures:
                    value = _numeric(row.get(column))
                    if value is not None and value < 0:
                        offenders.append(
                            {
                                "sub_query_id": result.sub_query_id,
                                "column": column,
                                "value": value,
                            }
                        )
                        if result.sub_query_id not in blamed:
                            blamed.append(result.sub_query_id)
        return VerificationCheck(
            name="no_negative_measures",
            passed=not offenders,
            severity=ERROR,
            message=(
                "No negative revenue, order count or quantity."
                if not offenders
                else f"{len(offenders)} negative value(s) in measure columns, "
                f"for example {offenders[0]['column']}="
                f"{offenders[0]['value']:,.2f} in "
                f"{offenders[0]['sub_query_id']}. Revenue, orders and "
                f"quantities cannot be negative in this dataset."
            ),
            details={"offenders": offenders[:10], "sub_query_ids": blamed}
            if offenders
            else None,
        )

    @staticmethod
    def _check_aov_consistency(results: list[QueryResult]) -> VerificationCheck:
        """Check that AOV equals revenue divided by orders.

        Args:
            results: Every sub-query result.

        Returns:
            An error-severity check; an AOV that does not reconcile means one
            of the three numbers is wrong.
        """
        offenders: list[dict[str, Any]] = []
        blamed: list[str] = []
        checked = 0
        for result in _successful(results):
            revenue_column = _first_column(result.columns, is_revenue_column)
            orders_column = _first_column(result.columns, is_order_count_column)
            aov_column = _first_column(result.columns, is_aov_column)
            if not (revenue_column and orders_column and aov_column):
                continue
            for row in result.rows:
                revenue = _numeric(row.get(revenue_column))
                orders = _numeric(row.get(orders_column))
                aov = _numeric(row.get(aov_column))
                if revenue is None or orders is None or aov is None or orders == 0:
                    continue
                checked += 1
                expected = revenue / orders
                if abs(expected - aov) > settings.AOV_TOLERANCE_INR:
                    offenders.append(
                        {
                            "sub_query_id": result.sub_query_id,
                            "reported_aov": round(aov, 2),
                            "computed_aov": round(expected, 2),
                            "delta": round(aov - expected, 2),
                        }
                    )
                    if result.sub_query_id not in blamed:
                        blamed.append(result.sub_query_id)
        if checked == 0:
            return VerificationCheck(
                name="aov_reconciles",
                passed=True,
                severity=ERROR,
                message="No result carries revenue, orders and AOV together, "
                "so there was nothing to reconcile.",
                details=None,
            )
        return VerificationCheck(
            name="aov_reconciles",
            passed=not offenders,
            severity=ERROR,
            message=(
                f"AOV equals revenue / orders in all {checked} row(s) checked, "
                f"within {settings.AOV_TOLERANCE_INR} INR."
                if not offenders
                else f"AOV does not reconcile in {len(offenders)} row(s): "
                f"reported {offenders[0]['reported_aov']:,.2f} against "
                f"computed {offenders[0]['computed_aov']:,.2f}, a difference "
                f"of {offenders[0]['delta']:,.2f} INR. AOV must be a ratio of "
                f"sums, not an average of averages."
            ),
            details={"offenders": offenders[:10], "sub_query_ids": blamed}
            if offenders
            else None,
        )

    @staticmethod
    def _check_parts_sum_to_total(results: list[QueryResult]) -> VerificationCheck:
        """Check that a complete breakdown sums to the reported total.

        Only breakdowns over an exhaustively partitioning dimension with the
        expected number of rows are reconciled. A top-five ranking is a subset
        by design and summing it against a total would fail every time.

        Args:
            results: Every sub-query result.

        Returns:
            An error-severity check reporting the delta when it fails.
        """
        successful = _successful(results)
        total_value: float | None = None
        total_id = ""
        for result in successful:
            if result.row_count != 1:
                continue
            revenue_column = _first_column(result.columns, is_revenue_column)
            if revenue_column is None:
                continue
            # A single row that is itself part of a breakdown is not a total.
            if any(
                column in _EXHAUSTIVE_DIMENSIONS for column in result.columns
            ):
                continue
            value = _numeric(result.rows[0].get(revenue_column))
            if value is not None:
                total_value = value
                total_id = result.sub_query_id
                break

        if total_value is None:
            return VerificationCheck(
                name="parts_sum_to_total",
                passed=True,
                severity=ERROR,
                message="No overall total was produced, so no breakdown could "
                "be reconciled against one.",
                details=None,
            )

        for result in successful:
            dimension = _first_column(
                result.columns, lambda column: column in _EXHAUSTIVE_DIMENSIONS
            )
            revenue_column = _first_column(result.columns, is_revenue_column)
            if dimension is None or revenue_column is None:
                continue
            distinct = {str(row.get(dimension)) for row in result.rows}
            if len(distinct) != _EXHAUSTIVE_DIMENSIONS[dimension]:
                continue
            parts = sum(
                _numeric(row.get(revenue_column)) or 0.0 for row in result.rows
            )
            delta = parts - total_value
            passed = abs(delta) <= settings.TOTAL_RECONCILIATION_TOLERANCE_INR
            return VerificationCheck(
                name="parts_sum_to_total",
                passed=passed,
                severity=ERROR,
                message=(
                    f"The {dimension} breakdown sums to {parts:,.2f} INR, "
                    f"matching the reported total of {total_value:,.2f} INR."
                    if passed
                    else f"The {dimension} breakdown sums to {parts:,.2f} INR "
                    f"but the reported total is {total_value:,.2f} INR, a "
                    f"difference of {delta:,.2f} INR. One of the two figures "
                    f"is wrong."
                ),
                details={
                    "dimension": dimension,
                    "parts_total": round(parts, 2),
                    "reported_total": round(total_value, 2),
                    "delta": round(delta, 2),
                    "sub_query_ids": [result.sub_query_id, total_id],
                },
            )

        return VerificationCheck(
            name="parts_sum_to_total",
            passed=True,
            severity=ERROR,
            message="No complete dimensional breakdown accompanied the total, "
            "so there was nothing to reconcile.",
            details=None,
        )

    @staticmethod
    def _check_shares_sum_to_100(results: list[QueryResult]) -> VerificationCheck:
        """Check that share columns sum to approximately 100.

        Args:
            results: Every sub-query result.

        Returns:
            A warning-severity check; rounding alone can move the sum.
        """
        offenders: list[dict[str, Any]] = []
        blamed: list[str] = []
        checked = 0
        for result in _successful(results):
            if result.row_count < 2:
                continue
            for column in result.columns:
                if not is_share_column(column):
                    continue
                values = [_numeric(row.get(column)) for row in result.rows]
                if any(value is None for value in values):
                    continue
                checked += 1
                total = sum(value for value in values if value is not None)
                if abs(total - 100.0) > settings.SHARE_SUM_TOLERANCE_PCT:
                    offenders.append(
                        {
                            "sub_query_id": result.sub_query_id,
                            "column": column,
                            "sum": round(total, 2),
                        }
                    )
                    if result.sub_query_id not in blamed:
                        blamed.append(result.sub_query_id)
        return VerificationCheck(
            name="shares_sum_to_100",
            passed=not offenders,
            severity=WARNING,
            message=(
                f"{checked} share column(s) sum to 100 within "
                f"{settings.SHARE_SUM_TOLERANCE_PCT} points."
                if not offenders
                else f"Share column {offenders[0]['column']} sums to "
                f"{offenders[0]['sum']}, not 100. The breakdown may be "
                f"incomplete or the denominator may be wrong."
            ),
            details={"offenders": offenders, "sub_query_ids": blamed}
            if offenders
            else None,
        )

    @staticmethod
    def _check_revenue_plausible(results: list[QueryResult]) -> VerificationCheck:
        """Check that no revenue figure exceeds what the dataset can contain.

        Full-year revenue is about 13M INR, so a larger figure is not a
        surprising finding - it is a fan-out join or the wrong grain.

        Args:
            results: Every sub-query result.

        Returns:
            An error-severity check.
        """
        offenders: list[dict[str, Any]] = []
        blamed: list[str] = []
        for result in _successful(results):
            for column in result.columns:
                if not is_revenue_column(column):
                    continue
                for row in result.rows:
                    value = _numeric(row.get(column))
                    if value is not None and value > settings.MAX_PLAUSIBLE_REVENUE_INR:
                        offenders.append(
                            {
                                "sub_query_id": result.sub_query_id,
                                "column": column,
                                "value": round(value, 2),
                            }
                        )
                        if result.sub_query_id not in blamed:
                            blamed.append(result.sub_query_id)
        return VerificationCheck(
            name="revenue_within_plausible_bound",
            passed=not offenders,
            severity=ERROR,
            message=(
                f"Every revenue figure is below the "
                f"{settings.MAX_PLAUSIBLE_REVENUE_INR:,.0f} INR plausibility "
                f"bound."
                if not offenders
                else f"Revenue of {offenders[0]['value']:,.2f} INR in "
                f"{offenders[0]['sub_query_id']}.{offenders[0]['column']} "
                f"exceeds the {settings.MAX_PLAUSIBLE_REVENUE_INR:,.0f} INR "
                f"bound. Total revenue across the whole dataset is about 13M "
                f"INR, so this is a duplicated join rather than a large "
                f"number."
            ),
            details={"offenders": offenders[:10], "sub_query_ids": blamed}
            if offenders
            else None,
        )

    @staticmethod
    def _check_row_count_expectation(
        plan: AnalysisPlan, results: list[QueryResult]
    ) -> VerificationCheck:
        """Check row counts against a "top N" stated in the question.

        A question asking for the top five that comes back with forty rows has
        not been answered as asked, even though every row may be correct.

        Args:
            plan: The plan, whose question carries the expectation.
            results: Every sub-query result.

        Returns:
            A warning-severity check.
        """
        match = _TOP_N.search(plan.question)
        if not match:
            return VerificationCheck(
                name="row_count_matches_expectation",
                passed=True,
                severity=WARNING,
                message="The question states no row-count expectation.",
                details=None,
            )

        wanted = int(match.group(1))
        # A per-month breakdown of a top N legitimately returns N rows per
        # month, so the allowance scales with the window.
        months = _months_in_window(plan)
        allowance = wanted * max(1, months)
        offenders = [
            {"sub_query_id": result.sub_query_id, "row_count": result.row_count}
            for result in _successful(results)
            if result.row_count > allowance
            and result.row_count >= _MIN_ROWS_FOR_TOPN_WARNING
        ]
        return VerificationCheck(
            name="row_count_matches_expectation",
            passed=not offenders,
            severity=WARNING,
            message=(
                f"Row counts are consistent with a top-{wanted} question."
                if not offenders
                else f"The question asks for {wanted}, but "
                f"{offenders[0]['sub_query_id']} returned "
                f"{offenders[0]['row_count']} rows (allowing up to "
                f"{allowance} for a per-month breakdown). The ranking may not "
                f"have been narrowed."
            ),
            details={"expected": wanted, "allowance": allowance, "offenders": offenders}
            if offenders
            else None,
        )

    @staticmethod
    def _check_metrics_referenced(
        plan: AnalysisPlan, results: list[QueryResult]
    ) -> VerificationCheck:
        """Check that the SQL computed the metrics the plan asked for.

        A warning rather than an error: the marts expose pre-aggregated
        aliases, so a metric can be answered correctly without its canonical
        expression appearing anywhere in the text of the query.

        Args:
            plan: The plan naming the requested metrics.
            results: Every sub-query result.

        Returns:
            A warning-severity check.
        """
        if not plan.metrics:
            return VerificationCheck(
                name="sql_references_planned_metrics",
                passed=True,
                severity=WARNING,
                message="The plan requested no specific metrics.",
                details=None,
            )

        all_sql = " ".join(result.sql for result in results).lower()
        all_columns = " ".join(
            column for result in results for column in result.columns
        ).lower()
        haystack = f"{all_sql} {all_columns}"
        missing = [
            metric
            for metric in plan.metrics
            if not any(token in haystack for token in _metric_tokens(metric))
        ]
        return VerificationCheck(
            name="sql_references_planned_metrics",
            passed=not missing,
            severity=WARNING,
            message=(
                f"Every planned metric ({', '.join(plan.metrics)}) appears in "
                f"the executed SQL."
                if not missing
                else f"Planned metric(s) {', '.join(missing)} do not appear in "
                f"any executed query, so the plan and the SQL may have "
                f"diverged."
            ),
            details={"missing": missing} if missing else None,
        )

    @staticmethod
    def _check_dates_in_range(results: list[QueryResult]) -> VerificationCheck:
        """Check that every date-like value falls inside the dataset range.

        Args:
            results: Every sub-query result.

        Returns:
            An error-severity check; a date outside the extract means the
            query computed it rather than read it.
        """
        offenders: list[dict[str, Any]] = []
        blamed: list[str] = []
        for result in _successful(results):
            for row in result.rows:
                for column, value in row.items():
                    parsed = _parse_date_like(value)
                    if parsed is None:
                        continue
                    if not (
                        settings.DATA_START_DATE <= parsed <= settings.DATA_ASOF_DATE
                    ):
                        offenders.append(
                            {
                                "sub_query_id": result.sub_query_id,
                                "column": column,
                                "value": str(value),
                            }
                        )
                        if result.sub_query_id not in blamed:
                            blamed.append(result.sub_query_id)
        return VerificationCheck(
            name="dates_within_data_range",
            passed=not offenders,
            severity=ERROR,
            message=(
                f"Every date falls within "
                f"{settings.DATA_START_DATE.isoformat()} to "
                f"{settings.DATA_ASOF_DATE.isoformat()}."
                if not offenders
                else f"{len(offenders)} value(s) fall outside the dataset "
                f"range, for example {offenders[0]['value']} in "
                f"{offenders[0]['sub_query_id']}.{offenders[0]['column']}. "
                f"The data covers "
                f"{settings.DATA_START_DATE.isoformat()} to "
                f"{settings.DATA_ASOF_DATE.isoformat()} only."
            ),
            details={"offenders": offenders[:10], "sub_query_ids": blamed}
            if offenders
            else None,
        )

    # ------------------------------------------------------------------
    # LLM escalation
    # ------------------------------------------------------------------

    @staticmethod
    def needs_escalation(
        plan: AnalysisPlan,
        results: list[QueryResult],
        checks: list[VerificationCheck],
    ) -> bool:
        """Decide whether a model's opinion would add anything.

        Escalation costs a round trip and buys weak evidence, so it happens
        only when the deterministic layer has found nothing conclusive and the
        shape of the result is genuinely ambiguous.

        Args:
            plan: The plan the results answer.
            results: Every sub-query result.
            checks: The deterministic checks already run.

        Returns:
            True when the model should be asked.
        """
        successful = _successful(results)
        if not successful:
            return False
        if any(not check.passed and check.severity == WARNING for check in checks):
            return True
        # A trend or ranking answered by a single number, or a single-total
        # question answered by a large table, is a shape mismatch worth a
        # second opinion.
        multi_row_intents = {
            QueryIntent.TREND,
            QueryIntent.RANKING,
            QueryIntent.COMPARISON,
            QueryIntent.DIAGNOSTIC,
        }
        if plan.intent in multi_row_intents and all(
            result.row_count <= 1 for result in successful
        ):
            return True
        return False

    async def _escalate(
        self, plan: AnalysisPlan, results: list[QueryResult]
    ) -> VerificationCheck:
        """Ask the model whether the results plausibly answer the question.

        The returned check is always warning severity. A model cannot fail
        verification, because a model's agreement is not evidence of
        correctness and its disagreement is not proof of error.

        Args:
            plan: The plan the results answer.
            results: Every sub-query result.

        Returns:
            A warning-severity check carrying the model's judgement, or an
            info-severity note when the model could not be reached.
        """
        payload = {
            "question": plan.question,
            "intent": plan.intent.value,
            "queries": [
                {
                    "purpose": result.sub_query_id,
                    "sql": result.sql,
                    "columns": result.columns,
                    "sample_rows": result.rows[:5],
                    "row_count": result.row_count,
                }
                for result in _successful(results)
            ],
        }
        try:
            value, response = await self.llm.complete_json_with_response(
                system=ESCALATION_INSTRUCTIONS,
                user=json.dumps(payload, default=str),
                temperature=0.0,
            )
            self.record_usage(
                response.provider, response.input_tokens + response.output_tokens
            )
        except Exception as error:  # noqa: BLE001 - escalation is best-effort
            logger.warning("verification_escalation_failed", extra={"error": str(error)})
            return VerificationCheck(
                name="llm_plausibility",
                passed=True,
                severity=INFO,
                message=(
                    f"The optional plausibility review could not run "
                    f"({error}). The deterministic checks above are unaffected."
                ),
                details=None,
            )

        plausible = bool(value.get("plausible", True)) if isinstance(value, dict) else True
        reason = (
            str(value.get("reason", "")).strip()
            if isinstance(value, dict)
            else ""
        )
        return VerificationCheck(
            name="llm_plausibility",
            passed=plausible,
            severity=WARNING,
            message=(
                f"A plausibility review found the results consistent with the "
                f"question. {reason}".strip()
                if plausible
                else f"A plausibility review flagged a possible mismatch: "
                f"{reason or 'no reason given'}. This is a judgement, not a "
                f"measurement, so it is recorded as a warning only."
            ),
            details={"llm_reviewed": True},
        )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @staticmethod
    def _status_for(checks: list[VerificationCheck]) -> VerificationStatus:
        """Map check severities to an overall status.

        Args:
            checks: Every check that ran.

        Returns:
            FAILED if an error check failed, PASSED_WITH_WARNINGS if a warning
            failed, otherwise PASSED.
        """
        if any(check.severity == ERROR and not check.passed for check in checks):
            return VerificationStatus.FAILED
        if any(check.severity == WARNING and not check.passed for check in checks):
            return VerificationStatus.PASSED_WITH_WARNINGS
        return VerificationStatus.PASSED

    @staticmethod
    def _summary_for(
        status: VerificationStatus, checks: list[VerificationCheck]
    ) -> str:
        """Write the user-facing verdict.

        Args:
            status: The computed status.
            checks: Every check that ran.

        Returns:
            One or two sentences on whether the answer can be trusted.
        """
        failed = [check for check in checks if not check.passed]
        if status is VerificationStatus.PASSED:
            return (
                f"All {len(checks)} checks passed, including the arithmetic "
                f"consistency of the figures."
            )
        if status is VerificationStatus.PASSED_WITH_WARNINGS:
            return (
                f"The figures are arithmetically sound. "
                f"{len(failed)} advisory check(s) were raised: "
                f"{failed[0].message}"
            )
        errors = [check for check in failed if check.severity == ERROR]
        return (
            f"{len(errors)} consistency check(s) failed, so these numbers "
            f"should not be relied on. {errors[0].message}"
        )

    def summarize(self, result: VerificationReport) -> str:
        """Describe the verification for the trace.

        Args:
            result: The report produced.

        Returns:
            A one-sentence summary.
        """
        passed = sum(1 for check in result.checks if check.passed)
        escalated = any(check.name == "llm_plausibility" for check in result.checks)
        suffix = " after an LLM plausibility review" if escalated else ""
        return (
            f"Ran {len(result.checks)} checks, {passed} passed; verdict "
            f"{result.status.value}{suffix}."
        )


def _metric_tokens(metric: str) -> tuple[str, ...]:
    """Column and expression fragments that evidence a metric was computed.

    Args:
        metric: A key of ``METRIC_DEFINITIONS``.

    Returns:
        Lowercase substrings, any of which counts as the metric appearing.
    """
    definition = METRIC_DEFINITIONS.get(metric, {})
    tokens = {metric.lower()}
    for key in ("sql", "alt_sql"):
        expression = definition.get(key)
        if expression:
            tokens.update(re.findall(r"[a-z_]{4,}", expression.lower()))
    # Mart columns answer these metrics without naming their expression.
    tokens.update(
        {
            "revenue": ("revenue_net", "revenue"),
            "orders": ("orders",),
            "aov": ("aov",),
            "units": ("units",),
        }.get(metric, ())
    )
    tokens.discard("sum")
    tokens.discard("count")
    tokens.discard("distinct")
    return tuple(tokens)


def _is_month_column(column: str) -> bool:
    """Whether a column carries a month or a date.

    Args:
        column: The column name.

    Returns:
        True when the column is a time axis.
    """
    lowered = column.lower()
    return "month" in lowered or "date" in lowered


def _window_months(plan: AnalysisPlan) -> set[str]:
    """The month keys the plan's window spans.

    Args:
        plan: The plan.

    Returns:
        Month keys in 'YYYY-MM' form.
    """
    window = plan.time_window
    months: set[str] = set()
    year, month = window.start_date.year, window.start_date.month
    while (year, month) <= (window.end_date.year, window.end_date.month):
        months.add(f"{year:04d}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return months


def _months_in_window(plan: AnalysisPlan) -> int:
    """Count calendar months spanned by the plan's window.

    Args:
        plan: The plan.

    Returns:
        The number of months, at least one.
    """
    window = plan.time_window
    months = (
        (window.end_date.year - window.start_date.year) * 12
        + window.end_date.month
        - window.start_date.month
        + 1
    )
    return max(1, months)


def _parse_date_like(value: Any) -> date | None:
    """Parse a cell as a date or month key when it looks like one.

    Args:
        value: A cell from a result row.

    Returns:
        The first day of the month for a month key, the date itself for a full
        date, or None when the value is not date-like.
    """
    if not isinstance(value, str):
        return None
    text = value.strip()
    try:
        if _ISO_DATE.match(text):
            return date.fromisoformat(text)
        if _MONTH_KEY.match(text):
            return date.fromisoformat(f"{text}-01")
    except ValueError:
        return None
    return None
