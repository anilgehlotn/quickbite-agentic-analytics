"""SQL analyst agent: turns one sub-query into executed rows.

The agent writes SQL, validates it through :class:`SQLGuard`, and executes it
through :class:`SafeQueryExecutor`. When either step fails it feeds the exact
error back to the model and regenerates, which is why both the guard and the
executor put the failing SQL in their error messages: those messages are the
repair instructions.

Two deliberate behaviours:

* **Failure returns, it does not raise.** A sub-query that cannot be answered
  produces a :class:`QueryResult` carrying the error. A partial answer with one
  missing piece is more useful than no answer at all, and the orchestrator - not
  this agent - decides how to degrade.
* **Sub-queries run concurrently.** They are independent by construction, so
  :meth:`SQLAnalystAgent.run_many` gathers them. On a diagnostic question with
  five sub-queries this is the difference between five sequential model calls
  and one round trip's worth of latency.
"""

from __future__ import annotations

import asyncio
import re
import time
from datetime import date
from typing import Final

from app.agents.base import Agent
from app.agents.contracts import AnalysisPlan, QueryResult, SubQuery
from app.core.logging import get_logger
from app.core.sql_guard import (
    QueryExecutionError,
    SafeQueryExecutor,
    SQLGuard,
)
from app.semantic.schema import BUSINESS_RULES, EXAMPLE_QUERIES, get_schema_context

logger = get_logger(__name__)

# Total attempts per sub-query, spanning both validation and execution
# failures. Two gives the model one chance to read its own error and fix it;
# beyond that it tends to loop on the same mistake.
MAX_SQL_ATTEMPTS: Final[int] = 2

# Models wrap SQL in fences even when told not to.
_SQL_FENCE = re.compile(r"^```(?:sql)?\s*\n?(.*?)\n?```$", re.DOTALL | re.IGNORECASE)

SQL_RULES: Final[str] = """
SQL RULES - violating any of these produces a wrong answer

1. DIALECT. SQLite. No window-function syntax SQLite lacks, no full outer
   join, no RIGHT JOIN. Date columns are TEXT in 'YYYY-MM-DD' form and
   compare correctly as strings.

2. NEVER use DATE('now'), CURRENT_DATE, datetime('now') or any function that
   reads the real clock. The dataset ends on {asof}. A query anchored to the
   real date returns zero rows.

3. REVENUE is net_before_tax. Never net_revenue, which includes the {tax:.0%}
   tax, unless the question explicitly asks for a tax-inclusive figure.

4. GRAIN. Use fact_orders for revenue, order counts and AOV. Use
   fact_order_lines ONLY for SKU, product, category and margin questions. The
   two grains do not reconcile exactly, so a revenue total taken from the line
   grain will not match the canonical figure.

5. LEFT JOIN dimension tables, never INNER JOIN. customer_id is NULL on 28% of
   orders and promo_id is NULL on 96%; an inner join silently drops them.

6. MARTS. Prefer mart_store_month, mart_city_month and mart_channel_month for
   monthly trend and ranking questions. They are pre-aggregated and reconcile
   exactly with fact_orders.

7. MONTH_KEY IS 'YYYY-MM', NOT A DATE. Compare it only against 'YYYY-MM'
   literals. month_key BETWEEN '2026-05-01' AND '2026-07-31' looks right and
   silently drops every May row, because '2026-05' sorts before '2026-05-01'
   as a string. Write month_key BETWEEN '2026-05' AND '2026-07'. Use full
   dates only with order_date on fact_orders.

8. BASELINE COMPARISONS. This rule applies ONLY to a sub-query whose purpose
   is the prior-period comparison. It does not apply to a monthly trend
   sub-query, which must still return one row per entity per month: collapsing
   a trend into two period totals destroys the very thing a trend question
   asks about. When this sub-query IS the baseline comparison, do not return
   the two periods as separate rows for someone else to subtract. Return ONE
   ROW PER ENTITY with the comparison already computed, using exactly these
   aliases:
       window_revenue, baseline_revenue, delta_abs, delta_pct,
       is_above_baseline
   where is_above_baseline is
   CASE WHEN window_revenue > baseline_revenue THEN 1 ELSE 0 END.
   Conditional aggregation over month_key gives both periods in one pass:
   SUM(CASE WHEN month_key BETWEEN <window> THEN revenue_net ELSE 0 END).
   A reader comparing two columns across fifty rows will misread one; a
   computed column cannot be misread.

   The baseline is a COLUMN, never a FILTER. If the sub-query identifies
   entities by a pattern over months - "declined every month" - that pattern
   alone decides membership. Do NOT also require the entity to be below its
   baseline, or the answer silently reports a subset and the reverting
   entities disappear from a question that asked about all of them.

9. OUTPUT. Return ONLY the SQL. No explanation, no markdown fences, no
   trailing semicolon. One statement.
"""


# Words that mark a sub-query as the prior-period comparison, and words that
# mark it as the monthly series. Checked against the id and the purpose, which
# the planner writes in business language.
_BASELINE_MARKERS: Final[tuple[str, ...]] = (
    "baseline",
    "prior period",
    "prior-period",
    "prior quarter",
    "previous period",
    "period comparison",
    "versus prior",
    "against prior",
)
_TREND_MARKERS: Final[tuple[str, ...]] = (
    "month",
    "monthly",
    "trend",
    "trajectory",
    "over time",
    "per month",
    "consecutive",
)

# Words that mark a sub-query as the one deciding WHICH entities qualify.
# Checked before the trend markers, because the qualifying query for "declined
# every consecutive month" contains "consecutive" and "month" and would
# otherwise be told to return the raw monthly series - which is exactly the
# work it exists to avoid.
_QUALIFYING_MARKERS: Final[tuple[str, ...]] = (
    "identify",
    "exactly the",
    "exactly those",
    "qualify",
    "qualifying",
    "which stores",
    "which cities",
    "which products",
    "declined every",
    "fell in every",
    "every consecutive",
    "consistently declin",
    "strictly declin",
    "are any",
)

# Words describing a directional move over time. Only meaningful in
# combination with a qualifying marker: a sub-query must be BOTH a membership
# test AND about a decline before it is treated as the monotonic-decline set.
# Requiring both is what keeps this off the monthly trend and baseline
# sub-queries of the same plan, which describe declines too.
_DIRECTIONAL_MARKERS: Final[tuple[str, ...]] = (
    "declin",
    "fell",
    "fallen",
    "falling",
    "drop",
    "decreas",
    "worsen",
    "deteriorat",
    "slid",
    "shrink",
    "shrank",
    "improv",
    "grew",
    "grow",
    "rising",
    "increas",
)

# Column name for the computed membership flag. Chosen to start with "is_" so
# the insight agent's flag grouping picks it up automatically and hands the
# model the membership lists rather than the rows to derive them from.
DECLINE_FLAG_COLUMN: Final[str] = "is_strictly_declining"


def window_months(start: date, end: date) -> list[str]:
    """List the month keys a window spans.

    Args:
        start: First date of the window.
        end: Last date of the window.

    Returns:
        Month keys in 'YYYY-MM' form, in order.
    """
    months: list[str] = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        months.append(f"{year:04d}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return months


def _describes(sub_query: SubQuery, markers: tuple[str, ...]) -> bool:
    """Test a sub-query's id and purpose against a set of markers.

    Args:
        sub_query: The sub-query to classify.
        markers: Lowercase substrings to look for.

    Returns:
        True when any marker appears.
    """
    text = f"{sub_query.id} {sub_query.purpose}".lower()
    return any(marker in text for marker in markers)


def is_baseline_sub_query(sub_query: SubQuery) -> bool:
    """Whether a sub-query is the prior-period comparison.

    Args:
        sub_query: The sub-query to classify.

    Returns:
        True when its purpose is comparing the window against a baseline.
    """
    return _describes(sub_query, _BASELINE_MARKERS)


def is_qualifying_sub_query(sub_query: SubQuery) -> bool:
    """Whether a sub-query decides which entities meet a criterion.

    Args:
        sub_query: The sub-query to classify.

    Returns:
        True when it should return the membership set rather than the data
        from which membership could be derived.
    """
    return _describes(sub_query, _QUALIFYING_MARKERS)


def is_monotonic_decline_sub_query(sub_query: SubQuery) -> bool:
    """Whether a sub-query decides which members moved in one direction.

    A stricter form of :func:`is_qualifying_sub_query`: the sub-query must be
    both a membership test and about a directional move. Both conditions are
    required so that the monthly trend and the baseline comparison of the same
    diagnostic plan - which also talk about declines - keep their own
    instructions.

    Args:
        sub_query: The sub-query to classify.

    Returns:
        True when it should return one row per member carrying a computed
        decline flag.
    """
    return is_qualifying_sub_query(sub_query) and _describes(
        sub_query, _DIRECTIONAL_MARKERS
    )


def is_trend_sub_query(sub_query: SubQuery) -> bool:
    """Whether a sub-query is the monthly series.

    A sub-query that is both is treated as a baseline comparison, because the
    baseline test is checked first at the call site; the trend then belongs in
    its own sub-query, which is what the planner is told to produce.

    Args:
        sub_query: The sub-query to classify.

    Returns:
        True when it should return one row per period.
    """
    return _describes(sub_query, _TREND_MARKERS) or "month_key" in sub_query.dimensions


class SQLAnalystAgent(Agent[QueryResult]):
    """Generates, validates and executes SQL for one sub-query."""

    name = "sql_analyst"

    def __init__(
        self,
        llm: object | None = None,
        guard: SQLGuard | None = None,
        executor: SafeQueryExecutor | None = None,
    ) -> None:
        """Initialise the agent.

        Args:
            llm: LLM client. Defaults to the shared singleton.
            guard: SQL validator. Defaults to a fresh :class:`SQLGuard`.
            executor: Query executor. Defaults to one bound to the configured
                database and sharing this agent's guard.
        """
        super().__init__(llm)  # type: ignore[arg-type]
        self.guard = guard or SQLGuard()
        self.executor = executor or SafeQueryExecutor(guard=self.guard)

    def build_system_prompt(self) -> str:
        """Assemble the analyst's system prompt.

        Returns:
            The schema context, the business rules, worked examples and the
            SQLite dialect rules.
        """
        from app.config import settings

        rules = SQL_RULES.format(
            asof=settings.DATA_ASOF_DATE.isoformat(), tax=settings.TAX_RATE
        )
        numbered_rules = "\n".join(
            f"{index}. {rule}" for index, rule in enumerate(BUSINESS_RULES, start=1)
        )
        examples = "\n\n".join(
            f"Q: {example['question']}\nSQL:\n{example['sql']}"
            for example in EXAMPLE_QUERIES
        )

        return (
            "You are the SQL analyst for a QSR business analytics system. You "
            "write one SQLite SELECT statement that answers exactly the "
            "sub-query you are given.\n\n"
            f"{get_schema_context()}\n"
            f"{rules}\n\n"
            "BUSINESS RULES\n"
            f"{numbered_rules}\n\n"
            "WORKED EXAMPLES\n"
            f"{examples}\n"
        )

    def build_user_prompt(self, sub_query: SubQuery, plan: AnalysisPlan) -> str:
        """Describe one sub-query and the plan context around it.

        Args:
            sub_query: The piece of analysis to write SQL for.
            plan: The plan it belongs to, for the window and overall question.

        Returns:
            The user message.
        """
        window = plan.time_window
        lines = [
            f"Overall question: {plan.question}",
            f"This sub-query: {sub_query.purpose}",
            f"Time window: {window.start_date.isoformat()} to "
            f"{window.end_date.isoformat()} ({window.label})",
        ]
        has_comparison = bool(window.comparison_start and window.comparison_end)
        if has_comparison:
            lines.append(
                f"Comparison window: {window.comparison_start.isoformat()} to "
                f"{window.comparison_end.isoformat()}"
            )

        # Which instruction is appended is decided here, in code, rather than
        # left to the model to infer from the plan. Stating the baseline rule
        # on every sub-query of a diagnostic plan was not a harmless
        # over-instruction: it turned the monthly trend query into a second
        # copy of the baseline query, the monthly series vanished, and the
        # answer concluded that nothing had declined. The order matters too -
        # the query that decides "declined every consecutive month" mentions
        # both "consecutive" and "month", so it must be claimed as the
        # qualifying query before the trend rule can take it.
        if is_monotonic_decline_sub_query(sub_query):
            months = window_months(window.start_date, window.end_date)
            period_columns = [f"revenue_{month.replace('-', '_')}" for month in months]
            comparisons = " AND ".join(
                f"{later} < {earlier}"
                for earlier, later in zip(period_columns, period_columns[1:])
            )
            lines.append(
                "THIS DECIDES WHICH MEMBERS DECLINED. Return ONE ROW PER "
                "MEMBER of the dimension - every member, not only the ones "
                "that qualify - with these columns:\n"
                f"  the dimension column, then {', '.join(period_columns)}, "
                f"one per month in the window;\n"
                f"  {DECLINE_FLAG_COLUMN}, which is 1 when the value fell in "
                f"every consecutive month ({comparisons}) and 0 otherwise;\n"
                "  change_abs and change_pct, last month minus first.\n"
                "Use conditional aggregation over month_key to pivot the "
                "months into columns, then compute the flag from those "
                "columns. Restrict the test to the months listed above and no "
                "others. The point of the flag is that the answer reads it "
                "instead of comparing numbers across rows, so it must be a "
                "real column in the output. If no member qualifies every flag "
                "is 0, which is a complete and correct answer."
            )
            if has_comparison:
                # Both flags on the same row, so "declining AND above its own
                # baseline" is a column pair rather than an intersection of
                # two result sets computed by hand. Without this the reverting
                # members get named as the top concern, which is the specific
                # mistake the baseline exists to prevent.
                lines.append(
                    "ALSO include on the same row: baseline_revenue (the same "
                    "measure summed over the comparison window), delta_abs, "
                    "delta_pct and is_above_baseline. A member can be "
                    "declining every month and still be above its own "
                    "baseline; both facts must be readable from one row."
                )
        elif is_qualifying_sub_query(sub_query):
            months = window_months(window.start_date, window.end_date)
            lines.append(
                "THIS DECIDES WHICH ENTITIES QUALIFY. Return one row per "
                "qualifying entity and nothing else, with the test itself "
                "computed in SQL - self-joins on month_key, or window "
                "functions over the monthly values. Returning every entity's "
                "monthly rows and leaving the test to the reader is the "
                "failure this instruction exists to prevent. If no entity "
                "qualifies, return no rows: that is a valid answer.\n"
                f"Apply the test across EXACTLY these months and no others: "
                f"{', '.join(months)}. The comparison window belongs to the "
                f"baseline sub-query and must not be folded into this test - "
                f"extending the run of months makes the criterion stricter "
                f"than the question asked and silently returns nothing."
            )
        elif has_comparison and is_baseline_sub_query(sub_query):
            lines.append(
                "THIS IS THE BASELINE COMPARISON. Return one row per entity "
                "with window_revenue, baseline_revenue, delta_abs, delta_pct "
                "and is_above_baseline computed in SQL. Do not return "
                "per-month rows."
            )
        elif is_trend_sub_query(sub_query):
            lines.append(
                "THIS IS THE MONTHLY SERIES. Return one row per entity per "
                "month, including month_key. Do NOT collapse the months into "
                "period totals, and do not add baseline comparison columns "
                "here."
            )
        if sub_query.metrics:
            lines.append(f"Metrics: {', '.join(sub_query.metrics)}")
        if sub_query.dimensions:
            lines.append(f"Group by: {', '.join(sub_query.dimensions)}")
        if sub_query.tables:
            lines.append(f"Suggested tables: {', '.join(sub_query.tables)}")
        if sub_query.filters:
            lines.append(f"Filters: {sub_query.filters}")
        lines.append("\nReturn only the SQL.")
        return "\n".join(lines)

    @staticmethod
    def strip_sql_fences(text: str) -> str:
        """Remove markdown fences and a trailing semicolon from model output.

        Args:
            text: The model's reply.

        Returns:
            Bare SQL.
        """
        cleaned = text.strip()
        match = _SQL_FENCE.match(cleaned)
        if match:
            cleaned = match.group(1).strip()
        return cleaned.rstrip(";").strip()

    async def execute(
        self,
        sub_query: SubQuery,
        plan: AnalysisPlan,
        feedback: str | None = None,
    ) -> QueryResult:
        """Generate, validate and run SQL for one sub-query.

        Never raises for a query failure: an unanswerable sub-query returns a
        :class:`QueryResult` with ``error`` populated so the rest of the answer
        can still be assembled.

        Args:
            sub_query: The piece of analysis to run.
            plan: The plan it belongs to.
            feedback: Why a previous attempt at this sub-query was rejected,
                appended to the first prompt. The orchestrator passes the
                verifier's failure here when self-healing, so the model is
                repairing a specific stated problem rather than retrying blind.

        Returns:
            The result, or a result carrying the error after all attempts.
        """
        system = self.build_system_prompt()
        user = self.build_user_prompt(sub_query, plan)

        last_sql = ""
        last_error = "no attempt was made"
        started = time.perf_counter()

        for attempt in range(1, MAX_SQL_ATTEMPTS + 1):
            prompt = user
            if feedback is not None:
                previous = (
                    f"Your previous SQL was:\n{last_sql}\n\n" if last_sql else ""
                )
                prompt = (
                    f"{user}\n\n"
                    f"{previous}"
                    f"A previous attempt at this sub-query was rejected: "
                    f"{feedback}\n\n"
                    f"Return corrected SQL that fixes exactly that problem."
                )

            try:
                response = await self.llm.complete(
                    system=system, user=prompt, temperature=0.0
                )
            except Exception as error:  # noqa: BLE001 - degrade, do not raise
                last_error = f"LLM call failed: {error}"
                break

            self.record_usage(
                response.provider, response.input_tokens + response.output_tokens
            )
            last_sql = self.strip_sql_fences(response.text)

            validation = self.guard.validate(last_sql)
            if not validation.valid:
                feedback = f"SQL validation failed: {validation.error_message}"
                last_error = feedback
                logger.warning(
                    "sql_validation_failed",
                    extra={
                        "sub_query_id": sub_query.id,
                        "attempt": attempt,
                        "errors": validation.errors,
                    },
                )
                continue

            try:
                result = self.executor.execute(validation.sql, validate=False)
            except QueryExecutionError as error:
                feedback = f"SQL execution failed: {error.reason}"
                last_error = feedback
                logger.warning(
                    "sql_execution_failed",
                    extra={
                        "sub_query_id": sub_query.id,
                        "attempt": attempt,
                        "error": error.reason,
                    },
                )
                continue

            logger.info(
                "sub_query_completed",
                extra={
                    "sub_query_id": sub_query.id,
                    "row_count": result.row_count,
                    "attempts": attempt,
                },
            )
            return QueryResult(
                sub_query_id=sub_query.id,
                sql=result.sql,
                columns=result.columns,
                rows=result.rows,
                row_count=result.row_count,
                execution_ms=result.execution_ms,
                error=None,
                attempts=attempt,
            )

        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.error(
            "sub_query_failed",
            extra={"sub_query_id": sub_query.id, "error": last_error},
        )
        return QueryResult(
            sub_query_id=sub_query.id,
            sql=last_sql,
            columns=[],
            rows=[],
            row_count=0,
            execution_ms=elapsed_ms,
            error=last_error,
            attempts=MAX_SQL_ATTEMPTS,
        )

    async def run_many(
        self,
        sub_queries: list[SubQuery],
        plan: AnalysisPlan,
        feedback: str | None = None,
    ) -> list[QueryResult]:
        """Run several sub-queries concurrently.

        Sub-queries are independent by construction, so they are gathered
        rather than awaited in sequence. Results come back in the order the
        sub-queries were given, regardless of which finished first.

        Args:
            sub_queries: The sub-queries to run.
            plan: The plan they belong to.
            feedback: Optional rejection reason applied to every sub-query,
                used by the orchestrator's self-healing retry.

        Returns:
            One result per sub-query, in input order.
        """
        if not sub_queries:
            return []

        results = await asyncio.gather(
            *(self.execute(sub_query, plan, feedback) for sub_query in sub_queries)
        )
        logger.info(
            "sub_queries_completed",
            extra={
                "count": len(results),
                "failed": sum(1 for result in results if result.error),
            },
        )
        return list(results)

    def summarize(self, result: QueryResult) -> str:
        """Describe the query outcome for the trace.

        Args:
            result: The result produced.

        Returns:
            A one-sentence summary.
        """
        if result.error:
            return f"Could not answer '{result.sub_query_id}': {result.error}"
        return (
            f"Ran '{result.sub_query_id}' and returned {result.row_count} "
            f"{'row' if result.row_count == 1 else 'rows'} in "
            f"{result.execution_ms:.0f}ms."
        )
