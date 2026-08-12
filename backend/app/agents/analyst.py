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

8. OUTPUT. Return ONLY the SQL. No explanation, no markdown fences, no
   trailing semicolon. One statement.
"""


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
        if window.comparison_start and window.comparison_end:
            lines.append(
                f"Comparison window: {window.comparison_start.isoformat()} to "
                f"{window.comparison_end.isoformat()}"
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
