"""Tests for the SQL analyst agent.

The LLM is scripted, but the **database is real**. Mocking the database would
make the self-correction tests meaningless: the whole point is that a genuine
SQLite error message is fed back to the model and produces working SQL on the
next attempt, and a fake executor would only prove the retry plumbing loops.
"""

from __future__ import annotations

import asyncio
from datetime import date
from typing import Any

import pytest

from app.agents.analyst import MAX_SQL_ATTEMPTS, SQLAnalystAgent
from app.agents.contracts import (
    AgentStatus,
    AnalysisPlan,
    QueryIntent,
    SubQuery,
    TimeWindow,
)
from app.core.llm import LLMError, LLMResponse

# A query that runs against the real database and returns exactly one row.
VALID_SQL = (
    "SELECT ROUND(SUM(net_before_tax), 2) AS revenue_inr, "
    "COUNT(DISTINCT order_id) AS orders FROM fact_orders "
    "WHERE order_date BETWEEN '2026-05-01' AND '2026-07-31'"
)


class FakeLLM:
    """A scripted stand-in for :class:`LLMClient` returning SQL text."""

    def __init__(self, *replies: str | Exception) -> None:
        """Initialise the fake.

        Args:
            *replies: SQL strings to return in order, or an exception to
                raise. The last entry repeats once exhausted.
        """
        self.replies = list(replies)
        self.calls: list[dict[str, str]] = []

    async def complete(self, system: str, user: str, **kwargs: Any) -> LLMResponse:
        """Return the next scripted reply.

        Args:
            system: The system prompt, recorded for assertions.
            user: The user prompt, recorded for assertions.
            **kwargs: Ignored generation settings.

        Returns:
            A synthetic completion carrying the scripted SQL.

        Raises:
            Exception: When the script says this call should fail.
        """
        self.calls.append({"system": system, "user": user})
        index = min(len(self.calls) - 1, len(self.replies) - 1)
        reply = self.replies[index]
        if isinstance(reply, Exception):
            raise reply
        return LLMResponse(
            text=reply,
            provider="anthropic",
            model="claude-opus-5",
            input_tokens=200,
            output_tokens=60,
            latency_ms=15.0,
            attempts=["anthropic"],
        )

    @property
    def call_count(self) -> int:
        """How many completions were requested.

        Returns:
            The number of calls made.
        """
        return len(self.calls)


@pytest.fixture
def plan() -> AnalysisPlan:
    """A minimal plan providing window context to the analyst.

    Returns:
        A valid AnalysisPlan.
    """
    return AnalysisPlan(
        question="What was revenue in the last 3 months?",
        intent=QueryIntent.AGGREGATE,
        time_window=TimeWindow(
            start_date=date(2026, 5, 1),
            end_date=date(2026, 7, 31),
            label="last 3 months",
        ),
        metrics=["revenue"],
        sub_queries=[SubQuery(id="headline", purpose="Total revenue.")],
        reasoning="One aggregate answers it.",
        confidence=0.95,
    )


@pytest.fixture
def sub_query() -> SubQuery:
    """The sub-query under test.

    Returns:
        A simple revenue sub-query.
    """
    return SubQuery(
        id="headline",
        purpose="Total revenue and order count for the window.",
        tables=["fact_orders"],
        metrics=["revenue", "orders"],
    )


class TestSuccessfulExecution:
    """Valid SQL runs against the real database."""

    @pytest.mark.asyncio
    async def test_valid_sql_returns_rows(
        self, plan: AnalysisPlan, sub_query: SubQuery
    ) -> None:
        """The query executes and the real figures come back."""
        agent = SQLAnalystAgent(llm=FakeLLM(VALID_SQL))

        result = await agent.execute(sub_query, plan)

        assert result.error is None
        assert result.row_count == 1
        assert result.rows[0]["revenue_inr"] == 3197076.5
        assert result.rows[0]["orders"] == 4930
        assert result.attempts == 1
        assert result.sub_query_id == "headline"

    @pytest.mark.asyncio
    async def test_executed_sql_is_recorded_verbatim(
        self, plan: AnalysisPlan, sub_query: SubQuery
    ) -> None:
        """The result carries what actually ran, including the added LIMIT."""
        agent = SQLAnalystAgent(llm=FakeLLM(VALID_SQL))

        result = await agent.execute(sub_query, plan)

        assert "fact_orders" in result.sql
        assert "LIMIT" in result.sql  # injected by the guard

    @pytest.mark.asyncio
    async def test_markdown_fences_are_stripped(
        self, plan: AnalysisPlan, sub_query: SubQuery
    ) -> None:
        """Models fence SQL despite instructions; the agent copes."""
        agent = SQLAnalystAgent(llm=FakeLLM(f"```sql\n{VALID_SQL}\n```"))

        result = await agent.execute(sub_query, plan)

        assert result.error is None
        assert not result.sql.startswith("```")

    @pytest.mark.asyncio
    async def test_trailing_semicolon_is_stripped(
        self, plan: AnalysisPlan, sub_query: SubQuery
    ) -> None:
        """A terminator would otherwise look like a second statement."""
        agent = SQLAnalystAgent(llm=FakeLLM(f"{VALID_SQL};"))

        result = await agent.execute(sub_query, plan)

        assert result.error is None


class TestValidationRetry:
    """SQL rejected by the guard is regenerated with the error fed back."""

    @pytest.mark.asyncio
    async def test_invalid_sql_triggers_regeneration(
        self, plan: AnalysisPlan, sub_query: SubQuery
    ) -> None:
        """A forbidden statement is caught and the second attempt succeeds."""
        llm = FakeLLM("DROP TABLE fact_orders", VALID_SQL)
        agent = SQLAnalystAgent(llm=llm)

        result = await agent.execute(sub_query, plan)

        assert llm.call_count == 2
        assert result.error is None
        assert result.attempts == 2

    @pytest.mark.asyncio
    async def test_retry_prompt_contains_the_validation_error(
        self, plan: AnalysisPlan, sub_query: SubQuery
    ) -> None:
        """The model is told exactly why its SQL was refused."""
        llm = FakeLLM("SELECT * FROM secret_table", VALID_SQL)
        agent = SQLAnalystAgent(llm=llm)

        await agent.execute(sub_query, plan)

        retry_prompt = llm.calls[1]["user"]
        assert "validation failed" in retry_prompt
        assert "secret_table" in retry_prompt
        assert "Your previous SQL was" in retry_prompt

    @pytest.mark.asyncio
    async def test_unknown_table_is_rejected_before_execution(
        self, plan: AnalysisPlan, sub_query: SubQuery
    ) -> None:
        """The guard stops it, so the database is never asked."""
        llm = FakeLLM("SELECT * FROM sqlite_master")
        agent = SQLAnalystAgent(llm=llm)

        result = await agent.execute(sub_query, plan)

        assert result.error is not None
        assert "validation" in result.error


class TestExecutionRetry:
    """SQL the database rejects is regenerated with the real error fed back."""

    @pytest.mark.asyncio
    async def test_execution_error_triggers_regeneration(
        self, plan: AnalysisPlan, sub_query: SubQuery
    ) -> None:
        """A bad column passes validation but fails execution, then recovers."""
        llm = FakeLLM(
            "SELECT no_such_column FROM fact_orders LIMIT 1",
            VALID_SQL,
        )
        agent = SQLAnalystAgent(llm=llm)

        result = await agent.execute(sub_query, plan)

        assert llm.call_count == 2
        assert result.error is None
        assert result.attempts == 2

    @pytest.mark.asyncio
    async def test_retry_prompt_contains_the_database_error(
        self, plan: AnalysisPlan, sub_query: SubQuery
    ) -> None:
        """The genuine SQLite message is what the model gets to work from."""
        llm = FakeLLM("SELECT no_such_column FROM fact_orders LIMIT 1", VALID_SQL)
        agent = SQLAnalystAgent(llm=llm)

        await agent.execute(sub_query, plan)

        retry_prompt = llm.calls[1]["user"]
        assert "execution failed" in retry_prompt
        assert "no_such_column" in retry_prompt


class TestExhaustedAttempts:
    """A sub-query that cannot be answered degrades rather than raising."""

    @pytest.mark.asyncio
    async def test_returns_a_result_with_an_error_not_an_exception(
        self, plan: AnalysisPlan, sub_query: SubQuery
    ) -> None:
        """The orchestrator decides how to degrade, so nothing is raised."""
        agent = SQLAnalystAgent(llm=FakeLLM("DROP TABLE fact_orders"))

        result = await agent.execute(sub_query, plan)

        assert result.error is not None
        assert result.rows == []
        assert result.row_count == 0
        assert result.attempts == MAX_SQL_ATTEMPTS
        assert result.sub_query_id == "headline"

    @pytest.mark.asyncio
    async def test_stops_after_max_attempts(
        self, plan: AnalysisPlan, sub_query: SubQuery
    ) -> None:
        """The loop is bounded; it does not retry forever."""
        llm = FakeLLM("DROP TABLE fact_orders")
        agent = SQLAnalystAgent(llm=llm)

        await agent.execute(sub_query, plan)

        assert llm.call_count == MAX_SQL_ATTEMPTS

    @pytest.mark.asyncio
    async def test_llm_failure_degrades_to_an_error_result(
        self, plan: AnalysisPlan, sub_query: SubQuery
    ) -> None:
        """Even a total provider outage produces a well-formed result."""
        agent = SQLAnalystAgent(llm=FakeLLM(LLMError("all providers failed")))

        result = await agent.execute(sub_query, plan)

        assert result.error is not None
        assert "LLM call failed" in result.error


class TestRunMany:
    """Independent sub-queries run concurrently."""

    @pytest.mark.asyncio
    async def test_preserves_sub_query_ids_and_order(
        self, plan: AnalysisPlan
    ) -> None:
        """Results come back in input order regardless of completion order."""
        sub_queries = [
            SubQuery(id="first", purpose="First piece of the analysis."),
            SubQuery(id="second", purpose="Second piece of the analysis."),
            SubQuery(id="third", purpose="Third piece of the analysis."),
        ]
        agent = SQLAnalystAgent(llm=FakeLLM(VALID_SQL))

        results = await agent.run_many(sub_queries, plan)

        assert [result.sub_query_id for result in results] == [
            "first",
            "second",
            "third",
        ]
        assert all(result.error is None for result in results)

    @pytest.mark.asyncio
    async def test_runs_concurrently_not_sequentially(
        self, plan: AnalysisPlan
    ) -> None:
        """Total time is close to one call, not the sum of all of them.

        A sequential implementation would take at least 3 x the per-call delay;
        a concurrent one takes roughly one.
        """
        delay = 0.15

        class SlowLLM(FakeLLM):
            """A fake that sleeps before replying."""

            async def complete(
                self, system: str, user: str, **kwargs: Any
            ) -> LLMResponse:
                """Sleep, then return the scripted SQL.

                Args:
                    system: The system prompt.
                    user: The user prompt.
                    **kwargs: Ignored.

                Returns:
                    The scripted completion.
                """
                await asyncio.sleep(delay)
                return await super().complete(system, user, **kwargs)

        sub_queries = [
            SubQuery(id=f"q{i}", purpose=f"Piece {i} of the analysis.")
            for i in range(3)
        ]
        agent = SQLAnalystAgent(llm=SlowLLM(VALID_SQL))

        started = asyncio.get_running_loop().time()
        results = await agent.run_many(sub_queries, plan)
        elapsed = asyncio.get_running_loop().time() - started

        assert len(results) == 3
        assert elapsed < delay * 2, f"took {elapsed:.2f}s; likely sequential"

    @pytest.mark.asyncio
    async def test_empty_list_returns_empty(self, plan: AnalysisPlan) -> None:
        """No sub-queries means no work and no error."""
        agent = SQLAnalystAgent(llm=FakeLLM(VALID_SQL))

        assert await agent.run_many([], plan) == []

    @pytest.mark.asyncio
    async def test_one_failure_does_not_lose_the_others(
        self, plan: AnalysisPlan
    ) -> None:
        """A partial answer beats no answer.

        The failing sub-query is reported, and the successful ones still carry
        their rows.
        """

        class MixedLLM(FakeLLM):
            """Fails for one sub-query id, succeeds for the rest."""

            async def complete(
                self, system: str, user: str, **kwargs: Any
            ) -> LLMResponse:
                """Return bad SQL for the doomed sub-query only.

                Args:
                    system: The system prompt.
                    user: The user prompt.
                    **kwargs: Ignored.

                Returns:
                    A completion carrying either bad or good SQL.
                """
                self.calls.append({"system": system, "user": user})
                sql = (
                    "DROP TABLE fact_orders"
                    if "doomed" in user
                    else VALID_SQL
                )
                return LLMResponse(
                    text=sql,
                    provider="anthropic",
                    model="claude-opus-5",
                    input_tokens=10,
                    output_tokens=5,
                    latency_ms=1.0,
                    attempts=["anthropic"],
                )

        sub_queries = [
            SubQuery(id="good", purpose="A query that works fine."),
            SubQuery(id="bad", purpose="The doomed query that cannot work."),
        ]
        agent = SQLAnalystAgent(llm=MixedLLM())

        results = await agent.run_many(sub_queries, plan)

        by_id = {result.sub_query_id: result for result in results}
        assert by_id["good"].error is None
        assert by_id["good"].row_count == 1
        assert by_id["bad"].error is not None


class TestPrompt:
    """The rules that prevent wrong SQL must actually be in the prompt."""

    def test_prompt_forbids_the_system_clock(self) -> None:
        """DATE('now') would return zero rows against this dataset."""
        prompt = SQLAnalystAgent(llm=FakeLLM()).build_system_prompt()

        assert "DATE('now')" in prompt
        assert "CURRENT_DATE" in prompt

    def test_prompt_states_the_revenue_column(self) -> None:
        """net_before_tax, never net_revenue."""
        prompt = SQLAnalystAgent(llm=FakeLLM()).build_system_prompt()

        assert "net_before_tax" in prompt
        assert "net_revenue" in prompt

    def test_prompt_states_the_grain_rule(self) -> None:
        """fact_orders for revenue, fact_order_lines for products only."""
        prompt = SQLAnalystAgent(llm=FakeLLM()).build_system_prompt()

        assert "fact_order_lines ONLY for SKU" in prompt

    def test_prompt_demands_left_joins(self) -> None:
        """An inner join silently drops anonymous orders."""
        prompt = SQLAnalystAgent(llm=FakeLLM()).build_system_prompt()

        assert "LEFT JOIN" in prompt

    def test_prompt_includes_worked_examples(self) -> None:
        """The few-shot examples from the semantic layer reach the model."""
        from app.semantic.schema import EXAMPLE_QUERIES

        prompt = SQLAnalystAgent(llm=FakeLLM()).build_system_prompt()

        for example in EXAMPLE_QUERIES:
            assert example["sql"] in prompt

    def test_user_prompt_carries_the_window(self, plan: AnalysisPlan) -> None:
        """The resolved dates reach the model, not a relative phrase."""
        agent = SQLAnalystAgent(llm=FakeLLM())

        prompt = agent.build_user_prompt(
            SubQuery(id="x", purpose="Some analysis to perform."), plan
        )

        assert "2026-05-01" in prompt
        assert "2026-07-31" in prompt


class TestInstrumentation:
    """The analyst reports itself into the trace like any other agent."""

    @pytest.mark.asyncio
    async def test_run_produces_a_trace_step(
        self, plan: AnalysisPlan, sub_query: SubQuery
    ) -> None:
        """A successful run records provider, tokens and timing."""
        agent = SQLAnalystAgent(llm=FakeLLM(VALID_SQL))

        outcome = await agent.run(sub_query, plan)

        assert outcome.succeeded is True
        assert outcome.step.agent_name == "sql_analyst"
        assert outcome.step.status is AgentStatus.SUCCEEDED
        assert outcome.step.tokens == 260
        assert "1 row" in outcome.step.summary

    @pytest.mark.asyncio
    async def test_failed_query_still_succeeds_as_a_step(
        self, plan: AnalysisPlan, sub_query: SubQuery
    ) -> None:
        """The agent completed its work even though the query did not.

        The failure is carried in the QueryResult, not the step, because the
        agent behaved correctly: it tried, failed, and reported.
        """
        agent = SQLAnalystAgent(llm=FakeLLM("DROP TABLE fact_orders"))

        outcome = await agent.run(sub_query, plan)

        assert outcome.succeeded is True
        assert outcome.result is not None
        assert outcome.result.error is not None
        assert "Could not answer" in outcome.step.summary
