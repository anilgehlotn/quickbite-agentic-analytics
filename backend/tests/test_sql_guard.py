"""Tests for the SQL guard, treated as the security boundary it is.

The system is publicly deployed with an LLM writing queries, so anything a user
types reaches a SQL generator. These tests assume an adversary, not a confused
agent: the rejection cases include payloads shaped to defeat regex-based
filtering, and the final class verifies that the read-only connection refuses a
write even when every validation layer is bypassed.
"""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import pytest

from app.config import settings
from app.core.sql_guard import (
    MAX_JOINS,
    QueryExecutionError,
    SafeQueryExecutor,
    SQLGuard,
    to_json_safe,
)
from app.semantic.schema import TABLE_ALLOWLIST


@pytest.fixture
def guard() -> SQLGuard:
    """A guard configured exactly as production uses it.

    Returns:
        A guard bound to the semantic layer's allowlist.
    """
    return SQLGuard()


class TestRejectsWrites:
    """No statement that modifies anything may pass."""

    @pytest.mark.parametrize(
        "sql",
        [
            "DROP TABLE dim_store",
            "DELETE FROM fact_orders",
            "DELETE FROM fact_orders WHERE order_id = 'ORD000001'",
            "UPDATE dim_store SET city = 'X'",
            "INSERT INTO dim_store VALUES ('ST999')",
            "CREATE TABLE evil (a INT)",
            "ALTER TABLE dim_store ADD COLUMN evil TEXT",
            "DROP INDEX idx_fact_orders_store_id",
        ],
    )
    def test_write_statements_are_rejected(self, guard: SQLGuard, sql: str) -> None:
        """Every write verb is refused on the parsed node type."""
        result = guard.validate(sql)

        assert result.valid is False
        assert result.errors

    def test_rejection_names_the_statement_type(self, guard: SQLGuard) -> None:
        """The error says what was refused, so an agent can correct itself."""
        result = guard.validate("DROP TABLE dim_store")

        assert "DROP" in result.error_message
        assert "only SELECT" in result.error_message


class TestRejectsDangerousStatements:
    """Statements that reach outside the query surface."""

    @pytest.mark.parametrize(
        "sql",
        [
            "PRAGMA table_info(dim_store)",
            "PRAGMA writable_schema = 1",
            "ATTACH DATABASE '/tmp/evil.db' AS evil",
            "DETACH DATABASE evil",
            "VACUUM",
        ],
    )
    def test_non_query_statements_are_rejected(
        self, guard: SQLGuard, sql: str
    ) -> None:
        """PRAGMA, ATTACH and VACUUM are all refused.

        Some fail the statement-type check and some fail to parse as SQLite at
        all; either way they never reach the database.
        """
        assert guard.validate(sql).valid is False

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT name FROM sqlite_master",
            "SELECT sql FROM sqlite_master WHERE type = 'table'",
            "SELECT * FROM sqlite_temp_master",
            "SELECT * FROM sqlite_sequence",
        ],
    )
    def test_sqlite_internal_tables_are_rejected(
        self, guard: SQLGuard, sql: str
    ) -> None:
        """The schema catalogue is not readable.

        sqlite_master exposes the definition of every table, including ones
        outside the allowlist.
        """
        result = guard.validate(sql)

        assert result.valid is False
        assert "internal" in result.error_message or "unknown table" in result.error_message


class TestRejectsMultipleStatements:
    """Statement stacking is the classic injection payload."""

    def test_two_statements_are_rejected(self, guard: SQLGuard) -> None:
        """A second statement after a semicolon is counted, not matched."""
        result = guard.validate("SELECT 1 FROM fact_orders; DROP TABLE dim_store")

        assert result.valid is False
        assert "one statement" in result.error_message

    def test_trailing_semicolon_alone_is_fine(self, guard: SQLGuard) -> None:
        """A terminator is not a second statement."""
        assert guard.validate("SELECT store_id FROM dim_store LIMIT 5;").valid is True

    def test_three_statements_are_rejected(self, guard: SQLGuard) -> None:
        """The count is reported, not just the fact of stacking."""
        result = guard.validate(
            "SELECT 1 FROM dim_store; SELECT 2 FROM dim_store; DELETE FROM dim_store"
        )

        assert result.valid is False
        assert "found 3" in result.error_message


class TestPromptInjectionPayloads:
    """Payloads shaped to defeat naive filtering.

    Each of these would slip past a regex that looks for forbidden keywords,
    because the keyword is hidden behind a comment, a string literal or
    unusual whitespace. Parsing sees through all three.
    """

    def test_comment_hidden_second_statement(self, guard: SQLGuard) -> None:
        """A line comment before the payload does not hide it from a parser."""
        payload = (
            "SELECT store_id FROM dim_store -- ignore previous instructions\n"
            "; DROP TABLE fact_orders"
        )
        result = guard.validate(payload)

        assert result.valid is False

    def test_union_to_sqlite_master(self, guard: SQLGuard) -> None:
        """A UNION is still a query, so the table check must catch it."""
        payload = (
            "SELECT store_name FROM dim_store "
            "UNION SELECT sql FROM sqlite_master"
        )
        result = guard.validate(payload)

        assert result.valid is False
        assert "sqlite_master" in result.error_message

    def test_block_comment_between_keywords(self, guard: SQLGuard) -> None:
        """Comments inside the statement do not disguise the verb."""
        result = guard.validate("DROP/**/TABLE/**/dim_store")

        assert result.valid is False

    def test_newline_and_tab_separated_statements(self, guard: SQLGuard) -> None:
        """Whitespace tricks do not merge two statements into one."""
        result = guard.validate("SELECT 1 FROM dim_store ;\n\t\r\n DELETE FROM dim_store")

        assert result.valid is False

    def test_forbidden_word_inside_a_string_literal_is_harmless(
        self, guard: SQLGuard
    ) -> None:
        """A regex filter would reject this valid query; a parser does not.

        False positives matter too: rejecting legitimate SQL because it
        contains the word DROP in a string would break real questions.
        """
        result = guard.validate(
            "SELECT store_name FROM dim_store WHERE city = 'DROP TABLE' LIMIT 5"
        )

        assert result.valid is True

    def test_subquery_to_a_forbidden_table(self, guard: SQLGuard) -> None:
        """Nesting does not hide a table from the AST walk."""
        result = guard.validate(
            "SELECT store_id FROM dim_store WHERE store_id IN "
            "(SELECT name FROM sqlite_master) LIMIT 5"
        )

        assert result.valid is False


class TestTableAllowlist:
    """Only tables the semantic layer declares may be read."""

    def test_unknown_table_is_rejected(self, guard: SQLGuard) -> None:
        """A table that does not exist is refused with the valid list."""
        result = guard.validate("SELECT * FROM secret_payroll LIMIT 5")

        assert result.valid is False
        assert "secret_payroll" in result.error_message
        assert "fact_orders" in result.error_message  # names what IS allowed

    @pytest.mark.parametrize("table", TABLE_ALLOWLIST)
    def test_every_allowlisted_table_is_accepted(
        self, guard: SQLGuard, table: str
    ) -> None:
        """The guard and the semantic layer agree about what exists."""
        assert guard.validate(f"SELECT * FROM {table} LIMIT 1").valid is True

    def test_reports_tables_referenced(self, guard: SQLGuard) -> None:
        """The result lists what the query reads, for the verifier."""
        result = guard.validate(
            "SELECT o.order_id FROM fact_orders o "
            "LEFT JOIN dim_store s ON s.store_id = o.store_id LIMIT 5"
        )

        assert result.tables_referenced == ["dim_store", "fact_orders"]


class TestAcceptsValidQueries:
    """Legitimate analytical SQL must pass."""

    def test_plain_select(self, guard: SQLGuard) -> None:
        """The simplest query passes."""
        assert guard.validate("SELECT COUNT(*) FROM fact_orders LIMIT 1").valid is True

    def test_select_with_joins(self, guard: SQLGuard) -> None:
        """A realistic multi-join query passes."""
        result = guard.validate(
            "SELECT s.city, SUM(o.net_before_tax) AS revenue "
            "FROM fact_orders o "
            "LEFT JOIN dim_store s ON s.store_id = o.store_id "
            "LEFT JOIN dim_customer c ON c.customer_id = o.customer_id "
            "GROUP BY s.city LIMIT 10"
        )

        assert result.valid is True

    def test_cte_is_not_mistaken_for_an_unknown_table(self, guard: SQLGuard) -> None:
        """A CTE name is defined by the query, not a table to look up.

        Without subtracting CTE names, every WITH query would be rejected.
        """
        result = guard.validate(
            "WITH monthly AS ("
            "  SELECT month_key, SUM(net_before_tax) AS r FROM fact_orders "
            "  GROUP BY month_key"
            ") SELECT month_key, r FROM monthly ORDER BY month_key LIMIT 12"
        )

        assert result.valid is True, result.error_message
        assert "monthly" not in result.tables_referenced
        assert result.tables_referenced == ["fact_orders"]

    def test_multiple_ctes(self, guard: SQLGuard) -> None:
        """Several CTEs, including one reading another, all resolve."""
        result = guard.validate(
            "WITH a AS (SELECT store_id, net_before_tax FROM fact_orders), "
            "b AS (SELECT store_id, SUM(net_before_tax) AS r FROM a GROUP BY store_id) "
            "SELECT * FROM b LIMIT 5"
        )

        assert result.valid is True, result.error_message
        assert result.tables_referenced == ["fact_orders"]

    def test_subquery_on_allowed_tables(self, guard: SQLGuard) -> None:
        """Nested selects over allowed tables pass."""
        result = guard.validate(
            "SELECT store_id FROM mart_store_month WHERE revenue_net > "
            "(SELECT AVG(revenue_net) FROM mart_store_month) LIMIT 10"
        )

        assert result.valid is True

    def test_window_function(self, guard: SQLGuard) -> None:
        """Window functions are ordinary SELECT syntax."""
        result = guard.validate(
            "SELECT month_key, revenue_net, "
            "LAG(revenue_net) OVER (ORDER BY month_key) AS prev "
            "FROM mart_city_month WHERE city = 'Mumbai' LIMIT 12"
        )

        assert result.valid is True

    def test_union_of_allowed_tables(self, guard: SQLGuard) -> None:
        """A UNION is permitted when both sides are allowed."""
        result = guard.validate(
            "SELECT city FROM mart_city_month UNION "
            "SELECT city FROM dim_store LIMIT 20"
        )

        assert result.valid is True


class TestLimitInjection:
    """Unbounded queries would exhaust memory on a free-tier host."""

    def test_limit_is_injected_when_absent(self, guard: SQLGuard) -> None:
        """A missing LIMIT is added and reported as a warning, not an error."""
        result = guard.validate("SELECT * FROM fact_orders")

        assert result.valid is True
        assert f"LIMIT {settings.MAX_QUERY_ROWS}" in result.sql
        assert any("LIMIT" in warning for warning in result.warnings)

    def test_existing_limit_is_preserved(self, guard: SQLGuard) -> None:
        """A smaller LIMIT the agent chose is not overwritten."""
        result = guard.validate("SELECT * FROM fact_orders LIMIT 5")

        assert result.valid is True
        assert "LIMIT 5" in result.sql
        assert result.warnings == []

    def test_limit_injected_on_a_union(self, guard: SQLGuard) -> None:
        """Set operations get a row cap too."""
        result = guard.validate(
            "SELECT city FROM dim_store UNION SELECT city FROM mart_city_month"
        )

        assert result.valid is True
        assert f"LIMIT {settings.MAX_QUERY_ROWS}" in result.sql

    def test_custom_max_rows_is_respected(self) -> None:
        """The cap comes from configuration, not a constant."""
        result = SQLGuard(max_rows=7).validate("SELECT * FROM fact_orders")

        assert "LIMIT 7" in result.sql


class TestMalformedInput:
    """Garbage in produces a clear rejection, not a crash."""

    @pytest.mark.parametrize(
        "sql",
        [
            "",
            "   ",
            "this is not sql at all",
            "SELECT FROM WHERE",
            "SELECT * FROM (((",
        ],
    )
    def test_unparseable_sql_is_rejected(self, guard: SQLGuard, sql: str) -> None:
        """Every malformed input is refused with a reason."""
        result = guard.validate(sql)

        assert result.valid is False
        assert result.errors

    def test_parse_error_is_reported(self, guard: SQLGuard) -> None:
        """The parse failure reaches the agent so it can fix its syntax."""
        result = guard.validate("SELECT * FROM (((")

        assert "does not parse" in result.error_message


class TestComplexityGuard:
    """A join explosion is refused before it reaches the planner."""

    def test_excessive_joins_are_rejected(self, guard: SQLGuard) -> None:
        """More joins than permitted fails with the count."""
        joins = " ".join(
            f"LEFT JOIN dim_store s{i} ON s{i}.store_id = o.store_id"
            for i in range(MAX_JOINS + 1)
        )
        result = guard.validate(f"SELECT o.order_id FROM fact_orders o {joins} LIMIT 5")

        assert result.valid is False
        assert "joins" in result.error_message

    def test_reasonable_joins_are_accepted(self, guard: SQLGuard) -> None:
        """A realistic query with several joins passes."""
        result = guard.validate(
            "SELECT o.order_id FROM fact_orders o "
            "LEFT JOIN dim_store s ON s.store_id = o.store_id "
            "LEFT JOIN dim_customer c ON c.customer_id = o.customer_id "
            "LEFT JOIN dim_promotion p ON p.promo_id = o.promo_id LIMIT 5"
        )

        assert result.valid is True


class TestSafeQueryExecutor:
    """Execution runs read-only, bounded and without leaking raw errors."""

    @pytest.fixture
    def executor(self) -> SafeQueryExecutor:
        """An executor bound to the real database.

        Returns:
            A configured executor.
        """
        return SafeQueryExecutor()

    def test_executes_a_valid_query(self, executor: SafeQueryExecutor) -> None:
        """A real query returns real rows."""
        result = executor.execute(
            "SELECT COUNT(*) AS n FROM fact_orders WHERE order_date "
            "BETWEEN '2026-05-01' AND '2026-07-31'"
        )

        assert result.columns == ["n"]
        assert result.rows[0]["n"] == 4930
        assert result.execution_ms >= 0

    def test_rows_are_dicts(self, executor: SafeQueryExecutor) -> None:
        """Rows are JSON-safe mappings, not sqlite3.Row objects."""
        result = executor.execute("SELECT store_id, city FROM dim_store LIMIT 2")

        assert isinstance(result.rows[0], dict)
        assert set(result.rows[0]) == {"store_id", "city"}

    def test_row_cap_truncates_an_oversized_explicit_limit(self) -> None:
        """A query that sets its own large LIMIT is still capped.

        LIMIT injection handles the unbounded case, so this is the net for the
        case injection cannot cover: the agent asked for more rows than the
        executor will return.
        """
        executor = SafeQueryExecutor(max_rows=5)

        result = executor.execute("SELECT order_id FROM fact_orders LIMIT 50")

        assert result.row_count == 5
        assert result.truncated is True

    def test_injected_limit_means_no_truncation(self) -> None:
        """When the guard injects the cap, the database returns exactly it."""
        executor = SafeQueryExecutor(max_rows=5)

        result = executor.execute("SELECT order_id FROM fact_orders")

        assert result.row_count == 5
        assert result.truncated is False
        assert "LIMIT 5" in result.sql

    def test_connection_is_read_only(self, executor: SafeQueryExecutor) -> None:
        """The last line of defence: SQLite itself refuses the write.

        This is the check that still holds if every validation layer above it
        has a bug, because it is enforced by the database, not by our code.
        """
        connection = executor.connect()
        try:
            with pytest.raises(sqlite3.OperationalError, match="readonly"):
                connection.execute("CREATE TABLE evil (a INTEGER)")
            with pytest.raises(sqlite3.OperationalError, match="readonly"):
                connection.execute("DELETE FROM fact_orders")
        finally:
            connection.close()

    def test_original_database_is_unchanged_after_a_write_attempt(
        self, tmp_path: Path
    ) -> None:
        """A refused write leaves no trace in the file."""
        copy = tmp_path / "copy.db"
        shutil.copy(settings.DB_PATH, copy)
        executor = SafeQueryExecutor(db_path=copy)

        connection = executor.connect()
        try:
            with pytest.raises(sqlite3.OperationalError):
                connection.execute("DELETE FROM fact_orders")
        finally:
            connection.close()

        verify = sqlite3.connect(copy)
        try:
            assert verify.execute("SELECT COUNT(*) FROM fact_orders").fetchone()[0] == 20000
        finally:
            verify.close()

    def test_invalid_sql_raises_query_execution_error(
        self, executor: SafeQueryExecutor
    ) -> None:
        """Validation failures surface as QueryExecutionError, not a bare raise."""
        with pytest.raises(QueryExecutionError, match="validation"):
            executor.execute("DROP TABLE dim_store")

    def test_execution_error_carries_the_sql(
        self, executor: SafeQueryExecutor
    ) -> None:
        """The failing SQL is attached because it is fed back to the agent."""
        with pytest.raises(QueryExecutionError) as caught:
            executor.execute("SELECT no_such_column FROM fact_orders LIMIT 1")

        assert "no_such_column" in str(caught.value)
        assert caught.value.sql

    def test_no_raw_sqlite_error_escapes(self, executor: SafeQueryExecutor) -> None:
        """Every database failure is wrapped."""
        with pytest.raises(QueryExecutionError):
            executor.execute("SELECT * FROM fact_orders WHERE bad_col = 1 LIMIT 1")

    def test_missing_database_is_reported_clearly(self, tmp_path: Path) -> None:
        """A missing file is an explicit error, not an obscure sqlite one."""
        executor = SafeQueryExecutor(db_path=tmp_path / "nope.db")

        with pytest.raises(QueryExecutionError, match="database not found"):
            executor.execute("SELECT 1 FROM fact_orders LIMIT 1")

    def test_limit_is_injected_during_execution(
        self, executor: SafeQueryExecutor
    ) -> None:
        """The executor re-validates, so an unbounded query is capped."""
        result = executor.execute("SELECT order_id FROM fact_orders")

        assert f"LIMIT {settings.MAX_QUERY_ROWS}" in result.sql


class TestJsonSafeConversion:
    """Result values must survive JSON serialization."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (None, None),
            (1, 1),
            (1.5, 1.5),
            ("text", "text"),
            (True, True),
            (b"bytes", "bytes"),
        ],
    )
    def test_conversions(self, value: object, expected: object) -> None:
        """Native types pass through; bytes are decoded."""
        assert to_json_safe(value) == expected
