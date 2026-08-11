"""Tests for the semantic layer that agents read to write SQL.

The load-bearing test here is ``TestExampleQueries``: every few-shot example is
executed against the real database and must return rows. Few-shot SQL that
references a column that does not exist teaches an agent to hallucinate the same
column, so the examples are verified rather than trusted.

``TestTableAllowlist`` compares the allowlist against ``sqlite_master``, which
catches drift the moment the physical schema changes without the semantic layer
following.
"""

from __future__ import annotations

import sqlite3
from typing import Iterator

import pytest

from app.config import settings
from app.semantic.schema import (
    BUSINESS_RULES,
    COMPACT_SCHEMA,
    EXAMPLE_QUERIES,
    METRIC_DEFINITIONS,
    SCHEMA_DESCRIPTION,
    TABLE_ALLOWLIST,
    TABLES,
    TIME_ANCHOR,
    get_compact_schema,
    get_schema_context,
    month_keys,
)

# A compact schema that is not appreciably smaller than the full context has no
# reason to exist.
MAX_COMPACT_RATIO: float = 0.5


@pytest.fixture(scope="module")
def connection() -> Iterator[sqlite3.Connection]:
    """Open a connection to the real database.

    Yields:
        An open SQLite connection, closed on teardown.
    """
    conn = sqlite3.connect(settings.DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


class TestSchemaContext:
    """The assembled prompt block."""

    def test_context_is_non_empty(self) -> None:
        """The context has substantial content."""
        context = get_schema_context()
        assert context.strip()
        assert len(context) > 1_000

    @pytest.mark.parametrize("table", TABLE_ALLOWLIST)
    def test_context_names_every_table(self, table: str) -> None:
        """Every table an agent may query is described in the context."""
        assert table in get_schema_context()

    def test_context_has_all_sections(self) -> None:
        """The six prompt sections are present and ordered."""
        context = get_schema_context()
        headings = [
            "TIME ANCHOR",
            "DATABASE SCHEMA",
            "METRIC DEFINITIONS",
            "BUSINESS RULES",
            "EXAMPLE QUERIES",
            "TABLE ALLOWLIST",
        ]
        positions = [context.find(heading) for heading in headings]
        assert all(position > 0 for position in positions), dict(
            zip(headings, positions)
        )
        assert positions == sorted(positions)

    def test_context_includes_the_business_rules(self) -> None:
        """Every rule reaches the prompt."""
        context = get_schema_context()
        for rule in BUSINESS_RULES:
            assert rule in context

    def test_context_includes_the_example_sql(self) -> None:
        """Every worked example reaches the prompt."""
        context = get_schema_context()
        for example in EXAMPLE_QUERIES:
            assert example["sql"] in context

    def test_context_warns_against_the_system_clock(self) -> None:
        """The single most costly mistake is called out explicitly."""
        context = get_schema_context()
        assert "system clock" in context
        assert "DATE('now')" in context


class TestSchemaDescription:
    """The rendered physical schema."""

    def test_every_table_is_described(self) -> None:
        """Each table in the spec appears with its purpose and grain."""
        for table in TABLES:
            assert f"TABLE {table.name}" in SCHEMA_DESCRIPTION
            assert table.purpose in SCHEMA_DESCRIPTION
            assert table.grain in SCHEMA_DESCRIPTION

    def test_spec_matches_the_physical_columns(
        self, connection: sqlite3.Connection
    ) -> None:
        """Documented columns match the database exactly, in order."""
        for table in TABLES:
            physical = [
                row[1]
                for row in connection.execute(f"PRAGMA table_info({table.name})")
            ]
            documented = [column.name for column in table.columns]
            assert documented == physical, table.name

    def test_nullable_columns_explain_their_nulls(self) -> None:
        """Every nullable column says what a NULL means."""
        for table in TABLES:
            for column in table.columns:
                if column.nullable:
                    assert column.null_meaning, f"{table.name}.{column.name}"

    def test_nullable_columns_are_exactly_the_two_known_ones(self) -> None:
        """Only customer_id and promo_id are documented as nullable."""
        nullable = {
            f"{table.name}.{column.name}"
            for table in TABLES
            for column in table.columns
            if column.nullable
        }
        assert nullable == {"fact_orders.customer_id", "fact_orders.promo_id"}

    def test_row_counts_match_the_database(
        self, connection: sqlite3.Connection
    ) -> None:
        """Documented row counts are the real ones."""
        for table in TABLES:
            actual = connection.execute(
                f"SELECT COUNT(*) FROM {table.name}"
            ).fetchone()[0]
            assert table.row_count == actual, table.name


class TestMetricDefinitions:
    """The metric catalogue."""

    @pytest.mark.parametrize("name", sorted(METRIC_DEFINITIONS))
    def test_metric_is_complete(self, name: str) -> None:
        """Each metric carries a SQL expression, a description and a source."""
        metric = METRIC_DEFINITIONS[name]
        assert metric["sql"].strip()
        assert metric["description"].strip()
        assert metric["source_table"].strip()
        assert metric["unit"].strip()

    @pytest.mark.parametrize("name", sorted(METRIC_DEFINITIONS))
    def test_metric_source_table_is_allowed(self, name: str) -> None:
        """Metrics only reference tables an agent may query."""
        assert METRIC_DEFINITIONS[name]["source_table"] in TABLE_ALLOWLIST

    @pytest.mark.parametrize("name", sorted(METRIC_DEFINITIONS))
    def test_metric_sql_executes(
        self, connection: sqlite3.Connection, name: str
    ) -> None:
        """Each metric expression is valid SQL against its source table."""
        metric = METRIC_DEFINITIONS[name]
        value = connection.execute(
            f"SELECT {metric['sql']} FROM {metric['source_table']}"
        ).fetchone()[0]
        assert value is not None

    def test_revenue_metric_is_canonical(self) -> None:
        """The headline revenue metric uses the configured revenue column."""
        assert settings.REVENUE_METRIC in METRIC_DEFINITIONS["revenue"]["sql"]

    def test_revenue_and_revenue_with_tax_differ(
        self, connection: sqlite3.Connection
    ) -> None:
        """The tax-exclusive and tax-inclusive metrics are not the same number."""
        net = connection.execute(
            f"SELECT {METRIC_DEFINITIONS['revenue']['sql']} FROM fact_orders"
        ).fetchone()[0]
        gross = connection.execute(
            f"SELECT {METRIC_DEFINITIONS['revenue_with_tax']['sql']} FROM fact_orders"
        ).fetchone()[0]
        assert gross > net
        assert gross == pytest.approx(net * (1 + settings.TAX_RATE), rel=1e-6)


class TestTimeAnchor:
    """The time anchoring section."""

    def test_anchor_date_is_from_settings(self) -> None:
        """The stated 'today' is the configured as-of date."""
        assert settings.DATA_ASOF_DATE.isoformat() in TIME_ANCHOR

    def test_data_window_is_stated(self) -> None:
        """Both ends of the data window appear."""
        assert settings.DATA_START_DATE.isoformat() in TIME_ANCHOR
        assert settings.LAST_3M_START.isoformat() in TIME_ANCHOR
        assert settings.LAST_3M_END.isoformat() in TIME_ANCHOR

    def test_all_twelve_month_keys_are_listed(self) -> None:
        """Every valid month key is enumerated for the agent."""
        keys = month_keys()
        assert len(keys) == 12
        for key in keys:
            assert key in TIME_ANCHOR

    def test_month_keys_match_the_database(
        self, connection: sqlite3.Connection
    ) -> None:
        """The enumerated month keys are the ones that actually exist."""
        actual = [
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT month_key FROM fact_orders ORDER BY month_key"
            )
        ]
        assert month_keys() == actual

    def test_month_keys_bound_the_dataset(self) -> None:
        """The first and last keys match the configured window."""
        keys = month_keys()
        assert keys[0] == settings.DATA_START_DATE.strftime("%Y-%m")
        assert keys[-1] == settings.DATA_ASOF_DATE.strftime("%Y-%m")


class TestBusinessRules:
    """The rules that prevent the mistakes this dataset invites."""

    def test_rules_are_non_empty(self) -> None:
        """Every rule has content."""
        assert BUSINESS_RULES
        for rule in BUSINESS_RULES:
            assert rule.strip()

    def test_left_join_rules_are_present(self) -> None:
        """The two nullable dimensions are both covered."""
        joined = " ".join(BUSINESS_RULES)
        assert "LEFT JOIN dim_customer" in joined
        assert "LEFT JOIN dim_promotion" in joined

    def test_revenue_column_rule_is_present(self) -> None:
        """The canonical revenue column is stated, and its trap named."""
        joined = " ".join(BUSINESS_RULES)
        assert settings.REVENUE_METRIC in joined
        assert "net_revenue" in joined

    def test_grain_rule_is_present(self) -> None:
        """The line-versus-order grain rule is stated."""
        joined = " ".join(BUSINESS_RULES)
        assert "fact_order_lines" in joined
        assert "fact_orders" in joined


class TestExampleQueries:
    """Few-shot examples must be real, executable SQL."""

    def test_four_distinct_patterns(self) -> None:
        """The examples teach four different things."""
        assert len(EXAMPLE_QUERIES) == 4
        assert len({example["pattern"] for example in EXAMPLE_QUERIES}) == 4

    @pytest.mark.parametrize(
        "example", EXAMPLE_QUERIES, ids=[e["pattern"] for e in EXAMPLE_QUERIES]
    )
    def test_example_is_complete(self, example: dict[str, str]) -> None:
        """Each example carries a question, a lesson and SQL."""
        assert example["question"].strip()
        assert example["teaches"].strip()
        assert example["sql"].strip()

    @pytest.mark.parametrize(
        "example", EXAMPLE_QUERIES, ids=[e["pattern"] for e in EXAMPLE_QUERIES]
    )
    def test_example_sql_executes_and_returns_rows(
        self, connection: sqlite3.Connection, example: dict[str, str]
    ) -> None:
        """The SQL runs against the real database and returns data."""
        rows = connection.execute(example["sql"]).fetchall()
        assert rows, example["question"]
        assert all(row is not None for row in rows)

    @pytest.mark.parametrize(
        "example", EXAMPLE_QUERIES, ids=[e["pattern"] for e in EXAMPLE_QUERIES]
    )
    def test_example_sql_only_touches_allowed_tables(
        self, example: dict[str, str]
    ) -> None:
        """Examples never demonstrate a table outside the allowlist."""
        sql = example["sql"]
        for token in ("FROM ", "JOIN "):
            for fragment in sql.split(token)[1:]:
                table = fragment.split()[0].strip("(),")
                assert table in TABLE_ALLOWLIST, table

    def test_examples_use_the_canonical_revenue_column(self) -> None:
        """No example teaches revenue from the tax-inclusive column."""
        for example in EXAMPLE_QUERIES:
            assert "net_revenue" not in example["sql"], example["pattern"]

    def test_examples_never_use_the_system_clock(self) -> None:
        """No example resolves dates against the real clock."""
        for example in EXAMPLE_QUERIES:
            lowered = example["sql"].lower()
            assert "date('now')" not in lowered
            assert "current_date" not in lowered

    def test_time_windowed_examples_use_the_configured_window(self) -> None:
        """Date literals come from the config, not from invention."""
        windowed = [
            example
            for example in EXAMPLE_QUERIES
            if settings.LAST_3M_START.isoformat() in example["sql"]
        ]
        assert len(windowed) == 3
        for example in windowed:
            assert settings.LAST_3M_END.isoformat() in example["sql"]


class TestTableAllowlist:
    """The allowlist is the guardrail's whitelist; it must match reality."""

    def test_allowlist_matches_the_database(
        self, connection: sqlite3.Connection
    ) -> None:
        """The allowlist equals the set of real tables, exactly."""
        actual = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        assert set(TABLE_ALLOWLIST) == actual

    def test_allowlist_matches_the_documented_tables(self) -> None:
        """Every allowed table is also a documented table."""
        assert set(TABLE_ALLOWLIST) == {table.name for table in TABLES}

    def test_allowlist_has_no_duplicates(self) -> None:
        """Each table appears once."""
        assert len(TABLE_ALLOWLIST) == len(set(TABLE_ALLOWLIST))


class TestCompactSchema:
    """The cheap variant."""

    def test_compact_is_non_empty(self) -> None:
        """The compact schema has content."""
        assert get_compact_schema().strip()

    def test_compact_is_meaningfully_shorter(self) -> None:
        """The compact variant is worth having."""
        compact = len(get_compact_schema())
        full = len(get_schema_context())
        assert compact < full * MAX_COMPACT_RATIO, f"{compact} vs {full}"

    @pytest.mark.parametrize("table", TABLE_ALLOWLIST)
    def test_compact_names_every_table(self, table: str) -> None:
        """Name resolution still works from the compact variant."""
        assert table in get_compact_schema()

    def test_compact_lists_every_column(self) -> None:
        """Every column name is resolvable from the compact variant."""
        for table in TABLES:
            for column in table.columns:
                assert column.name in COMPACT_SCHEMA

    def test_compact_keeps_the_time_anchor(self) -> None:
        """Even the cheap variant states the fixed anchor date."""
        assert settings.DATA_ASOF_DATE.isoformat() in get_compact_schema()

    def test_compact_keeps_the_null_join_rule(self) -> None:
        """The cheap variant still warns about the nullable dimensions."""
        compact = get_compact_schema()
        assert "LEFT JOIN" in compact
        assert settings.REVENUE_METRIC in compact
