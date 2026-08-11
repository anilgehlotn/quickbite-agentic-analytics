"""Tests for the ETL pipeline that builds the SQLite star schema.

The database is built once per test session into a temporary directory, so the
tests never depend on (or overwrite) the committed ``data/quickbite.db``.

These tests encode the dataset's known shape as hard assertions. The most
important ones guard against silent data loss: 5,664 orders are anonymous
walk-ins with a NULL ``customer_id`` and 19,160 orders carry no promotion. A
pipeline that inner-joins its dimensions would quietly drop those rows and every
downstream number would be wrong but plausible.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterator

import pytest

from app.config import settings
from app.etl.build_db import NON_FESTIVE_PERIOD, WEEKEND_DAY_TYPE, build_database

# The dataset's expected shape and the reconciliation tolerances live with the
# quality gate, which is their production home; the tests assert against the
# same constants rather than restating them.
from app.etl.quality_checks import (
    EXPECTED_ANONYMOUS_ORDERS,
    EXPECTED_MONTH_COUNT,
    EXPECTED_PROMO_ORDERS,
    EXPECTED_ROW_COUNTS,
    MONTH_KEY_GLOB,
    REVENUE_TOLERANCE_INR,
)

# Tables carrying a month_key column.
MONTH_KEY_TABLES: tuple[str, ...] = (
    "dim_calendar",
    "fact_orders",
    "mart_store_month",
    "mart_city_month",
    "mart_channel_month",
)

EXPECTED_INDEXES: tuple[str, ...] = (
    "idx_fact_orders_order_date",
    "idx_fact_orders_store_id",
    "idx_fact_orders_channel",
    "idx_fact_orders_month_key",
    "idx_fact_orders_is_weekend",
    "idx_fact_orders_festive_period",
    "idx_fact_order_lines_order_id",
    "idx_fact_order_lines_sku_id",
    "idx_mart_store_month_month_key",
    "idx_mart_city_month_month_key",
    "idx_mart_channel_month_month_key",
)

MART_TABLES: tuple[str, ...] = (
    "mart_store_month",
    "mart_city_month",
    "mart_channel_month",
)


@pytest.fixture(scope="session")
def database_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the database once for the whole session.

    Args:
        tmp_path_factory: pytest factory for session-scoped temp directories.

    Returns:
        Path to the freshly built SQLite database.
    """
    path = tmp_path_factory.mktemp("etl") / "quickbite_test.db"
    build_database(db_path=path, verbose=False)
    return path


@pytest.fixture(scope="session")
def connection(database_path: Path) -> Iterator[sqlite3.Connection]:
    """Open a read connection to the session database.

    Args:
        database_path: Path to the built database.

    Yields:
        An open SQLite connection, closed on teardown.
    """
    conn = sqlite3.connect(database_path)
    try:
        yield conn
    finally:
        conn.close()


def scalar(connection: sqlite3.Connection, sql: str) -> Any:
    """Run a query and return its first column of its first row.

    Args:
        connection: Open SQLite connection.
        sql: Query returning a single value.

    Returns:
        The single value produced by the query.
    """
    return connection.execute(sql).fetchone()[0]


def column_values(connection: sqlite3.Connection, sql: str) -> list[Any]:
    """Run a query and return its first column as a list.

    Args:
        connection: Open SQLite connection.
        sql: Query returning one column.

    Returns:
        Every value in the first column.
    """
    return [row[0] for row in connection.execute(sql).fetchall()]


class TestSchema:
    """Tables, row counts and indexes."""

    def test_all_tables_exist(self, connection: sqlite3.Connection) -> None:
        """Every dimension, fact and mart table was created."""
        tables = set(
            column_values(
                connection,
                "SELECT name FROM sqlite_master WHERE type = 'table'",
            )
        )
        expected = set(EXPECTED_ROW_COUNTS) | set(MART_TABLES)
        assert expected <= tables

    @pytest.mark.parametrize(
        ("table", "expected"), sorted(EXPECTED_ROW_COUNTS.items())
    )
    def test_row_counts(
        self, connection: sqlite3.Connection, table: str, expected: int
    ) -> None:
        """Each table holds exactly the number of rows the dataset ships."""
        assert scalar(connection, f"SELECT COUNT(*) FROM {table}") == expected

    def test_mart_grain_is_one_row_per_entity_month(
        self, connection: sqlite3.Connection
    ) -> None:
        """Marts hold one row per entity per month across all twelve months."""
        store_count = scalar(connection, "SELECT COUNT(*) FROM dim_store")
        assert (
            scalar(connection, "SELECT COUNT(*) FROM mart_store_month")
            == store_count * EXPECTED_MONTH_COUNT
        )
        assert scalar(
            connection, "SELECT COUNT(*) FROM mart_channel_month"
        ) == len(settings.CHANNELS) * EXPECTED_MONTH_COUNT

    def test_expected_indexes_exist(self, connection: sqlite3.Connection) -> None:
        """The query indexes on the facts and marts were created."""
        indexes = set(
            column_values(
                connection,
                "SELECT name FROM sqlite_master WHERE type = 'index'",
            )
        )
        assert set(EXPECTED_INDEXES) <= indexes


class TestNullPreservation:
    """Anonymous orders and promotion-free orders must survive the load."""

    def test_anonymous_orders_preserved_as_null(
        self, connection: sqlite3.Connection
    ) -> None:
        """Exactly 5,664 orders keep a NULL customer_id."""
        assert (
            scalar(
                connection,
                "SELECT COUNT(*) FROM fact_orders WHERE customer_id IS NULL",
            )
            == EXPECTED_ANONYMOUS_ORDERS
        )

    def test_identified_orders_count(self, connection: sqlite3.Connection) -> None:
        """Anonymous and identified orders together account for every order."""
        identified = scalar(
            connection,
            "SELECT COUNT(*) FROM fact_orders WHERE customer_id IS NOT NULL",
        )
        assert (
            identified + EXPECTED_ANONYMOUS_ORDERS
            == EXPECTED_ROW_COUNTS["fact_orders"]
        )

    def test_promo_orders_count(self, connection: sqlite3.Connection) -> None:
        """Exactly 840 orders carry a promotion; NULL is the normal case."""
        assert (
            scalar(
                connection,
                "SELECT COUNT(*) FROM fact_orders WHERE promo_id IS NOT NULL",
            )
            == EXPECTED_PROMO_ORDERS
        )


class TestDateRange:
    """The fact table must cover the full dataset window and nothing else."""

    def test_min_order_date_is_dataset_start(
        self, connection: sqlite3.Connection
    ) -> None:
        """The earliest order falls on the configured start date."""
        assert (
            scalar(connection, "SELECT MIN(order_date) FROM fact_orders")
            == settings.DATA_START_DATE.isoformat()
        )

    def test_max_order_date_is_asof_date(
        self, connection: sqlite3.Connection
    ) -> None:
        """The latest order falls on the configured as-of date."""
        assert (
            scalar(connection, "SELECT MAX(order_date) FROM fact_orders")
            == settings.DATA_ASOF_DATE.isoformat()
        )

    def test_calendar_covers_dataset_window(
        self, connection: sqlite3.Connection
    ) -> None:
        """dim_calendar spans the same window as the orders."""
        assert (
            scalar(connection, "SELECT MIN(date) FROM dim_calendar")
            == settings.DATA_START_DATE.isoformat()
        )
        assert (
            scalar(connection, "SELECT MAX(date) FROM dim_calendar")
            == settings.DATA_ASOF_DATE.isoformat()
        )

    def test_order_hour_is_a_valid_hour(
        self, connection: sqlite3.Connection
    ) -> None:
        """order_hour was derived from the timestamp, not miscast."""
        assert (
            scalar(
                connection,
                "SELECT COUNT(*) FROM fact_orders "
                "WHERE order_hour < 0 OR order_hour > 23",
            )
            == 0
        )


class TestDomainVocabulary:
    """Fact values must stay inside the vocabularies declared in config."""

    def test_channels_match_config(self, connection: sqlite3.Connection) -> None:
        """Every channel in the fact table is a configured channel."""
        channels = column_values(
            connection, "SELECT DISTINCT channel FROM fact_orders"
        )
        assert set(channels) <= set(settings.CHANNELS)
        assert set(channels) == set(settings.CHANNELS)

    def test_festive_periods_match_config(
        self, connection: sqlite3.Connection
    ) -> None:
        """Festive periods are the configured ones plus the non-festive value."""
        periods = set(
            column_values(
                connection, "SELECT DISTINCT festive_period FROM fact_orders"
            )
        )
        assert periods <= set(settings.FESTIVE_PERIODS) | {NON_FESTIVE_PERIOD}

    def test_is_festive_flag_matches_festive_period(
        self, connection: sqlite3.Connection
    ) -> None:
        """is_festive is 1 exactly when the day is in a festive period."""
        placeholders = ", ".join("?" for _ in settings.FESTIVE_PERIODS)
        mismatches = connection.execute(
            "SELECT COUNT(*) FROM fact_orders "
            f"WHERE is_festive != (festive_period IN ({placeholders}))",
            settings.FESTIVE_PERIODS,
        ).fetchone()[0]
        assert mismatches == 0


class TestDerivedColumns:
    """Columns computed during the load."""

    @pytest.mark.parametrize("table", MONTH_KEY_TABLES)
    def test_month_key_format(
        self, connection: sqlite3.Connection, table: str
    ) -> None:
        """Every month_key is formatted YYYY-MM."""
        assert (
            scalar(
                connection,
                f"SELECT COUNT(*) FROM {table} "
                f"WHERE month_key NOT GLOB '{MONTH_KEY_GLOB}'",
            )
            == 0
        )

    @pytest.mark.parametrize("table", MONTH_KEY_TABLES)
    def test_twelve_distinct_month_keys(
        self, connection: sqlite3.Connection, table: str
    ) -> None:
        """The dataset spans exactly twelve months everywhere."""
        assert (
            scalar(connection, f"SELECT COUNT(DISTINCT month_key) FROM {table}")
            == EXPECTED_MONTH_COUNT
        )

    def test_month_key_agrees_with_order_date(
        self, connection: sqlite3.Connection
    ) -> None:
        """month_key is the year-month prefix of the order date."""
        assert (
            scalar(
                connection,
                "SELECT COUNT(*) FROM fact_orders "
                "WHERE month_key != substr(order_date, 1, 7)",
            )
            == 0
        )

    def test_is_weekend_matches_day_type(
        self, connection: sqlite3.Connection
    ) -> None:
        """is_weekend is 1 exactly when day_type is 'Weekend'."""
        for table in ("dim_calendar", "fact_orders"):
            assert (
                connection.execute(
                    f"SELECT COUNT(*) FROM {table} "
                    "WHERE is_weekend != (day_type = ?)",
                    (WEEKEND_DAY_TYPE,),
                ).fetchone()[0]
                == 0
            ), table

    def test_line_margin_is_net_minus_cogs(
        self, connection: sqlite3.Connection
    ) -> None:
        """line_margin equals line_net_value minus est_cogs."""
        assert (
            scalar(
                connection,
                "SELECT COUNT(*) FROM fact_order_lines "
                "WHERE ABS(line_margin - (line_net_value - est_cogs)) > 0.005",
            )
            == 0
        )


class TestMartConsistency:
    """Marts must reconcile with the fact table they were built from."""

    @pytest.mark.parametrize("mart", MART_TABLES)
    def test_mart_revenue_matches_fact_orders(
        self, connection: sqlite3.Connection, mart: str
    ) -> None:
        """Mart revenue_net totals SUM(net_before_tax) from fact_orders."""
        fact_total = scalar(
            connection, "SELECT SUM(net_before_tax) FROM fact_orders"
        )
        mart_total = scalar(connection, f"SELECT SUM(revenue_net) FROM {mart}")
        assert abs(mart_total - fact_total) < REVENUE_TOLERANCE_INR

    @pytest.mark.parametrize("mart", MART_TABLES)
    def test_mart_order_counts_match_fact_orders(
        self, connection: sqlite3.Connection, mart: str
    ) -> None:
        """Mart order counts total the number of orders."""
        assert (
            scalar(connection, f"SELECT SUM(orders) FROM {mart}")
            == EXPECTED_ROW_COUNTS["fact_orders"]
        )

    @pytest.mark.parametrize("mart", MART_TABLES)
    def test_mart_aov_is_revenue_over_orders(
        self, connection: sqlite3.Connection, mart: str
    ) -> None:
        """AOV is revenue_net divided by order count."""
        assert (
            scalar(
                connection,
                f"SELECT COUNT(*) FROM {mart} "
                "WHERE ABS(aov - revenue_net / orders) > 0.005",
            )
            == 0
        )

    def test_marts_use_net_before_tax_not_net_revenue(
        self, connection: sqlite3.Connection
    ) -> None:
        """Revenue is canonical (tax-exclusive), not the tax-inclusive figure."""
        net_revenue_total = scalar(
            connection, "SELECT SUM(net_revenue) FROM fact_orders"
        )
        mart_total = scalar(
            connection, "SELECT SUM(revenue_net) FROM mart_store_month"
        )
        assert abs(mart_total - net_revenue_total) > REVENUE_TOLERANCE_INR


class TestReferentialIntegrity:
    """Every foreign key must resolve."""

    def test_every_order_store_exists(self, connection: sqlite3.Connection) -> None:
        """fact_orders.store_id resolves to dim_store."""
        assert (
            scalar(
                connection,
                "SELECT COUNT(*) FROM fact_orders o "
                "LEFT JOIN dim_store s ON s.store_id = o.store_id "
                "WHERE s.store_id IS NULL",
            )
            == 0
        )

    def test_every_non_null_customer_exists(
        self, connection: sqlite3.Connection
    ) -> None:
        """Non-NULL customer ids resolve to dim_customer."""
        assert (
            scalar(
                connection,
                "SELECT COUNT(*) FROM fact_orders o "
                "LEFT JOIN dim_customer c ON c.customer_id = o.customer_id "
                "WHERE o.customer_id IS NOT NULL AND c.customer_id IS NULL",
            )
            == 0
        )

    def test_every_non_null_promo_exists(
        self, connection: sqlite3.Connection
    ) -> None:
        """Non-NULL promo ids resolve to dim_promotion."""
        assert (
            scalar(
                connection,
                "SELECT COUNT(*) FROM fact_orders o "
                "LEFT JOIN dim_promotion p ON p.promo_id = o.promo_id "
                "WHERE o.promo_id IS NOT NULL AND p.promo_id IS NULL",
            )
            == 0
        )

    def test_every_line_order_exists(self, connection: sqlite3.Connection) -> None:
        """fact_order_lines.order_id resolves to fact_orders."""
        assert (
            scalar(
                connection,
                "SELECT COUNT(*) FROM fact_order_lines l "
                "LEFT JOIN fact_orders o ON o.order_id = l.order_id "
                "WHERE o.order_id IS NULL",
            )
            == 0
        )

    def test_every_line_sku_exists(self, connection: sqlite3.Connection) -> None:
        """fact_order_lines.sku_id resolves to dim_product."""
        assert (
            scalar(
                connection,
                "SELECT COUNT(*) FROM fact_order_lines l "
                "LEFT JOIN dim_product p ON p.sku_id = l.sku_id "
                "WHERE p.sku_id IS NULL",
            )
            == 0
        )

    def test_every_order_date_exists_in_calendar(
        self, connection: sqlite3.Connection
    ) -> None:
        """Order dates resolve to dim_calendar, so no denormalized nulls."""
        assert (
            scalar(
                connection,
                "SELECT COUNT(*) FROM fact_orders o "
                "LEFT JOIN dim_calendar c ON c.date = o.order_date "
                "WHERE c.date IS NULL",
            )
            == 0
        )


class TestIdempotency:
    """Re-running the build must be safe and produce the same database."""

    def test_second_build_produces_identical_counts(
        self, tmp_path: Path
    ) -> None:
        """Building twice into the same file yields identical row counts."""
        path = tmp_path / "idempotency.db"

        first = build_database(db_path=path, verbose=False)
        second = build_database(db_path=path, verbose=False)

        assert first == second
        assert second["fact_orders"] == EXPECTED_ROW_COUNTS["fact_orders"]

        conn = sqlite3.connect(path)
        try:
            for table, expected in EXPECTED_ROW_COUNTS.items():
                assert scalar(conn, f"SELECT COUNT(*) FROM {table}") == expected, table
        finally:
            conn.close()
