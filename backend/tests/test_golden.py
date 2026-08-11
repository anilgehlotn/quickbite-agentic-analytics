"""Tests for the golden answers - the evaluation ground truth.

Two jobs, and the second is the important one.

First, internal consistency: totals computed independently in different
questions must agree, so a mistake in any one of them shows up as a
contradiction rather than a plausible number.

Second, the CROSS-CHECK. The golden answers are computed from the Excel workbook
with pandas; the application answers from the SQLite star schema. These tests run
the SQL equivalent of Q1, Q2, Q3, Q5 and Q6 and assert it lands within 1 INR of
the pandas figure. Two independent implementations agreeing on twelve months of
data is real evidence the ETL is faithful; either one alone proves nothing.

These tests do not exercise any agent - that comes later. They validate the
yardstick itself.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterator

import pytest

from app.config import settings

GOLDEN_PATH: Path = Path(__file__).resolve().parent / "golden_answers.json"

# Reconciliation tolerance between the pandas and SQL paths, in INR.
TOLERANCE_INR: float = 1.0

# Tolerance for derived ratios such as AOV.
TOLERANCE_RATIO: float = 0.01

# Tolerance for percentage shares that should total 100.
TOLERANCE_SHARE_PCT: float = 0.1

QUESTION_KEYS: tuple[str, ...] = ("q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8")


@pytest.fixture(scope="module")
def golden() -> dict[str, Any]:
    """Load the golden answers file.

    Returns:
        The parsed payload.
    """
    assert GOLDEN_PATH.exists(), (
        f"{GOLDEN_PATH} not found; run "
        "`python scripts/compute_golden_answers.py` first"
    )
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def connection() -> Iterator[sqlite3.Connection]:
    """Open a connection to the SQLite database for the cross-check.

    Yields:
        An open connection, closed on teardown.
    """
    conn = sqlite3.connect(settings.DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture(scope="module")
def window() -> tuple[str, str]:
    """The analysis window as ISO date strings.

    Returns:
        The inclusive start and end of the last-3-months window.
    """
    return settings.LAST_3M_START.isoformat(), settings.LAST_3M_END.isoformat()


class TestFileStructure:
    """The file exists, parses and carries every expected block."""

    def test_file_exists_and_parses(self, golden: dict[str, Any]) -> None:
        """The payload is valid JSON with content."""
        assert isinstance(golden, dict)
        assert golden

    @pytest.mark.parametrize("key", ("metadata", "context", *QUESTION_KEYS))
    def test_top_level_key_present(self, golden: dict[str, Any], key: str) -> None:
        """Every required top-level key is present."""
        assert key in golden

    def test_metadata_is_complete(self, golden: dict[str, Any]) -> None:
        """Metadata records how, when and from what the answers were computed."""
        metadata = golden["metadata"]
        for field in (
            "generated_at",
            "source_file",
            "data_asof_date",
            "analysis_window",
            "revenue_metric",
            "grain_rule",
        ):
            assert metadata.get(field), field

    def test_metadata_matches_config(self, golden: dict[str, Any]) -> None:
        """The recorded anchor and window come from the shared config."""
        metadata = golden["metadata"]
        assert metadata["data_asof_date"] == settings.DATA_ASOF_DATE.isoformat()
        assert metadata["revenue_metric"] == settings.REVENUE_METRIC
        assert (
            metadata["analysis_window"]["start"] == settings.LAST_3M_START.isoformat()
        )
        assert metadata["analysis_window"]["end"] == settings.LAST_3M_END.isoformat()

    def test_metadata_states_independence(self, golden: dict[str, Any]) -> None:
        """The file records that it was not computed from SQLite."""
        assert "pandas" in golden["metadata"]["computed_from"]
        assert golden["metadata"]["source_file"] == settings.EXCEL_PATH.name

    @pytest.mark.parametrize("key", QUESTION_KEYS)
    def test_every_question_states_its_question(
        self, golden: dict[str, Any], key: str
    ) -> None:
        """Each answer records the question it answers."""
        assert golden[key]["question"].strip()

    def test_context_has_twelve_months(self, golden: dict[str, Any]) -> None:
        """The seasonality context covers the whole dataset."""
        assert len(golden["context"]["monthly_revenue"]) == 12


class TestInternalConsistency:
    """Totals computed separately in different questions must agree."""

    def test_q1_equals_its_monthly_breakdown(self, golden: dict[str, Any]) -> None:
        """Q1's headline revenue is the sum of its own monthly rows."""
        monthly = sum(row["revenue_net_inr"] for row in golden["q1"]["monthly"])
        assert monthly == pytest.approx(
            golden["q1"]["revenue_net_inr"], abs=TOLERANCE_INR
        )

    def test_q1_orders_equal_its_monthly_breakdown(
        self, golden: dict[str, Any]
    ) -> None:
        """Q1's order count is the sum of its own monthly rows."""
        assert sum(row["orders"] for row in golden["q1"]["monthly"]) == (
            golden["q1"]["orders"]
        )

    def test_q1_aov_is_revenue_over_orders(self, golden: dict[str, Any]) -> None:
        """AOV is internally consistent with revenue and orders."""
        q1 = golden["q1"]
        assert q1["aov_inr"] == pytest.approx(
            q1["revenue_net_inr"] / q1["orders"], abs=TOLERANCE_RATIO
        )

    def test_q1_tax_relationship(self, golden: dict[str, Any]) -> None:
        """Tax-inclusive revenue is the net figure grossed up by the tax rate."""
        q1 = golden["q1"]
        assert q1["revenue_with_tax_inr"] == pytest.approx(
            q1["revenue_net_inr"] * (1 + settings.TAX_RATE), abs=TOLERANCE_INR
        )

    def test_q3_channels_sum_to_q1(self, golden: dict[str, Any]) -> None:
        """Channel revenue totals the headline revenue."""
        channels = sum(row["revenue_net_inr"] for row in golden["q3"]["channels"])
        assert channels == pytest.approx(
            golden["q1"]["revenue_net_inr"], abs=TOLERANCE_INR
        )

    def test_q3_orders_sum_to_q1(self, golden: dict[str, Any]) -> None:
        """Channel order counts total the headline order count."""
        assert sum(row["orders"] for row in golden["q3"]["channels"]) == (
            golden["q1"]["orders"]
        )

    def test_q3_shares_sum_to_100(self, golden: dict[str, Any]) -> None:
        """Revenue shares are a partition of the whole."""
        shares = sum(row["revenue_share_pct"] for row in golden["q3"]["channels"])
        assert shares == pytest.approx(100.0, abs=TOLERANCE_SHARE_PCT)

    def test_q3_channels_match_config(self, golden: dict[str, Any]) -> None:
        """Every channel is a configured channel."""
        channels = {row["CHANNEL"] for row in golden["q3"]["channels"]}
        assert channels == set(settings.CHANNELS)

    def test_q2_all_stores_sum_to_q1(self, golden: dict[str, Any]) -> None:
        """The full store list totals the headline revenue."""
        stores = sum(row["revenue_net_inr"] for row in golden["q2"]["all_stores"])
        assert stores == pytest.approx(
            golden["q1"]["revenue_net_inr"], abs=TOLERANCE_INR
        )

    def test_q2_shares_sum_to_100(self, golden: dict[str, Any]) -> None:
        """Store revenue shares partition the whole."""
        shares = sum(row["revenue_share_pct"] for row in golden["q2"]["all_stores"])
        assert shares == pytest.approx(100.0, abs=TOLERANCE_SHARE_PCT)

    def test_q2_top_beats_bottom(self, golden: dict[str, Any]) -> None:
        """Every top-5 store out-earns every bottom-5 store."""
        worst_top = min(row["revenue_net_inr"] for row in golden["q2"]["top_5"])
        best_bottom = max(row["revenue_net_inr"] for row in golden["q2"]["bottom_5"])
        assert worst_top >= best_bottom

    def test_q2_ranks_are_consistent(self, golden: dict[str, Any]) -> None:
        """Ranks run 1..N in descending revenue order."""
        stores = golden["q2"]["all_stores"]
        assert [row["rank"] for row in stores] == list(range(1, len(stores) + 1))
        revenues = [row["revenue_net_inr"] for row in stores]
        assert revenues == sorted(revenues, reverse=True)

    def test_q5_cities_sum_to_q1(self, golden: dict[str, Any]) -> None:
        """City revenue across the three months totals the headline revenue."""
        total = sum(
            value
            for row in golden["q5"]["cities"]
            for value in row["monthly_revenue_inr"].values()
        )
        assert total == pytest.approx(
            golden["q1"]["revenue_net_inr"], abs=TOLERANCE_INR
        )

    def test_q6_weekend_plus_weekday_equals_q1(self, golden: dict[str, Any]) -> None:
        """Weekend and weekday revenue partition the window."""
        total = sum(
            row["revenue_net_inr"]
            for row in golden["q6"]["last_3_months"]["day_types"]
        )
        assert total == pytest.approx(
            golden["q1"]["revenue_net_inr"], abs=TOLERANCE_INR
        )

    def test_q6_full_year_equals_context(self, golden: dict[str, Any]) -> None:
        """Full-year weekend plus weekday equals the full-year total."""
        total = sum(
            row["revenue_net_inr"] for row in golden["q6"]["full_year"]["day_types"]
        )
        assert total == pytest.approx(
            golden["context"]["full_year"]["revenue_net_inr"], abs=TOLERANCE_INR
        )

    def test_q6_days_partition_the_window(self, golden: dict[str, Any]) -> None:
        """Weekend and weekday days total the days in the year."""
        days = sum(
            row["distinct_days"] for row in golden["q6"]["full_year"]["day_types"]
        )
        assert days == 365

    def test_q7_festive_plus_normal_equals_full_year(
        self, golden: dict[str, Any]
    ) -> None:
        """Festive and normal revenue partition the year."""
        total = (
            golden["q7"]["normal"]["revenue_net_inr"]
            + golden["q7"]["all_festive_combined"]["revenue_net_inr"]
        )
        assert total == pytest.approx(
            golden["context"]["full_year"]["revenue_net_inr"], abs=TOLERANCE_INR
        )

    def test_q7_individual_periods_sum_to_combined(
        self, golden: dict[str, Any]
    ) -> None:
        """The named festive periods total the combined figure."""
        total = sum(row["revenue_net_inr"] for row in golden["q7"]["festive_periods"])
        assert total == pytest.approx(
            golden["q7"]["all_festive_combined"]["revenue_net_inr"], abs=TOLERANCE_INR
        )

    def test_q7_periods_match_config(self, golden: dict[str, Any]) -> None:
        """The festive periods are the configured ones."""
        periods = [row["period"] for row in golden["q7"]["festive_periods"]]
        assert periods == settings.FESTIVE_PERIODS

    def test_q8_stores_sum_to_q1(self, golden: dict[str, Any]) -> None:
        """Store revenue across the window totals the headline revenue."""
        total = sum(row["window_revenue_inr"] for row in golden["q8"]["stores"])
        assert total == pytest.approx(
            golden["q1"]["revenue_net_inr"], abs=TOLERANCE_INR
        )

    def test_q4_is_labelled_as_line_grain(self, golden: dict[str, Any]) -> None:
        """Q4 declares its grain and explains the variance."""
        q4 = golden["q4"]
        assert q4["grain"] == "line"
        assert "LINE GRAIN" in q4["note"]
        assert q4["order_revenue_total_inr"] == pytest.approx(
            golden["q1"]["revenue_net_inr"], abs=TOLERANCE_INR
        )

    def test_order_grain_questions_are_labelled(self, golden: dict[str, Any]) -> None:
        """Every question except Q4 is computed at order grain."""
        for key in ("q1", "q2", "q3", "q5", "q6", "q7", "q8"):
            assert golden[key]["grain"] == "order", key


class TestDeclineFlags:
    """A flagged decline must be a real decline in the underlying numbers."""

    def test_flagged_cities_are_strictly_decreasing(
        self, golden: dict[str, Any]
    ) -> None:
        """Every declining city really falls in every consecutive month."""
        months = golden["q5"]["months"]
        for city in golden["q5"]["declining_cities"]:
            values = [city["monthly_revenue_inr"][month] for month in months]
            assert all(
                values[i] > values[i + 1] for i in range(len(values) - 1)
            ), city["city"]

    def test_unflagged_cities_are_not_strictly_decreasing(
        self, golden: dict[str, Any]
    ) -> None:
        """No genuine decline was missed."""
        months = golden["q5"]["months"]
        flagged = {city["city"] for city in golden["q5"]["declining_cities"]}
        for city in golden["q5"]["cities"]:
            if city["city"] in flagged:
                continue
            values = [city["monthly_revenue_inr"][month] for month in months]
            assert not all(
                values[i] > values[i + 1] for i in range(len(values) - 1)
            ), city["city"]

    def test_flagged_stores_are_strictly_decreasing(
        self, golden: dict[str, Any]
    ) -> None:
        """Every declining store really falls in every consecutive month."""
        months = golden["q8"]["months"]
        for store in golden["q8"]["declining_stores"]:
            values = [store["monthly_revenue_inr"][month] for month in months]
            assert all(
                values[i] > values[i + 1] for i in range(len(values) - 1)
            ), store["store_id"]

    def test_unflagged_stores_are_not_strictly_decreasing(
        self, golden: dict[str, Any]
    ) -> None:
        """No genuine store decline was missed."""
        months = golden["q8"]["months"]
        flagged = {store["store_id"] for store in golden["q8"]["declining_stores"]}
        for store in golden["q8"]["stores"]:
            if store["store_id"] in flagged:
                continue
            values = [store["monthly_revenue_inr"][month] for month in months]
            assert not all(
                values[i] > values[i + 1] for i in range(len(values) - 1)
            ), store["store_id"]

    def test_declining_store_count_matches_the_list(
        self, golden: dict[str, Any]
    ) -> None:
        """The reported count matches the list length."""
        assert golden["q8"]["declining_store_count"] == len(
            golden["q8"]["declining_stores"]
        )

    def test_every_declining_store_has_diagnostics(
        self, golden: dict[str, Any]
    ) -> None:
        """Each flagged store carries the evidence needed to explain it."""
        months = golden["q8"]["months"]
        for store in golden["q8"]["declining_stores"]:
            diagnostics = store["diagnostics"]
            assert len(diagnostics["monthly"]) == len(months)
            assert diagnostics["by_channel"]
            assert diagnostics["decomposition"]["primary_driver"] in (
                "order volume",
                "basket size",
            )
            assert diagnostics["city_trend"]["monthly_revenue_inr"]
            assert diagnostics["store_specific_vs_market_wide"] in (
                "store-specific",
                "market-wide",
            )

    def test_diagnostic_monthly_revenue_matches_the_store_row(
        self, golden: dict[str, Any]
    ) -> None:
        """Diagnostics are computed for the same store and window."""
        months = golden["q8"]["months"]
        for store in golden["q8"]["declining_stores"]:
            for index, month in enumerate(months):
                assert store["diagnostics"]["monthly"][index][
                    "revenue_net_inr"
                ] == pytest.approx(
                    store["monthly_revenue_inr"][month], abs=TOLERANCE_INR
                )

    def test_diagnostic_channels_sum_to_store_revenue(
        self, golden: dict[str, Any]
    ) -> None:
        """A store's channel breakdown totals its own revenue."""
        for store in golden["q8"]["declining_stores"]:
            channel_total = sum(
                value
                for channel in store["diagnostics"]["by_channel"]
                for value in channel["monthly_revenue_inr"].values()
            )
            assert channel_total == pytest.approx(
                store["window_revenue_inr"], abs=TOLERANCE_INR
            )


class TestSqliteCrossCheck:
    """The pandas ground truth and the SQLite star schema must agree.

    This is the test that proves the ETL is faithful. The golden answers never
    read SQLite and the ETL never reads the golden answers, so agreement here is
    two independent implementations landing on the same numbers.
    """

    def test_q1_revenue_matches_sql(
        self,
        golden: dict[str, Any],
        connection: sqlite3.Connection,
        window: tuple[str, str],
    ) -> None:
        """Headline revenue agrees between pandas and SQL."""
        revenue = connection.execute(
            "SELECT SUM(net_before_tax) FROM fact_orders "
            "WHERE order_date BETWEEN ? AND ?",
            window,
        ).fetchone()[0]
        assert revenue == pytest.approx(
            golden["q1"]["revenue_net_inr"], abs=TOLERANCE_INR
        )

    def test_q1_orders_and_units_match_sql(
        self,
        golden: dict[str, Any],
        connection: sqlite3.Connection,
        window: tuple[str, str],
    ) -> None:
        """Order count and units agree between pandas and SQL."""
        orders, units = connection.execute(
            "SELECT COUNT(DISTINCT order_id), SUM(total_qty) FROM fact_orders "
            "WHERE order_date BETWEEN ? AND ?",
            window,
        ).fetchone()
        assert orders == golden["q1"]["orders"]
        assert units == golden["q1"]["units"]

    def test_q1_monthly_matches_sql(
        self,
        golden: dict[str, Any],
        connection: sqlite3.Connection,
        window: tuple[str, str],
    ) -> None:
        """Each month's revenue agrees between pandas and SQL."""
        rows = dict(
            connection.execute(
                "SELECT month_key, SUM(net_before_tax) FROM fact_orders "
                "WHERE order_date BETWEEN ? AND ? GROUP BY month_key",
                window,
            ).fetchall()
        )
        for month in golden["q1"]["monthly"]:
            assert rows[month["MONTH_KEY"]] == pytest.approx(
                month["revenue_net_inr"], abs=TOLERANCE_INR
            ), month["MONTH_KEY"]

    def test_q2_store_revenue_matches_sql(
        self,
        golden: dict[str, Any],
        connection: sqlite3.Connection,
        window: tuple[str, str],
    ) -> None:
        """Every store's revenue agrees between pandas and SQL."""
        rows = dict(
            connection.execute(
                "SELECT store_id, SUM(net_before_tax) FROM fact_orders "
                "WHERE order_date BETWEEN ? AND ? GROUP BY store_id",
                window,
            ).fetchall()
        )
        assert len(rows) == golden["q2"]["store_count"]
        for store in golden["q2"]["all_stores"]:
            assert rows[store["STORE_ID"]] == pytest.approx(
                store["revenue_net_inr"], abs=TOLERANCE_INR
            ), store["STORE_ID"]

    def test_q2_ranking_matches_sql(
        self,
        golden: dict[str, Any],
        connection: sqlite3.Connection,
        window: tuple[str, str],
    ) -> None:
        """The top five stores are the same five, in the same order."""
        rows = connection.execute(
            "SELECT store_id FROM fact_orders "
            "WHERE order_date BETWEEN ? AND ? "
            "GROUP BY store_id ORDER BY SUM(net_before_tax) DESC LIMIT 5",
            window,
        ).fetchall()
        assert [row[0] for row in rows] == [
            store["STORE_ID"] for store in golden["q2"]["top_5"]
        ]

    def test_q3_channel_revenue_matches_sql(
        self,
        golden: dict[str, Any],
        connection: sqlite3.Connection,
        window: tuple[str, str],
    ) -> None:
        """Every channel's revenue agrees between pandas and SQL."""
        rows = dict(
            connection.execute(
                "SELECT channel, SUM(net_before_tax) FROM fact_orders "
                "WHERE order_date BETWEEN ? AND ? GROUP BY channel",
                window,
            ).fetchall()
        )
        for channel in golden["q3"]["channels"]:
            assert rows[channel["CHANNEL"]] == pytest.approx(
                channel["revenue_net_inr"], abs=TOLERANCE_INR
            ), channel["CHANNEL"]

    def test_q3_matches_the_channel_mart(
        self, golden: dict[str, Any], connection: sqlite3.Connection
    ) -> None:
        """The pre-aggregated mart also agrees with the pandas ground truth."""
        months = golden["metadata"]["analysis_window"]["months"]
        placeholders = ", ".join("?" for _ in months)
        rows = dict(
            connection.execute(
                f"SELECT channel, SUM(revenue_net) FROM mart_channel_month "
                f"WHERE month_key IN ({placeholders}) GROUP BY channel",
                months,
            ).fetchall()
        )
        for channel in golden["q3"]["channels"]:
            assert rows[channel["CHANNEL"]] == pytest.approx(
                channel["revenue_net_inr"], abs=TOLERANCE_INR
            ), channel["CHANNEL"]

    def test_q5_city_months_match_sql(
        self,
        golden: dict[str, Any],
        connection: sqlite3.Connection,
        window: tuple[str, str],
    ) -> None:
        """Every city-month revenue agrees between pandas and SQL."""
        rows = {
            (city, month): revenue
            for city, month, revenue in connection.execute(
                "SELECT s.city, o.month_key, SUM(o.net_before_tax) "
                "FROM fact_orders AS o "
                "JOIN dim_store AS s ON s.store_id = o.store_id "
                "WHERE o.order_date BETWEEN ? AND ? "
                "GROUP BY s.city, o.month_key",
                window,
            ).fetchall()
        }
        for city in golden["q5"]["cities"]:
            for month, revenue in city["monthly_revenue_inr"].items():
                assert rows[(city["city"], month)] == pytest.approx(
                    revenue, abs=TOLERANCE_INR
                ), f"{city['city']} {month}"

    def test_q5_matches_the_city_mart(
        self, golden: dict[str, Any], connection: sqlite3.Connection
    ) -> None:
        """The pre-aggregated city mart also agrees."""
        rows = {
            (city, month): revenue
            for city, month, revenue in connection.execute(
                "SELECT city, month_key, revenue_net FROM mart_city_month"
            ).fetchall()
        }
        for city in golden["q5"]["cities"]:
            for month, revenue in city["monthly_revenue_inr"].items():
                assert rows[(city["city"], month)] == pytest.approx(
                    revenue, abs=TOLERANCE_INR
                ), f"{city['city']} {month}"

    def test_q6_day_type_matches_sql(
        self,
        golden: dict[str, Any],
        connection: sqlite3.Connection,
        window: tuple[str, str],
    ) -> None:
        """Weekend and weekday revenue agree between pandas and SQL."""
        rows = dict(
            connection.execute(
                "SELECT day_type, SUM(net_before_tax) FROM fact_orders "
                "WHERE order_date BETWEEN ? AND ? GROUP BY day_type",
                window,
            ).fetchall()
        )
        for day_type in golden["q6"]["last_3_months"]["day_types"]:
            assert rows[day_type["DAY_TYPE"]] == pytest.approx(
                day_type["revenue_net_inr"], abs=TOLERANCE_INR
            ), day_type["DAY_TYPE"]

    def test_q6_full_year_day_counts_match_sql(
        self, golden: dict[str, Any], connection: sqlite3.Connection
    ) -> None:
        """The distinct trading-day counts agree with dim_calendar."""
        rows = dict(
            connection.execute(
                "SELECT day_type, COUNT(*) FROM dim_calendar GROUP BY day_type"
            ).fetchall()
        )
        for day_type in golden["q6"]["full_year"]["day_types"]:
            assert rows[day_type["DAY_TYPE"]] == day_type["distinct_days"]

    def test_q8_declining_stores_match_sql(
        self, golden: dict[str, Any], connection: sqlite3.Connection
    ) -> None:
        """SQL independently identifies the same declining stores."""
        months = golden["q8"]["months"]
        rows = {
            (store, month): revenue
            for store, month, revenue in connection.execute(
                "SELECT store_id, month_key, revenue_net FROM mart_store_month"
            ).fetchall()
        }
        declining = {
            store
            for store in {key[0] for key in rows}
            if all(
                rows.get((store, months[i]), 0.0) > rows.get((store, months[i + 1]), 0.0)
                for i in range(len(months) - 1)
            )
        }
        assert declining == {
            store["store_id"] for store in golden["q8"]["declining_stores"]
        }

    def test_full_year_total_matches_sql(
        self, golden: dict[str, Any], connection: sqlite3.Connection
    ) -> None:
        """The full-year revenue agrees, covering all twelve months at once."""
        revenue = connection.execute(
            "SELECT SUM(net_before_tax) FROM fact_orders"
        ).fetchone()[0]
        assert revenue == pytest.approx(
            golden["context"]["full_year"]["revenue_net_inr"], abs=TOLERANCE_INR
        )

    def test_every_month_matches_sql(
        self, golden: dict[str, Any], connection: sqlite3.Connection
    ) -> None:
        """All twelve months agree, not just the analysis window."""
        rows = dict(
            connection.execute(
                "SELECT month_key, SUM(net_before_tax) FROM fact_orders "
                "GROUP BY month_key"
            ).fetchall()
        )
        for month in golden["context"]["monthly_revenue"]:
            assert rows[month["MONTH_KEY"]] == pytest.approx(
                month["revenue_net_inr"], abs=TOLERANCE_INR
            ), month["MONTH_KEY"]
