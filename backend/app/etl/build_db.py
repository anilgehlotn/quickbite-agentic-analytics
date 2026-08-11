"""Build the QuickBite SQLite star schema from the source Excel workbook.

Reads ``settings.EXCEL_PATH`` and writes a star schema to ``settings.DB_PATH``:
five dimension tables, two fact tables and three precomputed monthly aggregate
marts. Every table is dropped and recreated on each run, so the module is fully
idempotent and safe to re-run.

Two shape decisions are deliberate:

* ``fact_orders`` denormalizes the calendar attributes (month key, day name,
  day type, weekend and festive flags) that would otherwise require a join to
  ``dim_calendar``. Agent-generated SQL is simpler and less error-prone when the
  common filters live on the fact table.
* Revenue in the marts is ``net_before_tax`` (``settings.REVENUE_METRIC``), the
  canonical revenue measure, which excludes the 5% tax carried in
  ``net_revenue``.

Anonymous walk-in orders carry a NULL ``customer_id`` and orders without a
promotion carry a NULL ``promo_id``. Both are normal states in this dataset and
are preserved as NULL; no row is ever dropped.

Run with::

    python -m app.etl.build_db
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Final

import pandas as pd

from app.config import settings

# --- Source sheet names ----------------------------------------------------
SHEET_STORE_MASTER: Final[str] = "Store_Master"
SHEET_PRODUCT_MASTER: Final[str] = "Product_Master"
SHEET_CUSTOMER_MASTER: Final[str] = "Customer_Master"
SHEET_PROMOTIONS: Final[str] = "Promotions"
SHEET_CALENDAR: Final[str] = "Calendar"
SHEET_ORDERS: Final[str] = "Orders"
SHEET_ORDER_DETAILS: Final[str] = "Order_Details"

# --- Domain vocabulary -----------------------------------------------------
# Value of dim_calendar.day_type that marks a non-working day. Anything else is
# a weekday.
WEEKEND_DAY_TYPE: Final[str] = "Weekend"

# --- Formatting ------------------------------------------------------------
BYTES_PER_MB: Final[int] = 1024 * 1024
INSERT_CHUNK_SIZE: Final[int] = 5_000

# Tables in creation order; also the order used for the console summary.
# Children precede parents when dropping so foreign keys never block a drop.
TABLE_NAMES: Final[tuple[str, ...]] = (
    "dim_store",
    "dim_product",
    "dim_customer",
    "dim_promotion",
    "dim_calendar",
    "fact_orders",
    "fact_order_lines",
    "mart_store_month",
    "mart_city_month",
    "mart_channel_month",
)

# --- DDL -------------------------------------------------------------------
# Dates are ISO TEXT (SQLite convention), money is REAL, counts are INTEGER.

DDL_STATEMENTS: Final[dict[str, str]] = {
    "dim_store": """
        CREATE TABLE dim_store (
            store_id           TEXT PRIMARY KEY,
            store_name         TEXT NOT NULL,
            city               TEXT NOT NULL,
            state              TEXT NOT NULL,
            region             TEXT NOT NULL,
            store_format       TEXT NOT NULL,
            opening_date       TEXT NOT NULL,
            city_price_index   REAL NOT NULL,
            performance_factor REAL NOT NULL,
            status             TEXT NOT NULL
        )
    """,
    "dim_product": """
        CREATE TABLE dim_product (
            sku_id         TEXT PRIMARY KEY,
            sku_name       TEXT NOT NULL,
            category       TEXT NOT NULL,
            veg_nonveg     TEXT NOT NULL,
            base_price_inr REAL NOT NULL,
            est_cogs_pct   REAL NOT NULL,
            status         TEXT NOT NULL
        )
    """,
    "dim_customer": """
        CREATE TABLE dim_customer (
            customer_id      TEXT PRIMARY KEY,
            home_city        TEXT NOT NULL,
            customer_segment TEXT NOT NULL,
            join_date        TEXT NOT NULL
        )
    """,
    "dim_promotion": """
        CREATE TABLE dim_promotion (
            promo_id          TEXT PRIMARY KEY,
            promo_name        TEXT NOT NULL,
            promo_type        TEXT NOT NULL,
            start_date        TEXT NOT NULL,
            end_date          TEXT NOT NULL,
            applicable_days   TEXT,
            applicability     TEXT,
            discount_pct      REAL NOT NULL,
            min_bill_value    REAL NOT NULL,
            max_discount_inr  REAL NOT NULL
        )
    """,
    "dim_calendar": """
        CREATE TABLE dim_calendar (
            date           TEXT PRIMARY KEY,
            year           INTEGER NOT NULL,
            month          TEXT NOT NULL,
            month_no       INTEGER NOT NULL,
            day_name       TEXT NOT NULL,
            day_type       TEXT NOT NULL,
            festive_period TEXT NOT NULL,
            -- Derived for convenience in agent-generated SQL.
            month_key      TEXT NOT NULL,   -- 'YYYY-MM'
            is_weekend     INTEGER NOT NULL,
            is_festive     INTEGER NOT NULL
        )
    """,
    "fact_orders": """
        CREATE TABLE fact_orders (
            order_id         TEXT PRIMARY KEY,
            order_datetime   TEXT NOT NULL,
            order_date       TEXT NOT NULL,
            order_hour       INTEGER NOT NULL,
            store_id         TEXT NOT NULL,
            customer_id      TEXT,            -- NULL for anonymous walk-ins
            channel          TEXT NOT NULL,
            promo_id         TEXT,            -- NULL when no promotion applied
            total_qty        INTEGER NOT NULL,
            gross_bill_value REAL NOT NULL,
            discount_amount  REAL NOT NULL,
            net_before_tax   REAL NOT NULL,   -- canonical revenue (excl. tax)
            tax_amount       REAL NOT NULL,
            net_revenue      REAL NOT NULL,   -- includes tax
            -- Denormalized from dim_calendar on order_date.
            month_key        TEXT NOT NULL,
            day_name         TEXT NOT NULL,
            day_type         TEXT NOT NULL,
            is_weekend       INTEGER NOT NULL,
            festive_period   TEXT NOT NULL,
            is_festive       INTEGER NOT NULL,
            FOREIGN KEY (store_id)    REFERENCES dim_store (store_id),
            FOREIGN KEY (customer_id) REFERENCES dim_customer (customer_id),
            FOREIGN KEY (promo_id)    REFERENCES dim_promotion (promo_id),
            FOREIGN KEY (order_date)  REFERENCES dim_calendar (date)
        )
    """,
    "fact_order_lines": """
        CREATE TABLE fact_order_lines (
            order_detail_id  TEXT PRIMARY KEY,
            order_id         TEXT NOT NULL,
            sku_id           TEXT NOT NULL,
            quantity         INTEGER NOT NULL,
            unit_price       REAL NOT NULL,
            line_gross_value REAL NOT NULL,
            line_discount    REAL NOT NULL,
            line_net_value   REAL NOT NULL,
            est_cogs         REAL NOT NULL,
            line_margin      REAL NOT NULL,   -- line_net_value - est_cogs
            FOREIGN KEY (order_id) REFERENCES fact_orders (order_id),
            FOREIGN KEY (sku_id)   REFERENCES dim_product (sku_id)
        )
    """,
    "mart_store_month": """
        CREATE TABLE mart_store_month (
            store_id      TEXT NOT NULL,
            store_name    TEXT NOT NULL,
            city          TEXT NOT NULL,
            region        TEXT NOT NULL,
            month_key     TEXT NOT NULL,
            orders        INTEGER NOT NULL,
            revenue_net   REAL NOT NULL,
            revenue_gross REAL NOT NULL,
            units         INTEGER NOT NULL,
            aov           REAL NOT NULL,
            PRIMARY KEY (store_id, month_key)
        )
    """,
    "mart_city_month": """
        CREATE TABLE mart_city_month (
            city          TEXT NOT NULL,
            month_key     TEXT NOT NULL,
            orders        INTEGER NOT NULL,
            revenue_net   REAL NOT NULL,
            revenue_gross REAL NOT NULL,
            units         INTEGER NOT NULL,
            aov           REAL NOT NULL,
            PRIMARY KEY (city, month_key)
        )
    """,
    "mart_channel_month": """
        CREATE TABLE mart_channel_month (
            channel       TEXT NOT NULL,
            month_key     TEXT NOT NULL,
            orders        INTEGER NOT NULL,
            revenue_net   REAL NOT NULL,
            revenue_gross REAL NOT NULL,
            units         INTEGER NOT NULL,
            aov           REAL NOT NULL,
            PRIMARY KEY (channel, month_key)
        )
    """,
}

INDEX_STATEMENTS: Final[tuple[str, ...]] = (
    "CREATE INDEX idx_fact_orders_order_date ON fact_orders (order_date)",
    "CREATE INDEX idx_fact_orders_store_id ON fact_orders (store_id)",
    "CREATE INDEX idx_fact_orders_channel ON fact_orders (channel)",
    "CREATE INDEX idx_fact_orders_month_key ON fact_orders (month_key)",
    "CREATE INDEX idx_fact_orders_is_weekend ON fact_orders (is_weekend)",
    "CREATE INDEX idx_fact_orders_festive_period ON fact_orders (festive_period)",
    "CREATE INDEX idx_fact_order_lines_order_id ON fact_order_lines (order_id)",
    "CREATE INDEX idx_fact_order_lines_sku_id ON fact_order_lines (sku_id)",
    "CREATE INDEX idx_mart_store_month_month_key ON mart_store_month (month_key)",
    "CREATE INDEX idx_mart_city_month_month_key ON mart_city_month (month_key)",
    "CREATE INDEX idx_mart_channel_month_month_key ON mart_channel_month (month_key)",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def read_sheet(excel_path: Path, sheet_name: str) -> pd.DataFrame:
    """Read one sheet from the source workbook with snake_case column names.

    Args:
        excel_path: Path to the source Excel workbook.
        sheet_name: Name of the sheet to read.

    Returns:
        The sheet's contents with columns lowercased (source headers are
        already underscore-separated upper case, e.g. ``STORE_ID``).
    """
    frame = pd.read_excel(excel_path, sheet_name=sheet_name)
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    return frame


def to_iso_date(values: pd.Series) -> pd.Series:
    """Convert a datetime column to ISO date strings (``YYYY-MM-DD``).

    Args:
        values: Column of datetime-like values.

    Returns:
        The same column as ``YYYY-MM-DD`` strings, with nulls preserved.
    """
    return pd.to_datetime(values).dt.strftime("%Y-%m-%d")


def to_iso_datetime(values: pd.Series) -> pd.Series:
    """Convert a datetime column to ISO timestamp strings.

    Args:
        values: Column of datetime-like values.

    Returns:
        The same column as ``YYYY-MM-DD HH:MM:SS`` strings, nulls preserved.
    """
    return pd.to_datetime(values).dt.strftime("%Y-%m-%d %H:%M:%S")


def nulls_to_none(values: pd.Series) -> pd.Series:
    """Replace pandas nulls with ``None`` so SQLite stores real NULLs.

    Args:
        values: Column that may contain NaN/NaT values.

    Returns:
        The column with every null replaced by ``None``.
    """
    return values.astype(object).where(values.notna(), None)


def load_table(connection: sqlite3.Connection, table: str, frame: pd.DataFrame) -> int:
    """Insert a prepared DataFrame into an existing table.

    The table must already exist; ``append`` is used so the explicit DDL types,
    primary keys and foreign keys defined in this module are preserved.

    Args:
        connection: Open SQLite connection.
        table: Destination table name.
        frame: Rows to insert, with columns matching the table definition.

    Returns:
        The number of rows inserted.
    """
    frame.to_sql(
        table,
        connection,
        if_exists="append",
        index=False,
        chunksize=INSERT_CHUNK_SIZE,
        method="multi",
    )
    return len(frame)


def create_schema(connection: sqlite3.Connection) -> None:
    """Drop and recreate every table, making the build idempotent.

    Tables are dropped children-first so foreign key references never block a
    drop, then recreated in dependency order.

    Args:
        connection: Open SQLite connection.
    """
    for table in reversed(TABLE_NAMES):
        connection.execute(f"DROP TABLE IF EXISTS {table}")
    for table in TABLE_NAMES:
        connection.execute(DDL_STATEMENTS[table])
    connection.commit()


def create_indexes(connection: sqlite3.Connection) -> None:
    """Create the query indexes on the fact tables and marts.

    Args:
        connection: Open SQLite connection.
    """
    for statement in INDEX_STATEMENTS:
        connection.execute(statement)
    connection.commit()


# ---------------------------------------------------------------------------
# Dimension builders
# ---------------------------------------------------------------------------


def build_dim_store(connection: sqlite3.Connection, excel_path: Path) -> int:
    """Load Store_Master into ``dim_store``.

    Args:
        connection: Open SQLite connection.
        excel_path: Path to the source workbook.

    Returns:
        Number of rows loaded.
    """
    frame = read_sheet(excel_path, SHEET_STORE_MASTER)
    frame["opening_date"] = to_iso_date(frame["opening_date"])
    return load_table(connection, "dim_store", frame)


def build_dim_product(connection: sqlite3.Connection, excel_path: Path) -> int:
    """Load Product_Master into ``dim_product``.

    Args:
        connection: Open SQLite connection.
        excel_path: Path to the source workbook.

    Returns:
        Number of rows loaded.
    """
    frame = read_sheet(excel_path, SHEET_PRODUCT_MASTER)
    return load_table(connection, "dim_product", frame)


def build_dim_customer(connection: sqlite3.Connection, excel_path: Path) -> int:
    """Load Customer_Master into ``dim_customer``.

    Args:
        connection: Open SQLite connection.
        excel_path: Path to the source workbook.

    Returns:
        Number of rows loaded.
    """
    frame = read_sheet(excel_path, SHEET_CUSTOMER_MASTER)
    frame["join_date"] = to_iso_date(frame["join_date"])
    return load_table(connection, "dim_customer", frame)


def build_dim_promotion(connection: sqlite3.Connection, excel_path: Path) -> int:
    """Load Promotions into ``dim_promotion``.

    Args:
        connection: Open SQLite connection.
        excel_path: Path to the source workbook.

    Returns:
        Number of rows loaded.
    """
    frame = read_sheet(excel_path, SHEET_PROMOTIONS)
    frame["start_date"] = to_iso_date(frame["start_date"])
    frame["end_date"] = to_iso_date(frame["end_date"])
    return load_table(connection, "dim_promotion", frame)


def build_dim_calendar(connection: sqlite3.Connection, excel_path: Path) -> int:
    """Load Calendar into ``dim_calendar`` with derived convenience columns.

    Adds ``month_key`` (``YYYY-MM``), ``is_weekend`` and ``is_festive``. The
    festive flag is derived from ``settings.FESTIVE_PERIODS`` rather than from a
    literal, so the config file stays the single source of truth.

    Args:
        connection: Open SQLite connection.
        excel_path: Path to the source workbook.

    Returns:
        Number of rows loaded.
    """
    frame = read_sheet(excel_path, SHEET_CALENDAR)
    frame["date"] = to_iso_date(frame["date"])
    frame["month_key"] = frame["date"].str.slice(0, len("YYYY-MM"))
    frame["is_weekend"] = (frame["day_type"] == WEEKEND_DAY_TYPE).astype(int)
    frame["is_festive"] = (
        frame["festive_period"].isin(settings.FESTIVE_PERIODS).astype(int)
    )
    return load_table(connection, "dim_calendar", frame)


# ---------------------------------------------------------------------------
# Fact builders
# ---------------------------------------------------------------------------


def build_fact_orders(connection: sqlite3.Connection, excel_path: Path) -> int:
    """Load Orders into ``fact_orders`` with calendar attributes denormalized.

    Derives ``order_date`` and ``order_hour`` from ``order_datetime``, then
    joins ``dim_calendar`` on the date to carry the month key, day name, day
    type, weekend flag, festive period and festive flag onto the fact row.

    NULL ``customer_id`` (anonymous walk-in) and NULL ``promo_id`` (no
    promotion) are preserved; no rows are filtered out.

    Args:
        connection: Open SQLite connection.
        excel_path: Path to the source workbook.

    Returns:
        Number of rows loaded.
    """
    frame = read_sheet(excel_path, SHEET_ORDERS)
    order_datetime = pd.to_datetime(frame["order_datetime"])
    frame["order_datetime"] = to_iso_datetime(order_datetime)
    frame["order_date"] = order_datetime.dt.strftime("%Y-%m-%d")
    frame["order_hour"] = order_datetime.dt.hour.astype(int)

    # Nullable foreign keys must reach SQLite as real NULLs.
    frame["customer_id"] = nulls_to_none(frame["customer_id"])
    frame["promo_id"] = nulls_to_none(frame["promo_id"])

    calendar = pd.read_sql_query(
        "SELECT date, month_key, day_name, day_type, is_weekend, "
        "festive_period, is_festive FROM dim_calendar",
        connection,
    )
    # Left join keeps every order even if a date were missing from the
    # calendar; the NOT NULL constraints would then surface the gap loudly.
    frame = frame.merge(
        calendar, how="left", left_on="order_date", right_on="date"
    ).drop(columns=["date"])

    columns = [
        "order_id",
        "order_datetime",
        "order_date",
        "order_hour",
        "store_id",
        "customer_id",
        "channel",
        "promo_id",
        "total_qty",
        "gross_bill_value",
        "discount_amount",
        "net_before_tax",
        "tax_amount",
        "net_revenue",
        "month_key",
        "day_name",
        "day_type",
        "is_weekend",
        "festive_period",
        "is_festive",
    ]
    return load_table(connection, "fact_orders", frame[columns])


def build_fact_order_lines(connection: sqlite3.Connection, excel_path: Path) -> int:
    """Load Order_Details into ``fact_order_lines`` with line margin derived.

    Args:
        connection: Open SQLite connection.
        excel_path: Path to the source workbook.

    Returns:
        Number of rows loaded.
    """
    frame = read_sheet(excel_path, SHEET_ORDER_DETAILS)
    frame["line_margin"] = frame["line_net_value"] - frame["est_cogs"]
    return load_table(connection, "fact_order_lines", frame)


# ---------------------------------------------------------------------------
# Aggregate marts
# ---------------------------------------------------------------------------


def build_mart_store_month(connection: sqlite3.Connection) -> int:
    """Build the store-by-month aggregate mart from ``fact_orders``.

    Args:
        connection: Open SQLite connection.

    Returns:
        Number of rows written.
    """
    cursor = connection.execute(
        """
        INSERT INTO mart_store_month (
            store_id, store_name, city, region, month_key,
            orders, revenue_net, revenue_gross, units, aov
        )
        SELECT
            o.store_id,
            s.store_name,
            s.city,
            s.region,
            o.month_key,
            COUNT(*)                        AS orders,
            SUM(o.net_before_tax)           AS revenue_net,
            SUM(o.gross_bill_value)         AS revenue_gross,
            SUM(o.total_qty)                AS units,
            SUM(o.net_before_tax) / COUNT(*) AS aov
        FROM fact_orders AS o
        JOIN dim_store AS s ON s.store_id = o.store_id
        GROUP BY o.store_id, s.store_name, s.city, s.region, o.month_key
        """
    )
    connection.commit()
    return cursor.rowcount


def build_mart_city_month(connection: sqlite3.Connection) -> int:
    """Build the city-by-month aggregate mart from ``fact_orders``.

    Args:
        connection: Open SQLite connection.

    Returns:
        Number of rows written.
    """
    cursor = connection.execute(
        """
        INSERT INTO mart_city_month (
            city, month_key, orders, revenue_net, revenue_gross, units, aov
        )
        SELECT
            s.city,
            o.month_key,
            COUNT(*)                        AS orders,
            SUM(o.net_before_tax)           AS revenue_net,
            SUM(o.gross_bill_value)         AS revenue_gross,
            SUM(o.total_qty)                AS units,
            SUM(o.net_before_tax) / COUNT(*) AS aov
        FROM fact_orders AS o
        JOIN dim_store AS s ON s.store_id = o.store_id
        GROUP BY s.city, o.month_key
        """
    )
    connection.commit()
    return cursor.rowcount


def build_mart_channel_month(connection: sqlite3.Connection) -> int:
    """Build the channel-by-month aggregate mart from ``fact_orders``.

    Args:
        connection: Open SQLite connection.

    Returns:
        Number of rows written.
    """
    cursor = connection.execute(
        """
        INSERT INTO mart_channel_month (
            channel, month_key, orders, revenue_net, revenue_gross, units, aov
        )
        SELECT
            o.channel,
            o.month_key,
            COUNT(*)                        AS orders,
            SUM(o.net_before_tax)           AS revenue_net,
            SUM(o.gross_bill_value)         AS revenue_gross,
            SUM(o.total_qty)                AS units,
            SUM(o.net_before_tax) / COUNT(*) AS aov
        FROM fact_orders AS o
        GROUP BY o.channel, o.month_key
        """
    )
    connection.commit()
    return cursor.rowcount


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def count_rows(connection: sqlite3.Connection, table: str) -> int:
    """Count the rows in a table.

    Args:
        connection: Open SQLite connection.
        table: Table to count.

    Returns:
        The table's row count.
    """
    return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def total_revenue(connection: sqlite3.Connection) -> float:
    """Total canonical revenue across all orders.

    Args:
        connection: Open SQLite connection.

    Returns:
        ``SUM(net_before_tax)`` from ``fact_orders``.
    """
    return float(
        connection.execute("SELECT SUM(net_before_tax) FROM fact_orders").fetchone()[0]
    )


def print_summary(connection: sqlite3.Connection, db_path: Path) -> None:
    """Print the build summary: row counts, file size and total revenue.

    Args:
        connection: Open SQLite connection.
        db_path: Path to the database file, used for its size on disk.
    """
    name_width = max(len(table) for table in TABLE_NAMES)
    separator = "-" * (name_width + 14)

    print()
    print(f"{'TABLE'.ljust(name_width)}  {'ROWS':>12}")
    print(separator)
    for table in TABLE_NAMES:
        print(f"{table.ljust(name_width)}  {count_rows(connection, table):>12,}")
    print(separator)

    size_mb = db_path.stat().st_size / BYTES_PER_MB
    revenue = total_revenue(connection)
    print(f"{'database file'.ljust(name_width)}  {size_mb:>9,.2f} MB")
    print(
        f"{f'revenue ({settings.REVENUE_METRIC})'.ljust(name_width)}  "
        f"{revenue:>12,.2f} INR"
    )
    print()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def build_database(
    excel_path: Path | None = None,
    db_path: Path | None = None,
    verbose: bool = True,
) -> dict[str, int]:
    """Build the complete star schema from the source workbook.

    Drops and recreates every table, loads the dimensions, then the facts (the
    order fact depends on ``dim_calendar`` for its denormalized attributes),
    then the aggregate marts, then the indexes. Re-running is safe and produces
    an identical database.

    Args:
        excel_path: Source workbook. Defaults to ``settings.EXCEL_PATH``.
        db_path: Destination SQLite file. Defaults to ``settings.DB_PATH``.
        verbose: Whether to print the build summary to stdout.

    Returns:
        Mapping of table name to the number of rows it holds.
    """
    source = excel_path or settings.EXCEL_PATH
    destination = db_path or settings.DB_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)

    row_counts: dict[str, int] = {}
    connection = sqlite3.connect(destination)
    try:
        # Foreign keys stay off while dropping and recreating; they are enabled
        # for the load so referential integrity is validated as rows go in.
        connection.execute("PRAGMA foreign_keys = OFF")
        create_schema(connection)
        connection.execute("PRAGMA foreign_keys = ON")

        row_counts["dim_store"] = build_dim_store(connection, source)
        row_counts["dim_product"] = build_dim_product(connection, source)
        row_counts["dim_customer"] = build_dim_customer(connection, source)
        row_counts["dim_promotion"] = build_dim_promotion(connection, source)
        row_counts["dim_calendar"] = build_dim_calendar(connection, source)
        connection.commit()

        row_counts["fact_orders"] = build_fact_orders(connection, source)
        row_counts["fact_order_lines"] = build_fact_order_lines(connection, source)
        connection.commit()

        row_counts["mart_store_month"] = build_mart_store_month(connection)
        row_counts["mart_city_month"] = build_mart_city_month(connection)
        row_counts["mart_channel_month"] = build_mart_channel_month(connection)

        create_indexes(connection)
        connection.commit()

        # Reclaim space left by the dropped tables so re-runs do not grow the
        # file, and refresh the planner's statistics.
        connection.execute("ANALYZE")
        connection.commit()
        connection.execute("VACUUM")

        if verbose:
            print_summary(connection, destination)
    finally:
        connection.close()

    return row_counts


def main() -> None:
    """Command-line entrypoint: build the database and print the summary."""
    print(f"Source : {settings.EXCEL_PATH}")
    print(f"Target : {settings.DB_PATH}")
    build_database()


if __name__ == "__main__":
    main()
