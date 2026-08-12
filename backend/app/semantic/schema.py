"""Semantic layer: the single source of truth agents read to write correct SQL.

Everything an agent needs to turn a business question into a correct query lives
here — the physical schema, the canonical metric expressions, the fixed time
anchor, the rules that prevent the specific mistakes this dataset invites, and
worked examples that are executed by the test suite so they cannot rot.

The schema text is rendered from a structured table specification rather than
hand-written prose, so the full and compact renderings can never drift apart or
from each other. Row counts, dates, tax rate, channel and festive vocabularies,
and the anonymous/promotion rates are all pulled from ``app.config`` and
``app.etl.quality_checks`` rather than restated, so a change in the data shows
up here automatically.

Two entrypoints:

``get_schema_context()``
    The full prompt block, for planning and SQL generation.
``get_compact_schema()``
    Tables and columns only, for cheaper calls that just need to resolve names.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Final

from app.config import settings
from app.etl.build_db import (
    DAY_TYPES,
    NON_FESTIVE_PERIOD,
    TABLE_NAMES,
    WEEKEND_DAY_TYPE,
)
from app.etl.quality_checks import (
    EXPECTED_ANONYMOUS_ORDERS,
    EXPECTED_MONTH_COUNT,
    EXPECTED_PROMO_ORDERS,
    EXPECTED_ROW_COUNTS,
    MAX_LINE_VARIANCE_REVENUE_PCT,
    percentage,
)

# --- Derived dataset facts used in the rendered text -----------------------
_STORE_COUNT: Final[int] = EXPECTED_ROW_COUNTS["dim_store"]
_SKU_COUNT: Final[int] = EXPECTED_ROW_COUNTS["dim_product"]
_ORDER_COUNT: Final[int] = EXPECTED_ROW_COUNTS["fact_orders"]
_CHANNEL_COUNT: Final[int] = len(settings.CHANNELS)

# Distinct cities across the store estate. Not derivable from config; it is a
# property of dim_store and is asserted by the quality gate's cardinality check.
_CITY_COUNT: Final[int] = 8

_ANONYMOUS_PCT: Final[float] = percentage(EXPECTED_ANONYMOUS_ORDERS, _ORDER_COUNT)
_PROMO_PCT: Final[float] = percentage(EXPECTED_PROMO_ORDERS, _ORDER_COUNT)

# Enumerated vocabularies, small enough to inline so agents never guess a
# spelling ('Dine-in', not 'Dine In').
_REGIONS: Final[tuple[str, ...]] = ("North", "South", "East", "West")
_STORE_FORMATS: Final[tuple[str, ...]] = ("Mall", "High Street", "Food Court")
_CATEGORIES: Final[tuple[str, ...]] = (
    "Burgers",
    "Pizza",
    "Wraps",
    "Sides",
    "Beverages",
    "Desserts",
)
_VEG_FLAGS: Final[tuple[str, ...]] = ("Veg", "Non-Veg")
_CUSTOMER_SEGMENTS: Final[tuple[str, ...]] = ("Loyal", "Regular", "Occasional")
_CITIES: Final[tuple[str, ...]] = (
    "Bengaluru",
    "Chennai",
    "Delhi",
    "Gurugram",
    "Hyderabad",
    "Kolkata",
    "Mumbai",
    "Pune",
)

_FESTIVE_VALUES: Final[tuple[str, ...]] = (
    NON_FESTIVE_PERIOD,
    *settings.FESTIVE_PERIODS,
)

# Public alias. The verifier needs the cardinality of each exhaustively
# partitioning dimension to know whether a breakdown is complete and should
# therefore reconcile to a total, or is a top-N subset that must not.
CITIES: Final[tuple[str, ...]] = _CITIES


def _quoted(values: tuple[str, ...] | list[str]) -> str:
    """Render a vocabulary as a quoted, comma-separated list.

    Args:
        values: The permitted values.

    Returns:
        A string such as ``'Veg' | 'Non-Veg'``.
    """
    return " | ".join(f"'{value}'" for value in values)


def month_keys() -> list[str]:
    """Enumerate every valid ``month_key`` in the dataset.

    Walks month by month from the configured data start to the as-of date, so
    the list follows the config rather than restating it.

    Returns:
        The ``YYYY-MM`` keys in chronological order.
    """
    keys: list[str] = []
    cursor = settings.DATA_START_DATE.replace(day=1)
    last = settings.DATA_ASOF_DATE.replace(day=1)
    while cursor <= last:
        keys.append(cursor.strftime("%Y-%m"))
        cursor = (
            date(cursor.year + 1, 1, 1)
            if cursor.month == 12
            else date(cursor.year, cursor.month + 1, 1)
        )
    return keys


# ---------------------------------------------------------------------------
# Structured table specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Column:
    """One column in the physical schema.

    Attributes:
        name: Column name.
        type: SQLite storage type.
        description: What the column means, including its vocabulary when the
            set of values is small enough to enumerate.
        key: ``PK``, or ``FK -> table.column``, or None.
        nullable: Whether NULL is a legitimate value.
        null_meaning: What a NULL signifies, required when nullable.
    """

    name: str
    type: str
    description: str
    key: str | None = None
    nullable: bool = False
    null_meaning: str | None = None


@dataclass(frozen=True)
class Table:
    """One table in the physical schema.

    Attributes:
        name: Table name.
        purpose: What questions the table answers.
        grain: What exactly one row represents.
        row_count: Number of rows in the built database.
        columns: The table's columns, in physical order.
    """

    name: str
    purpose: str
    grain: str
    row_count: int
    columns: tuple[Column, ...]


TABLES: Final[tuple[Table, ...]] = (
    Table(
        name="dim_store",
        purpose="Store attributes for location, region and format analysis.",
        grain="one row per store",
        row_count=_STORE_COUNT,
        columns=(
            Column("store_id", "TEXT", "Store identifier, e.g. 'ST001'.", key="PK"),
            Column("store_name", "TEXT", "Human-readable store name."),
            Column("city", "TEXT", f"City. One of {_quoted(_CITIES)}."),
            Column("state", "TEXT", "Indian state the store sits in."),
            Column("region", "TEXT", f"Region. One of {_quoted(_REGIONS)}."),
            Column(
                "store_format",
                "TEXT",
                f"Format. One of {_quoted(_STORE_FORMATS)}.",
            ),
            Column("opening_date", "TEXT", "ISO date 'YYYY-MM-DD' the store opened."),
            Column(
                "city_price_index",
                "REAL",
                "Relative price level of the city; ~1.03-1.22. Explains price "
                "differences between cities, not demand.",
            ),
            Column(
                "performance_factor",
                "REAL",
                "Baseline demand multiplier for the store, independent of city.",
            ),
            Column(
                "status",
                "TEXT",
                f"Lifecycle status. All {_STORE_COUNT} stores are 'Active'; this "
                "column has no filtering value.",
            ),
        ),
    ),
    Table(
        name="dim_product",
        purpose="SKU attributes for product, category and margin analysis.",
        grain="one row per SKU",
        row_count=_SKU_COUNT,
        columns=(
            Column("sku_id", "TEXT", "SKU identifier, e.g. 'SKU001'.", key="PK"),
            Column("sku_name", "TEXT", "Menu item name."),
            Column("category", "TEXT", f"Category. One of {_quoted(_CATEGORIES)}."),
            Column("veg_nonveg", "TEXT", f"Diet flag. One of {_quoted(_VEG_FLAGS)}."),
            Column("base_price_inr", "REAL", "List price in INR before city indexing."),
            Column(
                "est_cogs_pct",
                "REAL",
                "Estimated cost of goods as a fraction of price, e.g. 0.32.",
            ),
            Column(
                "status",
                "TEXT",
                f"Lifecycle status. All {_SKU_COUNT} SKUs are 'Active'; this "
                "column has no filtering value.",
            ),
        ),
    ),
    Table(
        name="dim_customer",
        purpose="Known customers, for segment and loyalty analysis.",
        grain="one row per registered customer",
        row_count=EXPECTED_ROW_COUNTS["dim_customer"],
        columns=(
            Column("customer_id", "TEXT", "Customer identifier.", key="PK"),
            Column("home_city", "TEXT", "City the customer is registered in."),
            Column(
                "customer_segment",
                "TEXT",
                f"Segment. One of {_quoted(_CUSTOMER_SEGMENTS)}.",
            ),
            Column("join_date", "TEXT", "ISO date 'YYYY-MM-DD' the customer joined."),
        ),
    ),
    Table(
        name="dim_promotion",
        purpose="Promotion definitions, for discount and campaign analysis.",
        grain="one row per promotion",
        row_count=EXPECTED_ROW_COUNTS["dim_promotion"],
        columns=(
            Column("promo_id", "TEXT", "Promotion identifier, e.g. 'PR001'.", key="PK"),
            Column("promo_name", "TEXT", "Campaign name, e.g. 'Weekend Saver'."),
            Column(
                "promo_type",
                "TEXT",
                "Mechanic. One of 'Order Discount' | 'Beverage Bundle'.",
            ),
            Column("start_date", "TEXT", "ISO date the promotion opens."),
            Column("end_date", "TEXT", "ISO date the promotion closes, inclusive."),
            Column(
                "applicable_days",
                "TEXT",
                "Day restriction as free text, e.g. 'All Days' | 'Sat/Sun'.",
            ),
            Column(
                "applicability",
                "TEXT",
                "Scope as free text, e.g. 'All' | 'Kolkata' | 'Swiggy/Zomato'.",
            ),
            Column("discount_pct", "REAL", "Headline discount percent, e.g. 10."),
            Column("min_bill_value", "REAL", "Minimum bill in INR to qualify."),
            Column("max_discount_inr", "REAL", "Discount cap in INR."),
        ),
    ),
    Table(
        name="dim_calendar",
        purpose=(
            "Date spine with weekend and festive attributes. Rarely needed "
            "directly because fact_orders already carries these columns."
        ),
        grain="one row per calendar date in the dataset window",
        row_count=EXPECTED_ROW_COUNTS["dim_calendar"],
        columns=(
            Column("date", "TEXT", "ISO date 'YYYY-MM-DD'.", key="PK"),
            Column("year", "INTEGER", "Calendar year, 2025 or 2026."),
            Column("month", "TEXT", "Short month name, e.g. 'Aug'."),
            Column("month_no", "INTEGER", "Month number 1-12."),
            Column("day_name", "TEXT", "Day of week, e.g. 'Monday'."),
            Column("day_type", "TEXT", f"One of {_quoted(DAY_TYPES)}."),
            Column(
                "festive_period",
                "TEXT",
                f"One of {_quoted(_FESTIVE_VALUES)}.",
            ),
            Column("month_key", "TEXT", "Derived 'YYYY-MM' key. Sorts chronologically."),
            Column(
                "is_weekend",
                "INTEGER",
                f"Derived flag: 1 when day_type = '{WEEKEND_DAY_TYPE}', else 0.",
            ),
            Column(
                "is_festive",
                "INTEGER",
                f"Derived flag: 1 when festive_period != '{NON_FESTIVE_PERIOD}', "
                "else 0.",
            ),
        ),
    ),
    Table(
        name="fact_orders",
        purpose=(
            "THE CANONICAL FACT for revenue, order counts and AOV. Calendar "
            "attributes are denormalized onto every row, so weekend, festive and "
            "monthly analysis needs no join."
        ),
        grain="one row per order",
        row_count=_ORDER_COUNT,
        columns=(
            Column("order_id", "TEXT", "Order identifier.", key="PK"),
            Column(
                "order_datetime", "TEXT", "ISO timestamp 'YYYY-MM-DD HH:MM:SS'."
            ),
            Column(
                "order_date",
                "TEXT",
                "ISO date 'YYYY-MM-DD'. Use this for all date filtering.",
                key="FK -> dim_calendar.date",
            ),
            Column("order_hour", "INTEGER", "Hour of day 0-23, for daypart analysis."),
            Column("store_id", "TEXT", "Store that took the order.", key="FK -> dim_store.store_id"),
            Column(
                "customer_id",
                "TEXT",
                "Identified customer, when there is one.",
                key="FK -> dim_customer.customer_id",
                nullable=True,
                null_meaning=(
                    f"anonymous walk-in; {EXPECTED_ANONYMOUS_ORDERS:,} orders "
                    f"({_ANONYMOUS_PCT:.2f}%) are NULL"
                ),
            ),
            Column(
                "channel",
                "TEXT",
                f"Sales channel. One of {_quoted(settings.CHANNELS)}.",
            ),
            Column(
                "promo_id",
                "TEXT",
                "Promotion applied to the order.",
                key="FK -> dim_promotion.promo_id",
                nullable=True,
                null_meaning=(
                    f"no promotion; only {EXPECTED_PROMO_ORDERS:,} orders "
                    f"({_PROMO_PCT:.2f}%) are non-NULL"
                ),
            ),
            Column("total_qty", "INTEGER", "Units on the order. Use for 'units sold'."),
            Column("gross_bill_value", "REAL", "Bill in INR before discount and tax."),
            Column("discount_amount", "REAL", "Discount in INR. 0 when no promotion."),
            Column(
                "net_before_tax",
                "REAL",
                f"CANONICAL REVENUE in INR, excluding the "
                f"{settings.TAX_RATE:.0%} tax. Equals gross_bill_value - "
                f"discount_amount. This is what 'revenue' means.",
            ),
            Column("tax_amount", "REAL", f"Tax in INR at {settings.TAX_RATE:.0%}."),
            Column(
                "net_revenue",
                "REAL",
                f"Revenue INCLUDING tax; equals net_before_tax x "
                f"{1 + settings.TAX_RATE}. Use only when the question explicitly "
                "asks for tax-inclusive figures.",
            ),
            Column("month_key", "TEXT", "Denormalized 'YYYY-MM'. Group by this for months."),
            Column("day_name", "TEXT", "Denormalized day of week."),
            Column("day_type", "TEXT", f"Denormalized; one of {_quoted(DAY_TYPES)}."),
            Column("is_weekend", "INTEGER", "Denormalized flag, 1 or 0."),
            Column(
                "festive_period",
                "TEXT",
                f"Denormalized; one of {_quoted(_FESTIVE_VALUES)}.",
            ),
            Column("is_festive", "INTEGER", "Denormalized flag, 1 or 0."),
        ),
    ),
    Table(
        name="fact_order_lines",
        purpose=(
            "Line detail for product, category and margin analysis ONLY. Do not "
            "total revenue here; see the business rules."
        ),
        grain="one row per SKU per order",
        row_count=EXPECTED_ROW_COUNTS["fact_order_lines"],
        columns=(
            Column("order_detail_id", "TEXT", "Line identifier.", key="PK"),
            Column("order_id", "TEXT", "Parent order.", key="FK -> fact_orders.order_id"),
            Column("sku_id", "TEXT", "Product sold.", key="FK -> dim_product.sku_id"),
            Column("quantity", "INTEGER", "Units of this SKU on the order."),
            Column("unit_price", "REAL", "Price per unit in INR, city-indexed."),
            Column("line_gross_value", "REAL", "quantity x unit_price, in INR."),
            Column("line_discount", "REAL", "Discount allocated to this line, in INR."),
            Column("line_net_value", "REAL", "Line revenue in INR after discount, pre-tax."),
            Column("est_cogs", "REAL", "Estimated cost of goods for the line, in INR."),
            Column(
                "line_margin",
                "REAL",
                "Derived: line_net_value - est_cogs. Use for margin questions.",
            ),
        ),
    ),
    Table(
        name="mart_store_month",
        purpose=(
            "Pre-aggregated store performance by month. Prefer this over "
            "fact_orders for store trend and decline questions."
        ),
        grain="one row per store per month",
        row_count=_STORE_COUNT * EXPECTED_MONTH_COUNT,
        columns=(
            Column("store_id", "TEXT", "Store.", key="PK part; FK -> dim_store.store_id"),
            Column("store_name", "TEXT", "Denormalized store name; no join needed."),
            Column("city", "TEXT", "Denormalized city."),
            Column("region", "TEXT", "Denormalized region."),
            Column("month_key", "TEXT", "'YYYY-MM'.", key="PK part"),
            Column("orders", "INTEGER", "Order count in the month."),
            Column("revenue_net", "REAL", "SUM(net_before_tax) in INR. Canonical revenue."),
            Column("revenue_gross", "REAL", "SUM(gross_bill_value) in INR."),
            Column("units", "INTEGER", "SUM(total_qty)."),
            Column("aov", "REAL", "revenue_net / orders, in INR."),
        ),
    ),
    Table(
        name="mart_city_month",
        purpose="Pre-aggregated city performance by month.",
        grain="one row per city per month",
        row_count=_CITY_COUNT * EXPECTED_MONTH_COUNT,
        columns=(
            Column("city", "TEXT", f"One of {_quoted(_CITIES)}.", key="PK part"),
            Column("month_key", "TEXT", "'YYYY-MM'.", key="PK part"),
            Column("orders", "INTEGER", "Order count in the month."),
            Column("revenue_net", "REAL", "SUM(net_before_tax) in INR. Canonical revenue."),
            Column("revenue_gross", "REAL", "SUM(gross_bill_value) in INR."),
            Column("units", "INTEGER", "SUM(total_qty)."),
            Column("aov", "REAL", "revenue_net / orders, in INR."),
        ),
    ),
    Table(
        name="mart_channel_month",
        purpose="Pre-aggregated channel performance by month, for channel mix.",
        grain="one row per channel per month",
        row_count=_CHANNEL_COUNT * EXPECTED_MONTH_COUNT,
        columns=(
            Column(
                "channel",
                "TEXT",
                f"One of {_quoted(settings.CHANNELS)}.",
                key="PK part",
            ),
            Column("month_key", "TEXT", "'YYYY-MM'.", key="PK part"),
            Column("orders", "INTEGER", "Order count in the month."),
            Column("revenue_net", "REAL", "SUM(net_before_tax) in INR. Canonical revenue."),
            Column("revenue_gross", "REAL", "SUM(gross_bill_value) in INR."),
            Column("units", "INTEGER", "SUM(total_qty)."),
            Column("aov", "REAL", "revenue_net / orders, in INR."),
        ),
    ),
)


def _render_column(column: Column) -> str:
    """Render one column as a schema line.

    Args:
        column: The column to render.

    Returns:
        A single line describing the column, its type, keys and nullability.
    """
    parts = [f"    {column.name} {column.type}"]
    if column.key:
        parts.append(f"[{column.key}]")
    if column.nullable:
        parts.append(f"[NULLABLE: {column.null_meaning}]")
    parts.append(f"- {column.description}")
    return " ".join(parts)


def _render_schema_description() -> str:
    """Render the full physical schema description.

    Returns:
        Every table with its purpose, grain, row count and columns.
    """
    blocks: list[str] = []
    for table in TABLES:
        lines = [
            f"TABLE {table.name}  ({table.row_count:,} rows)",
            f"  purpose: {table.purpose}",
            f"  grain:   {table.grain}",
            "  columns:",
        ]
        lines.extend(_render_column(column) for column in table.columns)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


SCHEMA_DESCRIPTION: Final[str] = _render_schema_description()


def _render_compact_schema() -> str:
    """Render the tables-and-columns-only schema.

    Returns:
        One line per table listing its column names.
    """
    return "\n".join(
        f"{table.name}({', '.join(column.name for column in table.columns)})"
        for table in TABLES
    )


COMPACT_SCHEMA: Final[str] = _render_compact_schema()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

METRIC_DEFINITIONS: Final[dict[str, dict[str, str]]] = {
    "revenue": {
        "sql": "SUM(net_before_tax)",
        "description": (
            f"Canonical revenue in INR, excluding the {settings.TAX_RATE:.0%} "
            "tax. This is what 'revenue', 'sales' and 'turnover' mean unless the "
            "question says otherwise."
        ),
        "source_table": "fact_orders",
        "unit": "INR",
    },
    "revenue_with_tax": {
        "sql": "SUM(net_revenue)",
        "description": (
            "Revenue including tax. Use only when the question explicitly asks "
            "for a tax-inclusive figure."
        ),
        "source_table": "fact_orders",
        "unit": "INR",
    },
    "gross_revenue": {
        "sql": "SUM(gross_bill_value)",
        "description": "Billed value before discount and before tax.",
        "source_table": "fact_orders",
        "unit": "INR",
    },
    "discount": {
        "sql": "SUM(discount_amount)",
        "description": (
            "Total discount given. Equals gross_revenue - revenue. Zero on the "
            "~96% of orders with no promotion."
        ),
        "source_table": "fact_orders",
        "unit": "INR",
    },
    "orders": {
        "sql": "COUNT(DISTINCT order_id)",
        "description": (
            "Number of orders. Use COUNT(DISTINCT order_id) rather than COUNT(*) "
            "so the expression stays correct if fact_order_lines is joined in."
        ),
        "source_table": "fact_orders",
        "unit": "count",
    },
    "units": {
        "sql": "SUM(total_qty)",
        "description": (
            "Units sold. From fact_orders use SUM(total_qty); when already at "
            "line grain for a product question use SUM(quantity) on "
            "fact_order_lines instead. The two agree exactly."
        ),
        "source_table": "fact_orders",
        "alt_sql": "SUM(quantity)",
        "alt_source_table": "fact_order_lines",
        "unit": "count",
    },
    "aov": {
        "sql": "SUM(net_before_tax) / COUNT(DISTINCT order_id)",
        "description": (
            "Average order value in INR. Always compute as a ratio of sums, "
            "never AVG(net_before_tax) of pre-aggregated rows."
        ),
        "source_table": "fact_orders",
        "unit": "INR",
    },
    "units_per_order": {
        "sql": "SUM(total_qty) / COUNT(DISTINCT order_id)",
        "description": "Average basket size in units.",
        "source_table": "fact_orders",
        "unit": "count",
    },
    "gross_margin": {
        "sql": "SUM(line_net_value - est_cogs)",
        "description": (
            "Gross margin in INR. Only available at line grain; equivalently "
            "SUM(line_margin), which is precomputed."
        ),
        "source_table": "fact_order_lines",
        "alt_sql": "SUM(line_margin)",
        "unit": "INR",
    },
    "margin_pct": {
        "sql": "100.0 * SUM(line_net_value - est_cogs) / SUM(line_net_value)",
        "description": (
            "Gross margin as a percent of line revenue. Denominator is line "
            "revenue, not fact_orders revenue - the two grains do not reconcile "
            "exactly, so never mix them in one ratio."
        ),
        "source_table": "fact_order_lines",
        "unit": "percent",
    },
}


# ---------------------------------------------------------------------------
# Time anchor
# ---------------------------------------------------------------------------


def _render_time_anchor() -> str:
    """Render the time anchoring rules and the valid month keys.

    Returns:
        The time anchor prompt section.
    """
    keys = month_keys()
    rows = "\n".join(
        f"    {key}" + ("   <- last 3 months" if key in keys[-3:] else "")
        for key in keys
    )
    return f"""TODAY IS {settings.DATA_ASOF_DATE.isoformat()}.

This is a fixed historical dataset. The system's notion of "today" is the
constant {settings.DATA_ASOF_DATE.isoformat()}. NEVER use the system clock,
CURRENT_DATE, DATE('now') or any equivalent - the real date is well past the end
of the data, so every window built from it returns zero rows.

Data coverage : {settings.DATA_START_DATE.isoformat()} to {settings.DATA_ASOF_DATE.isoformat()} ({EXPECTED_MONTH_COUNT} complete months)
Today         : {settings.DATA_ASOF_DATE.isoformat()}
Last 3 months : {settings.LAST_3M_START.isoformat()} to {settings.LAST_3M_END.isoformat()}

Resolve relative time expressions against the anchor above:
  "today"                 -> order_date = '{settings.DATA_ASOF_DATE.isoformat()}'
  "last 3 months"         -> order_date BETWEEN '{settings.LAST_3M_START.isoformat()}' AND '{settings.LAST_3M_END.isoformat()}'
  "this month" / "latest" -> month_key = '{keys[-1]}'
  "last month"            -> month_key = '{keys[-2]}'
  "last 6 months"         -> month_key >= '{keys[-6]}'
  "year to date"          -> order_date BETWEEN '{settings.DATA_ASOF_DATE.year}-01-01' AND '{settings.DATA_ASOF_DATE.isoformat()}'
  "full year" / "overall" -> the entire range; no date filter needed

month_key is TEXT in 'YYYY-MM' form and sorts chronologically as a string, so
BETWEEN, >= and ORDER BY all work directly on it. The {EXPECTED_MONTH_COUNT} valid
month_key values are:

{rows}

Any month_key outside this list returns no rows."""


TIME_ANCHOR: Final[str] = _render_time_anchor()


# ---------------------------------------------------------------------------
# Business rules
# ---------------------------------------------------------------------------

BUSINESS_RULES: Final[list[str]] = [
    (
        f"ALWAYS LEFT JOIN dim_customer, never INNER JOIN. "
        f"{EXPECTED_ANONYMOUS_ORDERS:,} of {_ORDER_COUNT:,} orders "
        f"({_ANONYMOUS_PCT:.2f}%) are anonymous walk-ins with customer_id IS "
        f"NULL. An inner join silently drops more than a quarter of all revenue "
        f"and the result still looks plausible."
    ),
    (
        f"ALWAYS LEFT JOIN dim_promotion, never INNER JOIN. Only "
        f"{EXPECTED_PROMO_ORDERS:,} orders ({_PROMO_PCT:.2f}%) carry a promo_id; "
        f"NULL is the normal case."
    ),
    (
        f"Revenue means net_before_tax, NEVER net_revenue. net_revenue includes "
        f"the {settings.TAX_RATE:.0%} tax and overstates revenue by "
        f"{settings.TAX_RATE:.0%}. Use net_revenue only when the question "
        f"explicitly asks for a tax-inclusive figure."
    ),
    (
        f"Use fact_orders for revenue, order counts, AOV and units. Use "
        f"fact_order_lines ONLY for SKU, product, category and margin questions. "
        f"A small share of revenue does not reconcile between the two grains - a "
        f"known defect in the source data, held below "
        f"{MAX_LINE_VARIANCE_REVENUE_PCT}% of revenue by the quality gate - so a "
        f"revenue total taken from the line grain will not match the canonical "
        f"figure. The affected orders are concentrated in the most recent "
        f"months, so the gap is several times wider inside a recent window "
        f"(around 0.4% over the last 3 months) than across the full year "
        f"(around 0.1%). This makes the rule matter most for exactly the "
        f"questions people ask most."
    ),
    (
        "Prefer the marts (mart_store_month, mart_city_month, "
        "mart_channel_month) for monthly trend, ranking and decline questions. "
        "They are pre-aggregated, carry denormalized names, and reconcile "
        "exactly with fact_orders."
    ),
    (
        f"All {_STORE_COUNT} stores and all {_SKU_COUNT} SKUs have status = "
        f"'Active'. Filtering on status is never necessary and never changes a "
        f"result."
    ),
    (
        "is_weekend, day_type, festive_period, is_festive and month_key are "
        "already denormalized onto fact_orders. Weekend, festive and monthly "
        "analysis needs NO join to dim_calendar."
    ),
    (
        "All monetary values are INR. Never convert, never assume another "
        "currency, and label outputs as INR."
    ),
    (
        "Compute averages as ratios of sums (SUM(x) / COUNT(...)), not as AVG() "
        "over already-aggregated rows, or the result is a mean of means."
    ),
    (
        f"Only these tables exist: {', '.join(TABLE_NAMES)}. Never reference any "
        f"other table or column name."
    ),
]


# ---------------------------------------------------------------------------
# Few-shot examples
# ---------------------------------------------------------------------------

_LAST_3M_START: Final[str] = settings.LAST_3M_START.isoformat()
_LAST_3M_END: Final[str] = settings.LAST_3M_END.isoformat()

EXAMPLE_QUERIES: Final[list[dict[str, str]]] = [
    {
        "pattern": "time-windowed aggregate",
        "teaches": (
            "Resolve a relative window against the fixed anchor; use "
            "net_before_tax; compute AOV as a ratio of sums."
        ),
        "question": "What was our total revenue in the last 3 months?",
        "sql": f"""SELECT
    ROUND(SUM(net_before_tax), 2)                            AS revenue_inr,
    COUNT(DISTINCT order_id)                                 AS orders,
    ROUND(SUM(net_before_tax) / COUNT(DISTINCT order_id), 2) AS aov_inr
FROM fact_orders
WHERE order_date BETWEEN '{_LAST_3M_START}' AND '{_LAST_3M_END}'""",
    },
    {
        "pattern": "ranking with a dimension join",
        "teaches": (
            "Join a dimension for attributes, group by the key plus the selected "
            "attributes, order by the metric and limit."
        ),
        "question": "Which 5 stores generated the most revenue in the last 3 months?",
        "sql": f"""SELECT
    s.store_name,
    s.city,
    s.region,
    COUNT(DISTINCT o.order_id)      AS orders,
    ROUND(SUM(o.net_before_tax), 2) AS revenue_inr
FROM fact_orders AS o
JOIN dim_store AS s ON s.store_id = o.store_id
WHERE o.order_date BETWEEN '{_LAST_3M_START}' AND '{_LAST_3M_END}'
GROUP BY s.store_id, s.store_name, s.city, s.region
ORDER BY revenue_inr DESC
LIMIT 5""",
    },
    {
        "pattern": "month-over-month trend from a mart",
        "teaches": (
            "Use a pre-aggregated mart for trends; use LAG() over month_key for "
            "month-over-month change; month_key sorts correctly as text."
        ),
        "question": "How has revenue trended month over month in Mumbai?",
        "sql": """SELECT
    month_key,
    orders,
    ROUND(revenue_net, 2) AS revenue_inr,
    ROUND(
        100.0 * (revenue_net - LAG(revenue_net) OVER (ORDER BY month_key))
        / LAG(revenue_net) OVER (ORDER BY month_key),
        2
    ) AS mom_change_pct
FROM mart_city_month
WHERE city = 'Mumbai'
ORDER BY month_key""",
    },
    {
        "pattern": "product analysis at line grain",
        "teaches": (
            "Product and margin questions use fact_order_lines joined back to "
            "fact_orders for the date filter; margin comes from line_margin; "
            "the margin denominator stays at line grain."
        ),
        "question": "Which 5 SKUs made us the most gross margin in the last 3 months?",
        "sql": f"""SELECT
    p.sku_name,
    p.category,
    SUM(l.quantity)                  AS units,
    ROUND(SUM(l.line_net_value), 2)  AS line_revenue_inr,
    ROUND(SUM(l.line_margin), 2)     AS gross_margin_inr,
    ROUND(100.0 * SUM(l.line_margin) / SUM(l.line_net_value), 2) AS margin_pct
FROM fact_order_lines AS l
JOIN fact_orders AS o  ON o.order_id = l.order_id
JOIN dim_product AS p  ON p.sku_id = l.sku_id
WHERE o.order_date BETWEEN '{_LAST_3M_START}' AND '{_LAST_3M_END}'
GROUP BY p.sku_id, p.sku_name, p.category
ORDER BY gross_margin_inr DESC
LIMIT 5""",
    },
]


# ---------------------------------------------------------------------------
# Allowlist and assembled context
# ---------------------------------------------------------------------------

# The only tables an agent may reference. Sourced from the ETL's table list so
# the allowlist cannot drift from what is actually built.
TABLE_ALLOWLIST: Final[list[str]] = list(TABLE_NAMES)


def _render_metrics() -> str:
    """Render the metric catalogue.

    Returns:
        One block per metric with its SQL expression and meaning.
    """
    lines: list[str] = []
    for name, metric in METRIC_DEFINITIONS.items():
        lines.append(f"  {name} ({metric['unit']}) from {metric['source_table']}")
        lines.append(f"    SQL: {metric['sql']}")
        if "alt_sql" in metric:
            alt_source = metric.get("alt_source_table", metric["source_table"])
            lines.append(f"    alt: {metric['alt_sql']}  (on {alt_source})")
        lines.append(f"    {metric['description']}")
    return "\n".join(lines)


def _render_examples() -> str:
    """Render the few-shot examples.

    Returns:
        One block per example with the question, the pattern and the SQL.
    """
    blocks: list[str] = []
    for index, example in enumerate(EXAMPLE_QUERIES, start=1):
        blocks.append(
            f"Example {index} - {example['pattern']}\n"
            f"Q: {example['question']}\n"
            f"Teaches: {example['teaches']}\n"
            f"SQL:\n{example['sql']}"
        )
    return "\n\n".join(blocks)


def _section(title: str) -> str:
    """Render a section header.

    Args:
        title: Section title.

    Returns:
        The title wrapped in a rule for readability in a prompt.
    """
    return f"\n{'=' * 78}\n{title}\n{'=' * 78}\n"


def get_schema_context() -> str:
    """Assemble the full schema context injected into agent prompts.

    Returns:
        A sectioned prompt block covering the time anchor, the physical schema,
        the metric catalogue, the business rules, worked examples and the table
        allowlist.
    """
    rules = "\n".join(
        f"{index}. {rule}" for index, rule in enumerate(BUSINESS_RULES, start=1)
    )
    return "\n".join(
        [
            f"{settings.APP_NAME} - SQL generation context",
            f"Dialect: SQLite. Database: {settings.DB_PATH.name}. Currency: INR.",
            _section("1. TIME ANCHOR (read this first)"),
            TIME_ANCHOR,
            _section("2. DATABASE SCHEMA"),
            SCHEMA_DESCRIPTION,
            _section("3. METRIC DEFINITIONS (use these expressions verbatim)"),
            _render_metrics(),
            _section("4. BUSINESS RULES (violating these produces wrong answers)"),
            rules,
            _section("5. EXAMPLE QUERIES"),
            _render_examples(),
            _section("6. TABLE ALLOWLIST"),
            "Only these tables may appear in a query:\n"
            + "\n".join(f"  - {table}" for table in TABLE_ALLOWLIST),
            "",
        ]
    )


def get_compact_schema() -> str:
    """Assemble a short schema block for cheaper, name-resolution-only calls.

    Returns:
        The anchor dates, the table and column listing, the metric expressions
        and the two rules that most often cause wrong SQL.
    """
    metrics = "\n".join(
        f"  {name} = {metric['sql']}" for name, metric in METRIC_DEFINITIONS.items()
    )
    return "\n".join(
        [
            f"SQLite. Today = {settings.DATA_ASOF_DATE.isoformat()} (fixed; never "
            f"use the system clock).",
            f"Data {settings.DATA_START_DATE.isoformat()} to "
            f"{settings.DATA_ASOF_DATE.isoformat()}. Last 3 months = "
            f"{settings.LAST_3M_START.isoformat()} to "
            f"{settings.LAST_3M_END.isoformat()}. month_key is 'YYYY-MM'.",
            "",
            "TABLES:",
            COMPACT_SCHEMA,
            "",
            "METRICS:",
            metrics,
            "",
            "Revenue = net_before_tax (tax-exclusive). customer_id and promo_id "
            "are nullable: always LEFT JOIN their dimensions.",
        ]
    )
