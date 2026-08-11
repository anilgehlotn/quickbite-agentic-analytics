"""Compute ground-truth answers to the eight evaluation questions.

This is the system's INDEPENDENT verification path. It reads the source Excel
workbook with pandas and never touches the SQLite database, the ETL module or
the semantic layer. That independence is the whole point: if the ground truth
and the agents both read the same SQL layer, a bug in that layer is invisible to
every test. The only thing shared with the rest of the application is
``app.config.settings``, so the time anchor and the revenue metric cannot drift
apart between the two paths.

Sheet and column names are declared locally rather than imported from the ETL
for the same reason - a wrong constant in the ETL must not silently propagate
into the answers that are supposed to catch it.

Grain rule, applied throughout:

* Revenue, order counts and AOV come from the **Orders** sheet (order grain).
* SKU, product and category questions come from **Order_Details** (line grain).

About 216 orders carry line values that do not sum to their header
``NET_BEFORE_TAX``. The order grain is canonical, so line-grain revenue is
reported only in Q4 and is explicitly labelled there.

Run with::

    python scripts/compute_golden_answers.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

# This script sits outside the backend package. Add it to the path so the
# canonical configuration can be imported rather than restated.
_BACKEND_DIR: Final[Path] = Path(__file__).resolve().parents[1] / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.config import settings  # noqa: E402  (path set up above)

# --- Source sheets ---------------------------------------------------------
SHEET_STORES: Final[str] = "Store_Master"
SHEET_PRODUCTS: Final[str] = "Product_Master"
SHEET_CALENDAR: Final[str] = "Calendar"
SHEET_ORDERS: Final[str] = "Orders"
SHEET_ORDER_DETAILS: Final[str] = "Order_Details"

# --- Domain vocabulary -----------------------------------------------------
NON_FESTIVE_PERIOD: Final[str] = "Normal"
WEEKEND_DAY_TYPE: Final[str] = "Weekend"
WEEKDAY_DAY_TYPE: Final[str] = "Weekday"

# --- Output ----------------------------------------------------------------
GOLDEN_PATH: Final[Path] = _BACKEND_DIR / "tests" / "golden_answers.json"

# Number of months in the analysis window and in the comparison window that
# precedes it.
WINDOW_MONTHS: Final[int] = 3

# Console layout.
LINE_WIDTH: Final[int] = 100
TOP_N: Final[int] = 5


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def inr(value: float) -> str:
    """Format a monetary amount with thousands separators.

    Args:
        value: Amount in INR.

    Returns:
        The amount to two decimals, e.g. ``3,197,076.50``.
    """
    return f"{value:,.2f}"


def pct(value: float) -> str:
    """Format a percentage with an explicit sign.

    Args:
        value: Percentage value.

    Returns:
        The percentage to two decimals, e.g. ``-4.31%``.
    """
    return f"{value:+.2f}%"


def heading(title: str) -> None:
    """Print a section heading.

    Args:
        title: Section title.
    """
    print()
    print("=" * LINE_WIDTH)
    print(title)
    print("=" * LINE_WIDTH)


def subheading(title: str) -> None:
    """Print a subsection heading.

    Args:
        title: Subsection title.
    """
    print()
    print(f"-- {title} " + "-" * max(0, LINE_WIDTH - len(title) - 4))


def key_value(label: str, value: str, label_width: int = 34) -> None:
    """Print an aligned label/value pair.

    Args:
        label: Left-hand label.
        value: Right-hand value, already formatted.
        label_width: Column width for the label.
    """
    print(f"  {label.ljust(label_width)} {value}")


def print_table(
    headers: list[str], rows: list[list[str]], aligns: str | None = None
) -> None:
    """Print an aligned text table.

    Args:
        headers: Column headers.
        rows: Row values, already formatted as strings.
        aligns: One character per column, ``l`` or ``r``. Defaults to left for
            the first column and right for the rest.
    """
    if aligns is None:
        aligns = "l" + "r" * (len(headers) - 1)
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]
    header_line = "  ".join(
        headers[i].ljust(widths[i]) if aligns[i] == "l" else headers[i].rjust(widths[i])
        for i in range(len(headers))
    )
    print(f"  {header_line}")
    print(f"  {'-' * len(header_line)}")
    for row in rows:
        print(
            "  "
            + "  ".join(
                row[i].ljust(widths[i]) if aligns[i] == "l" else row[i].rjust(widths[i])
                for i in range(len(headers))
            )
        )


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------


def month_key_of(value: date) -> str:
    """Render a date as a ``YYYY-MM`` month key.

    Args:
        value: The date.

    Returns:
        The month key.
    """
    return value.strftime("%Y-%m")


def shift_months(anchor: date, months: int) -> date:
    """Shift a date by a whole number of months, landing on the first of it.

    Args:
        anchor: Starting date.
        months: Months to add; negative shifts backwards.

    Returns:
        The first day of the shifted month.
    """
    total = anchor.year * 12 + (anchor.month - 1) + months
    return date(total // 12, total % 12 + 1, 1)


def month_keys_between(start: date, end: date) -> list[str]:
    """Enumerate the month keys spanning two dates, inclusive.

    Args:
        start: First date in the range.
        end: Last date in the range.

    Returns:
        The ``YYYY-MM`` keys in chronological order.
    """
    keys: list[str] = []
    cursor = start.replace(day=1)
    last = end.replace(day=1)
    while cursor <= last:
        keys.append(month_key_of(cursor))
        cursor = shift_months(cursor, 1)
    return keys


def safe_pct_change(old: float, new: float) -> float:
    """Percentage change between two values, guarding a zero base.

    Args:
        old: Earlier value.
        new: Later value.

    Returns:
        Percent change, or 0.0 when the base is zero.
    """
    if not old:
        return 0.0
    return (new - old) / old * 100


def safe_ratio(numerator: float, denominator: float) -> float:
    """Divide, guarding a zero denominator.

    Args:
        numerator: Top of the ratio.
        denominator: Bottom of the ratio.

    Returns:
        The ratio, or 0.0 when the denominator is zero.
    """
    if not denominator:
        return 0.0
    return numerator / denominator


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Dataset:
    """The source workbook, loaded and enriched for analysis.

    Attributes:
        orders: Order-grain facts joined to store and calendar attributes.
        lines: Line-grain facts joined to product attributes and order dates.
        calendar: The full date spine with a month key.
        stores: Store master.
        products: Product master.
    """

    orders: pd.DataFrame
    lines: pd.DataFrame
    calendar: pd.DataFrame
    stores: pd.DataFrame
    products: pd.DataFrame


def load_dataset(excel_path: Path) -> Dataset:
    """Read the workbook and build the analysis frames.

    Joins are all left joins from the fact frames, so the 5,664 orders with a
    NULL ``CUSTOMER_ID`` and the orders with no promotion are preserved.

    Args:
        excel_path: Path to the source Excel workbook.

    Returns:
        The loaded and enriched dataset.
    """
    stores = pd.read_excel(excel_path, sheet_name=SHEET_STORES)
    products = pd.read_excel(excel_path, sheet_name=SHEET_PRODUCTS)
    calendar = pd.read_excel(excel_path, sheet_name=SHEET_CALENDAR)
    orders = pd.read_excel(excel_path, sheet_name=SHEET_ORDERS)
    lines = pd.read_excel(excel_path, sheet_name=SHEET_ORDER_DETAILS)

    calendar["DATE"] = pd.to_datetime(calendar["DATE"]).dt.normalize()
    calendar["MONTH_KEY"] = calendar["DATE"].dt.strftime("%Y-%m")

    orders["ORDER_DATETIME"] = pd.to_datetime(orders["ORDER_DATETIME"])
    orders["ORDER_DATE"] = orders["ORDER_DATETIME"].dt.normalize()
    orders["MONTH_KEY"] = orders["ORDER_DATE"].dt.strftime("%Y-%m")

    orders = orders.merge(
        calendar[["DATE", "DAY_NAME", "DAY_TYPE", "FESTIVE_PERIOD"]],
        how="left",
        left_on="ORDER_DATE",
        right_on="DATE",
    ).drop(columns=["DATE"])

    orders = orders.merge(
        stores[["STORE_ID", "STORE_NAME", "CITY", "REGION", "STORE_FORMAT"]],
        how="left",
        on="STORE_ID",
    )

    lines = lines.merge(
        orders[["ORDER_ID", "ORDER_DATE", "MONTH_KEY", "STORE_ID", "CHANNEL"]],
        how="left",
        on="ORDER_ID",
    )
    lines = lines.merge(
        products[["SKU_ID", "SKU_NAME", "CATEGORY", "VEG_NONVEG"]],
        how="left",
        on="SKU_ID",
    )

    return Dataset(
        orders=orders,
        lines=lines,
        calendar=calendar,
        stores=stores,
        products=products,
    )


def in_window(frame: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Filter a frame to an inclusive date window.

    Args:
        frame: Frame carrying an ``ORDER_DATE`` column.
        start: First date to include.
        end: Last date to include.

    Returns:
        The rows falling inside the window.
    """
    mask = (frame["ORDER_DATE"] >= pd.Timestamp(start)) & (
        frame["ORDER_DATE"] <= pd.Timestamp(end)
    )
    return frame.loc[mask]


def calendar_days(calendar: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Select the calendar rows inside a date window.

    Args:
        calendar: The full date spine.
        start: First date to include.
        end: Last date to include.

    Returns:
        The calendar rows in the window.
    """
    mask = (calendar["DATE"] >= pd.Timestamp(start)) & (
        calendar["DATE"] <= pd.Timestamp(end)
    )
    return calendar.loc[mask]


def summarise(frame: pd.DataFrame) -> dict[str, float]:
    """Compute the headline order-grain measures for a set of orders.

    Args:
        frame: Order-grain rows.

    Returns:
        Revenue, orders, AOV, units and tax-inclusive revenue.
    """
    orders = int(frame["ORDER_ID"].nunique())
    revenue = float(frame["NET_BEFORE_TAX"].sum())
    units = int(frame["TOTAL_QTY"].sum())
    return {
        "revenue_net_inr": revenue,
        "orders": orders,
        "aov_inr": safe_ratio(revenue, orders),
        "units": units,
        "units_per_order": safe_ratio(units, orders),
        "revenue_with_tax_inr": float(frame["NET_REVENUE"].sum()),
        "gross_revenue_inr": float(frame["GROSS_BILL_VALUE"].sum()),
        "discount_inr": float(frame["DISCOUNT_AMOUNT"].sum()),
    }


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


def compute_context(data: Dataset) -> dict[str, Any]:
    """Compute the seasonality context every other answer is read against.

    Args:
        data: The loaded dataset.

    Returns:
        Full-year totals and a per-month revenue series, with the peak, the
        trough and the spread between them.
    """
    monthly = (
        data.orders.groupby("MONTH_KEY")
        .agg(
            orders=("ORDER_ID", "nunique"),
            revenue_net_inr=("NET_BEFORE_TAX", "sum"),
            units=("TOTAL_QTY", "sum"),
        )
        .reset_index()
        .sort_values("MONTH_KEY")
    )
    monthly["aov_inr"] = monthly["revenue_net_inr"] / monthly["orders"]

    festive_by_month = (
        data.calendar[data.calendar["FESTIVE_PERIOD"] != NON_FESTIVE_PERIOD]
        .groupby("MONTH_KEY")["FESTIVE_PERIOD"]
        .agg(lambda values: sorted(set(values)))
        .to_dict()
    )
    records = monthly.to_dict("records")
    for record in records:
        record["festive_periods"] = festive_by_month.get(record["MONTH_KEY"], [])

    peak = max(records, key=lambda r: r["revenue_net_inr"])
    trough = min(records, key=lambda r: r["revenue_net_inr"])

    return {
        "full_year": summarise(data.orders),
        "monthly_revenue": records,
        "peak_month": peak["MONTH_KEY"],
        "peak_revenue_inr": peak["revenue_net_inr"],
        "trough_month": trough["MONTH_KEY"],
        "trough_revenue_inr": trough["revenue_net_inr"],
        "peak_to_trough_ratio": safe_ratio(
            peak["revenue_net_inr"], trough["revenue_net_inr"]
        ),
        "seasonality_note": (
            f"Revenue peaks in {peak['MONTH_KEY']} "
            f"({inr(peak['revenue_net_inr'])} INR, festive) and troughs in "
            f"{trough['MONTH_KEY']} ({inr(trough['revenue_net_inr'])} INR), a "
            f"{safe_ratio(peak['revenue_net_inr'], trough['revenue_net_inr']):.2f}x "
            f"spread. Any decline claim must be read against this seasonal "
            f"shape, not treated as evidence of a problem on its own."
        ),
    }


# ---------------------------------------------------------------------------
# Q1 - headline
# ---------------------------------------------------------------------------


def compute_q1(data: Dataset, start: date, end: date) -> dict[str, Any]:
    """Q1: headline performance for the analysis window.

    Args:
        data: The loaded dataset.
        start: First date of the window.
        end: Last date of the window.

    Returns:
        Headline totals plus a per-month breakdown so the trend is visible.
    """
    window = in_window(data.orders, start, end)
    totals = summarise(window)

    monthly = (
        window.groupby("MONTH_KEY")
        .agg(
            orders=("ORDER_ID", "nunique"),
            revenue_net_inr=("NET_BEFORE_TAX", "sum"),
            revenue_with_tax_inr=("NET_REVENUE", "sum"),
            units=("TOTAL_QTY", "sum"),
        )
        .reset_index()
        .sort_values("MONTH_KEY")
    )
    monthly["aov_inr"] = monthly["revenue_net_inr"] / monthly["orders"]
    monthly["units_per_order"] = monthly["units"] / monthly["orders"]

    return {
        "question": "What was our total revenue, order count and AOV in the last 3 months?",
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "grain": "order",
        **totals,
        "monthly": monthly.to_dict("records"),
    }


# ---------------------------------------------------------------------------
# Q2 - store ranking
# ---------------------------------------------------------------------------


def compute_q2(data: Dataset, start: date, end: date) -> dict[str, Any]:
    """Q2: best and worst performing stores by revenue.

    Args:
        data: The loaded dataset.
        start: First date of the window.
        end: Last date of the window.

    Returns:
        The full ranked store list plus the top and bottom five.
    """
    window = in_window(data.orders, start, end)
    total_revenue = float(window["NET_BEFORE_TAX"].sum())

    grouped = (
        window.groupby(["STORE_ID", "STORE_NAME", "CITY", "REGION"])
        .agg(
            revenue_net_inr=("NET_BEFORE_TAX", "sum"),
            orders=("ORDER_ID", "nunique"),
            units=("TOTAL_QTY", "sum"),
        )
        .reset_index()
    )
    grouped["aov_inr"] = grouped["revenue_net_inr"] / grouped["orders"]
    grouped["revenue_share_pct"] = grouped["revenue_net_inr"] / total_revenue * 100
    grouped = grouped.sort_values("revenue_net_inr", ascending=False).reset_index(
        drop=True
    )
    grouped["rank"] = grouped.index + 1

    records = grouped.to_dict("records")
    return {
        "question": "Which stores are performing best and worst by revenue in the last 3 months?",
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "grain": "order",
        "store_count": len(records),
        "total_revenue_inr": total_revenue,
        "top_5": records[:TOP_N],
        "bottom_5": records[-TOP_N:][::-1],
        "all_stores": records,
    }


# ---------------------------------------------------------------------------
# Q3 - channel mix
# ---------------------------------------------------------------------------


def compute_q3(data: Dataset, start: date, end: date) -> dict[str, Any]:
    """Q3: channel performance for the analysis window.

    Args:
        data: The loaded dataset.
        start: First date of the window.
        end: Last date of the window.

    Returns:
        Per-channel revenue, orders, AOV, units, share and basket size.
    """
    window = in_window(data.orders, start, end)
    total_revenue = float(window["NET_BEFORE_TAX"].sum())
    total_orders = int(window["ORDER_ID"].nunique())

    grouped = (
        window.groupby("CHANNEL")
        .agg(
            revenue_net_inr=("NET_BEFORE_TAX", "sum"),
            orders=("ORDER_ID", "nunique"),
            units=("TOTAL_QTY", "sum"),
        )
        .reset_index()
    )
    grouped["aov_inr"] = grouped["revenue_net_inr"] / grouped["orders"]
    grouped["revenue_share_pct"] = grouped["revenue_net_inr"] / total_revenue * 100
    grouped["order_share_pct"] = grouped["orders"] / total_orders * 100
    grouped["units_per_order"] = grouped["units"] / grouped["orders"]
    grouped = grouped.sort_values("revenue_net_inr", ascending=False)

    return {
        "question": "How do our sales channels compare in the last 3 months?",
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "grain": "order",
        "total_revenue_inr": total_revenue,
        "total_orders": total_orders,
        "channels": grouped.to_dict("records"),
    }


# ---------------------------------------------------------------------------
# Q4 - product ranking
# ---------------------------------------------------------------------------


def compute_q4(data: Dataset, start: date, end: date) -> dict[str, Any]:
    """Q4: top selling SKUs by volume and by revenue.

    This is the only question computed at line grain, because SKU identity does
    not exist on the order header.

    Args:
        data: The loaded dataset.
        start: First date of the window.
        end: Last date of the window.

    Returns:
        The top five SKUs by quantity and by line revenue, with a note
        explaining why line-grain revenue differs slightly from the canonical
        order-grain figure.
    """
    window_lines = in_window(data.lines, start, end)
    window_orders = in_window(data.orders, start, end)

    grouped = (
        window_lines.groupby(["SKU_ID", "SKU_NAME", "CATEGORY"])
        .agg(
            quantity=("QUANTITY", "sum"),
            line_revenue_inr=("LINE_NET_VALUE", "sum"),
            orders=("ORDER_ID", "nunique"),
            est_cogs_inr=("EST_COGS", "sum"),
        )
        .reset_index()
    )
    grouped["gross_margin_inr"] = (
        grouped["line_revenue_inr"] - grouped["est_cogs_inr"]
    )
    grouped["margin_pct"] = (
        grouped["gross_margin_inr"] / grouped["line_revenue_inr"] * 100
    )

    line_total = float(window_lines["LINE_NET_VALUE"].sum())
    order_total = float(window_orders["NET_BEFORE_TAX"].sum())
    variance = line_total - order_total

    by_quantity = grouped.sort_values("quantity", ascending=False).head(TOP_N)
    by_revenue = grouped.sort_values("line_revenue_inr", ascending=False).head(TOP_N)

    return {
        "question": "Which products sell the most in the last 3 months?",
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "grain": "line",
        "top_5_by_quantity": by_quantity.to_dict("records"),
        "top_5_by_revenue": by_revenue.to_dict("records"),
        "line_revenue_total_inr": line_total,
        "order_revenue_total_inr": order_total,
        "grain_variance_inr": variance,
        "grain_variance_pct": safe_pct_change(order_total, line_total),
        "note": (
            "These figures are LINE GRAIN, computed from Order_Details, because "
            "SKU identity exists only on the line. Line revenue in this window "
            f"totals {inr(line_total)} INR against the canonical order-grain "
            f"figure of {inr(order_total)} INR, a difference of "
            f"{inr(variance)} INR "
            f"({safe_pct_change(order_total, line_total):+.4f}%). The gap comes "
            "from orders in the source workbook whose line values do not sum to "
            "their header NET_BEFORE_TAX. IMPORTANT: every one of those orders "
            "falls inside this three-month window - the preceding nine months "
            "have none - so the variance is about 0.44% of revenue here against "
            "0.11% across the full year. Use these numbers for product mix, "
            "ranking and margin; use the order grain (Q1) for any revenue total."
        ),
    }


# ---------------------------------------------------------------------------
# Q5 - city trend
# ---------------------------------------------------------------------------


def monthly_pivot(
    frame: pd.DataFrame, index: list[str], months: list[str]
) -> pd.DataFrame:
    """Pivot revenue into one column per month.

    Args:
        frame: Order-grain rows.
        index: Grouping columns.
        months: Month keys to produce, in order. Missing months become 0.

    Returns:
        One row per group with one revenue column per month key.
    """
    pivot = frame.pivot_table(
        index=index,
        columns="MONTH_KEY",
        values="NET_BEFORE_TAX",
        aggfunc="sum",
        fill_value=0.0,
    )
    for month in months:
        if month not in pivot.columns:
            pivot[month] = 0.0
    return pivot[months].reset_index()


def compute_q5(
    data: Dataset, start: date, end: date, months: list[str]
) -> dict[str, Any]:
    """Q5: cities whose revenue is declining across the window.

    Args:
        data: The loaded dataset.
        start: First date of the window.
        end: Last date of the window.
        months: The three month keys in the window, in order.

    Returns:
        The full per-city table and the filtered declining list, sorted by
        percent change ascending.
    """
    window = in_window(data.orders, start, end)
    pivot = monthly_pivot(window, ["CITY"], months)

    records: list[dict[str, Any]] = []
    for row in pivot.to_dict("records"):
        values = [float(row[month]) for month in months]
        first, last = values[0], values[-1]
        strictly_declining = all(
            values[i] > values[i + 1] for i in range(len(values) - 1)
        )
        records.append(
            {
                "city": row["CITY"],
                "monthly_revenue_inr": dict(zip(months, values)),
                "first_month": months[0],
                "last_month": months[-1],
                "first_month_revenue_inr": first,
                "last_month_revenue_inr": last,
                "change_inr": last - first,
                "change_pct": safe_pct_change(first, last),
                "strictly_declining": strictly_declining,
                "last_below_first": last < first,
            }
        )

    records.sort(key=lambda record: record["change_pct"])
    declining = [record for record in records if record["strictly_declining"]]

    return {
        "question": "Are any cities showing declining revenue in the last 3 months?",
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "grain": "order",
        "months": months,
        "city_count": len(records),
        "cities": records,
        "declining_cities": declining,
        "declining_city_count": len(declining),
        "note": (
            "'strictly_declining' means revenue fell in every consecutive month. "
            "'last_below_first' is the weaker test of the endpoint alone."
        ),
    }


# ---------------------------------------------------------------------------
# Q6 - weekend vs weekday
# ---------------------------------------------------------------------------


def day_type_breakdown(
    orders: pd.DataFrame, calendar: pd.DataFrame, start: date, end: date
) -> dict[str, Any]:
    """Compare weekend and weekday trading over a window, normalised per day.

    Raw totals mislead here: a window holds roughly 2.5 times more weekdays than
    weekend days, so weekday revenue is larger even when each weekend day trades
    far harder. Every comparison is therefore made per trading day.

    Args:
        orders: Order-grain rows.
        calendar: The full date spine.
        start: First date of the window.
        end: Last date of the window.

    Returns:
        Per-day-type measures and the weekend-to-weekday daily revenue ratio.
    """
    window = in_window(orders, start, end)
    days = calendar_days(calendar, start, end)
    days_by_type = days.groupby("DAY_TYPE")["DATE"].nunique().to_dict()

    grouped = (
        window.groupby("DAY_TYPE")
        .agg(
            revenue_net_inr=("NET_BEFORE_TAX", "sum"),
            orders=("ORDER_ID", "nunique"),
            units=("TOTAL_QTY", "sum"),
        )
        .reset_index()
    )
    grouped["distinct_days"] = grouped["DAY_TYPE"].map(days_by_type)
    grouped["aov_inr"] = grouped["revenue_net_inr"] / grouped["orders"]
    grouped["avg_revenue_per_day_inr"] = (
        grouped["revenue_net_inr"] / grouped["distinct_days"]
    )
    grouped["avg_orders_per_day"] = grouped["orders"] / grouped["distinct_days"]
    grouped["units_per_order"] = grouped["units"] / grouped["orders"]

    records = {row["DAY_TYPE"]: row for row in grouped.to_dict("records")}
    weekend = records.get(WEEKEND_DAY_TYPE, {})
    weekday = records.get(WEEKDAY_DAY_TYPE, {})

    return {
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "day_types": grouped.to_dict("records"),
        "weekend_to_weekday_daily_revenue_ratio": safe_ratio(
            float(weekend.get("avg_revenue_per_day_inr", 0.0)),
            float(weekday.get("avg_revenue_per_day_inr", 0.0)),
        ),
        "weekend_to_weekday_daily_orders_ratio": safe_ratio(
            float(weekend.get("avg_orders_per_day", 0.0)),
            float(weekday.get("avg_orders_per_day", 0.0)),
        ),
        "weekend_to_weekday_aov_ratio": safe_ratio(
            float(weekend.get("aov_inr", 0.0)), float(weekday.get("aov_inr", 0.0))
        ),
    }


def compute_q6(data: Dataset, start: date, end: date) -> dict[str, Any]:
    """Q6: weekend versus weekday performance, full year and last 3 months.

    Args:
        data: The loaded dataset.
        start: First date of the recent window.
        end: Last date of the recent window.

    Returns:
        The comparison over both windows, with a note on why per-day
        normalisation is required.
    """
    return {
        "question": "How does weekend trading compare with weekday trading?",
        "grain": "order",
        "full_year": day_type_breakdown(
            data.orders,
            data.calendar,
            settings.DATA_START_DATE,
            settings.DATA_ASOF_DATE,
        ),
        "last_3_months": day_type_breakdown(
            data.orders, data.calendar, start, end
        ),
        "note": (
            "Compare avg_revenue_per_day_inr, never the raw totals. A year holds "
            "about 2.5x more weekdays than weekend days, so weekday revenue is "
            "larger in total even though each weekend day trades far harder."
        ),
    }


# ---------------------------------------------------------------------------
# Q7 - festive vs normal
# ---------------------------------------------------------------------------


def compute_q7(data: Dataset) -> dict[str, Any]:
    """Q7: festive period uplift against normal trading, over the full year.

    Args:
        data: The loaded dataset.

    Returns:
        Each festive period individually and all festive combined, compared
        against Normal on a per-trading-day basis.
    """
    orders = data.orders
    calendar = data.calendar
    days_by_period = calendar.groupby("FESTIVE_PERIOD")["DATE"].nunique().to_dict()

    def block(label: str, mask: pd.Series, distinct_days: int) -> dict[str, Any]:
        """Summarise one slice of the year.

        Args:
            label: Name of the period.
            mask: Boolean mask selecting the orders.
            distinct_days: Calendar days the period covers.

        Returns:
            Revenue, orders, AOV and per-day averages for the slice.
        """
        subset = orders.loc[mask]
        revenue = float(subset["NET_BEFORE_TAX"].sum())
        order_count = int(subset["ORDER_ID"].nunique())
        units = int(subset["TOTAL_QTY"].sum())
        return {
            "period": label,
            "revenue_net_inr": revenue,
            "orders": order_count,
            "units": units,
            "aov_inr": safe_ratio(revenue, order_count),
            "distinct_days": distinct_days,
            "avg_revenue_per_day_inr": safe_ratio(revenue, distinct_days),
            "avg_orders_per_day": safe_ratio(order_count, distinct_days),
            "units_per_order": safe_ratio(units, order_count),
        }

    normal = block(
        NON_FESTIVE_PERIOD,
        orders["FESTIVE_PERIOD"] == NON_FESTIVE_PERIOD,
        int(days_by_period.get(NON_FESTIVE_PERIOD, 0)),
    )

    periods: list[dict[str, Any]] = []
    for period in settings.FESTIVE_PERIODS:
        entry = block(
            period,
            orders["FESTIVE_PERIOD"] == period,
            int(days_by_period.get(period, 0)),
        )
        entry["revenue_uplift_vs_normal_x"] = safe_ratio(
            entry["avg_revenue_per_day_inr"], normal["avg_revenue_per_day_inr"]
        )
        entry["orders_uplift_vs_normal_x"] = safe_ratio(
            entry["avg_orders_per_day"], normal["avg_orders_per_day"]
        )
        entry["aov_uplift_vs_normal_x"] = safe_ratio(
            entry["aov_inr"], normal["aov_inr"]
        )
        periods.append(entry)

    combined_days = sum(
        int(days_by_period.get(period, 0)) for period in settings.FESTIVE_PERIODS
    )
    combined = block(
        "All festive",
        orders["FESTIVE_PERIOD"].isin(settings.FESTIVE_PERIODS),
        combined_days,
    )
    combined["revenue_uplift_vs_normal_x"] = safe_ratio(
        combined["avg_revenue_per_day_inr"], normal["avg_revenue_per_day_inr"]
    )
    combined["orders_uplift_vs_normal_x"] = safe_ratio(
        combined["avg_orders_per_day"], normal["avg_orders_per_day"]
    )
    combined["aov_uplift_vs_normal_x"] = safe_ratio(
        combined["aov_inr"], normal["aov_inr"]
    )

    return {
        "question": "How much do festive periods lift trading versus normal days?",
        "window": {
            "start": settings.DATA_START_DATE.isoformat(),
            "end": settings.DATA_ASOF_DATE.isoformat(),
        },
        "grain": "order",
        "normal": normal,
        "festive_periods": periods,
        "all_festive_combined": combined,
        "note": (
            "Uplift is measured per trading day, because the festive periods "
            "cover far fewer days than Normal. Raw totals would understate the "
            "effect."
        ),
    }


# ---------------------------------------------------------------------------
# Q8 - declining stores with diagnostics
# ---------------------------------------------------------------------------


def store_diagnostics(
    data: Dataset,
    store_id: str,
    months: list[str],
    start: date,
    end: date,
    prior_start: date,
    prior_end: date,
    city_trends: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build the evidence needed to explain one store's decline.

    Answers four diagnostic questions: is it fewer orders or smaller baskets, is
    the fall concentrated in one channel, is the store losing customers, and is
    this a break in the store's own trend or a market-wide move.

    Args:
        data: The loaded dataset.
        store_id: The store under investigation.
        months: The three month keys in the window, in order.
        start: First date of the window.
        end: Last date of the window.
        prior_start: First date of the preceding comparison window.
        prior_end: Last date of the preceding comparison window.
        city_trends: Per-city trend records keyed by city name.

    Returns:
        The store's monthly measures, channel breakdown, customer counts,
        prior-window comparison and city context.
    """
    store_orders = data.orders[data.orders["STORE_ID"] == store_id]
    window = in_window(store_orders, start, end)
    prior = in_window(store_orders, prior_start, prior_end)

    monthly = (
        window.groupby("MONTH_KEY")
        .agg(
            revenue_net_inr=("NET_BEFORE_TAX", "sum"),
            orders=("ORDER_ID", "nunique"),
            units=("TOTAL_QTY", "sum"),
        )
        .reindex(months, fill_value=0)
        .reset_index()
    )
    monthly["aov_inr"] = monthly.apply(
        lambda row: safe_ratio(row["revenue_net_inr"], row["orders"]), axis=1
    )
    monthly["units_per_order"] = monthly.apply(
        lambda row: safe_ratio(row["units"], row["orders"]), axis=1
    )
    # Anonymous walk-ins have no customer id, so distinct customers counts only
    # identified ones. It is a directional signal, not a footfall count.
    customers = (
        window[window["CUSTOMER_ID"].notna()]
        .groupby("MONTH_KEY")["CUSTOMER_ID"]
        .nunique()
        .reindex(months, fill_value=0)
    )
    monthly["distinct_identified_customers"] = customers.to_numpy()

    monthly_records = monthly.to_dict("records")
    first, last = monthly_records[0], monthly_records[-1]

    channel_pivot = monthly_pivot(window, ["CHANNEL"], months)
    channels: list[dict[str, Any]] = []
    for row in channel_pivot.to_dict("records"):
        values = [float(row[month]) for month in months]
        channels.append(
            {
                "channel": row["CHANNEL"],
                "monthly_revenue_inr": dict(zip(months, values)),
                "change_inr": values[-1] - values[0],
                "change_pct": safe_pct_change(values[0], values[-1]),
                "strictly_declining": all(
                    values[i] > values[i + 1] for i in range(len(values) - 1)
                ),
            }
        )
    channels.sort(key=lambda entry: entry["change_inr"])

    window_revenue = float(window["NET_BEFORE_TAX"].sum())
    prior_revenue = float(prior["NET_BEFORE_TAX"].sum())
    total_change = last["revenue_net_inr"] - first["revenue_net_inr"]
    worst_channel = channels[0] if channels else None

    city = str(store_orders["CITY"].iloc[0])
    city_trend = city_trends.get(city, {})

    # Attribute the fall between order volume and basket size, holding the other
    # constant in turn.
    orders_effect = (first["aov_inr"]) * (last["orders"] - first["orders"])
    aov_effect = (first["orders"]) * (last["aov_inr"] - first["aov_inr"])

    return {
        "monthly": monthly_records,
        "orders_change_pct": safe_pct_change(first["orders"], last["orders"]),
        "aov_change_pct": safe_pct_change(first["aov_inr"], last["aov_inr"]),
        "units_per_order_change_pct": safe_pct_change(
            first["units_per_order"], last["units_per_order"]
        ),
        "identified_customers_change_pct": safe_pct_change(
            first["distinct_identified_customers"],
            last["distinct_identified_customers"],
        ),
        "decomposition": {
            "revenue_change_inr": total_change,
            "attributable_to_order_volume_inr": orders_effect,
            "attributable_to_basket_size_inr": aov_effect,
            "primary_driver": (
                "order volume"
                if abs(orders_effect) >= abs(aov_effect)
                else "basket size"
            ),
        },
        "by_channel": channels,
        "worst_channel": worst_channel["channel"] if worst_channel else None,
        "decline_concentrated_in_one_channel": bool(
            worst_channel
            and total_change < 0
            and safe_ratio(worst_channel["change_inr"], total_change) >= 0.5
        ),
        "prior_window": {
            "start": prior_start.isoformat(),
            "end": prior_end.isoformat(),
            "revenue_net_inr": prior_revenue,
        },
        "current_window_revenue_inr": window_revenue,
        "vs_prior_window_change_inr": window_revenue - prior_revenue,
        "vs_prior_window_change_pct": safe_pct_change(prior_revenue, window_revenue),
        "is_break_in_trend": window_revenue < prior_revenue,
        "city": city,
        "city_trend": {
            "change_pct": city_trend.get("change_pct"),
            "strictly_declining": city_trend.get("strictly_declining"),
            "monthly_revenue_inr": city_trend.get("monthly_revenue_inr"),
        },
        "store_specific_vs_market_wide": (
            "market-wide"
            if city_trend.get("strictly_declining")
            else "store-specific"
        ),
    }


def compute_q8(
    data: Dataset,
    start: date,
    end: date,
    months: list[str],
    city_trends: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Q8: stores declining in both consecutive months, with diagnostics.

    Args:
        data: The loaded dataset.
        start: First date of the window.
        end: Last date of the window.
        months: The three month keys in the window, in order.
        city_trends: Per-city trend records keyed by city name.

    Returns:
        The full per-store table and the declining stores with their
        diagnostics attached.
    """
    prior_start = shift_months(start, -WINDOW_MONTHS)
    prior_end = start - timedelta(days=1)

    window = in_window(data.orders, start, end)
    pivot = monthly_pivot(window, ["STORE_ID", "STORE_NAME", "CITY", "REGION"], months)

    records: list[dict[str, Any]] = []
    for row in pivot.to_dict("records"):
        values = [float(row[month]) for month in months]
        records.append(
            {
                "store_id": row["STORE_ID"],
                "store_name": row["STORE_NAME"],
                "city": row["CITY"],
                "region": row["REGION"],
                "monthly_revenue_inr": dict(zip(months, values)),
                "window_revenue_inr": sum(values),
                "change_inr": values[-1] - values[0],
                "change_pct": safe_pct_change(values[0], values[-1]),
                "declined_every_month": all(
                    values[i] > values[i + 1] for i in range(len(values) - 1)
                ),
            }
        )
    records.sort(key=lambda record: record["change_pct"])

    declining: list[dict[str, Any]] = []
    for record in records:
        if not record["declined_every_month"]:
            continue
        entry = dict(record)
        entry["diagnostics"] = store_diagnostics(
            data,
            record["store_id"],
            months,
            start,
            end,
            prior_start,
            prior_end,
            city_trends,
        )
        declining.append(entry)

    return {
        "question": (
            "Which stores have consistently declining revenue, and why are they "
            "declining?"
        ),
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "prior_window": {
            "start": prior_start.isoformat(),
            "end": prior_end.isoformat(),
        },
        "grain": "order",
        "months": months,
        "store_count": len(records),
        "declining_store_count": len(declining),
        "stores": records,
        "declining_stores": declining,
        "note": (
            "'declined_every_month' requires revenue to fall in every "
            "consecutive month in the window, which is a stricter and more "
            "meaningful test than simply comparing the endpoints."
        ),
    }


# ---------------------------------------------------------------------------
# Console rendering
# ---------------------------------------------------------------------------


def print_context(context: dict[str, Any]) -> None:
    """Print the seasonality context block.

    Args:
        context: The computed context.
    """
    heading("CONTEXT - FULL YEAR SEASONALITY")
    totals = context["full_year"]
    key_value("Revenue (net_before_tax)", f"{inr(totals['revenue_net_inr'])} INR")
    key_value("Orders", f"{totals['orders']:,}")
    key_value("AOV", f"{inr(totals['aov_inr'])} INR")
    key_value("Units", f"{totals['units']:,}")
    print()
    print_table(
        ["MONTH", "ORDERS", "REVENUE (INR)", "AOV (INR)", "FESTIVE"],
        [
            [
                row["MONTH_KEY"],
                f"{row['orders']:,}",
                inr(row["revenue_net_inr"]),
                inr(row["aov_inr"]),
                ", ".join(row["festive_periods"]) or "-",
            ]
            for row in context["monthly_revenue"]
        ],
        aligns="lrrrl",
    )
    print()
    print(f"  {context['seasonality_note']}")


def print_q1(answer: dict[str, Any]) -> None:
    """Print the Q1 headline block.

    Args:
        answer: The Q1 result.
    """
    heading(
        f"Q1 - LAST 3 MONTHS HEADLINE ({answer['window']['start']} to "
        f"{answer['window']['end']})"
    )
    key_value("Revenue (net_before_tax)", f"{inr(answer['revenue_net_inr'])} INR")
    key_value("Revenue (incl. 5% tax)", f"{inr(answer['revenue_with_tax_inr'])} INR")
    key_value("Orders", f"{answer['orders']:,}")
    key_value("AOV", f"{inr(answer['aov_inr'])} INR")
    key_value("Units", f"{answer['units']:,}")
    key_value("Units per order", f"{answer['units_per_order']:.2f}")
    print()
    print_table(
        ["MONTH", "ORDERS", "REVENUE (INR)", "AOV (INR)", "UNITS"],
        [
            [
                row["MONTH_KEY"],
                f"{row['orders']:,}",
                inr(row["revenue_net_inr"]),
                inr(row["aov_inr"]),
                f"{row['units']:,}",
            ]
            for row in answer["monthly"]
        ],
    )


def print_q2(answer: dict[str, Any]) -> None:
    """Print the Q2 store ranking block.

    Args:
        answer: The Q2 result.
    """
    heading("Q2 - TOP AND BOTTOM STORES BY REVENUE (LAST 3 MONTHS)")
    columns = ["RANK", "STORE", "CITY", "REGION", "REVENUE (INR)", "ORDERS", "AOV", "SHARE"]

    def rows(records: list[dict[str, Any]]) -> list[list[str]]:
        """Render store records as table rows.

        Args:
            records: Store result records.

        Returns:
            Formatted rows.
        """
        return [
            [
                str(record["rank"]),
                record["STORE_NAME"],
                record["CITY"],
                record["REGION"],
                inr(record["revenue_net_inr"]),
                f"{record['orders']:,}",
                inr(record["aov_inr"]),
                f"{record['revenue_share_pct']:.2f}%",
            ]
            for record in records
        ]

    subheading("TOP 5")
    print_table(columns, rows(answer["top_5"]), aligns="rlllrrrr")
    subheading("BOTTOM 5")
    print_table(columns, rows(answer["bottom_5"]), aligns="rlllrrrr")


def print_q3(answer: dict[str, Any]) -> None:
    """Print the Q3 channel block.

    Args:
        answer: The Q3 result.
    """
    heading("Q3 - CHANNEL PERFORMANCE (LAST 3 MONTHS)")
    print_table(
        ["CHANNEL", "REVENUE (INR)", "SHARE", "ORDERS", "AOV (INR)", "UNITS", "UNITS/ORDER"],
        [
            [
                row["CHANNEL"],
                inr(row["revenue_net_inr"]),
                f"{row['revenue_share_pct']:.2f}%",
                f"{row['orders']:,}",
                inr(row["aov_inr"]),
                f"{row['units']:,}",
                f"{row['units_per_order']:.2f}",
            ]
            for row in answer["channels"]
        ],
    )


def print_q4(answer: dict[str, Any]) -> None:
    """Print the Q4 product block.

    Args:
        answer: The Q4 result.
    """
    heading("Q4 - TOP SKUS (LAST 3 MONTHS, LINE GRAIN)")
    columns = ["SKU", "NAME", "CATEGORY", "QTY", "LINE REVENUE (INR)", "ORDERS", "MARGIN %"]

    def rows(records: list[dict[str, Any]]) -> list[list[str]]:
        """Render SKU records as table rows.

        Args:
            records: SKU result records.

        Returns:
            Formatted rows.
        """
        return [
            [
                record["SKU_ID"],
                record["SKU_NAME"],
                record["CATEGORY"],
                f"{record['quantity']:,}",
                inr(record["line_revenue_inr"]),
                f"{record['orders']:,}",
                f"{record['margin_pct']:.2f}%",
            ]
            for record in records
        ]

    subheading("TOP 5 BY QUANTITY")
    print_table(columns, rows(answer["top_5_by_quantity"]), aligns="lllrrrr")
    subheading("TOP 5 BY LINE REVENUE")
    print_table(columns, rows(answer["top_5_by_revenue"]), aligns="lllrrrr")
    print()
    print(f"  NOTE: {answer['note']}")


def print_q5(answer: dict[str, Any]) -> None:
    """Print the Q5 city trend block.

    Args:
        answer: The Q5 result.
    """
    heading("Q5 - CITY REVENUE TREND (LAST 3 MONTHS)")
    months = answer["months"]
    print_table(
        ["CITY", *[m.upper() for m in months], "CHANGE (INR)", "CHANGE %", "EVERY MONTH?"],
        [
            [
                row["city"],
                *[inr(row["monthly_revenue_inr"][m]) for m in months],
                inr(row["change_inr"]),
                pct(row["change_pct"]),
                "YES" if row["strictly_declining"] else "no",
            ]
            for row in answer["cities"]
        ],
    )
    print()
    if answer["declining_cities"]:
        names = ", ".join(row["city"] for row in answer["declining_cities"])
        print(f"  DECLINING EVERY MONTH ({answer['declining_city_count']}): {names}")
    else:
        print("  No city declined in every consecutive month.")


def print_q6(answer: dict[str, Any]) -> None:
    """Print the Q6 weekend/weekday block.

    Args:
        answer: The Q6 result.
    """
    heading("Q6 - WEEKEND VS WEEKDAY")
    for label, key in (("FULL YEAR", "full_year"), ("LAST 3 MONTHS", "last_3_months")):
        block = answer[key]
        subheading(f"{label} ({block['window']['start']} to {block['window']['end']})")
        print_table(
            [
                "DAY TYPE",
                "REVENUE (INR)",
                "ORDERS",
                "AOV (INR)",
                "DAYS",
                "REV/DAY (INR)",
                "ORDERS/DAY",
            ],
            [
                [
                    row["DAY_TYPE"],
                    inr(row["revenue_net_inr"]),
                    f"{row['orders']:,}",
                    inr(row["aov_inr"]),
                    f"{row['distinct_days']:,}",
                    inr(row["avg_revenue_per_day_inr"]),
                    f"{row['avg_orders_per_day']:.2f}",
                ]
                for row in block["day_types"]
            ],
        )
        print()
        key_value(
            "Weekend:weekday revenue per day",
            f"{block['weekend_to_weekday_daily_revenue_ratio']:.3f}x",
        )
        key_value(
            "Weekend:weekday orders per day",
            f"{block['weekend_to_weekday_daily_orders_ratio']:.3f}x",
        )
        key_value(
            "Weekend:weekday AOV",
            f"{block['weekend_to_weekday_aov_ratio']:.3f}x",
        )
    print()
    print(f"  NOTE: {answer['note']}")


def print_q7(answer: dict[str, Any]) -> None:
    """Print the Q7 festive block.

    Args:
        answer: The Q7 result.
    """
    heading("Q7 - FESTIVE VS NORMAL (FULL YEAR)")
    rows = [answer["normal"], *answer["festive_periods"], answer["all_festive_combined"]]
    print_table(
        [
            "PERIOD",
            "REVENUE (INR)",
            "ORDERS",
            "AOV (INR)",
            "DAYS",
            "REV/DAY (INR)",
            "ORDERS/DAY",
            "UPLIFT",
        ],
        [
            [
                row["period"],
                inr(row["revenue_net_inr"]),
                f"{row['orders']:,}",
                inr(row["aov_inr"]),
                f"{row['distinct_days']:,}",
                inr(row["avg_revenue_per_day_inr"]),
                f"{row['avg_orders_per_day']:.2f}",
                f"{row['revenue_uplift_vs_normal_x']:.3f}x"
                if "revenue_uplift_vs_normal_x" in row
                else "baseline",
            ]
            for row in rows
        ],
    )
    print()
    print(f"  NOTE: {answer['note']}")


def print_q8(answer: dict[str, Any]) -> None:
    """Print the Q8 declining stores block with diagnostics.

    Args:
        answer: The Q8 result.
    """
    heading("Q8 - CONSISTENTLY DECLINING STORES (LAST 3 MONTHS)")
    months = answer["months"]
    key_value("Stores analysed", f"{answer['store_count']}")
    key_value("Declining every month", f"{answer['declining_store_count']}")
    print()
    subheading("WORST 10 STORES BY CHANGE")
    print_table(
        ["STORE", "NAME", "CITY", *[m.upper() for m in months], "CHANGE %", "EVERY MONTH?"],
        [
            [
                row["store_id"],
                row["store_name"],
                row["city"],
                *[inr(row["monthly_revenue_inr"][m]) for m in months],
                pct(row["change_pct"]),
                "YES" if row["declined_every_month"] else "no",
            ]
            for row in answer["stores"][:10]
        ],
        aligns="lll" + "r" * (len(months) + 2),
    )

    for store in answer["declining_stores"]:
        diagnostics = store["diagnostics"]
        subheading(
            f"DIAGNOSTIC: {store['store_id']} {store['store_name']} "
            f"({store['city']}, {store['region']}) {pct(store['change_pct'])}"
        )
        print_table(
            ["MONTH", "REVENUE (INR)", "ORDERS", "AOV (INR)", "UNITS/ORDER", "CUSTOMERS"],
            [
                [
                    row["MONTH_KEY"],
                    inr(row["revenue_net_inr"]),
                    f"{row['orders']:,}",
                    inr(row["aov_inr"]),
                    f"{row['units_per_order']:.2f}",
                    f"{row['distinct_identified_customers']:,}",
                ]
                for row in diagnostics["monthly"]
            ],
        )
        print()
        decomposition = diagnostics["decomposition"]
        key_value("Orders change", pct(diagnostics["orders_change_pct"]))
        key_value("AOV change", pct(diagnostics["aov_change_pct"]))
        key_value("Units/order change", pct(diagnostics["units_per_order_change_pct"]))
        key_value(
            "Identified customers change",
            pct(diagnostics["identified_customers_change_pct"]),
        )
        key_value("Primary driver", decomposition["primary_driver"])
        key_value(
            "  from order volume",
            f"{inr(decomposition['attributable_to_order_volume_inr'])} INR",
        )
        key_value(
            "  from basket size",
            f"{inr(decomposition['attributable_to_basket_size_inr'])} INR",
        )
        print()
        print_table(
            ["CHANNEL", *[m.upper() for m in months], "CHANGE (INR)", "CHANGE %"],
            [
                [
                    row["channel"],
                    *[inr(row["monthly_revenue_inr"][m]) for m in months],
                    inr(row["change_inr"]),
                    pct(row["change_pct"]),
                ]
                for row in diagnostics["by_channel"]
            ],
        )
        print()
        key_value("Worst channel", str(diagnostics["worst_channel"]))
        key_value(
            "Concentrated in one channel",
            "YES" if diagnostics["decline_concentrated_in_one_channel"] else "no",
        )
        key_value(
            f"vs prior 3 months ({diagnostics['prior_window']['start']} to "
            f"{diagnostics['prior_window']['end']})",
            f"{inr(diagnostics['prior_window']['revenue_net_inr'])} INR -> "
            f"{inr(diagnostics['current_window_revenue_inr'])} INR "
            f"({pct(diagnostics['vs_prior_window_change_pct'])})",
            label_width=52,
        )
        key_value(
            f"City trend ({diagnostics['city']})",
            f"{pct(diagnostics['city_trend']['change_pct'])} over the window, "
            f"declining every month: "
            f"{'YES' if diagnostics['city_trend']['strictly_declining'] else 'no'}",
            label_width=52,
        )
        key_value(
            "Verdict", diagnostics["store_specific_vs_market_wide"], label_width=52
        )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class GoldenJSONEncoder(json.JSONEncoder):
    """JSON encoder that understands numpy and pandas scalar types."""

    def default(self, o: Any) -> Any:
        """Convert a non-native value to something JSON can hold.

        Args:
            o: The value to convert.

        Returns:
            A JSON-serializable equivalent.
        """
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (pd.Timestamp, datetime, date)):
            return o.isoformat()
        if o is pd.NaT or (isinstance(o, float) and pd.isna(o)):
            return None
        return super().default(o)


def normalise(value: Any, places: int = 2) -> Any:
    """Recursively convert numpy types to Python types and round floats.

    Args:
        value: Any nested structure of dicts, lists and scalars.
        places: Decimal places to round floats to.

    Returns:
        The same structure with native Python types and rounded floats.
    """
    if isinstance(value, dict):
        return {key: normalise(item, places) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalise(item, places) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if pd.isna(value) else round(float(value), places)
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if value is None or value is pd.NaT:
        return None
    return value


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def compute_all() -> dict[str, Any]:
    """Compute every golden answer.

    Returns:
        The complete payload: metadata, context and Q1 through Q8.
    """
    start = settings.LAST_3M_START
    end = settings.LAST_3M_END
    months = month_keys_between(start, end)

    data = load_dataset(settings.EXCEL_PATH)
    context = compute_context(data)

    q5 = compute_q5(data, start, end, months)
    city_trends = {record["city"]: record for record in q5["cities"]}

    payload = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_file": settings.EXCEL_PATH.name,
            "source_path": str(settings.EXCEL_PATH),
            "computed_from": (
                "Excel workbook via pandas - an independent path that never "
                "reads the SQLite database, the ETL module or the semantic layer"
            ),
            "data_asof_date": settings.DATA_ASOF_DATE.isoformat(),
            "data_start_date": settings.DATA_START_DATE.isoformat(),
            "analysis_window": {
                "start": start.isoformat(),
                "end": end.isoformat(),
                "months": months,
            },
            "revenue_metric": settings.REVENUE_METRIC,
            "revenue_metric_note": (
                f"Revenue means NET_BEFORE_TAX, excluding the "
                f"{settings.TAX_RATE:.0%} tax carried in NET_REVENUE."
            ),
            "grain_rule": (
                "Revenue, order counts and AOV come from the Orders sheet "
                "(order grain). SKU, product and category figures come from "
                "Order_Details (line grain). About 216 orders have line values "
                "that do not sum to their header NET_BEFORE_TAX, so the two "
                "grains differ by roughly 0.11% of revenue. The order grain is "
                "canonical; only Q4 reports line-grain revenue and says so."
            ),
            "null_handling": (
                f"All joins are left joins. The {int(data.orders['CUSTOMER_ID'].isna().sum()):,} "
                "orders with a NULL CUSTOMER_ID are anonymous walk-ins and are "
                "never dropped."
            ),
        },
        "context": context,
        "q1": compute_q1(data, start, end),
        "q2": compute_q2(data, start, end),
        "q3": compute_q3(data, start, end),
        "q4": compute_q4(data, start, end),
        "q5": q5,
        "q6": compute_q6(data, start, end),
        "q7": compute_q7(data),
        "q8": compute_q8(data, start, end, months, city_trends),
    }
    return payload


def main() -> None:
    """Compute, print and persist the golden answers."""
    print()
    print("=" * LINE_WIDTH)
    print("QUICKBITE GOLDEN ANSWERS - GROUND TRUTH".center(LINE_WIDTH))
    print("=" * LINE_WIDTH)
    print(f"  source        : {settings.EXCEL_PATH}")
    print(f"  computed via  : pandas, direct from Excel (independent of SQLite)")
    print(f"  data as-of    : {settings.DATA_ASOF_DATE.isoformat()}")
    print(
        f"  window        : {settings.LAST_3M_START.isoformat()} to "
        f"{settings.LAST_3M_END.isoformat()}"
    )
    print(f"  revenue metric: {settings.REVENUE_METRIC} (excludes "
          f"{settings.TAX_RATE:.0%} tax)")

    payload = compute_all()

    print_context(payload["context"])
    print_q1(payload["q1"])
    print_q2(payload["q2"])
    print_q3(payload["q3"])
    print_q4(payload["q4"])
    print_q5(payload["q5"])
    print_q6(payload["q6"])
    print_q7(payload["q7"])
    print_q8(payload["q8"])

    normalised = normalise(payload)
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(
        json.dumps(normalised, indent=2, cls=GoldenJSONEncoder), encoding="utf-8"
    )

    heading("OUTPUT")
    size_kb = GOLDEN_PATH.stat().st_size / 1024
    key_value("Written to", str(GOLDEN_PATH))
    key_value("Size", f"{size_kb:,.1f} KB")
    print()


if __name__ == "__main__":
    main()
