"""Data quality gate for the QuickBite star schema.

Runs a suite of structured validations against a built database and returns a
:class:`QualityReport`. The gate is the contract between the ETL and everything
downstream: if it passes, agent-generated SQL can trust the schema's shape,
its referential integrity, its date coverage and its financial reconciliation.

Checks carry one of three severities:

``error``
    A defect that makes downstream answers wrong. Any failing error check fails
    the whole report and, when the gate runs as part of the ETL, raises.
``warning``
    A known, quantified imperfection in the source data that is documented and
    worked around rather than fixed. It does not fail the build, but it is
    always reported so it cannot be forgotten.
``info``
    An observation, never a failure. These describe the data's shape (anonymous
    order rate, cardinalities, seasonality) so the semantic layer and the agents
    can be written against reality rather than assumption.

Run standalone as a CI gate::

    python -m app.etl.quality_checks   # exit code 1 if any error check fails
"""

from __future__ import annotations

import sqlite3
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final, Sequence

from app.config import settings
from app.etl.build_db import (
    DAY_TYPES,
    NON_FESTIVE_PERIOD,
    TABLE_NAMES,
    WEEKEND_DAY_TYPE,
)

# --- Severities ------------------------------------------------------------
SEVERITY_ERROR: Final[str] = "error"
SEVERITY_WARNING: Final[str] = "warning"
SEVERITY_INFO: Final[str] = "info"

# --- Categories, in report order -------------------------------------------
CATEGORY_COMPLETENESS: Final[str] = "completeness"
CATEGORY_REFERENTIAL: Final[str] = "referential_integrity"
CATEGORY_TEMPORAL: Final[str] = "temporal"
CATEGORY_FINANCIAL: Final[str] = "financial_reconciliation"
CATEGORY_DOMAIN: Final[str] = "domain_validity"
CATEGORY_DISTRIBUTION: Final[str] = "distribution"

CATEGORY_ORDER: Final[tuple[str, ...]] = (
    CATEGORY_COMPLETENESS,
    CATEGORY_REFERENTIAL,
    CATEGORY_TEMPORAL,
    CATEGORY_FINANCIAL,
    CATEGORY_DOMAIN,
    CATEGORY_DISTRIBUTION,
)

# --- Expected dataset shape ------------------------------------------------
# Taken from the workbook's README sheet. These are assertions about the source
# data, not tunables: if the source changes, the gate must fail loudly.
EXPECTED_ROW_COUNTS: Final[dict[str, int]] = {
    "dim_store": 50,
    "dim_product": 30,
    "dim_customer": 5_000,
    "dim_promotion": 6,
    "dim_calendar": 365,
    "fact_orders": 20_000,
    "fact_order_lines": 49_834,
}

# Orders with no identified customer (anonymous walk-ins).
EXPECTED_ANONYMOUS_ORDERS: Final[int] = 5_664

# Orders with a promotion applied.
EXPECTED_PROMO_ORDERS: Final[int] = 840

# The dataset spans exactly twelve calendar months.
EXPECTED_MONTH_COUNT: Final[int] = 12

# --- Tolerances ------------------------------------------------------------
# Marts are sums of the same REAL column, so only float drift is expected.
REVENUE_TOLERANCE_INR: Final[float] = 1.0

# Per-row arithmetic identities (tax, gross less discount) should be exact to
# the paisa.
ROW_TOLERANCE_INR: Final[float] = 0.01

# An order's lines are considered to disagree with its header beyond this.
LINE_RECONCILIATION_TOLERANCE_INR: Final[float] = 1.0

# The line-to-header variance is a known defect in the source data. It stays a
# warning while it remains this small; past either threshold the grain
# disagreement is large enough to corrupt answers and becomes an error.
MAX_LINE_VARIANCE_ORDER_PCT: Final[float] = 2.0
MAX_LINE_VARIANCE_REVENUE_PCT: Final[float] = 1.0

# SQLite GLOB pattern matching a 'YYYY-MM' month key.
MONTH_KEY_GLOB: Final[str] = "[0-9][0-9][0-9][0-9]-[0-9][0-9]"

# --- Column groups checked for nulls and negatives -------------------------
# Columns that must never be NULL for a row to be usable.
REQUIRED_COLUMNS: Final[dict[str, tuple[str, ...]]] = {
    "fact_orders": (
        "order_id",
        "store_id",
        "channel",
        "net_before_tax",
        "order_date",
    ),
}

# Columns that must never be negative.
NON_NEGATIVE_COLUMNS: Final[dict[str, tuple[str, ...]]] = {
    "fact_orders": (
        "total_qty",
        "gross_bill_value",
        "discount_amount",
        "net_before_tax",
        "tax_amount",
        "net_revenue",
    ),
    "fact_order_lines": (
        "quantity",
        "unit_price",
        "line_gross_value",
        "line_discount",
        "line_net_value",
        "est_cogs",
    ),
    "dim_product": ("base_price_inr", "est_cogs_pct"),
}

MART_TABLES: Final[tuple[str, ...]] = (
    "mart_store_month",
    "mart_city_month",
    "mart_channel_month",
)


class QualityCheckError(RuntimeError):
    """Raised when the quality gate fails with one or more error checks."""


@dataclass(frozen=True)
class QualityCheck:
    """The outcome of a single validation.

    Attributes:
        name: Stable identifier for the check.
        category: Group the check belongs to, one of ``CATEGORY_ORDER``.
        severity: ``error``, ``warning`` or ``info``.
        passed: Whether the validation held. Info checks are always ``True``.
        message: Human-readable outcome, including the observed numbers.
        details: Optional structured evidence, JSON-serializable.
    """

    name: str
    category: str
    severity: str
    passed: bool
    message: str
    details: dict[str, Any] | None = None


@dataclass
class QualityReport:
    """The full set of check outcomes for one database.

    Attributes:
        checks: Every check that was run, in execution order.
        db_path: The database the checks ran against.
    """

    checks: list[QualityCheck] = field(default_factory=list)
    db_path: str = ""

    @property
    def passed(self) -> bool:
        """Whether the gate passed.

        Returns:
            True when no error-severity check failed. Failing warnings do not
            fail the gate.
        """
        return self.error_count == 0

    @property
    def error_count(self) -> int:
        """Number of failing error-severity checks.

        Returns:
            Count of checks that failed at ``error`` severity.
        """
        return sum(
            1 for c in self.checks if c.severity == SEVERITY_ERROR and not c.passed
        )

    @property
    def warning_count(self) -> int:
        """Number of failing warning-severity checks.

        Returns:
            Count of checks that failed at ``warning`` severity.
        """
        return sum(
            1 for c in self.checks if c.severity == SEVERITY_WARNING and not c.passed
        )

    @property
    def info_count(self) -> int:
        """Number of informational observations.

        Returns:
            Count of ``info`` severity checks.
        """
        return sum(1 for c in self.checks if c.severity == SEVERITY_INFO)

    def by_category(self, category: str) -> list[QualityCheck]:
        """Select the checks belonging to one category.

        Args:
            category: Category name, one of ``CATEGORY_ORDER``.

        Returns:
            The checks in that category, in execution order.
        """
        return [c for c in self.checks if c.category == category]

    def failures(self) -> list[QualityCheck]:
        """Select every check that did not pass.

        Returns:
            Failing error and warning checks, in execution order.
        """
        return [c for c in self.checks if not c.passed]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report for API exposure or logging.

        Returns:
            A JSON-serializable summary containing the verdict, the counts and
            every individual check.
        """
        return {
            "passed": self.passed,
            "db_path": self.db_path,
            "total_checks": len(self.checks),
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "info_count": self.info_count,
            "categories": {
                category: [asdict(c) for c in self.by_category(category)]
                for category in CATEGORY_ORDER
            },
            "checks": [asdict(c) for c in self.checks],
        }


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


def scalar(
    connection: sqlite3.Connection, sql: str, params: Sequence[Any] = ()
) -> Any:
    """Run a query and return the first column of the first row.

    Args:
        connection: Open SQLite connection.
        sql: Query returning a single value.
        params: Optional bound parameters.

    Returns:
        The single value produced by the query.
    """
    return connection.execute(sql, params).fetchone()[0]


def count_orphans(
    connection: sqlite3.Connection,
    child_table: str,
    child_column: str,
    parent_table: str,
    parent_column: str,
    allow_null: bool,
) -> int:
    """Count child rows whose foreign key does not resolve to a parent.

    Args:
        connection: Open SQLite connection.
        child_table: Table holding the foreign key.
        child_column: Foreign key column.
        parent_table: Referenced table.
        parent_column: Referenced column.
        allow_null: When True, NULL keys are legitimate and are not counted.

    Returns:
        Number of unresolved references.
    """
    null_clause = f"c.{child_column} IS NOT NULL AND " if allow_null else ""
    return int(
        scalar(
            connection,
            f"SELECT COUNT(*) FROM {child_table} AS c "
            f"LEFT JOIN {parent_table} AS p "
            f"  ON p.{parent_column} = c.{child_column} "
            f"WHERE {null_clause}p.{parent_column} IS NULL",
        )
    )


def percentage(part: float, whole: float) -> float:
    """Express one quantity as a percentage of another.

    Args:
        part: Numerator.
        whole: Denominator.

    Returns:
        ``part / whole * 100`` rounded to four decimals, or 0.0 when ``whole``
        is zero.
    """
    if not whole:
        return 0.0
    return round(part / whole * 100, 4)


# ---------------------------------------------------------------------------
# Completeness
# ---------------------------------------------------------------------------


def check_row_count(
    connection: sqlite3.Connection, table: str, expected: int
) -> QualityCheck:
    """Verify a table holds exactly the expected number of rows.

    Args:
        connection: Open SQLite connection.
        table: Table to count.
        expected: Row count the source dataset declares.

    Returns:
        The check outcome, with the observed and expected counts in details.
    """
    actual = int(scalar(connection, f"SELECT COUNT(*) FROM {table}"))
    passed = actual == expected
    return QualityCheck(
        name=f"row_count::{table}",
        category=CATEGORY_COMPLETENESS,
        severity=SEVERITY_ERROR,
        passed=passed,
        message=(
            f"{table} holds {actual:,} rows (expected {expected:,})"
            if passed
            else f"{table} holds {actual:,} rows but {expected:,} were expected "
            f"(delta {actual - expected:+,})"
        ),
        details={"table": table, "actual": actual, "expected": expected},
    )


def check_no_empty_tables(connection: sqlite3.Connection) -> QualityCheck:
    """Verify no table in the schema is empty.

    Args:
        connection: Open SQLite connection.

    Returns:
        The check outcome, listing any empty tables.
    """
    counts = {
        table: int(scalar(connection, f"SELECT COUNT(*) FROM {table}"))
        for table in TABLE_NAMES
    }
    empty = [table for table, count in counts.items() if count == 0]
    return QualityCheck(
        name="no_empty_tables",
        category=CATEGORY_COMPLETENESS,
        severity=SEVERITY_ERROR,
        passed=not empty,
        message=(
            f"all {len(counts)} tables are populated"
            if not empty
            else f"{len(empty)} table(s) are empty: {', '.join(empty)}"
        ),
        details={"row_counts": counts, "empty_tables": empty},
    )


def check_required_columns_not_null(connection: sqlite3.Connection) -> QualityCheck:
    """Verify columns that must always carry a value contain no NULLs.

    Nullable foreign keys (``customer_id``, ``promo_id``) are deliberately
    excluded: NULL is a valid business state for both.

    Args:
        connection: Open SQLite connection.

    Returns:
        The check outcome, with a per-column NULL count for any offender.
    """
    offenders: dict[str, int] = {}
    for table, columns in REQUIRED_COLUMNS.items():
        for column in columns:
            nulls = int(
                scalar(
                    connection,
                    f"SELECT COUNT(*) FROM {table} WHERE {column} IS NULL",
                )
            )
            if nulls:
                offenders[f"{table}.{column}"] = nulls
    checked = sum(len(columns) for columns in REQUIRED_COLUMNS.values())
    return QualityCheck(
        name="required_columns_not_null",
        category=CATEGORY_COMPLETENESS,
        severity=SEVERITY_ERROR,
        passed=not offenders,
        message=(
            f"all {checked} required columns are fully populated"
            if not offenders
            else f"NULLs found in required columns: {offenders}"
        ),
        details={"columns_checked": checked, "offenders": offenders},
    )


def run_completeness_checks(connection: sqlite3.Connection) -> list[QualityCheck]:
    """Run every completeness check.

    Args:
        connection: Open SQLite connection.

    Returns:
        The completeness check outcomes.
    """
    checks = [
        check_row_count(connection, table, expected)
        for table, expected in EXPECTED_ROW_COUNTS.items()
    ]
    checks.append(check_no_empty_tables(connection))
    checks.append(check_required_columns_not_null(connection))
    return checks


# ---------------------------------------------------------------------------
# Referential integrity
# ---------------------------------------------------------------------------


def check_foreign_key(
    connection: sqlite3.Connection,
    child_table: str,
    child_column: str,
    parent_table: str,
    parent_column: str,
    allow_null: bool = False,
) -> QualityCheck:
    """Verify every foreign key value resolves to its parent row.

    Args:
        connection: Open SQLite connection.
        child_table: Table holding the foreign key.
        child_column: Foreign key column.
        parent_table: Referenced table.
        parent_column: Referenced column.
        allow_null: When True, NULL keys are legitimate and are not counted.

    Returns:
        The check outcome, with the number of unresolved references.
    """
    orphans = count_orphans(
        connection, child_table, child_column, parent_table, parent_column, allow_null
    )
    reference = f"{child_table}.{child_column} -> {parent_table}.{parent_column}"
    null_note = " (NULL permitted)" if allow_null else ""
    return QualityCheck(
        name=f"foreign_key::{child_table}.{child_column}",
        category=CATEGORY_REFERENTIAL,
        severity=SEVERITY_ERROR,
        passed=orphans == 0,
        message=(
            f"{reference} fully resolves{null_note}"
            if orphans == 0
            else f"{reference} has {orphans:,} unresolved reference(s)"
        ),
        details={"reference": reference, "orphans": orphans, "null_allowed": allow_null},
    )


def check_every_order_has_lines(connection: sqlite3.Connection) -> QualityCheck:
    """Verify every order has at least one detail line.

    Args:
        connection: Open SQLite connection.

    Returns:
        The check outcome, with the count of line-less orders.
    """
    missing = int(
        scalar(
            connection,
            "SELECT COUNT(*) FROM fact_orders AS o "
            "WHERE NOT EXISTS ("
            "  SELECT 1 FROM fact_order_lines AS l WHERE l.order_id = o.order_id"
            ")",
        )
    )
    return QualityCheck(
        name="every_order_has_lines",
        category=CATEGORY_REFERENTIAL,
        severity=SEVERITY_ERROR,
        passed=missing == 0,
        message=(
            "every order has at least one detail line"
            if missing == 0
            else f"{missing:,} order(s) have no detail lines"
        ),
        details={"orders_without_lines": missing},
    )


def run_referential_checks(connection: sqlite3.Connection) -> list[QualityCheck]:
    """Run every referential integrity check.

    Args:
        connection: Open SQLite connection.

    Returns:
        The referential integrity check outcomes.
    """
    return [
        check_foreign_key(connection, "fact_orders", "store_id", "dim_store", "store_id"),
        check_foreign_key(
            connection,
            "fact_orders",
            "customer_id",
            "dim_customer",
            "customer_id",
            allow_null=True,
        ),
        check_foreign_key(
            connection,
            "fact_orders",
            "promo_id",
            "dim_promotion",
            "promo_id",
            allow_null=True,
        ),
        check_foreign_key(connection, "fact_orders", "order_date", "dim_calendar", "date"),
        check_foreign_key(
            connection, "fact_order_lines", "order_id", "fact_orders", "order_id"
        ),
        check_foreign_key(
            connection, "fact_order_lines", "sku_id", "dim_product", "sku_id"
        ),
        check_every_order_has_lines(connection),
    ]


# ---------------------------------------------------------------------------
# Temporal
# ---------------------------------------------------------------------------


def check_date_boundaries(connection: sqlite3.Connection) -> QualityCheck:
    """Verify the order fact spans exactly the configured dataset window.

    Args:
        connection: Open SQLite connection.

    Returns:
        The check outcome, comparing observed bounds to the configured ones.
    """
    observed_min = scalar(connection, "SELECT MIN(order_date) FROM fact_orders")
    observed_max = scalar(connection, "SELECT MAX(order_date) FROM fact_orders")
    expected_min = settings.DATA_START_DATE.isoformat()
    expected_max = settings.DATA_ASOF_DATE.isoformat()
    passed = observed_min == expected_min and observed_max == expected_max
    return QualityCheck(
        name="order_date_boundaries",
        category=CATEGORY_TEMPORAL,
        severity=SEVERITY_ERROR,
        passed=passed,
        message=(
            f"orders span {observed_min} to {observed_max}, matching the "
            f"configured dataset window"
            if passed
            else f"orders span {observed_min} to {observed_max} but "
            f"{expected_min} to {expected_max} was expected"
        ),
        details={
            "observed_min": observed_min,
            "observed_max": observed_max,
            "expected_min": expected_min,
            "expected_max": expected_max,
        },
    )


def check_month_keys(connection: sqlite3.Connection) -> QualityCheck:
    """Verify month keys are well formed and cover exactly twelve months.

    Args:
        connection: Open SQLite connection.

    Returns:
        The check outcome, with any malformed keys and the distinct count.
    """
    malformed: dict[str, int] = {}
    for table in ("dim_calendar", "fact_orders", *MART_TABLES):
        bad = int(
            scalar(
                connection,
                f"SELECT COUNT(*) FROM {table} "
                f"WHERE month_key NOT GLOB '{MONTH_KEY_GLOB}'",
            )
        )
        if bad:
            malformed[table] = bad

    distinct = int(
        scalar(connection, "SELECT COUNT(DISTINCT month_key) FROM fact_orders")
    )
    mismatched = int(
        scalar(
            connection,
            "SELECT COUNT(*) FROM fact_orders "
            "WHERE month_key != substr(order_date, 1, 7)",
        )
    )
    passed = (
        not malformed and distinct == EXPECTED_MONTH_COUNT and mismatched == 0
    )
    return QualityCheck(
        name="month_key_integrity",
        category=CATEGORY_TEMPORAL,
        severity=SEVERITY_ERROR,
        passed=passed,
        message=(
            f"{distinct} distinct month keys, all formatted YYYY-MM and "
            f"consistent with order_date"
            if passed
            else f"month key problems: {distinct} distinct (expected "
            f"{EXPECTED_MONTH_COUNT}), malformed={malformed}, "
            f"inconsistent_with_order_date={mismatched:,}"
        ),
        details={
            "distinct_month_keys": distinct,
            "expected_month_keys": EXPECTED_MONTH_COUNT,
            "malformed_by_table": malformed,
            "inconsistent_with_order_date": mismatched,
        },
    )


def check_no_future_orders(connection: sqlite3.Connection) -> QualityCheck:
    """Verify no order is dated after the fixed as-of date.

    Args:
        connection: Open SQLite connection.

    Returns:
        The check outcome, with the count of future-dated orders.
    """
    asof = settings.DATA_ASOF_DATE.isoformat()
    future = int(
        scalar(
            connection,
            "SELECT COUNT(*) FROM fact_orders WHERE order_date > ?",
            (asof,),
        )
    )
    return QualityCheck(
        name="no_future_orders",
        category=CATEGORY_TEMPORAL,
        severity=SEVERITY_ERROR,
        passed=future == 0,
        message=(
            f"no orders dated after the as-of date ({asof})"
            if future == 0
            else f"{future:,} order(s) dated after the as-of date ({asof})"
        ),
        details={"asof_date": asof, "future_orders": future},
    )


def run_temporal_checks(connection: sqlite3.Connection) -> list[QualityCheck]:
    """Run every temporal check.

    Args:
        connection: Open SQLite connection.

    Returns:
        The temporal check outcomes.
    """
    return [
        check_date_boundaries(connection),
        check_month_keys(connection),
        check_no_future_orders(connection),
    ]


# ---------------------------------------------------------------------------
# Financial reconciliation
# ---------------------------------------------------------------------------


def check_mart_reconciliation(
    connection: sqlite3.Connection, mart: str
) -> QualityCheck:
    """Verify a mart's revenue totals back to the order fact.

    Args:
        connection: Open SQLite connection.
        mart: Mart table to reconcile.

    Returns:
        The check outcome, reporting the actual delta in INR.
    """
    fact_total = float(
        scalar(connection, "SELECT SUM(net_before_tax) FROM fact_orders")
    )
    mart_total = float(scalar(connection, f"SELECT SUM(revenue_net) FROM {mart}"))
    delta = mart_total - fact_total
    passed = abs(delta) < REVENUE_TOLERANCE_INR
    return QualityCheck(
        name=f"mart_reconciliation::{mart}",
        category=CATEGORY_FINANCIAL,
        severity=SEVERITY_ERROR,
        passed=passed,
        message=(
            f"{mart} revenue_net reconciles to fact_orders "
            f"(delta {delta:+.4f} INR, tolerance {REVENUE_TOLERANCE_INR} INR)"
        ),
        details={
            "mart": mart,
            "mart_total": round(mart_total, 4),
            "fact_total": round(fact_total, 4),
            "delta_inr": round(delta, 4),
            "tolerance_inr": REVENUE_TOLERANCE_INR,
        },
    )


def check_tax_reconciliation(connection: sqlite3.Connection) -> QualityCheck:
    """Verify ``net_before_tax * (1 + TAX_RATE)`` equals ``net_revenue``.

    Args:
        connection: Open SQLite connection.

    Returns:
        The check outcome, with the count of breaches and the worst deviation.
    """
    multiplier = 1 + settings.TAX_RATE
    breaches = int(
        scalar(
            connection,
            "SELECT COUNT(*) FROM fact_orders "
            "WHERE ABS(net_before_tax * ? - net_revenue) > ?",
            (multiplier, ROW_TOLERANCE_INR),
        )
    )
    worst = float(
        scalar(
            connection,
            "SELECT MAX(ABS(net_before_tax * ? - net_revenue)) FROM fact_orders",
            (multiplier,),
        )
    )
    return QualityCheck(
        name="tax_reconciliation",
        category=CATEGORY_FINANCIAL,
        severity=SEVERITY_ERROR,
        passed=breaches == 0,
        message=(
            f"net_before_tax x {multiplier} equals net_revenue on all "
            f"{EXPECTED_ROW_COUNTS['fact_orders']:,} orders "
            f"(worst deviation {worst:.4f} INR)"
            if breaches == 0
            else f"{breaches:,} order(s) breach the {settings.TAX_RATE:.0%} tax "
            f"identity (worst deviation {worst:.4f} INR)"
        ),
        details={
            "tax_rate": settings.TAX_RATE,
            "breaches": breaches,
            "worst_deviation_inr": round(worst, 4),
            "tolerance_inr": ROW_TOLERANCE_INR,
        },
    )


def check_gross_less_discount(connection: sqlite3.Connection) -> QualityCheck:
    """Verify ``gross_bill_value - discount_amount`` equals ``net_before_tax``.

    Args:
        connection: Open SQLite connection.

    Returns:
        The check outcome, with the count of breaches and the worst deviation.
    """
    breaches = int(
        scalar(
            connection,
            "SELECT COUNT(*) FROM fact_orders "
            "WHERE ABS(gross_bill_value - discount_amount - net_before_tax) > ?",
            (ROW_TOLERANCE_INR,),
        )
    )
    worst = float(
        scalar(
            connection,
            "SELECT MAX(ABS(gross_bill_value - discount_amount - net_before_tax)) "
            "FROM fact_orders",
        )
    )
    return QualityCheck(
        name="gross_less_discount_equals_net",
        category=CATEGORY_FINANCIAL,
        severity=SEVERITY_ERROR,
        passed=breaches == 0,
        message=(
            f"gross_bill_value - discount_amount equals net_before_tax on every "
            f"order (worst deviation {worst:.4f} INR)"
            if breaches == 0
            else f"{breaches:,} order(s) breach the gross-less-discount identity "
            f"(worst deviation {worst:.4f} INR)"
        ),
        details={
            "breaches": breaches,
            "worst_deviation_inr": round(worst, 4),
            "tolerance_inr": ROW_TOLERANCE_INR,
        },
    )


def check_line_to_header_reconciliation(
    connection: sqlite3.Connection,
) -> QualityCheck:
    """Quantify the known disagreement between line grain and order grain.

    A small number of orders carry a header ``net_before_tax`` that does not
    equal the sum of their line ``line_net_value``. This is a defect in the
    source workbook, not in the ETL: both grains are loaded exactly as supplied.

    It is reported as a warning rather than an error because the variance is
    tiny and its consequence is documented and worked around: the order grain is
    canonical for revenue, and the line grain is used only for product mix and
    margin. The check escalates to an error if the variance ever grows past
    ``MAX_LINE_VARIANCE_ORDER_PCT`` of orders or
    ``MAX_LINE_VARIANCE_REVENUE_PCT`` of revenue, at which point the two grains
    are too far apart for that workaround to hold.

    Args:
        connection: Open SQLite connection.

    Returns:
        The check outcome, with the affected order count, the total variance in
        INR and both percentages.
    """
    total_orders = int(scalar(connection, "SELECT COUNT(*) FROM fact_orders"))
    total_revenue = float(
        scalar(connection, "SELECT SUM(net_before_tax) FROM fact_orders")
    )

    affected_orders, variance_inr = connection.execute(
        """
        SELECT COUNT(*), COALESCE(SUM(variance), 0)
        FROM (
            SELECT ABS(SUM(l.line_net_value) - o.net_before_tax) AS variance
            FROM fact_orders AS o
            JOIN fact_order_lines AS l ON l.order_id = o.order_id
            GROUP BY o.order_id, o.net_before_tax
            HAVING ABS(SUM(l.line_net_value) - o.net_before_tax) > ?
        )
        """,
        (LINE_RECONCILIATION_TOLERANCE_INR,),
    ).fetchone()

    # Where the variance sits matters as much as its size: it is concentrated in
    # the most recent months, so its share of a recent window is several times
    # its share of the full year.
    by_month = {
        month: {"orders": int(count), "variance_inr": round(float(variance), 2)}
        for month, count, variance in connection.execute(
            """
            SELECT month_key, COUNT(*), SUM(variance)
            FROM (
                SELECT o.month_key AS month_key,
                       ABS(SUM(l.line_net_value) - o.net_before_tax) AS variance
                FROM fact_orders AS o
                JOIN fact_order_lines AS l ON l.order_id = o.order_id
                GROUP BY o.order_id, o.month_key, o.net_before_tax
                HAVING ABS(SUM(l.line_net_value) - o.net_before_tax) > ?
            )
            GROUP BY month_key
            ORDER BY month_key
            """,
            (LINE_RECONCILIATION_TOLERANCE_INR,),
        )
    }

    order_pct = percentage(affected_orders, total_orders)
    revenue_pct = percentage(variance_inr, total_revenue)

    within_tolerance = (
        order_pct <= MAX_LINE_VARIANCE_ORDER_PCT
        and revenue_pct <= MAX_LINE_VARIANCE_REVENUE_PCT
    )
    severity = SEVERITY_WARNING if within_tolerance else SEVERITY_ERROR
    passed = affected_orders == 0

    consequence = (
        "CONSEQUENCE: fact_orders is canonical for revenue and order counts; "
        "fact_order_lines is for product mix and margin only. Never total "
        "revenue from the line grain."
    )
    if passed:
        message = "every order's lines sum to its header net_before_tax"
    else:
        span = (
            f" All affected orders fall in {min(by_month)}..{max(by_month)}, "
            f"{len(by_month)} of {EXPECTED_MONTH_COUNT} months, so the variance "
            f"is a larger share of any window inside that span than of the full "
            f"year."
            if by_month
            else ""
        )
        message = (
            f"{affected_orders:,} of {total_orders:,} orders ({order_pct:.2f}%) "
            f"have lines that do not sum to the header net_before_tax; total "
            f"variance {variance_inr:,.2f} INR ({revenue_pct:.4f}% of full-year "
            f"revenue).{span} "
            f"{'Within' if within_tolerance else 'EXCEEDS'} the accepted "
            f"threshold of {MAX_LINE_VARIANCE_ORDER_PCT}% of orders / "
            f"{MAX_LINE_VARIANCE_REVENUE_PCT}% of revenue. {consequence}"
        )

    return QualityCheck(
        name="line_to_header_reconciliation",
        category=CATEGORY_FINANCIAL,
        severity=severity,
        passed=passed,
        message=message,
        details={
            "affected_orders": affected_orders,
            "total_orders": total_orders,
            "affected_order_pct": order_pct,
            "variance_inr": round(float(variance_inr), 2),
            "total_revenue_inr": round(total_revenue, 2),
            "variance_revenue_pct": revenue_pct,
            "by_month": by_month,
            "affected_months": sorted(by_month),
            "tolerance_inr": LINE_RECONCILIATION_TOLERANCE_INR,
            "max_order_pct": MAX_LINE_VARIANCE_ORDER_PCT,
            "max_revenue_pct": MAX_LINE_VARIANCE_REVENUE_PCT,
            "within_tolerance": within_tolerance,
            "canonical_grain": "fact_orders",
        },
    )


def check_no_negative_values(connection: sqlite3.Connection) -> QualityCheck:
    """Verify quantity, price and revenue columns are never negative.

    Args:
        connection: Open SQLite connection.

    Returns:
        The check outcome, with a per-column count for any offender.
    """
    offenders: dict[str, int] = {}
    for table, columns in NON_NEGATIVE_COLUMNS.items():
        for column in columns:
            negatives = int(
                scalar(connection, f"SELECT COUNT(*) FROM {table} WHERE {column} < 0")
            )
            if negatives:
                offenders[f"{table}.{column}"] = negatives
    checked = sum(len(columns) for columns in NON_NEGATIVE_COLUMNS.values())
    return QualityCheck(
        name="no_negative_values",
        category=CATEGORY_FINANCIAL,
        severity=SEVERITY_ERROR,
        passed=not offenders,
        message=(
            f"no negative values across {checked} quantity, price and revenue "
            f"columns"
            if not offenders
            else f"negative values found: {offenders}"
        ),
        details={"columns_checked": checked, "offenders": offenders},
    )


def run_financial_checks(connection: sqlite3.Connection) -> list[QualityCheck]:
    """Run every financial reconciliation check.

    Args:
        connection: Open SQLite connection.

    Returns:
        The financial check outcomes.
    """
    checks = [check_mart_reconciliation(connection, mart) for mart in MART_TABLES]
    checks.append(check_tax_reconciliation(connection))
    checks.append(check_gross_less_discount(connection))
    checks.append(check_line_to_header_reconciliation(connection))
    checks.append(check_no_negative_values(connection))
    return checks


# ---------------------------------------------------------------------------
# Domain validity
# ---------------------------------------------------------------------------


def check_allowed_values(
    connection: sqlite3.Connection,
    name: str,
    table: str,
    column: str,
    allowed: Sequence[str],
) -> QualityCheck:
    """Verify a column contains only values from a declared vocabulary.

    Args:
        connection: Open SQLite connection.
        name: Check name.
        table: Table to inspect.
        column: Column to inspect.
        allowed: The permitted values.

    Returns:
        The check outcome, listing any unexpected values.
    """
    observed = {
        row[0] for row in connection.execute(f"SELECT DISTINCT {column} FROM {table}")
    }
    unexpected = sorted(observed - set(allowed))
    return QualityCheck(
        name=name,
        category=CATEGORY_DOMAIN,
        severity=SEVERITY_ERROR,
        passed=not unexpected,
        message=(
            f"{table}.{column} holds only the {len(allowed)} declared values: "
            f"{', '.join(sorted(observed))}"
            if not unexpected
            else f"{table}.{column} holds undeclared value(s): "
            f"{', '.join(unexpected)}"
        ),
        details={
            "table": table,
            "column": column,
            "allowed": list(allowed),
            "observed": sorted(observed),
            "unexpected": unexpected,
        },
    )


def check_derived_flag(
    connection: sqlite3.Connection,
    name: str,
    flag_column: str,
    condition_sql: str,
    params: Sequence[Any],
    description: str,
) -> QualityCheck:
    """Verify a denormalized boolean flag agrees with the column it derives from.

    Args:
        connection: Open SQLite connection.
        name: Check name.
        flag_column: The integer flag column on ``fact_orders``.
        condition_sql: SQL boolean expression the flag should equal.
        params: Parameters bound into ``condition_sql``.
        description: Human-readable statement of the rule.

    Returns:
        The check outcome, with the count of disagreeing rows.
    """
    mismatches = 0
    for table in ("dim_calendar", "fact_orders"):
        mismatches += int(
            scalar(
                connection,
                f"SELECT COUNT(*) FROM {table} "
                f"WHERE {flag_column} != ({condition_sql})",
                params,
            )
        )
    return QualityCheck(
        name=name,
        category=CATEGORY_DOMAIN,
        severity=SEVERITY_ERROR,
        passed=mismatches == 0,
        message=(
            f"{description} on every row of dim_calendar and fact_orders"
            if mismatches == 0
            else f"{mismatches:,} row(s) where {description.lower()} does not hold"
        ),
        details={"flag": flag_column, "mismatches": mismatches, "rule": description},
    )


def run_domain_checks(connection: sqlite3.Connection) -> list[QualityCheck]:
    """Run every domain validity check.

    Args:
        connection: Open SQLite connection.

    Returns:
        The domain validity check outcomes.
    """
    festive_placeholders = ", ".join("?" for _ in settings.FESTIVE_PERIODS)
    return [
        check_allowed_values(
            connection, "channels_in_vocabulary", "fact_orders", "channel", settings.CHANNELS
        ),
        check_allowed_values(
            connection,
            "festive_periods_in_vocabulary",
            "fact_orders",
            "festive_period",
            [*settings.FESTIVE_PERIODS, NON_FESTIVE_PERIOD],
        ),
        check_allowed_values(
            connection, "day_types_in_vocabulary", "fact_orders", "day_type", DAY_TYPES
        ),
        check_derived_flag(
            connection,
            name="is_weekend_matches_day_type",
            flag_column="is_weekend",
            condition_sql="day_type = ?",
            params=(WEEKEND_DAY_TYPE,),
            description=f"is_weekend = 1 exactly when day_type = '{WEEKEND_DAY_TYPE}'",
        ),
        check_derived_flag(
            connection,
            name="is_festive_matches_festive_period",
            flag_column="is_festive",
            condition_sql=f"festive_period IN ({festive_placeholders})",
            params=tuple(settings.FESTIVE_PERIODS),
            description=(
                f"is_festive = 1 exactly when festive_period != "
                f"'{NON_FESTIVE_PERIOD}'"
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Distribution (informational)
# ---------------------------------------------------------------------------


def describe_anonymous_orders(connection: sqlite3.Connection) -> QualityCheck:
    """Report the share of orders with no identified customer.

    Args:
        connection: Open SQLite connection.

    Returns:
        An informational check describing the anonymous order rate.
    """
    total = int(scalar(connection, "SELECT COUNT(*) FROM fact_orders"))
    anonymous = int(
        scalar(connection, "SELECT COUNT(*) FROM fact_orders WHERE customer_id IS NULL")
    )
    share = percentage(anonymous, total)
    return QualityCheck(
        name="anonymous_order_share",
        category=CATEGORY_DISTRIBUTION,
        severity=SEVERITY_INFO,
        passed=True,
        message=(
            f"{anonymous:,} of {total:,} orders ({share:.2f}%) are anonymous "
            f"walk-ins with a NULL customer_id - always LEFT JOIN dim_customer"
        ),
        details={
            "anonymous_orders": anonymous,
            "total_orders": total,
            "share_pct": share,
            "expected": EXPECTED_ANONYMOUS_ORDERS,
        },
    )


def describe_promo_orders(connection: sqlite3.Connection) -> QualityCheck:
    """Report the share of orders carrying a promotion.

    Args:
        connection: Open SQLite connection.

    Returns:
        An informational check describing the promotion attachment rate.
    """
    total = int(scalar(connection, "SELECT COUNT(*) FROM fact_orders"))
    promoted = int(
        scalar(connection, "SELECT COUNT(*) FROM fact_orders WHERE promo_id IS NOT NULL")
    )
    share = percentage(promoted, total)
    return QualityCheck(
        name="promotion_attachment_share",
        category=CATEGORY_DISTRIBUTION,
        severity=SEVERITY_INFO,
        passed=True,
        message=(
            f"{promoted:,} of {total:,} orders ({share:.2f}%) carry a promotion - "
            f"NULL promo_id is the normal case, always LEFT JOIN dim_promotion"
        ),
        details={
            "promo_orders": promoted,
            "total_orders": total,
            "share_pct": share,
            "expected": EXPECTED_PROMO_ORDERS,
        },
    )


def describe_cardinalities(connection: sqlite3.Connection) -> QualityCheck:
    """Report the distinct counts of the main dimensions.

    Args:
        connection: Open SQLite connection.

    Returns:
        An informational check listing dimension cardinalities.
    """
    cardinalities = {
        "stores": int(scalar(connection, "SELECT COUNT(DISTINCT store_id) FROM dim_store")),
        "cities": int(scalar(connection, "SELECT COUNT(DISTINCT city) FROM dim_store")),
        "regions": int(scalar(connection, "SELECT COUNT(DISTINCT region) FROM dim_store")),
        "store_formats": int(
            scalar(connection, "SELECT COUNT(DISTINCT store_format) FROM dim_store")
        ),
        "skus": int(scalar(connection, "SELECT COUNT(DISTINCT sku_id) FROM dim_product")),
        "categories": int(
            scalar(connection, "SELECT COUNT(DISTINCT category) FROM dim_product")
        ),
        "customers": int(
            scalar(connection, "SELECT COUNT(DISTINCT customer_id) FROM dim_customer")
        ),
        "customer_segments": int(
            scalar(connection, "SELECT COUNT(DISTINCT customer_segment) FROM dim_customer")
        ),
        "channels": int(scalar(connection, "SELECT COUNT(DISTINCT channel) FROM fact_orders")),
        "promotions": int(
            scalar(connection, "SELECT COUNT(DISTINCT promo_id) FROM dim_promotion")
        ),
    }
    summary = ", ".join(f"{name}={count}" for name, count in cardinalities.items())
    return QualityCheck(
        name="dimension_cardinalities",
        category=CATEGORY_DISTRIBUTION,
        severity=SEVERITY_INFO,
        passed=True,
        message=summary,
        details=cardinalities,
    )


def describe_monthly_revenue(connection: sqlite3.Connection) -> QualityCheck:
    """Report revenue by month to surface the dataset's seasonal shape.

    Args:
        connection: Open SQLite connection.

    Returns:
        An informational check with per-month revenue and order counts.
    """
    rows = connection.execute(
        "SELECT month_key, COUNT(*), SUM(net_before_tax) FROM fact_orders "
        "GROUP BY month_key ORDER BY month_key"
    ).fetchall()
    monthly = {
        month: {"orders": int(orders), "revenue_net": round(float(revenue), 2)}
        for month, orders, revenue in rows
    }
    peak = max(monthly.items(), key=lambda item: item[1]["revenue_net"])
    trough = min(monthly.items(), key=lambda item: item[1]["revenue_net"])
    return QualityCheck(
        name="monthly_revenue_shape",
        category=CATEGORY_DISTRIBUTION,
        severity=SEVERITY_INFO,
        passed=True,
        message=(
            f"{len(monthly)} months; peak {peak[0]} at "
            f"{peak[1]['revenue_net']:,.0f} INR, trough {trough[0]} at "
            f"{trough[1]['revenue_net']:,.0f} INR "
            f"(spread {peak[1]['revenue_net'] / trough[1]['revenue_net']:.2f}x)"
        ),
        details={"monthly": monthly, "peak_month": peak[0], "trough_month": trough[0]},
    )


def describe_aov(connection: sqlite3.Connection) -> QualityCheck:
    """Report the overall average order value and units per order.

    Args:
        connection: Open SQLite connection.

    Returns:
        An informational check with headline averages.
    """
    orders, revenue, units = connection.execute(
        "SELECT COUNT(*), SUM(net_before_tax), SUM(total_qty) FROM fact_orders"
    ).fetchone()
    aov = float(revenue) / int(orders)
    upo = float(units) / int(orders)
    return QualityCheck(
        name="headline_averages",
        category=CATEGORY_DISTRIBUTION,
        severity=SEVERITY_INFO,
        passed=True,
        message=(
            f"AOV {aov:,.2f} INR ({settings.REVENUE_METRIC}) across "
            f"{int(orders):,} orders; {upo:.2f} units per order; total revenue "
            f"{float(revenue):,.2f} INR"
        ),
        details={
            "orders": int(orders),
            "revenue_net_inr": round(float(revenue), 2),
            "units": int(units),
            "aov_inr": round(aov, 2),
            "units_per_order": round(upo, 4),
            "revenue_metric": settings.REVENUE_METRIC,
        },
    )


def run_distribution_checks(connection: sqlite3.Connection) -> list[QualityCheck]:
    """Run every informational distribution check.

    Args:
        connection: Open SQLite connection.

    Returns:
        The distribution observations.
    """
    return [
        describe_anonymous_orders(connection),
        describe_promo_orders(connection),
        describe_cardinalities(connection),
        describe_monthly_revenue(connection),
        describe_aov(connection),
    ]


# ---------------------------------------------------------------------------
# Orchestration and reporting
# ---------------------------------------------------------------------------


def run_quality_checks(db_path: Path | None = None) -> QualityReport:
    """Run the full quality gate against a database.

    Args:
        db_path: Database to validate. Defaults to ``settings.DB_PATH``.

    Returns:
        The completed report.

    Raises:
        FileNotFoundError: If the database file does not exist.
    """
    path = db_path or settings.DB_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"database not found at {path}; run `python -m app.etl.build_db` first"
        )

    report = QualityReport(db_path=str(path))
    connection = sqlite3.connect(path)
    try:
        report.checks.extend(run_completeness_checks(connection))
        report.checks.extend(run_referential_checks(connection))
        report.checks.extend(run_temporal_checks(connection))
        report.checks.extend(run_financial_checks(connection))
        report.checks.extend(run_domain_checks(connection))
        report.checks.extend(run_distribution_checks(connection))
    finally:
        connection.close()
    return report


def marker(check: QualityCheck) -> str:
    """Render the status marker for a check.

    Args:
        check: The check to render.

    Returns:
        ``INFO``, ``PASS``, ``WARN`` or ``FAIL``.
    """
    if check.severity == SEVERITY_INFO:
        return "INFO"
    if check.passed:
        return "PASS"
    return "WARN" if check.severity == SEVERITY_WARNING else "FAIL"


def print_report(report: QualityReport, width: int = 88) -> None:
    """Print the report as a sectioned console summary.

    Args:
        report: The report to render.
        width: Line width for the section rules.
    """
    print()
    print("=" * width)
    print("DATA QUALITY REPORT".center(width))
    print("=" * width)
    print(f"database : {report.db_path}")
    print(f"checks   : {len(report.checks)}")

    for category in CATEGORY_ORDER:
        checks = report.by_category(category)
        if not checks:
            continue
        print()
        print(f"-- {category.replace('_', ' ').upper()} " + "-" * (width - len(category) - 4))
        for check in checks:
            print(f"  [{marker(check)}] {check.name}")
            print(f"         {check.message}")

    print()
    print("=" * width)
    verdict = "PASSED" if report.passed else "FAILED"
    print(
        f"VERDICT: {verdict}  |  {report.error_count} error(s), "
        f"{report.warning_count} warning(s), {report.info_count} observation(s), "
        f"{len(report.checks)} checks total"
    )
    if report.warning_count:
        print("Warnings are known, quantified source-data defects; see messages above.")
    if not report.passed:
        print()
        print("Failing checks:")
        for check in report.failures():
            if check.severity == SEVERITY_ERROR:
                print(f"  - {check.name}: {check.message}")
    print("=" * width)
    print()


def main() -> None:
    """Command-line entrypoint: run the gate and exit non-zero on failure."""
    report = run_quality_checks()
    print_report(report)
    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
