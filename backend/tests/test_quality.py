"""Tests for the data quality gate.

The gate is the contract that lets everything downstream trust the database, so
these tests check both directions: that it passes on the real, correct database,
and that it actually fails on a deliberately corrupted copy. A gate that only
ever returns "passed" proves nothing.
"""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import pytest

from app.config import settings
from app.etl.quality_checks import (
    CATEGORY_DISTRIBUTION,
    CATEGORY_ORDER,
    SEVERITY_ERROR,
    SEVERITY_INFO,
    SEVERITY_WARNING,
    QualityCheck,
    QualityReport,
    run_quality_checks,
)

# The line-to-header variance is the one known, accepted defect in the source
# data. Any other warning is new and must be looked at.
EXPECTED_WARNING_NAMES: tuple[str, ...] = ("line_to_header_reconciliation",)


@pytest.fixture(scope="module")
def report() -> QualityReport:
    """Run the gate once against the real database.

    Returns:
        The report produced for ``settings.DB_PATH``.
    """
    return run_quality_checks(settings.DB_PATH)


class TestGatePasses:
    """The committed database must clear the gate."""

    def test_report_passes(self, report: QualityReport) -> None:
        """No error-severity check fails."""
        failures = [c.name for c in report.failures() if c.severity == SEVERITY_ERROR]
        assert report.passed, f"error checks failed: {failures}"

    def test_no_errors(self, report: QualityReport) -> None:
        """The error count is zero."""
        assert report.error_count == 0

    def test_report_is_not_trivially_empty(self, report: QualityReport) -> None:
        """The gate actually ran a meaningful number of checks."""
        assert len(report.checks) >= len(CATEGORY_ORDER)

    def test_every_category_has_at_least_one_check(
        self, report: QualityReport
    ) -> None:
        """No category was silently skipped."""
        for category in CATEGORY_ORDER:
            assert report.by_category(category), category

    def test_every_check_belongs_to_a_known_category(
        self, report: QualityReport
    ) -> None:
        """No check carries an unrecognised category."""
        for check in report.checks:
            assert check.category in CATEGORY_ORDER, check.name

    def test_every_check_has_a_message(self, report: QualityReport) -> None:
        """Each check explains its outcome."""
        for check in report.checks:
            assert check.message.strip(), check.name

    def test_check_names_are_unique(self, report: QualityReport) -> None:
        """Names identify checks unambiguously in the report and the API."""
        names = [check.name for check in report.checks]
        assert len(names) == len(set(names))


class TestWarnings:
    """Known source-data defects are reported as warnings, never as errors."""

    def test_expected_warning_count(self, report: QualityReport) -> None:
        """Exactly the known defects warn; nothing new has appeared."""
        assert report.warning_count == len(EXPECTED_WARNING_NAMES)

    def test_line_reconciliation_warning_is_present(
        self, report: QualityReport
    ) -> None:
        """The line-to-header variance is surfaced, not hidden."""
        names = {c.name for c in report.failures()}
        assert "line_to_header_reconciliation" in names

    def test_line_reconciliation_is_a_warning_not_an_error(
        self, report: QualityReport
    ) -> None:
        """The known variance must not fail the build."""
        check = next(
            c for c in report.checks if c.name == "line_to_header_reconciliation"
        )
        assert check.severity == SEVERITY_WARNING
        assert not check.passed

    def test_line_reconciliation_quantifies_the_variance(
        self, report: QualityReport
    ) -> None:
        """The warning carries the numbers needed to judge it."""
        check = next(
            c for c in report.checks if c.name == "line_to_header_reconciliation"
        )
        assert check.details is not None
        assert check.details["affected_orders"] > 0
        assert check.details["variance_inr"] > 0
        assert check.details["within_tolerance"] is True
        assert check.details["canonical_grain"] == "fact_orders"

    def test_line_reconciliation_states_the_consequence(
        self, report: QualityReport
    ) -> None:
        """The message tells a reader which grain to trust."""
        check = next(
            c for c in report.checks if c.name == "line_to_header_reconciliation"
        )
        assert "CONSEQUENCE" in check.message
        assert "fact_orders" in check.message


class TestDistributionChecks:
    """Distribution checks inform; they never fail the gate."""

    def test_distribution_checks_are_informational(
        self, report: QualityReport
    ) -> None:
        """Every distribution check is info severity and passes."""
        checks = report.by_category(CATEGORY_DISTRIBUTION)
        assert checks
        for check in checks:
            assert check.severity == SEVERITY_INFO, check.name
            assert check.passed, check.name

    def test_anonymous_share_is_reported(self, report: QualityReport) -> None:
        """The anonymous order rate is measured and matches the dataset."""
        check = next(
            c for c in report.checks if c.name == "anonymous_order_share"
        )
        assert check.details is not None
        assert check.details["anonymous_orders"] == check.details["expected"]

    def test_promo_share_is_reported(self, report: QualityReport) -> None:
        """The promotion attachment rate is measured and matches the dataset."""
        check = next(
            c for c in report.checks if c.name == "promotion_attachment_share"
        )
        assert check.details is not None
        assert check.details["promo_orders"] == check.details["expected"]


class TestReportSerialization:
    """The report must survive the trip to an API response."""

    def test_to_dict_shape(self, report: QualityReport) -> None:
        """to_dict exposes the verdict, the counts and every check."""
        payload = report.to_dict()
        assert payload["passed"] is True
        assert payload["total_checks"] == len(report.checks)
        assert payload["error_count"] == report.error_count
        assert payload["warning_count"] == report.warning_count
        assert len(payload["checks"]) == len(report.checks)
        assert set(payload["categories"]) == set(CATEGORY_ORDER)

    def test_to_dict_is_json_serializable(self, report: QualityReport) -> None:
        """The payload contains only JSON-native types."""
        import json

        assert json.loads(json.dumps(report.to_dict()))["passed"] is True


class TestGateCatchesCorruption:
    """The gate must fail on a broken database, not just pass on a good one."""

    @pytest.fixture
    def corrupted_db(self, tmp_path: Path) -> Path:
        """Copy the database and delete a store that orders reference.

        Args:
            tmp_path: pytest-provided temporary directory.

        Returns:
            Path to the corrupted copy.
        """
        path = tmp_path / "corrupted.db"
        shutil.copy(settings.DB_PATH, path)

        connection = sqlite3.connect(path)
        try:
            victim = connection.execute(
                "SELECT store_id FROM fact_orders LIMIT 1"
            ).fetchone()[0]
            connection.execute("PRAGMA foreign_keys = OFF")
            connection.execute("DELETE FROM dim_store WHERE store_id = ?", (victim,))
            connection.commit()
        finally:
            connection.close()
        return path

    def test_corrupted_database_fails_the_gate(self, corrupted_db: Path) -> None:
        """Deleting a referenced store makes the report fail."""
        result = run_quality_checks(corrupted_db)
        assert result.passed is False
        assert result.error_count > 0

    def test_referential_check_identifies_the_break(
        self, corrupted_db: Path
    ) -> None:
        """The store foreign key check is the one that catches it."""
        result = run_quality_checks(corrupted_db)
        check = next(
            c for c in result.checks if c.name == "foreign_key::fact_orders.store_id"
        )
        assert not check.passed
        assert check.severity == SEVERITY_ERROR
        assert check.details is not None
        assert check.details["orphans"] > 0

    def test_row_count_check_also_catches_the_deletion(
        self, corrupted_db: Path
    ) -> None:
        """The completeness check notices dim_store lost a row."""
        result = run_quality_checks(corrupted_db)
        check = next(c for c in result.checks if c.name == "row_count::dim_store")
        assert not check.passed
        assert check.details is not None
        assert check.details["actual"] == check.details["expected"] - 1

    def test_to_dict_reports_failure(self, corrupted_db: Path) -> None:
        """A failing report serializes as failing."""
        assert run_quality_checks(corrupted_db).to_dict()["passed"] is False


class TestReportSemantics:
    """The report's own accounting logic."""

    def test_warnings_do_not_fail_the_report(self) -> None:
        """A failing warning leaves the verdict passing."""
        result = QualityReport(
            checks=[
                QualityCheck(
                    name="w",
                    category=CATEGORY_DISTRIBUTION,
                    severity=SEVERITY_WARNING,
                    passed=False,
                    message="known defect",
                )
            ]
        )
        assert result.passed is True
        assert result.warning_count == 1
        assert result.error_count == 0

    def test_errors_fail_the_report(self) -> None:
        """A failing error fails the verdict."""
        result = QualityReport(
            checks=[
                QualityCheck(
                    name="e",
                    category=CATEGORY_DISTRIBUTION,
                    severity=SEVERITY_ERROR,
                    passed=False,
                    message="broken",
                )
            ]
        )
        assert result.passed is False
        assert result.error_count == 1

    def test_missing_database_raises(self, tmp_path: Path) -> None:
        """Pointing the gate at a nonexistent file is an explicit error."""
        with pytest.raises(FileNotFoundError):
            run_quality_checks(tmp_path / "does_not_exist.db")
