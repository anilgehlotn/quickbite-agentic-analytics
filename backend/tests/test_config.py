"""Tests for the application configuration.

These tests guard the invariants that the rest of the system depends on: the
fixed time anchor, the business constants, the absolute filesystem paths and
the LLM provider fallback order. A regression in the time anchor in particular
would silently produce empty results for every relative-date question, so it is
asserted explicitly rather than derived.
"""

from datetime import date
from pathlib import Path

import pytest

from app.config import Settings, settings

# Provider environment variable names, in the same order as LLM_PROVIDER_ORDER.
PROVIDER_ENV_VARS: tuple[str, ...] = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GROK_API_KEY",
)


@pytest.fixture
def clean_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Build a Settings instance isolated from the ambient environment.

    Removes every provider key from the environment and disables ``.env``
    loading so provider tests are deterministic on any machine.

    Args:
        monkeypatch: pytest fixture used to clear the provider variables.

    Returns:
        A Settings instance with no provider credentials configured.
    """
    for env_var in PROVIDER_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)
    return Settings(_env_file=None)


class TestTimeAnchoring:
    """The dataset's fixed notion of 'today' and the windows derived from it."""

    def test_asof_date_is_end_of_dataset(self) -> None:
        """The as-of anchor is exactly 2026-07-31, never the system clock."""
        assert settings.DATA_ASOF_DATE == date(2026, 7, 31)

    def test_start_date_is_beginning_of_dataset(self) -> None:
        """The dataset starts on 2025-08-01."""
        assert settings.DATA_START_DATE == date(2025, 8, 1)

    def test_asof_date_is_not_system_today(self) -> None:
        """The anchor is a constant, so it must not track the real clock."""
        assert settings.DATA_ASOF_DATE != date.today()

    def test_data_range_is_ordered(self) -> None:
        """The dataset window runs forwards in time."""
        assert settings.DATA_START_DATE < settings.DATA_ASOF_DATE

    def test_last_3m_window_bounds(self) -> None:
        """'Last 3 months' spans 2026-05-01 through 2026-07-31 inclusive."""
        assert settings.LAST_3M_START == date(2026, 5, 1)
        assert settings.LAST_3M_END == date(2026, 7, 31)

    def test_last_3m_window_ends_on_asof_date(self) -> None:
        """The last-3-months window is anchored to the as-of date."""
        assert settings.LAST_3M_END == settings.DATA_ASOF_DATE

    def test_last_3m_window_falls_within_data_range(self) -> None:
        """The window must sit inside the data, or queries return nothing."""
        assert settings.DATA_START_DATE <= settings.LAST_3M_START
        assert settings.LAST_3M_START <= settings.LAST_3M_END
        assert settings.LAST_3M_END <= settings.DATA_ASOF_DATE


class TestPaths:
    """Filesystem paths must be absolute and point at real locations."""

    def test_project_root_resolves_to_repository_root(self) -> None:
        """PROJECT_ROOT is the directory containing backend/ and data/."""
        assert (settings.PROJECT_ROOT / "backend" / "app" / "config.py").is_file()
        assert (settings.PROJECT_ROOT / "data").is_dir()

    def test_all_paths_are_absolute(self) -> None:
        """Paths work regardless of the process's working directory."""
        for path in (
            settings.PROJECT_ROOT,
            settings.DATA_DIR,
            settings.EXCEL_PATH,
            settings.DB_PATH,
        ):
            assert isinstance(path, Path)
            assert path.is_absolute()

    def test_data_dir_is_under_project_root(self) -> None:
        """The data directory sits directly beneath the project root."""
        assert settings.DATA_DIR == settings.PROJECT_ROOT / "data"

    def test_excel_path_points_at_existing_file(self) -> None:
        """The source workbook ships with the repository."""
        assert settings.EXCEL_PATH.is_file()
        assert settings.EXCEL_PATH.name == "QSR_Agentic_Insights_Dataset.xlsx"

    def test_db_path_is_in_data_dir(self) -> None:
        """The SQLite database lives alongside the source workbook."""
        assert settings.DB_PATH.parent == settings.DATA_DIR
        assert settings.DB_PATH.name == "quickbite.db"


class TestBusinessConstants:
    """Canonical revenue definition, tax rate and domain vocabularies."""

    def test_revenue_metric_is_net_before_tax(self) -> None:
        """Revenue is reported net of tax."""
        assert settings.REVENUE_METRIC == "net_before_tax"

    def test_tax_rate(self) -> None:
        """The dataset applies a 5% tax."""
        assert settings.TAX_RATE == 0.05

    def test_channels_are_exactly_the_four_known_channels(self) -> None:
        """Channel vocabulary matches the dataset exactly."""
        assert settings.CHANNELS == ["Dine-in", "Takeaway", "Swiggy", "Zomato"]

    def test_festive_periods_are_exactly_the_three_known_periods(self) -> None:
        """Festive period vocabulary matches the dataset exactly."""
        assert settings.FESTIVE_PERIODS == ["Pujo", "Diwali", "New Year"]


class TestAvailableProviders:
    """LLM provider detection and preference ordering."""

    def test_provider_order(self) -> None:
        """Providers are preferred in a fixed order."""
        assert settings.LLM_PROVIDER_ORDER == [
            "anthropic",
            "openai",
            "gemini",
            "grok",
        ]

    def test_returns_empty_when_no_keys_configured(
        self, clean_settings: Settings
    ) -> None:
        """No credentials means no available providers."""
        assert clean_settings.available_providers() == []

    def test_returns_only_configured_providers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only providers with a key are reported."""
        for env_var in PROVIDER_ENV_VARS:
            monkeypatch.delenv(env_var, raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")

        assert Settings(_env_file=None).available_providers() == ["openai"]

    def test_preserves_provider_order_not_env_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Results follow LLM_PROVIDER_ORDER, not the order keys were set."""
        for env_var in PROVIDER_ENV_VARS:
            monkeypatch.delenv(env_var, raising=False)
        monkeypatch.setenv("GROK_API_KEY", "xai-test-grok")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setenv("GEMINI_API_KEY", "test-gemini")

        assert Settings(_env_file=None).available_providers() == [
            "anthropic",
            "gemini",
            "grok",
        ]

    def test_blank_keys_are_not_counted_as_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty or whitespace-only key is treated as absent."""
        for env_var in PROVIDER_ENV_VARS:
            monkeypatch.delenv(env_var, raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")
        monkeypatch.setenv("OPENAI_API_KEY", "   ")

        assert Settings(_env_file=None).available_providers() == []

    def test_all_providers_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With every key set, all providers are reported in order."""
        for env_var in PROVIDER_ENV_VARS:
            monkeypatch.setenv(env_var, f"test-{env_var.lower()}")

        instance = Settings(_env_file=None)
        assert instance.available_providers() == instance.LLM_PROVIDER_ORDER


class TestAppSettings:
    """Application metadata and query safety rails."""

    def test_app_metadata(self) -> None:
        """Name and version identify the service."""
        assert settings.APP_NAME == "QuickBite Agentic Analytics"
        assert settings.APP_VERSION == "0.1.0"

    def test_query_limits_are_positive(self) -> None:
        """Row and timeout limits bound generated SQL."""
        assert settings.MAX_QUERY_ROWS > 0
        assert settings.QUERY_TIMEOUT_SECONDS > 0

    def test_singleton_is_a_settings_instance(self) -> None:
        """The module-level singleton is usable directly."""
        assert isinstance(settings, Settings)
