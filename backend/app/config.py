"""Central configuration for the QuickBite Agentic Analytics backend.

This module is the single source of truth for every constant in the system:
time anchoring, business rules, filesystem paths, LLM provider credentials and
application settings. No other module should hardcode dates, rates or paths.

The most important thing defined here is the time anchor. The QSR dataset is a
fixed historical extract covering 01-Aug-2025 through 31-Jul-2026, so the
system's notion of "today" is the constant ``DATA_ASOF_DATE`` (2026-07-31) and
never ``date.today()``. Every relative time expression ("last 3 months",
"year to date") must resolve against that anchor; resolving against the system
clock would place every window outside the data and return empty results.
"""

from datetime import date
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Filesystem anchors resolved from this file's location so that every path is
# absolute and correct regardless of the process's current working directory.
# config.py lives at <project_root>/backend/app/config.py, so the project root
# is three parents up.
_APP_DIR: Path = Path(__file__).resolve().parent
_BACKEND_DIR: Path = _APP_DIR.parent
_PROJECT_ROOT: Path = _BACKEND_DIR.parent
_DATA_DIR: Path = _PROJECT_ROOT / "data"

# Filename of the source workbook shipped with the repository.
_EXCEL_FILENAME: str = "QSR_Agentic_Insights_Dataset.xlsx"

# Filename of the SQLite database built by the ETL step. It is committed to the
# repository so the application needs no external infrastructure to run.
_DB_FILENAME: str = "quickbite.db"


class Settings(BaseSettings):
    """Application settings for QuickBite Agentic Analytics.

    Fields fall into two groups. Domain constants (time anchoring, business
    rules, paths) carry class-level defaults and are intentionally fixed: they
    describe the shipped dataset, not the deployment environment. Credentials
    and application settings are read from the environment or a ``.env`` file.
    """

    # ------------------------------------------------------------------
    # Time anchoring
    # ------------------------------------------------------------------
    # Constants, not deployment configuration. The dataset is a fixed extract,
    # so these describe the data itself and must not drift with the system
    # clock or be tuned per environment.

    # The system's "today". All relative time expressions resolve against this.
    DATA_ASOF_DATE: date = date(2026, 7, 31)

    # First day covered by the dataset.
    DATA_START_DATE: date = date(2025, 8, 1)

    # Canonical "last 3 months" window: the three full calendar months ending
    # on the as-of date (May, June, July 2026).
    LAST_3M_START: date = date(2026, 5, 1)
    LAST_3M_END: date = date(2026, 7, 31)

    # ------------------------------------------------------------------
    # Business constants
    # ------------------------------------------------------------------

    # Canonical revenue measure: net of the 5% tax component.
    REVENUE_METRIC: str = "net_before_tax"

    # Tax applied on top of net revenue in the source data.
    TAX_RATE: float = 0.05

    # Named festive windows recognised by the semantic layer.
    FESTIVE_PERIODS: list[str] = ["Pujo", "Diwali", "New Year"]

    # Sales channels present in the dataset.
    CHANNELS: list[str] = ["Dine-in", "Takeaway", "Swiggy", "Zomato"]

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    # Absolute, derived from this file's location at import time.

    PROJECT_ROOT: Path = _PROJECT_ROOT
    DATA_DIR: Path = _DATA_DIR
    EXCEL_PATH: Path = _DATA_DIR / _EXCEL_FILENAME
    DB_PATH: Path = _DATA_DIR / _DB_FILENAME

    # ------------------------------------------------------------------
    # LLM provider credentials
    # ------------------------------------------------------------------
    # All optional. The system falls back through LLM_PROVIDER_ORDER using
    # whichever keys are actually present.

    ANTHROPIC_API_KEY: str | None = None
    OPENAI_API_KEY: str | None = None
    GEMINI_API_KEY: str | None = None
    GROK_API_KEY: str | None = None

    # Preference order used when selecting a provider.
    LLM_PROVIDER_ORDER: list[str] = ["anthropic", "openai", "gemini", "grok"]

    # ------------------------------------------------------------------
    # Application settings
    # ------------------------------------------------------------------

    APP_NAME: str = "QuickBite Agentic Analytics"
    APP_VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"
    CORS_ORIGINS: list[str] = ["http://localhost:3000"]

    # Safety rails for generated SQL.
    MAX_QUERY_ROWS: int = 1000
    QUERY_TIMEOUT_SECONDS: int = 10

    model_config = SettingsConfigDict(
        # Absolute path so the .env file is found no matter where the process
        # was started from.
        env_file=_BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        # Unknown environment variables are ignored rather than rejected, so an
        # unrelated variable in the shell cannot break startup.
        extra="ignore",
    )

    def available_providers(self) -> list[str]:
        """Return the configured LLM providers in preference order.

        A provider counts as configured when its API key is present and not
        blank. Providers are returned in ``LLM_PROVIDER_ORDER`` sequence so
        callers can simply take the first entry as the preferred provider.

        Returns:
            Provider names that have a usable key, ordered by preference.
            Empty when no keys are configured.
        """
        keys_by_provider: dict[str, str | None] = {
            "anthropic": self.ANTHROPIC_API_KEY,
            "openai": self.OPENAI_API_KEY,
            "gemini": self.GEMINI_API_KEY,
            "grok": self.GROK_API_KEY,
        }
        return [
            provider
            for provider in self.LLM_PROVIDER_ORDER
            if (keys_by_provider.get(provider) or "").strip()
        ]


# Module-level singleton imported throughout the application.
settings = Settings()
