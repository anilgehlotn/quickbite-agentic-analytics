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

import json
from datetime import date
from pathlib import Path
from typing import Annotated, Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

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

# Filename of the warmed answer cache. Committed to the repository so a
# deployment can answer the evaluation questions with no provider available.
_CACHE_FILENAME: str = "answer_cache.json"


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
    # Seasonality
    # ------------------------------------------------------------------
    # Measured facts about the shipped extract, used to caveat month-to-month
    # comparisons. Without them an agent reads any three-month fall as a
    # deterioration, when part of it is the annual shape of the business.

    SEASONAL_PEAK_MONTH: str = "2025-10"
    SEASONAL_TROUGH_MONTH: str = "2026-02"

    # Peak month revenue divided by trough month revenue.
    SEASONAL_SPREAD: float = 1.36

    # ------------------------------------------------------------------
    # Verification bounds
    # ------------------------------------------------------------------
    # Sanity limits for agent-produced numbers. A figure outside these is not
    # a surprising result, it is a broken query - a fan-out join or the wrong
    # grain - so the verifier treats it as an error rather than a finding.

    # Full-year revenue is about 13M INR, so nothing in this dataset can
    # legitimately exceed this.
    MAX_PLAUSIBLE_REVENUE_INR: float = 15_000_000.0

    # AOV is derived, so revenue / orders must reproduce it to within this
    # many INR. Wider than rounding, narrower than a real inconsistency.
    AOV_TOLERANCE_INR: float = 0.5

    # A breakdown's parts must sum to the reported total to within this.
    TOTAL_RECONCILIATION_TOLERANCE_INR: float = 1.0

    # Share and percentage columns must sum to 100 within this many points.
    SHARE_SUM_TOLERANCE_PCT: float = 1.0

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    # Absolute, derived from this file's location at import time.

    PROJECT_ROOT: Path = _PROJECT_ROOT
    DATA_DIR: Path = _DATA_DIR
    EXCEL_PATH: Path = _DATA_DIR / _EXCEL_FILENAME
    DB_PATH: Path = _DATA_DIR / _DB_FILENAME

    # Warmed answer cache, committed to the repository. It is what lets the
    # eight evaluation questions be answered after every API key has expired.
    CACHE_PATH: Path = _DATA_DIR / _CACHE_FILENAME

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
    # LLM models and runtime
    # ------------------------------------------------------------------
    # Model IDs are deployment configuration, not constants. Providers release
    # and retire models on their own schedules, so a default that is current
    # today will eventually 404. Every one is env-overridable specifically so a
    # stale default can be corrected from the hosting dashboard without a code
    # change or redeploy of the image.
    # Where a provider publishes a floating "latest" alias, prefer it: a pinned
    # version is a dated default that eventually 404s, and a 404 is a permanent
    # failure that silently drops the provider out of the failover chain.
    ANTHROPIC_MODEL: str = "claude-opus-5"
    OPENAI_MODEL: str = "gpt-4o"
    # The lite alias rather than gemini-flash-latest: the latter currently
    # resolves to a model whose free tier allows 20 requests per day, which one
    # analysis exhausts in four questions. A deployment that a reviewer may
    # open weeks from now needs headroom more than it needs the stronger model,
    # and this is exactly the kind of thing an env override exists to correct.
    GEMINI_MODEL: str = "gemini-flash-lite-latest"
    GROK_MODEL: str = "grok-4-latest"

    # Per-request budget and determinism. Analytics answers must be
    # reproducible, so the default temperature is 0.
    LLM_MAX_TOKENS: int = 4096
    LLM_TEMPERATURE: float = 0.0

    # The insight agent writes the only long output in the system: a headline,
    # a narrative, several findings, caveats and actions, plus a chart spec. On
    # a diagnostic question that overruns the general budget, and a truncated
    # reply is invalid JSON, which costs the whole narrative rather than its
    # tail. This was observed in practice, not anticipated.
    INSIGHT_MAX_TOKENS: int = 8192

    # Per-request timeout, and retry policy for transient provider failures.
    LLM_TIMEOUT_SECONDS: int = 60
    LLM_MAX_RETRIES: int = 2
    LLM_RETRY_BACKOFF_SECONDS: float = 0.5

    # ------------------------------------------------------------------
    # Provider health
    #
    # A dead provider is worse than an absent one: every request pays its
    # timeout before failing over. The breaker turns that recurring cost into
    # a one-off, and the probe front-loads the discovery to startup.
    # ------------------------------------------------------------------

    #: Consecutive failures before a provider is taken out of the rotation.
    #: Two rather than one, because a single failure is often the request's
    #: fault rather than the provider's.
    CIRCUIT_BREAKER_THRESHOLD: int = 2

    #: How long a provider stays out before it is probed again. Long enough
    #: that a rate-limited provider has recovered, short enough that a brief
    #: outage does not sideline it for a whole session.
    CIRCUIT_BREAKER_COOLDOWN_SECONDS: float = 120.0

    #: Whether to probe each configured provider at startup and order the
    #: chain by which ones actually answered.
    PROVIDER_PROBE_ON_STARTUP: bool = True

    #: Seconds allowed for one startup probe. Short: the probe is a liveness
    #: check, and a provider that cannot answer a two-token prompt quickly is
    #: not one to put first.
    PROVIDER_PROBE_TIMEOUT_SECONDS: float = 12.0

    #: Minimum seconds between re-probes, so periodic health checks cannot
    #: turn into a spend.
    PROVIDER_PROBE_INTERVAL_SECONDS: float = 900.0

    # ------------------------------------------------------------------
    # Application settings
    # ------------------------------------------------------------------

    APP_NAME: str = "QuickBite Agentic Analytics"
    APP_VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"

    # Browser origins allowed to call the API. NoDecode disables the default
    # JSON parsing so a plain comma-separated string works: hosting dashboards
    # take flat strings, and a bare URL would otherwise fail JSON decoding and
    # crash the app at import. See the validator below.
    CORS_ORIGINS: Annotated[list[str], NoDecode] = ["http://localhost:3000"]

    # Port the server binds to. Hosting platforms inject this.
    PORT: int = 8000

    # Root logging level.
    LOG_LEVEL: str = "INFO"

    # Safety rails for generated SQL.
    MAX_QUERY_ROWS: int = 1000
    QUERY_TIMEOUT_SECONDS: int = 10

    # ------------------------------------------------------------------
    # API limits
    # ------------------------------------------------------------------
    # A public URL in front of paid APIs needs a cost boundary. Cached answers
    # are free and are deliberately exempt from both limits.

    RATE_LIMIT_PER_MINUTE: int = 10
    RATE_LIMIT_PER_DAY: int = 200

    # Accepted question length. The lower bound rejects noise; the upper bound
    # stops a pasted document becoming a prompt.
    MIN_QUESTION_LENGTH: int = 3
    MAX_QUESTION_LENGTH: int = 500

    # Whether answers are cached and served from cache at all.
    CACHE_ENABLED: bool = True

    # ------------------------------------------------------------------
    # Per-agent timeouts
    # ------------------------------------------------------------------
    # A hung agent must not hang the request, so every stage is bounded
    # independently. The analyst gets the largest budget because it may run
    # several sub-queries concurrently, each with its own repair retry.

    PLANNER_TIMEOUT_SECONDS: float = 45.0
    ANALYST_TIMEOUT_SECONDS: float = 90.0
    VERIFIER_TIMEOUT_SECONDS: float = 45.0
    INSIGHT_TIMEOUT_SECONDS: float = 90.0

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: Any) -> list[str]:
        """Accept CORS origins as a comma-separated string or a JSON array.

        Hosting dashboards (Render, Vercel) store environment variables as flat
        strings, so the common case is ``https://a.example,https://b.example``.
        A JSON array is still accepted for backwards compatibility.

        Args:
            value: The raw value from the environment, a .env file or a default.

        Returns:
            The origins as a list, with blanks and surrounding whitespace
            stripped.
        """
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("["):
                value = json.loads(text)
            else:
                return [origin.strip() for origin in text.split(",") if origin.strip()]
        if isinstance(value, (list, tuple)):
            return [str(origin).strip() for origin in value if str(origin).strip()]
        raise ValueError(f"cannot parse CORS_ORIGINS from {value!r}")

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
