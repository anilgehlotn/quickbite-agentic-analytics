"""Typed contracts for every hand-off between agents.

These models are the interfaces that make the orchestration real. Each one is
both a runtime validator and a piece of documentation: the ``description`` on
every field is written for an LLM audience, because those descriptions are
injected into agent prompts as the schema the agent must produce. Vague field
descriptions here become malformed agent output later.

Two validations are load-bearing rather than decorative:

* :class:`AnalysisPlan` rejects any metric that is not a key of
  ``METRIC_DEFINITIONS``. A planner that invents ``total_sales`` fails at the
  plan boundary, where the error names the problem, instead of producing SQL
  that silently returns nothing.
* Every model carries a realistic example in ``json_schema_extra``. Those
  examples are few-shot material for the agents, so the test suite validates
  each one against its own model - a wrong example would teach the wrong shape.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.config import settings
from app.semantic.schema import METRIC_DEFINITIONS


class QueryIntent(str, Enum):
    """What kind of question the user asked.

    The intent drives how the plan is shaped: a RANKING question needs an ORDER
    BY and a LIMIT, a DIAGNOSTIC question needs several supporting sub-queries
    rather than one.
    """

    AGGREGATE = "aggregate"
    RANKING = "ranking"
    COMPARISON = "comparison"
    TREND = "trend"
    DIAGNOSTIC = "diagnostic"
    UNSUPPORTED = "unsupported"
    AMBIGUOUS = "ambiguous"

    @property
    def is_terminal(self) -> bool:
        """Whether this intent ends the run before any SQL is written.

        Returns:
            True for the two intents that produce a reply rather than an
            analysis.
        """
        return self in {QueryIntent.UNSUPPORTED, QueryIntent.AMBIGUOUS}


class ResponseStatus(str, Enum):
    """What kind of reply the pipeline produced.

    ``answered`` is not enough on its own: a clarification request and a
    genuine failure are both "not answered" but mean completely different
    things to a client, and rendering them the same way would tell a user their
    question broke the system when it was merely underspecified.
    """

    ANSWERED = "answered"
    CLARIFICATION_NEEDED = "clarification_needed"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"


class VerificationStatus(str, Enum):
    """Outcome of verifying a set of query results."""

    PASSED = "passed"
    PASSED_WITH_WARNINGS = "passed_with_warnings"
    FAILED = "failed"


class ChartType(str, Enum):
    """The chart shape that best fits a result set."""

    BAR = "bar"
    LINE = "line"
    GROUPED_BAR = "grouped_bar"
    NONE = "none"


class AgentStatus(str, Enum):
    """Execution state of one agent step."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class TimeWindow(BaseModel):
    """The date range a question covers, resolved against the fixed anchor.

    Dates are always absolute by the time they reach this model. Resolving
    "last 3 months" against the dataset's as-of date is the planner's job, and
    doing it here rather than in SQL keeps every downstream query anchored to
    the same window.
    """

    start_date: date = Field(
        description=(
            "First date of the window, inclusive, as YYYY-MM-DD. Must fall "
            "within the dataset range 2025-08-01 to 2026-07-31."
        )
    )
    end_date: date = Field(
        description=(
            "Last date of the window, inclusive, as YYYY-MM-DD. Must be on or "
            "after start_date."
        )
    )
    label: str = Field(
        description=(
            "Human-readable name for this window as the user would say it, for "
            "example 'last 3 months', 'July 2026' or 'the full year'. Used "
            "verbatim when explaining the answer."
        )
    )
    comparison_start: date | None = Field(
        default=None,
        description=(
            "First date of the period being compared against, for "
            "period-over-period questions. Null when the question does not "
            "compare two periods."
        ),
    )
    comparison_end: date | None = Field(
        default=None,
        description=(
            "Last date of the comparison period, inclusive. Null when the "
            "question does not compare two periods."
        ),
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "start_date": "2026-05-01",
                "end_date": "2026-07-31",
                "label": "last 3 months",
                "comparison_start": "2026-02-01",
                "comparison_end": "2026-04-30",
            }
        }
    )

    @field_validator("end_date")
    @classmethod
    def validate_range(cls, value: date, info: Any) -> date:
        """Reject windows that run backwards.

        Args:
            value: The proposed end date.
            info: Validation context carrying the already-validated fields.

        Returns:
            The end date, unchanged.

        Raises:
            ValueError: If the end date precedes the start date.
        """
        start = info.data.get("start_date")
        if start is not None and value < start:
            raise ValueError(f"end_date {value} precedes start_date {start}")
        return value


class Clarification(BaseModel):
    """A single focused question back to the user, with ways to answer it.

    Only produced when guessing would give a misleading answer. Most questions
    have a sensible default and should simply be answered - defaulting is
    cheaper for the user than a round trip, and a system that asks about
    everything is more annoying than one that occasionally assumes.

    The options are complete questions rather than fragments, so a client can
    submit one verbatim as the next question and a user can see exactly what
    they will get before clicking.
    """

    question: str = Field(
        min_length=8,
        description=(
            "The single thing that must be settled before the question can be "
            "answered, phrased as a direct question to the user."
        ),
    )
    reason: str = Field(
        description=(
            "Why this cannot be defaulted: what would be wrong about the "
            "answer if the wrong interpretation were assumed."
        )
    )
    options: list[str] = Field(
        min_length=2,
        max_length=4,
        description=(
            "Two to four complete, self-contained questions the user can ask "
            "instead. Each must be answerable on its own without further "
            "context, because a client submits it verbatim."
        ),
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "Which sense of 'best' did you mean?",
                "reason": (
                    "The highest-selling product by units is a different "
                    "product from the highest by revenue, so answering the "
                    "wrong one would name the wrong item."
                ),
                "options": [
                    "Which products sold the most units in the last 3 months?",
                    "Which products generated the most revenue in the last 3 "
                    "months?",
                ],
            }
        }
    )


class SubQuery(BaseModel):
    """One piece of analysis within a plan.

    A sub-query is a business question small enough to answer with a single SQL
    statement. Splitting a diagnostic question into several named sub-queries is
    what makes the final explanation traceable to specific numbers.
    """

    id: str = Field(
        description=(
            "Short stable identifier, lowercase with underscores, for example "
            "'monthly_revenue' or 'channel_breakdown'. Query results reference "
            "this id, so it must be unique within the plan."
        )
    )
    purpose: str = Field(
        description=(
            "The business question this sub-query answers, in one sentence. "
            "Written for a reader, not as a restatement of the SQL."
        )
    )
    tables: list[str] = Field(
        default_factory=list,
        description=(
            "Optional hint: which tables this sub-query is expected to read. "
            "Must be names from the table allowlist. Leave empty to let the "
            "SQL agent decide."
        ),
    )
    metrics: list[str] = Field(
        default_factory=list,
        description=(
            "Optional hint: which canonical metrics this sub-query computes. "
            "Must be keys of METRIC_DEFINITIONS."
        ),
    )
    dimensions: list[str] = Field(
        default_factory=list,
        description=(
            "Optional hint: which columns to group by, for example 'city', "
            "'channel' or 'month_key'."
        ),
    )
    filters: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Optional hint: column-to-value filters beyond the plan's time "
            "window, for example {'channel': 'Zomato'}."
        ),
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "store_monthly_revenue",
                "purpose": (
                    "Revenue per store for each of the last three months, to "
                    "identify stores declining every month."
                ),
                "tables": ["mart_store_month"],
                "metrics": ["revenue"],
                "dimensions": ["store_id", "month_key"],
                "filters": {},
            }
        }
    )


class AnalysisPlan(BaseModel):
    """The planner agent's output: how the question will be answered.

    The plan is written before any SQL exists, so it is the cheapest place to
    catch a misunderstanding. A metric that is not in the semantic layer is
    rejected here rather than becoming a query that returns nothing.
    """

    question: str = Field(
        description="The user's original question, verbatim and unedited."
    )
    intent: QueryIntent = Field(
        description=(
            "The kind of question this is. Use DIAGNOSTIC when the user asks "
            "why something happened, TREND for change over time, RANKING for "
            "best or worst, COMPARISON for one group against another, "
            "AGGREGATE for a single total, and UNSUPPORTED when the question "
            "cannot be answered from this dataset."
        )
    )
    time_window: TimeWindow = Field(
        description=(
            "The date range to analyse, already resolved to absolute dates "
            "against the fixed as-of date of 2026-07-31."
        )
    )
    metrics: list[str] = Field(
        description=(
            "Canonical metric names this analysis computes. Every entry must be "
            "a key of METRIC_DEFINITIONS; inventing a metric name is an error."
        )
    )
    dimensions: list[str] = Field(
        default_factory=list,
        description=(
            "Columns to break the metrics down by, for example ['city'] or "
            "['channel', 'month_key']. Empty for a single overall number."
        ),
    )
    sub_queries: list[SubQuery] = Field(
        default_factory=list,
        description=(
            "The individual queries to run, in order. At least one is "
            "required unless the intent is unsupported or ambiguous, neither "
            "of which runs any SQL. A diagnostic question typically needs "
            "several: one for the headline number and others for the "
            "supporting evidence."
        ),
    )
    clarification: Clarification | None = Field(
        default=None,
        description=(
            "The question to put back to the user. Required when intent is "
            "ambiguous and forbidden otherwise: a plan that both asks and "
            "answers leaves the caller to decide which it meant."
        ),
    )
    requires_diagnostics: bool = Field(
        default=False,
        description=(
            "True when the question asks why something happened rather than "
            "what happened, and therefore needs supporting evidence beyond the "
            "headline figure."
        )
    )
    reasoning: str = Field(
        description=(
            "Why the plan is shaped this way: why these metrics, this window "
            "and this set of sub-queries. Read by the verifier to check the "
            "plan actually addresses the question."
        )
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "How confident the planner is that this plan answers the question "
            "asked, from 0.0 to 1.0. Below 0.5 signals the question was "
            "ambiguous."
        ),
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "Which stores are declining and why?",
                "intent": "diagnostic",
                "time_window": {
                    "start_date": "2026-05-01",
                    "end_date": "2026-07-31",
                    "label": "last 3 months",
                    "comparison_start": "2026-02-01",
                    "comparison_end": "2026-04-30",
                },
                "metrics": ["revenue", "orders", "aov"],
                "dimensions": ["store_id", "month_key"],
                "sub_queries": [
                    {
                        "id": "consistently_declining_stores",
                        "purpose": (
                            "Return exactly the stores whose revenue fell in "
                            "every consecutive month, computed in SQL by "
                            "joining the monthly rows against each other, so "
                            "the qualifying set is not left to be worked out "
                            "by reading a table."
                        ),
                        "tables": ["mart_store_month"],
                        "metrics": ["revenue"],
                        "dimensions": ["store_id"],
                        "filters": {},
                    },
                    {
                        "id": "store_monthly_revenue",
                        "purpose": (
                            "Revenue per store per month, so the shape of "
                            "each decline can be shown rather than only its "
                            "endpoints."
                        ),
                        "tables": ["mart_store_month"],
                        "metrics": ["revenue", "orders", "aov"],
                        "dimensions": ["store_id", "month_key"],
                        "filters": {},
                    },
                    {
                        "id": "declining_store_baseline",
                        "purpose": (
                            "One row per store with the baseline comparison "
                            "already computed as columns: window_revenue, "
                            "baseline_revenue, delta_abs, delta_pct and "
                            "is_above_baseline. This separates stores that "
                            "are genuinely deteriorating from stores "
                            "reverting after an unusually strong month, "
                            "without anyone having to compare two numbers "
                            "across fifty rows."
                        ),
                        "tables": ["mart_store_month"],
                        "metrics": ["revenue", "orders"],
                        "dimensions": ["store_id"],
                        "filters": {},
                    },
                    {
                        "id": "declining_store_channels",
                        "purpose": (
                            "Revenue by channel for the declining stores, to "
                            "see whether the fall is concentrated in one "
                            "channel."
                        ),
                        "tables": ["fact_orders", "dim_store"],
                        "metrics": ["revenue"],
                        "dimensions": ["store_id", "channel", "month_key"],
                        "filters": {},
                    },
                ],
                "requires_diagnostics": True,
                "reasoning": (
                    "The question asks why, so a single revenue total is not "
                    "enough. The first sub-query settles which stores qualify, "
                    "in SQL rather than by inspection. The second shows the "
                    "shape of each decline month by month. The third compares "
                    "each store against its own prior quarter, because a "
                    "store falling from an unusually strong month may still "
                    "be above its historical run rate and naming it as the "
                    "top concern would be arithmetically correct and "
                    "analytically wrong. The fourth attributes the fall to a "
                    "channel, and orders against AOV separates a volume "
                    "problem from a basket-size one."
                ),
                "confidence": 0.9,
            }
        }
    )

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, value: list[str]) -> list[str]:
        """Reject metric names the semantic layer does not define.

        Args:
            value: The proposed metric names.

        Returns:
            The metric names, unchanged.

        Raises:
            ValueError: If any name is not a key of ``METRIC_DEFINITIONS``.
        """
        unknown = [metric for metric in value if metric not in METRIC_DEFINITIONS]
        if unknown:
            raise ValueError(
                f"unknown metric(s) {unknown}; valid metrics are "
                f"{sorted(METRIC_DEFINITIONS)}"
            )
        return value

    @model_validator(mode="after")
    def validate_intent_consistency(self) -> AnalysisPlan:
        """Tie the intent to the fields it implies.

        Three rules, each catching a plan that would otherwise fail confusingly
        much later:

        * An answerable intent needs at least one sub-query. Without this a
          plan can be valid and yet produce no numbers at all.
        * An ambiguous intent needs a clarification, or the caller has nothing
          to put back to the user.
        * Anything other than ambiguous must not carry a clarification. A plan
          that both asks a question and answers one leaves the caller to guess
          which was meant, and the safe guess is the wrong one.

        Returns:
            The validated plan.

        Raises:
            ValueError: If the intent and the fields disagree.
        """
        if not self.intent.is_terminal and not self.sub_queries:
            raise ValueError(
                f"intent '{self.intent.value}' requires at least one sub-query"
            )
        if self.intent is QueryIntent.AMBIGUOUS and self.clarification is None:
            raise ValueError(
                "intent 'ambiguous' requires a clarification with the question "
                "to ask and the interpretations to offer"
            )
        if self.intent is not QueryIntent.AMBIGUOUS and self.clarification is not None:
            raise ValueError(
                f"intent '{self.intent.value}' must not carry a clarification; "
                f"use intent 'ambiguous' to ask the user a question"
            )
        return self


class QueryResult(BaseModel):
    """The outcome of running one sub-query.

    Carries the exact SQL that ran, not a reconstruction. When a number is
    questioned later, this is the audit trail.
    """

    sub_query_id: str = Field(
        description="The id of the SubQuery this result belongs to."
    )
    sql: str = Field(
        description=(
            "The exact SQL that was executed, including any modifications made "
            "during retries. Not a template or a reconstruction."
        )
    )
    columns: list[str] = Field(
        default_factory=list,
        description="Column names of the result set, in order.",
    )
    rows: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "The result rows, each a mapping of column name to value. Empty "
            "when the query matched nothing, which is itself a finding."
        ),
    )
    row_count: int = Field(
        ge=0, description="Number of rows returned. Zero is a valid result."
    )
    execution_ms: float = Field(
        ge=0.0, description="Wall-clock execution time in milliseconds."
    )
    error: str | None = Field(
        default=None,
        description=(
            "The database error message if the query failed after all retries, "
            "otherwise null."
        ),
    )
    attempts: int = Field(
        default=1,
        ge=1,
        description=(
            "How many times this query was attempted, including the first. "
            "Greater than one means SQL was repaired after an error."
        ),
    )
    degraded: bool = Field(
        default=False,
        description=(
            "True when this SQL was assembled deterministically from the plan "
            "because no model was reachable. The numbers are exact - it is "
            "real SQL over the real database - but the query is a plain "
            "aggregate rather than a tailored one, so it may answer a simpler "
            "question than the one asked. Surfaced so the answer can say so."
        ),
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "sub_query_id": "store_monthly_revenue",
                "sql": (
                    "SELECT store_id, month_key, revenue_net FROM "
                    "mart_store_month WHERE month_key >= '2026-05' ORDER BY "
                    "store_id, month_key"
                ),
                "columns": ["store_id", "month_key", "revenue_net"],
                "rows": [
                    {
                        "store_id": "ST039",
                        "month_key": "2026-05",
                        "revenue_net": 25150.0,
                    },
                    {
                        "store_id": "ST039",
                        "month_key": "2026-06",
                        "revenue_net": 22545.0,
                    },
                ],
                "row_count": 2,
                "execution_ms": 4.2,
                "error": None,
                "attempts": 1,
            }
        }
    )


class VerificationCheck(BaseModel):
    """One assertion made about the query results."""

    name: str = Field(
        description=(
            "Stable identifier for the check, lowercase with underscores, for "
            "example 'time_window_matches_plan'."
        )
    )
    passed: bool = Field(description="Whether the assertion held.")
    severity: str = Field(
        description=(
            "How much a failure matters: 'error' means the answer cannot be "
            "trusted, 'warning' means it is usable but caveated, 'info' is an "
            "observation that never fails verification."
        )
    )
    message: str = Field(
        description=(
            "What was checked and what was found, including the actual numbers "
            "so a reader can judge the result without re-running anything."
        )
    )
    details: dict[str, Any] | None = Field(
        default=None,
        description="Optional structured evidence, JSON-serializable.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "revenue_metric_is_canonical",
                "passed": True,
                "severity": "error",
                "message": (
                    "Query uses net_before_tax, the canonical tax-exclusive "
                    "revenue column."
                ),
                "details": {"column": "net_before_tax"},
            }
        }
    )


class VerificationReport(BaseModel):
    """The verifier agent's judgement on a set of results."""

    status: VerificationStatus = Field(
        description=(
            "PASSED when every check held, PASSED_WITH_WARNINGS when only "
            "warnings failed, FAILED when any error-severity check failed."
        )
    )
    checks: list[VerificationCheck] = Field(
        default_factory=list, description="Every check that was run, in order."
    )
    summary: str = Field(
        description=(
            "One or two sentences on whether the answer can be trusted and "
            "why, written for the end user rather than for a developer."
        )
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "passed_with_warnings",
                "checks": [
                    {
                        "name": "revenue_metric_is_canonical",
                        "passed": True,
                        "severity": "error",
                        "message": "Query uses net_before_tax as required.",
                        "details": None,
                    },
                    {
                        "name": "line_grain_not_used_for_revenue",
                        "passed": False,
                        "severity": "warning",
                        "message": (
                            "Product breakdown uses line grain, which differs "
                            "from the order grain by about 0.44% in this window."
                        ),
                        "details": {"variance_pct": 0.44},
                    },
                ],
                "summary": (
                    "The headline revenue figure is sound. The per-SKU split "
                    "is line-grain and will not sum exactly to it."
                ),
            }
        }
    )

    @property
    def passed(self) -> bool:
        """Whether the answer can be trusted.

        Returns:
            True unless any error-severity check failed.
        """
        return not any(
            check.severity == "error" and not check.passed for check in self.checks
        )

    @property
    def has_warnings(self) -> bool:
        """Whether any non-fatal check failed.

        Returns:
            True when at least one warning-severity check failed.
        """
        return any(
            check.severity == "warning" and not check.passed for check in self.checks
        )


class Insight(BaseModel):
    """The explanation agent's output: what the numbers mean.

    The ``caveats`` field is not optional politeness. A careful analyst states
    the limits of their own answer, and an agent that never does is
    indistinguishable from one that has not noticed them.
    """

    headline: str = Field(
        description=(
            "The direct answer in one sentence, leading with the number. This "
            "is what a reader sees first, so it must stand alone."
        )
    )
    narrative: str = Field(
        description=(
            "The business explanation: what the numbers show, what drove it, "
            "and what it means. Written for a business reader, not an analyst; "
            "no SQL, no column names."
        )
    )
    key_findings: list[str] = Field(
        default_factory=list,
        description=(
            "The supporting points, each a complete sentence containing the "
            "specific figure it rests on."
        ),
    )
    caveats: list[str] = Field(
        default_factory=list,
        description=(
            "Data limitations and interpretation warnings a careful analyst "
            "would state unprompted: known grain variances, seasonality that "
            "could explain the result, small sample sizes, or a metric that "
            "does not mean what the reader may assume."
        ),
    )
    recommended_actions: list[str] = Field(
        default_factory=list,
        description=(
            "Concrete next steps the business could take, each tied to a "
            "finding. Empty when the question was purely informational."
        ),
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "How confident the explanation is, from 0.0 to 1.0. Lower this "
            "when verification raised warnings or the data is thin."
        ),
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "headline": (
                    "Nine of fifty stores declined in every one of the last "
                    "three months, together losing 48,860 INR."
                ),
                "narrative": (
                    "The decline is store-specific rather than market-wide: no "
                    "city fell in every consecutive month. Seven of the nine "
                    "stores are losing orders rather than basket size, and at "
                    "every one the fall concentrates in a single channel while "
                    "another channel grows, which points to an operational "
                    "cause rather than lost demand."
                ),
                "key_findings": [
                    (
                        "ST039 Kolkata 39 fell 41.4%, the steepest in the "
                        "estate."
                    ),
                    (
                        "Seven of the nine declines are driven by order volume; "
                        "only ST011 and ST007 are basket-size stories."
                    ),
                    (
                        "Only five of the nine are also below their own prior "
                        "three months, so four are reverting from a strong May."
                    ),
                ],
                "caveats": [
                    (
                        "Revenue peaks in October and troughs in February, a "
                        "1.36x seasonal spread, so a three-month decline is "
                        "not on its own evidence of a problem."
                    ),
                    (
                        "Distinct customer counts exclude the 28% of orders "
                        "that are anonymous walk-ins, so they are directional "
                        "rather than a footfall count."
                    ),
                ],
                "recommended_actions": [
                    (
                        "Review aggregator listings and delivery coverage at "
                        "ST042, ST015 and ST010, where one channel collapsed "
                        "while another grew."
                    ),
                    (
                        "Treat ST039 as lower priority despite its headline: "
                        "it is 28% above its own prior quarter."
                    ),
                ],
                "confidence": 0.85,
            }
        }
    )


class ChartSpec(BaseModel):
    """How to visualise a result set."""

    chart_type: ChartType = Field(
        description=(
            "LINE for a metric over time, BAR for comparing categories, "
            "GROUPED_BAR for categories split by a second dimension, NONE when "
            "the answer is a single number that a chart would not improve."
        )
    )
    x_field: str = Field(
        description=(
            "Column plotted on the x axis, for example 'month_key' or "
            "'store_name'. Empty string when chart_type is NONE."
        )
    )
    y_fields: list[str] = Field(
        default_factory=list,
        description=(
            "Columns plotted on the y axis. More than one draws multiple "
            "series."
        ),
    )
    title: str = Field(
        description="Chart title, phrased as the finding rather than the axes."
    )
    series_field: str | None = Field(
        default=None,
        description=(
            "Column that splits the data into series for a grouped bar or "
            "multi-line chart, for example 'channel'. Null for a single series."
        ),
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chart_type": "line",
                "x_field": "month_key",
                "y_fields": ["revenue_net"],
                "title": "Revenue fell in each of the last three months",
                "series_field": "store_name",
            }
        }
    )


class AgentStep(BaseModel):
    """One agent's execution record within a run."""

    agent_name: str = Field(
        description=(
            "Which agent ran, for example 'planner', 'sql_generator', "
            "'verifier' or 'explainer'."
        )
    )
    status: AgentStatus = Field(description="How the step ended.")
    started_at: datetime = Field(
        description="When the step began, as an ISO 8601 timestamp."
    )
    duration_ms: float = Field(
        ge=0.0, description="How long the step took, in milliseconds."
    )
    summary: str = Field(
        description=(
            "One sentence on what this step did or decided, written for a user "
            "watching the run rather than for a log."
        )
    )
    error: str | None = Field(
        default=None,
        description="The failure reason when status is FAILED, otherwise null.",
    )
    llm_provider: str | None = Field(
        default=None,
        description=(
            "Which LLM provider served this step, or null for steps that made "
            "no model call, such as SQL execution."
        ),
    )
    tokens: int | None = Field(
        default=None,
        ge=0,
        description="Total tokens consumed by this step, or null if none.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "agent_name": "planner",
                "status": "succeeded",
                "started_at": "2026-07-31T10:15:00Z",
                "duration_ms": 1240.5,
                "summary": (
                    "Classified the question as diagnostic and split it into "
                    "two sub-queries."
                ),
                "error": None,
                "llm_provider": "anthropic",
                "tokens": 1830,
            }
        }
    )


class AgentTrace(BaseModel):
    """The full record of one run, step by step.

    This is what the frontend renders to show the system is genuinely agentic:
    a claim that four agents collaborated is unfalsifiable without the trace
    that shows each one running, how long it took and what it decided.
    """

    steps: list[AgentStep] = Field(
        default_factory=list, description="Every step, in execution order."
    )
    total_duration_ms: float = Field(
        ge=0.0,
        description=(
            "End-to-end wall-clock time for the run. May exceed the sum of the "
            "steps if any ran concurrently."
        ),
    )
    total_tokens: int = Field(
        ge=0, description="Tokens consumed across every step."
    )
    providers_used: list[str] = Field(
        default_factory=list,
        description=(
            "Distinct LLM providers that served this run. More than one means "
            "failover occurred mid-run."
        ),
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "steps": [
                    {
                        "agent_name": "planner",
                        "status": "succeeded",
                        "started_at": "2026-07-31T10:15:00Z",
                        "duration_ms": 1240.5,
                        "summary": "Split the question into two sub-queries.",
                        "error": None,
                        "llm_provider": "anthropic",
                        "tokens": 1830,
                    },
                    {
                        "agent_name": "sql_generator",
                        "status": "succeeded",
                        "started_at": "2026-07-31T10:15:01Z",
                        "duration_ms": 890.0,
                        "summary": "Generated and ran two queries.",
                        "error": None,
                        "llm_provider": "anthropic",
                        "tokens": 2100,
                    },
                ],
                "total_duration_ms": 2130.5,
                "total_tokens": 3930,
                "providers_used": ["anthropic"],
            }
        }
    )


class AnalysisResponse(BaseModel):
    """The complete answer to one question: the top-level API response."""

    question: str = Field(description="The user's original question, verbatim.")
    answered: bool = Field(
        description=(
            "True when a trustworthy answer was produced. False when the "
            "question was unsupported or the run failed, in which case error "
            "explains why."
        )
    )
    plan: AnalysisPlan | None = Field(
        default=None,
        description="The plan that was executed, or null if planning failed.",
    )
    query_results: list[QueryResult] = Field(
        default_factory=list,
        description="Every query that ran, with its SQL and rows.",
    )
    verification: VerificationReport | None = Field(
        default=None,
        description="The verifier's judgement, or null if it did not run.",
    )
    insight: Insight | None = Field(
        default=None,
        description="The business explanation, or null if none was produced.",
    )
    chart: ChartSpec | None = Field(
        default=None,
        description=(
            "How to visualise the result, or null when no chart is warranted."
        ),
    )
    trace: AgentTrace = Field(
        description="The step-by-step record of the run, always present."
    )
    data_asof: date = Field(
        description=(
            "The dataset's fixed as-of date, always 2026-07-31. Included so a "
            "client can show what 'today' meant for this answer."
        )
    )
    from_cache: bool = Field(
        default=False,
        description="True when this answer was served from cache.",
    )
    request_id: str | None = Field(
        default=None,
        description=(
            "Correlation id for this request, matching the X-Request-ID header "
            "and the server logs. Quotable when reporting a problem."
        ),
    )
    error: str | None = Field(
        default=None,
        description="Why the run failed, when answered is false.",
    )
    status: ResponseStatus = Field(
        default=ResponseStatus.ANSWERED,
        description=(
            "What kind of reply this is. 'answered' carries numbers; "
            "'clarification_needed' carries a question back to the user; "
            "'unsupported' means the data cannot answer it; 'failed' means "
            "something broke. A client renders these four differently, and "
            "collapsing them into the answered flag makes a question that was "
            "merely underspecified look like an outage."
        ),
    )
    clarification: Clarification | None = Field(
        default=None,
        description=(
            "The question to put back to the user, present only when status "
            "is 'clarification_needed'."
        ),
    )
    suggested_questions: list[str] = Field(
        default_factory=list,
        description=(
            "Complete, answerable questions to offer instead of this one. "
            "Populated when the question was unsupported or ambiguous, so the "
            "user always has somewhere to go next rather than a dead end."
        ),
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "What was our revenue in the last 3 months?",
                "answered": True,
                "plan": {
                    "question": "What was our revenue in the last 3 months?",
                    "intent": "aggregate",
                    "time_window": {
                        "start_date": "2026-05-01",
                        "end_date": "2026-07-31",
                        "label": "last 3 months",
                        "comparison_start": None,
                        "comparison_end": None,
                    },
                    "metrics": ["revenue", "orders", "aov"],
                    "dimensions": [],
                    "sub_queries": [
                        {
                            "id": "headline_revenue",
                            "purpose": (
                                "Total revenue, order count and AOV for the "
                                "window."
                            ),
                            "tables": ["fact_orders"],
                            "metrics": ["revenue", "orders", "aov"],
                            "dimensions": [],
                            "filters": {},
                        }
                    ],
                    "requires_diagnostics": False,
                    "reasoning": (
                        "A single aggregate over the resolved window answers "
                        "the question directly."
                    ),
                    "confidence": 0.97,
                },
                "query_results": [
                    {
                        "sub_query_id": "headline_revenue",
                        "sql": (
                            "SELECT ROUND(SUM(net_before_tax), 2) AS "
                            "revenue_inr FROM fact_orders WHERE order_date "
                            "BETWEEN '2026-05-01' AND '2026-07-31'"
                        ),
                        "columns": ["revenue_inr"],
                        "rows": [{"revenue_inr": 3197076.5}],
                        "row_count": 1,
                        "execution_ms": 3.1,
                        "error": None,
                        "attempts": 1,
                    }
                ],
                "verification": {
                    "status": "passed",
                    "checks": [
                        {
                            "name": "revenue_metric_is_canonical",
                            "passed": True,
                            "severity": "error",
                            "message": "Query uses net_before_tax as required.",
                            "details": None,
                        }
                    ],
                    "summary": "All checks passed; the figure can be trusted.",
                },
                "insight": {
                    "headline": (
                        "Revenue was 3,197,077 INR over the last three months "
                        "across 4,930 orders."
                    ),
                    "narrative": (
                        "Trading was steady through the window, with July the "
                        "strongest of the three months. Average order value "
                        "held at 648 INR."
                    ),
                    "key_findings": [
                        "July was the strongest month at 1,103,236 INR.",
                        "AOV was flat across the window at about 648 INR.",
                    ],
                    "caveats": [
                        (
                            "Revenue excludes the 5% tax, which is the "
                            "canonical definition used throughout."
                        )
                    ],
                    "recommended_actions": [],
                    "confidence": 0.95,
                },
                "chart": {
                    "chart_type": "line",
                    "x_field": "month_key",
                    "y_fields": ["revenue_net"],
                    "title": "Revenue by month, last 3 months",
                    "series_field": None,
                },
                "trace": {
                    "steps": [
                        {
                            "agent_name": "planner",
                            "status": "succeeded",
                            "started_at": "2026-07-31T10:15:00Z",
                            "duration_ms": 1240.5,
                            "summary": "Classified the question as aggregate.",
                            "error": None,
                            "llm_provider": "anthropic",
                            "tokens": 1830,
                        }
                    ],
                    "total_duration_ms": 1240.5,
                    "total_tokens": 1830,
                    "providers_used": ["anthropic"],
                },
                "data_asof": "2026-07-31",
                "from_cache": False,
                "error": None,
            }
        }
    )

    @classmethod
    def unanswered(
        cls, question: str, error: str, trace: AgentTrace | None = None
    ) -> AnalysisResponse:
        """Build a failure response that is still well formed.

        A failed run must return the same shape as a successful one so the
        frontend never has to branch on whether the payload is an answer or an
        exception.

        Args:
            question: The user's original question.
            error: Why the run could not produce an answer.
            trace: The partial trace, when one exists.

        Returns:
            A response with ``answered`` false and the error attached.
        """
        return cls(
            question=question,
            answered=False,
            trace=trace
            or AgentTrace(total_duration_ms=0.0, total_tokens=0, providers_used=[]),
            data_asof=settings.DATA_ASOF_DATE,
            error=error,
            status=ResponseStatus.FAILED,
        )

    @classmethod
    def needs_clarification(
        cls,
        question: str,
        plan: AnalysisPlan,
        clarification: Clarification,
        trace: AgentTrace,
    ) -> AnalysisResponse:
        """Build a response that asks the user one question back.

        This is not a failure and must not read like one. The insight carries
        the clarifying question as its headline so a client that only renders
        insights still shows something sensible, and the interpretations are
        repeated in ``suggested_questions`` so they can be offered as one-click
        follow-ups.

        Args:
            question: The user's original question.
            plan: The plan that identified the ambiguity.
            clarification: What to ask and what to offer.
            trace: The trace of the run so far.

        Returns:
            A response with status ``clarification_needed``.
        """
        return cls(
            question=question,
            answered=False,
            plan=plan,
            trace=trace,
            data_asof=settings.DATA_ASOF_DATE,
            status=ResponseStatus.CLARIFICATION_NEEDED,
            clarification=clarification,
            suggested_questions=list(clarification.options),
            insight=Insight(
                headline=clarification.question,
                narrative=clarification.reason,
                key_findings=list(clarification.options),
                caveats=[],
                recommended_actions=[],
                confidence=0.0,
            ),
            error=None,
        )
