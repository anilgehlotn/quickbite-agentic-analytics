"""Tests for the inter-agent contracts.

Two groups matter most.

Round-tripping proves every model survives the trip through JSON that it makes
on every agent hand-off and every API response — a model that serializes but
does not deserialize breaks at the boundary, not at the point of definition.

Example validation proves the ``json_schema_extra`` examples are correct. Those
examples are injected into agent prompts as the shape the agent must produce, so
a wrong example teaches the wrong shape and the failure surfaces as
mysteriously malformed agent output much later.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from app.agents.contracts import (
    AgentStatus,
    AgentStep,
    AgentTrace,
    AnalysisPlan,
    AnalysisResponse,
    ChartSpec,
    ChartType,
    Insight,
    QueryIntent,
    QueryResult,
    SubQuery,
    TimeWindow,
    VerificationCheck,
    VerificationReport,
    VerificationStatus,
)
from app.config import settings
from app.semantic.schema import METRIC_DEFINITIONS

# Every contract model, for the round-trip and example sweeps.
ALL_MODELS: tuple[type[BaseModel], ...] = (
    TimeWindow,
    SubQuery,
    AnalysisPlan,
    QueryResult,
    VerificationCheck,
    VerificationReport,
    Insight,
    ChartSpec,
    AgentStep,
    AgentTrace,
    AnalysisResponse,
)


def minimal_plan(**overrides: Any) -> AnalysisPlan:
    """Build a valid plan for tests to mutate.

    Args:
        **overrides: Fields to replace on the base plan.

    Returns:
        A valid AnalysisPlan.
    """
    payload: dict[str, Any] = {
        "question": "What was revenue last month?",
        "intent": QueryIntent.AGGREGATE,
        "time_window": TimeWindow(
            start_date=date(2026, 7, 1),
            end_date=date(2026, 7, 31),
            label="July 2026",
        ),
        "metrics": ["revenue"],
        "dimensions": [],
        "sub_queries": [SubQuery(id="total", purpose="Total revenue for July.")],
        "requires_diagnostics": False,
        "reasoning": "A single aggregate answers the question.",
        "confidence": 0.9,
    }
    payload.update(overrides)
    return AnalysisPlan(**payload)


class TestRoundTrip:
    """Every model must survive JSON serialization and deserialization."""

    @pytest.mark.parametrize("model", ALL_MODELS, ids=lambda m: m.__name__)
    def test_example_round_trips(self, model: type[BaseModel]) -> None:
        """The example serializes and deserializes to an equal object."""
        example = model.model_config["json_schema_extra"]["example"]
        instance = model.model_validate(example)

        restored = model.model_validate_json(instance.model_dump_json())

        assert restored == instance

    @pytest.mark.parametrize("model", ALL_MODELS, ids=lambda m: m.__name__)
    def test_dump_json_is_valid_json(self, model: type[BaseModel]) -> None:
        """model_dump_json produces parseable JSON."""
        import json

        example = model.model_config["json_schema_extra"]["example"]
        instance = model.model_validate(example)

        assert isinstance(json.loads(instance.model_dump_json()), dict)


class TestExamplesAreCorrect:
    """json_schema_extra examples become few-shot material, so they must work."""

    @pytest.mark.parametrize("model", ALL_MODELS, ids=lambda m: m.__name__)
    def test_every_model_has_an_example(self, model: type[BaseModel]) -> None:
        """Each model documents its shape with an example."""
        extra = model.model_config.get("json_schema_extra")
        assert extra is not None, model.__name__
        assert "example" in extra, model.__name__

    @pytest.mark.parametrize("model", ALL_MODELS, ids=lambda m: m.__name__)
    def test_example_validates_against_its_own_model(
        self, model: type[BaseModel]
    ) -> None:
        """A wrong example would teach agents the wrong shape."""
        example = model.model_config["json_schema_extra"]["example"]

        assert isinstance(model.model_validate(example), model)

    def test_plan_example_uses_real_metrics(self) -> None:
        """The plan example must not demonstrate an invented metric."""
        example = AnalysisPlan.model_config["json_schema_extra"]["example"]

        for metric in example["metrics"]:
            assert metric in METRIC_DEFINITIONS, metric

    def test_json_schema_generation_works(self) -> None:
        """Every model can emit a JSON schema for prompt injection."""
        for model in ALL_MODELS:
            schema = model.model_json_schema()
            assert schema["type"] == "object", model.__name__


class TestFieldDescriptions:
    """Descriptions are injected into agent prompts, so they must exist."""

    @pytest.mark.parametrize("model", ALL_MODELS, ids=lambda m: m.__name__)
    def test_every_field_has_a_description(self, model: type[BaseModel]) -> None:
        """A field with no description gives an agent nothing to work from."""
        for name, field in model.model_fields.items():
            assert field.description, f"{model.__name__}.{name}"
            assert len(field.description) > 15, f"{model.__name__}.{name}"


class TestAnalysisPlanValidation:
    """The plan is where a misunderstanding is cheapest to catch."""

    def test_rejects_unknown_metric(self) -> None:
        """An invented metric fails at the plan, not as empty SQL results."""
        with pytest.raises(ValidationError) as caught:
            minimal_plan(metrics=["total_sales"])

        assert "total_sales" in str(caught.value)

    def test_rejects_unknown_metric_among_valid_ones(self) -> None:
        """One bad metric in a valid list still fails."""
        with pytest.raises(ValidationError):
            minimal_plan(metrics=["revenue", "made_up_metric"])

    @pytest.mark.parametrize("metric", sorted(METRIC_DEFINITIONS))
    def test_accepts_every_defined_metric(self, metric: str) -> None:
        """Every metric the semantic layer defines is usable in a plan."""
        assert minimal_plan(metrics=[metric]).metrics == [metric]

    def test_requires_at_least_one_sub_query(self) -> None:
        """A plan with no queries cannot answer anything."""
        with pytest.raises(ValidationError):
            minimal_plan(sub_queries=[])

    def test_rejects_confidence_above_one(self) -> None:
        """Confidence is a probability, not a score."""
        with pytest.raises(ValidationError):
            minimal_plan(confidence=1.5)

    def test_rejects_negative_confidence(self) -> None:
        """Confidence cannot be below zero."""
        with pytest.raises(ValidationError):
            minimal_plan(confidence=-0.1)

    @pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
    def test_accepts_confidence_within_range(self, value: float) -> None:
        """The full 0-1 range is valid, endpoints included."""
        assert minimal_plan(confidence=value).confidence == value

    def test_rejects_unknown_intent(self) -> None:
        """Intent is a closed set."""
        with pytest.raises(ValidationError):
            minimal_plan(intent="guessing")

    def test_empty_metrics_list_is_allowed(self) -> None:
        """A plan may legitimately compute no named metric."""
        assert minimal_plan(metrics=[]).metrics == []


class TestTimeWindowValidation:
    """The time window is where the fixed anchor lands."""

    def test_rejects_backwards_range(self) -> None:
        """An end date before the start date is a planner bug."""
        with pytest.raises(ValidationError, match="precedes"):
            TimeWindow(
                start_date=date(2026, 7, 31),
                end_date=date(2026, 5, 1),
                label="backwards",
            )

    def test_accepts_single_day_window(self) -> None:
        """Start and end may be the same day."""
        window = TimeWindow(
            start_date=date(2026, 7, 31), end_date=date(2026, 7, 31), label="today"
        )

        assert window.start_date == window.end_date

    def test_comparison_period_is_optional(self) -> None:
        """Most questions do not compare two periods."""
        window = TimeWindow(
            start_date=date(2026, 5, 1), end_date=date(2026, 7, 31), label="last 3m"
        )

        assert window.comparison_start is None
        assert window.comparison_end is None

    def test_parses_iso_date_strings(self) -> None:
        """Agents emit JSON, so dates arrive as strings."""
        window = TimeWindow.model_validate(
            {"start_date": "2026-05-01", "end_date": "2026-07-31", "label": "last 3m"}
        )

        assert window.start_date == settings.LAST_3M_START
        assert window.end_date == settings.LAST_3M_END


class TestVerificationReport:
    """The report's verdict logic."""

    def test_passed_is_false_when_an_error_check_fails(self) -> None:
        """A failed error-severity check means the answer is untrustworthy."""
        report = VerificationReport(
            status=VerificationStatus.FAILED,
            checks=[
                VerificationCheck(
                    name="ok", passed=True, severity="error", message="fine"
                ),
                VerificationCheck(
                    name="broken", passed=False, severity="error", message="wrong"
                ),
            ],
            summary="One check failed.",
        )

        assert report.passed is False

    def test_passed_is_true_when_only_a_warning_fails(self) -> None:
        """A failed warning is caveated, not fatal."""
        report = VerificationReport(
            status=VerificationStatus.PASSED_WITH_WARNINGS,
            checks=[
                VerificationCheck(
                    name="grain", passed=False, severity="warning", message="variance"
                )
            ],
            summary="Usable with a caveat.",
        )

        assert report.passed is True
        assert report.has_warnings is True

    def test_passed_with_no_checks(self) -> None:
        """An empty report vacuously passes and has no warnings."""
        report = VerificationReport(
            status=VerificationStatus.PASSED, checks=[], summary="Nothing to check."
        )

        assert report.passed is True
        assert report.has_warnings is False

    def test_failed_info_check_does_not_fail_the_report(self) -> None:
        """Info checks are observations and never fail verification."""
        report = VerificationReport(
            status=VerificationStatus.PASSED,
            checks=[
                VerificationCheck(
                    name="note", passed=False, severity="info", message="observation"
                )
            ],
            summary="Observation only.",
        )

        assert report.passed is True
        assert report.has_warnings is False


class TestInsight:
    """The explanation contract."""

    def test_confidence_is_bounded(self) -> None:
        """Confidence stays within 0-1 here too."""
        with pytest.raises(ValidationError):
            Insight(
                headline="h", narrative="n", confidence=2.0
            )

    def test_lists_default_to_empty(self) -> None:
        """A minimal insight is valid; the lists are optional."""
        insight = Insight(headline="h", narrative="n", confidence=0.5)

        assert insight.key_findings == []
        assert insight.caveats == []
        assert insight.recommended_actions == []

    def test_example_states_caveats(self) -> None:
        """The example models the behaviour we want: state the limits."""
        example = Insight.model_config["json_schema_extra"]["example"]

        assert example["caveats"], "the example must demonstrate caveats"


class TestAgentTrace:
    """The trace the frontend renders to prove the run was agentic."""

    def test_accepts_a_full_trace(self) -> None:
        """Steps, totals and providers round-trip."""
        trace = AgentTrace(
            steps=[
                AgentStep(
                    agent_name="planner",
                    status=AgentStatus.SUCCEEDED,
                    started_at=datetime(2026, 7, 31, 10, 0, tzinfo=timezone.utc),
                    duration_ms=120.0,
                    summary="Planned.",
                    llm_provider="anthropic",
                    tokens=500,
                )
            ],
            total_duration_ms=120.0,
            total_tokens=500,
            providers_used=["anthropic"],
        )

        assert len(trace.steps) == 1
        assert trace.steps[0].status is AgentStatus.SUCCEEDED

    def test_step_without_llm_call_has_null_provider(self) -> None:
        """SQL execution makes no model call, so provider and tokens are null."""
        step = AgentStep(
            agent_name="sql_executor",
            status=AgentStatus.SUCCEEDED,
            started_at=datetime(2026, 7, 31, 10, 0, tzinfo=timezone.utc),
            duration_ms=4.0,
            summary="Ran two queries.",
        )

        assert step.llm_provider is None
        assert step.tokens is None

    def test_rejects_negative_duration(self) -> None:
        """Durations cannot be negative."""
        with pytest.raises(ValidationError):
            AgentTrace(total_duration_ms=-1.0, total_tokens=0, providers_used=[])


class TestAnalysisResponse:
    """The top-level response shape."""

    def test_unanswered_builds_a_valid_response(self) -> None:
        """A failure returns the same shape as a success, never an exception."""
        response = AnalysisResponse.unanswered(
            question="Who is the CEO?", error="Not answerable from this dataset."
        )

        assert response.answered is False
        assert response.error == "Not answerable from this dataset."
        assert response.plan is None
        assert response.insight is None
        assert response.data_asof == settings.DATA_ASOF_DATE
        assert response.trace.total_tokens == 0

    def test_unanswered_round_trips(self) -> None:
        """The failure shape survives JSON like any other."""
        response = AnalysisResponse.unanswered(question="q", error="e")

        restored = AnalysisResponse.model_validate_json(response.model_dump_json())

        assert restored == response

    def test_data_asof_matches_the_configured_anchor(self) -> None:
        """The example carries the fixed as-of date, not a real date."""
        example = AnalysisResponse.model_config["json_schema_extra"]["example"]

        assert example["data_asof"] == settings.DATA_ASOF_DATE.isoformat()

    def test_from_cache_defaults_false(self) -> None:
        """A fresh run is not cached."""
        assert AnalysisResponse.unanswered("q", "e").from_cache is False


class TestEnums:
    """The closed vocabularies agents must choose from."""

    def test_query_intents(self) -> None:
        """Every intent the planner may emit."""
        assert {intent.value for intent in QueryIntent} == {
            "aggregate",
            "ranking",
            "comparison",
            "trend",
            "diagnostic",
            "unsupported",
        }

    def test_verification_statuses(self) -> None:
        """Every verdict the verifier may emit."""
        assert {status.value for status in VerificationStatus} == {
            "passed",
            "passed_with_warnings",
            "failed",
        }

    def test_chart_types(self) -> None:
        """Every chart shape, including the explicit no-chart case."""
        assert {chart.value for chart in ChartType} == {
            "bar",
            "line",
            "grouped_bar",
            "none",
        }

    def test_agent_statuses(self) -> None:
        """Every step state the trace may show."""
        assert {status.value for status in AgentStatus} == {
            "pending",
            "running",
            "succeeded",
            "failed",
            "skipped",
        }

    def test_enums_serialize_as_strings(self) -> None:
        """String enums keep JSON readable for the frontend."""
        spec = ChartSpec(
            chart_type=ChartType.LINE, x_field="month_key", y_fields=["r"], title="t"
        )

        assert '"line"' in spec.model_dump_json()
