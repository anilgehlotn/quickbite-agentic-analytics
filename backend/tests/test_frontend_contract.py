"""The TypeScript mirrors in the frontend must match the Pydantic contracts.

Nothing else catches this. The JSON crossing the wire is unchecked at the
boundary, so a field added on one side and forgotten on the other produces no
error anywhere: TypeScript compiles, the backend serialises happily, and the
client silently reads ``undefined``. That is precisely how Module 9 shipped an
``AnalysisPlan.clarification`` field that existed in Python and not in
``lib/types.ts``.

The comparison is by field *name*, not by type. Names are what the wire format
actually carries, and checking them catches the failure that matters - a field
that is missing entirely - without this test becoming a second, worse
implementation of the type system.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import BaseModel

from app.agents.contracts import (
    AgentStep,
    AgentTrace,
    AnalysisPlan,
    AnalysisResponse,
    ChartSpec,
    Clarification,
    Insight,
    QueryResult,
    SubQuery,
    TimeWindow,
    VerificationCheck,
    VerificationReport,
)
from app.api.routes import HealthResponse, ProviderHealth, QuestionSuggestion

#: Path to the frontend's hand-maintained mirrors.
TYPES_PATH: Path = Path(__file__).resolve().parents[2] / "frontend" / "lib" / "types.ts"

#: Every contract that crosses the wire, paired with its interface name.
MIRRORED_MODELS: list[tuple[str, type[BaseModel]]] = [
    ("AnalysisResponse", AnalysisResponse),
    ("AnalysisPlan", AnalysisPlan),
    ("QueryResult", QueryResult),
    ("Insight", Insight),
    ("ChartSpec", ChartSpec),
    ("AgentTrace", AgentTrace),
    ("AgentStep", AgentStep),
    ("VerificationReport", VerificationReport),
    ("VerificationCheck", VerificationCheck),
    ("TimeWindow", TimeWindow),
    ("SubQuery", SubQuery),
    ("Clarification", Clarification),
    ("HealthResponse", HealthResponse),
    ("ProviderHealth", ProviderHealth),
    ("QuestionSuggestion", QuestionSuggestion),
]


def typescript_source() -> str:
    """Read the frontend's type definitions.

    Returns:
        The file contents.
    """
    return TYPES_PATH.read_text(encoding="utf-8")


def interface_fields(source: str, name: str) -> set[str] | None:
    """Extract the field names declared by one TypeScript interface.

    Args:
        source: The contents of types.ts.
        name: The interface name.

    Returns:
        The declared field names, or None when the interface is absent.
    """
    match = re.search(rf"export interface {name} \{{(.*?)\n\}}", source, re.S)
    if match is None:
        return None
    # Field declarations only: a line like "  foo: string;" or "  foo?: X".
    # Comment bodies cannot match because the pattern is anchored to a line
    # start followed by an identifier and a colon.
    return set(re.findall(r"^\s*([a-z_][A-Za-z0-9_]*)\??:", match.group(1), re.M))


def test_the_types_file_exists() -> None:
    """A missing mirror would make every other check vacuously pass."""
    assert TYPES_PATH.is_file(), f"expected {TYPES_PATH}"


@pytest.mark.parametrize(
    ("name", "model"), MIRRORED_MODELS, ids=[name for name, _ in MIRRORED_MODELS]
)
def test_frontend_mirrors_every_backend_field(name: str, model: type[BaseModel]) -> None:
    """Each contract's fields appear in its TypeScript interface, and vice versa.

    Args:
        name: The interface name in types.ts.
        model: The Pydantic model it mirrors.
    """
    fields = interface_fields(typescript_source(), name)
    assert fields is not None, f"types.ts has no interface {name}"

    backend = set(model.model_fields)
    missing = backend - fields
    extra = fields - backend

    assert not missing, (
        f"{name}: backend fields missing from frontend/lib/types.ts: "
        f"{sorted(missing)}. A client reading these gets undefined, silently."
    )
    assert not extra, (
        f"{name}: frontend declares fields the backend does not send: "
        f"{sorted(extra)}. Either the backend dropped them or the mirror is stale."
    )


def test_query_intent_union_matches_the_enum() -> None:
    """A new intent the client cannot represent would break rendering."""
    from app.agents.contracts import QueryIntent

    match = re.search(r"export type QueryIntent =(.*?);", typescript_source(), re.S)
    assert match is not None
    declared = set(re.findall(r'"([a-z_]+)"', match.group(1)))
    assert declared == {intent.value for intent in QueryIntent}


def test_response_status_union_matches_the_enum() -> None:
    """The four reply kinds must be renderable by the client."""
    from app.agents.contracts import ResponseStatus

    match = re.search(r"export type ResponseStatus =(.*?);", typescript_source(), re.S)
    assert match is not None
    declared = set(re.findall(r'"([a-z_]+)"', match.group(1)))
    assert declared == {status.value for status in ResponseStatus}
