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

The same reasoning covers the second mirror in this file: the frontend's static
copy of the eight canonical questions in ``lib/questions.ts``. It exists so the
page can render its suggestion cards on first paint instead of waiting for a
backend that may be asleep, and the cost of that speed is a hand-maintained
duplicate. Drift there is worse than a missing type: a card would send text the
backend has no cached answer for, so a question advertised as instant would
quietly become a live analysis - or, with no provider reachable, no answer at
all.
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

#: Root of the frontend package.
FRONTEND: Path = Path(__file__).resolve().parents[2] / "frontend"

#: Path to the frontend's hand-maintained mirrors.
TYPES_PATH: Path = FRONTEND / "lib" / "types.ts"

#: Path to the frontend's build-time copy of the canonical questions.
QUESTIONS_PATH: Path = FRONTEND / "lib" / "questions.ts"

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


def static_questions() -> list[dict[str, str]]:
    """Parse the frontend's build-time question list.

    Reads the object literals out of ``lib/questions.ts`` in file order. The
    pattern requires the three keys in the order the file declares them, which
    is stricter than necessary but keeps the parse unambiguous - and Prettier
    never reorders object keys, so it is stable.

    Returns:
        One dict per question, with ``id``, ``question`` and ``label``.
    """
    source = QUESTIONS_PATH.read_text(encoding="utf-8")
    # Only the entries inside the exported array, so the interface declaration
    # and the docstring examples above it cannot be mistaken for data.
    array = re.search(
        r"export const CANONICAL_QUESTIONS[^=]*=\s*\[(.*?)\n\] as const;",
        source,
        re.S,
    )
    assert array is not None, f"no CANONICAL_QUESTIONS array in {QUESTIONS_PATH}"

    pattern = re.compile(
        r'id:\s*"([^"]*)",\s*question:\s*"([^"]*)",\s*label:\s*"([^"]*)",',
        re.S,
    )
    return [
        {"id": found.group(1), "question": found.group(2), "label": found.group(3)}
        for found in pattern.finditer(array.group(1))
    ]


def test_the_static_question_file_exists() -> None:
    """A missing file would make the drift check vacuous."""
    assert QUESTIONS_PATH.is_file(), f"expected {QUESTIONS_PATH}"


def test_static_questions_parse() -> None:
    """The parse must find eight entries, or the comparison proves nothing.

    Guards the regex itself: a reformat that broke the pattern would otherwise
    turn this file's real check into a comparison of two empty lists.
    """
    from app.agents.orchestrator import CANONICAL_QUESTIONS

    parsed = static_questions()
    assert len(parsed) == len(CANONICAL_QUESTIONS) == 8, (
        f"parsed {len(parsed)} questions from {QUESTIONS_PATH.name}; the file "
        f"may have been reformatted in a way the parser does not handle."
    )


def test_frontend_static_questions_match_the_backend() -> None:
    """The card a reviewer clicks must be the question the backend warmed.

    Compared in order and in full - id, exact text and label. The text is the
    part that matters most: it is sent verbatim to ``/api/ask`` and normalised
    into a cache key, so a single changed word turns a guaranteed instant
    answer into a live analysis that no provider may be available to run.
    """
    from app.agents.orchestrator import CANONICAL_QUESTIONS

    expected = [
        {"id": entry["id"], "question": entry["question"], "label": entry["label"]}
        for entry in CANONICAL_QUESTIONS
    ]
    assert static_questions() == expected, (
        f"{QUESTIONS_PATH} has drifted from CANONICAL_QUESTIONS in "
        f"app/agents/orchestrator.py. The frontend renders its suggestion "
        f"cards from that file before the backend answers, so the two must "
        f"agree exactly."
    )


def test_static_questions_are_the_warmed_ones() -> None:
    """Every card the page renders on first paint is answerable from cache.

    The point of rendering the cards immediately is that clicking one always
    works, including with no provider reachable. That holds only while the
    committed cache actually contains them.
    """
    from app.core.cache import get_cache, normalise_question

    cached = set(get_cache().keys())
    if not cached:
        pytest.skip("no committed answer cache in this environment")

    missing = [
        entry["question"]
        for entry in static_questions()
        if normalise_question(entry["question"]) not in cached
    ]
    assert not missing, (
        f"questions rendered on first paint but not warmed in the cache: "
        f"{missing}. A reviewer clicking these with no provider configured "
        f"gets the unavailable message instead of an answer."
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
