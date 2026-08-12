"""Prove the system answers its eight questions with no provider configured.

This is the module's central claim, so it is asserted by a script that runs in
CI rather than by a paragraph in the README. It clears every API key from the
environment, starts the application exactly as a fresh process would, and
checks that each canonical question returns a complete answer — headline,
narrative, executed SQL, verification report and full agent trace — served from
the committed cache without a single network call.

It then asks something uncached and checks that the refusal is a well-formed
answer with helpful content rather than an error shape.

Usage::

    python scripts/check_offline.py
    python scripts/check_offline.py --verbose
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Final

_ROOT = Path(__file__).resolve().parents[1]
_BACKEND = _ROOT / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

# Cleared before app import so the settings singleton is built without them.
# An empty string is treated as unconfigured by Settings.available_providers,
# and setting the variables explicitly also overrides anything in a local .env,
# which is what makes this a real test rather than a hopeful one.
_PROVIDER_KEYS: Final[tuple[str, ...]] = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GROK_API_KEY",
)
for _key in _PROVIDER_KEYS:
    os.environ[_key] = ""

from fastapi.testclient import TestClient  # noqa: E402

from app.agents.orchestrator import CANONICAL_QUESTIONS  # noqa: E402
from app.config import settings  # noqa: E402
from app.main import app  # noqa: E402

CONSOLE_WIDTH: Final[int] = 92

# A question deliberately absent from the cache, used to exercise the refusal.
UNCACHED_QUESTION: Final[str] = (
    "What was the revenue split by store format in February 2026?"
)


def check_answer(payload: dict[str, Any]) -> list[str]:
    """Verify one cached answer is complete rather than merely present.

    Args:
        payload: The parsed response body.

    Returns:
        Problems found. Empty when the answer is complete.
    """
    problems: list[str] = []

    if payload.get("answered") is not True:
        problems.append("answered is not true")
    if payload.get("from_cache") is not True:
        problems.append("from_cache is not true (a provider may have been used)")

    insight = payload.get("insight")
    if not isinstance(insight, dict) or not insight.get("headline"):
        problems.append("no insight headline")
    elif not insight.get("narrative"):
        problems.append("no insight narrative")

    verification = payload.get("verification")
    if not isinstance(verification, dict):
        problems.append("no verification report")
    else:
        if verification.get("status") == "failed":
            problems.append("verification status is failed")
        if not verification.get("checks"):
            problems.append("verification report has no checks")

    results = payload.get("query_results")
    if not isinstance(results, list) or len(results) == 0:
        problems.append("no query results")
    elif not any(isinstance(r, dict) and r.get("sql") for r in results):
        problems.append("no executed SQL")

    trace = payload.get("trace")
    if not isinstance(trace, dict) or not trace.get("steps"):
        problems.append("no agent trace")
    else:
        agents = {str(step.get("agent_name", "")) for step in trace["steps"]}
        for expected in ("planner", "sql_analyst", "verifier", "insight"):
            if not any(name.startswith(expected) for name in agents):
                problems.append(f"trace is missing the {expected} step")

    return problems


def main() -> int:
    """Run the offline checks and report the outcome.

    Returns:
        Process exit code: 0 when every check passed.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each answer's headline.",
    )
    arguments = parser.parse_args()

    print("=" * CONSOLE_WIDTH)
    print("OFFLINE CHECK — no API keys configured")
    print("=" * CONSOLE_WIDTH)
    print(f"providers available : {settings.available_providers() or 'none'}")
    print(f"cache file          : {settings.CACHE_PATH}")
    print(f"cache exists        : {settings.CACHE_PATH.exists()}")

    if settings.available_providers():
        print("\nFAILED: provider keys are still configured; the check is void.")
        return 1

    failures: list[str] = []

    with TestClient(app) as client:
        health = client.get("/api/health").json()
        print(f"health mode         : {health['mode']} (degraded={health['degraded']})")
        print(f"cached answers      : {health['cached_answers']}")
        print(f"data as of          : {health['data_asof']}")
        print()

        if health["mode"] != "cache_only":
            failures.append(
                f"health reports mode={health['mode']}, expected cache_only"
            )

        print(f"{'id':<5}{'status':<9}{'rows':>6}{'checks':>8}{'steps':>7}  headline")
        print("-" * CONSOLE_WIDTH)

        for entry in CANONICAL_QUESTIONS:
            response = client.post(
                "/api/ask", json={"question": entry["question"]}
            )
            if response.status_code != 200:
                failures.append(f"{entry['id']}: HTTP {response.status_code}")
                print(f"{entry['id']:<5}{'HTTP ' + str(response.status_code):<9}")
                continue

            payload = response.json()
            problems = check_answer(payload)
            if problems:
                failures.extend(f"{entry['id']}: {problem}" for problem in problems)

            rows = sum(
                int(result.get("row_count", 0))
                for result in payload.get("query_results", [])
            )
            checks = len(payload.get("verification", {}).get("checks", []))
            steps = len(payload.get("trace", {}).get("steps", []))
            headline = (payload.get("insight") or {}).get("headline", "")
            print(
                f"{entry['id']:<5}{'ok' if not problems else 'PROBLEM':<9}"
                f"{rows:>6}{checks:>8}{steps:>7}  {headline[:52]}"
            )
            if arguments.verbose and headline:
                print(f"       {headline}")

        # The refusal path: a question nobody warmed, with nothing to call.
        print()
        print("uncached question, no provider:")
        refusal = client.post("/api/ask", json={"question": UNCACHED_QUESTION})
        print(f"  HTTP {refusal.status_code}")
        if refusal.status_code != 200:
            failures.append(
                f"uncached question returned HTTP {refusal.status_code}, "
                f"expected a well-formed 200"
            )
        else:
            body = refusal.json()
            insight = body.get("insight") or {}
            print(f"  answered   : {body.get('answered')}")
            print(f"  headline   : {insight.get('headline', '')}")
            print(f"  offers     : {len(insight.get('key_findings', []))} questions")
            if body.get("answered") is not False:
                failures.append("uncached question did not report answered=false")
            if not insight.get("headline"):
                failures.append("uncached question returned no explanation")
            if len(insight.get("key_findings", [])) < len(CANONICAL_QUESTIONS):
                failures.append(
                    "uncached question did not list the available questions"
                )

    print()
    print("=" * CONSOLE_WIDTH)
    if failures:
        print(f"FAILED — {len(failures)} problem(s):")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(
        f"PASSED — all {len(CANONICAL_QUESTIONS)} canonical questions answered "
        f"completely from cache with no provider configured."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
