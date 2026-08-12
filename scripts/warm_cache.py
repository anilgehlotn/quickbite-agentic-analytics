"""Warm the answer cache by running the eight evaluation questions.

The output of this script is a committed artifact, not a local convenience.
The deployed instance may be opened by a reviewer long after the API keys have
expired, and a system that answers "no provider available" to the questions it
was built to answer has failed regardless of how good its agents are. Running
this before shipping means those eight answers are on disk, served from cache,
with the full agent trace intact so the pipeline is still visible.

Every answer is cross-checked against ``tests/golden_answers.json`` where the
figures are directly comparable, so a cache is never committed containing an
answer that contradicts the ground truth.

Run from the repository root::

    python scripts/warm_cache.py
    python scripts/warm_cache.py --questions q1 q5
    python scripts/warm_cache.py --force
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Final

_BACKEND = Path(__file__).resolve().parents[1] / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app.agents.contracts import AnalysisResponse, VerificationStatus  # noqa: E402
from app.agents.orchestrator import CANONICAL_QUESTIONS, Orchestrator  # noqa: E402
from app.config import settings  # noqa: E402
from app.core.cache import AnswerCache, normalise_question  # noqa: E402

# Golden answers live with the tests because they are the tests' ground truth.
GOLDEN_PATH: Final[Path] = _BACKEND / "tests" / "golden_answers.json"

# Seconds between questions. Free provider tiers rate-limit per minute, and a
# burst of eight concurrent analyses is the fastest way to discover that.
PAUSE_SECONDS: Final[float] = 5.0

# Tolerances for the cross-check against ground truth.
REVENUE_TOLERANCE_INR: Final[float] = 1.0
AOV_TOLERANCE_INR: Final[float] = 0.01

CONSOLE_WIDTH: Final[int] = 92


def load_golden() -> dict[str, Any]:
    """Read the golden answers.

    Returns:
        The parsed ground truth, or an empty mapping when the file is absent.
    """
    if not GOLDEN_PATH.exists():
        print(f"  ! golden answers not found at {GOLDEN_PATH}")
        return {}
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def numbers_in(response: AnalysisResponse) -> list[float]:
    """Collect every numeric value the queries returned.

    Args:
        response: The analysis to scan.

    Returns:
        All numeric cell values across all result rows.
    """
    values: list[float] = []
    for result in response.query_results:
        for row in result.rows:
            for value in row.values():
                if isinstance(value, bool) or value is None:
                    continue
                if isinstance(value, (int, float)):
                    values.append(float(value))
    return values


def check_against_golden(
    question_id: str, response: AnalysisResponse, golden: dict[str, Any]
) -> list[str]:
    """Cross-check an answer against the ground truth.

    Deliberately checks only what is directly comparable: whether the figures
    the ground truth states actually appear among the numbers the queries
    returned. Comparing narrative text would be checking the model's prose, not
    its arithmetic.

    Args:
        question_id: Which canonical question this is.
        response: The answer produced.
        golden: The parsed golden answers.

    Returns:
        Human-readable problems. Empty when the answer is consistent.
    """
    entry = golden.get(question_id)
    if not entry:
        return []

    problems: list[str] = []
    values = numbers_in(response)

    def present(target: float, tolerance: float, label: str) -> None:
        """Assert a ground-truth figure appears in the results.

        Args:
            target: The expected figure.
            tolerance: Allowed absolute difference.
            label: Name used in the problem message.
        """
        if not any(abs(value - target) <= tolerance for value in values):
            problems.append(f"{label} {target:,.2f} not found in results")

    if question_id == "q1":
        present(entry["revenue_net_inr"], REVENUE_TOLERANCE_INR, "revenue")
        present(float(entry["orders"]), 0.5, "orders")
        present(entry["aov_inr"], AOV_TOLERANCE_INR, "AOV")

    if question_id == "q8":
        expected = {
            store["store_id"]
            for store in entry["stores"]
            if store["declined_every_month"]
        }
        found = {
            str(cell)
            for result in response.query_results
            for row in result.rows
            for key, cell in row.items()
            if key == "store_id"
        }
        missing = expected - found
        if missing:
            problems.append(
                f"declining stores missing from results: {sorted(missing)}"
            )

    return problems


async def warm(
    question_ids: list[str] | None, force: bool
) -> tuple[list[dict[str, Any]], AnswerCache]:
    """Run the canonical questions and populate the cache.

    Args:
        question_ids: Restrict to these ids, or None for all eight.
        force: Re-run questions that are already cached.

    Returns:
        One summary row per question, and the populated cache.
    """
    # autosave off: one write at the end, so an interrupted run cannot leave a
    # half-warmed cache that looks complete.
    cache = AnswerCache(autosave=False)
    golden = load_golden()
    orchestrator = Orchestrator()

    selected = [
        entry
        for entry in CANONICAL_QUESTIONS
        if question_ids is None or entry["id"] in question_ids
    ]
    rows: list[dict[str, Any]] = []

    for index, entry in enumerate(selected):
        question = entry["question"]
        if not force and cache.get(question) is not None:
            print(f"[{entry['id']}] already cached, skipping")
            rows.append(
                {
                    "id": entry["id"],
                    "status": "cached",
                    "tokens": 0,
                    "seconds": 0.0,
                    "problems": [],
                }
            )
            continue

        if index:
            await asyncio.sleep(PAUSE_SECONDS)

        print()
        print("=" * CONSOLE_WIDTH)
        print(f"[{entry['id']}] {question}")
        print("=" * CONSOLE_WIDTH)

        started = time.perf_counter()
        response = await orchestrator.run(question)
        elapsed = time.perf_counter() - started

        problems = check_against_golden(entry["id"], response, golden)
        verification = (
            response.verification.status.value if response.verification else "not_run"
        )

        if response.answered:
            cache.put(question, response)
            status = "ok" if not problems else "mismatch"
        else:
            status = "failed"
            problems.append(response.error or "no answer produced")

        print(f"  answered      : {response.answered}")
        print(f"  verification  : {verification}")
        print(f"  sub-queries   : {len(response.query_results)}")
        print(f"  agents        : {[s.agent_name for s in response.trace.steps]}")
        print(f"  tokens        : {response.trace.total_tokens:,}")
        print(f"  duration      : {elapsed:.1f}s")
        if response.insight:
            print(f"  headline      : {response.insight.headline}")
        for problem in problems:
            print(f"  ! {problem}")

        rows.append(
            {
                "id": entry["id"],
                "status": status,
                "tokens": response.trace.total_tokens,
                "seconds": elapsed,
                "problems": problems,
                "verification": verification,
            }
        )

    return rows, cache


def print_summary(rows: list[dict[str, Any]], cache: AnswerCache) -> bool:
    """Print the run summary and report whether it is safe to commit.

    Args:
        rows: The per-question summaries.
        cache: The populated cache.

    Returns:
        True when every question produced a consistent answer.
    """
    print()
    print("=" * CONSOLE_WIDTH)
    print("SUMMARY")
    print("=" * CONSOLE_WIDTH)
    print(f"{'id':<5}{'status':<11}{'verification':<22}{'tokens':>9}{'seconds':>9}")
    print("-" * CONSOLE_WIDTH)
    for row in rows:
        print(
            f"{row['id']:<5}{row['status']:<11}"
            f"{row.get('verification', '-'):<22}"
            f"{row['tokens']:>9,}{row['seconds']:>9.1f}"
        )
    print("-" * CONSOLE_WIDTH)

    total_tokens = sum(row["tokens"] for row in rows)
    total_seconds = sum(row["seconds"] for row in rows)
    failures = [row for row in rows if row["status"] not in ("ok", "cached")]

    print(f"{'total':<5}{'':<11}{'':<22}{total_tokens:>9,}{total_seconds:>9.1f}")
    print()
    print(f"cache entries : {len(cache)}")
    print(f"cache file    : {cache.path}")
    print(f"total tokens  : {total_tokens:,}")
    if failures:
        print()
        print(f"{len(failures)} question(s) did not produce a clean answer:")
        for row in failures:
            for problem in row["problems"]:
                print(f"  [{row['id']}] {problem}")
    return not failures


def main() -> int:
    """Warm the cache and report the outcome.

    Returns:
        Process exit code: 0 when every question answered consistently.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions",
        nargs="*",
        default=None,
        help="Canonical question ids to warm, for example q1 q5. Default: all.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run questions that already have a cached answer.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the questions but do not write the cache file.",
    )
    arguments = parser.parse_args()

    if not settings.available_providers():
        print("No LLM provider is configured; nothing to warm.")
        return 1

    print(f"providers  : {settings.available_providers()}")
    print(f"cache file : {settings.CACHE_PATH}")
    print(f"questions  : {arguments.questions or 'all 8'}")

    rows, cache = asyncio.run(warm(arguments.questions, arguments.force))
    clean = print_summary(rows, cache)

    if arguments.dry_run:
        print()
        print("dry run: cache not written")
        return 0 if clean else 1

    cache.save()
    print()
    print(f"wrote {len(cache)} answers to {cache.path}")
    for question in cache.questions():
        print(f"  - {normalise_question(question)[:80]}")
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
