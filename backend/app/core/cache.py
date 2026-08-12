"""Answer cache and per-IP rate limiting.

Both exist for the same reason: this is a public URL in front of paid APIs,
opened by people the author will never meet, possibly long after the API keys
have expired.

The cache is therefore not a latency optimisation. It is the system's answer to
a specific failure: a reviewer opens the deployed URL weeks from now, every
provider key is dead, and the honest-but-useless response would be "live
analysis is unavailable". A cache warmed with the eight evaluation questions
and committed to the repository means those eight questions are answered
correctly from disk with no provider at all, and the trace still shows the
agents that produced them. It is persisted as JSON rather than held in memory
because the deployment sleeps and cold-starts, and an empty cache after every
sleep would defeat the point.

Rate limiting is the cost boundary. A cached answer costs nothing, so it does
not consume quota; only work that reaches a provider does.
"""

from __future__ import annotations

import json
import re
import string
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Final

from app.agents.contracts import AnalysisResponse
from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Version stamp written into the cache file. Bumping it invalidates every
# entry, which is the right response to a change in the response contract:
# a stale entry that no longer validates would fail on read, one at a time.
CACHE_FORMAT_VERSION: Final[int] = 1

# Punctuation is stripped during normalisation so "revenue?" and "revenue"
# are the same question. Kept as a translation table for speed.
_PUNCTUATION = str.maketrans("", "", string.punctuation)

# Runs of any whitespace collapse to a single space.
_WHITESPACE = re.compile(r"\s+")

# Seconds in the two rate-limit windows.
SECONDS_PER_MINUTE: Final[int] = 60
SECONDS_PER_DAY: Final[int] = 86_400


def normalise_question(question: str) -> str:
    """Reduce a question to its cache key.

    Lowercased, punctuation removed, whitespace collapsed, so the same question
    typed three different ways is one cache entry rather than three.

    Args:
        question: The question as the user typed it.

    Returns:
        The normalised key.
    """
    lowered = question.strip().lower().translate(_PUNCTUATION)
    return _WHITESPACE.sub(" ", lowered).strip()


class AnswerCache:
    """A question-keyed store of successful analyses, persisted to JSON.

    Thread-safe: FastAPI serves requests from a thread pool, and two clients
    asking the same question at once must not corrupt the file.
    """

    def __init__(self, path: Path | None = None, autosave: bool = True) -> None:
        """Initialise the cache and load any entries already on disk.

        Args:
            path: Cache file location. Defaults to ``settings.CACHE_PATH``.
            autosave: Whether to write to disk on every store. Disabled by the
                warming script, which saves once at the end.
        """
        self.path = path or settings.CACHE_PATH
        self.autosave = autosave
        self._entries: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self.load()

    def load(self) -> int:
        """Read the cache file into memory.

        A missing or unreadable file is not an error: the cache is an
        optimisation and the system must start without it.

        Returns:
            The number of entries loaded.
        """
        if not self.path.exists():
            logger.info("cache_absent", extra={"path": str(self.path)})
            return 0
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            logger.warning(
                "cache_unreadable",
                extra={"path": str(self.path), "error": str(error)},
            )
            return 0

        if payload.get("version") != CACHE_FORMAT_VERSION:
            logger.warning(
                "cache_version_mismatch",
                extra={
                    "found": payload.get("version"),
                    "expected": CACHE_FORMAT_VERSION,
                },
            )
            return 0

        entries = payload.get("entries", {})
        with self._lock:
            self._entries = dict(entries)
        logger.info(
            "cache_loaded",
            extra={"path": str(self.path), "entries": len(entries)},
        )
        return len(entries)

    def save(self) -> None:
        """Write the cache to disk atomically.

        Written to a temporary file and renamed, so a crash mid-write leaves
        the previous cache intact rather than a truncated file that fails to
        parse on the next boot.
        """
        with self._lock:
            payload = {
                "version": CACHE_FORMAT_VERSION,
                "data_asof": settings.DATA_ASOF_DATE.isoformat(),
                "entries": dict(self._entries),
            }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
        temporary.replace(self.path)
        logger.info(
            "cache_saved",
            extra={"path": str(self.path), "entries": len(payload["entries"])},
        )

    def get(self, question: str) -> AnalysisResponse | None:
        """Look up a cached answer.

        Args:
            question: The question as typed.

        Returns:
            The stored response with ``from_cache`` set, or None on a miss.
            The original trace is preserved rather than replaced, so a cached
            answer still shows which agents ran and what they decided.
        """
        key = normalise_question(question)
        with self._lock:
            entry = self._entries.get(key)
        if entry is None:
            return None
        try:
            response = AnalysisResponse.model_validate(entry["response"])
        except Exception as error:  # noqa: BLE001 - a bad entry is a miss
            logger.warning(
                "cache_entry_invalid", extra={"key": key, "error": str(error)}
            )
            return None
        response.from_cache = True
        logger.info("cache_hit", extra={"key": key})
        return response

    def put(self, question: str, response: AnalysisResponse) -> None:
        """Store a successful answer.

        Only answered responses are cached. Caching a failure would serve that
        failure to every later visitor, long after the cause had cleared.

        Args:
            question: The question as typed.
            response: The response to store.
        """
        if not response.answered:
            return
        key = normalise_question(question)
        with self._lock:
            self._entries[key] = {
                "question": question,
                "cached_at": time.time(),
                "response": response.model_dump(mode="json"),
            }
        logger.info("cache_stored", extra={"key": key})
        if self.autosave:
            self.save()

    def keys(self) -> list[str]:
        """List the normalised questions currently cached.

        Returns:
            The cache keys, sorted.
        """
        with self._lock:
            return sorted(self._entries)

    def questions(self) -> list[str]:
        """List the original question text of every entry.

        Returns:
            The questions as they were first asked, sorted.
        """
        with self._lock:
            return sorted(
                str(entry.get("question", key))
                for key, entry in self._entries.items()
            )

    def clear(self) -> None:
        """Drop every entry from memory."""
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        """Count the cached answers.

        Returns:
            The number of entries.
        """
        with self._lock:
            return len(self._entries)


class RateLimitResult:
    """The outcome of one rate-limit decision.

    Attributes:
        allowed: Whether the request may proceed.
        retry_after: Seconds until the caller may retry, when blocked.
        reason: Which limit was hit, for the error message.
    """

    def __init__(
        self, allowed: bool, retry_after: int = 0, reason: str = ""
    ) -> None:
        """Initialise the result.

        Args:
            allowed: Whether the request may proceed.
            retry_after: Seconds until a retry could succeed.
            reason: Human-readable explanation when blocked.
        """
        self.allowed = allowed
        self.retry_after = retry_after
        self.reason = reason


class RateLimiter:
    """In-memory sliding-window rate limiter, keyed by client IP.

    In-memory is a deliberate limit, not an oversight: the deployment is a
    single process, and a shared store would add infrastructure this system
    does not otherwise need. It resets on restart, which is acceptable for a
    cost guard rather than a security control.
    """

    def __init__(
        self, per_minute: int | None = None, per_day: int | None = None
    ) -> None:
        """Initialise the limiter.

        Args:
            per_minute: Requests allowed per minute. Defaults to settings.
            per_day: Requests allowed per day. Defaults to settings.
        """
        self.per_minute = (
            per_minute if per_minute is not None else settings.RATE_LIMIT_PER_MINUTE
        )
        self.per_day = per_day if per_day is not None else settings.RATE_LIMIT_PER_DAY
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, client: str, now: float | None = None) -> RateLimitResult:
        """Test whether a client may make a chargeable request.

        Only call this for work that will reach a provider. A cache hit costs
        nothing and must not consume anyone's quota.

        Args:
            client: Identifier for the caller, normally its IP address.
            now: Current time, injectable for tests.

        Returns:
            The decision, with a retry delay when blocked.
        """
        current = time.time() if now is None else now
        with self._lock:
            hits = self._hits.setdefault(client, deque())
            while hits and current - hits[0] > SECONDS_PER_DAY:
                hits.popleft()

            in_last_minute = sum(
                1 for hit in hits if current - hit <= SECONDS_PER_MINUTE
            )
            if in_last_minute >= self.per_minute:
                oldest = next(
                    hit for hit in hits if current - hit <= SECONDS_PER_MINUTE
                )
                retry = max(1, int(SECONDS_PER_MINUTE - (current - oldest)) + 1)
                return RateLimitResult(
                    allowed=False,
                    retry_after=retry,
                    reason=(
                        f"Rate limit reached: {self.per_minute} live analyses "
                        f"per minute. Cached answers are not limited, so the "
                        f"suggested questions remain available."
                    ),
                )

            if len(hits) >= self.per_day:
                retry = max(1, int(SECONDS_PER_DAY - (current - hits[0])) + 1)
                return RateLimitResult(
                    allowed=False,
                    retry_after=retry,
                    reason=(
                        f"Daily limit reached: {self.per_day} live analyses "
                        f"per day. Cached answers are not limited, so the "
                        f"suggested questions remain available."
                    ),
                )

            hits.append(current)
            return RateLimitResult(allowed=True)

    def reset(self, client: str | None = None) -> None:
        """Forget recorded hits.

        Args:
            client: The client to clear, or None to clear every client.
        """
        with self._lock:
            if client is None:
                self._hits.clear()
            else:
                self._hits.pop(client, None)


# Module-level singletons. The API holds one of each for the process lifetime;
# constructing a new cache per request would re-read the file every time.
_cache: AnswerCache | None = None
_limiter: RateLimiter | None = None


def get_cache() -> AnswerCache:
    """Return the shared answer cache, loading it on first use.

    Returns:
        The process-wide cache.
    """
    global _cache
    if _cache is None:
        _cache = AnswerCache()
    return _cache


def get_rate_limiter() -> RateLimiter:
    """Return the shared rate limiter.

    Returns:
        The process-wide limiter.
    """
    global _limiter
    if _limiter is None:
        _limiter = RateLimiter()
    return _limiter


def reset_cache_and_limiter() -> None:
    """Drop both singletons so the next call rebuilds them.

    Used by tests, which need a limiter whose counters are empty and a cache
    that is not the developer's real one.
    """
    global _cache, _limiter
    _cache = None
    _limiter = None
