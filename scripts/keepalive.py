"""Keep the deployed backend awake by polling its health endpoint.

The hosting tier suspends a service after about fifteen minutes idle, and
waking it again takes up to fifty seconds. A reviewer who opens the link once
and waits fifty seconds for the first click has already formed an opinion, so
the practical fix is to make sure the service is never actually idle.

This script is the local version, useful while demonstrating the system live.
The version that matters in normal operation is
``.github/workflows/keepalive.yml``, which does the same thing on GitHub's
infrastructure and therefore keeps working when this machine is closed.

Usage::

    python scripts/keepalive.py --url https://your-service.onrender.com
    python scripts/keepalive.py --url http://localhost:8000 --interval 60 --once
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Final

# Default poll interval. Comfortably inside the fifteen-minute idle window,
# with room for a missed run.
DEFAULT_INTERVAL_SECONDS: Final[int] = 600

# Per-request timeout. Generous enough to cover a cold start rather than
# reporting a failure for the very condition this script exists to prevent.
REQUEST_TIMEOUT_SECONDS: Final[int] = 90

DEFAULT_URL: Final[str] = "http://localhost:8000"


def timestamp() -> str:
    """Return the current UTC time for a log line.

    Returns:
        An ISO 8601 timestamp to the second.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ping(base_url: str) -> tuple[bool, str]:
    """Call the health endpoint once.

    Args:
        base_url: Backend base URL, without a trailing slash.

    Returns:
        Whether the call succeeded, and a one-line description of the result.
    """
    url = f"{base_url.rstrip('/')}/api/health"
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            payload: dict[str, Any] = json.loads(response.read())
    except urllib.error.HTTPError as error:
        return False, f"HTTP {error.code}"
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as error:
        return False, f"{type(error).__name__}: {error}"

    elapsed = time.perf_counter() - started
    return True, (
        f"{payload.get('status', '?')} / {payload.get('mode', '?')} · "
        f"{payload.get('cached_answers', 0)} cached · {elapsed:.1f}s"
    )


def main() -> int:
    """Poll the health endpoint until interrupted.

    Returns:
        Process exit code: 0 normally, 1 when a single ``--once`` ping failed.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"Backend base URL. Default: {DEFAULT_URL}",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_INTERVAL_SECONDS,
        help=(
            f"Seconds between pings. Default: {DEFAULT_INTERVAL_SECONDS} "
            f"(the idle window is about 900)."
        ),
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Ping once and exit, for use in a scheduler.",
    )
    arguments = parser.parse_args()

    print(f"pinging {arguments.url} every {arguments.interval}s (Ctrl-C to stop)")
    while True:
        ok, detail = ping(arguments.url)
        print(f"{timestamp()}  {'ok  ' if ok else 'FAIL'}  {detail}", flush=True)
        if arguments.once:
            return 0 if ok else 1
        try:
            time.sleep(arguments.interval)
        except KeyboardInterrupt:
            print("\nstopped")
            return 0


if __name__ == "__main__":
    sys.exit(main())
