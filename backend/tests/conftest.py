"""Test environment setup, applied before any application module is imported.

The suite must produce the same result on a developer's laptop and in CI. It
did not: ``backend/.env`` is gitignored, so a developer running the tests had
four real provider keys in ``settings`` while CI had none. Twelve tests in
``test_api.py`` depended on that difference without saying so - they POST to
``/api/ask``, which short-circuits at the provider guard when nothing is
configured, so the stubbed orchestrator was never reached and the assertions
about its output failed only in CI.

The guard itself is correct and deliberate: it is what lets the deployment
serve cached answers with no provider at all. The bug was that tests exercising
the *analysis* path never established the precondition that path requires.

So the environment is pinned here, before ``app.config`` is imported and
therefore before ``Settings`` reads anything:

* **One dummy provider key.** Enough for ``available_providers()`` to be
  non-empty, so tests reach the code they are about. It is never used to make a
  call: every test that touches a model either stubs the orchestrator or
  injects its own client. Tests that specifically want the no-provider state
  still clear the keys themselves.
* **The other three blank.** Otherwise a developer with three real keys and CI
  with none would still be running different code paths.
* **The startup probe off.** It would otherwise fire on every ``TestClient``
  lifespan and make real network calls to four vendors. In practice the task is
  usually cancelled at shutdown before it runs, which is worse than it firing
  reliably: a slower machine would let it through, the probe would fail against
  a dummy key, and the circuit breaker would open and change what
  ``/api/health`` reports. A test suite that is one scheduling decision away
  from a different answer is not a test suite.

Environment variables are set rather than ``settings`` attributes patched,
because pydantic-settings gives real environment variables precedence over the
``.env`` file - which is exactly the override needed here.
"""

from __future__ import annotations

import os
from typing import Final

# The single provider tests are allowed to see. Shaped like a key so that any
# accidental leak into a response body is caught by the key-scanning tests,
# and obviously fake so it can never be mistaken for a real credential.
DUMMY_PROVIDER_KEY: Final[str] = "sk-ant-test-key-not-a-real-credential"

# Set before app.config is imported anywhere; Settings reads the environment at
# construction, and pytest imports this file first.
os.environ["ANTHROPIC_API_KEY"] = DUMMY_PROVIDER_KEY
os.environ["OPENAI_API_KEY"] = ""
os.environ["GEMINI_API_KEY"] = ""
os.environ["GROK_API_KEY"] = ""
os.environ["PROVIDER_PROBE_ON_STARTUP"] = "false"
