"""Tests for provider resilience: the circuit breaker, exclusion and probing.

Every test runs against an ``httpx.MockTransport``, so nothing here makes a
network call or costs money. Providers are made to fail on purpose, because a
failover path that has never failed is a claim rather than a feature.

The transport routes by URL, which is what makes per-provider behaviour
testable: one handler can serve a 401 from Anthropic, valid JSON from OpenAI
and a timeout from Gemini within a single call, and the assertions can then be
about *which* provider answered.
"""

from __future__ import annotations

import json
from typing import Any, Callable

import httpx
import pytest

from app.config import settings
from app.core.llm import (
    ANTHROPIC_URL,
    GROK_URL,
    OPENAI_URL,
    LLMClient,
    LLMError,
    LLMJSONError,
    ProviderHealth,
)

PROVIDER_KEY_VARS: tuple[str, ...] = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GROK_API_KEY",
)

NO_BACKOFF: float = 0.0


def anthropic_body(text: str = "hello") -> dict[str, Any]:
    """Build an Anthropic response body.

    Args:
        text: Text the model appears to return.

    Returns:
        The response body.
    """
    return {
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": 5, "output_tokens": 3},
    }


def openai_body(text: str = "hello") -> dict[str, Any]:
    """Build an OpenAI response body.

    Args:
        text: Text the model appears to return.

    Returns:
        The response body.
    """
    return {
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3},
    }


def gemini_body(text: str = "hello") -> dict[str, Any]:
    """Build a Gemini response body.

    Args:
        text: Text the model appears to return.

    Returns:
        The response body.
    """
    return {
        "candidates": [{"content": {"parts": [{"text": text}]}}],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3},
    }


class Router:
    """Serves a different scripted response per provider, and counts calls."""

    def __init__(
        self,
        anthropic: Callable[[], httpx.Response] | httpx.Response | None = None,
        openai: Callable[[], httpx.Response] | httpx.Response | None = None,
        gemini: Callable[[], httpx.Response] | httpx.Response | None = None,
        grok: Callable[[], httpx.Response] | httpx.Response | None = None,
    ) -> None:
        """Initialise the router.

        Args:
            anthropic: Response for Anthropic's endpoint.
            openai: Response for OpenAI's endpoint.
            gemini: Response for Gemini's endpoint.
            grok: Response for xAI's endpoint.
        """
        self.scripted = {
            "anthropic": anthropic,
            "openai": openai,
            "gemini": gemini,
            "grok": grok,
        }
        self.calls: list[str] = []
        self.requests: list[httpx.Request] = []

    def provider_for(self, url: str) -> str:
        """Identify which provider a URL belongs to.

        Args:
            url: The intercepted request URL.

        Returns:
            The provider name.
        """
        if url.startswith(ANTHROPIC_URL):
            return "anthropic"
        if url.startswith(OPENAI_URL):
            return "openai"
        if url.startswith(GROK_URL):
            return "grok"
        return "gemini"

    def handler(self, request: httpx.Request) -> httpx.Response:
        """Serve the scripted response for this provider.

        Args:
            request: The intercepted request.

        Returns:
            The scripted response.

        Raises:
            httpx.TimeoutException: When the script says to time out.
        """
        provider = self.provider_for(str(request.url))
        self.calls.append(provider)
        self.requests.append(request)
        entry = self.scripted.get(provider)
        if entry is None:
            return httpx.Response(404, json={"error": "not scripted"})
        if callable(entry):
            return entry()
        return entry

    @property
    def transport(self) -> httpx.MockTransport:
        """The transport to hand to the client.

        Returns:
            A MockTransport bound to this router.
        """
        return httpx.MockTransport(self.handler)

    def count(self, provider: str) -> int:
        """How many times one provider was called.

        Args:
            provider: The provider name.

        Returns:
            The call count.
        """
        return self.calls.count(provider)


def timeout() -> httpx.Response:
    """Raise a timeout when called.

    Returns:
        Never returns.

    Raises:
        httpx.TimeoutException: Always.
    """
    raise httpx.TimeoutException("simulated timeout")


@pytest.fixture(autouse=True)
def isolate_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear real keys and make retries instant.

    Args:
        monkeypatch: pytest fixture used to patch settings.
    """
    for name in PROVIDER_KEY_VARS:
        monkeypatch.setattr(settings, name, None, raising=False)
    monkeypatch.setattr(settings, "LLM_MAX_RETRIES", 0, raising=False)
    monkeypatch.setattr(settings, "CIRCUIT_BREAKER_THRESHOLD", 2, raising=False)
    monkeypatch.setattr(
        settings, "CIRCUIT_BREAKER_COOLDOWN_SECONDS", 120.0, raising=False
    )


def configure(monkeypatch: pytest.MonkeyPatch, *providers: str) -> None:
    """Give the named providers a key and make them the whole chain.

    Args:
        monkeypatch: pytest fixture used to patch settings.
        *providers: Provider names to enable, in preference order.
    """
    for provider in providers:
        monkeypatch.setattr(
            settings, f"{provider.upper()}_API_KEY", f"key-{provider}", raising=False
        )
    monkeypatch.setattr(settings, "LLM_PROVIDER_ORDER", list(providers))


def build(router: Router) -> LLMClient:
    """Build a client wired to a router.

    Args:
        router: The routing transport.

    Returns:
        The client.
    """
    return LLMClient(
        transport=router.transport, max_retries=0, backoff_seconds=NO_BACKOFF
    )


class TestProviderHealth:
    """The breaker's state machine, tested without any HTTP at all."""

    def test_starts_closed(self) -> None:
        """A provider that has never failed is available."""
        health = ProviderHealth(name="x")
        assert health.is_open(now=0.0, cooldown=10.0) is False

    def test_opens_at_the_threshold(self) -> None:
        """Two consecutive failures open the breaker, one does not."""
        health = ProviderHealth(name="x")
        health.record_failure(now=0.0, reason="boom", threshold=2)
        assert health.is_open(now=0.0, cooldown=10.0) is False
        health.record_failure(now=0.0, reason="boom", threshold=2)
        assert health.is_open(now=0.0, cooldown=10.0) is True

    def test_closes_after_cooldown(self) -> None:
        """The breaker reopens the provider once the cooldown elapses."""
        health = ProviderHealth(name="x")
        for _ in range(2):
            health.record_failure(now=100.0, reason="boom", threshold=2)
        assert health.is_open(now=105.0, cooldown=10.0) is True
        assert health.is_open(now=110.0, cooldown=10.0) is False

    def test_reports_cooldown_remaining(self) -> None:
        """The remaining cooldown is exposed for the health endpoint."""
        health = ProviderHealth(name="x")
        for _ in range(2):
            health.record_failure(now=100.0, reason="boom", threshold=2)
        assert health.cooldown_remaining(now=104.0, cooldown=10.0) == 6.0
        assert health.cooldown_remaining(now=200.0, cooldown=10.0) == 0.0

    def test_success_resets_the_count(self) -> None:
        """One success clears the failure streak, so it must be consecutive."""
        health = ProviderHealth(name="x")
        health.record_failure(now=0.0, reason="boom", threshold=2)
        health.record_success(now=1.0)
        health.record_failure(now=2.0, reason="boom", threshold=2)
        assert health.is_open(now=2.0, cooldown=10.0) is False

    def test_success_closes_an_open_breaker(self) -> None:
        """A probe that succeeds puts the provider straight back in."""
        health = ProviderHealth(name="x")
        for _ in range(2):
            health.record_failure(now=0.0, reason="boom", threshold=2)
        health.record_success(now=1.0)
        assert health.is_open(now=1.0, cooldown=10.0) is False


class TestCircuitBreaker:
    """A failing provider stops being tried, then is tried again."""

    @pytest.mark.asyncio
    async def test_opens_after_threshold_and_skips_the_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Once open, the dead provider is not called again."""
        configure(monkeypatch, "anthropic", "openai")
        router = Router(
            anthropic=httpx.Response(401, json={"error": "bad key"}),
            openai=httpx.Response(200, json=openai_body()),
        )
        client = build(router)

        for _ in range(3):
            await client.complete(system="s", user="u")

        # Tried on the first two calls, skipped on the third.
        assert router.count("anthropic") == 2
        assert router.count("openai") == 3
        assert "anthropic" not in client.healthy_provider_names

    @pytest.mark.asyncio
    async def test_closed_breaker_keeps_the_preferred_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A healthy first choice is still tried first."""
        configure(monkeypatch, "anthropic", "openai")
        router = Router(
            anthropic=httpx.Response(200, json=anthropic_body()),
            openai=httpx.Response(200, json=openai_body()),
        )
        client = build(router)

        response = await client.complete(system="s", user="u")

        assert response.provider == "anthropic"
        assert router.count("openai") == 0

    @pytest.mark.asyncio
    async def test_reopens_after_cooldown(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After the cooldown the provider is probed by being tried again."""
        configure(monkeypatch, "anthropic", "openai")
        router = Router(
            anthropic=httpx.Response(401, json={"error": "bad key"}),
            openai=httpx.Response(200, json=openai_body()),
        )
        client = build(router)

        for _ in range(2):
            await client.complete(system="s", user="u")
        assert "anthropic" not in client.healthy_provider_names

        # Rewind the breaker rather than sleeping two minutes.
        health = client.health_by_provider["anthropic"]
        assert health.opened_at is not None
        health.opened_at -= settings.CIRCUIT_BREAKER_COOLDOWN_SECONDS + 1

        assert "anthropic" in client.healthy_provider_names
        before = router.count("anthropic")
        await client.complete(system="s", user="u")
        assert router.count("anthropic") == before + 1

    @pytest.mark.asyncio
    async def test_recovery_closes_the_breaker(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A provider that starts working again is trusted again."""
        configure(monkeypatch, "anthropic")
        state = {"fail": True}

        def flaky() -> httpx.Response:
            if state["fail"]:
                return httpx.Response(401, json={"error": "bad key"})
            return httpx.Response(200, json=anthropic_body())

        router = Router(anthropic=flaky)
        client = build(router)

        for _ in range(2):
            with pytest.raises(LLMError):
                await client.complete(system="s", user="u")
        assert client.healthy_provider_names == []

        state["fail"] = False
        # The only provider is in cooldown, so the breaker is bypassed rather
        # than the request being abandoned.
        response = await client.complete(system="s", user="u")
        assert response.provider == "anthropic"
        assert client.healthy_provider_names == ["anthropic"]

    @pytest.mark.asyncio
    async def test_breaker_is_bypassed_when_nothing_is_healthy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An open breaker predicts failure; it does not guarantee it.

        Refusing to try is only better than trying when something else might
        work. With every provider in cooldown, one timeout is a better outcome
        than a certain refusal.
        """
        configure(monkeypatch, "anthropic")
        router = Router(anthropic=httpx.Response(500, json={"error": "down"}))
        client = build(router)

        for _ in range(2):
            with pytest.raises(LLMError):
                await client.complete(system="s", user="u")

        calls_before = router.count("anthropic")
        with pytest.raises(LLMError):
            await client.complete(system="s", user="u")
        assert router.count("anthropic") > calls_before


class TestJSONFailoverToDifferentProvider:
    """Malformed structure is a property of the model, not of the moment."""

    @pytest.mark.asyncio
    async def test_unparsable_json_retries_elsewhere(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A provider returning prose is not asked again; the next one is."""
        configure(monkeypatch, "anthropic", "openai")
        router = Router(
            anthropic=httpx.Response(
                200, json=anthropic_body("I'm afraid I can't do that.")
            ),
            openai=httpx.Response(200, json=openai_body('{"ok": true}')),
        )
        client = build(router)

        value, response = await client.complete_json_with_response(
            system="s", user="u"
        )

        assert value == {"ok": True}
        assert response.provider == "openai"
        assert router.count("anthropic") == 1

    @pytest.mark.asyncio
    async def test_same_provider_is_never_retried(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The broken provider gets exactly one chance, not two."""
        configure(monkeypatch, "anthropic", "openai")
        router = Router(
            anthropic=httpx.Response(200, json=anthropic_body("not json")),
            openai=httpx.Response(200, json=openai_body("also not json")),
        )
        client = build(router)

        with pytest.raises(LLMJSONError):
            await client.complete_json_with_response(system="s", user="u")

        assert router.count("anthropic") == 1
        assert router.count("openai") == 1

    @pytest.mark.asyncio
    async def test_malformed_json_counts_towards_the_breaker(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An HTTP 200 carrying garbage is still a failure of that provider."""
        configure(monkeypatch, "anthropic", "openai")
        router = Router(
            anthropic=httpx.Response(200, json=anthropic_body("not json")),
            openai=httpx.Response(200, json=openai_body('{"ok": true}')),
        )
        client = build(router)

        for _ in range(2):
            await client.complete_json_with_response(system="s", user="u")

        assert "anthropic" not in client.healthy_provider_names

    @pytest.mark.asyncio
    async def test_error_names_the_last_provider_tried(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The raised error carries what the model actually said."""
        configure(monkeypatch, "anthropic")
        router = Router(
            anthropic=httpx.Response(200, json=anthropic_body("nope"))
        )
        client = build(router)

        with pytest.raises(LLMJSONError) as caught:
            await client.complete_json_with_response(system="s", user="u")

        assert caught.value.raw_text == "nope"

    @pytest.mark.asyncio
    async def test_no_provider_configured_still_raises_clearly(self) -> None:
        """With nothing configured the error explains what to set."""
        client = LLMClient()
        with pytest.raises(LLMError) as caught:
            await client.complete_json(system="s", user="u")
        assert "no LLM provider is configured" in str(caught.value)


class TestExplicitExclusion:
    """Callers can route around a provider for one call."""

    @pytest.mark.asyncio
    async def test_excluded_provider_is_not_called(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Exclusion skips the provider even though it is healthy."""
        configure(monkeypatch, "anthropic", "openai")
        router = Router(
            anthropic=httpx.Response(200, json=anthropic_body()),
            openai=httpx.Response(200, json=openai_body()),
        )
        client = build(router)

        response = await client.complete(
            system="s", user="u", exclude=["anthropic"]
        )

        assert response.provider == "openai"
        assert router.count("anthropic") == 0

    @pytest.mark.asyncio
    async def test_excluding_everything_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Excluding the whole chain is an error, not a silent success."""
        configure(monkeypatch, "anthropic")
        router = Router(anthropic=httpx.Response(200, json=anthropic_body()))
        client = build(router)

        with pytest.raises(LLMError) as caught:
            await client.complete(system="s", user="u", exclude=["anthropic"])
        assert "no LLM provider is available" in str(caught.value)


class TestStartupProbe:
    """One tiny call per provider, ordering the chain by what answered."""

    @pytest.mark.asyncio
    async def test_probe_records_health(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each provider's probe outcome is recorded."""
        configure(monkeypatch, "anthropic", "openai")
        router = Router(
            anthropic=httpx.Response(200, json=anthropic_body("OK")),
            openai=httpx.Response(401, json={"error": "bad key"}),
        )
        client = build(router)

        outcome = await client.probe_providers()

        assert outcome["anthropic"]["ok"] is True
        assert outcome["openai"]["ok"] is False
        assert client.health_by_provider["anthropic"].probe_ok is True
        assert client.health_by_provider["openai"].probe_ok is False

    @pytest.mark.asyncio
    async def test_failed_probe_opens_the_breaker_immediately(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The probe is the evidence; a user should not rediscover it."""
        configure(monkeypatch, "anthropic", "openai")
        router = Router(
            anthropic=httpx.Response(401, json={"error": "bad key"}),
            openai=httpx.Response(200, json=openai_body("OK")),
        )
        client = build(router)

        await client.probe_providers()

        assert client.healthy_provider_names == ["openai"]

    @pytest.mark.asyncio
    async def test_probe_uses_a_tiny_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A liveness check must not cost real tokens.

        The probe runs on every startup and on a periodic timer, so a probe
        that asked for a normal completion would turn health checking into a
        recurring bill.
        """
        configure(monkeypatch, "anthropic")
        router = Router(anthropic=httpx.Response(200, json=anthropic_body("OK")))
        client = build(router)

        await client.probe_providers()

        assert router.count("anthropic") == 1
        body = json.loads(router.requests[0].content)
        assert body["max_tokens"] <= 8
        assert len(body["messages"][0]["content"]) < 40

    @pytest.mark.asyncio
    async def test_probe_never_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A total outage at startup must not stop the app starting."""
        configure(monkeypatch, "anthropic", "openai")
        router = Router(anthropic=timeout, openai=timeout)
        client = build(router)

        outcome = await client.probe_providers()

        assert all(entry["ok"] is False for entry in outcome.values())
        assert client.healthy_provider_names == []

    @pytest.mark.asyncio
    async def test_probe_reorders_the_chain(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A provider that answered goes ahead of one that did not.

        Both are still tried, because the breaker's cooldown expires; the point
        is that the live one is tried first so the common case is fast.
        """
        configure(monkeypatch, "anthropic", "openai")
        router = Router(
            anthropic=httpx.Response(401, json={"error": "bad key"}),
            openai=httpx.Response(200, json=openai_body("OK")),
        )
        client = build(router)
        await client.probe_providers()

        # Let the breaker lapse so ordering, not exclusion, decides.
        health = client.health_by_provider["anthropic"]
        assert health.opened_at is not None
        health.opened_at -= settings.CIRCUIT_BREAKER_COOLDOWN_SECONDS + 1

        available, _ = client._selection()
        assert [provider.name for provider in available][0] == "openai"

    @pytest.mark.asyncio
    async def test_probe_respects_the_interval(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Re-probing too soon is skipped, so health checks cannot cost money."""
        configure(monkeypatch, "anthropic")
        router = Router(anthropic=httpx.Response(200, json=anthropic_body("OK")))
        client = build(router)

        await client.probe_providers()
        await client.probe_providers()

        assert router.count("anthropic") == 1

    @pytest.mark.asyncio
    async def test_force_reprobes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An explicit re-probe ignores the interval."""
        configure(monkeypatch, "anthropic")
        router = Router(anthropic=httpx.Response(200, json=anthropic_body("OK")))
        client = build(router)

        await client.probe_providers()
        await client.probe_providers(force=True)

        assert router.count("anthropic") == 2


class TestHealthReporting:
    """The health payload tells an operator what is actually happening."""

    @pytest.mark.asyncio
    async def test_reports_healthy_and_cooldown_sets(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Configured, healthy and in-cooldown are reported separately."""
        configure(monkeypatch, "anthropic", "openai")
        router = Router(
            anthropic=httpx.Response(401, json={"error": "bad key"}),
            openai=httpx.Response(200, json=openai_body()),
        )
        client = build(router)
        for _ in range(2):
            await client.complete(system="s", user="u")

        health = client.health()

        assert set(health["providers_configured"]) == {"anthropic", "openai"}
        assert health["providers_healthy"] == ["openai"]
        assert health["providers_in_cooldown"] == ["anthropic"]
        assert health["primary_provider"] == "openai"
        assert health["any_healthy"] is True

    @pytest.mark.asyncio
    async def test_per_provider_detail(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each provider reports its failure count and cooldown."""
        configure(monkeypatch, "anthropic")
        router = Router(anthropic=httpx.Response(401, json={"error": "bad key"}))
        client = build(router)
        for _ in range(2):
            with pytest.raises(LLMError):
                await client.complete(system="s", user="u")

        entry = client.health()["provider_health"][0]

        assert entry["name"] == "anthropic"
        assert entry["healthy"] is False
        assert entry["consecutive_failures"] == 2
        assert entry["cooldown_remaining_seconds"] > 0
        assert "bad key" in (entry["last_failure"] or "")

    def test_health_never_contains_key_material(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No part of any credential appears in the health payload."""
        monkeypatch.setattr(
            settings, "ANTHROPIC_API_KEY", "sk-ant-SUPERSECRETVALUE", raising=False
        )
        monkeypatch.setattr(settings, "LLM_PROVIDER_ORDER", ["anthropic"])
        rendered = json.dumps(LLMClient().health())
        assert "SUPERSECRET" not in rendered

    @pytest.mark.asyncio
    async def test_failure_detail_is_truncated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A provider error cannot flood the health payload."""
        configure(monkeypatch, "anthropic")
        router = Router(
            anthropic=httpx.Response(400, json={"error": {"message": "x" * 5000}})
        )
        client = build(router)
        with pytest.raises(LLMError):
            await client.complete(system="s", user="u")

        entry = client.health()["provider_health"][0]
        assert len(entry["last_failure"] or "") <= 200
