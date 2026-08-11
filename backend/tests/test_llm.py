"""Tests for the LLM client and its provider failover.

Every test runs against an ``httpx.MockTransport``. No test in this file makes
a network call, needs an API key, or costs money — the transport intercepts the
request before it leaves the process, which also lets each test assert on the
exact URL, headers and body the provider built.

The failover tests are the important ones. Provider fallback is a reliability
requirement for a system whose keys may be expired when a reviewer opens it, and
the only way to know it works is to make providers fail on purpose.
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
    AnthropicProvider,
    GeminiProvider,
    GrokProvider,
    LLMClient,
    LLMError,
    LLMJSONError,
    OpenAIProvider,
    strip_code_fences,
)

# All four provider key names, for clearing the environment.
PROVIDER_KEY_VARS: tuple[str, ...] = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GROK_API_KEY",
)

# Retries are exercised deliberately, so backoff is zeroed to keep tests fast.
NO_BACKOFF: float = 0.0


def anthropic_body(text: str = "hello") -> dict[str, Any]:
    """Build a minimal Anthropic Messages response.

    Args:
        text: The text the model should appear to have returned.

    Returns:
        A response body in Anthropic's schema.
    """
    return {
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": 11, "output_tokens": 7},
    }


def openai_body(text: str = "hello") -> dict[str, Any]:
    """Build a minimal OpenAI chat completions response.

    Args:
        text: The text the model should appear to have returned.

    Returns:
        A response body in OpenAI's schema.
    """
    return {
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 13, "completion_tokens": 5},
    }


def gemini_body(text: str = "hello") -> dict[str, Any]:
    """Build a minimal Gemini generateContent response.

    Args:
        text: The text the model should appear to have returned.

    Returns:
        A response body in Gemini's schema.
    """
    return {
        "candidates": [{"content": {"parts": [{"text": text}]}}],
        "usageMetadata": {"promptTokenCount": 17, "candidatesTokenCount": 3},
    }


class Recorder:
    """Captures the requests a provider makes and serves scripted responses."""

    def __init__(self, *responses: httpx.Response | Callable[[], httpx.Response]) -> None:
        """Initialise the recorder.

        Args:
            *responses: Responses to serve in order. The last one repeats once
                the list is exhausted.
        """
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        """Serve the next scripted response.

        Args:
            request: The intercepted request.

        Returns:
            The next response, or the last one once exhausted.

        Raises:
            httpx.TimeoutException: When the script says to time out.
        """
        self.requests.append(request)
        index = min(len(self.requests) - 1, len(self.responses) - 1)
        entry = self.responses[index]
        if callable(entry):
            return entry()
        return entry

    @property
    def transport(self) -> httpx.MockTransport:
        """The transport to hand to a provider or client.

        Returns:
            A MockTransport bound to this recorder's handler.
        """
        return httpx.MockTransport(self.handler)

    @property
    def count(self) -> int:
        """How many requests were made.

        Returns:
            The number of intercepted requests.
        """
        return len(self.requests)

    def body(self, index: int = 0) -> dict[str, Any]:
        """Decode the JSON body of a recorded request.

        Args:
            index: Which request to decode.

        Returns:
            The decoded request body.
        """
        return json.loads(self.requests[index].content)


def timeout() -> httpx.Response:
    """Raise a timeout when the transport is called.

    Returns:
        Never returns.

    Raises:
        httpx.TimeoutException: Always.
    """
    raise httpx.TimeoutException("simulated timeout")


@pytest.fixture(autouse=True)
def clear_provider_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every provider key from settings for the duration of a test.

    Tests opt providers in explicitly, so a real key in the developer's
    environment can never change a result.

    Args:
        monkeypatch: pytest fixture used to patch the settings object.
    """
    for name in PROVIDER_KEY_VARS:
        monkeypatch.setattr(settings, name, None, raising=False)


class TestAnthropicProvider:
    """Request shape and response parsing for Anthropic."""

    @pytest.mark.asyncio
    async def test_builds_correct_request(self) -> None:
        """URL, auth header, version header and body match the Messages API."""
        recorder = Recorder(httpx.Response(200, json=anthropic_body()))
        provider = AnthropicProvider(
            api_key="sk-ant-test",
            model="claude-sonnet-4-6",
            transport=recorder.transport,
        )

        await provider.complete(system="SYS", user="USER", max_tokens=256)

        request = recorder.requests[0]
        assert str(request.url) == ANTHROPIC_URL
        assert request.headers["x-api-key"] == "sk-ant-test"
        assert request.headers["anthropic-version"] == "2023-06-01"
        assert "authorization" not in request.headers

        body = recorder.body()
        assert body["model"] == "claude-sonnet-4-6"
        assert body["max_tokens"] == 256
        assert body["system"] == "SYS"
        assert body["messages"] == [{"role": "user", "content": "USER"}]

    @pytest.mark.asyncio
    async def test_omits_temperature_on_models_that_reject_it(self) -> None:
        """Newer Claude models reject sampling params, so they are dropped.

        Sending temperature to claude-opus-5 returns HTTP 400; passing it
        through would make every Anthropic call fail.
        """
        recorder = Recorder(httpx.Response(200, json=anthropic_body()))
        provider = AnthropicProvider(
            api_key="sk-ant-test",
            model="claude-opus-5",
            transport=recorder.transport,
        )

        await provider.complete(system="SYS", user="USER", temperature=0.7)

        assert provider.accepts_sampling_parameters is False
        assert "temperature" not in recorder.body()

    @pytest.mark.asyncio
    async def test_sends_temperature_on_models_that_accept_it(self) -> None:
        """Older Claude models still take temperature."""
        recorder = Recorder(httpx.Response(200, json=anthropic_body()))
        provider = AnthropicProvider(
            api_key="sk-ant-test",
            model="claude-haiku-4-5",
            transport=recorder.transport,
        )

        await provider.complete(system="SYS", user="USER", temperature=0.3)

        assert provider.accepts_sampling_parameters is True
        assert recorder.body()["temperature"] == 0.3

    @pytest.mark.asyncio
    async def test_parses_response(self) -> None:
        """Text blocks and token counts are read from Anthropic's schema."""
        recorder = Recorder(httpx.Response(200, json=anthropic_body("the answer")))
        provider = AnthropicProvider(
            api_key="k", model="claude-sonnet-4-6", transport=recorder.transport
        )

        response = await provider.complete(system="s", user="u")

        assert response.text == "the answer"
        assert response.provider == "anthropic"
        assert response.model == "claude-sonnet-4-6"
        assert response.input_tokens == 11
        assert response.output_tokens == 7
        assert response.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_joins_multiple_text_blocks(self) -> None:
        """Anthropic returns a list of blocks; the text ones are concatenated."""
        recorder = Recorder(
            httpx.Response(
                200,
                json={
                    "content": [
                        {"type": "text", "text": "one "},
                        {"type": "thinking", "thinking": "ignored"},
                        {"type": "text", "text": "two"},
                    ],
                    "usage": {"input_tokens": 1, "output_tokens": 2},
                },
            )
        )
        provider = AnthropicProvider(
            api_key="k", model="claude-sonnet-4-6", transport=recorder.transport
        )

        assert (await provider.complete(system="s", user="u")).text == "one two"


class TestOpenAIProvider:
    """Request shape and response parsing for OpenAI."""

    @pytest.mark.asyncio
    async def test_builds_correct_request(self) -> None:
        """URL, bearer auth and system-as-message match chat completions."""
        recorder = Recorder(httpx.Response(200, json=openai_body()))
        provider = OpenAIProvider(
            api_key="sk-openai-test", model="gpt-4o", transport=recorder.transport
        )

        await provider.complete(system="SYS", user="USER", temperature=0.2)

        request = recorder.requests[0]
        assert str(request.url) == OPENAI_URL
        assert request.headers["Authorization"] == "Bearer sk-openai-test"
        assert "x-api-key" not in request.headers

        body = recorder.body()
        assert body["model"] == "gpt-4o"
        assert body["temperature"] == 0.2
        assert body["messages"][0] == {"role": "system", "content": "SYS"}
        assert body["messages"][1] == {"role": "user", "content": "USER"}
        assert "system" not in body

    @pytest.mark.asyncio
    async def test_parses_response(self) -> None:
        """Choices and prompt/completion token counts are read correctly."""
        recorder = Recorder(httpx.Response(200, json=openai_body("gpt says hi")))
        provider = OpenAIProvider(
            api_key="k", model="gpt-4o", transport=recorder.transport
        )

        response = await provider.complete(system="s", user="u")

        assert response.text == "gpt says hi"
        assert response.provider == "openai"
        assert response.input_tokens == 13
        assert response.output_tokens == 5


class TestGeminiProvider:
    """Request shape and response parsing for Gemini."""

    @pytest.mark.asyncio
    async def test_builds_correct_request(self) -> None:
        """Model is in the URL, key is a header, system is systemInstruction."""
        recorder = Recorder(httpx.Response(200, json=gemini_body()))
        provider = GeminiProvider(
            api_key="gem-test",
            model="gemini-2.0-flash",
            transport=recorder.transport,
        )

        await provider.complete(system="SYS", user="USER", max_tokens=128)

        request = recorder.requests[0]
        assert "models/gemini-2.0-flash:generateContent" in str(request.url)
        assert request.headers["x-goog-api-key"] == "gem-test"

        body = recorder.body()
        assert body["systemInstruction"]["parts"][0]["text"] == "SYS"
        assert body["contents"][0]["parts"][0]["text"] == "USER"
        assert body["generationConfig"]["maxOutputTokens"] == 128

    @pytest.mark.asyncio
    async def test_key_is_not_in_the_url(self) -> None:
        """The key travels as a header, never a query string.

        Query strings are captured by access logs and proxies; headers are not.
        """
        recorder = Recorder(httpx.Response(200, json=gemini_body()))
        provider = GeminiProvider(
            api_key="gem-secret", model="gemini-2.0-flash", transport=recorder.transport
        )

        await provider.complete(system="s", user="u")

        assert "gem-secret" not in str(recorder.requests[0].url)

    @pytest.mark.asyncio
    async def test_parses_response(self) -> None:
        """Candidate parts and usageMetadata are read correctly."""
        recorder = Recorder(httpx.Response(200, json=gemini_body("gemini says hi")))
        provider = GeminiProvider(
            api_key="k", model="gemini-2.0-flash", transport=recorder.transport
        )

        response = await provider.complete(system="s", user="u")

        assert response.text == "gemini says hi"
        assert response.provider == "gemini"
        assert response.input_tokens == 17
        assert response.output_tokens == 3


class TestGrokProvider:
    """Request shape for Grok, which is OpenAI-compatible."""

    @pytest.mark.asyncio
    async def test_builds_correct_request(self) -> None:
        """Grok uses the xAI host with OpenAI's schema and bearer auth."""
        recorder = Recorder(httpx.Response(200, json=openai_body("grok says hi")))
        provider = GrokProvider(
            api_key="xai-test", model="grok-2-latest", transport=recorder.transport
        )

        response = await provider.complete(system="SYS", user="USER")

        request = recorder.requests[0]
        assert str(request.url) == GROK_URL
        assert request.headers["Authorization"] == "Bearer xai-test"
        assert recorder.body()["messages"][0]["role"] == "system"
        assert response.provider == "grok"
        assert response.text == "grok says hi"


class TestRetryPolicy:
    """Transient failures retry; permanent ones do not."""

    @pytest.mark.asyncio
    async def test_429_retries_within_the_same_provider(self) -> None:
        """A rate limit is transient, so the same provider is tried again."""
        recorder = Recorder(
            httpx.Response(429, json={"error": {"message": "slow down"}}),
            httpx.Response(200, json=anthropic_body("recovered")),
        )
        provider = AnthropicProvider(
            api_key="k",
            model="claude-sonnet-4-6",
            transport=recorder.transport,
            backoff_seconds=NO_BACKOFF,
        )

        response = await provider.complete(system="s", user="u")

        assert recorder.count == 2
        assert response.text == "recovered"

    @pytest.mark.asyncio
    async def test_401_does_not_retry(self) -> None:
        """A bad key is permanent; retrying it only wastes the user's time."""
        recorder = Recorder(httpx.Response(401, json={"error": {"message": "bad key"}}))
        provider = AnthropicProvider(
            api_key="k",
            model="claude-sonnet-4-6",
            transport=recorder.transport,
            backoff_seconds=NO_BACKOFF,
        )

        with pytest.raises(LLMError, match="401"):
            await provider.complete(system="s", user="u")

        assert recorder.count == 1

    @pytest.mark.asyncio
    async def test_400_does_not_retry(self) -> None:
        """A malformed request will be malformed on the retry too."""
        recorder = Recorder(httpx.Response(400, json={"error": {"message": "bad req"}}))
        provider = OpenAIProvider(
            api_key="k",
            model="gpt-4o",
            transport=recorder.transport,
            backoff_seconds=NO_BACKOFF,
        )

        with pytest.raises(LLMError):
            await provider.complete(system="s", user="u")

        assert recorder.count == 1

    @pytest.mark.asyncio
    async def test_timeout_is_transient(self) -> None:
        """A timeout is retried, then succeeds."""
        recorder = Recorder(timeout, httpx.Response(200, json=anthropic_body("ok")))
        provider = AnthropicProvider(
            api_key="k",
            model="claude-sonnet-4-6",
            transport=recorder.transport,
            backoff_seconds=NO_BACKOFF,
        )

        response = await provider.complete(system="s", user="u")

        assert recorder.count == 2
        assert response.text == "ok"

    @pytest.mark.asyncio
    async def test_retries_are_bounded(self) -> None:
        """Retries stop after max_retries; the loop cannot run forever."""
        recorder = Recorder(httpx.Response(503, text="unavailable"))
        provider = AnthropicProvider(
            api_key="k",
            model="claude-sonnet-4-6",
            transport=recorder.transport,
            max_retries=2,
            backoff_seconds=NO_BACKOFF,
        )

        with pytest.raises(LLMError):
            await provider.complete(system="s", user="u")

        assert recorder.count == 3  # first attempt plus two retries

    @pytest.mark.asyncio
    async def test_5xx_statuses_are_transient(self) -> None:
        """Every configured server-error status retries."""
        for status in (500, 502, 503, 504):
            recorder = Recorder(
                httpx.Response(status, text="err"),
                httpx.Response(200, json=anthropic_body("ok")),
            )
            provider = AnthropicProvider(
                api_key="k",
                model="claude-sonnet-4-6",
                transport=recorder.transport,
                backoff_seconds=NO_BACKOFF,
            )
            assert (await provider.complete(system="s", user="u")).text == "ok", status
            assert recorder.count == 2, status


class TestFailover:
    """Falling over between providers when one fails."""

    @pytest.mark.asyncio
    async def test_second_provider_serves_when_first_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A dead first provider is skipped and the second answers."""
        monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "sk-ant")
        monkeypatch.setattr(settings, "OPENAI_API_KEY", "sk-openai")

        def handler(request: httpx.Request) -> httpx.Response:
            """Fail Anthropic, serve OpenAI.

            Args:
                request: The intercepted request.

            Returns:
                A 401 for Anthropic, a 200 for OpenAI.
            """
            if "anthropic" in str(request.url):
                return httpx.Response(401, json={"error": {"message": "expired key"}})
            return httpx.Response(200, json=openai_body("from openai"))

        client = LLMClient(
            transport=httpx.MockTransport(handler), backoff_seconds=NO_BACKOFF
        )
        response = await client.complete(system="s", user="u")

        assert response.provider == "openai"
        assert response.text == "from openai"
        assert response.attempts == ["anthropic", "openai"]

    @pytest.mark.asyncio
    async def test_attempts_lists_only_the_first_provider_on_success(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No failover means a single-entry attempts list."""
        monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "sk-ant")
        monkeypatch.setattr(settings, "OPENAI_API_KEY", "sk-openai")

        client = LLMClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json=anthropic_body())
            ),
            backoff_seconds=NO_BACKOFF,
        )

        assert (await client.complete(system="s", user="u")).attempts == ["anthropic"]

    @pytest.mark.asyncio
    async def test_all_providers_failing_names_every_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The error names each provider and its distinct reason."""
        for name in PROVIDER_KEY_VARS:
            monkeypatch.setattr(settings, name, f"key-for-{name}")

        def handler(request: httpx.Request) -> httpx.Response:
            """Fail every provider differently.

            Args:
                request: The intercepted request.

            Returns:
                A distinct failure per host.
            """
            url = str(request.url)
            if "anthropic" in url:
                return httpx.Response(401, json={"error": {"message": "expired key"}})
            if "openai" in url:
                return httpx.Response(403, json={"error": {"message": "no access"}})
            if "googleapis" in url:
                return httpx.Response(404, json={"error": {"message": "no such model"}})
            return httpx.Response(400, json={"error": {"message": "bad request"}})

        client = LLMClient(
            transport=httpx.MockTransport(handler), backoff_seconds=NO_BACKOFF
        )

        with pytest.raises(LLMError) as caught:
            await client.complete(system="s", user="u")

        message = str(caught.value)
        for provider in ("anthropic", "openai", "gemini", "grok"):
            assert provider in message, provider
        assert "expired key" in message
        assert "no access" in message
        assert "no such model" in message
        assert len(caught.value.failures) == 4

    @pytest.mark.asyncio
    async def test_failure_details_are_serializable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The error can be returned from an API without further work."""
        monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "sk-ant")
        client = LLMClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(401, text="nope")
            ),
            backoff_seconds=NO_BACKOFF,
        )

        with pytest.raises(LLMError) as caught:
            await client.complete(system="s", user="u")

        payload = caught.value.to_dict()
        assert json.loads(json.dumps(payload))["failures"][0]["provider"] == "anthropic"

    @pytest.mark.asyncio
    async def test_no_providers_configured_raises_clearly(self) -> None:
        """With no keys at all the error explains what to set."""
        client = LLMClient()

        assert client.providers == []
        with pytest.raises(LLMError, match="no LLM provider is configured"):
            await client.complete(system="s", user="u")


class TestProviderSelection:
    """Only configured providers are instantiated, in preference order."""

    def test_only_configured_providers_are_built(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A provider without a key is never instantiated."""
        monkeypatch.setattr(settings, "OPENAI_API_KEY", "sk-openai")

        assert LLMClient().provider_names == ["openai"]

    def test_providers_follow_configured_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Order comes from LLM_PROVIDER_ORDER, not from which key was set."""
        monkeypatch.setattr(settings, "GROK_API_KEY", "xai")
        monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "sk-ant")
        monkeypatch.setattr(settings, "GEMINI_API_KEY", "gem")

        assert LLMClient().provider_names == ["anthropic", "gemini", "grok"]

    def test_blank_key_does_not_configure_a_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Whitespace is not a credential."""
        monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "   ")

        assert LLMClient().provider_names == []


class TestCompleteJson:
    """JSON extraction, fence stripping and parse failures."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ('{"a": 1}', '{"a": 1}'),
            ('```json\n{"a": 1}\n```', '{"a": 1}'),
            ('```\n{"a": 1}\n```', '{"a": 1}'),
            ('  ```json\n{"a": 1}\n```  ', '{"a": 1}'),
        ],
    )
    def test_strip_code_fences(self, raw: str, expected: str) -> None:
        """Fenced and unfenced JSON both come out clean."""
        assert strip_code_fences(raw) == expected

    @pytest.mark.asyncio
    async def test_parses_fenced_json(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ```json fenced reply is stripped and parsed."""
        monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "sk-ant")
        client = LLMClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, json=anthropic_body('```json\n{"intent": "trend"}\n```')
                )
            ),
            backoff_seconds=NO_BACKOFF,
        )

        assert await client.complete_json(system="s", user="u") == {"intent": "trend"}

    @pytest.mark.asyncio
    async def test_appends_json_instruction_to_system_prompt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The model is told to return JSON only."""
        monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "sk-ant")
        recorder = Recorder(httpx.Response(200, json=anthropic_body("{}")))
        client = LLMClient(transport=recorder.transport, backoff_seconds=NO_BACKOFF)

        await client.complete_json(system="ORIGINAL", user="u")

        system = recorder.body()["system"]
        assert system.startswith("ORIGINAL")
        assert "valid JSON" in system

    @pytest.mark.asyncio
    async def test_malformed_json_raises_with_the_raw_text(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The error carries what the model actually said, not a guess."""
        monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "sk-ant")
        garbage = "I think the answer is probably 42"
        client = LLMClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json=anthropic_body(garbage))
            ),
            backoff_seconds=NO_BACKOFF,
        )

        with pytest.raises(LLMJSONError) as caught:
            await client.complete_json(system="s", user="u")

        assert garbage in str(caught.value)
        assert caught.value.raw_text == garbage


class TestHealth:
    """Health output must never leak credentials."""

    def test_reports_configured_providers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Configured providers and the primary are reported."""
        monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "sk-ant")
        monkeypatch.setattr(settings, "GROK_API_KEY", "xai")

        health = LLMClient().health()

        assert health["providers_configured"] == ["anthropic", "grok"]
        assert health["primary_provider"] == "anthropic"
        assert health["any_configured"] is True

    def test_reports_nothing_configured(self) -> None:
        """With no keys the report says so rather than omitting the field."""
        health = LLMClient().health()

        assert health["providers_configured"] == []
        assert health["primary_provider"] is None
        assert health["any_configured"] is False

    def test_never_exposes_a_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Realistically formatted keys do not appear in the output."""
        secrets = {
            "ANTHROPIC_API_KEY": "sk-ant-api03-SUPERSECRETVALUE01",
            "OPENAI_API_KEY": "sk-proj-SUPERSECRETVALUE02",
            "GEMINI_API_KEY": "AIzaSUPERSECRETVALUE03",
            "GROK_API_KEY": "xai-SUPERSECRETVALUE04",
        }
        for name, value in secrets.items():
            monkeypatch.setattr(settings, name, value)

        serialized = json.dumps(LLMClient().health())

        for value in secrets.values():
            assert value not in serialized

    def test_never_exposes_even_a_fragment_of_a_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Not a prefix, not a masked fragment, not any six characters.

        A partial key is still a leaked secret, and health output reaches logs
        and screenshots. The keys here are opaque tokens rather than realistic
        ones so the scan cannot trip over an incidental substring such as the
        provider's own name inside 'sk-openai-...'.
        """
        secrets = {
            "ANTHROPIC_API_KEY": "QWXZJ7742HHVBBK01",
            "OPENAI_API_KEY": "PLMNB6613GGTYYJ02",
            "GEMINI_API_KEY": "ZXCVM5584FFRUUH03",
            "GROK_API_KEY": "TREWQ3396DDSIIG04",
        }
        for name, value in secrets.items():
            monkeypatch.setattr(settings, name, value)

        serialized = json.dumps(LLMClient().health())

        for value in secrets.values():
            for start in range(len(value) - 6):
                assert value[start : start + 6] not in serialized

    def test_exposes_model_ids(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Model ids are not secret and are useful for debugging a deploy."""
        monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "sk-ant")

        assert LLMClient().health()["models"]["anthropic"] == settings.ANTHROPIC_MODEL
