"""Provider-agnostic LLM client with automatic failover.

This system is a portfolio piece whose deployed URL may be opened months from
now, after whichever API keys were configured have expired or been revoked.
Provider failover is therefore a reliability requirement, not an optimisation:
the client tries every configured provider in order and, when all of them fail,
raises a structured :class:`LLMError` naming each provider and its reason rather
than surfacing a stack trace from whichever one happened to be tried last.

Each provider calls its HTTP API directly through ``httpx`` rather than through
a vendor SDK. Four SDKs would mean four dependency trees, four release cadences
and four sets of abstractions to read past; the raw request and response shapes
are short enough to read in one sitting and make the differences between
providers explicit.

Retries and failover handle different failures. A transient fault (429, 5xx,
timeout, connection reset) is retried *within* the same provider with
exponential backoff, because the provider is fine and the request is not. A
permanent fault (401 from a dead key, 400 from a malformed request) fails over
to the next provider immediately: retrying a rejected key only wastes the
seconds a user is waiting.

Usage::

    client = get_llm_client()
    response = await client.complete(system="...", user="...")
    plan = await client.complete_json(system="...", user="...")
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, replace
from typing import Any, Final, Sequence

import httpx

from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# --- Failure classification ------------------------------------------------
# Statuses worth retrying against the same provider: the provider is rate
# limiting us or having a bad moment, and the same request may well succeed.
# Everything else (401 bad key, 403 no access, 400 bad request, 404 retired
# model) is permanent for this provider - fail over instead of burning time.
TRANSIENT_STATUS_CODES: Final[frozenset[int]] = frozenset({429, 500, 502, 503, 504})

# --- Anthropic ------------------------------------------------------------
ANTHROPIC_URL: Final[str] = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION: Final[str] = "2023-06-01"

# Anthropic removed the sampling parameters (temperature, top_p, top_k) from
# its newer models: sending temperature to any of these returns HTTP 400, so
# the parameter is dropped for them rather than passed through. Older Claude
# models still accept it. Matched by prefix because the same family covers
# several published ids.
ANTHROPIC_NO_SAMPLING_PREFIXES: Final[tuple[str, ...]] = (
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-fable-5",
    "claude-mythos-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
)

# --- Other provider endpoints ---------------------------------------------
OPENAI_URL: Final[str] = "https://api.openai.com/v1/chat/completions"
GROK_URL: Final[str] = "https://api.x.ai/v1/chat/completions"
GEMINI_URL_TEMPLATE: Final[str] = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)

# Instruction appended to the system prompt by complete_json.
JSON_ONLY_INSTRUCTION: Final[str] = (
    "Respond with a single valid JSON value and nothing else. Do not wrap it in "
    "markdown code fences, do not prefix it with an explanation, and do not add "
    "trailing commentary."
)


class LLMError(RuntimeError):
    """An LLM call that could not be completed.

    Attributes:
        failures: Per-provider failure reasons, empty when the error is not a
            failover exhaustion (for example a configuration error).
    """

    def __init__(self, message: str, failures: Sequence[ProviderFailure] = ()) -> None:
        """Initialise the error.

        Args:
            message: Human-readable summary.
            failures: The per-provider failures that led here.
        """
        super().__init__(message)
        self.failures: list[ProviderFailure] = list(failures)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the error for an API response.

        Returns:
            The message and every provider failure, JSON-serializable.
        """
        return {
            "error": str(self),
            "failures": [asdict(failure) for failure in self.failures],
        }


class TransientLLMError(LLMError):
    """A failure worth retrying against the same provider."""


class LLMJSONError(LLMError):
    """The model's response could not be parsed as JSON.

    Attributes:
        raw_text: The exact text returned by the model, so the caller can see
            what was actually produced rather than guessing.
    """

    def __init__(self, message: str, raw_text: str) -> None:
        """Initialise the error.

        Args:
            message: Human-readable summary.
            raw_text: The unparsable model output.
        """
        super().__init__(message)
        self.raw_text = raw_text


@dataclass(frozen=True)
class ProviderFailure:
    """Why one provider failed.

    Attributes:
        provider: Provider name.
        model: Model that was attempted.
        reason: The failure, as a readable string.
        attempts: How many HTTP attempts were made, including retries.
    """

    provider: str
    model: str
    reason: str
    attempts: int


@dataclass(frozen=True)
class LLMResponse:
    """A successful completion.

    Attributes:
        text: The model's response text.
        provider: Provider that produced it.
        model: Model that produced it.
        input_tokens: Prompt tokens consumed.
        output_tokens: Completion tokens produced.
        latency_ms: Wall-clock time for the successful call.
        attempts: Provider names tried, in order, ending with the one that
            succeeded. A single-element list means the first choice worked.
    """

    text: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    attempts: list[str]


# ---------------------------------------------------------------------------
# Provider base
# ---------------------------------------------------------------------------


class LLMProvider(ABC):
    """One LLM vendor's HTTP API.

    Subclasses supply the vendor-specific request and response shapes through
    four small hooks; the HTTP call, the retry policy and the timing live here
    so every provider behaves identically in the ways that matter to callers.
    """

    def __init__(
        self,
        api_key: str | None,
        model: str,
        timeout_seconds: int | None = None,
        max_retries: int | None = None,
        backoff_seconds: float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Initialise the provider.

        Args:
            api_key: Credential, or None when unconfigured.
            model: Model identifier to request.
            timeout_seconds: Per-request timeout. Defaults to settings.
            max_retries: Retries after the first attempt. Defaults to settings.
            backoff_seconds: Base delay for exponential backoff. Defaults to
                settings.
            transport: Optional httpx transport, used by tests to serve mock
                responses without any network access.
        """
        self.api_key = (api_key or "").strip() or None
        self.model = model
        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else settings.LLM_TIMEOUT_SECONDS
        )
        self.max_retries = (
            max_retries if max_retries is not None else settings.LLM_MAX_RETRIES
        )
        self.backoff_seconds = (
            backoff_seconds
            if backoff_seconds is not None
            else settings.LLM_RETRY_BACKOFF_SECONDS
        )
        self._transport = transport

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name, matching the entries in ``LLM_PROVIDER_ORDER``.

        Returns:
            The lowercase provider name.
        """

    @property
    def is_configured(self) -> bool:
        """Whether this provider has a usable credential.

        Returns:
            True when a non-blank API key is present.
        """
        return self.api_key is not None

    @property
    @abstractmethod
    def endpoint(self) -> str:
        """The URL to POST to.

        Returns:
            The provider's completion endpoint.
        """

    @abstractmethod
    def build_headers(self) -> dict[str, str]:
        """Build the request headers, including authentication.

        Returns:
            Headers for the POST.
        """

    @abstractmethod
    def build_payload(
        self, system: str, user: str, max_tokens: int, temperature: float
    ) -> dict[str, Any]:
        """Build the request body in this provider's schema.

        Args:
            system: System prompt.
            user: User message.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            The JSON body to POST.
        """

    @abstractmethod
    def parse_response(self, data: dict[str, Any]) -> tuple[str, int, int]:
        """Extract text and token counts from this provider's response schema.

        Args:
            data: The decoded JSON response.

        Returns:
            The response text, input token count and output token count.

        Raises:
            LLMError: If the response does not have the expected shape.
        """

    async def complete(
        self,
        system: str,
        user: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Send one completion request, retrying transient failures.

        Args:
            system: System prompt.
            user: User message.
            max_tokens: Maximum tokens to generate. Defaults to settings.
            temperature: Sampling temperature. Defaults to settings.

        Returns:
            The completion, with timing and token counts attached.

        Raises:
            LLMError: If the provider is unconfigured, or the call fails after
                exhausting retries.
        """
        if not self.is_configured:
            raise LLMError(f"{self.name} has no API key configured")

        payload = self.build_payload(
            system=system,
            user=user,
            max_tokens=(
                max_tokens if max_tokens is not None else settings.LLM_MAX_TOKENS
            ),
            temperature=(
                temperature if temperature is not None else settings.LLM_TEMPERATURE
            ),
        )

        started = time.perf_counter()
        data, attempts = await self._post_with_retries(payload)
        latency_ms = (time.perf_counter() - started) * 1000

        text, input_tokens, output_tokens = self.parse_response(data)
        logger.info(
            "llm_call_succeeded",
            extra={
                "provider": self.name,
                "model": self.model,
                "latency_ms": round(latency_ms, 1),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "http_attempts": attempts,
            },
        )
        return LLMResponse(
            text=text,
            provider=self.name,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            attempts=[self.name],
        )

    async def _post_with_retries(
        self, payload: dict[str, Any]
    ) -> tuple[dict[str, Any], int]:
        """POST the payload, retrying only transient failures.

        Args:
            payload: The request body.

        Returns:
            The decoded response and the number of HTTP attempts made.

        Raises:
            LLMError: On a permanent failure, immediately and without retrying.
            TransientLLMError: When every retry is exhausted.
        """
        last_error: TransientLLMError | None = None

        for attempt in range(self.max_retries + 1):
            try:
                return await self._post_once(payload), attempt + 1
            except TransientLLMError as error:
                last_error = error
                if attempt < self.max_retries:
                    delay = self.backoff_seconds * (2**attempt)
                    logger.warning(
                        "llm_retry",
                        extra={
                            "provider": self.name,
                            "model": self.model,
                            "attempt": attempt + 1,
                            "delay_seconds": delay,
                            "reason": str(error),
                        },
                    )
                    await asyncio.sleep(delay)

        assert last_error is not None  # loop runs at least once
        raise last_error

    async def _post_once(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Make a single HTTP attempt.

        Args:
            payload: The request body.

        Returns:
            The decoded JSON response.

        Raises:
            TransientLLMError: On a timeout, connection error or retryable
                status code.
            LLMError: On a permanent status code or an undecodable body.
        """
        try:
            async with httpx.AsyncClient(
                transport=self._transport, timeout=self.timeout_seconds
            ) as client:
                response = await client.post(
                    self.endpoint, headers=self.build_headers(), json=payload
                )
        except httpx.TimeoutException as error:
            raise TransientLLMError(
                f"{self.name} timed out after {self.timeout_seconds}s"
            ) from error
        except httpx.HTTPError as error:
            raise TransientLLMError(
                f"{self.name} connection error: {type(error).__name__}: {error}"
            ) from error

        if response.status_code in TRANSIENT_STATUS_CODES:
            raise TransientLLMError(
                f"{self.name} returned HTTP {response.status_code}: "
                f"{self._error_detail(response)}"
            )
        if response.status_code >= 400:
            raise LLMError(
                f"{self.name} returned HTTP {response.status_code}: "
                f"{self._error_detail(response)}"
            )

        try:
            return dict(response.json())
        except (ValueError, TypeError) as error:
            raise LLMError(
                f"{self.name} returned a body that is not JSON: {error}"
            ) from error

    @staticmethod
    def _error_detail(response: httpx.Response, limit: int = 300) -> str:
        """Summarise an error response body for logging.

        Args:
            response: The failed response.
            limit: Maximum characters to include.

        Returns:
            The provider's error message when it can be found, otherwise the
            truncated raw body.
        """
        try:
            body = response.json()
        except ValueError:
            return response.text[:limit]
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict) and "message" in error:
                return str(error["message"])[:limit]
            if isinstance(error, str):
                return error[:limit]
        return json.dumps(body)[:limit]


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


class AnthropicProvider(LLMProvider):
    """Anthropic Claude via the Messages API."""

    @property
    def name(self) -> str:
        """Provider name.

        Returns:
            ``"anthropic"``.
        """
        return "anthropic"

    @property
    def endpoint(self) -> str:
        """The Messages endpoint.

        Returns:
            The Anthropic messages URL.
        """
        return ANTHROPIC_URL

    @property
    def accepts_sampling_parameters(self) -> bool:
        """Whether the configured model accepts ``temperature``.

        Anthropic removed the sampling parameters on its newer model families;
        sending ``temperature`` to one of those returns HTTP 400, so it must be
        omitted rather than passed through.

        Returns:
            False when the model belongs to a family that rejects them.
        """
        return not self.model.startswith(ANTHROPIC_NO_SAMPLING_PREFIXES)

    def build_headers(self) -> dict[str, str]:
        """Build Anthropic's headers.

        Anthropic authenticates with ``x-api-key`` rather than a bearer token,
        and requires an explicit API version.

        Returns:
            The request headers.
        """
        return {
            "x-api-key": self.api_key or "",
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

    def build_payload(
        self, system: str, user: str, max_tokens: int, temperature: float
    ) -> dict[str, Any]:
        """Build an Anthropic Messages request.

        The system prompt is a top-level field here, not a message with a
        ``system`` role, and ``max_tokens`` is required rather than optional.

        Args:
            system: System prompt.
            user: User message.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature, omitted for models that reject it.

        Returns:
            The request body.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if self.accepts_sampling_parameters:
            payload["temperature"] = temperature
        return payload

    def parse_response(self, data: dict[str, Any]) -> tuple[str, int, int]:
        """Parse an Anthropic Messages response.

        Content is a list of typed blocks; only the text blocks are joined.

        Args:
            data: The decoded response.

        Returns:
            Response text, input tokens and output tokens.

        Raises:
            LLMError: If the response contains no text block.
        """
        blocks = data.get("content") or []
        text = "".join(
            str(block.get("text", ""))
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "text"
        )
        if not text:
            raise LLMError(f"anthropic returned no text content: {json.dumps(data)[:300]}")
        usage = data.get("usage") or {}
        return (
            text,
            int(usage.get("input_tokens", 0)),
            int(usage.get("output_tokens", 0)),
        )


# ---------------------------------------------------------------------------
# OpenAI-compatible providers
# ---------------------------------------------------------------------------


class OpenAICompatibleProvider(LLMProvider):
    """Base for providers speaking OpenAI's chat completions schema.

    OpenAI and xAI accept the same request and response shapes, so they differ
    only in endpoint and name.
    """

    def build_headers(self) -> dict[str, str]:
        """Build bearer-token headers.

        Returns:
            The request headers.
        """
        return {
            "Authorization": f"Bearer {self.api_key or ''}",
            "Content-Type": "application/json",
        }

    def build_payload(
        self, system: str, user: str, max_tokens: int, temperature: float
    ) -> dict[str, Any]:
        """Build a chat completions request.

        The system prompt is the first message with role ``system``, unlike
        Anthropic where it is a top-level field.

        Args:
            system: System prompt.
            user: User message.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            The request body.
        """
        return {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

    def parse_response(self, data: dict[str, Any]) -> tuple[str, int, int]:
        """Parse a chat completions response.

        Args:
            data: The decoded response.

        Returns:
            Response text, prompt tokens and completion tokens.

        Raises:
            LLMError: If the response contains no choices.
        """
        choices = data.get("choices") or []
        if not choices:
            raise LLMError(
                f"{self.name} returned no choices: {json.dumps(data)[:300]}"
            )
        text = str((choices[0].get("message") or {}).get("content") or "")
        if not text:
            raise LLMError(
                f"{self.name} returned an empty message: {json.dumps(data)[:300]}"
            )
        usage = data.get("usage") or {}
        return (
            text,
            int(usage.get("prompt_tokens", 0)),
            int(usage.get("completion_tokens", 0)),
        )


class OpenAIProvider(OpenAICompatibleProvider):
    """OpenAI via the chat completions API."""

    @property
    def name(self) -> str:
        """Provider name.

        Returns:
            ``"openai"``.
        """
        return "openai"

    @property
    def endpoint(self) -> str:
        """The chat completions endpoint.

        Returns:
            The OpenAI chat completions URL.
        """
        return OPENAI_URL


class GrokProvider(OpenAICompatibleProvider):
    """xAI Grok, which exposes an OpenAI-compatible API."""

    @property
    def name(self) -> str:
        """Provider name.

        Returns:
            ``"grok"``.
        """
        return "grok"

    @property
    def endpoint(self) -> str:
        """The chat completions endpoint.

        Returns:
            The xAI chat completions URL.
        """
        return GROK_URL


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------


class GeminiProvider(LLMProvider):
    """Google Gemini via the generateContent API."""

    @property
    def name(self) -> str:
        """Provider name.

        Returns:
            ``"gemini"``.
        """
        return "gemini"

    @property
    def endpoint(self) -> str:
        """The generateContent endpoint for the configured model.

        Returns:
            The model-specific Gemini URL.
        """
        return GEMINI_URL_TEMPLATE.format(model=self.model)

    def build_headers(self) -> dict[str, str]:
        """Build Gemini's headers.

        The key is sent as a header rather than a query parameter: query
        strings end up in access logs and error messages, headers do not.

        Returns:
            The request headers.
        """
        return {
            "x-goog-api-key": self.api_key or "",
            "Content-Type": "application/json",
        }

    def build_payload(
        self, system: str, user: str, max_tokens: int, temperature: float
    ) -> dict[str, Any]:
        """Build a generateContent request.

        Gemini carries the system prompt in ``systemInstruction`` and nests
        generation limits inside ``generationConfig``.

        Args:
            system: System prompt.
            user: User message.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            The request body.
        """
        return {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            },
        }

    def parse_response(self, data: dict[str, Any]) -> tuple[str, int, int]:
        """Parse a generateContent response.

        Args:
            data: The decoded response.

        Returns:
            Response text, prompt tokens and candidate tokens.

        Raises:
            LLMError: If the response contains no candidate text.
        """
        candidates = data.get("candidates") or []
        if not candidates:
            raise LLMError(f"gemini returned no candidates: {json.dumps(data)[:300]}")
        parts = ((candidates[0].get("content") or {}).get("parts")) or []
        text = "".join(str(part.get("text", "")) for part in parts)
        if not text:
            raise LLMError(f"gemini returned no text parts: {json.dumps(data)[:300]}")
        usage = data.get("usageMetadata") or {}
        return (
            text,
            int(usage.get("promptTokenCount", 0)),
            int(usage.get("candidatesTokenCount", 0)),
        )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

# Provider name -> (class, settings attribute holding the key, settings
# attribute holding the model id).
PROVIDER_REGISTRY: Final[dict[str, tuple[type[LLMProvider], str, str]]] = {
    "anthropic": (AnthropicProvider, "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL"),
    "openai": (OpenAIProvider, "OPENAI_API_KEY", "OPENAI_MODEL"),
    "gemini": (GeminiProvider, "GEMINI_API_KEY", "GEMINI_MODEL"),
    "grok": (GrokProvider, "GROK_API_KEY", "GROK_MODEL"),
}


def strip_code_fences(text: str) -> str:
    """Remove a surrounding markdown code fence from model output.

    Models routinely wrap JSON in ```json fences despite being told not to.
    Stripping them is cheaper and more reliable than a retry.

    Args:
        text: Raw model output.

    Returns:
        The text with an enclosing fence removed, if one was present.
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()[1:]  # drop the opening ``` or ```json
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


#: Prompt used to check a provider is alive. Deliberately trivial: the probe
#: measures reachability and credential validity, not capability, and a longer
#: prompt would cost real money on every startup.
PROBE_SYSTEM: Final[str] = "Reply with the single word OK."
PROBE_USER: Final[str] = "ping"
PROBE_MAX_TOKENS: Final[int] = 8


@dataclass
class ProviderHealth:
    """The rolling health of one provider.

    A provider is not simply up or down. It can be untried, known good, or
    failing repeatedly enough that continuing to try it costs a user real
    seconds for no chance of success. The breaker exists for that third state.

    Attributes:
        name: Provider name.
        consecutive_failures: Failures since the last success.
        opened_at: Monotonic time the breaker opened, or None when closed.
        last_success: Monotonic time of the last successful call.
        last_failure_reason: Why the most recent failure happened.
        last_probe: Wall-clock ISO timestamp of the last startup/periodic
            probe, or None when never probed.
        probe_ok: Whether that probe succeeded.
        probe_latency_ms: How long the probe took, when it ran.
    """

    name: str
    consecutive_failures: int = 0
    opened_at: float | None = None
    last_success: float | None = None
    last_failure_reason: str | None = None
    last_probe: str | None = None
    probe_ok: bool | None = None
    probe_latency_ms: float | None = None

    def is_open(self, now: float, cooldown: float) -> bool:
        """Whether the breaker is currently holding this provider back.

        Args:
            now: Current monotonic time.
            cooldown: Seconds the breaker stays open.

        Returns:
            True while the provider should be skipped.
        """
        if self.opened_at is None:
            return False
        if now - self.opened_at >= cooldown:
            return False
        return True

    def cooldown_remaining(self, now: float, cooldown: float) -> float:
        """Seconds until the provider is probed again.

        Args:
            now: Current monotonic time.
            cooldown: Seconds the breaker stays open.

        Returns:
            Remaining seconds, or 0.0 when the breaker is closed.
        """
        if not self.is_open(now, cooldown):
            return 0.0
        assert self.opened_at is not None
        return max(0.0, cooldown - (now - self.opened_at))

    def record_success(self, now: float) -> None:
        """Note a successful call, closing the breaker.

        Args:
            now: Current monotonic time.
        """
        self.consecutive_failures = 0
        self.opened_at = None
        self.last_success = now
        self.last_failure_reason = None

    def record_failure(self, now: float, reason: str, threshold: int) -> None:
        """Note a failed call, opening the breaker at the threshold.

        Args:
            now: Current monotonic time.
            reason: Why the call failed.
            threshold: Consecutive failures that open the breaker.
        """
        self.consecutive_failures += 1
        self.last_failure_reason = reason
        if self.consecutive_failures >= threshold and self.opened_at is None:
            self.opened_at = now


class LLMClient:
    """Calls LLM providers in preference order, failing over on error.

    Three behaviours beyond plain failover, each addressing a way the naive
    version wastes a user's time:

    * **A circuit breaker.** After
      :data:`Settings.CIRCUIT_BREAKER_THRESHOLD` consecutive failures a
      provider is skipped for a cooldown, then tried again. Without this every
      request pays the full timeout of a provider whose key expired months ago.
    * **Provider exclusion.** Callers can ask for a completion that avoids
      named providers. The JSON path uses it: a model that returned malformed
      structure once is likely to do it again, so the retry goes elsewhere.
    * **A liveness probe.** One tiny call per provider at startup reorders the
      chain so the first provider tried is one that actually answered.
    """

    def __init__(
        self,
        transport: httpx.AsyncBaseTransport | None = None,
        max_retries: int | None = None,
        backoff_seconds: float | None = None,
    ) -> None:
        """Build the client from configured providers only.

        Providers without a key are not instantiated at all, so an unconfigured
        provider can never be tried and can never appear in a failure list.

        Args:
            transport: Optional httpx transport shared by every provider, used
                by tests to serve mock responses without network access.
            max_retries: Override the configured retry count.
            backoff_seconds: Override the configured backoff base.
        """
        self.providers: list[LLMProvider] = []
        for name in settings.LLM_PROVIDER_ORDER:
            entry = PROVIDER_REGISTRY.get(name)
            if entry is None:
                logger.warning("unknown_llm_provider", extra={"provider": name})
                continue
            provider_class, key_attr, model_attr = entry
            provider = provider_class(
                api_key=getattr(settings, key_attr, None),
                model=getattr(settings, model_attr),
                max_retries=max_retries,
                backoff_seconds=backoff_seconds,
                transport=transport,
            )
            if provider.is_configured:
                self.providers.append(provider)

        self.health_by_provider: dict[str, ProviderHealth] = {
            provider.name: ProviderHealth(name=provider.name)
            for provider in self.providers
        }
        self._probed_at: float | None = None

    @property
    def provider_names(self) -> list[str]:
        """Names of the configured providers, in preference order.

        Returns:
            The provider names that will actually be tried.
        """
        return [provider.name for provider in self.providers]

    @property
    def healthy_provider_names(self) -> list[str]:
        """Providers not currently held back by the breaker.

        Returns:
            The provider names that would be tried right now.
        """
        now = time.monotonic()
        return [
            provider.name
            for provider in self.providers
            if not self.health_by_provider[provider.name].is_open(
                now, settings.CIRCUIT_BREAKER_COOLDOWN_SECONDS
            )
        ]

    def _selection(
        self, exclude: Sequence[str] = ()
    ) -> tuple[list[LLMProvider], list[LLMProvider]]:
        """Choose which providers to try, and in what order.

        Args:
            exclude: Provider names to skip entirely for this call.

        Returns:
            The providers to try, and the ones the breaker is holding back.
            The held-back list is returned rather than discarded because it is
            used as a last resort: refusing to try a provider is only better
            than trying it while some other provider might still work.
        """
        now = time.monotonic()
        excluded = {name.lower() for name in exclude}
        available: list[LLMProvider] = []
        blocked: list[LLMProvider] = []

        for provider in self.providers:
            if provider.name in excluded:
                continue
            health = self.health_by_provider[provider.name]
            if health.is_open(now, settings.CIRCUIT_BREAKER_COOLDOWN_SECONDS):
                blocked.append(provider)
            else:
                available.append(provider)

        # A provider that answered a probe goes first. Ordering is otherwise
        # the configured preference, and the sort is stable so it is preserved
        # within each group.
        available.sort(
            key=lambda provider: self.health_by_provider[provider.name].probe_ok
            is False
        )
        return available, blocked

    async def probe_providers(self, force: bool = False) -> dict[str, Any]:
        """Check every configured provider with one minimal call.

        Run at startup and periodically. Failures are recorded but never
        raised: a probe that cannot reach anything must not stop the
        application starting, because the cache can still answer the canonical
        questions with no provider at all.

        Args:
            force: Probe even if the interval has not elapsed.

        Returns:
            Provider name to its probe outcome.
        """
        now = time.monotonic()
        if (
            not force
            and self._probed_at is not None
            and now - self._probed_at < settings.PROVIDER_PROBE_INTERVAL_SECONDS
        ):
            return {
                name: {"skipped": "probed recently"}
                for name in self.provider_names
            }

        self._probed_at = now
        results = await asyncio.gather(
            *(self._probe_one(provider) for provider in self.providers),
            return_exceptions=False,
        )
        outcome = dict(results)
        logger.info(
            "provider_probe_completed",
            extra={
                "healthy": [
                    name for name, entry in outcome.items() if entry.get("ok")
                ],
                "unhealthy": [
                    name for name, entry in outcome.items() if not entry.get("ok")
                ],
            },
        )
        return outcome

    async def _probe_one(
        self, provider: LLMProvider
    ) -> tuple[str, dict[str, Any]]:
        """Probe one provider, recording the result in its health.

        Args:
            provider: The provider to check.

        Returns:
            Its name and the probe outcome.
        """
        health = self.health_by_provider[provider.name]
        started = time.perf_counter()
        health.last_probe = datetime.now(timezone.utc).isoformat()
        try:
            await asyncio.wait_for(
                provider.complete(
                    system=PROBE_SYSTEM,
                    user=PROBE_USER,
                    max_tokens=PROBE_MAX_TOKENS,
                ),
                timeout=settings.PROVIDER_PROBE_TIMEOUT_SECONDS,
            )
        except (LLMError, asyncio.TimeoutError, Exception) as error:  # noqa: BLE001
            latency_ms = (time.perf_counter() - started) * 1000
            health.probe_ok = False
            health.probe_latency_ms = round(latency_ms, 1)
            # A failed probe opens the breaker immediately rather than
            # counting towards the threshold: the probe *is* the evidence, and
            # making a user's first request rediscover it defeats the point.
            health.record_failure(
                time.monotonic(),
                f"probe failed: {error}",
                threshold=1,
            )
            logger.warning(
                "provider_probe_failed",
                extra={"provider": provider.name, "reason": str(error)[:200]},
            )
            return provider.name, {"ok": False, "reason": str(error)[:200]}

        latency_ms = (time.perf_counter() - started) * 1000
        health.probe_ok = True
        health.probe_latency_ms = round(latency_ms, 1)
        health.record_success(time.monotonic())
        return provider.name, {"ok": True, "latency_ms": round(latency_ms, 1)}

    async def complete(
        self,
        system: str,
        user: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        exclude: Sequence[str] = (),
        record_success: bool = True,
    ) -> LLMResponse:
        """Complete a prompt, failing over across providers.

        Providers held back by the circuit breaker are skipped, but only while
        another provider might still work. If the breaker has taken everything
        out of rotation they are tried anyway: an open breaker is a prediction
        that a call will fail, and a prediction is not a good enough reason to
        return nothing when trying costs one timeout.

        Args:
            system: System prompt.
            user: User message.
            max_tokens: Maximum tokens to generate. Defaults to settings.
            temperature: Sampling temperature. Defaults to settings.
            exclude: Provider names to skip. Used by the JSON path to retry
                somewhere other than the provider that just produced garbage.
            record_success: Whether a completed HTTP call counts as a success
                for the breaker. The JSON path passes False and records the
                outcome itself: a 200 carrying unparsable text is a transport
                success but a failure of the operation, and recording it here
                would reset the failure streak a moment before the parse error
                increments it, so a provider returning garbage forever would
                never trip the breaker.

        Returns:
            The first successful completion, with ``attempts`` listing every
            provider tried including the one that succeeded.

        Raises:
            LLMError: If no provider is configured, or all of them failed. The
                message names each provider and its reason.
        """
        if not self.providers:
            raise LLMError(
                "no LLM provider is configured; set one of "
                f"{', '.join(f'{n.upper()}_API_KEY' for n in PROVIDER_REGISTRY)}"
            )

        available, blocked = self._selection(exclude)
        if not available and blocked:
            logger.warning(
                "llm_breaker_bypassed",
                extra={"providers": [p.name for p in blocked]},
            )
            available = blocked
            blocked = []
        if not available:
            excluded = ", ".join(exclude) or "none"
            raise LLMError(
                f"no LLM provider is available (excluded: {excluded}; "
                f"configured: {', '.join(self.provider_names)})"
            )

        attempted: list[str] = []
        failures: list[ProviderFailure] = []

        for provider in available:
            attempted.append(provider.name)
            health = self.health_by_provider[provider.name]
            try:
                response = await provider.complete(
                    system=system,
                    user=user,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except LLMError as error:
                health.record_failure(
                    time.monotonic(),
                    str(error),
                    settings.CIRCUIT_BREAKER_THRESHOLD,
                )
                failures.append(
                    ProviderFailure(
                        provider=provider.name,
                        model=provider.model,
                        reason=str(error),
                        attempts=provider.max_retries + 1,
                    )
                )
                logger.warning(
                    "llm_provider_failed",
                    extra={
                        "provider": provider.name,
                        "model": provider.model,
                        "reason": str(error),
                        "consecutive_failures": health.consecutive_failures,
                        "breaker_open": health.is_open(
                            time.monotonic(),
                            settings.CIRCUIT_BREAKER_COOLDOWN_SECONDS,
                        ),
                    },
                )
                continue
            if record_success:
                health.record_success(time.monotonic())
            return replace(response, attempts=list(attempted))

        summary = "; ".join(
            f"{failure.provider} ({failure.model}): {failure.reason}"
            for failure in failures
        )
        skipped = [provider.name for provider in blocked]
        logger.error(
            "llm_all_providers_failed",
            extra={
                "providers": attempted,
                "skipped_by_breaker": skipped,
                "failures": summary,
            },
        )
        raise LLMError(
            f"all {len(failures)} LLM provider(s) failed: {summary}", failures
        )

    async def complete_json(
        self,
        system: str,
        user: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Any:
        """Complete a prompt and parse the response as JSON.

        Args:
            system: System prompt; the JSON instruction is appended to it.
            user: User message.
            max_tokens: Maximum tokens to generate. Defaults to settings.
            temperature: Sampling temperature. Defaults to settings.

        Returns:
            The decoded JSON value.

        Raises:
            LLMJSONError: If the response is not valid JSON.
            LLMError: If every provider failed.
        """
        value, _ = await self.complete_json_with_response(
            system=system, user=user, max_tokens=max_tokens, temperature=temperature
        )
        return value

    async def complete_json_with_response(
        self,
        system: str,
        user: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> tuple[Any, LLMResponse]:
        """Complete a prompt as JSON, also returning the raw response.

        Appends a JSON-only instruction to the system prompt, strips markdown
        fences from the reply and decodes it. Callers that need the token
        counts and provider for a trace use this; callers that only want the
        value use :meth:`complete_json`.

        A response that will not parse fails over to a *different* provider
        rather than being retried on the same one. Malformed structure is a
        property of the model, not of the moment: at temperature zero the same
        model given the same prompt will usually produce the same broken
        output, so retrying it burns the user's time to arrive back where it
        started. Every provider is given one chance before the call is
        abandoned.

        Args:
            system: System prompt; the JSON instruction is appended to it.
            user: User message.
            max_tokens: Maximum tokens to generate. Defaults to settings.
            temperature: Sampling temperature. Defaults to settings.

        Returns:
            The decoded JSON value and the completion that produced it.

        Raises:
            LLMJSONError: If no provider returned valid JSON. The error carries
                the raw text of the last attempt so the caller can see what the
                model actually said.
            LLMError: If every provider failed to respond at all.
        """
        tried: list[str] = []
        last_error: LLMJSONError | None = None

        # One pass per configured provider at most; the exclusion list grows
        # by one each time so a provider is never asked twice.
        for _ in range(max(1, len(self.providers))):
            response = await self.complete(
                system=f"{system}\n\n{JSON_ONLY_INSTRUCTION}",
                user=user,
                max_tokens=max_tokens,
                temperature=temperature,
                exclude=tried,
                record_success=False,
            )
            health = self.health_by_provider[response.provider]
            cleaned = strip_code_fences(response.text)
            try:
                value = json.loads(cleaned)
            except json.JSONDecodeError as error:
                reason = str(error)
            else:
                health.record_success(time.monotonic())
                return value, response

            tried.append(response.provider)
            last_error = LLMJSONError(
                f"{response.provider} ({response.model}) did not return valid "
                f"JSON: {reason}. Raw response: {response.text!r}",
                raw_text=response.text,
            )
            # Malformed JSON is a failure of this provider for this task, so it
            # counts towards the breaker even though the HTTP call succeeded.
            health.record_failure(
                time.monotonic(),
                f"returned unparsable JSON: {reason}",
                settings.CIRCUIT_BREAKER_THRESHOLD,
            )
            logger.warning(
                "llm_json_parse_failed",
                extra={
                    "provider": response.provider,
                    "error": reason,
                    "will_retry_elsewhere": len(tried) < len(self.providers),
                },
            )

        assert last_error is not None  # loop body runs at least once
        logger.error(
            "llm_json_unparsable_everywhere",
            extra={"providers": tried},
        )
        raise last_error

    def health(self) -> dict[str, Any]:
        """Report provider configuration without exposing credentials.

        No part of any API key appears in the output, not even a masked prefix:
        a partial key is still a leaked secret, and health output ends up in
        logs and screenshots.

        Returns:
            Configured provider names, the preference order, the model ids and
            the retry settings.
        """
        now = time.monotonic()
        cooldown = settings.CIRCUIT_BREAKER_COOLDOWN_SECONDS
        healthy = self.healthy_provider_names
        return {
            "providers_configured": self.provider_names,
            "providers_healthy": healthy,
            "providers_in_cooldown": [
                name for name in self.provider_names if name not in healthy
            ],
            "provider_order": list(settings.LLM_PROVIDER_ORDER),
            "primary_provider": healthy[0] if healthy else None,
            "any_configured": bool(self.providers),
            "any_healthy": bool(healthy),
            "provider_health": [
                {
                    "name": health.name,
                    "healthy": not health.is_open(now, cooldown),
                    "consecutive_failures": health.consecutive_failures,
                    "cooldown_remaining_seconds": round(
                        health.cooldown_remaining(now, cooldown), 1
                    ),
                    "last_probe": health.last_probe,
                    "probe_ok": health.probe_ok,
                    "probe_latency_ms": health.probe_latency_ms,
                    # Truncated: a provider error can embed a whole request
                    # body, and health output ends up in logs and screenshots.
                    "last_failure": (
                        health.last_failure_reason[:200]
                        if health.last_failure_reason
                        else None
                    ),
                }
                for health in (
                    self.health_by_provider[name] for name in self.provider_names
                )
            ],
            "models": {
                name: getattr(settings, model_attr)
                for name, (_, _, model_attr) in PROVIDER_REGISTRY.items()
            },
            "timeout_seconds": settings.LLM_TIMEOUT_SECONDS,
            "max_retries": settings.LLM_MAX_RETRIES,
            "breaker_threshold": settings.CIRCUIT_BREAKER_THRESHOLD,
            "breaker_cooldown_seconds": cooldown,
        }


_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """Return the shared client, building it on first use.

    Returns:
        The module-level singleton.
    """
    global _client
    if _client is None:
        _client = LLMClient()
    return _client


def reset_llm_client() -> None:
    """Discard the cached client so the next call rebuilds it.

    Used by tests that change provider credentials between cases.
    """
    global _client
    _client = None
