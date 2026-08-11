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


class LLMClient:
    """Calls LLM providers in preference order, failing over on error."""

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

    @property
    def provider_names(self) -> list[str]:
        """Names of the configured providers, in preference order.

        Returns:
            The provider names that will actually be tried.
        """
        return [provider.name for provider in self.providers]

    async def complete(
        self,
        system: str,
        user: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Complete a prompt, failing over across providers.

        Args:
            system: System prompt.
            user: User message.
            max_tokens: Maximum tokens to generate. Defaults to settings.
            temperature: Sampling temperature. Defaults to settings.

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

        attempted: list[str] = []
        failures: list[ProviderFailure] = []

        for provider in self.providers:
            attempted.append(provider.name)
            try:
                response = await provider.complete(
                    system=system,
                    user=user,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except LLMError as error:
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
                    },
                )
                continue
            return replace(response, attempts=list(attempted))

        summary = "; ".join(
            f"{failure.provider} ({failure.model}): {failure.reason}"
            for failure in failures
        )
        logger.error(
            "llm_all_providers_failed",
            extra={"providers": attempted, "failures": summary},
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

        Appends a JSON-only instruction to the system prompt, strips markdown
        fences from the reply and decodes it.

        Args:
            system: System prompt; the JSON instruction is appended to it.
            user: User message.
            max_tokens: Maximum tokens to generate. Defaults to settings.
            temperature: Sampling temperature. Defaults to settings.

        Returns:
            The decoded JSON value.

        Raises:
            LLMJSONError: If the response is not valid JSON. The error carries
                the raw text so the caller can see what the model actually said.
            LLMError: If every provider failed.
        """
        response = await self.complete(
            system=f"{system}\n\n{JSON_ONLY_INSTRUCTION}",
            user=user,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        cleaned = strip_code_fences(response.text)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as error:
            logger.error(
                "llm_json_parse_failed",
                extra={"provider": response.provider, "error": str(error)},
            )
            raise LLMJSONError(
                f"{response.provider} ({response.model}) did not return valid JSON: "
                f"{error}. Raw response: {response.text!r}",
                raw_text=response.text,
            ) from error

    def health(self) -> dict[str, Any]:
        """Report provider configuration without exposing credentials.

        No part of any API key appears in the output, not even a masked prefix:
        a partial key is still a leaked secret, and health output ends up in
        logs and screenshots.

        Returns:
            Configured provider names, the preference order, the model ids and
            the retry settings.
        """
        return {
            "providers_configured": self.provider_names,
            "provider_order": list(settings.LLM_PROVIDER_ORDER),
            "primary_provider": self.provider_names[0] if self.providers else None,
            "any_configured": bool(self.providers),
            "models": {
                name: getattr(settings, model_attr)
                for name, (_, _, model_attr) in PROVIDER_REGISTRY.items()
            },
            "timeout_seconds": settings.LLM_TIMEOUT_SECONDS,
            "max_retries": settings.LLM_MAX_RETRIES,
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
