"""Base class shared by every agent.

The base owns instrumentation so no agent can forget to report itself. Timing,
structured logging, error capture and the :class:`AgentStep` that goes into the
user-visible trace are all produced here; a subclass implements only
:meth:`Agent.execute` and gets the rest.

That split matters because the trace is the system's evidence that it is
genuinely agentic. If producing a step were each agent's own responsibility,
the one agent that forgot would be invisible in the UI, and a failing agent
would be invisible exactly when the trace is most useful.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Generic, TypeVar

from app.agents.contracts import AgentStatus, AgentStep
from app.core.llm import LLMClient, get_llm_client
from app.core.logging import get_logger

logger = get_logger(__name__)

ResultT = TypeVar("ResultT")


class AgentError(RuntimeError):
    """An agent that could not complete its work.

    Attributes:
        agent_name: Which agent failed.
    """

    def __init__(self, agent_name: str, message: str) -> None:
        """Initialise the error.

        Args:
            agent_name: The failing agent.
            message: What went wrong.
        """
        super().__init__(f"{agent_name}: {message}")
        self.agent_name = agent_name


class AgentOutcome(Generic[ResultT]):
    """An agent's result paired with its trace step.

    Attributes:
        result: What the agent produced, or None when it failed.
        step: The trace record, always present.
    """

    def __init__(self, result: ResultT | None, step: AgentStep) -> None:
        """Initialise the outcome.

        Args:
            result: The agent's output, or None on failure.
            step: The trace record for this run.
        """
        self.result = result
        self.step = step

    @property
    def succeeded(self) -> bool:
        """Whether the agent completed its work.

        Returns:
            True when a result was produced.
        """
        return self.step.status is AgentStatus.SUCCEEDED


class Agent(ABC, Generic[ResultT]):
    """One step of the analysis pipeline.

    Subclasses implement :meth:`execute` and set :attr:`name`; the base wraps
    every run with timing, logging and trace production.
    """

    #: Name shown in the trace and in logs. Overridden by every subclass.
    name: str = "agent"

    def __init__(self, llm: LLMClient | None = None) -> None:
        """Initialise the agent.

        Args:
            llm: LLM client to use. Defaults to the shared singleton; tests
                inject a client wired to a mock transport.
        """
        self.llm = llm or get_llm_client()
        self._provider: str | None = None
        self._tokens: int = 0

    @property
    def provider(self) -> str | None:
        """The provider that served the most recent model call.

        Returns:
            The provider name, or None when no call has been made.
        """
        return self._provider

    @property
    def tokens(self) -> int:
        """Tokens consumed since usage was last reset.

        Returns:
            The accumulated token count.
        """
        return self._tokens

    def reset_usage(self) -> None:
        """Clear the recorded provider and token spend.

        :meth:`run` calls this so every step reports only its own usage.
        Callers that drive an agent outside :meth:`run` - the orchestrator
        invokes :meth:`SQLAnalystAgent.run_many` directly - must call it
        themselves, or one request's tokens leak into the next request's trace.
        """
        self._provider = None
        self._tokens = 0

    def record_usage(self, provider: str, tokens: int) -> None:
        """Record the provider and token spend of one model call.

        Subclasses call this after each completion so the trace reports real
        usage rather than an estimate. Calling it more than once accumulates,
        which is what a retry should do.

        Args:
            provider: The provider that served the call.
            tokens: Total tokens consumed by the call.
        """
        self._provider = provider
        self._tokens += tokens

    @abstractmethod
    async def execute(self, *args: Any, **kwargs: Any) -> ResultT:
        """Do the agent's work.

        Args:
            *args: Agent-specific positional arguments.
            **kwargs: Agent-specific keyword arguments.

        Returns:
            The agent's output.

        Raises:
            Exception: Any failure; the base class converts it into a failed
                trace step.
        """

    @abstractmethod
    def summarize(self, result: ResultT) -> str:
        """Describe what this run decided, in one sentence.

        Written for a user watching the trace, not for a log file.

        Args:
            result: The agent's output.

        Returns:
            A one-sentence summary.
        """

    async def run(self, *args: Any, **kwargs: Any) -> AgentOutcome[ResultT]:
        """Execute the agent with full instrumentation.

        Never raises: a failure becomes an :class:`AgentOutcome` with a FAILED
        step and a null result, so the orchestrator decides how to degrade
        rather than being interrupted by an exception.

        Args:
            *args: Passed through to :meth:`execute`.
            **kwargs: Passed through to :meth:`execute`.

        Returns:
            The result and its trace step.
        """
        self.reset_usage()
        started_at = datetime.now(timezone.utc)
        started = time.perf_counter()

        logger.info("agent_started", extra={"agent": self.name})
        try:
            result = await self.execute(*args, **kwargs)
        except Exception as error:  # noqa: BLE001 - failure must reach the trace
            duration_ms = (time.perf_counter() - started) * 1000
            logger.error(
                "agent_failed",
                extra={
                    "agent": self.name,
                    "duration_ms": round(duration_ms, 1),
                    "error": str(error),
                },
                exc_info=True,
            )
            return AgentOutcome(
                result=None,
                step=AgentStep(
                    agent_name=self.name,
                    status=AgentStatus.FAILED,
                    started_at=started_at,
                    duration_ms=duration_ms,
                    summary=f"{self.name} failed: {error}",
                    error=str(error),
                    llm_provider=self._provider,
                    tokens=self._tokens or None,
                ),
            )

        duration_ms = (time.perf_counter() - started) * 1000
        summary = self.summarize(result)
        logger.info(
            "agent_succeeded",
            extra={
                "agent": self.name,
                "duration_ms": round(duration_ms, 1),
                "provider": self._provider,
                "tokens": self._tokens,
            },
        )
        return AgentOutcome(
            result=result,
            step=AgentStep(
                agent_name=self.name,
                status=AgentStatus.SUCCEEDED,
                started_at=started_at,
                duration_ms=duration_ms,
                summary=summary,
                error=None,
                llm_provider=self._provider,
                tokens=self._tokens or None,
            ),
        )
