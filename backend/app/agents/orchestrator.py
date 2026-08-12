"""Orchestrator: runs the pipeline and owns every degradation decision.

The individual agents are deliberately narrow - the planner does not know that
the analyst can fail, and the analyst does not know what the verifier will make
of its rows. All of that judgement lives here, in one place, so the policy is
readable rather than scattered through four files.

Two properties are non-negotiable:

* **The pipeline never raises.** Every path out of :meth:`Orchestrator.run`
  returns a valid :class:`AnalysisResponse`. A caller renders one shape whether
  the run succeeded, partially succeeded, timed out or was refused, and an
  exception can never escape to become a 500 in front of a user.
* **The trace is complete and honest.** Every stage contributes an
  :class:`AgentStep`, including the ones that were skipped and the ones that
  failed. The trace is the evidence that this is genuinely a multi-agent
  system, and evidence that omits the failures is not evidence. A skipped
  verifier appears as SKIPPED with the reason, not as an absence.

The self-healing retry is the one place the orchestrator loops. When
verification fails on an error-severity check, the failing sub-queries are
re-run once with the verifier's own message appended to the analyst's prompt,
and the results are verified again. Exactly once: a model that cannot fix a
stated arithmetic inconsistency on the second attempt is not going to fix it on
the fifth, and the user is waiting.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Final

from app.agents.analyst import SQLAnalystAgent
from app.agents.base import Agent, AgentOutcome
from app.agents.contracts import (
    AgentStatus,
    AgentStep,
    AgentTrace,
    AnalysisPlan,
    AnalysisResponse,
    ChartSpec,
    ChartType,
    Insight,
    QueryIntent,
    QueryResult,
    SubQuery,
    VerificationReport,
    VerificationStatus,
)
from app.agents.insight import (
    InsightAgent,
    InsightBundle,
    build_degraded_insight,
    choose_chart,
)
from app.agents.planner import PlannerAgent
from app.agents.verifier import VerifierAgent, failing_sub_query_ids
from app.config import settings
from app.core.logging import get_logger
from app.semantic.schema import METRIC_DEFINITIONS, TABLE_ALLOWLIST

logger = get_logger(__name__)

# One self-healing attempt. See the module docstring for why it is not more.
MAX_VERIFICATION_RETRIES: Final[int] = 1

# The eight evaluation questions, exposed for the frontend's suggestion chips
# so the UI and the golden answers cannot drift apart.
CANONICAL_QUESTIONS: Final[list[dict[str, str]]] = [
    {
        "id": "q1",
        "question": (
            "What was our total revenue, order count and AOV in the last 3 "
            "months?"
        ),
        "label": "Revenue, orders and AOV",
    },
    {
        "id": "q2",
        "question": (
            "Which stores are performing best and worst by revenue in the last "
            "3 months?"
        ),
        "label": "Best and worst stores",
    },
    {
        "id": "q3",
        "question": "How do our sales channels compare in the last 3 months?",
        "label": "Channel comparison",
    },
    {
        "id": "q4",
        "question": "Which products sell the most in the last 3 months?",
        "label": "Top products",
    },
    {
        "id": "q5",
        "question": "Are any cities showing declining revenue in the last 3 months?",
        "label": "Declining cities",
    },
    {
        "id": "q6",
        "question": "How does weekend trading compare with weekday trading?",
        "label": "Weekend vs weekday",
    },
    {
        "id": "q7",
        "question": "How much do festive periods lift trading versus normal days?",
        "label": "Festive lift",
    },
    {
        "id": "q8",
        "question": (
            "Which stores have consistently declining revenue, and why are they "
            "declining?"
        ),
        "label": "Consistent decliners and why",
    },
]


def _skipped_step(agent_name: str, reason: str) -> AgentStep:
    """Build a SKIPPED trace step.

    A stage that did not run still belongs in the trace; omitting it would
    misrepresent what the system did.

    Args:
        agent_name: The stage that was skipped.
        reason: Why it did not run.

    Returns:
        The step.
    """
    return AgentStep(
        agent_name=agent_name,
        status=AgentStatus.SKIPPED,
        started_at=datetime.now(timezone.utc),
        duration_ms=0.0,
        summary=reason,
        error=None,
        llm_provider=None,
        tokens=None,
    )


def _unsupported_explanation() -> str:
    """Describe what this system can and cannot answer.

    Built from the semantic layer rather than written by hand, so it cannot
    describe a dataset the system no longer has.

    Returns:
        A short paragraph naming the coverage, the metrics and the tables.
    """
    return (
        f"This system answers questions about QuickBite's own sales data "
        f"between {settings.DATA_START_DATE.isoformat()} and "
        f"{settings.DATA_ASOF_DATE.isoformat()}. It can measure "
        f"{', '.join(sorted(METRIC_DEFINITIONS))} broken down by store, city, "
        f"region, channel, product, category, customer segment, day type and "
        f"month, across {len(TABLE_ALLOWLIST)} tables. It holds no data on "
        f"competitors, staff, marketing spend, weather, customer opinions or "
        f"anything after {settings.DATA_ASOF_DATE.isoformat()}, so questions "
        f"about those cannot be answered from this dataset."
    )


class Orchestrator:
    """Runs planner, analyst, verifier and insight as one request."""

    def __init__(
        self,
        planner: PlannerAgent | None = None,
        analyst: SQLAnalystAgent | None = None,
        verifier: VerifierAgent | None = None,
        insight: InsightAgent | None = None,
    ) -> None:
        """Initialise the orchestrator.

        Args:
            planner: Planning agent. Defaults to a new :class:`PlannerAgent`.
            analyst: SQL agent. Defaults to a new :class:`SQLAnalystAgent`.
            verifier: Verification agent. Defaults to a new
                :class:`VerifierAgent`.
            insight: Explanation agent. Defaults to a new
                :class:`InsightAgent`.
        """
        self.planner = planner or PlannerAgent()
        self.analyst = analyst or SQLAnalystAgent()
        self.verifier = verifier or VerifierAgent()
        self.insight = insight or InsightAgent()

    async def run(
        self, question: str, request_id: str | None = None
    ) -> AnalysisResponse:
        """Answer one question end to end.

        Args:
            question: The user's natural-language question.
            request_id: Correlation id for the logs. Generated when omitted.

        Returns:
            A complete response. Never raises; failures are expressed as
            ``answered=False`` with an explanation and a full trace.
        """
        request_id = request_id or uuid.uuid4().hex[:12]
        started = time.perf_counter()
        steps: list[AgentStep] = []
        logger.info(
            "run_started", extra={"request_id": request_id, "question": question}
        )

        try:
            return await self._run(question, request_id, started, steps)
        except Exception as error:  # noqa: BLE001 - the pipeline never raises
            logger.error(
                "run_crashed",
                extra={"request_id": request_id, "error": str(error)},
                exc_info=True,
            )
            return AnalysisResponse.unanswered(
                question=question,
                error=f"The analysis could not be completed: {error}",
                trace=self._build_trace(steps, started),
            )

    async def _run(
        self,
        question: str,
        request_id: str,
        started: float,
        steps: list[AgentStep],
    ) -> AnalysisResponse:
        """Execute the pipeline stages in order.

        Args:
            question: The user's question.
            request_id: Correlation id for the logs.
            started: ``perf_counter`` reading at the start of the run.
            steps: The trace under construction, appended to in place.

        Returns:
            The response.
        """
        # --- 1. Plan --------------------------------------------------
        plan_outcome = await self._with_timeout(
            self.planner, settings.PLANNER_TIMEOUT_SECONDS, question
        )
        steps.append(plan_outcome.step)
        plan = plan_outcome.result
        if plan is None:
            logger.warning(
                "planning_failed",
                extra={"request_id": request_id, "error": plan_outcome.step.error},
            )
            for stage in ("sql_analyst", "verifier", "insight"):
                steps.append(_skipped_step(stage, "Planning did not produce a plan."))
            return AnalysisResponse.unanswered(
                question=question,
                error=(
                    f"The question could not be turned into an analysis plan: "
                    f"{plan_outcome.step.error}"
                ),
                trace=self._build_trace(steps, started),
            )

        # --- 2. Refuse unsupported questions --------------------------
        if plan.intent is QueryIntent.UNSUPPORTED:
            logger.info("question_unsupported", extra={"request_id": request_id})
            return self._unsupported_response(question, plan, steps, started)

        # --- 3. Run the sub-queries -----------------------------------
        results, analyst_step = await self._run_analyst(
            plan, plan.sub_queries, request_id, feedback=None
        )
        steps.append(analyst_step)

        if all(result.error for result in results):
            reasons = "; ".join(
                f"{result.sub_query_id}: {result.error}" for result in results
            )
            logger.error(
                "all_sub_queries_failed",
                extra={"request_id": request_id, "reasons": reasons},
            )
            for stage in ("verifier", "insight"):
                steps.append(_skipped_step(stage, "No query returned any data."))
            response = AnalysisResponse.unanswered(
                question=question,
                error=(
                    f"Every query for this question failed, so there are no "
                    f"numbers to report. {reasons}"
                ),
                trace=self._build_trace(steps, started),
            )
            response.plan = plan
            response.query_results = results
            return response

        partial = [result.sub_query_id for result in results if result.error]
        if partial:
            logger.warning(
                "partial_results",
                extra={"request_id": request_id, "failed": partial},
            )

        # --- 4. Verify, with one self-healing retry -------------------
        verification, results = await self._verify_with_retry(
            plan, results, request_id, steps
        )

        # --- 5. Explain ------------------------------------------------
        bundle = await self._run_insight(plan, results, verification, steps)

        trace = self._build_trace(steps, started)
        logger.info(
            "run_completed",
            extra={
                "request_id": request_id,
                "duration_ms": round(trace.total_duration_ms, 1),
                "tokens": trace.total_tokens,
                "verification": (
                    verification.status.value if verification else "not_run"
                ),
            },
        )
        return AnalysisResponse(
            question=question,
            answered=True,
            plan=plan,
            query_results=results,
            verification=verification,
            insight=bundle.insight,
            chart=bundle.chart,
            trace=trace,
            data_asof=settings.DATA_ASOF_DATE,
            from_cache=False,
            error=None,
        )

    # ------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------

    async def _run_analyst(
        self,
        plan: AnalysisPlan,
        sub_queries: list[SubQuery],
        request_id: str,
        feedback: str | None,
        label: str = "sql_analyst",
    ) -> tuple[list[QueryResult], AgentStep]:
        """Run a set of sub-queries and produce their trace step.

        ``run_many`` is driven directly rather than through
        :meth:`Agent.run`, because one step should describe the whole
        concurrent batch rather than one step per sub-query. Usage is reset
        explicitly so this batch reports only its own tokens.

        Args:
            plan: The plan being executed.
            sub_queries: The sub-queries to run.
            request_id: Correlation id for the logs.
            feedback: Rejection reason to append to the prompt, for a retry.
            label: Agent name recorded in the trace.

        Returns:
            The results and the step describing the batch.
        """
        started_at = datetime.now(timezone.utc)
        started = time.perf_counter()
        self.analyst.reset_usage()
        try:
            results = await asyncio.wait_for(
                self.analyst.run_many(sub_queries, plan, feedback),
                timeout=settings.ANALYST_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            duration_ms = (time.perf_counter() - started) * 1000
            logger.error(
                "analyst_timed_out",
                extra={"request_id": request_id, "timeout": settings.ANALYST_TIMEOUT_SECONDS},
            )
            timed_out = [
                QueryResult(
                    sub_query_id=sub_query.id,
                    sql="",
                    columns=[],
                    rows=[],
                    row_count=0,
                    execution_ms=duration_ms,
                    error=(
                        f"timed out after {settings.ANALYST_TIMEOUT_SECONDS}s"
                    ),
                    attempts=1,
                )
                for sub_query in sub_queries
            ]
            return timed_out, AgentStep(
                agent_name=label,
                status=AgentStatus.FAILED,
                started_at=started_at,
                duration_ms=duration_ms,
                summary=(
                    f"Querying timed out after "
                    f"{settings.ANALYST_TIMEOUT_SECONDS:.0f}s."
                ),
                error=f"timed out after {settings.ANALYST_TIMEOUT_SECONDS}s",
                llm_provider=self.analyst.provider,
                tokens=self.analyst.tokens or None,
            )

        duration_ms = (time.perf_counter() - started) * 1000
        failed = [result.sub_query_id for result in results if result.error]
        total_rows = sum(result.row_count for result in results)
        status = (
            AgentStatus.FAILED
            if failed and len(failed) == len(results)
            else AgentStatus.SUCCEEDED
        )
        summary = (
            f"Ran {len(results)} quer"
            f"{'y' if len(results) == 1 else 'ies'} concurrently, returning "
            f"{total_rows} row{'' if total_rows == 1 else 's'}."
        )
        if failed:
            summary += f" {len(failed)} failed: {', '.join(failed)}."
        return list(results), AgentStep(
            agent_name=label,
            status=status,
            started_at=started_at,
            duration_ms=duration_ms,
            summary=summary,
            error=f"{len(failed)} sub-quer{'y' if len(failed) == 1 else 'ies'} failed"
            if failed
            else None,
            llm_provider=self.analyst.provider,
            tokens=self.analyst.tokens or None,
        )

    async def _verify_with_retry(
        self,
        plan: AnalysisPlan,
        results: list[QueryResult],
        request_id: str,
        steps: list[AgentStep],
    ) -> tuple[VerificationReport | None, list[QueryResult]]:
        """Verify the results, re-running the failing sub-queries once.

        Args:
            plan: The plan being executed.
            results: The results to verify.
            request_id: Correlation id for the logs.
            steps: The trace under construction, appended to in place.

        Returns:
            The final report (None when the verifier itself failed) and the
            results it applies to, which may be the repaired set.
        """
        outcome = await self._with_timeout(
            self.verifier, settings.VERIFIER_TIMEOUT_SECONDS, plan, results
        )
        steps.append(outcome.step)
        report = outcome.result
        if report is None or report.status is not VerificationStatus.FAILED:
            return report, results

        blamed = failing_sub_query_ids(report) or [
            sub_query.id for sub_query in plan.sub_queries
        ]
        retry_targets = [
            sub_query for sub_query in plan.sub_queries if sub_query.id in blamed
        ]
        logger.warning(
            "verification_failed_retrying",
            extra={
                "request_id": request_id,
                "sub_queries": [sub_query.id for sub_query in retry_targets],
                "summary": report.summary,
            },
        )

        feedback = (
            f"The results of this query failed automated verification: "
            f"{report.summary} Rewrite the SQL so that this specific problem "
            f"cannot occur."
        )
        repaired, retry_step = await self._run_analyst(
            plan, retry_targets, request_id, feedback, label="sql_analyst_retry"
        )
        steps.append(retry_step)

        merged = list(results)
        by_id = {result.sub_query_id: index for index, result in enumerate(merged)}
        for result in repaired:
            if result.error is not None:
                # A failed repair must not overwrite a result that at least
                # produced rows.
                continue
            if result.sub_query_id in by_id:
                merged[by_id[result.sub_query_id]] = result
            else:
                merged.append(result)

        recheck = await self._with_timeout(
            self.verifier, settings.VERIFIER_TIMEOUT_SECONDS, plan, merged
        )
        steps.append(recheck.step)
        if recheck.result is None:
            return report, merged
        if recheck.result.status is VerificationStatus.FAILED:
            logger.error(
                "verification_failed_after_retry",
                extra={"request_id": request_id, "summary": recheck.result.summary},
            )
        return recheck.result, merged

    async def _run_insight(
        self,
        plan: AnalysisPlan,
        results: list[QueryResult],
        verification: VerificationReport | None,
        steps: list[AgentStep],
    ) -> InsightBundle:
        """Explain the results, degrading to deterministic prose on failure.

        Args:
            plan: The plan being executed.
            results: The verified results.
            verification: The verifier's report, when it ran.
            steps: The trace under construction, appended to in place.

        Returns:
            The insight bundle; never None, even when the agent failed.
        """
        outcome = await self._with_timeout(
            self.insight,
            settings.INSIGHT_TIMEOUT_SECONDS,
            plan,
            results,
            verification,
        )
        steps.append(outcome.step)
        if outcome.result is not None:
            return outcome.result

        return InsightBundle(
            insight=build_degraded_insight(plan.question, plan, results),
            chart=choose_chart(plan, results),
            degraded=True,
        )

    def _unsupported_response(
        self,
        question: str,
        plan: AnalysisPlan,
        steps: list[AgentStep],
        started: float,
    ) -> AnalysisResponse:
        """Answer a question the dataset cannot support.

        No SQL is generated and no model is asked to explain: the honest reply
        is a statement of scope, and it is written from the semantic layer so
        it is always accurate.

        Args:
            question: The user's question.
            plan: The plan, whose reasoning explains the refusal.
            steps: The trace under construction, appended to in place.
            started: ``perf_counter`` reading at the start of the run.

        Returns:
            A response with ``answered`` false and a useful explanation.
        """
        for stage in ("sql_analyst", "verifier", "insight"):
            steps.append(
                _skipped_step(
                    stage, "The question cannot be answered from this dataset."
                )
            )
        insight = Insight(
            headline=(
                "This question cannot be answered from QuickBite's sales data."
            ),
            narrative=f"{plan.reasoning}\n\n{_unsupported_explanation()}",
            key_findings=[],
            caveats=[],
            recommended_actions=[],
            confidence=0.0,
        )
        return AnalysisResponse(
            question=question,
            answered=False,
            plan=plan,
            query_results=[],
            verification=None,
            insight=insight,
            chart=ChartSpec(
                chart_type=ChartType.NONE, x_field="", y_fields=[], title=question
            ),
            trace=self._build_trace(steps, started),
            data_asof=settings.DATA_ASOF_DATE,
            from_cache=False,
            error=_unsupported_explanation(),
        )

    # ------------------------------------------------------------------
    # Plumbing
    # ------------------------------------------------------------------

    @staticmethod
    async def _with_timeout(
        agent: Agent[Any], timeout: float, *args: Any
    ) -> AgentOutcome[Any]:
        """Run an agent under a deadline.

        :meth:`Agent.run` already converts its own failures into a FAILED
        step, so the only thing that escapes it is a hang. That is converted
        here rather than being allowed to hold the request open.

        Args:
            agent: The agent to run.
            timeout: Seconds to allow.
            *args: Passed through to the agent.

        Returns:
            The agent's outcome, or one carrying a FAILED timeout step.
        """
        started_at = datetime.now(timezone.utc)
        started = time.perf_counter()
        try:
            return await asyncio.wait_for(agent.run(*args), timeout=timeout)
        except asyncio.TimeoutError:
            duration_ms = (time.perf_counter() - started) * 1000
            logger.error(
                "agent_timed_out",
                extra={"agent": agent.name, "timeout": timeout},
            )
            return AgentOutcome(
                result=None,
                step=AgentStep(
                    agent_name=agent.name,
                    status=AgentStatus.FAILED,
                    started_at=started_at,
                    duration_ms=duration_ms,
                    summary=f"{agent.name} timed out after {timeout:.0f}s.",
                    error=f"timed out after {timeout}s",
                    llm_provider=agent.provider,
                    tokens=agent.tokens or None,
                ),
            )

    @staticmethod
    def _build_trace(steps: list[AgentStep], started: float) -> AgentTrace:
        """Aggregate the steps into a trace.

        Args:
            steps: Every step recorded so far.
            started: ``perf_counter`` reading at the start of the run.

        Returns:
            The trace, with totals summed across the steps.
        """
        providers: list[str] = []
        for step in steps:
            if step.llm_provider and step.llm_provider not in providers:
                providers.append(step.llm_provider)
        return AgentTrace(
            steps=list(steps),
            total_duration_ms=(time.perf_counter() - started) * 1000,
            total_tokens=sum(step.tokens or 0 for step in steps),
            providers_used=providers,
        )
