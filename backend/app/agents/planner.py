"""Planner agent: turns a natural-language question into a validated plan.

Planning happens before any SQL exists, which makes it the cheapest place to
catch a misunderstanding. A plan that names a metric the semantic layer does not
define is rejected here with a message the model can act on, rather than
becoming a query that runs successfully and returns nothing.

Two rules in the prompt carry most of the weight for answer quality:

* **Decomposition.** A question that asks *why* cannot be answered by measuring
  the thing that changed. Explaining a change means decomposing it - by channel,
  by month, by order count versus basket size, and against the prior comparable
  period - so a diagnostic plan is required to add those sub-queries.
* **Never only the endpoints.** A trend question must return every monthly
  value, because a metric can fall from start to end while rising in the middle.
  Those are different business situations and an endpoint comparison cannot tell
  them apart.
"""

from __future__ import annotations

import json
from typing import Any, Final

from pydantic import ValidationError

from app.agents.base import Agent, AgentError
from app.agents.contracts import AnalysisPlan, QueryIntent
from app.config import settings
from app.core.logging import get_logger
from app.semantic.schema import METRIC_DEFINITIONS, get_schema_context

logger = get_logger(__name__)

# One retry, with the validation error appended. A second failure means the
# model is not going to produce a valid plan for this question and pretending
# otherwise just spends tokens.
MAX_PLANNING_ATTEMPTS: Final[int] = 2

PLANNING_RULES: Final[str] = """
PLANNING RULES

1. TIME. Resolve every relative expression against the fixed anchor above,
   never against the real calendar. "Last 3 months" is
   {last_3m_start} to {last_3m_end}. Emit absolute dates only.

2. DECOMPOSITION. One sub-query per distinct thing the question asks for.
   - "revenue, orders and AOV" is ONE sub-query: three metrics, one grouping.
   - "top 5 AND bottom 5 stores" is TWO sub-queries: different orderings.
   - "compare X and Y" is usually one sub-query grouped by the dimension.

3. DIAGNOSTIC QUESTIONS. Set requires_diagnostics=true whenever the question
   asks why, asks for a reason or cause, or asks about a decline, a drop or a
   problem. Measuring the change is not explaining it, so when it is true you
   MUST add sub-queries that decompose the change:
   - by channel (is the fall concentrated in one channel?)
   - by month (is it steady or a single bad month?)
   - order count versus AOV (fewer customers, or smaller baskets?)
   - against the prior comparable period (is this a break in trend, or a
     return to normal after an unusually strong period?)
   Set time_window.comparison_start and comparison_end for that last one.
   The baseline sub-query must return ONE ROW PER ENTITY carrying the window
   total, the prior-period total and the change between them, aggregated in
   SQL. Do not return the prior months as raw rows: a decline that is still
   above its own prior quarter is a reverting store, not a deteriorating one,
   and that distinction is lost if it has to be worked out by adding up
   monthly rows by eye.

4. TRENDS. Any trend or decline question must return the individual monthly values.
   Never return only the first and last month. A metric can fall from start to
   end while rising in the middle; those are different business situations and
   endpoints cannot distinguish them.

5. IDENTIFY THE SET IN SQL. When the question asks WHICH entities did
   something consistently - declined every month, grew every month, stayed
   below a threshold - one sub-query must return exactly that set, computed in
   SQL with a self-join or window functions over the monthly values. Do not
   return every entity's monthly series and leave the qualifying set to be
   worked out by reading it. Fifty stores across three months is a hundred and
   fifty numbers, and a reader comparing them by eye will miss some. Keep the
   full monthly series as a separate sub-query, per rule 4, so the answer can
   still show the shape of each decline.

6. UNSUPPORTED. If the question cannot be answered from this data - it asks
   about competitors, weather, staff, marketing spend, customer opinions, or
   anything in the future - set intent="unsupported", confidence below 0.3,
   and explain in reasoning what data would be needed. Do NOT invent a plan
   for a question the data cannot answer.

7. METRICS. The metrics list may contain ONLY these names: {metric_names}.
   Any other name is invalid and the plan will be rejected.

8. BASELINES. When the question implies a comparison ("is it down?",
   "better than before?"), set the comparison window in time_window.
"""


class PlannerAgent(Agent[AnalysisPlan]):
    """Produces a validated :class:`AnalysisPlan` from a question."""

    name = "planner"

    def build_system_prompt(self) -> str:
        """Assemble the planner's system prompt.

        The schema context carries the time anchor, the tables, the metric
        catalogue and the business rules, so the planner and the SQL agent
        cannot develop different ideas about what exists.

        Returns:
            The full system prompt.
        """
        schema = AnalysisPlan.model_json_schema()
        rules = PLANNING_RULES.format(
            last_3m_start=settings.LAST_3M_START.isoformat(),
            last_3m_end=settings.LAST_3M_END.isoformat(),
            metric_names=", ".join(sorted(METRIC_DEFINITIONS)),
        )
        example = AnalysisPlan.model_config["json_schema_extra"]["example"]

        return (
            "You are the planning agent for a QSR business analytics system. "
            "You turn a business question into a structured plan that other "
            "agents execute. You do not write SQL.\n\n"
            f"{get_schema_context()}\n"
            f"{rules}\n\n"
            "OUTPUT\n"
            "Return a single JSON object matching this schema:\n"
            f"{json.dumps(schema, separators=(',', ':'))}\n\n"
            "A worked example of a good plan:\n"
            f"{json.dumps(example, indent=2)}\n"
        )

    async def execute(self, question: str) -> AnalysisPlan:
        """Plan how to answer one question.

        Args:
            question: The user's natural-language question.

        Returns:
            A validated plan.

        Raises:
            AgentError: If the model cannot produce a valid plan within
                :data:`MAX_PLANNING_ATTEMPTS`.
        """
        system = self.build_system_prompt()
        user = f"Question: {question}"
        last_error: str | None = None

        for attempt in range(1, MAX_PLANNING_ATTEMPTS + 1):
            prompt = user
            if last_error is not None:
                prompt = (
                    f"{user}\n\n"
                    f"Your previous plan was rejected by schema validation:\n"
                    f"{last_error}\n\n"
                    f"Produce a corrected plan that fixes exactly that problem."
                )

            payload, response = await self.llm.complete_json_with_response(
                system=system,
                user=prompt,
                temperature=0.0,
            )
            self.record_usage(
                response.provider, response.input_tokens + response.output_tokens
            )

            try:
                plan = self._validate(payload, question)
            except ValidationError as error:
                last_error = self._format_validation_error(error)
                logger.warning(
                    "plan_validation_failed",
                    extra={
                        "attempt": attempt,
                        "error": last_error,
                        "question": question,
                    },
                )
                continue

            logger.info(
                "plan_created",
                extra={
                    "intent": plan.intent.value,
                    "metrics": plan.metrics,
                    "sub_queries": len(plan.sub_queries),
                    "requires_diagnostics": plan.requires_diagnostics,
                    "attempt": attempt,
                },
            )
            return plan

        raise AgentError(
            self.name,
            f"could not produce a valid plan after {MAX_PLANNING_ATTEMPTS} "
            f"attempts; last validation error: {last_error}",
        )

    def _validate(self, payload: Any, question: str) -> AnalysisPlan:
        """Coerce the model's JSON into an :class:`AnalysisPlan`.

        The user's original question is restored from the caller rather than
        trusted from the payload, so a model that paraphrases the question
        cannot change what the answer claims to be about.

        Args:
            payload: The decoded model output.
            question: The user's original question.

        Returns:
            The validated plan.

        Raises:
            ValidationError: If the payload does not satisfy the contract.
        """
        if not isinstance(payload, dict):
            raise ValidationError.from_exception_data(
                "AnalysisPlan",
                [
                    {
                        "type": "dict_type",
                        "loc": (),
                        "input": payload,
                    }
                ],
            )
        data = dict(payload)
        data["question"] = question
        return AnalysisPlan.model_validate(data)

    @staticmethod
    def _format_validation_error(error: ValidationError) -> str:
        """Render a validation error compactly enough to put in a prompt.

        Args:
            error: The pydantic failure.

        Returns:
            One line per problem, naming the field and the reason.
        """
        return "; ".join(
            f"{'.'.join(str(part) for part in item['loc']) or 'root'}: {item['msg']}"
            for item in error.errors()
        )

    def summarize(self, result: AnalysisPlan) -> str:
        """Describe the plan for the trace.

        Args:
            result: The plan produced.

        Returns:
            A one-sentence summary.
        """
        if result.intent is QueryIntent.UNSUPPORTED:
            return "Determined the question cannot be answered from this dataset."
        diagnostics = " with diagnostics" if result.requires_diagnostics else ""
        return (
            f"Classified as {result.intent.value}{diagnostics} and planned "
            f"{len(result.sub_queries)} "
            f"{'query' if len(result.sub_queries) == 1 else 'queries'} over "
            f"{result.time_window.label}."
        )
