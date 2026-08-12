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
from app.semantic.entities import render_for_prompt, resolve
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
   - An ambiguous superlative is TWO sub-queries, not a guess. "Sells the
     most", "biggest", "best performing" can mean volume or value, and the
     two give different answers: the highest-selling product by units is
     usually not the highest by revenue. Rank both ways and let the answer
     say which is which.

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

   The baseline sub-query must return ONE ROW PER ENTITY with the comparison
   ALREADY COMPUTED as columns, not two result sets to be joined by reading:
       window_revenue, baseline_revenue, delta_abs, delta_pct,
       is_above_baseline   (1 when window_revenue > baseline_revenue, else 0)
   State those column names in the sub-query's purpose. A decline that is
   still above its own prior quarter is a reverting entity, not a
   deteriorating one, and naming it as the top concern is arithmetically
   correct and analytically wrong. Comparing two numbers across fifty rows is
   exactly the step that gets silently mis-read, so SQL must do it and the
   answer must only have to read the result.

4. TRENDS. Any trend or decline question must return the individual monthly values.
   Never return only the first and last month. A metric can fall from start to
   end while rising in the middle; those are different business situations and
   endpoints cannot distinguish them.

5. IDENTIFY THE SET IN SQL. When the question asks WHICH entities did
   something consistently, or whether ANY did - "which stores declined every
   month", "are any cities declining", "did anything grow every month" - one
   sub-query must return exactly that set, computed in SQL with a self-join or
   window functions over the monthly values. A yes/no question needs the set
   just as much as a which question does: "are any cities declining" is
   answered by the list, and an empty list is the answer "no".

   This applies to EVERY dimension - city, store, channel, category, product -
   and to improvement as well as decline. Whenever the question asks which
   members of a dimension declined, fell, dropped, worsened, grew or improved
   over consecutive periods, the plan MUST contain a sub-query whose purpose
   says it identifies exactly those members and computes the test in SQL.
   Word that purpose with both parts - the membership ("identify which cities
   ...") and the direction ("... declined in every consecutive month") - so
   the SQL agent knows which shape to produce. Never leave a membership test
   to be inferred from a raw monthly breakdown: a reader comparing values
   across a table will miss cases, and a question answered that way is wrong
   in a way no arithmetic check can detect. Do not
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

9. DEFAULT RATHER THAN ASK. Most questions have a sensible default and should
   be answered, not queried back. Apply these silently and never ask about
   them:
   - No period given -> {last_3m_start} to {last_3m_end}.
   - Revenue -> always tax-exclusive (net_before_tax).
   - No channel, store, city or segment named -> all of them.
   - "Best", "top", "worst" with no measure named -> revenue.
   A bare month name means the most recent occurrence of that month inside
   the data window, so "in June" is 2026-06 and "last December" is 2025-12.

10. AMBIGUOUS. Set intent="ambiguous" ONLY when guessing would produce a
   misleading answer, and supply `clarification` with one focused question and
   two to four complete, self-contained questions the user could ask instead.
   Do not set sub_queries. Genuine cases are narrow:
   - A named reference that matched several different KINDS of entity, where
     the answers differ (a product category versus one product).
   - A metric word that could mean materially different things and the two
     answers would name different winners.
   Everything in rule 9 is NOT ambiguous. When in doubt, answer with the
   default and record the assumption in reasoning.

11. SHARES AND PERCENTAGES. "What percentage / share / proportion of X"
   requires the part, the whole and the percentage as COLUMNS of one result,
   computed in SQL with a window function:
       SUM(...) AS revenue,
       SUM(SUM(...)) OVER () AS total_revenue,
       100.0 * SUM(...) / SUM(SUM(...)) OVER () AS pct_of_total
   Never return the parts alone and leave the division to be done by reading.
   "Delivery" is not a channel: it means Swiggy and Zomato together, so a
   delivery share must group them and say so.

12. ENTITY DEEP DIVES. "How is ST015 doing?" is not one number. Plan a
   profile: the entity's headline metrics for the window, its month-by-month
   series, and the same metrics for the comparable population so the reader
   can tell good from average. Put the comparison in the SAME row where
   possible - the entity's value, the peer average, and the difference as
   columns.

13. TWO NAMED ENTITIES. A comparison between two named things is ONE
   sub-query grouped by that dimension and filtered to both, not two
   sub-queries to be read side by side. Add the difference as a column when
   there are exactly two.

14. RANKING INSIDE A FILTER. "Best store in Bengaluru" ranks within the
   filter, so the filter belongs in WHERE and the ranking in ORDER BY. Never
   rank the whole population and then hope the filter's members appear.

15. THE DIMENSIONS THAT ARE OFTEN FORGOTTEN. These all exist and questions
   about them are answerable:
   - Time of day -> fact_orders.order_hour (0-23).
   - Day of week -> fact_orders.day_name; weekday/weekend -> day_type.
   - Product mix -> dim_product.category and dim_product.veg_nonveg, joined
     through fact_order_lines.
   - Promotions -> dim_promotion via fact_orders.promo_id, LEFT JOIN always.
     Promotion effectiveness compares promoted with non-promoted orders on
     AOV and units, not on revenue alone: a discount that raises basket size
     can still lower revenue per order.
   - Margin -> line-level est_cogs, so gross_margin and margin_pct come from
     fact_order_lines. This is ESTIMATED cost of goods only; it is not profit,
     because the data holds no rent, staff or overhead.
   - Customer segments -> dim_customer.customer_segment, LEFT JOIN always.
     A large share of orders are anonymous walk-ins with no customer_id; a
     segment breakdown MUST either include them as their own group or state
     what share it excludes. Silently dropping 28% of orders is the single
     easiest way to produce a confidently wrong segment answer.

16. YEAR-ON-YEAR IS NOT AVAILABLE. The data covers {data_start} to {data_end},
   a single twelve-month window. There is no prior year to compare against,
   so "versus last year" cannot be answered for any period. Set
   intent="unsupported" and say so plainly. Do NOT quietly compare against an
   earlier part of the same year and present it as year-on-year: that is the
   kind of answer that is wrong in a way the reader cannot detect.
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
            data_start=settings.DATA_START_DATE.isoformat(),
            data_end=settings.DATA_ASOF_DATE.isoformat(),
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

    def build_user_prompt(self, question: str) -> str:
        """Assemble the user message, with any resolved entities attached.

        Entity resolution happens in code against the real dimension values,
        and the canonical values are handed over as facts. Left to itself a
        model asked about "the Bangalore stores" will filter on the string it
        was given, produce SQL that runs perfectly and returns nothing, and the
        answer will be confidently empty. Resolving here is the same principle
        applied throughout the system: compute what the model would otherwise
        have to infer.

        Args:
            question: The user's natural-language question.

        Returns:
            The user message.
        """
        block = render_for_prompt(resolve(question))
        if not block:
            return f"Question: {question}"
        return f"Question: {question}\n\n{block}"

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
        user = self.build_user_prompt(question)
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
        if result.intent is QueryIntent.AMBIGUOUS:
            options = len(result.clarification.options) if result.clarification else 0
            return (
                f"Found the question ambiguous and prepared {options} "
                f"interpretations to choose between."
            )
        diagnostics = " with diagnostics" if result.requires_diagnostics else ""
        return (
            f"Classified as {result.intent.value}{diagnostics} and planned "
            f"{len(result.sub_queries)} "
            f"{'query' if len(result.sub_queries) == 1 else 'queries'} over "
            f"{result.time_window.label}."
        )
