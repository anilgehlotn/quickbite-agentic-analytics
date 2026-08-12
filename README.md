# QuickBite Agentic Analytics

Natural-language business analytics over a QSR sales dataset, answered by four
cooperating agents that plan the analysis, write and execute SQL, verify the
numbers arithmetically, and explain what the result means.

[![ci](https://github.com/anilgehlotn/quickbite-agentic-analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/anilgehlotn/quickbite-agentic-analytics/actions/workflows/ci.yml)

- **Live application:** _(add the Vercel URL here)_
- **API:** _(add the Render URL here)_ — try `/api/health` and `/api/verify`
- **Demo video:** _(add the link here)_

---

## What it does

You ask a question in plain English. A planner interprets it and decides what
to measure, a SQL analyst writes and runs the queries against a SQLite star
schema, a verifier checks the returned numbers against each other before anyone
sees them, and an insight agent explains what happened and what to do about it.

Every answer ships with its evidence: the exact SQL that ran, the verification
checks that passed or failed, the duration and token cost of each agent, and
the plan the system worked from. The interface puts that trace beside the
answer rather than behind a link, because a claim that four agents collaborated
is worth nothing without the record.

The dataset is a fixed historical extract covering 1 August 2025 to 31 July
2026: 20,000 orders across 50 stores, 8 cities and 4 sales channels.

---

## Architecture

```mermaid
flowchart TD
    Q["Question<br/>(natural language)"] --> API["FastAPI /api/ask"]
    API --> CACHE{"Cached?"}
    CACHE -->|hit| RESP
    CACHE -->|miss| ORCH["Orchestrator<br/>timeouts · degradation · trace"]

    ORCH --> ER["Entity resolver<br/>ST7 → ST007 · Bangalore → Bengaluru"]
    ER --> P["1 · Planner"]
    P -->|"intent: unsupported"| SCOPE["Scope reply<br/>+ adjacent questions"]
    P -->|"intent: ambiguous"| CLR["Clarifying question<br/>+ interpretations"]
    SCOPE --> RESP
    CLR --> RESP
    P -->|AnalysisPlan| A["2 · SQL Analyst"]
    A -->|QueryResult| V["3 · Verifier"]
    V -->|VerificationReport| I["4 · Insight"]
    I -->|Insight + ChartSpec| RESP["AnalysisResponse<br/>answer · trace · SQL · checks"]

    SEM["Semantic layer<br/>schema · metrics · business rules · time anchor"] -.-> P
    SEM -.-> A
    GUARD["SQL Guard<br/>sqlglot AST · allowlist · LIMIT"] --> DB[("SQLite<br/>read-only URI")]
    A --> GUARD

    V -->|failed| A

    classDef agent fill:#f5eff4,stroke:#4c2a4d,color:#1a1917
    classDef infra fill:#f4f3ef,stroke:#8c877d,color:#57534e
    class P,A,V,I agent
    class SEM,GUARD,DB,CACHE,ORCH,API,ER,SCOPE,CLR infra
```

The two dotted edges matter as much as the solid ones. The planner and the SQL
analyst read the **same** semantic layer, so they cannot develop different
ideas about what a column means. And every query the analyst writes passes
through the **SQL guard** before it reaches the database — the analyst has no
path to SQLite that skips it.

The edge from the verifier back to the analyst is the self-healing retry: when
an arithmetic check fails, the failing sub-queries are re-run once with the
verifier's own message appended to the prompt, then verified again.

The two branches out of the planner are the paths that produce an answer
without running any SQL. A question the data cannot support returns a statement
of scope naming what is missing, plus two or three adjacent questions that
*are* answerable. A question that is genuinely ambiguous returns one focused
clarifying question with two to four complete interpretations, each
submittable in a click.

---

## The agents

| Agent | Role | Produces |
|---|---|---|
| **Planner** | Interprets the question, resolves relative dates against the fixed anchor, classifies intent, and decomposes the work into sub-queries. Rejects questions the data cannot answer instead of inventing a plan. | `AnalysisPlan` — intent, time window, metrics, sub-queries, reasoning, confidence |
| **SQL Analyst** | Writes one SQLite statement per sub-query, validates it through the guard, executes it, and repairs it once from the database's own error message. Sub-queries run concurrently. | `QueryResult[]` — exact SQL, columns, rows, timing, attempt count |
| **Verifier** | Runs **13 deterministic checks** in pure Python, then escalates to a model only for genuine ambiguity. A model can add a warning; it can never decide the verdict. | `VerificationReport` — status and every individual check |
| **Insight** | Explains the result as a senior analyst would: decomposes changes into volume and basket size, separates level from trend, distinguishes deterioration from reversion, and states caveats unprompted. | `Insight` + `ChartSpec` — headline, narrative, findings, caveats, actions |

The **Orchestrator** owns everything the agents do not: per-agent timeouts,
partial-failure handling, the self-healing retry, and the guarantee that the
pipeline never raises. Every path out of it returns a valid response with a
complete trace, including the failures.

---

## Key design decisions

### Deterministic verification runs before model verification

Asking a model whether its own pipeline produced a good answer is weak
evidence: it is the same class of system that produced the answer, it has no
independent access to the data, and it is agreeable by construction. Asserting
that a channel breakdown sums to its total, that AOV equals revenue over
orders, or that no revenue figure exceeds the annual total of the entire
dataset is strong evidence — it either holds or it does not, and when it fails
the failure names the number that is wrong.

So the arithmetic runs first, in code. If any error-severity check fails, the
model is never consulted. When it is consulted, its verdict is hardcoded to
warning severity: it can add a caveat, never overturn a sum. A test asserts the
mock was not called after a deterministic failure, because an untested claim
about ordering is just a comment.

### Qualifying sets are computed in SQL, not inferred by the model

"Which stores declined every month?" was answered wrongly, repeatedly, while
every individual number in the answer was correct. Handed 150 rows of monthly
revenue, the model would enumerate three stores whose figures declined
monotonically and then conclude that none had.

The fix was structural rather than another instruction: the plan now contains a
sub-query that returns one row per store with the monthly values pivoted into
columns, a computed `is_strictly_declining` flag and the endpoint change. The
model reads a flag instead of comparing numbers across rows. The same applies
to the prior-period baseline, which is emitted as
`window_revenue, baseline_revenue, delta_abs, delta_pct, is_above_baseline` on
the same row — because a store can be declining every month and still be above
its own historical run rate, and naming that store as the top concern is
arithmetically correct and analytically wrong.

Where a comparison is still at risk of being misread, the insight layer is
handed the grouping precomputed in Python rather than the rows to derive it
from. The principle throughout: **never ask the model to derive a comparison it
could read.**

### The data anchor is fixed, not taken from the clock

The dataset ends on 31 July 2026. The system's notion of "today" is the
constant `DATA_ASOF_DATE = 2026-07-31`, and every relative expression resolves
against it. Reading the clock instead would place "last 3 months" outside the
data and return zero rows for every time-bounded question — the most damaging
possible failure, because empty results look like findings. The anchor lives in
`config.py`, is injected into every agent prompt, and the SQL rules forbid
`DATE('now')` and `CURRENT_DATE` explicitly.

### Order grain is canonical for revenue

The source workbook has a reconciliation defect: line-level values do not sum
exactly to their order headers. Across the full year the gap is about 0.1% of
revenue, but the affected orders are concentrated in the most recent months, so
inside the three-month evaluation window it widens to about **0.44%** — which
makes it matter most for exactly the questions people ask most.

`fact_orders` is therefore canonical for revenue, order counts and AOV.
`fact_order_lines` is used only for SKU, product, category and margin
questions, where line identity is the only place the information exists. The
quality gate reports the variance with a per-month breakdown and escalates from
warning to error past a threshold, so the defect is quantified and bounded
rather than hidden.

### Why a semantic layer exists

Without one, every agent invents its own vocabulary and the failures are
plausible rather than obvious: a query that uses `net_revenue` instead of
`net_before_tax` overstates revenue by exactly the tax rate and looks
completely normal. The semantic layer is a single rendered description of the
tables, the ten canonical metrics with their SQL, ten business rules and the
time anchor, and it is generated from one structured specification so the full
and compact renderings cannot drift apart. Both the planner and the analyst
read it; the guard's table allowlist is derived from the same source.

### Entity resolution happens in code, before the planner sees the question

The likeliest cause of a confidently wrong answer is a reviewer naming
something in a way the data does not use. The database holds `ST007`, but a
person types `ST7`; it holds `Bengaluru`, but a person types `Bangalore`; it
holds `Veg Burger 5`, but a person types `veg burgers`. A model asked to invent
a filter value from any of those writes SQL that runs perfectly and returns
nothing.

So `app/semantic/entities.py` loads 211 distinct values across 16 dimensions at
startup and resolves references deterministically, in decreasing order of
certainty: exact, alias, identifier normalisation, word-boundary substring,
then fuzzy matching at a 0.84 threshold restricted to single tokens. The
canonical values and the columns they live in are handed to the planner as
facts. It covers store and SKU ids, store names, cities (17 aliases), regions,
formats, categories, veg/non-veg, channels, customer segments, promotions,
festive periods and day names.

Three properties matter more than recall, and each is tested:

- **It never invents.** `ST999` resolves to nothing, not to the nearest store.
- **It never widens.** `veg burger` is a substring of `Non-Veg Burger 2`, so
  matches must begin at a word boundary — otherwise a question about veg
  burgers is answered with non-veg data.
- **Ambiguity survives.** `pizzas` matches both a category and several product
  names, and that is reported rather than decided.

### Asking beats guessing, but only when guessing would mislead

The planner can return an `ambiguous` intent carrying one focused clarifying
question and two to four complete interpretations. The API returns this as a
valid response with `status: "clarification_needed"`, and the frontend renders
the interpretations as one-click follow-ups rather than as an error.

It is deliberately conservative. Most questions have a sensible default and
should be answered: no period given means the last three months, revenue means
tax-exclusive, no channel named means all channels. Clarification is reserved
for cases where guessing wrong produces a misleading answer — a term matching
several *kinds* of entity, or a metric word whose two readings would name
different winners. A system that asks about everything is more annoying than
one that occasionally assumes.

Every response now carries a `status` field — `answered`,
`clarification_needed`, `unsupported` or `failed` — because a clarification
request and a genuine failure are both "not answered" and mean opposite things
to a reader.

### Why provider failover and answer caching exist

This is a public URL in front of paid APIs, opened by people the author will
never meet, possibly after the keys have expired. Both mechanisms exist for
that reality rather than for elegance.

Failover tries each configured provider in order, retrying transient failures
(429, 5xx, timeouts) within a provider and moving on immediately for permanent
ones (401, 400, 403, 404). When a live run exhausted all four, it produced one
structured error naming all four reasons instead of a traceback — and exposed
two stale model IDs of mine in the process.

Three mechanisms sit on top of plain failover, each addressing a way the naive
version wastes a user's time:

- **A circuit breaker.** After two consecutive failures a provider is skipped
  for a 120-second cooldown, then tried again. Without it, every request pays
  the full timeout of a provider whose key expired months ago. When the breaker
  has taken *everything* out of rotation it is bypassed and the providers are
  tried anyway — an open breaker predicts failure, and a prediction is not a
  good enough reason to return nothing.
- **A startup liveness probe.** One minimal call per provider at boot, run in
  the background so a slow provider cannot delay the port opening. Providers
  that answered are ordered first, and a failed probe opens the breaker
  immediately. On the current deployment this identified in about four seconds
  that three of the four providers had no credit remaining.
- **Per-agent failover with independent state.** Each agent fails over on its
  own; the planner succeeding on one provider and the analyst failing on it
  does not restart the pipeline. A response that will not parse as JSON fails
  over to a *different* provider rather than retrying the same one, because
  malformed structure is a property of the model rather than of the moment.

`GET /api/providers` reports which providers are configured, which are healthy,
which are in cooldown and for how long, and each one's last probe result.
`POST /api/providers/probe` re-probes on demand, so a provider whose key was
fixed comes back without a redeploy. Neither payload contains credential
material.

The cache is not a latency optimisation. The eight evaluation questions are
warmed by `scripts/warm_cache.py` and the resulting file is committed, so those
answers are served from disk with **no provider involved at all**, complete
with their original traces. `scripts/check_offline.py` proves it in CI by
clearing every key and asserting all eight still return a headline, narrative,
executed SQL, verification report and four-agent trace.

---

## Data model

A star schema built from the workbook by `app/etl/build_db.py`:

| Table | Rows | Notes |
|---|---|---|
| `dim_store` | 50 | City, region, format; all active |
| `dim_product` | 30 | Category, veg flag, price, cost |
| `dim_customer` | 5,000 | Segment; **5,664 of 20,000 orders (28%) have no customer** |
| `dim_promotion` | 6 | **Only 840 orders (4%) carry a promotion** |
| `dim_calendar` | 365 | Day type, festive period, month key |
| `fact_orders` | 20,000 | **Canonical grain for revenue, orders and AOV** |
| `fact_order_lines` | 49,834 | Line grain; SKU, quantity, margin |
| `mart_store_month` | 600 | 50 stores × 12 months, pre-aggregated |
| `mart_city_month` | 96 | 8 cities × 12 months |
| `mart_channel_month` | 48 | 4 channels × 12 months |

Calendar attributes (`month_key`, `day_type`, `festive_period`, `is_weekend`)
are denormalised onto `fact_orders`, so weekend, festive and monthly analysis
needs no join. Dimensions are always LEFT JOINed: an inner join to
`dim_customer` silently drops more than a quarter of all revenue and the result
still looks plausible.

Revenue means `net_before_tax` throughout. `net_revenue` includes the 5% tax
and overstates revenue by exactly that much.

The database is committed to the repository. It deploys with no infrastructure
to provision and no build step on the host.

---

## Testing

**813 tests**, all passing without any API key.

### Ground truth is computed independently

`scripts/compute_golden_answers.py` answers the eight evaluation questions with
pandas, reading the **original Excel workbook** — not the SQLite database. That
independence is the point: if the ground truth and the agents both read the
same SQL layer, a bug in that layer is invisible to both. The two paths share
only `config.py`.

`tests/test_golden.py` then cross-checks them, asserting that the pandas
figures and equivalent SQL queries agree to within 1 INR across all 50 stores,
4 channels, 24 city-months, 12 months and both marts.

### The end-to-end suite

`tests/test_golden_e2e.py` runs the real orchestrator over the real database
for all eight questions and asserts against ground truth: exact top-five and
bottom-five store IDs in rank order, all four channels with matching revenue,
the top SKUs by both quantity and revenue, the nine consistently declining
stores, and the four-versus-five split between stores above and below their own
baseline.

Two assertions are hallucination tests rather than correctness tests: question
five must report plainly that **no** city declined monotonically (the
temptation is to name one anyway), and question eight must not lead its
recommendations with the steepest decliner, because that store is above its own
prior quarter. A universal check asserts that no figure appears in any
narrative that is absent from the query results.

### Replay and live mode

By default the suite runs in **replay** mode: the model is replaced by a fake
that returns the plan, SQL and narrative the live system actually produced,
taken from the warmed cache. Everything downstream is real — the guard
validates, SQLite executes, the verifier checks, and the assertions compare to
ground truth. No network, no cost, and the fixtures cannot drift from reality
because they *are* what the system did.

Setting `QUICKBITE_E2E_LIVE=1` calls the configured provider instead. Replay
mode's honest limitation is that it never asks a model anything, so it cannot
catch a prompt regression; it catches regressions in everything else, which is
the part that must never break silently.

---

## Running locally

Requires Python 3.11+ and Node 20+.

```bash
# Backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m app.etl.build_db          # builds data/quickbite.db from the workbook
python -m app.etl.quality_checks    # 36 checks; exits non-zero on error
uvicorn app.main:app --reload       # http://localhost:8000
```

```bash
# Frontend, in a second terminal
cd frontend
npm install
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
npm run dev                         # http://localhost:3000
```

Open http://localhost:3000 and click any suggested question. **No API key is
required for those eight** — they are served from the committed cache. To ask
new questions, copy `backend/.env.example` to `backend/.env` and add at least
one provider key.

Useful commands:

```bash
python -m pytest tests/ -q          # the full suite
python scripts/check_offline.py     # proves the cache answers with no keys
python scripts/warm_cache.py        # re-warms the cache (needs a key)
python scripts/keepalive.py --url https://your-service.onrender.com
```

---

## Security

Agent-generated SQL reaching a database is the obvious risk in a system like
this, and prompt injection through a public input is a real threat rather than
a theoretical one. Six layers stand between the model and the data:

1. **Parse-based validation with sqlglot, never regex.** Regex on SQL loses to
   comments, string literals and whitespace tricks. `DROP/**/TABLE/**/x` is
   caught because the AST node is a `Drop`, and a query containing the string
   `'DROP TABLE'` in a WHERE clause is correctly *allowed*.
2. **Exactly one statement.** Stacking behind a comment fails on the statement
   count, not on pattern matching.
3. **Reads only.** `exp.Query` is the discriminator; anything else is refused.
4. **Table allowlist**, derived from the semantic layer, with CTE names
   subtracted so `WITH` queries validate correctly, and `sqlite_%` internals
   blocked.
5. **Injected row cap and a query deadline** enforced by a progress handler.
6. **Read-only connection.** `file:path?mode=ro` — SQLite itself refuses the
   write, so the last layer holds even if the five above have a bug.

Also: per-IP rate limiting (10/minute, 200/day) with cached answers exempt
because they cost nothing; a global exception handler that returns a structured
error with a request id and never a traceback; and API keys read from the
environment only, never logged, never returned by `/api/health` — which reports
which providers are *configured*, not what their keys are. `.env` is
gitignored; `.env.example` carries no values.

---

## Known limitations, and what I would do next

**The line-to-header reconciliation variance.** Around 0.44% of revenue does
not reconcile between the two grains inside the three-month evaluation window,
against 0.1% across the full year. Product-level answers use line grain and
will not sum exactly to the canonical totals. It is measured, bounded by the
quality gate and stated in the affected answers, but it is a defect in the
source data that no amount of care in the query layer removes.

**Only one provider is funded.** The Anthropic and OpenAI keys are out of
credit and the xAI team has none, so Gemini is doing all the work — currently
`gemini-flash-lite-latest`, chosen over the stronger `gemini-flash-latest`
because the latter's free tier allows 20 requests per day, which one afternoon
of testing exhausts. A stronger model would improve answer quality
measurably; every model ID is environment-overridable precisely so that is a
dashboard change, not a redeploy.

**The live trace advances on a schedule, not a measurement.** While a request
is in flight the frontend highlights each stage on an elapsed-time schedule,
because the backend returns one response rather than streaming events. The UI
says so, in-flight steps show no duration or token count, and every figure
after the response lands is real. Server-sent events would make it truthful
rather than merely honest, and that is the change I would make first.

**Answer quality is model-dependent and varies between runs.** The cached eight
are verified correct against ground truth. New questions are not guaranteed to
be: the same question asked twice can produce different decompositions, and
quality degrades noticeably when the plan does not isolate a membership test
into its own sub-query. Most of the work in the later modules was converting
such failures from prompt instructions into computed columns, and more of that
remains — the pattern generalises further than I have taken it.

**A caveat about the caveats.** The insight agent's numeric traceability check
skips percentages and values below 100, because a growth rate is arithmetic the
writer is entitled to do. A fabricated small figure or an invented percentage
would pass it. Widening the check without producing false positives needs a
better idea than the one I have.

**Narrative referential slips pass verification.** Asked to compare two stores,
one live answer said "ST015 outperformed ST015" where the second should have
been ST007. Every *number* was correct, so all thirteen checks passed. The
verifier checks arithmetic and the traceability of figures; it does not check
that entity references in prose are internally consistent. A check comparing
entity mentions in the narrative against the entities present in the result
rows would catch it, and is the next check I would write.

**Insight degrades under provider rate limits.** In live testing, two of
nineteen questions lost their narrative to a free-tier rate limit and fell back
to deterministic prose. The data and the verification were intact and the trace
said what had happened, but the answer was noticeably poorer.

**A question naming a non-existent entity returns empty results rather than
saying so.** The resolver correctly refuses to invent a match for `ST999`, but
nothing downstream converts "the question named an identifier and it resolved
to nothing" into a clear message, so the user gets an honest but unhelpful "no
data could be retrieved".

**The deterministic SQL fallback covers only single-table queries.** When no
provider is reachable, SQL can be assembled from the plan alone — but only over
`fact_orders`, and only for metrics and dimensions available there. Anything
needing a join is refused rather than guessed, which is the correct trade,
since a wrong join changes the grain silently. It does mean margin, product and
city questions have no degraded path.

**Clarification is conservative, and its false-negative rate was not measured.**
I verified it triggers on two genuinely ambiguous questions and that the round
trip works. I did not sweep for questions that *should* have been clarified and
were silently answered under an assumption instead. That rate is unknown.

**Coverage is estimated, not measured.** From nineteen live questions across
the supported question types, I would estimate 80–85% of reasonable questions
about this dataset now answer correctly. That is an estimate from a small
sample, not a benchmark. A proper evaluation set of a hundred questions with
human-labelled expected answers is the single thing I would build next.

**Not built:** conversational follow-up (each question is independent),
authentication, and any write path. The rate limiter is in-memory and resets on
restart, which is adequate for a cost guard and inadequate as a security
control.
