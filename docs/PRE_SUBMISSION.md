# Pre-submission checklist

Run through this immediately before sending the link. It is ordered by how
expensive the failure is to discover late: the things a reviewer sees in the
first ten seconds come first.

Every check has an expected result written next to it. A check whose expected
result you have to guess at is not a check.

---

## 1. Both URLs respond

- [ ] **Backend health.** `curl https://quickbite-analytics-api.onrender.com/api/health`
      Expect HTTP 200 and a body containing `"status": "ok"`,
      `"database_ready": true`, `"fact_orders_rows": 20000` and
      `"data_asof": "2026-07-31"`.
- [ ] **`mode` is what you expect.** `"full"` if a provider key is funded,
      `"cache_only"` if not. `"offline"` means the database did not load and
      is a stop-the-line failure.
- [ ] **`cached_answers` is at least 8.** Zero means the cache file did not
      ship — the single most damaging deployment error, because everything else
      looks fine until someone clicks a question. The count is currently 21:
      the eight canonical answers plus thirteen written by live testing during
      Module 9. Extra entries are harmless (they make more questions instant
      and provider-free), but see section 8 before shipping.
- [ ] **Provider health.** `curl https://quickbite-analytics-api.onrender.com/api/providers` shows
      `providers_healthy` non-empty if any key is funded. A provider listed in
      `providers_in_cooldown` with a non-zero `cooldown_remaining_seconds` is
      the circuit breaker working, not a fault; `last_failure` says why.
- [ ] **Frontend loads.** Open the Vercel URL. The header status pill reads
      `connected` within a few seconds, not `offline`.
- [ ] **Time the first request.** If the backend was asleep this can take up
      to 50 seconds. Confirm the frontend shows the waking message rather than
      appearing to hang, then reload and confirm the second load is fast.

## 2. All eight questions answer correctly, from a fresh browser session

Use a private window so nothing is warm on the client side. Click each chip and
check the headline against `backend/tests/golden_answers.json`.

- [ ] **Q1 revenue, orders, AOV** — 3,197,076.50 INR / 4,930 orders / 648.49
- [ ] **Q2 best and worst stores** — Gurugram 04 highest at 99,985.0; Delhi 19
      lowest at 37,498.5
- [ ] **Q3 channels** — all four present; Zomato highest at 907,336.50
- [ ] **Q4 products** — Veg Burger 5 highest by units (706); Non-Veg Pizza 4
      highest by revenue (258,564.12)
- [ ] **Q5 declining cities** — the headline states plainly that **no** city
      declined in every consecutive month. If it names a city as declining
      without that qualification, the answer has regressed.
- [ ] **Q6 weekend vs weekday** — compares **per trading day** (weekend
      ~40,117 vs weekday ~33,684), not raw totals
- [ ] **Q7 festive lift** — expressed per trading day, roughly 18–36% uplift
- [ ] **Q8 consistent decliners** — exactly nine stores, and the answer
      separates the four above their own baseline (ST002, ST016, ST030, ST039)
      from the five genuinely below it (ST007, ST010, ST011, ST015, ST042)
- [ ] **No store or SKU identifier is malformed.** `ST007`, never `ST07`.
- [ ] Each answer shows a verification badge; none reads `Unverified`.

## 3. The trace renders

- [ ] Trace panel shows **all four agents** with durations and token counts.
- [ ] The **plan summary** shows intent, window and sub-query count.
- [ ] **Executed SQL** is present for every sub-query, and the copy button
      works.
- [ ] Any **retry or skipped step** is visible rather than hidden. A visible
      recovery is a feature; hiding it would defeat the panel.
- [ ] Ask one **free-form** question (if a provider is funded) and watch the
      stages advance during the wait.
- [ ] A **degraded** sub-query, if one occurs, shows the `simplified query`
      badge. That means no provider was reachable and the SQL was assembled
      from the plan; the figures are exact, the query is a plain aggregate.

## 4. Mobile layout holds

- [ ] At **380px** the layout is a single column with no horizontal scrolling.
- [ ] Metric cards stack; numbers use Indian grouping (`₹31,97,076.50`).
- [ ] The trace collapses into an expandable section beneath each answer and
      opens correctly.
- [ ] The chart is readable and does not overflow its container.

## 5. Failure states are dignified

- [ ] Ask a question the data cannot answer ("How do we compare with
      McDonald's?"). Expect `status: "unsupported"`, a statement naming what
      specifically is missing, and two or three adjacent questions that *are*
      answerable, offered as clickable follow-ups.
- [ ] Ask an **ambiguous** question ("How are pizzas doing?"). Expect
      `status: "clarification_needed"` rendered as a question back to you with
      two to four interpretations, styled as a normal card and **not** as an
      error. Click one and confirm it answers.
- [ ] Ask a **year-on-year** question ("Compare this year to last year").
      Expect a refusal explaining that the data covers a single twelve-month
      window — not a silent comparison against an earlier part of the same year.
- [ ] Ask about a **non-existent store** ("How is ST999 doing?"). Expect empty
      results reported honestly. Known gap: it does not yet say the store does
      not exist.
- [ ] If `mode` is `cache_only`, confirm the banner above the input says so
      **before** anyone types, and the suggestion chips stay prominent.
- [ ] Trigger the rate limit if convenient (11 uncached questions in a
      minute). Expect a calm 429 explaining when to retry, in caution amber
      rather than red.

## 6. No key material anywhere

- [ ] `git grep -nEi "sk-[a-zA-Z0-9]{10}|AIza[0-9A-Za-z_-]{10}|xai-[a-zA-Z0-9]{10}"`
      returns **only** the two synthetic fixtures in `backend/tests/test_llm.py`
      (`AIzaSUPERSECRETVALUE03`, `xai-SUPERSECRETVALUE04`). Those are
      deliberately key-shaped: the test asserts that no real key of that shape
      can leak through the health output. Anything else in the results is a
      genuine leak.
- [ ] `.env` is **not** tracked: `git ls-files backend/.env` prints nothing.
- [ ] `backend/.env.example` contains placeholders only, no real values.
- [ ] `curl https://quickbite-analytics-api.onrender.com/api/health | grep -Ei "sk-|AIza|xai-"` returns
      nothing — the endpoint reports which providers are *configured*, never
      their keys.
- [ ] Skim the Render logs for the deploy: no key material, no tracebacks
      reaching a client.
- [ ] The committed `data/answer_cache.json` contains answers and SQL only.

## 7. CI is green

- [ ] The Actions tab shows the latest `ci` run passing on the default branch.
- [ ] The README badge renders green (it points at the same workflow).
- [ ] The `keepalive` workflow has run recently and succeeded. If it has never
      run, check that the `BACKEND_URL` repository variable is set.

## 8. Repository presentation

- [ ] **The three README links open.** They are filled in, so the risk is no
      longer a forgotten placeholder but a link that resolves for you and not
      for a stranger. Check each from a private window:
      - <https://quickbite-agentic-analytics.vercel.app>
      - <https://quickbite-analytics-api.onrender.com>
      - <https://drive.google.com/drive/folders/1deZuT6HP__SPFqAJHvHHBWLCWxX0Jt3_?usp=sharing>
- [ ] **The Drive folder is shared with anyone holding the link**, and the
      video inside it plays without a Google sign-in. A folder set to
      "restricted" looks fine to its owner and is a dead end to everyone else.
- [ ] `docs/DEPLOYMENT.md` matches what was actually deployed.
- [ ] `docs/Agent_Architecture.pdf` regenerated from `docs/ARCHITECTURE.md`
      (`python scripts/build_architecture_pdf.py`) and opens cleanly.
- [ ] **Review the answer cache.** `data/answer_cache.json` holds 21 entries:
      the eight canonical answers plus thirteen written by live testing during
      Module 9. The extras are harmless — they make more questions instant and
      provider-free — but they are cached output a reviewer can reach, so read
      any you have not read. Two were removed for that reason:

      * `how is store st999 performing` — cached an empty-result answer for a
        store that does not exist. Asked live, the question now takes the
        unsupported path and says so; the cache was short-circuiting that with
        a worse answer.
      * `how do loyal customers compare with occasional ones` — compared the
        two segments without stating the anonymous-orders share, which the
        business rules require whenever customer segments are discussed. The
        surviving `how do loyal regular and occasional customers compare by
        revenue` answers the same question and does state it.

      Nothing else was touched, and no cache entry was regenerated. To remove
      another key, edit the JSON and re-run `python scripts/check_offline.py`
      plus `pytest tests/test_golden_e2e.py`; do not re-warm the file.

---

## Local pre-flight, before deploying at all

```bash
cd backend
python -m pytest tests/ -q            # expect 813 passed
python -m app.etl.quality_checks      # expect PASSED, 0 errors
cd .. && python scripts/check_offline.py   # expect all 8 answered with no keys
cd frontend && npm run lint && npm run build
```

The offline check is the one that matters most. It clears every API key,
starts the app as a fresh process would, and asserts that all eight questions
still return a headline, narrative, executed SQL, verification report and
four-agent trace. If that passes, the deployment survives every key expiring.
