# Pre-submission checklist

Run through this immediately before sending the link. It is ordered by how
expensive the failure is to discover late: the things a reviewer sees in the
first ten seconds come first.

Every check has an expected result written next to it. A check whose expected
result you have to guess at is not a check.

---

## 1. Both URLs respond

- [ ] **Backend health.** `curl https://<render-url>/api/health`
      Expect HTTP 200 and a body containing `"status": "ok"`,
      `"database_ready": true`, `"fact_orders_rows": 20000` and
      `"data_asof": "2026-07-31"`.
- [ ] **`mode` is what you expect.** `"full"` if a provider key is funded,
      `"cache_only"` if not. `"offline"` means the database did not load and
      is a stop-the-line failure.
- [ ] **`cached_answers` is 8.** Zero means the cache file did not ship — the
      single most damaging deployment error, because everything else looks
      fine until someone clicks a question.
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

## 4. Mobile layout holds

- [ ] At **380px** the layout is a single column with no horizontal scrolling.
- [ ] Metric cards stack; numbers use Indian grouping (`₹31,97,076.50`).
- [ ] The trace collapses into an expandable section beneath each answer and
      opens correctly.
- [ ] The chart is readable and does not overflow its container.

## 5. Failure states are dignified

- [ ] Ask a question the data cannot answer ("How do we compare with
      McDonald's?"). Expect a clear statement of scope, not an error.
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
- [ ] `curl https://<render-url>/api/health | grep -Ei "sk-|AIza|xai-"` returns
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

- [ ] README live-application, API and demo-video links are filled in — the
      placeholders are the easiest thing in this list to forget.
- [ ] The demo video plays from a private window.
- [ ] `docs/DEPLOYMENT.md` matches what was actually deployed.

---

## Local pre-flight, before deploying at all

```bash
cd backend
python -m pytest tests/ -q            # expect 692 passed
python -m app.etl.quality_checks      # expect PASSED, 0 errors
cd .. && python scripts/check_offline.py   # expect all 8 answered with no keys
cd frontend && npm run lint && npm run build
```

The offline check is the one that matters most. It clears every API key,
starts the app as a fresh process would, and asserts that all eight questions
still return a headline, narrative, executed SQL, verification report and
four-agent trace. If that passes, the deployment survives every key expiring.
