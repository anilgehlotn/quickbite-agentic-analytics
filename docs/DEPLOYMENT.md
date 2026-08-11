# Deployment

QuickBite runs as two services:

| Service  | Platform | Root directory | URL shape                                    |
| -------- | -------- | -------------- | -------------------------------------------- |
| API      | Render   | `backend`      | `https://quickbite-analytics-api.onrender.com` |
| Frontend | Vercel   | `frontend`     | `https://<project>.vercel.app`               |

There is no database to provision. The SQLite file is committed to the
repository, so the API deploys as a stateless, read-only service.

> **Read this first: the two services depend on each other.**
> The frontend needs the API's URL, and the API needs the frontend's origin in
> its CORS allowlist. Neither URL exists until its service is deployed, so the
> order in [Step 4](#step-4--resolve-the-circular-dependency) matters. Deploy the
> backend first, then the frontend, then come back and update the backend.

---

## Prerequisites

- A GitHub account.
- A [Render](https://render.com) account (free tier is sufficient).
- A [Vercel](https://vercel.com) account (Hobby tier is sufficient).
- `data/quickbite.db` committed. Confirm with `git ls-files data/quickbite.db`;
  if it prints nothing, the API will deploy but `/health` will report
  `"status": "degraded"`.

---

## Step 1 — Push to GitHub

1. Create a new repository on GitHub. Do **not** initialise it with a README,
   `.gitignore` or licence — the repository already has them.
2. From the project root:

   ```bash
   git init                       # skip if already a repository
   git add .
   git commit -m "QuickBite agentic analytics"
   git branch -M main
   git remote add origin https://github.com/<you>/quickbite-agentic-analytics.git
   git push -u origin main
   ```

3. Verify the database was pushed. On the GitHub file listing, `data/` should
   contain `quickbite.db` at roughly 10 MB. The `.gitignore` deliberately does
   **not** exclude `data/*.db`; if you see the file missing, check that rule
   before anything else.

---

## Step 2 — Deploy the API on Render

1. In the Render dashboard: **New → Blueprint**.
2. Connect your GitHub account and select the repository.
3. Render reads [`render.yaml`](../render.yaml) and shows a service named
   **quickbite-analytics-api**. Click **Apply**.
4. Render prompts for the variables marked `sync: false`. Set them now:

   | Variable            | Value                                         | Notes                                                                 |
   | ------------------- | --------------------------------------------- | --------------------------------------------------------------------- |
   | `ENVIRONMENT`       | `production`                                  | Appears in `/health`.                                                  |
   | `CORS_ORIGINS`      | `http://localhost:3000`                        | **Placeholder.** Updated in Step 4 once the Vercel URL exists.         |
   | `ANTHROPIC_API_KEY` | your key, or leave blank                       | Optional. Blank is valid; the app starts and reports no providers.     |
   | `OPENAI_API_KEY`    | your key, or leave blank                       | Optional.                                                              |
   | `GEMINI_API_KEY`    | your key, or leave blank                       | Optional.                                                              |
   | `GROK_API_KEY`      | your key, or leave blank                       | Optional.                                                              |

   `PYTHON_VERSION` and `LOG_LEVEL` are already set in `render.yaml` and need no
   input.

5. Wait for the first build (typically 2–4 minutes). Render runs
   `pip install -r requirements.txt` and starts
   `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
6. Note the assigned URL, e.g. `https://quickbite-analytics-api.onrender.com`.
   **You need it in Step 3.**
7. Confirm the service is live:

   ```bash
   curl https://<your-api>.onrender.com/health
   ```

   Expect `"status": "ok"` and `"fact_orders_rows": 20000`. If you get
   `"degraded"`, the database did not ship — return to Step 1.3.

### Reading the startup log

The first line the app logs is a single JSON object recording the resolved
configuration. It is the fastest way to diagnose a bad deploy:

```json
{"ts":"...","level":"INFO","logger":"app.main","msg":"startup",
 "environment":"production","database_ready":true,"fact_orders_rows":20000,
 "providers_configured":["anthropic"],"cors_origins":["http://localhost:3000"]}
```

Check `cors_origins` here after Step 4 — it is the authoritative view of what
the running service will actually allow.

---

## Step 3 — Deploy the frontend on Vercel

1. In the Vercel dashboard: **Add New → Project**, then import the same
   repository.
2. **Set the root directory to `frontend`.** This is the one setting that is
   easy to miss and it will fail the build if wrong. Use the **Edit** control
   beside "Root Directory" on the import screen.
3. Framework preset should auto-detect as **Next.js**. Leave the build and
   output settings at their defaults.
4. Add an environment variable:

   | Variable              | Value                                          |
   | --------------------- | ---------------------------------------------- |
   | `NEXT_PUBLIC_API_URL` | `https://<your-api>.onrender.com` from Step 2.6 |

   No trailing slash. Apply it to Production, Preview and Development.

   > `NEXT_PUBLIC_*` variables are inlined into the browser bundle **at build
   > time**, not read at runtime. Changing this value later requires a redeploy,
   > not just a restart.

5. Click **Deploy** and note the assigned URL, e.g.
   `https://quickbite-analytics.vercel.app`.

At this point the page loads but the status card will show **Unreachable** — the
API is rejecting the browser's origin. That is expected, and Step 4 fixes it.

---

## Step 4 — Resolve the circular dependency

The frontend needs the API's URL; the API needs the frontend's origin. Each is
only known after the other is deployed, so the loop is broken by deploying with
a placeholder and correcting it:

```
Step 2   deploy API          → API URL now known, CORS_ORIGINS is a placeholder
Step 3   deploy frontend     → uses the API URL; frontend origin now known
Step 4   update API's CORS   → API now accepts the frontend  ✅
```

1. Return to the Render dashboard → your service → **Environment**.
2. Edit `CORS_ORIGINS` to the comma-separated list:

   ```
   https://quickbite-analytics.vercel.app,http://localhost:3000
   ```

   - No spaces are required, though they are tolerated and stripped.
   - No trailing slashes — an origin is scheme + host + port only.
   - Keep `http://localhost:3000` so local development continues to work
     against the deployed API.
   - A JSON array is also accepted, but the plain comma-separated form is what
     dashboards handle best.

3. Save. Render restarts the service automatically (about 30 seconds).
4. Reload the Vercel URL. The status card should now show **Connected** with
   20,000 orders.

### Adding Vercel preview deployments

Every Vercel preview gets its own generated URL, which will not be in the
allowlist. If you need previews to reach the API, add the specific preview
origin to `CORS_ORIGINS`, or use a Vercel custom domain for a stable origin.

---

## Step 5 — Cold starts on the Render free tier

Render suspends free-tier services after **15 minutes** of inactivity. The next
request wakes the container, which takes **up to ~50 seconds**. This is normal
and is not a fault.

How the app handles it today:

- `lib/api.ts` uses a **60 second timeout**, comfortably above the worst
  observed cold start, so a waking backend resolves rather than aborting.
- After 4 seconds without a response, the status card explains that the backend
  may be waking and that the request has not failed. It never reports a false
  outage while still waiting.

A keep-alive ping to hold the service warm is planned for a later step. Until
then, if you are demonstrating the system live, load the page once a minute or
two beforehand so the container is already awake.

---

## Verification checklist

Run these against the deployed URLs. All four must pass.

**API**

```bash
API=https://<your-api>.onrender.com

# 1. Service identifies itself
curl -s $API/ | python3 -m json.tool
#    expect: name, version, "status": "running"

# 2. Health, with a live database query
curl -s $API/health | python3 -m json.tool
#    expect: "status": "ok", "database_ready": true, "fact_orders_rows": 20000
#            "data_asof": "2026-07-31"

# 3. Semantic layer loaded
curl -s $API/api/schema | python3 -m json.tool | head -20
#    expect: compact_schema text, 10 tables, 10 metric names
```

**CORS**

```bash
# 4. The deployed frontend origin is allowed
curl -s -I -X OPTIONS $API/health \
  -H "Origin: https://<your-project>.vercel.app" \
  -H "Access-Control-Request-Method: GET" | grep -i access-control-allow-origin
#    expect: access-control-allow-origin: https://<your-project>.vercel.app
#    no header returned means Step 4 is incomplete or the origin does not match
#    exactly (check for a trailing slash or http vs https)
```

**Frontend**

- [ ] The Vercel URL loads and shows the QuickBite heading.
- [ ] The status card reads **Connected** (not Connecting, not Unreachable).
- [ ] "Orders in database" shows **20,000**.
- [ ] "Data as of" shows **2026-07-31** — not today's date. If it shows today's
      date, the time anchor has been broken somewhere.
- [ ] Configured providers match the keys you set on Render.
- [ ] Stopping the Render service makes the card show a clear error with a
      working **Try again** button, rather than a blank or falsely healthy card.

---

## Troubleshooting

| Symptom                                                | Cause                                                                | Fix                                                                          |
| ------------------------------------------------------ | -------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| `/health` returns `"status": "degraded"`               | `data/quickbite.db` not in the repository                             | Confirm `git ls-files data/quickbite.db`; check `.gitignore` does not exclude it |
| Card shows "Unreachable", API responds to `curl`       | Origin not in `CORS_ORIGINS`                                          | Step 4; match scheme and host exactly, no trailing slash                       |
| Vercel build fails, cannot find `package.json`          | Root directory not set to `frontend`                                  | Project Settings → General → Root Directory → `frontend`                       |
| Card still points at `localhost:8000` in production     | `NEXT_PUBLIC_API_URL` missing, or set after the build                 | Add the variable, then **redeploy** — it is inlined at build time              |
| App crashes at startup with a JSON decode error         | An older build parsing `CORS_ORIGINS`                                 | Current code accepts comma-separated values; redeploy from `main`              |
| First request takes ~50 seconds                        | Free-tier cold start                                                  | Expected; see Step 5                                                          |
| `providers_configured` is `[]`                          | No LLM keys set                                                       | Expected until keys are added; the app runs fine without them                  |

---

## What is deliberately not automated

- **No database migrations.** The SQLite file is a committed build artefact.
  Rebuild it locally with `python -m app.etl.build_db` and commit the result.
- **No secrets in the repository.** Every key is `sync: false` in
  `render.yaml` and set through the dashboard.
- **No keep-alive yet.** Added in a later step.
