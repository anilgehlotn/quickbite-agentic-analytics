# QuickBite Agentic Analytics

A natural-language business analytics system over quick-service restaurant (QSR)
sales data. Ask a question in plain English and a team of AI agents plans the
analysis, writes and runs the SQL, verifies the numbers against the semantic
layer, and explains the result in business terms. The dataset is a fixed
historical extract covering **01-Aug-2025 to 31-Jul-2026**, so the system treats
**2026-07-31** as "today" — every relative time expression such as "last 3
months" resolves against that anchor rather than the system clock.

## Tech stack

| Layer    | Technology                                      |
| -------- | ----------------------------------------------- |
| Backend  | Python 3.11, FastAPI, Uvicorn, pydantic-settings |
| Data     | SQLite (committed to the repo), pandas, openpyxl, sqlglot |
| Frontend | Next.js *(added in a later step)*               |
| LLM      | Anthropic / OpenAI / Gemini / Grok, with fallback |

## Quick start

Requires **Python 3.11+** and **Node 18+**. You need two terminals: the backend
and the frontend run as separate services.

### Terminal 1 — backend (port 8000)

```bash
cd backend

python3 -m venv .venv                 # first time only
source .venv/bin/activate             # Windows: .venv\Scripts\activate
pip install -r requirements.txt       # first time only
cp .env.example .env                  # optional; add LLM keys if you have them

uvicorn app.main:app --reload --port 8000
```

- API root: <http://localhost:8000/>
- Health check: <http://localhost:8000/health>
- Semantic schema: <http://localhost:8000/api/schema>
- Interactive docs: <http://localhost:8000/docs>

### Terminal 2 — frontend (port 3000)

```bash
cd frontend

npm install                           # first time only
cp .env.local.example .env.local      # points at http://localhost:8000

npm run dev
```

Open <http://localhost:3000>. The status card should read **Connected** with
20,000 orders and a data-as-of date of **2026-07-31**. If it reads
**Unreachable**, the backend in terminal 1 is not running.

> The frontend's default origin, `http://localhost:3000`, is already in the
> backend's `CORS_ORIGINS` default, so no CORS setup is needed locally.

### Rebuild the database

`data/quickbite.db` is committed, so this is only needed if the source workbook
changes. The ETL runs the data quality gate automatically and fails the build if
any error-severity check does not pass.

```bash
cd backend
python -m app.etl.build_db            # rebuild + validate
python -m app.etl.quality_checks      # validate only; exits 1 on failure
```

### Regenerate the golden answers

Ground truth for the eight evaluation questions, computed from the Excel file
with pandas as an independent check on the SQL path.

```bash
python scripts/compute_golden_answers.py   # from the project root
```

### Run the tests

```bash
cd backend
python -m pytest tests/ -v
```

## Deployment

The API deploys to Render and the frontend to Vercel. The two have a circular
dependency — the frontend needs the API URL and the API needs the frontend's
origin for CORS — so the order matters. See **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)**
for the full sequence and a verification checklist.

## Project layout

```
backend/
  app/
    config.py           # single source of truth: time anchor, constants, paths
    main.py             # FastAPI app (root, health, schema)
    etl/
      build_db.py       # workbook -> SQLite star schema
      quality_checks.py # data quality gate, runs on every build
    semantic/
      schema.py         # what agents read to write correct SQL
    core/
      logging.py        # structured JSON logging
    agents/             # plan, query, verify, explain     (later step)
    api/                # analytical endpoints             (later step)
  tests/
    golden_answers.json # ground truth for the 8 evaluation questions
  runtime.txt
frontend/               # Next.js 14 App Router + Tailwind
  app/                  # layout, landing page, global styles
  lib/api.ts            # typed backend client
data/                   # source workbook + committed SQLite database
scripts/
  compute_golden_answers.py
docs/
  DEPLOYMENT.md
render.yaml             # Render blueprint for the API
```

> This README is intentionally brief and will be expanded as the system is built out.
