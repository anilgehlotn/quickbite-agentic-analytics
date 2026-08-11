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

## Setup

Requires Python 3.11 or newer.

```bash
# 1. Create and activate a virtual environment
cd backend
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment variables (all optional)
cp .env.example .env               # then add your LLM API keys
```

### Run the dev server

```bash
cd backend
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

- API root: <http://localhost:8000/>
- Health check: <http://localhost:8000/health>
- Interactive docs: <http://localhost:8000/docs>

### Run the tests

```bash
cd backend
python -m pytest tests/ -v
```

## Project layout

```
backend/
  app/
    config.py      # single source of truth: time anchor, business constants, paths
    main.py        # FastAPI app (root + health)
    etl/           # workbook -> SQLite               (later step)
    semantic/      # metric and dimension definitions (later step)
    agents/        # plan, query, verify, explain     (later step)
    api/           # analytical endpoints             (later step)
    core/          # shared infrastructure            (later step)
  tests/
data/              # source workbook + SQLite database
scripts/
docs/
```

> This README is intentionally brief and will be expanded as the system is built out.
