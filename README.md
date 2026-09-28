# CryptoGuard

An explainable BTC/USDT advisor: forecast → policy → action relative to a virtual position →
explanation → cost-aware evaluation of the advice itself. It executes no real trades and holds no user
funds.

University of London BSc Computer Science final project, Project Idea 4.2 (Financial Advisor Bot).

## Getting started

```sh
make setup   # uv sync, then npm install
make help    # every other target, with one line each
```

`make check` is the one to run before committing: lint, both type checkers, both test suites.

The sections below explain what the targets do; the Makefile is the short version.

`make setup-python` alone is enough for everything except the page:

```sh
uv sync
```

That is the whole setup. It provisions a project-local `.venv` on a uv-managed CPython 3.12 and installs
every dependency at the exact version recorded in `uv.lock`. It deliberately does **not** use a
system or conda Python: the environment must be reproducible on a machine that has neither.

Requires [uv](https://docs.astral.sh/uv/). Everything else, including the interpreter, uv installs.

Then:

```sh
uv run pytest          # test suite
uv run ruff check .    # lint
uv run mypy            # type check
uv run cryptoguard-job # batch job entry point
```

## The database

Published runs live in PostgreSQL. Nothing else is containerised: the job, the training and the tests
run natively through uv, because training on this machine uses MPS and Docker on macOS does not offer
it.

```sh
docker compose up -d db
export CRYPTOGUARD_DATABASE_URL=postgresql://cryptoguard:cryptoguard@localhost:5432/cryptoguard
uv run cryptoguard-job decide --day 2021-02-01
```

The job creates its own schema on first run and is idempotent: running the same Decision Day twice
publishes once, and the second run books no second trade. A named volume keeps published runs across
a restart of the container.

To run the job-seam tests against a database, point them at one:

```sh
CRYPTOGUARD_TEST_DATABASE_URL=postgresql://cryptoguard:cryptoguard@localhost:5432/cryptoguard \
  uv run pytest tests/test_decide_job.py
```

They **drop and recreate** their tables, so give them a database of their own. Without that variable
they skip, and the rest of the suite runs with no database at all. `make test` refuses to run when
`CRYPTOGUARD_TEST_DATABASE_URL` names the same database as `CRYPTOGUARD_DATABASE_URL`, because that
mistake destroys published runs.

## The read API

```sh
uv run uvicorn --factory cryptoguard_api:create_app --port 8000
```

- `GET /api/health` — liveness. Needs no database and no configuration.
- `GET /api/advice` — the newest Published Run: action, probability, position, explanation, the data
  cutoff, and two independent fields, `freshness` (`current` / `stale` / `unavailable`) and
  `model_mode` (`price-only` / `fused` / `replay`).
- `GET /api/paper-track` — the live track since go-live, with its genesis day.

The process loads no model weights and holds no data provider credentials: it reads published state
and nothing else, and a test asserts that by inspecting what the package imports.

## The page

```sh
cd apps/web && npm install && npm run dev
```

It serves on http://localhost:5173 and proxies `/api` to the API on port 8000, so run both. The page
shows today's action with the one-day horizon, the probability and the rule behind it, the
explanation, the data cutoff and freshness, the simulated portfolio, and the track since launch with
a price chart. See `apps/web/README.md`.

## Layout

| Path             | What lives there                                                        |
| ---------------- | ----------------------------------------------------------------------- |
| `packages/core`  | Shared domain core: data contracts, features, models, policy, backtest   |
| `apps/api`       | Read-only HTTP surface over published state; no weights, no credentials  |
| `apps/web`       | Web UI: React and TypeScript, built with Vite                           |
| `jobs`           | Batch entry points: ingest, decide, publish                             |
| `config`         | The Experiment Contract and other frozen configuration                  |
| `migrations`     | Database schema changes                                                 |
| `scripts/audit`  | Standalone data audits, reproducible from the raw snapshot              |
| `tests`          | Test suite                                                              |
| `reports`        | Audit and evaluation write-ups                                          |

Raw datasets, model weights and caches stay out of git; `uv.lock` and the hash manifests under
`data/manifests` are committed, so a result can be reproduced from a recorded input.
