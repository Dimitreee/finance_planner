# CryptoGuard

An explainable BTC/USDT advisor: forecast → policy → action relative to a virtual position →
explanation → cost-aware evaluation of the advice itself. It executes no real trades and holds no user
funds.

University of London BSc Computer Science final project, Project Idea 4.2 (Financial Advisor Bot).

## Getting started

One command, from a clean checkout:

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
they skip, and the rest of the suite runs with no database at all.

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

## Layout

| Path             | What lives there                                                        |
| ---------------- | ----------------------------------------------------------------------- |
| `packages/core`  | Shared domain core: data contracts, features, models, policy, backtest   |
| `apps/api`       | Read-only HTTP surface over published state; no weights, no credentials  |
| `apps/web`       | Web UI                                                                  |
| `jobs`           | Batch entry points: ingest, decide, publish                             |
| `config`         | The Experiment Contract and other frozen configuration                  |
| `migrations`     | Database schema changes                                                 |
| `scripts/audit`  | Standalone data audits, reproducible from the raw snapshot              |
| `tests`          | Test suite                                                              |
| `reports`        | Audit and evaluation write-ups                                          |

Raw datasets, model weights and caches stay out of git; `uv.lock` and the hash manifests under
`data/manifests` are committed, so a result can be reproduced from a recorded input.

## Where the thinking lives

**These paths are deliberately not tracked in git, so a fresh clone will not contain them.** They exist
in the working copy and are the normative reference for anyone working on this repository:

- `CONTEXT.md` — the glossary. Every domain term used in the code is defined there.
- `docs/adr/` — the decisions, each with the alternatives that were rejected and why.
- `JOURNAL.md` — dated record of what was decided, what broke and what was measured.
- `reports/` — audits and evaluations.
- `.scratch/` — specs and implementation tickets.

Keeping them out of version control was a deliberate choice recorded in `JOURNAL.md` on 2026-09-27. The
consequence is that the design history lives in dated journal entries rather than in commits, and that
tracked files do not cite these documents by filename.
