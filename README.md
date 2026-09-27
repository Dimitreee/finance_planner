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
