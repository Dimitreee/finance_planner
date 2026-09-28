# CryptoGuard — one place for the commands that already exist.
#
# Nothing here is new behaviour: every target runs something documented in the README or in a
# script. `make help` lists them.

SHELL := /bin/bash
.DEFAULT_GOAL := help

WEB := apps/web
SNAPSHOT := data/raw/binance/klines/BTCUSDT/1h
COMPOSE_DB_URL := postgresql://cryptoguard:cryptoguard@localhost:5432/cryptoguard

.PHONY: help
help: ## List the targets
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk -F':.*?## ' '{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# --- setup -------------------------------------------------------------------------------------

.PHONY: setup
setup: setup-python setup-web ## Install everything from a clean checkout

.PHONY: setup-python
setup-python: ## Provision .venv on a uv-managed CPython and install the pinned dependencies
	uv sync

.PHONY: setup-web
setup-web: ## Install the web dependencies
	cd $(WEB) && npm install

# --- checks ------------------------------------------------------------------------------------

.PHONY: check
check: lint typecheck test ## Everything: the target to run before committing

.PHONY: lint
lint: ## Lint and formatting, both sides
	uv run ruff check .
	uv run ruff format --check .

.PHONY: format
format: ## Apply the formatter and the safe lint fixes
	uv run ruff check --fix .
	uv run ruff format .

.PHONY: typecheck
typecheck: ## mypy --strict and tsc --noEmit
	uv run mypy
	cd $(WEB) && npx tsc --noEmit

.PHONY: test
test: test-python test-web ## Both test suites

.PHONY: test-python
test-python: guard-test-db ## pytest. The job and API seams skip unless a test database is reachable
	uv run pytest

.PHONY: test-web
test-web: ## vitest
	cd $(WEB) && npx vitest run

.PHONY: guard-test-db
guard-test-db:
	@# The job and API seam tests DROP their tables. Pointing them at the database the app uses
	@# would destroy published runs, which is a mistake worth making impossible rather than
	@# documenting. This guard exists because it has already happened once.
	@if [ -n "$$CRYPTOGUARD_TEST_DATABASE_URL" ] \
	   && [ "$$CRYPTOGUARD_TEST_DATABASE_URL" = "$$CRYPTOGUARD_DATABASE_URL" ]; then \
		echo "refusing: CRYPTOGUARD_TEST_DATABASE_URL is the same database as" >&2; \
		echo "CRYPTOGUARD_DATABASE_URL, and the seam tests drop their tables." >&2; \
		exit 1; \
	fi

# --- running -----------------------------------------------------------------------------------

.PHONY: db
db: ## Start PostgreSQL in Docker and wait for it
	docker compose up -d db
	@until docker compose exec -T db pg_isready -U cryptoguard -d cryptoguard >/dev/null 2>&1; do \
		sleep 1; \
	done
	@echo "database ready: $(COMPOSE_DB_URL)"

.PHONY: db-stop
db-stop: ## Stop the database, keeping its volume
	docker compose stop db

.PHONY: api
api: guard-db-url ## Serve the read API on :8000
	uv run uvicorn --factory cryptoguard_api:create_app --host 127.0.0.1 --port 8000

.PHONY: web
web: ## Serve the page on :5173, proxying /api to :8000
	cd $(WEB) && npm run dev

.PHONY: build-web
build-web: ## Type-check and build the page
	cd $(WEB) && npm run build

.PHONY: guard-db-url
guard-db-url:
	@if [ -z "$$CRYPTOGUARD_DATABASE_URL" ]; then \
		echo "CRYPTOGUARD_DATABASE_URL is not set. With 'make db' running:" >&2; \
		echo "  export CRYPTOGUARD_DATABASE_URL=$(COMPOSE_DB_URL)" >&2; \
		exit 1; \
	fi

# --- data --------------------------------------------------------------------------------------

.PHONY: data
data: data-market data-news ## Fetch and verify both raw datasets

.PHONY: data-market
data-market: ## Fetch the monthly kline archives and verify them against the manifest
	./scripts/fetch_binance_monthly.sh

.PHONY: data-daily
data-daily: ## Fetch one month of daily archives, e.g. make data-daily MONTH=2026-08
	@if [ -z "$(MONTH)" ]; then echo "set MONTH, e.g. make data-daily MONTH=2026-08" >&2; exit 1; fi
	./scripts/fetch_binance_daily.sh $(MONTH)

.PHONY: data-news
data-news: ## Fetch the news archive and verify both digests
	./scripts/fetch_news_archive.sh

.PHONY: data-sentiment
data-sentiment: ## Score the BTC headlines with the frozen extractor, filling its cache
	@# Not part of `data`: that target fetches raw archives, this one derives from one. Idempotent —
	@# a second run loads no model, because every reading is cached by (text_hash, revision).
	uv run python scripts/score_news_sentiment.py

.PHONY: annotation
annotation: ## Draw the Annotation Sample and write the blind labelling file
	@# Refuses to overwrite an existing draw: the seed is pre-registered, so redrawing is a contract
	@# edit rather than a rerun (ADR-0016).
	uv run python scripts/annotation_sample.py draw

.PHONY: annotation-score
annotation-score: ## Measure agreement once the labels file has been filled in by hand
	uv run python scripts/annotation_sample.py score

.PHONY: audit
audit: ## Re-run every data audit against the local snapshots
	uv run python scripts/audit/audit_binance_archives.py
	uv run python scripts/audit/audit_binance_gaps.py
	uv run python scripts/audit/audit_news_archive.py
	uv run python scripts/audit/audit_market_seam.py

# --- the pipeline ------------------------------------------------------------------------------

.PHONY: evaluate
evaluate: guard-db-url ## Fit Arm A across the folds, publish the Replay Result, promote the release
	uv run cryptoguard-job evaluate --promote

.PHONY: compare
compare: ## Fit arms B and C across the lag grid and answer the Primary Comparison
	@# Spends one Trial per (arm, lag). Refuses to refit a pair the Trial log already holds, and
	@# refuses to start unless the remaining budget covers every pair. --dry-run prints the plan.
	uv run python scripts/compare_arms.py

.PHONY: intervals
intervals: ## Bootstrap Intervals for the Primary Comparison and the Headline Metric
	@# Spends no Trial: it re-derives results already on the record and estimates uncertainty around
	@# them. Takes several minutes at the pre-registered 10 000 resamples.
	uv run python scripts/bootstrap_intervals.py

.PHONY: holdout
holdout: ## Spend the single permitted evaluation of the Final Holdout (irreversible)
	@# Succeeds exactly once. A second invocation refuses and exits non-zero without producing a
	@# number (ADR-0018). Use --dry-run first to see the frozen configuration and the digest.
	uv run python scripts/final_holdout.py

.PHONY: holdout-interval
holdout-interval: ## Bootstrap Intervals describing the recorded Final Holdout result
	@# Spends nothing and is not a second evaluation: it re-derives the deterministic per-day series
	@# and refuses unless what it re-derived is identical to what the marker records.
	uv run python scripts/holdout_interval.py

.PHONY: decide
decide: guard-day guard-db-url ## Publish one Decision Day, e.g. make decide DAY=2026-02-01
	uv run cryptoguard-job decide --day $(DAY)

.PHONY: guard-day
guard-day:
	@# Before the environment check, so a missing argument is reported as a missing argument
	@# rather than as a configuration problem.
	@if [ -z "$(DAY)" ]; then echo "set DAY, e.g. make decide DAY=2026-02-01" >&2; exit 1; fi

# --- housekeeping ------------------------------------------------------------------------------

.PHONY: clean
clean: ## Remove build output and caches. Raw data, manifests and artifacts are left alone
	rm -rf $(WEB)/dist .pytest_cache .ruff_cache .mypy_cache
	find . -name __pycache__ -type d -prune -not -path "./.venv/*" -exec rm -rf {} +
