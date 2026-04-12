# CLAUDE.md

Guidance for Claude Code when working in this repository. Keep this short —
link to the detailed memos rather than duplicating them.

## Project

`market-data-warehouse` (package name) / finance-quant (repo name) is a
quantitative trading platform built around a time-correct market data
warehouse. It ingests multi-source data (FRED, GDELT, Polymarket, Yahoo
Finance, SEC EDGAR, options, Reddit sentiment, ETF flows, CFTC, Fama-French
factors), curates it with strict `event_time` vs `available_time` separation
to prevent look-ahead bias, builds point-in-time snapshots for training,
backtests strategies, and executes live/paper trades via Alpaca. Also
includes market-making utilities. See `README.md` for the longer pitch.

## Tech stack

- **Python 3.11+** (`pyproject.toml:10`; also supports 3.12).
- **Dependencies:** pip + lock files (`requirements.lock`,
  `requirements-dev.lock`). Install with `make install` / `make install-dev`.
- **Core libs:** SQLAlchemy 2, DuckDB (default, zero-config) or PostgreSQL,
  pandas 2, pyarrow, pydantic 2 + pydantic-settings, Typer + Rich (CLI),
  httpx, tenacity, prometheus-client, alpaca-py, scikit-learn.
- **Optional extras:** `ml` (lightgbm, xgboost), `postgresql` (psycopg2),
  `web` (jinja2), `dev` (black, ruff, mypy, pytest, pytest-asyncio, pytest-cov).
- **Docker:** `docker-compose.yml` is optional and only runs PostgreSQL;
  the default DuckDB backend needs no server.

## Layout

```
src/pipeline/
  extract/           # 14+ source extractors (fred, gdelt, prices, polymarket,
                     #   sec_*, options, earnings, reddit, short_interest,
                     #   etf_flows, cftc_cot, factors_ff, …)
  load/              # Raw-to-warehouse loaders
  transform/         # Curated table transformations
  snapshot/          # Point-in-time snapshot builders
  dq/                # Data-quality tests (time monotonicity, referential,
                     #   anti-look-ahead)
  features/          # Feature engineering
  backtesting/       # Backtest engine (SEE GOTCHAS BELOW)
  strategy/          # Signal generation, sizing, risk
  execution/         # Alpaca broker integration (live / paper)
  eval/              # Performance metrics, stress tests
  market_making/     # Quoting, spreads, inventory hedging
  historical/        # Historical data management
  infrastructure/    # Async pools, circuit breakers, metrics, validation
  web/               # Jinja2 templates for reports
  cli.py             # Typer CLI; main() + __main__ at line 1899
  settings.py        # Pydantic BaseSettings (env + YAML)
  db.py              # DB session / engine helpers
  logging_config.py  # Logging setup
src/sql/
  ddl/               # Postgres DDL migrations
  ddl_duckdb/        # DuckDB DDL migrations
tests/               # ~47 pytest modules
data/
  signals/           # Daily signal CSVs (2025-12-08 … 2026-04-10)
  prediction_history.json  # Prediction outcomes
config.yaml          # Runtime config (consumed by settings.py)
```

## Common commands

All from the `Makefile`; run from repo root.

| Command | What it does |
|---|---|
| `make setup` | `install` + `db-init` (DuckDB, no server) |
| `make install` / `make install-dev` | `pip install -e .` / `.[dev]` |
| `make db-init` / `make db-reset` | Initialize / recreate DuckDB schema |
| `make pg-up` / `make pg-init` | Start + init the optional Postgres container |
| `make extract-prices` | Required extractor (others are best-effort) |
| `make extract-all` | Prices + every other configured source |
| `make transform` | Run curated transformations |
| `make snapshots` | Build training snapshots |
| `make dq` / `make test-dq` | Run data-quality tests |
| `make latency-stats` | Compute source latency distributions |
| `make full-pipeline` | extract-all → transform → latency-stats → snapshots → dq → inventory |
| `make historical-backfill` | Full 2010-present backfill (all sources, end-to-end) |
| `make historical-backfill-prices` | Backfill prices + factors only |
| `make daily-predictions` | Generate daily predictions + update `site/` |
| `make test` | `pytest tests/ -v` |
| `make test-snapshots` | `pytest tests/test_snapshots.py -v` |
| `make lint` | `ruff check src/ tests/` + `mypy src/` |
| `make format` | `black src/ tests/` + `ruff check --fix src/ tests/` |
| `make inventory` | Show data inventory |

CLI entry point: `python -m pipeline.cli <subcommand>` or the installed `mdw`
script (`pyproject.toml:70`, `mdw = "pipeline.cli:main"`). All subcommands are
defined in `src/pipeline/cli.py`.

## Conventions

- **Type hints** throughout (PEP 604 unions like `date | None` are fine — we're on 3.11+).
- **Line length 100** (black + ruff, `pyproject.toml:76,80`). Ruff lint set:
  `E, F, I, N, W, UP, B, C4, SIM`.
- **Logging:** `logger = logging.getLogger(__name__)` at the module top.
  Rich is reserved for CLI output, not library code.
- **Config:** Pydantic `BaseSettings` in `src/pipeline/settings.py`, with
  `@model_validator(mode="after")` for runtime invariants. Prefer editing
  `config.yaml` + env vars over hard-coding.
- **Resilience:** wrap external calls with `tenacity` retries and, for hot
  paths, the circuit breakers in `src/pipeline/infrastructure/circuit_breaker.py`.
- **Time-correctness (non-negotiable):** any new extractor or transform must
  preserve the `event_time` vs `available_time` split. Never join on
  `event_time` alone — doing so silently reintroduces look-ahead bias.

## Gotchas — READ BEFORE TOUCHING BACKTEST OR STRATEGY CODE

- **Same-bar execution bias.** The backtest engine currently generates signals
  and executes fills on the same bar's close. See `CRITICAL_LIMITATION.md`.
  Treat backtest PnL as upper-bound noise, not ground truth.
- **47 statistical-robustness gaps** documented in `GAP_ANALYSIS.md`: no
  confidence intervals, no multiple-testing correction, no proper out-of-sample
  validation. Adding a "good" Sharpe from a fresh backtest proves nothing.
- **Do not deploy signals to live trading** without addressing both of the
  above. Paper trading on Alpaca is fine.
- **No bundled historical OHLCV.** `data/` only contains recent daily signals
  and a prediction-history JSON. Extractors fetch prices on demand — e.g. the
  `FLASH_CRASH_2010` stress scenario in `src/pipeline/eval/stress.py:42` is
  just a date range; the actual 2010 bars have to be pulled from an external
  source at runtime.

## Where to look first

- CLI & subcommands → `src/pipeline/cli.py`
- Settings / env / YAML wiring → `src/pipeline/settings.py`, `config.yaml`
- DB schemas → `src/sql/ddl/`, `src/sql/ddl_duckdb/`
- Historical backfill runbook → `HISTORICAL_BACKFILL.md` (copy-pasteable
  CLI commands for 2010-present, for running outside a restricted sandbox)
- Known issues → `CRITICAL_LIMITATION.md`, `GAP_ANALYSIS.md`, `AUDIT_REPORT.md`
- Strategy design → `STRATEGY_SPEC.md`, `STRATEGY_MEMO.md`
- Live-trading readiness → `LIVE_READINESS_CHECKLIST.md`
