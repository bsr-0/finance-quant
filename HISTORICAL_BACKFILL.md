# Historical Data Backfill — Runbook

Exact commands to populate the market data warehouse with historical data
from **2010-01-01 to present** for the 34 tickers configured in
`config.yaml:92-127`.

> **Why this is a separate doc:** the Claude Code sandbox has a strict
> egress-proxy allowlist (package managers + GitHub only). Yahoo Finance,
> FRED, Ken French, and SEC EDGAR are all blocked, so the existing
> extractors cannot run inside the sandbox. Run these commands on your own
> machine (or CI) where outbound HTTPS to those providers is allowed.

> **Note on the Makefile:** the `make extract-*` targets hardcode
> `2024-01-01`/`2024-12-31` (see `Makefile:82`, `89`, etc.). Do **not** use
> them for backfill — invoke the CLI directly with explicit `--start` and
> `--end`.

## 0. Prerequisites

```bash
# Python 3.11+ (pyproject.toml:10)
python --version   # must be >= 3.11

# Clone and checkout the branch
git clone <repo-url> finance-quant
cd finance-quant
git checkout claude/find-historical-data-2010-IrZKr

# Install the package (editable) + dev deps
pip install -e ".[dev,ml]"

# Initialize the DuckDB schema (zero-config, no server)
python -m pipeline.cli init-db
```

### Known-issue fix: `multitasking` wheel

`yfinance` depends on `multitasking`, whose source distribution fails to
build with modern `setuptools`. If `pip install -e .` errors on
`Failed to build multitasking`, install the prebuilt wheel first:

```bash
pip install --only-binary :all: multitasking
pip install -e ".[dev,ml]"
```

## 1. Tier 1 — Prices + Factors (no API keys)

Fetches Yahoo Finance daily OHLCV for the 34 tickers in `config.yaml` and
the Ken French factor library from Dartmouth. **Run sequentially — DuckDB
is single-writer, so running two extract commands concurrently will fail
with `IOException: Could not set lock on file`.**

```bash
START=2010-01-01
END=$(date +%Y-%m-%d)

# --- Prices (Yahoo Finance, ~34 tickers × ~16 years, 5-15 min) ---
python -m pipeline.cli extract prices --start "$START" --end "$END"
python -m pipeline.cli load-raw prices

# --- Ken French factors (Fama-French 3 & 5 factor, momentum) ---
python -m pipeline.cli extract factors
python -m pipeline.cli load-raw factors
```

**Expected output:** `data/raw/prices/<ticker>/*.parquet` (one file per
ticker), `data/raw/factors/*.parquet`, and rows loaded into
`prices_daily_raw` and `factors_raw` tables in
`data/market_data.duckdb`.

**If some tickers fail** (Yahoo rate-limits aggressively), re-run the
extract command — it's idempotent and will refetch only the missing
tickers. Tickers that don't exist for the full date range (e.g. `XLC`
listed 2018, `XLRE` 2015, `META` IPO 2012, `TSLA` IPO 2010, `V` IPO 2008)
will simply start from their inception date.

## 2. Tier 2 — Add FRED Macro (free API key)

Adds the 25 macro series in `config.yaml:22-59` (rates, CPI, VIX, credit
spreads, housing, labor, money supply, FX, financial conditions).

```bash
# One-time: get a free API key from https://fred.stlouisfed.org/docs/api/api_key.html
export FRED_API_KEY="your_key_here"

python -m pipeline.cli extract fred --start 2010-01-01 --end "$END"
python -m pipeline.cli load-raw fred
```

## 3. Tier 3 — SEC EDGAR + CFTC COT (no keys, slow)

SEC EDGAR is rate-limited to 10 req/s by User-Agent identification, so
these runs are slower. See `config.yaml:128-131`.

```bash
python -m pipeline.cli extract sec-fundamentals --start 2010-01-01 --end "$END"
python -m pipeline.cli load-raw sec_fundamentals

python -m pipeline.cli extract sec-insider --start 2010-01-01 --end "$END"
python -m pipeline.cli load-raw sec_insider

python -m pipeline.cli extract sec-13f --start 2010-01-01 --end "$END"
python -m pipeline.cli load-raw sec_13f

python -m pipeline.cli extract cftc-cot --start 2010-01-01 --end "$END"
python -m pipeline.cli load-raw cftc_cot
```

## 4. Shallow-history sources (skip for pre-2020 backfill)

These do not meaningfully extend before ~2015–2020 and will return empty
for most of the 2010-2019 range, but you can run them for the recent
window:

- `gdelt` — available only from 2015 onwards
- `polymarket` — exists from ~2020
- `options`, `earnings`, `reddit-sentiment`, `short-interest`,
  `etf-flows` — mostly current-state snapshots, not historical

## 5. Post-extract pipeline

After any tier of extraction, run the curated transforms, snapshots, and
data-quality checks:

```bash
python -m pipeline.cli transform-curated
python -m pipeline.cli build-snapshots --start 2010-01-01T00:00:00 --end "${END}T00:00:00" --freq 1d
python -m pipeline.cli dq
python -m pipeline.cli inventory
```

Or chain them via the Makefile target (it does not take dates, but reads
what's already in the warehouse):

```bash
make transform snapshots dq inventory
```

## 6. Verification

```bash
python -m pipeline.cli inventory
```

Or query DuckDB directly:

```bash
duckdb data/market_data.duckdb "
  SELECT 'prices_daily' AS table, COUNT(*) AS rows,
         MIN(event_time) AS first, MAX(event_time) AS last
  FROM prices_daily
  UNION ALL
  SELECT 'factors_ff', COUNT(*), MIN(event_time), MAX(event_time)
  FROM factors_ff;
"
```

Expected (Tier 1, full 2010-2026 backfill): roughly **130k–135k** price
rows (34 tickers × ~252 trading days × ~16 years, minus pre-inception
gaps) and **~4k** factor rows (daily Fama-French back to 2010).

## 7. Troubleshooting

- **`IOException: Could not set lock on file market_data.duckdb`** —
  another python process is already connected to DuckDB. Only one
  extractor at a time. Use `lsof data/market_data.duckdb` to find it.
- **`CircuitBreakerOpenError` on every ticker** — the Yahoo circuit
  breaker tripped after 5 consecutive failures (see
  `src/pipeline/infrastructure/circuit_breaker.py`). The root cause is
  upstream (Yahoo rate-limit, DNS, proxy). Fix the network issue, wait
  for the circuit breaker timeout, then re-run. A single `ProxyError` at
  the very start is a strong signal that outbound HTTPS to
  `query1.finance.yahoo.com` is blocked.
- **yfinance returns empty DataFrames** — Yahoo occasionally serves empty
  responses for no good reason. Re-run the same command; the extractor
  is idempotent.
- **SEC EDGAR 403** — make sure the `User-Agent` header is set (the
  extractor does this automatically from `sec_edgar` config, but some
  corporate proxies strip it).
- **FRED `API key missing`** — export `FRED_API_KEY` before running the
  `extract fred` command.

## 8. What NOT to do

- **Do not run two `extract` or `load-raw` commands in parallel** —
  DuckDB will raise `IOException: Could not set lock on file`.
- **Do not trust backtest PnL** after populating the warehouse until the
  same-bar execution bias in `CRITICAL_LIMITATION.md` is fixed. The data
  is fine; the backtest harness is what's biased.
- **Do not use `make extract-prices`** for backfill — its date range is
  hardcoded to 2024 in `Makefile:89-91`.
