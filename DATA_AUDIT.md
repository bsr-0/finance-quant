# Data Integrity and Bias Audit

**Document ID:** QSG-AUDIT-001
**Version:** 1.0
**Date:** 2026-03-04

---

## 1. Data Sources Inventory

| Source | Extractor | Data Type | Frequency | Lag |
|--------|-----------|-----------|-----------|-----|
| Yahoo Finance | `extract_prices` | Daily OHLCV | Daily | ~15 min after close |
| FRED | `extract_fred` | Macro indicators (28 series) | Monthly/quarterly | 1–60 days |
| SEC EDGAR | `extract_sec_fundamentals` | Quarterly financials | Quarterly | 60+ days |
| SEC EDGAR | `extract_sec_insider` | Insider trades | Event-driven | 2–5 days |
| SEC EDGAR | `extract_sec_13f` | Institutional holdings | Quarterly | 45 days |
| Fama-French | `extract_factors_ff` | Factor returns | Daily | T+1 (09:30 ET) |
| GDELT | `extract_gdelt` | World events | Continuous | ~180 min |
| Polymarket | `extract_polymarket` | Prediction markets | Continuous | ~2 min |
| Options chains | `extract_options` | IV, greeks, OI | Daily | Same day |
| Earnings calendar | `extract_earnings` | Report dates | Event-driven | Varies |
| Reddit sentiment | `extract_reddit_sentiment` | Social signals | Hourly | ~1 hour |
| Short interest | `extract_short_interest` | SI data | Bi-monthly | 10+ days |
| ETF flows | `extract_etf_flows` | Fund flows | Daily | T+1 |

**Implementation:** `src/pipeline/extract/` (12 extractor modules), `config.yaml`

---

## 2. Timestamp and Timezone Handling

### 2.1 Policy

- All raw data timestamps stored in **UTC** in the database
- Original timezones preserved in metadata columns
- Market close time: **16:00:00 America/New_York** (configurable in `config.yaml`)
- Macro data release: `available_time_release_time: "09:30"` + `available_time_lag_days: 1`

### 2.2 Assessment

| Check | Status | Evidence |
|-------|--------|----------|
| UTC storage | PASS | `raw_loader.py` converts all timestamps to UTC on ingestion |
| Timezone configuration | PASS | `settings.py:FactorSettings.exchange_timezone = "America/New_York"` |
| Market hours respected | PASS | `prices.market_close_time: "16:00:00"` in config.yaml |
| Holiday calendar | PARTIAL | Uses `pd.bdate_range` which handles US business days but not exchange-specific holidays (early closes, etc.) |
| Daylight savings | PASS | `America/New_York` handles DST transitions |

### 2.3 Remaining Risk

- Exchange-specific holidays (Christmas Eve early close, Good Friday) are not explicitly handled. Prices would simply be missing for those dates, which is safe (no phantom bars) but could affect lookback windows.

---

## 3. Point-in-Time Integrity

### 3.1 Look-Ahead Bias Checks

| Check | Status | Evidence |
|-------|--------|----------|
| Feature computation | PASS | All indicators use `.rolling()` and `.shift(1)` — no forward-looking windows. Verified in `signals.py`, `technical_indicators.py` |
| Signal lag in backtest harness | PASS | `backtest_harness.py:218`: `composite_signals.shift(cfg.signal_lag_days)` with default lag = 1 day |
| Signal lag in swing engine | PASS | Engine processes signals on current date but entries use `close` price, executed next day implicitly |
| Evaluator hit rate | **FIXED** | Was using `pct_change().shift(-1)` (future return). Fixed to `pct_change()` in `evaluator.py:354` |
| Walk-forward validation | PASS | `walk_forward.py` with expanding/rolling windows, embargo periods |
| Purged k-fold | PASS | Contiguous time blocks with embargo buffer after test set |
| Bias detection tests | PASS | `bias_checks.py`: random shuffle test, data shift test, timestamp ordering |

### 3.2 Issue Found and Fixed

**Critical:** `src/pipeline/eval/evaluator.py` line 354 used `pct_change().shift(-1)` which computed the *next* day's return and attributed it to the current day's signal. This inflated hit rate metrics. Fixed to `pct_change()` which correctly uses the current day's return (already forward-looking relative to the signal lag).

### 3.3 Fundamental Data Point-in-Time

- SEC filings use `filed_date` as the availability timestamp (not `period_end`), ensuring point-in-time correctness
- FRED macro data uses configurable `available_time_lag_days: 1` plus release time `09:30` ET
- Factor returns use `available_time_lag_days: 1` from `settings.py:FactorSettings`

**Implementation:** `src/pipeline/features/feature_asof.py` — as-of joins ensure features are only available after their publication time.

---

## 4. Corporate Actions Handling

### 4.1 Price Adjustments

| Aspect | Status | Evidence |
|--------|--------|----------|
| Split adjustment | PARTIAL | Yahoo Finance provides adjusted close; raw OHLC may be unadjusted depending on API parameters |
| Dividend adjustment | PARTIAL | Same as above — adjusted close accounts for dividends |
| Consistency check | IMPLEMENTED | `QUANT_FIXES.md` documents split/dividend adjustment ratio tracking |
| Mixed adjusted/unadjusted | RISK | If indicators are computed on adjusted close but stops use raw prices, there could be inconsistency |

### 4.2 Survivorship Bias

| Check | Status | Evidence |
|-------|--------|----------|
| Survivorship bias module | IMPLEMENTED | `src/pipeline/backtesting/survivorship.py` |
| DQ monitor check | IMPLEMENTED | `data_quality_monitor.py` checks for minimum % of delisted symbols |
| Historical constituents | NOT IMPLEMENTED | Universe uses current tickers (SPY, QQQ, etc.), not historical index constituents |

### 4.3 Recommendation

The current universe (20 liquid mega-caps and ETFs) has minimal survivorship risk because:
- ETFs (SPY, QQQ, IWM) don't get delisted
- Mega-caps (AAPL, MSFT, GOOGL) have been continuously listed
- However, if the universe is expanded to smaller names, historical constituent lists must be used

---

## 5. Missing Data and Outlier Treatment

### 5.1 Missing Data

| Source | Handling | Implementation |
|--------|----------|----------------|
| Missing OHLCV bars | Skipped (no phantom bars) | `engine.py` checks `date not in df.index` |
| Missing indicators | NaN propagation (safe) | Rolling windows produce NaN for warmup period |
| Missing volume | Defaults to 0 | Volume checks guard against division by zero |
| Missing SPY prices | Default BULL regime | `engine.py:174-175` falls back to "bull" |

### 5.2 Outlier Detection

| Method | Threshold | Implementation |
|--------|-----------|----------------|
| MAD-based z-score | 4.0 standard deviations | `data_quality_monitor.py:_check_price_anomalies()` |
| Return spike detection | 5.0σ z-score | `risk_controls.py:IntradayRiskMonitor._check_pnl_anomaly()` |
| Robust statistics | Median, MAD, Winsorization | `features/robust_stats.py` |

### 5.3 Data Quality Monitoring

- **Freshness:** Maximum age thresholds per table (48h prices, 2h contracts, 168h macro)
- **Completeness:** Minimum 95% non-null for required columns
- **Anomaly alerts:** Stored in `meta_data_quality_alerts` with severity levels (CRITICAL/ERROR/WARNING/INFO)
- **Pipeline gating:** CRITICAL alerts block pipeline progression (`cli.py:240-243`)

**Implementation:** `src/pipeline/dq/data_quality_monitor.py`

---

## 6. Reproducibility

### 6.1 Determinism Checks

| Aspect | Status | Evidence |
|--------|--------|----------|
| Fixed random seeds | PASS | Test fixtures use explicit seeds (`seed=42`) |
| Indicator computation | DETERMINISTIC | Pure rolling-window operations, no randomness |
| Signal scoring | DETERMINISTIC | Pure function of indicator values |
| Backtest execution | DETERMINISTIC | Fixed order of operations, no parallelism in main loop |
| Monte Carlo | CONTROLLABLE | `monte_carlo.py` accepts seed parameter |

### 6.2 Configuration Versioning

- Pipeline runs record `config_hash` (SHA-256 of serialized config) in `meta_pipeline_runs`
- Git SHA recorded per run via `get_git_sha()`
- Configuration stored as JSON in the pipeline run record

---

## 7. Summary of Issues and Fixes

| # | Severity | Issue | Fix | File |
|---|----------|-------|-----|------|
| 1 | **CRITICAL** | Forward-return look-ahead in evaluator hit rate | Changed `pct_change().shift(-1)` to `pct_change()` | `eval/evaluator.py:354` |
| 2 | LOW | No exchange holiday calendar | Prices simply missing (safe but reduces lookback) | N/A — acceptable |
| 3 | LOW | Historical index constituents not used | Acceptable for current mega-cap universe | N/A |
| 4 | MEDIUM | Adjusted vs unadjusted price consistency | Document and enforce single policy | Config-level |
| 5 | LOW | Yahoo Finance single-source risk | API failure = no signals (fail-safe) | By design |
