# Quantitative Engineer's Fixes to Historical Data Pipelines

## Executive Summary

As a skeptical quantitative engineer reviewing this codebase, I've identified **critical flaws** that would render any prediction model trained on this data unreliable. This document details the issues found and the robust fixes implemented.

---

## Critical Issues Found

### 1. **SURVIVOR BIAS - CRITICAL** 🔴

**Problem:** The original code marks ALL symbols as `is_delisted = false`:

```python
# FROM: transform/curated.py (line 263-270)
INSERT INTO dim_symbol (..., is_delisted)
SELECT DISTINCT 
    r.ticker,
    ...
    false as is_delisted  # <-- ALWAYS FALSE!
```

**Impact:** Your backtests only see companies that survived. Failed companies (Enron, Lehman, etc.) are excluded, inflating performance by 2-5% annually.

**Fix:** `prices_daily_fixed.py` now:
- Detects delistings by checking for 30+ days of missing data
- Tracks `delisted_date` for each ticker
- Records universe composition statistics
- Provides `build_survivor_unbiased_universe()` for point-in-time universe construction

```python
# Detect delisting
days_since_last_trade = (datetime.now() - last_date).days
if days_since_last_trade > 30:
    ticker_info.is_delisted = True
    ticker_info.delisted_date = last_date.date()
```

---

### 2. **CORPORATE ACTIONS IGNORED - CRITICAL** 🔴

**Problem:** The original code fetches splits/dividends from Yahoo but **discards them**:

```python
# FROM: prices_daily.py (line 50)
params = {
    "events": "history,div,splits",  # Fetched but ignored!
    ...
}
# ... never processes events
```

**Impact:** 
- Split-adjusted prices appear to have massive jumps
- Dividend-adjusted returns are wrong
- Technical indicators computed on unadjusted data are meaningless

**Fix:** `prices_daily_fixed.py` now:
- Extracts and stores all corporate actions
- Computes adjustment ratios
- Provides `apply_point_in_time_adjustments()` for backtesting

```python
# Extract corporate actions
splits = result.get("events", {}).get("splits", {})
dividends = result.get("events", {}).get("dividends", {})

# Calculate adjustment factor
adj_factor = 1.0
for action in actions_up_to_asof_date:
    if action.action_type == "split":
        adj_factor *= action.ratio
```

---

### 3. **NO DATA QUALITY FLAGS - HIGH** 🟡

**Problem:** No tracking of data quality issues:
- Zero volume days
- OHLC logic errors (high < low)
- Suspicious price spikes (>50% daily change)
- Missing values silently accepted

**Fix:** `curated_fixed.py` now:
- Flags data quality issues during extraction
- SQL anomaly detection for price/volume spikes
- Separate handling for quality-flagged records

```python
# Flag suspicious data
df.loc[df["volume"] == 0, "data_quality_flag"] = "zero_volume"
df.loc[df["high"] < df["low"], "data_quality_flag"] = "ohlc_error"
df.loc[df["close"] <= 0, "data_quality_flag"] = "invalid_price"

# SQL anomaly detection
UPDATE cur_prices_ohlcv_daily
SET data_quality_flag = 'price_spike'
WHERE ABS(close / LAG(close) OVER (PARTITION BY symbol_id ORDER BY date) - 1) > 0.50
```

---

### 4. **NO STALENESS DETECTION - HIGH** 🟡

**Problem:** Snapshots use data without checking freshness:

```python
# FROM: contract_snapshots.py (line 81)
macro_panel = self._get_macro_panel(asof_ts)
# No check if macro data is days/weeks old!
```

**Impact:** 
- Trading on stale macro data
- Features computed with outdated information
- Model performance degrades silently

**Fix:** `contract_snapshots_fixed.py` now:
- Tracks `price_staleness_hours` and `macro_staleness_days`
- Warns when data is >24 hours stale
- Includes staleness in `data_quality_score`

```python
quality_report = self._get_staleness_metrics(contract_id, asof_ts)
snapshot["price_staleness_hours"] = quality_report.price_staleness_hours
snapshot["macro_staleness_days"] = quality_report.macro_staleness_days

if quality_report.price_staleness_hours > 24:
    logger.warning(f"Price data is {quality_report.price_staleness_hours:.1f} hours stale")
```

---

### 5. **NO OUTLIER DETECTION - HIGH** 🟡

**Problem:** Price outliers (fat-finger trades, data errors) are included in features:

```python
# FROM: contract_snapshots.py (line 147-152)
SELECT 
    COALESCE(STDDEV(price), 0) as price_std  # Outliers inflate std!
```

**Impact:**
- Volatility features meaningless
- Technical indicators skewed
- Model trained on bad data

**Fix:** `contract_snapshots_fixed.py` now:
- Multiple outlier detection methods (IQR, Z-score, MAD)
- Flags outliers in snapshot
- Uses robust statistics (median, MAD) instead of mean/std

```python
def _detect_price_outliers(self, prices: pd.Series, method: str = "mad"):
    # Median Absolute Deviation - more robust than Z-score
    median = prices.median()
    mad = np.median(np.abs(prices - median))
    modified_z = 0.6745 * (prices - median) / mad
    outlier_mask = np.abs(modified_z) > 3.5
    return outlier_mask, outlier_score
```

---

### 6. **NO MICROSTRUCTURE FEATURES - MEDIUM** 🟠

**Problem:** Only basic price/volume features, ignoring market microstructure:

**Impact:**
- Missing trade direction (buyer/seller initiated)
- No trade size analysis
- No price impact measurement
- Can't detect informed trading

**Fix:** `contract_snapshots_fixed.py` now includes:
- `trade_imbalance`: Buy vs sell volume ratio
- `buy_sell_ratio`: Buy volume / sell volume
- `price_impact`: Correlation between trade size and price change
- `avg_trade_size`, `trade_size_variance`

```python
def _calculate_microstructure_features(self, trades_df: pd.DataFrame):
    buy_volume = trades_df[trades_df["side"] == "buy"]["size"].sum()
    sell_volume = trades_df[trades_df["side"] == "sell"]["size"].sum()
    trade_imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume)
    
    # Price impact
    trades_df["price_change"] = trades_df["price"].diff()
    price_impact = trades_df["size"].corr(trades_df["price_change"].abs())
    
    return {
        "trade_imbalance": trade_imbalance,
        "price_impact": price_impact,
        ...
    }
```

---

### 7. **MACRO DATA WITHOUT VINTAGES - MEDIUM** 🟠

**Problem:** Macro data uses `extracted_at` as `available_time`, ignoring FRED's `realtime_start`:

```python
# FROM: transform/curated.py (line 91-93)
available_time = r.extracted_at  # Wrong! Should use realtime_start
```

**Impact:**
- Using revised macro data that wasn't available at the time
- Look-ahead bias in macro features
- Backtests unrealistically optimistic

**Fix:** `curated_fixed.py` now:
- Uses FRED's `realtime_start` when available
- Tracks vintage information
- Falls back to `extracted_at` only when necessary

```python
available_time = COALESCE(r.realtime_start, r.extracted_at) as available_time,
CASE 
    WHEN r.realtime_start IS NOT NULL THEN 'confirmed'
    ELSE 'assumed'
END as time_quality
```

---

### 8. **NO DATA QUALITY MONITORING - MEDIUM** 🟠

**Problem:** No systematic monitoring of data quality issues.

**Fix:** `data_quality_monitor.py` provides:
- Freshness checks (alert if data > X hours old)
- Completeness checks (alert if null rate > threshold)
- Price anomaly detection (Z-score > 4)
- Survivor bias monitoring
- Look-ahead bias detection

```python
def check_look_ahead_bias(self, table_name: str) -> Optional[DataQualityAlert]:
    result = self.db.run_query(f"""
        SELECT s.contract_id, s.asof_ts, p.available_time
        FROM {table_name} s
        JOIN cur_contract_prices p ON s.contract_id = p.contract_id
        WHERE p.available_time > s.asof_ts  # CRITICAL: Future data!
          AND p.ts <= s.asof_ts
    """)
    if result:
        return DataQualityAlert(
            severity=Severity.CRITICAL,
            message=f"Found {len(result)} instances of look-ahead bias!"
        )
```

---

## New Files Added

| File | Purpose |
|------|---------|
| `extract/prices_daily_fixed.py` | Fixed price extraction with corporate actions |
| `transform/curated_fixed.py` | Fixed transforms with survivor bias handling |
| `snapshot/contract_snapshots_fixed.py` | Fixed snapshots with staleness/outlier detection |
| `dq/data_quality_monitor.py` | Data quality monitoring and alerting |
| `sql/ddl/10_data_quality_tables.sql` | SQL schema for quality tracking |

---

## How to Use the Fixed Pipelines

### 1. Extract with Corporate Actions

```python
from pipeline.extract.prices_daily_fixed import extract_prices_fixed

results = extract_prices_fixed(
    output_dir=Path("data/raw"),
    tickers=["AAPL", "MSFT", "LEHMQ"],  # Include delisted!
    start_date="2000-01-01",
    end_date="2024-12-31"
)
# Returns: prices, corporate_actions, ticker_info, universe_stats
```

### 2. Transform with Survivor Bias Tracking

```python
from pipeline.transform.curated_fixed import CuratedTransformerFixed

transformer = CuratedTransformerFixed()
results = transformer.transform_all_fixed()

# Check survivor bias
bias_stats = transformer.db.run_query("""
    SELECT 
        COUNT(*) as total,
        SUM(CASE WHEN is_delisted THEN 1 ELSE 0 END) as delisted
    FROM dim_symbol
""")
print(f"Survivor bias: {bias_stats[0]['delisted']}/{bias_stats[0]['total']} delisted")
```

### 3. Build Snapshots with Quality Checks

```python
from pipeline.snapshot.contract_snapshots_fixed import ContractSnapshotBuilderFixed

builder = ContractSnapshotBuilderFixed()
snapshot = builder.build_contract_snapshot_fixed(
    contract_id=uuid,
    asof_ts=datetime(2024, 11, 1, 12, 0, 0)
)

# Check quality
print(f"Data quality score: {snapshot['data_quality_score']}")
print(f"Price staleness: {snapshot['price_staleness_hours']:.1f} hours")
print(f"Has outliers: {snapshot['has_price_outliers']}")
```

### 4. Monitor Data Quality

```python
from pipeline.dq.data_quality_monitor import run_quality_monitor

report = run_quality_monitor()
# Prints summary of all quality checks
```

---

## Data Quality Score

Each snapshot now includes a `data_quality_score` (0-100):

| Deduction | Condition |
|-----------|-----------|
| -30 max | Price staleness > 1 hour |
| -20 max | Macro staleness > 1 day |
| -20 max | Outliers detected |
| -10 | Missing microstructure features |

**Usage:** Filter snapshots with `data_quality_score >= 80` for model training.

---

## Backtesting Checklist

Before running any backtest, verify:

- [ ] **Survivor bias handled**: Include delisted tickers in universe
- [ ] **Point-in-time adjustments**: Use `apply_point_in_time_adjustments()`
- [ ] **No look-ahead**: All features use `available_time <= asof_ts`
- [ ] **Data freshness**: Check `staleness_hours < 24`
- [ ] **Outliers handled**: Filter or winsorize extreme values
- [ ] **Quality score**: Only use snapshots with score >= 80

---

## Performance Impact

| Metric | Before | After | Notes |
|--------|--------|-------|-------|
| Survivor bias | 0% delisted | Tracked | Critical fix |
| Corporate actions | Ignored | Tracked | Critical fix |
| Data staleness | Unknown | Monitored | High value |
| Outliers | Included | Flagged | High value |
| Quality score | N/A | 0-100 | New feature |

---

## Conclusion

The original codebase had **fundamental flaws** that would produce misleading backtests and unreliable models. These fixes ensure:

1. **Accurate backtests** with proper survivor bias handling
2. **Correct price adjustments** for splits/dividends
3. **Fresh data** with staleness monitoring
4. **Clean features** with outlier detection
5. **Observable quality** with scoring and alerting

**Recommendation:** Do not use the original pipelines for production trading. Use the fixed versions with quality checks enabled.
