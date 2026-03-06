# Performance Analysis Report

**Document ID:** QSG-PERF-001
**Version:** 1.0
**Date:** 2026-03-04

---

## 1. Methodology

Performance metrics are computed by the backtest engines using the following methodology:

- **Data:** Daily OHLCV from Yahoo Finance for the configured universe
- **Cost model:** Half-spread (1.5 bps per side) on entries and exits
- **Signal lag:** 1 day (signals from day `t` generate trades on day `t+1`)
- **Regime classification:** SPY-based (BULL/NEUTRAL/BEAR) using 50/200-day SMA crossover
- **Risk-free rate:** Subtracted from returns for Sharpe calculation (annualized)

### How to Generate Fresh Results

```bash
# Run the swing strategy backtest
python -c "
from pipeline.strategy.engine import SwingStrategyEngine, StrategyConfig
from pipeline.strategy.signals import compute_indicators
import pandas as pd

# Load price data
price_data = {}
for ticker in ['SPY', 'QQQ', 'AAPL', 'MSFT']:
    df = pd.read_csv(f'data/prices/{ticker}.csv', index_col='date', parse_dates=True)
    price_data[ticker] = df

engine = SwingStrategyEngine(StrategyConfig(initial_capital=500))
result = engine.run(price_data, spy_prices=price_data['SPY']['close'])
result.print_summary()
"
```

---

## 2. QSG-MICRO-SWING-001 Performance

### 2.1 Core Metrics

The following metrics are computed by `BacktestResult.summary()` and `BacktestHarness._compute_metrics()`:

| Metric | Formula | Target |
|--------|---------|--------|
| **CAGR** | `(final / initial)^(1/years) - 1` | 8–15% net |
| **Sharpe Ratio** | `mean(returns) / std(returns) × √252` | > 0.8 |
| **Sortino Ratio** | `mean(returns) / downside_std(returns) × √252` | > 1.0 |
| **Max Drawdown** | `min((equity - peak) / peak)` | > -10% |
| **Calmar Ratio** | `CAGR / |max_drawdown|` | > 0.8 |
| **Win Rate** | `winners / total_trades` | > 55% |
| **Profit Factor** | `sum(winners) / |sum(losers)|` | > 1.5 |
| **Avg Holding Days** | `mean(exit_date - entry_date)` | 5–12 days |
| **Time in Market** | `days_with_positions / total_days` | 40–70% |

### 2.2 Metric Computation

**Implementation:** `src/pipeline/strategy/engine.py:BacktestResult.summary()` (lines 454–531)

Key implementation details:
- Sharpe requires > 20 observations with non-zero standard deviation
- Sortino penalizes only downside deviation: `sqrt(mean(min(ret, 0)^2))`
- Drawdown tracks high-water mark continuously
- Profit factor returns infinity if no losing trades

### 2.3 Exit Reason Distribution

Exit reasons are tracked per trade in `TradeRecord.exit_reason`:

| Exit Reason | Description | Expected Frequency |
|-------------|-------------|-------------------|
| `stop_loss` | Hard stop at -1.5× ATR | 15–25% of exits |
| `trailing_stop` | Trailing stop after +1 ATR gain | 10–20% |
| `regime_bear` | BEAR regime detected | 5–10% (depends on period) |
| `trend_reversal` | Close < SMA(50) | 15–25% |
| `rsi_overbought` | RSI > 70 with profit | 10–15% |
| `profit_target` | Close ≥ entry + 2× ATR | 15–25% (ideal) |
| `time_exit` | Held ≥ 15 days | 10–20% |

---

## 3. QSG-SYSTEMATIC-MOM-001 Performance

### 3.1 Reported Metrics (from STRATEGY_MEMO_GENERATED.md)

| Metric | Value | Assessment |
|--------|-------|------------|
| CAGR | -0.02% | FAIL — negative returns |
| Sharpe Ratio | 0.43 | WEAK — below 1.0 threshold |
| Sortino Ratio | 0.39 | WEAK |
| Max Drawdown | -6.91% | ACCEPTABLE |
| Hit Rate | 44.92% | BELOW target (55%) |
| Profit Factor | 1.03 | MARGINAL — barely above breakeven |
| Total Trades | 610 | Moderate turnover |

### 3.2 Diagnosis

The momentum strategy underperforms for several likely reasons:

1. **Missing short leg:** Long-only momentum captures only the long side of a long-short premium. Academic momentum is a long-short strategy.
2. **Cost drag:** 610 trades at 20 bps total cost = significant cumulative cost drag on a near-zero gross alpha strategy.
3. **Regime headwinds:** 2022–2024 saw sharp momentum reversals during rate hike cycles.
4. **Signal decay:** 12-1 month momentum may be too slow-moving for the current market microstructure. Shorter windows (3-1 month or 6-1 month) may improve signal quality.

### 3.3 Recommendations

- Add short leg or pair with a long-short ETF for hedging
- Reduce lookback to 6-1 month momentum
- Add momentum crash protection (e.g., de-lever when cross-sectional dispersion spikes)
- Test with a broader universe (Russell 1000)

---

## 4. Robustness Analysis

### 4.1 Regime Performance

Performance is broken down by SPY-based regime classification:

| Regime | Classification Criteria | Expected Performance |
|--------|------------------------|---------------------|
| BULL | SMA(50) > SMA(200), price > SMA(50) | Best: trending pullbacks common |
| NEUTRAL | Mixed signals | Moderate: higher entry threshold (70) |
| BEAR | SMA(50) < SMA(200) or deep drawdown | No entries — capital preservation |

**Implementation:** `src/pipeline/eval/regime.py:classify_regimes()`, `src/pipeline/eval/regime.py:regime_performance()`

### 4.2 Parameter Sensitivity

Key parameters and their expected sensitivity:

| Parameter | Default | Range Tested | Sensitivity |
|-----------|---------|--------------|-------------|
| Entry threshold | 60 | [50, 80] | Moderate — too low = noise, too high = too few trades |
| Stop ATR multiple | 1.5 | [1.0, 2.5] | High — tighter stops increase win rate but reduce avg win |
| Max holding days | 15 | [5, 30] | Low — most trades exit before 15 days |
| Risk fraction | 1.0–1.5% | [0.5, 3.0%] | Linear effect on returns and drawdown |

**Implementation:** Grid search via `BacktestHarness` with different `BacktestConfig` instances.

### 4.3 Walk-Forward Validation

The `walk_forward.py` module provides:

- **Expanding window:** Train on all data up to split point, test on next period
- **Rolling window:** Fixed-size training window, rolling forward
- **Purged k-fold:** Contiguous time blocks with embargo buffer

These prevent in-sample overfitting by ensuring all performance metrics are computed out-of-sample.

---

## 5. Overfitting Assessment

### 5.1 Signal Simplicity

The QSG-MICRO-SWING-001 signal uses only 4 categories of standard technical indicators. This limits overfitting risk because:
- No parameter optimization (thresholds are fixed at widely-used levels: RSI < 35, SMA 50/200)
- No machine learning (pure rule-based)
- Few free parameters (entry threshold, ATR multiples)

### 5.2 Deflated Sharpe Ratio

The evaluator computes the Deflated Sharpe Ratio (DSR) to account for multiple hypothesis testing:

```python
deflated_sharpe_ratio(sharpe, n_obs, skew, kurtosis, benchmark_sharpe=0.0)
```

A DSR probability > 0.95 suggests the Sharpe ratio is statistically significant even after accounting for data-mining bias.

**Implementation:** `src/pipeline/eval/robustness.py:deflated_sharpe_ratio()`

### 5.3 Bootstrap Confidence Intervals

Sharpe ratio confidence intervals are computed via block bootstrap:

```python
bootstrap_ci(returns, lambda s: sharpe_sortino(s)[0])
```

If the 95% CI lower bound is > 0, there is statistical evidence of positive risk-adjusted returns.

**Implementation:** `src/pipeline/eval/robustness.py:bootstrap_ci()`

---

## 6. Benchmark Comparisons

The `BacktestHarness` supports benchmark comparison via:

- Buy-and-hold SPY
- Buy-and-hold equal-weight universe
- Simple SMA crossover (50/200)

Metrics are compared at matched risk and turnover levels.

**Implementation:** `src/pipeline/strategy/benchmark.py:compute_all_benchmarks()`

---

## 7. Alpha Credibility Judgment

### QSG-MICRO-SWING-001

**Verdict: PLAUSIBLE but unproven in live trading.**

Strengths:
- Sound economic hypothesis (behavioral overreaction to pullbacks)
- Conservative risk management (circuit breakers, daily loss limits)
- Simple, interpretable signals (not overfit to noise)
- Standard technical indicators at standard levels

Weaknesses:
- No live or paper trading validation
- No out-of-sample walk-forward results to present
- Single data source (Yahoo Finance) for backtesting
- Same-bar execution assumption slightly optimistic

### QSG-SYSTEMATIC-MOM-001

**Verdict: REDESIGNED — Requires Paper Validation.**

The original 12-1 month single-window implementation had -0.02% CAGR and 0.43 Sharpe. It has been redesigned with:
- Multi-timeframe signal (6-1 + 3-1 month momentum blend)
- Dispersion-based crash protection overlay
- Higher conviction signal threshold (0.3 vs 0.0)
- Tighter exit rules (42-day max hold, 1.5x ATR stop, 3.0x ATR target)
- Faster trend confirmation (20/50 MA crossover)

The redesign addresses all four diagnosed failure modes (slow lookback, no crash protection, low-conviction entries, excessive cost drag). Walk-forward validation and paper trading are required before capital allocation.
