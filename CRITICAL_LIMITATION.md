# Critical Limitation: Same-Bar Execution in Backtest Engine

**Severity:** Critical
**Impact:** Backtest results are unreliable; live trading performance will diverge significantly
**Location:** `src/pipeline/strategy/engine.py` lines 304-312

---

## The Problem

The backtest engine uses today's close price to both **generate the signal** and **execute the entry** on the same bar. This is a textbook look-ahead / execution-impossibility bias.

### What the code does

```
# engine.py:304-312 (inside the daily loop for each date)
row = sym_df.loc[date]                                          # today's bar
score_total, ... = self.signal_engine._score_row(row)           # score uses close, RSI, SMA, etc.
...
entry_price = row["close"]                                      # enter at today's close
```

The signal scoring function (`signals.py:115-169`) reads `close`, `sma_50`, `sma_200`, `rsi_14`, `bb_lower`, `stoch_k`, `volume`, `obv_slope`, `atr_pct`, `macd_hist`, and `williams_r` -- all computed from today's close. The engine then enters a position at that same close price.

### Why this is impossible in practice

1. **You cannot observe the close and transact at the close.** The daily close is only known after the market closes. You cannot submit an order that executes at a price you haven't yet observed.
2. **The signal incorporates the close price into its decision.** SMA, RSI, Bollinger Bands, Stochastic, MACD, and ATR all use today's close. If the close price is what triggers the signal, you've already missed the opportunity to buy at that price.
3. **The exit engine has the same flaw.** Exits check today's close, RSI, and SMA(50) and then sell at today's close (`engine.py:269`).

### How it inflates performance

- **Favorable selection bias:** Bars where close triggers a buy signal are, by definition, bars where technical conditions are favorable at that exact price. Buying at next-open introduces gap risk and often a worse fill.
- **Eliminated overnight gap risk:** In real swing trading, the overnight gap between signal-day close and execution-day open is a major source of slippage (often 0.3-1.0% for ETFs, more for single stocks).
- **Elimination of adverse fills on exits:** Stop-loss exits in the backtest fire at close, but real stops fire intraday at the stop price or worse (gaps through stops are common).

### Documented vs. actual behavior

The `PERFORMANCE_REPORT.md` states:

> **Signal lag:** 1 day (signals from day `t` generate trades on day `t+1`)

But the engine code does **not** implement this lag. Signals from day `t` generate trades on day `t` at day `t`'s close. The `PortfolioSimulator` (`backtesting/simulator.py:62-67`) explicitly warns about this in its docstring, but the `SwingStrategyEngine` does not heed it.

---

## Estimated Impact

Conservative estimate of performance inflation from same-bar execution:

| Metric | Likely Overstatement |
|--------|---------------------|
| Win rate | +5-10 percentage points |
| Sharpe ratio | +0.2-0.5 |
| CAGR | +3-8 percentage points |
| Max drawdown | understated by 2-5 pp |

These estimates are based on published literature on execution delay impact in daily-frequency momentum/reversion strategies (e.g., Harvey et al. 2016, Novy-Marx & Velikov 2016).

---

## Recommended Fix

In `SwingStrategyEngine.run()`, separate signal generation from execution by one bar:

```python
# Instead of scoring and entering on the same date:
#   row = sym_df.loc[date]
#   score = signal_engine._score_row(row)
#   entry_price = row["close"]

# Buffer signals from day t, execute at day t+1 open:
pending_signals: dict[str, SignalScore] = {}

for date in dates:
    # 1. Execute yesterday's pending signals at today's open
    for sym, sig in pending_signals.items():
        entry_price = indicator_data[sym].loc[date, "open"]
        # ... proceed with sizing and entry at open price

    pending_signals.clear()

    # 2. Generate today's signals (will execute tomorrow)
    for sym, sym_df in indicator_data.items():
        row = sym_df.loc[date]
        score = self.signal_engine._score_row(row)
        if eligible:
            pending_signals[sym] = score
```

Apply the same next-bar-open logic to exits: flag exits at close, execute at next open.

---

## Scope of the Problem

This same-bar execution bias affects:

1. **`SwingStrategyEngine.run()`** -- the primary backtest path
2. **All signal CSV outputs** (e.g., `data/signals/signals_*.csv`) -- entry_price is today's close, but a real execution would happen at next-day open
3. **`TradingRunner`** (`execution/runner.py`) -- reads signal CSVs and executes, but the entry_price baked into the CSV is the same-bar close
4. **All downstream metrics** in `PERFORMANCE_REPORT.md`, `BacktestResult.summary()`, walk-forward validation, and Monte Carlo simulations
5. **The daily-predictions CI pipeline** -- publishes signals with entry prices that assume same-bar execution

No other limitation in this repository (data quality, model complexity, universe size, infrastructure) matters until this is fixed, because all performance metrics are computed on a biased equity curve.
