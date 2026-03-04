# Backtest Engine Realism Assessment

**Document ID:** QSG-BACKTEST-001
**Version:** 1.0
**Date:** 2026-03-04

---

## 1. Architecture Overview

The system has two complementary backtest paths:

### 1.1 SwingStrategyEngine (`strategy/engine.py`, 560 lines)

End-to-end event-driven backtester for the QSG-MICRO-SWING-001 strategy. Processes daily OHLCV data chronologically, computing signals, checking entries/exits, and tracking positions at trade-level granularity.

**Components orchestrated:**
- `SignalEngine` → composite scoring
- `PositionSizer` → risk-based sizing
- `ExitEngine` → multi-trigger exits
- `SwingRiskManager` → drawdown circuit breakers
- `EdgeDecayMonitor` → strategy health monitoring

### 1.2 BacktestHarness (`strategy/backtest_harness.py`, 576 lines)

Modular backtest coordinator for arbitrary signal pipelines. Accepts pluggable signal pipelines, entry rules, exit engines, sizing models, and risk constraints.

**Components:**
- `SignalPipeline` → pluggable signal generation
- `EntryRuleSet` → AND-logic entry conditions (8 built-in rules)
- `ExitEngine` → same as swing engine
- `PositionSizingModel` → volatility-scaled or risk-based
- `RiskConstraintSet` → hard/soft/advisory constraints

### 1.3 Event-Driven Engine (`backtesting/event_engine.py`, 500 lines)

Priority-queue event processor for prediction market backtesting:
- Event priorities: MARKET_DATA(0) → RISK_CHECK(1) → TIMER(2) → ORDER(3) → FILL(4)
- Models maker/taker fees, latency, and order book simulation

---

## 2. Execution Assumptions

### 2.1 Order Types

| Order Type | Supported | Implementation |
|------------|-----------|----------------|
| Market-on-close | YES | Swing engine uses `close` price for entries/exits |
| Market order | YES | Event engine uses ask for buys, bid for sells |
| Limit order | PARTIAL | Event engine supports but not used in strategies |
| Stop order | IMPLICIT | Stop-loss logic in ExitEngine, not as order type |
| Stop-limit | NO | Not implemented |

### 2.2 Slippage Model

| Engine | Model | Default |
|--------|-------|---------|
| SwingStrategyEngine | Fixed half-spread | `spread_bps = 3.0` → 1.5 bps per side |
| BacktestHarness | Half-spread + slippage | `spread_bps=3.0` + `slippage_bps=2.0` |
| Event engine | Taker fee + latency | `taker_fee_bps=3.0`, `latency_ms=0` |

### 2.3 Transaction Costs

Two pluggable models in `backtesting/transaction_costs.py`:

**FixedPlusSpreadModel:**
```
cost = half_spread + commission
half_spread = notional × (spread_bps / 10,000 / 2)
commission = max(qty × per_share, min_commission)
```
Default: 5 bps spread, $0.005/share, $1.00 minimum

**SquareRootImpactModel (Almgren-Chriss):**
```
cost = spread + commission + market_impact
market_impact = σ × η × √(qty / ADV) × notional
```
Default: η=0.25, σ=0.02

### 2.4 Partial Fills and Non-Fills

| Feature | Status | Notes |
|---------|--------|-------|
| Partial fills | NOT MODELED | All orders fill completely or not at all |
| Volume participation cap | IMPLEMENTED | `simulator.py:76`: max_adv_pct=10% caps order size relative to ADV |
| Queue priority for limits | NOT MODELED | Limit orders fill if price touches |

---

## 3. Market Microstructure Realism

### 3.1 Fill Assumptions

| Aspect | Assessment |
|--------|------------|
| Fill price | Swing engine: exact close price; this is slightly optimistic but standard for daily strategies |
| Fill certainty | 100% fill at close — acceptable for liquid mega-caps with ADV > $500M |
| Mid-price assumption | NOT present — uses actual close, which is better than mid |
| Capacity constraint | ADV participation capped at 10% (`simulator.py`) |

### 3.2 Assessment

For the target universe (SPY, QQQ, AAPL, MSFT, etc. — all with ADV > $1B), the fill assumptions are **realistic**. These instruments trade billions of dollars daily at the close. A micro-capital account ($100–$1,000) or even institutional ($10M) would have zero market impact.

For the momentum strategy targeting smaller names, the `SquareRootImpactModel` properly penalizes large orders relative to ADV.

---

## 4. Time Handling

### 4.1 Signal Lag

| Engine | Lag | Implementation |
|--------|-----|----------------|
| BacktestHarness | Configurable (default: 1 day) | `composite_signals.shift(cfg.signal_lag_days)` at line 218 |
| SwingStrategyEngine | Implicit (same-bar) | Signals computed on bar `t`, entries at `close[t]`. This is a borderline same-bar execution — see §4.2 |
| Evaluator | Now correct | Fixed from `shift(-1)` to `pct_change()` |

### 4.2 Same-Bar Execution Concern

The swing engine computes signals from indicators at bar `t` and enters at `close[t]`. This assumes:
- All indicators are computed from data available at the close
- The trader can observe the close price and execute within the same closing auction

**Mitigation:** For daily bars on liquid US equities, this is a reasonable assumption — the closing auction on NYSE/NASDAQ allows market-on-close orders. However, a more conservative approach would enter at `open[t+1]`. The `BacktestHarness` addresses this with its explicit `signal_lag_days` parameter.

### 4.3 Market Hours

- Business day calendar via `pd.bdate_range` handles weekends
- Market close time: 16:00 ET (configured in `config.yaml`)
- No explicit handling of early closes or market holidays

---

## 5. Multi-Asset and Portfolio Handling

### 5.1 Portfolio Modeling

| Feature | SwingEngine | BacktestHarness | Simulator |
|---------|-------------|-----------------|-----------|
| Cash tracking | YES | YES | YES |
| Multiple positions | YES (1–4 by bracket) | YES (up to 50) | YES |
| Leverage | NO (long-only) | YES (max 2x) | YES (configurable) |
| Borrowing costs | NO | YES | YES (`borrow_cost_bps=30`) |
| Portfolio P&L | YES (equity curve) | YES (equity curve) | YES |
| Margin modeling | NO | NO | PARTIAL |
| Cross-currency | NO | NO | NO |

### 5.2 Risk Constraints

The `RiskConstraintSet` framework provides 13 constraint types at portfolio level:
- Position weight limits (hard)
- Sector exposure caps (hard)
- Gross/net exposure limits (hard)
- ADV participation limits (hard)
- Turnover limits (soft)
- Country exposure limits (soft)

---

## 6. Testing Infrastructure

### 6.1 Entry Points

| Entry Point | Command |
|-------------|---------|
| Swing backtest | `SwingStrategyEngine(config).run(price_data, spy_prices)` |
| Harness backtest | `BacktestHarness(...).run(price_data, universe, benchmarks)` |
| Event backtest | `EventEngine(config).run(events)` |
| CLI evaluate | `python -m pipeline.cli evaluate --scope equity ...` |
| Generate signals | `python -m pipeline.cli generate-signals --prices-dir ...` |

### 6.2 Automated Tests

| Test File | Coverage |
|-----------|----------|
| `test_strategy.py` | Signal engine, sizing, exits, risk, edge decay, engine integration |
| `test_backtesting.py` | Simulator, walk-forward, transaction costs |
| `test_backtesting_extended.py` | Extended backtesting scenarios |
| `test_signal_output.py` | Signal formatting, pre-trade checks, look-ahead regression |
| `test_eval_metrics.py` | Sharpe, Sortino, drawdown, calibration |

### 6.3 Edge Case Tests

- Empty universe (no tickers) — returns empty result
- No qualifying signals — returns empty trade log
- Extreme volatility data — ATR filters prevent entries
- Single-day data — handles gracefully

---

## 7. Gaps and Recommendations

| # | Gap | Severity | Recommendation |
|---|-----|----------|----------------|
| 1 | Same-bar execution in swing engine | MEDIUM | Add configurable `entry_delay` parameter (default 0 for MOC, 1 for next-day open) |
| 2 | No partial fill model | LOW | Acceptable for target liquidity universe; add if expanding to small-caps |
| 3 | No short-selling in swing strategy | LOW | By design (long-only micro-capital); momentum strategy supports via evaluator |
| 4 | No overnight gap model for stops | MEDIUM | Stops trigger at close, not at next open. For daily bars this is standard but should be documented |
| 5 | No exchange holiday calendar | LOW | Missing bars are handled safely (no trades) |
| 6 | No benchmark comparison in swing engine | LOW | BacktestHarness supports benchmarks; swing engine returns raw equity curve |
