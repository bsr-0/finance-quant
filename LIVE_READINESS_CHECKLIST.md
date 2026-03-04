# Live Trading Readiness Checklist

**Document ID:** QSG-LIVE-001
**Version:** 1.0
**Date:** 2026-03-04

---

## Scoring: 0–3 per criterion

| Score | Meaning |
|-------|---------|
| 0 | Not implemented |
| 1 | Partial — exists but incomplete or untested |
| 2 | Implemented — needs live environment testing |
| 3 | Production-ready — tested and documented |

---

## 1. Data Infrastructure

| # | Criterion | Score | Notes |
|---|-----------|-------|-------|
| 1.1 | Real-time price feed | **0** | No live data integration. Yahoo Finance is batch-only. |
| 1.2 | Data quality monitoring | **2** | `data_quality_monitor.py` monitors freshness, completeness, anomalies. Needs live environment testing. |
| 1.3 | Survivorship bias handling | **2** | `survivorship.py` implemented. Current universe (mega-caps) has minimal risk. |
| 1.4 | Corporate actions pipeline | **1** | Relies on Yahoo Finance adjusted prices. No explicit split/dividend event processing. |
| 1.5 | Data backup and recovery | **1** | PostgreSQL storage with run_id tracking. No explicit backup strategy. |

**Subtotal: 6 / 15**

---

## 2. Signal Generation

| # | Criterion | Score | Notes |
|---|-----------|-------|-------|
| 2.1 | Signal computation pipeline | **3** | `SignalEngine` fully implemented with 4-category composite scoring. |
| 2.2 | Signal lag enforcement | **3** | `BacktestHarness` applies `signal_lag_days`. Fixed evaluator look-ahead bug. |
| 2.3 | Pre-trade checks | **2** | `pre_trade_checks.py` validates tradability, liquidity, risk limits. New module — needs live testing. |
| 2.4 | Standardized output format | **2** | `signal_output.py` writes `signals_YYYYMMDD.csv` with full trade blotter. New module. |
| 2.5 | CLI entry point | **2** | `generate-signals` command added to CLI. |

**Subtotal: 12 / 15**

---

## 3. Execution

| # | Criterion | Score | Notes |
|---|-----------|-------|-------|
| 3.1 | Broker API integration | **0** | No broker connection (IBKR, Alpaca, etc.) |
| 3.2 | Order management system | **0** | No live order submission, tracking, or reconciliation. |
| 3.3 | Fill confirmation | **0** | No live fill processing. |
| 3.4 | Position reconciliation | **0** | No end-of-day position comparison between system and broker. |
| 3.5 | Execution cost monitoring | **2** | Transaction cost models implemented for backtesting. Not connected to live fills. |

**Subtotal: 2 / 15**

---

## 4. Risk Management

| # | Criterion | Score | Notes |
|---|-----------|-------|-------|
| 4.1 | Kill switch | **3** | Thread-safe `KillSwitch` in `risk_controls.py`. Tested. |
| 4.2 | Drawdown circuit breakers | **3** | 4-level system (GREEN/YELLOW/ORANGE/RED). Fully implemented and tested. |
| 4.3 | Daily loss limits | **2** | Added to `SwingRiskManager.get_risk_state()`. Blocks entries when daily loss > 2%. |
| 4.4 | Position sizing constraints | **3** | Risk-based sizing with bracket system, conviction scaling, regime adjustment. |
| 4.5 | Correlation monitoring | **2** | `correlation_monitor.py` with clustering and factor exposure tracking. Needs integration testing. |

**Subtotal: 13 / 15**

---

## 5. Monitoring and Logging

| # | Criterion | Score | Notes |
|---|-----------|-------|-------|
| 5.1 | Structured logging | **3** | JSON formatter with configurable output. All modules use standard Python logging. |
| 5.2 | Edge decay monitoring | **3** | 6 metrics, 3 alert levels (YELLOW/ORANGE/RED). Monthly breach tracking. |
| 5.3 | Performance attribution | **2** | Equity curve, trade log, exit reason tracking. No real-time P&L streaming. |
| 5.4 | Risk dashboard | **1** | `risk_dashboard.py` exists but needs web UI integration. |
| 5.5 | Alert notifications | **0** | No email/Slack/PagerDuty integration for alerts. |

**Subtotal: 9 / 15**

---

## 6. Backtesting and Validation

| # | Criterion | Score | Notes |
|---|-----------|-------|-------|
| 6.1 | Walk-forward validation | **3** | Expanding/rolling window with purged k-fold and embargo periods. |
| 6.2 | Look-ahead bias prevention | **3** | `bias_checks.py` (4 tests), signal lag, fixed evaluator bug. |
| 6.3 | Transaction cost modeling | **3** | Two models: FixedPlusSpread, Almgren-Chriss SquareRootImpact. |
| 6.4 | Monte Carlo simulation | **2** | Block bootstrap and execution stress tests. Needs validation on live data. |
| 6.5 | Capacity analysis | **2** | `capacity.py` estimates max AUM and participation rates. |

**Subtotal: 13 / 15**

---

## 7. Testing

| # | Criterion | Score | Notes |
|---|-----------|-------|-------|
| 7.1 | Unit tests | **3** | 24 test files covering signals, sizing, exits, risk, metrics, indicators. |
| 7.2 | Integration tests | **2** | `test_strategy.py` runs full engine. No end-to-end pipeline test with DB. |
| 7.3 | Regression tests | **2** | Signal and metric tests act as regression guards. No baseline comparison. |
| 7.4 | Edge case tests | **2** | Empty universe, extreme vol, single-day data handled. |

**Subtotal: 9 / 12**

---

## 8. Documentation

| # | Criterion | Score | Notes |
|---|-----------|-------|-------|
| 8.1 | Strategy specification | **3** | STRATEGY_SPEC.md with full formalization. |
| 8.2 | Data audit | **3** | DATA_AUDIT.md with source inventory, bias checks, fixes. |
| 8.3 | Backtest review | **3** | BACKTEST_ENGINE_REVIEW.md with realism assessment. |
| 8.4 | Risk framework | **3** | RISK_FRAMEWORK.md with full parameter tables. |
| 8.5 | Trader summary | **3** | TRADER_SUMMARY.md — 10-minute operational guide. |

**Subtotal: 15 / 15**

---

## Overall Score

| Category | Score | Max | % |
|----------|-------|-----|---|
| Data Infrastructure | 6 | 15 | 40% |
| Signal Generation | 12 | 15 | 80% |
| Execution | 2 | 15 | 13% |
| Risk Management | 13 | 15 | 87% |
| Monitoring | 9 | 15 | 60% |
| Backtesting | 13 | 15 | 87% |
| Testing | 9 | 12 | 75% |
| Documentation | 15 | 15 | 100% |
| **TOTAL** | **79** | **117** | **68%** |

---

## Required Fixes Before Live Deployment

### Critical (Must Have)

1. **Broker API integration** — Connect to Interactive Brokers, Alpaca, or equivalent for order submission and fill confirmation
2. **Real-time or near-real-time data feed** — WebSocket or polling-based price updates, not batch-only Yahoo Finance
3. **Order management system** — Track order lifecycle: submitted → filled → rejected → cancelled
4. **Position reconciliation** — End-of-day comparison between system state and broker positions

### Important (Should Have)

5. **Alert notifications** — Slack/email alerts for circuit breaker triggers, edge decay warnings, data quality issues
6. **Risk dashboard UI** — Web-based real-time view of positions, P&L, exposure, drawdown
7. **Exchange holiday calendar** — Explicit handling of early closes and holidays
8. **Corporate actions pipeline** — Process splits, dividends, and mergers from a dedicated data source

### Nice to Have

9. **Paper trading mode** — Simulated fills against live market data before committing real capital
10. **Multi-strategy orchestration** — Run QSG-MICRO-SWING-001 and QSG-SYSTEMATIC-MOM-001 in a single portfolio with combined risk limits
11. **Automated backtest comparison** — Compare current vs previous parameter versions on a consistent dataset

---

## Recommended Go-Live Sequence

1. Paper trade QSG-MICRO-SWING-001 for 3 months using signal CSV output + manual execution
2. Build broker API integration (Alpaca recommended for simplicity)
3. Connect real-time data feed
4. Add position reconciliation
5. Run live with minimum capital ($100) for 1 month
6. Scale up based on observed Sharpe and drawdown vs backtest
