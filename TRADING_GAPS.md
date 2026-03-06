# Trading Gaps Analysis: $100-$500 Micro-Capital Fund

## Overview

Assessment of the biggest gaps in this repository for profitably trading
with $100 to $500 in capital, using the existing QSG-MICRO-SWING-001
(Trend-Aligned Pullback Reversion) strategy.

---

## Critical Gaps (Must Fix Before Live Trading)

### 1. No Broker Integration

**Impact:** Cannot execute trades programmatically.

The system outputs signal CSVs (`signals_YYYYMMDD.csv`) but has zero
broker connectivity. No order submission, fill tracking, or position
reconciliation exists.

**Recommendation:** Integrate Alpaca API (free, $0 commissions, fractional
shares, paper trading). Build a thin wrapper:

```
signal CSV → validate → Alpaca REST order → track fill → reconcile at close
```

Key components needed:
- Order submission (market/limit)
- Fill confirmation and logging
- Position reconciliation (system state vs broker state)
- Account balance sync

### 2. No Paper Trading Mode

**Impact:** Cannot validate signals on live data without risking capital.

The backtester operates on historical data only. There is no mechanism to
run signals against live market data in a simulated environment.

**Recommendation:** Use Alpaca paper trading environment for 1-3 months
before deploying real capital. Track paper results vs backtest expectations.

### 3. No Real-Time Data Feed

**Impact:** Stops only model closing prices; gaps can blow through them.

All data sources are daily batch (Yahoo Finance). No WebSocket or polling
for intraday price monitoring. This means:
- Cannot enforce intraday stop-losses
- Cannot react to gap openings
- Cannot detect flash crashes intraday

**Recommendation:** Add Alpaca WebSocket or Polygon.io for real-time price
monitoring. At minimum, poll every 5 minutes during market hours for stop
enforcement.

---

## High Priority Gaps

### 4. No Alert/Notification System

**Impact:** Circuit breakers and edge decay warnings fire silently.

The risk management system includes 3-tier circuit breakers, daily loss
limits, consecutive loss blocks, and edge decay monitoring — but none of
these generate external notifications.

**Recommendation:** Add Slack webhook or email alerts for:
- Circuit breaker state changes (YELLOW/ORANGE/RED)
- Kill switch activation
- Edge decay warnings
- Daily signal generation summary
- Fill confirmations and rejections

### 5. Only One Viable Strategy

**Impact:** Single point of failure if the pullback-reversion edge decays.

- QSG-MICRO-SWING-001 (pullback reversion): VIABLE, designed for micro-capital
- QSG-SYSTEMATIC-MOM-001 (momentum): REDESIGNED — multi-timeframe (6-1 + 3-1 month),
  crash protection, higher signal threshold (0.3), tighter exits. Requires paper validation.

Running a single strategy on mega-cap US stocks concentrates risk.

**Recommendation:** Develop 1-2 additional uncorrelated strategies:
- Breakout/trend-following (uncorrelated to mean-reversion)
- ETF pair mean-reversion
- Earnings drift (leveraging existing earnings calendar data)

### 6. Unvalidated Backtest Results

**Impact:** No evidence that backtested performance translates to live.

The backtesting framework is well-built (walk-forward, bias checks, Monte
Carlo, transaction cost modeling), but:
- No published walk-forward validation results
- No out-of-sample test results
- 47 material deficiencies identified in gap analysis
- No live vs. backtest comparison data

**Recommendation:** Run formal walk-forward validation with embargo periods,
publish results, and track paper trading performance against expectations.

---

## Medium Priority Gaps

### 7. Incomplete Corporate Actions Processing

Stock splits and dividends are tracked in the database schema but not fully
integrated into the signal pipeline. This can corrupt signals if a stock
splits and prices are not properly adjusted.

### 8. Single Data Source Risk

Yahoo Finance is the sole price data source. API outages or rate limiting
means zero signals for the day. Config supports AlphaVantage/Polygon
fallback but it is not wired up.

### 9. No Exchange Holiday Calendar

The system does not handle early closes (Christmas Eve, Good Friday) or
market holidays. Signals may be generated for non-trading days.

### 10. No Execution Quality Analytics

No tracking of slippage (model price vs actual fill), execution timing, or
fill quality. This data is essential for calibrating transaction cost models.

---

## What Already Works Well

| Component | Status | Notes |
|-----------|--------|-------|
| Risk management | Strong | 3-tier circuit breakers, ATR stops, kill switch |
| Position sizing | Strong | Micro-capital brackets, conviction scaling |
| Backtesting framework | Strong | Walk-forward, bias checks, Monte Carlo |
| Data pipeline | Strong | 10+ sources, PostgreSQL warehouse |
| Signal generation | Working | Composite scoring, regime filtering |
| Documentation | Excellent | Strategy memos, risk frameworks, trader guides |

---

## Recommended Go-Live Sequence

| Phase | Timeline | Action | Capital at Risk |
|-------|----------|--------|-----------------|
| 1 | Week 1-2 | Build Alpaca broker integration | $0 |
| 2 | Week 3-4 | Paper trade with live signals | $0 |
| 3 | Month 2-3 | Compare paper vs backtest, tune | $0 |
| 4 | Month 4 | Go live with minimum capital | $100 |
| 5 | Month 5+ | Scale if metrics hold | $250-500 |

### Go-Live Criteria (Phase 3 → Phase 4)

- Paper trading Sharpe > 0.5 (half of backtest target)
- Paper max drawdown < 15%
- Paper win rate > 45%
- At least 20 completed paper trades
- All circuit breakers tested and verified
- Position reconciliation working daily

---

## Minimum Viable Broker Integration (Phase 1)

```
src/pipeline/execution/
├── broker.py          # Abstract broker interface
├── alpaca_broker.py   # Alpaca API implementation
├── order_manager.py   # Order lifecycle management
├── reconciler.py      # Position reconciliation
└── paper_mode.py      # Paper trading wrapper
```

Core interface:
- `submit_order(ticker, side, qty, order_type, limit_price=None)`
- `cancel_order(order_id)`
- `get_positions() -> list[Position]`
- `get_account() -> AccountInfo`
- `reconcile() -> list[Discrepancy]`
