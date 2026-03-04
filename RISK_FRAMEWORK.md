# Risk Management Framework

**Document ID:** QSG-RISK-001
**Version:** 1.0
**Date:** 2026-03-04

---

## 1. Risk Hierarchy

```
Position-Level
  ├── Hard stop-loss (1.5× ATR)
  ├── Trailing stop (2.0× ATR from high)
  ├── Profit targets (2× and 3× ATR)
  └── Time exit (15 days max)

Portfolio-Level
  ├── Drawdown circuit breakers (5%/10%/15%)
  ├── Daily loss limit (2% of equity)
  ├── Consecutive loss limit (4 trades)
  ├── Portfolio risk budget (3% max)
  ├── Correlation monitoring (0.70 threshold)
  └── Position count limits (1–4 by bracket)

Strategy-Level
  ├── Edge decay monitoring (6 metrics)
  ├── Regime filter (no entries in BEAR)
  ├── Kill switch (thread-safe global halt)
  └── Intraday risk monitor (anomaly detection)

Institutional Constraints
  ├── Gross/net exposure limits
  ├── Sector exposure caps (30%)
  ├── ADV participation limits (5–10%)
  └── Leverage limits (2–4×)
```

---

## 2. Position-Level Controls

### 2.1 Stop-Loss Framework

| Type | Formula | Behavior |
|------|---------|----------|
| **Hard stop** | `entry - 1.5 × ATR(14)` | Non-negotiable. Triggers at close. |
| **Trailing stop** | `highest_high - 2.0 × current_ATR` | Activates after +1 ATR gain. Ratchets up only. |

**Implementation:** `src/pipeline/strategy/exits.py:ExitEngine.check_exit()` (lines 90–198)

### 2.2 Profit Targets

| Target | Formula | Action |
|--------|---------|--------|
| Target 1 | `entry + 2.0 × ATR(14)` | Scale out 50% (accounts > $500) |
| Target 2 | `entry + 3.0 × ATR(14)` | Close remainder |

**Note:** Scale-out is described in the strategy memo but currently exits the full position. Partial position exits are a future enhancement.

### 2.3 Time Exit

Maximum holding period: **15 trading days**. Mandatory exit to prevent dead-money positions.

---

## 3. Portfolio-Level Controls

### 3.1 Drawdown Circuit Breakers

| Level | Threshold | Position Sizing | New Entries | Score Required | Action |
|-------|-----------|-----------------|-------------|----------------|--------|
| GREEN | < 5% DD | 100% | Allowed | ≥ 60 | Normal operation |
| YELLOW | 5–10% DD | 50% | Allowed (restricted) | ≥ 75 | Review trades manually |
| ORANGE | 10–15% DD | 0% | **Blocked** | N/A | Tighten stops to 1.0× ATR |
| RED | ≥ 15% DD | N/A | **Blocked** | N/A | **Close all positions. 30-day cooldown.** |

**Implementation:** `src/pipeline/strategy/risk.py:SwingRiskManager` (lines 43–194)

### 3.2 Daily Loss Limit

- Threshold: **2% of equity** (`max_daily_loss_pct = 0.02`)
- Behavior: Blocks all new entries for the remainder of the day
- Implementation: `risk.py:get_risk_state()` — checks `daily_return < -max_daily_loss_pct`

### 3.3 Consecutive Loss Protection

- Threshold: **4 consecutive losing trades**
- Behavior: Blocks new entries until a winning trade resets the counter
- Implementation: `risk.py:record_trade_result()` tracks win/loss streak

### 3.4 Portfolio Risk Budget

- Maximum total risk: **3% of equity** across all open positions
- Risk per position = `(entry_price - stop_price) × shares / equity`
- New entries rejected when budget is exhausted
- Implementation: `sizing.py:PositionSizer.compute()` checks `remaining_risk`

### 3.5 Correlation Monitoring

- Threshold: **0.70** maximum pairwise correlation (60-day rolling window)
- Behavior: Rejects entry if candidate is too correlated with existing positions
- Implementation: `risk.py:SwingRiskManager.check_correlation()`

Additional institutional-grade monitoring:
- Cluster detection for highly correlated instruments
- Factor exposure tracking (beta, size, value, momentum)
- Implementation: `infrastructure/correlation_monitor.py`

---

## 4. Strategy-Level Controls

### 4.1 Edge Decay Monitoring

Six metrics tracked on a 60-day rolling window:

| Metric | Alert Threshold | Rationale |
|--------|-----------------|-----------|
| Win rate | < 45% | Strategy losing predictive accuracy |
| Profit factor | < 1.0 | Losses exceeding gains |
| Rolling Sharpe | < 0.0 | Negative risk-adjusted returns |
| Signal hit rate | Decays > 20% from inception | Signal quality deteriorating |
| Winner/loser ratio | < 1.0 | Average win < average loss |
| Hurst exponent | < 0.45 | Equity curve losing trend (mean-reverting) |

**Alert levels:**
- YELLOW (1 breach): Reduce position size 50%, review
- ORANGE (2+ breaches): Halt entries 2 weeks, recalibrate
- RED (3+ breaches for 3+ months): Full shutdown, thesis review

**Implementation:** `src/pipeline/strategy/edge_decay.py:EdgeDecayMonitor`

### 4.2 Regime Filter

- BEAR regime: **No entries allowed**
- Regime classification: SPY SMA(50) vs SMA(200) crossover + drawdown detection
- Implementation: `src/pipeline/eval/regime.py:classify_regimes()`

### 4.3 Kill Switch

Thread-safe global trading halt:
- Engaged by intraday monitor on hard limit breach (2× drawdown threshold)
- Requires explicit manual reset
- All pre-trade checks reject orders when kill switch is engaged

**Implementation:** `src/pipeline/infrastructure/risk_controls.py:KillSwitch`

---

## 5. Position Sizing Parameters

### 5.1 Micro-Capital Bracket System

| Equity Bracket | Risk Fraction | Max Position % | Max Positions |
|----------------|---------------|----------------|---------------|
| $100 – $250 | 1.5% | 100% | 1 |
| $250 – $500 | 1.5% | 60% | 2 |
| $500 – $1,000 | 1.0% | 40% | 3 |
| $1,000+ | 1.0% | 30% | 4 |

### 5.2 Conviction Scaling

| Signal Score | Conviction Multiplier |
|-------------|----------------------|
| ≥ 80 | 1.00 (full risk budget) |
| ≥ 70 | 0.75 |
| ≥ 60 | 0.50 |

### 5.3 Regime Multiplier

| Regime | Multiplier |
|--------|-----------|
| BULL | 1.0 |
| NEUTRAL | 0.5 |
| BEAR | 0.0 (no entries) |

**Implementation:** `src/pipeline/strategy/sizing.py:PositionSizer` (lines 65–231)

---

## 6. Institutional Risk Constraints

The `RiskConstraintSet` provides a framework of hard/soft/advisory constraints:

| Constraint | Type | Default | Implementation |
|------------|------|---------|----------------|
| Max position weight | Hard | 5% | `risk_constraints.py` |
| Max sector exposure | Hard | 30% | `risk_constraints.py` |
| Max gross exposure | Hard | 100% | `risk_constraints.py` |
| Max net exposure | Hard | 100% | `risk_constraints.py` |
| Max ADV participation | Hard | 5% | `risk_constraints.py` |
| Max leverage | Hard | 4× | `risk_controls.py:RiskLimits` |
| Max order notional | Hard | $1M | `risk_controls.py:RiskLimits` |
| Max concentration | Hard | 20% | `risk_controls.py:RiskLimits` |
| Max daily loss | Hard | $100K | `risk_controls.py:RiskLimits` |
| Turnover limit | Soft | Varies | `risk_constraints.py` |

---

## 7. Liquidity and Capacity

### 7.1 ADV Participation

- Default cap: **10%** of average daily volume (`simulator.py:max_adv_pct`)
- At institutional level: **5%** recommended (`risk_constraints.py`)
- For micro-capital ($100–$1K): capacity is effectively unlimited

### 7.2 Market Impact Model

The Almgren-Chriss square root impact model is available:

```
impact = σ × η × √(qty / ADV) × notional
```

With η = 0.25 and σ = 0.02, a $1M order in a stock with $100M ADV experiences ~0.5 bps of impact.

**Implementation:** `src/pipeline/backtesting/transaction_costs.py:SquareRootImpactModel`

### 7.3 Capacity Estimation

The `backtesting/capacity.py` module estimates strategy capacity by:
- Computing maximum trade size per signal
- Tracking ADV participation rates
- Estimating aggregate impact at different AUM levels
