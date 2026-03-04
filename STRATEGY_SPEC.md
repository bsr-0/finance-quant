# Strategy Specification

**Document ID:** QSG-SPEC-001
**Version:** 1.0
**Classification:** Internal — Quantitative Research
**Date:** 2026-03-04

---

## 1. Strategy Registry

| Field | QSG-MICRO-SWING-001 | QSG-SYSTEMATIC-MOM-001 |
|-------|---------------------|------------------------|
| **Name** | Trend-Aligned Pullback Reversion | Cross-Sectional Momentum |
| **Style** | Mean Reversion (within trend) | Momentum / Cross-Sectional |
| **Direction** | Long-Only | Long-Only |
| **Holding Period** | 5–15 trading days | 21–252 trading days (monthly rebalance) |
| **Target AUM** | $100 – $1,000 (micro-capital) | $100M – $10B (institutional) |
| **Status** | Research / Pre-Production | Research / Underperforming |
| **Bar Size** | Daily OHLCV | Daily OHLCV |
| **Rebalance** | Event-driven (signal threshold) | Monthly |

---

## 2. QSG-MICRO-SWING-001 — Trend-Aligned Pullback Reversion

### 2.1 Economic Hypothesis

Retail investors systematically overreact to short-term pullbacks in established uptrends. When a liquid stock in a confirmed uptrend (SMA50 > SMA200, price > SMA50) pulls back to oversold levels (RSI < 35, close ≤ Bollinger Lower Band), the selling is disproportionate to the fundamental news. The strategy exploits this behavioral overreaction by buying the pullback and holding for mean reversion over 5–15 days.

**Edge source:** Behavioral bias (loss aversion, recency bias) among retail participants who sell into short-term weakness despite an intact primary uptrend.

**Expected regime:** Works best in bull markets with moderate volatility (VIX 15–25). Degrades in sustained bear markets, which is why the BEAR regime filter blocks entries entirely.

### 2.2 Universe

**Tier 1 — Core ETFs:** SPY, QQQ, IWM, DIA
**Tier 2 — Sector ETFs:** XLF, XLK, XLE, XLV
**Tier 3 — Mega-Cap Equities:** AAPL, MSFT, AMZN, GOOGL, META, NVDA, TSLA, JPM, V, UNH

All instruments are US-listed, highly liquid (ADV > $500M), with tight bid-ask spreads. No OTC, penny stocks, or thinly traded instruments.

### 2.3 Signal Construction

The composite signal score S ∈ [0, 100] is computed from four independent categories:

**Trend Alignment (max 40 points):**
```
trend_pts = 0
if close > SMA(50) and SMA(50) > SMA(200):  trend_pts += 25
if close > SMA(200):                         trend_pts += 10
if slope(SMA(50), 5) > 0:                   trend_pts += 5
```

**Pullback Depth (max 30 points):**
```
pullback_pts = 0
if RSI(14) < 35:             pullback_pts += 15
if close ≤ BB_lower(20, 2):  pullback_pts += 10
if Stochastic_K(14) < 20:    pullback_pts += 5
```

**Volume Confirmation (max 15 points):**
```
volume_pts = 0
if volume < 0.8 * SMA(volume, 20):  volume_pts += 10
if slope(OBV, 5) > 0:               volume_pts += 5
```

**Volatility / Momentum Context (max 15 points):**
```
volatility_pts = 0
if 0.5% < ATR(14)/close*100 < 4.0%:     volatility_pts += 5
if MACD_histogram > MACD_histogram(-1):   volatility_pts += 5
if Williams_%R(14) > -80:                 volatility_pts += 5
```

**Composite:** `S = trend_pts + pullback_pts + volume_pts + volatility_pts`

**Implementation:** `src/pipeline/strategy/signals.py:SignalEngine._score_row()`

### 2.4 Entry Rules

All conditions must pass (AND logic):

1. `regime ≠ "BEAR"` (SPY-based regime classification)
2. `trend_pts ≥ 25` (primary uptrend confirmed)
3. `pullback_pts > 0` (some pullback signal present)
4. `S ≥ threshold` (60 in BULL, 70 in NEUTRAL)
5. No existing position in the same instrument
6. Number of open positions < bracket maximum
7. Portfolio risk budget not exhausted (< 3% of equity)
8. Sufficient cash for the position + spread cost

**Implementation:** `src/pipeline/strategy/engine.py` (lines 284–368)

### 2.5 Exit Rules (Priority-Ordered)

| Priority | Trigger | Condition | Action |
|----------|---------|-----------|--------|
| 1 | Hard stop-loss | `close < entry - 1.5 × ATR(14)` | Immediate exit |
| 1b | Trailing stop | `close < trailing_stop` (activated at +1 ATR gain) | Immediate exit |
| 2 | Regime change | `regime = "BEAR"` | Immediate exit |
| 3 | Trend reversal | `close < SMA(50)` | Exit on close |
| 3b | RSI overbought | `RSI(14) > 70` and position profitable | Exit on close |
| 4 | Profit target | `close ≥ entry + 2 × ATR(14)` | Exit (scale-out if >$500) |
| 5 | Time exit | `days_held ≥ 15` | Exit on close |

**Trailing stop mechanics:**
- Activated when unrealized gain ≥ 1 × ATR from entry
- Trail value = `highest_high - 2.0 × current_ATR`
- Ratchets up only (never moves down)

**Implementation:** `src/pipeline/strategy/exits.py:ExitEngine.check_exit()`

### 2.6 Position Sizing

Risk-based sizing formula:

```
risk_fraction = 1.5% if equity < $500 else 1.0%
risk_budget = equity × risk_fraction × conviction × regime_mult
stop_distance = ATR(14) × 1.5
shares = floor(risk_budget / stop_distance)
```

**Conviction scaling:**
- Score ≥ 80: conviction = 1.0
- Score ≥ 70: conviction = 0.75
- Score ≥ 60: conviction = 0.50

**Regime multiplier:** BULL = 1.0, NEUTRAL = 0.5, BEAR = 0.0

**Portfolio constraints:**
- Max portfolio risk: 3% of equity
- Max position size: 30–100% of equity (by account bracket)
- Max positions: 1–4 (by account bracket)

**Implementation:** `src/pipeline/strategy/sizing.py:PositionSizer.compute()`

---

## 3. QSG-SYSTEMATIC-MOM-001 — Cross-Sectional Momentum

### 3.1 Economic Hypothesis

Exploits the well-documented 12-1 month momentum premium (Jegadeesh & Titman 1993). Stocks that have outperformed over the past 12 months (excluding the most recent month to avoid short-term reversal) tend to continue outperforming. The mechanism is attributed to:

- Institutional herding behavior
- Underreaction to fundamental news
- Disposition effect (selling winners too early)

**Status:** Currently underperforming (CAGR -0.02%, Sharpe 0.43). See §3.5 for diagnosis.

### 3.2 Signal Construction

Three components, z-scored cross-sectionally:

| Component | Weight | Formula |
|-----------|--------|---------|
| momentum_return | 0.60 | 12-1 month total return, z-scored |
| ma_crossover | 0.25 | SMA(50)/SMA(200) - 1, z-scored |
| volatility | 0.15 | -1 × realized_vol(60d), z-scored |

`composite = 0.60 × z(mom_ret) + 0.25 × z(ma_cross) + 0.15 × z(-vol)`

### 3.3 Position Sizing (Volatility-Scaled)

```
w_i = (σ_target / (σ_i × √N)) × sign(s_i) × conviction_i
```

- Target annual volatility: 10%
- Max single position: 5% of capital
- Min position: 0.50%

### 3.4 Risk Constraints

- Max single position: 5%
- Max gross exposure: 100%
- Max ADV participation: 5%
- Sector caps: 30% per sector
- Max drawdown halt: 15%

### 3.5 Diagnosis of Underperformance

The momentum strategy shows CAGR of -0.02% and Sharpe 0.43. Likely causes:
- **Momentum crash risk:** Classic momentum strategies suffered during 2020–2024 due to rapid regime rotations
- **Missing short leg:** Long-only momentum captures only half the premium
- **Cost drag:** 20 bps total cost on monthly rebalance with 610 trades
- **Insufficient universe filtering:** May be selecting momentum among already-expensive names

**Implementation:** `src/pipeline/strategy/example_momentum.py`, `src/pipeline/strategy/signal_library.py`

---

## 4. Dependencies

| Dependency | Purpose | Source |
|------------|---------|--------|
| Daily OHLCV | Signal computation | Yahoo Finance via `extract_prices` |
| SPY close prices | Regime classification | Yahoo Finance |
| Calendar (business days) | Date alignment | `pd.bdate_range` |
| Technical indicators | Signal features | `pipeline.features.technical_indicators` |
| Configuration | Parameters | `config.yaml`, `StrategyConfig` dataclass |

No external API calls are required at signal generation time — all data is pre-loaded.

---

## 5. Signal Generation Schedule

| Step | Timing | Description |
|------|--------|-------------|
| Data refresh | T+0 after market close (16:00 ET) | Load latest OHLCV |
| Indicator computation | T+0 + 5 min | Compute all technical indicators |
| Signal scoring | T+0 + 10 min | Score universe, apply entry rules |
| Pre-trade checks | T+0 + 15 min | Validate tradability, liquidity, risk limits |
| Signal output | T+0 + 20 min | Write `signals_YYYYMMDD.csv` |

Signals are actionable the following trading day at market open.

**CLI entry point:** `python -m pipeline.cli generate-signals --prices-dir data/prices/`
