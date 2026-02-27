# Goldman Sachs Quantitative Strategies Group
## Systematic Trading Strategy Memo

**Strategy Name:** QSG-MICRO-SWING-001 — Trend-Aligned Pullback Reversion
**Classification:** Systematic Equity | Long-Only | Small-Capital Swing
**Target AUM:** $100 – $1,000 (Micro-Capital Growth Program)
**Author:** Quantitative Strategies Group
**Date:** February 2026
**Status:** RESEARCH — Pre-Production

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Strategy Thesis](#2-strategy-thesis)
3. [Universe Selection](#3-universe-selection)
4. [Signal Generation Logic](#4-signal-generation-logic)
5. [Entry Rules](#5-entry-rules)
6. [Exit Rules](#6-exit-rules)
7. [Position Sizing Model](#7-position-sizing-model)
8. [Risk Parameters](#8-risk-parameters)
9. [Backtesting Framework](#9-backtesting-framework)
10. [Benchmark Selection](#10-benchmark-selection)
11. [Edge Decay Monitoring](#11-edge-decay-monitoring)
12. [Appendix: Mathematical Reference](#appendix-mathematical-reference)

---

## 1. Executive Summary

This memo presents a rules-based swing trading strategy designed for micro-capital accounts ($100–$1,000). The strategy exploits a well-documented behavioral inefficiency: **retail investors overreact to short-term pullbacks within established uptrends**, creating temporary mispricings in liquid US equities and ETFs that revert over 3–15 trading days.

The strategy is **long-only, cash-secured, zero-margin**, and designed for capital preservation first, compounding second. It sits in cash when no high-quality setups exist and deploys capital only when a strict confluence of trend, momentum, volatility, and volume conditions are met simultaneously.

**Target Performance Profile:**

| Metric | Target | Hard Floor |
|---|---|---|
| Annual CAGR (net of costs) | 8–15% | > 0% |
| Maximum Drawdown | < 10% | < 15% |
| Win Rate | > 55% | > 50% |
| Profit Factor | > 1.5 | > 1.2 |
| Average Holding Period | 5–12 trading days | — |
| Time in Market | 30–50% | — |
| Sharpe Ratio (annualized) | > 0.8 | > 0.4 |

---

## 2. Strategy Thesis

### 2.1 The Inefficiency

**Thesis:** Within an established intermediate-term uptrend, short-term pullbacks to key technical support levels (moving averages, Bollinger Band lower boundary) create statistically favorable long entries because:

1. **Behavioral overreaction:** Retail sellers panic during 3–5 day declines even when the larger trend structure remains intact. This is well-documented in behavioral finance literature (Barberis, Shleifer & Vishny 1998; Daniel, Hirshleifer & Subrahmanyam 1998).

2. **Mean reversion in bounded series:** RSI and stochastic oscillators exhibit strong mean-reverting properties within trending regimes. When RSI drops below 35 in an uptrend, the 10-day forward return distribution is significantly right-skewed.

3. **Institutional accumulation:** Large funds treat pullbacks in quality names as accumulation opportunities. Volume signatures (declining volume on pullbacks, expanding volume on reversals) confirm institutional participation.

4. **Volatility compression:** After a pullback, realized volatility often compresses before the next leg up, creating favorable risk/reward as ATR-based stops tighten.

### 2.2 Why This Edge Persists

- **Structural:** Retail order flow is behaviorally anchored to recent price action (recency bias, loss aversion).
- **Scale-independent:** This edge exists precisely because it is too small for institutional capital to arbitrage — a $500 position in SPY has zero market impact.
- **Time-horizoned:** The 5–15 day holding period sits in a neglected timeframe — too long for day traders, too short for buy-and-hold. This reduces competition.

### 2.3 When the Edge Disappears

The strategy will **not** work in:
- Sustained bear markets (price below 200-day SMA)
- High-correlation panic regimes (VIX > 35, all correlations → 1)
- Trendless chop (ADX < 15 for extended periods)

The regime filter (Section 4.3) explicitly disables trading in these conditions.

---

## 3. Universe Selection

### 3.1 Tradeable Instruments

The universe is restricted to **highly liquid US equities and ETFs** that a micro-capital account can trade with minimal friction:

**Tier 1 — Core ETFs (Preferred):**

| Symbol | Name | Rationale |
|---|---|---|
| SPY | S&P 500 ETF | Ultimate liquidity, broad market proxy |
| QQQ | Nasdaq 100 ETF | Tech-heavy growth exposure |
| IWM | Russell 2000 ETF | Small-cap mean reversion tends to be strongest |
| DIA | Dow Jones ETF | Blue-chip stability, lower volatility |

**Tier 2 — Sector ETFs (Selective):**

| Symbol | Name | Rationale |
|---|---|---|
| XLF | Financial Select | Cyclical, strong trend/pullback patterns |
| XLK | Technology Select | Growth sector, liquid |
| XLE | Energy Select | Commodity-linked, distinct regime behavior |
| XLV | Health Care Select | Defensive, lower correlation |

**Tier 3 — Individual Equities (Most Selective):**

Only mega-cap, highly liquid names with average daily dollar volume > $500M:

AAPL, MSFT, AMZN, GOOGL, META, NVDA, TSLA, JPM, V, UNH

### 3.2 Universe Filtering Criteria

A symbol must pass ALL of the following to remain in the tradeable universe on any given day:

```
UNIVERSE_FILTER:
  1. Average daily dollar volume (20-day) > $100M
  2. Share price > $5.00 (no penny stocks)
  3. Listed on NYSE, NASDAQ, or ARCA
  4. Bid-ask spread < 5 bps (ensures minimal slippage)
  5. No pending corporate actions (earnings within 3 days, splits, etc.)
  6. No current trading halt
```

### 3.3 Earnings Blackout

**Critical rule:** No new positions are opened within **3 trading days before** a scheduled earnings announcement. Existing positions are closed 2 days before earnings if still held. Binary event risk is inconsistent with the strategy's thesis.

---

## 4. Signal Generation Logic

### 4.1 Indicator Suite

The strategy uses four indicator categories, each providing independent confirmation:

#### A. Trend Identification

```
Primary Trend (intermediate):
  SMA_50  = Simple Moving Average(Close, 50)
  SMA_200 = Simple Moving Average(Close, 200)

  UPTREND    = (Close > SMA_50) AND (SMA_50 > SMA_200)
  DOWNTREND  = (Close < SMA_50) AND (SMA_50 < SMA_200)
  NEUTRAL    = otherwise
```

#### B. Pullback Detection (Mean Reversion Trigger)

```
RSI_14 = RSI(Close, 14)

Pullback Conditions (at least ONE must be true):
  PB_RSI       = RSI_14 < 35
  PB_BOLLINGER = Close <= BB_Lower(Close, 20, 2.0)
  PB_SMA       = (Close <= SMA_20) AND (Close > SMA_50)
  PB_STOCH     = Stochastic_K(14) < 20

PULLBACK_DETECTED = PB_RSI OR PB_BOLLINGER OR PB_SMA OR PB_STOCH
```

#### C. Volume Confirmation

```
Volume_SMA_20 = SMA(Volume, 20)

Healthy pullback (supply drying up):
  VOLUME_DECLINING = Volume < Volume_SMA_20 * 0.8

Reversal confirmation (demand returning):
  VOLUME_EXPANDING = Volume > Volume_SMA_20 * 1.2

VOLUME_CONFIRMED = VOLUME_DECLINING OR VOLUME_EXPANDING
```

#### D. Volatility Context

```
ATR_14  = ATR(High, Low, Close, 14)
ATR_PCT = ATR_14 / Close * 100

Volatility filter (avoid extreme chop):
  VOL_ACCEPTABLE = (ATR_PCT > 0.5%) AND (ATR_PCT < 4.0%)
```

### 4.2 Composite Signal Score

Each condition that is true adds to a **signal score** (0–100):

```python
def compute_signal_score(data):
    score = 0

    # Trend alignment (40 points max)
    if close > sma_50 and sma_50 > sma_200:
        score += 25                                    # Primary uptrend
    if close > sma_200:
        score += 10                                    # Above long-term trend
    if sma_50_slope > 0:
        score += 5                                     # 50-day trending up

    # Pullback depth (30 points max)
    if rsi_14 < 35:
        score += 15                                    # RSI oversold in uptrend
    if close <= bb_lower:
        score += 10                                    # At Bollinger lower band
    if stoch_k < 20:
        score += 5                                     # Stochastic oversold

    # Volume confirmation (15 points max)
    if volume < volume_sma_20 * 0.8:
        score += 10                                    # Low volume pullback
    if obv_slope > 0:
        score += 5                                     # OBV still rising (accumulation)

    # Volatility / regime (15 points max)
    if 0.5 < atr_pct < 4.0:
        score += 5                                     # Reasonable volatility
    if macd_histogram > macd_histogram_prev:
        score += 5                                     # MACD momentum improving
    if williams_r > -80:
        score += 5                                     # Not deeply oversold (confirmation)

    return score
```

### 4.3 Regime Filter (Master Switch)

The strategy has a **global regime filter** that overrides all signals:

```
REGIME = classify_regime(SPY)

classify_regime(prices):
  ma_200 = SMA(Close, 200)
  drawdown = (Close - Peak) / Peak

  if Close > ma_200 AND drawdown > -0.10:
    return "BULL"
  elif Close < ma_200 OR drawdown < -0.20:
    return "BEAR"
  else:
    return "NEUTRAL"

Trading allowed:
  BULL:    Full signal generation, normal position sizing
  NEUTRAL: Reduced position sizing (50% of normal), only score >= 70
  BEAR:    NO new positions. Close existing positions on strength.
```

---

## 5. Entry Rules

### 5.1 Required Confluence (ALL Must Be True)

```
ENTRY SIGNAL = TRUE when ALL of:

  1. REGIME != "BEAR"                    # Not in bear market
  2. UPTREND == TRUE                     # Primary trend is up
  3. PULLBACK_DETECTED == TRUE           # Price has pulled back
  4. SIGNAL_SCORE >= 60                  # Sufficient confluence
  5. VOL_ACCEPTABLE == TRUE              # Volatility in range
  6. NO_EARNINGS_WITHIN_3_DAYS == TRUE   # No binary event risk
  7. NO_EXISTING_POSITION == TRUE        # Not already in this name
  8. CASH_AVAILABLE >= MIN_POSITION_SIZE # Have capital to deploy
```

### 5.2 Entry Execution

```
On ENTRY SIGNAL = TRUE:
  1. Compute position size (Section 7)
  2. Place LIMIT order at:
       entry_price = Close                # Market-on-close
       — OR —
       entry_price = Close - (ATR_14 * 0.1)  # Slight discount
  3. If limit not filled within 1 trading day, cancel order
  4. Record: entry_date, entry_price, signal_score, stop_loss, target
```

### 5.3 Stale Signal Rule

If the signal score was >= 60 yesterday but price moved up > 1.5% today before entry, the signal is **stale** — do not chase. Wait for the next pullback.

---

## 6. Exit Rules

The strategy employs a **layered exit framework** with four independent triggers. The first trigger hit closes the position.

### 6.1 Stop-Loss Exit (Capital Preservation)

```
STOP LOSS:
  stop_price = entry_price - (ATR_14_at_entry * 1.5)

  Trailing stop (activated after +1 ATR profit):
    if unrealized_profit > ATR_14_at_entry:
      trail_stop = max(trail_stop, high - ATR_14_current * 2.0)

  EXIT if Close < stop_price OR Close < trail_stop
```

**Maximum loss per trade:** Capped at 1.5x ATR from entry. For a typical stock with ATR_PCT = 1.5%, this means a maximum loss of ~2.25% of the position.

### 6.2 Profit Target Exit (Capture Gains)

```
TARGET PRICE:
  target_1 = entry_price + (ATR_14_at_entry * 2.0)   # 2:1 reward/risk
  target_2 = entry_price + (ATR_14_at_entry * 3.0)   # 3:1 stretch target

  Scale-out rule (optional for accounts > $500):
    At target_1: close 50% of position
    At target_2: close remaining 50%

  For accounts < $500 (single lot):
    Close 100% at target_1
```

### 6.3 Time-Based Exit (Avoid Dead Money)

```
TIME EXIT:
  max_holding_days = 15

  if days_held >= max_holding_days AND position is profitable:
    EXIT at market
  if days_held >= max_holding_days AND position is losing:
    EXIT at market (accept the loss, don't hope)
```

**Rationale:** The strategy thesis is that pullback reversion occurs within 5–12 days. If it hasn't worked within 15 days, the thesis is broken for this trade.

### 6.4 Signal Reversal Exit

```
SIGNAL REVERSAL:
  if UPTREND becomes FALSE (close drops below SMA_50):
    EXIT immediately at market
  if RSI_14 > 70 AND position is profitable:
    EXIT (overbought, take profit)
  if REGIME changes to "BEAR":
    EXIT all positions immediately
```

### 6.5 Exit Priority

```
Priority order (highest to lowest):
  1. Stop-loss (non-negotiable)
  2. Regime change to BEAR (systemic risk)
  3. Signal reversal (thesis invalidated)
  4. Profit target (greed management)
  5. Time-based exit (opportunity cost)
```

---

## 7. Position Sizing Model

### 7.1 Core Principle: Risk-Based Sizing

Position size is determined by **how much capital you are willing to lose if the stop-loss is hit**, not by how much you want to invest.

```
POSITION SIZING FORMULA:

  risk_per_trade = account_equity * risk_fraction
  dollar_risk    = entry_price - stop_price          # Per-share risk
  shares         = floor(risk_per_trade / dollar_risk)
  position_value = shares * entry_price

  Constraints:
    position_value <= account_equity * max_position_pct
    shares >= 1                                       # Minimum 1 share
```

### 7.2 Risk Fraction Schedule

The risk fraction scales with account size to account for the discrete nature of small positions:

| Account Equity | risk_fraction | max_position_pct | Max Positions |
|---|---|---|---|
| $100 – $250 | 1.5% | 100% | 1 |
| $250 – $500 | 1.5% | 60% | 2 |
| $500 – $1,000 | 1.0% | 40% | 3 |
| $1,000+ | 1.0% | 30% | 4 |

### 7.3 Conviction Scaling

Signal score modulates position size within the allowed range:

```
conviction_multiplier:
  score >= 80: 1.00x (full size)
  score >= 70: 0.75x
  score >= 60: 0.50x (minimum entry)

adjusted_risk = risk_per_trade * conviction_multiplier
```

### 7.4 Regime Adjustment

```
regime_multiplier:
  BULL:    1.0x
  NEUTRAL: 0.5x
  BEAR:    0.0x (no new positions)

final_position_size = base_size * conviction_multiplier * regime_multiplier
```

### 7.5 Worked Example

```
Account: $500
Signal: SPY pullback, score = 75
SPY price: $520.00
ATR_14: $6.50
Stop: $520 - ($6.50 * 1.5) = $510.25
Dollar risk per share: $520.00 - $510.25 = $9.75
Risk budget: $500 * 1.0% = $5.00
Conviction: 0.75x → adjusted risk = $5.00 * 0.75 = $3.75
Shares: floor($3.75 / $9.75) = 0 → minimum 1 share

Position: 1 share of SPY at $520 = $520 (exceeds 40% cap)
→ Reduce to max_position_pct: $500 * 40% = $200 → 0 shares

Resolution: Account too small for SPY at this price.
→ Switch to a lower-priced ETF (e.g., IWM at ~$220) or wait.

IWM: price $220, ATR $4.00, stop = $220 - $6.00 = $214.00
Dollar risk: $6.00, shares = floor($3.75 / $6.00) = 0 → 1 share
Position: 1 share * $220 = $220 = 44% of account → OK (within 60%)
Max loss if stopped: $6.00 = 1.2% of account ✓
```

---

## 8. Risk Parameters

### 8.1 Hard Limits Table

| Parameter | Value | Rationale |
|---|---|---|
| Max positions | 1–4 (by account size) | Concentration for small capital |
| Max single position | 30–100% (by account size) | Practical constraint |
| Max portfolio risk | 3% of equity at risk | Sum of all stop-loss distances |
| Max drawdown (strategy halt) | 15% from equity peak | Preserve capital, reassess |
| Max consecutive losses before pause | 4 | Reassess signal quality |
| Max correlation between positions | 0.7 | Diversification |
| Leverage | 0x (cash only) | No margin, ever |
| Min cash reserve | 0% (can be fully invested) | — |
| Max sector exposure | 50% of capital | Avoid sector concentration |
| Max daily loss | 2% of equity | Intraday circuit breaker |

### 8.2 Position Risk Decomposition

```
For each position p:
  position_risk_p = shares_p * (entry_p - stop_p) / account_equity

Total portfolio risk:
  portfolio_risk = sum(position_risk_p for all p)
  REQUIRE: portfolio_risk <= 3.0%
```

### 8.3 Drawdown Circuit Breakers

```
Level 1 — Yellow (5% drawdown from peak):
  - Reduce max position size by 50%
  - Only enter trades with score >= 75
  - Log warning

Level 2 — Orange (10% drawdown from peak):
  - No new positions
  - Tighten stops on existing positions to 1.0x ATR
  - Log alert

Level 3 — Red (15% drawdown from peak):
  - Close ALL positions immediately
  - Strategy enters 30-day cooldown
  - Manual review required before restart
  - Log critical alert
```

### 8.4 Correlation Monitoring

```
Before opening position in symbol X:
  for each existing_position Y:
    corr_XY = rolling_correlation(returns_X, returns_Y, window=60)
    if abs(corr_XY) > 0.7:
      REJECT entry in X
      LOG: "Correlation breach: {X} vs {Y} = {corr_XY:.2f}"
```

---

## 9. Backtesting Framework

### 9.1 Methodology

The strategy must be validated using the existing `pipeline.backtesting` infrastructure with the following protocol:

```
Backtest Protocol:
  1. Data period:       Minimum 5 years (2019-01-01 to 2024-12-31)
  2. Walk-forward:      12-month train / 3-month test, expanding window
  3. Cost model:        FixedPlusSpreadModel(spread_bps=3, commission=0.00)
                        (Commission-free brokers, but model spread)
  4. Slippage:          2 bps per trade
  5. Fill assumption:   Market-on-close at adjusted close price
  6. Survivorship bias: Use delisting-adjusted prices from curated tables
  7. Corporate actions: Split/dividend-adjusted OHLCV
  8. Look-ahead bias:   ALL indicators computed on data available at t-1
```

### 9.2 Walk-Forward Validation

```
Using pipeline.backtesting.walk_forward:

  walk_forward_validate(
      df=feature_data,
      train_fn=calibrate_thresholds,
      predict_fn=generate_signals,
      eval_fn=compute_strategy_metrics,
      target_col="forward_return_10d",
      train_size=252,           # 1 year
      test_size=63,             # 1 quarter
      expanding=True
  )
```

### 9.3 Required Out-of-Sample Tests

| Test | Method | Pass Criterion |
|---|---|---|
| Walk-forward Sharpe | Expanding window OOS | Sharpe > 0.4 in every fold |
| Regime robustness | Segment by bull/bear/flat | Positive return in bull & flat |
| Year-by-year returns | Annual P&L decomposition | No year worse than -10% |
| Monte Carlo | 10,000 path simulations | 5th percentile CAGR > 0% |
| Transaction cost sensitivity | 2x and 3x base costs | Still profitable at 3x costs |
| Parameter stability | +/- 20% on all thresholds | Sharpe stays > 0.3 |

### 9.4 Backtest Integrity Checklist

```
□ No future data leakage (all indicators use data[0:t], never data[t+1:])
□ Adjusted prices used for all computations
□ Transaction costs applied to every trade
□ Slippage modeled
□ Positions rounded to whole shares
□ Minimum trade size enforced ($1 minimum)
□ Cash earns 0% (conservative)
□ No survivorship bias (delisted names included)
□ Walk-forward (not in-sample optimization)
□ Multiple regimes tested (2020 COVID crash, 2022 bear)
```

---

## 10. Benchmark Selection

### 10.1 Primary Benchmark

**SPY (S&P 500 ETF Trust)** — Total Return Index

**Rationale:** The strategy trades US equities and ETFs. SPY represents the opportunity cost of simply buying and holding the market. Any systematic strategy must justify its complexity by outperforming this simple alternative on a risk-adjusted basis.

### 10.2 Secondary Benchmarks

| Benchmark | Rationale |
|---|---|
| 60/40 SPY/AGG | Classic balanced portfolio — risk-adjusted comparison |
| Risk-free rate (3-month T-bill) | Sharpe ratio denominator; does the strategy beat cash? |
| Equal-weight buy-and-hold (universe) | Does timing add value vs. passive exposure? |
| Max drawdown of SPY | Did we avoid the worst of market downturns? |

### 10.3 Key Relative Metrics

```
vs. SPY:
  Information Ratio = (strategy_return - SPY_return) / tracking_error
  Target: IR > 0.3

  Up-capture ratio   = strategy_return_up_months / SPY_return_up_months
  Down-capture ratio  = strategy_return_down_months / SPY_return_down_months
  Target: Up-capture > 60%, Down-capture < 40%

  Beta to SPY < 0.5 (due to significant cash allocation)
```

---

## 11. Edge Decay Monitoring

### 11.1 What Edge Decay Looks Like

The strategy edge will decay when:
- Market microstructure changes (tighter spreads reduce the pullback amplitude)
- Crowding (too many participants trade the same pullback pattern)
- Regime shift (extended bear market or structural change in volatility)

### 11.2 Monitoring Dashboard (Rolling 60-Day)

```
MONITOR CONTINUOUSLY:

  1. Rolling Win Rate (60-day):
     wr_60 = winning_trades_60d / total_trades_60d
     ALERT if wr_60 < 0.45 for 2 consecutive months

  2. Rolling Profit Factor (60-day):
     pf_60 = sum(winning_pnl_60d) / abs(sum(losing_pnl_60d))
     ALERT if pf_60 < 1.0 for 2 consecutive months

  3. Rolling Sharpe (60-day):
     sharpe_60 = annualized_sharpe(returns_60d)
     ALERT if sharpe_60 < 0.0 for 3 consecutive months

  4. Signal Hit Rate:
     signal_accuracy = trades_hitting_target / total_trades
     ALERT if signal_accuracy declines > 20% from inception average

  5. Average Winner / Average Loser Ratio:
     wl_ratio = avg_winner / abs(avg_loser)
     ALERT if wl_ratio < 1.0

  6. Hurst Exponent of Equity Curve:
     H = hurst_exponent(equity_curve)
     ALERT if H < 0.45 (equity curve becoming mean-reverting = no trend)
```

### 11.3 Response Protocol

```
YELLOW ALERT (1 metric breached):
  → Review last 20 trades manually
  → Check if market regime has shifted
  → Continue trading with 50% reduced size

ORANGE ALERT (2+ metrics breached):
  → Halt new entries for 2 weeks
  → Run full parameter re-calibration on walk-forward
  → Check for structural market changes

RED ALERT (3+ metrics breached for 3+ months):
  → Full strategy shutdown
  → Comprehensive review of thesis
  → Determine if the edge has permanently decayed
  → Consider thesis revision or strategy retirement
```

### 11.4 Automatic Recalibration

```
Every 3 months (quarterly):
  1. Re-run walk-forward validation on last 12 months
  2. Compare OOS metrics to inception averages
  3. If OOS Sharpe < 50% of inception Sharpe:
     → Flag for manual review
  4. If parameter sensitivity widened by > 30%:
     → Flag parameter instability
```

---

## Appendix: Mathematical Reference

### A.1 Indicator Formulas

**Simple Moving Average:**
```
SMA(t, n) = (1/n) * Σ(i=0 to n-1) Close(t-i)
```

**Exponential Moving Average:**
```
EMA(t, n) = α * Close(t) + (1 - α) * EMA(t-1, n)
where α = 2 / (n + 1)
```

**Relative Strength Index:**
```
RS(t, n) = EMA(gains, n) / EMA(losses, n)
RSI(t, n) = 100 - 100 / (1 + RS(t, n))
```

**Average True Range:**
```
TR(t) = max(High(t) - Low(t), |High(t) - Close(t-1)|, |Low(t) - Close(t-1)|)
ATR(t, n) = SMA(TR, n)
```

**Bollinger Bands:**
```
BB_mid(t, n)   = SMA(Close, n)
BB_upper(t, n) = BB_mid + k * σ(Close, n)
BB_lower(t, n) = BB_mid - k * σ(Close, n)
where k = 2.0, σ = rolling standard deviation
```

**Stochastic Oscillator:**
```
%K(t, n) = 100 * (Close(t) - LL(n)) / (HH(n) - LL(n))
%D(t, m) = SMA(%K, m)
where LL = lowest low, HH = highest high over n periods
```

### A.2 Risk Formulas

**Annualized Sharpe Ratio:**
```
Sharpe = (μ_excess / σ) * √252
where μ_excess = mean daily excess return, σ = daily return std
```

**Maximum Drawdown:**
```
DD(t) = (NAV(t) - Peak(t)) / Peak(t)
MaxDD = min(DD(t)) over all t
```

**Position Size (shares):**
```
S = floor( (E * r * c * g) / (P_entry - P_stop) )
where:
  E = account equity
  r = risk fraction (1.0–1.5%)
  c = conviction multiplier (0.5–1.0)
  g = regime multiplier (0.0–1.0)
  P_entry = entry price
  P_stop = stop-loss price
```

### A.3 Implementation Pseudocode

```python
# Main trading loop (daily, after market close)
def daily_update(account, market_data, date):
    # 1. Update regime
    regime = classify_regime(market_data["SPY"])

    # 2. Check existing positions for exits
    for position in account.open_positions:
        exit_signal = check_exits(position, market_data, date)
        if exit_signal:
            execute_exit(account, position, exit_signal)

    # 3. Check drawdown circuit breakers
    dd_level = check_drawdown(account)
    if dd_level >= 3:
        close_all_positions(account)
        return

    # 4. Scan universe for entry signals (if allowed)
    if regime != "BEAR" and dd_level < 2:
        for symbol in tradeable_universe:
            score = compute_signal_score(symbol, market_data)
            if score >= entry_threshold(dd_level):
                size = compute_position_size(
                    account, symbol, market_data, score, regime
                )
                if size > 0 and passes_risk_checks(account, symbol, size):
                    execute_entry(account, symbol, size)

    # 5. Log daily state
    log_portfolio_state(account, date)
    check_edge_decay(account)
```

---

*This document is for research and educational purposes. Past performance in backtests does not guarantee future results. All trading involves risk of loss. The mathematical edge described herein is probabilistic, not deterministic.*

*Quantitative Strategies Group — Systematic Trading Research*
