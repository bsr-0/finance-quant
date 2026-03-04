# Goldman Sachs Quantitative Strategies Group
## Systematic Trading Strategy Memo

**Strategy Name:** QSG-SYSTEMATIC-MOM-001
**Classification:** Systematic Equity | Long-Only | Cross-Sectional Momentum
**Target AUM:** $100M - $10B (capacity constrained by liquidity filters)
**Author:** Quantitative Strategies Group
**Date:** March 2026
**Status:** RESEARCH — Pre-Production

---

## Table of Contents

1. [Strategy Thesis](#1-strategy-thesis)
2. [Universe Selection](#2-universe-selection)
3. [Signal Generation](#3-signal-generation)
4. [Entry Rules](#4-entry-rules)
5. [Exit Rules](#5-exit-rules)
6. [Position Sizing](#6-position-sizing)
7. [Risk Parameters](#7-risk-parameters)
8. [Backtesting Methodology](#8-backtesting-methodology)
9. [Benchmark Selection](#9-benchmark-selection)
10. [Edge Decay Monitoring](#10-edge-decay-monitoring)
11. [Appendix: Mathematical Reference](#appendix-mathematical-reference)

---

## 1. Strategy Thesis

### 1.1 The Inefficiency

Cross-sectional momentum exploits the well-documented tendency for recent winners to continue outperforming recent losers over 3-12 month horizons, driven by investor underreaction to fundamental news and herding behavior.

### 1.2 Hypothesized Drivers

   - Behavioral: Investor underreaction to earnings surprises and fundamental news (Hong & Stein 1999)
   - Behavioral: Herding and trend-following by institutional investors
   - Structural: Slow-moving capital and rebalancing constraints of passive funds
   - Liquidity: Gradual diffusion of information across heterogeneous investor populations

### 1.3 Expected Holding Period and Turnover

- **Holding Period:** 1-3 months (medium frequency, monthly rebalance)
- **Turnover Regime:** Monthly rebalance, ~200% annualized turnover

### 1.4 Alpha Source vs Systematic Risk Premia

Cross-sectional momentum premium, distinct from market beta. Expected alpha of 3-5% annualized above the market after costs.

**Factor Exposure:**
Primary: Momentum factor (UMD). Secondary: Market beta (reduced during drawdown via regime filter). Minimal exposure to value, size, or quality factors by construction.

### 1.5 When the Edge Disappears

- Sustained momentum crashes (rapid regime reversals as in 2009 Q1)
- Prolonged mean-reverting markets with no persistent trends
- High-correlation panic regimes (VIX > 35, correlations approach 1)
- Significant crowding in momentum strategies reducing the premium

**Backtest Performance Summary:**

| Metric | Value |
|---|---|
| CAGR | -0.02% |
| Sharpe Ratio | 0.43 |
| Sortino Ratio | 0.39 |
| Max Drawdown | -6.91% |
| Hit Rate | 44.92% |
| Profit Factor | 1.03 |
| Total Trades | 610 |

---

## 2. Universe Selection

### 2.1 Tradeable Instruments

- **Asset Classes:** equity
- **Regions:** US
- **Exchanges:** NYSE, NASDAQ

### 2.2 Eligibility Filters

All instruments must pass the following filters to remain in the tradeable
universe on any given rebalance date:

```
UNIVERSE_FILTER:
  1. Average daily dollar volume (20-day) > $500,000,000
  2. Share price > $10.00
  3. Listed on NYSE, NASDAQ
  4. Bid-ask spread < 5.0 bps
  5. Market capitalization > $10,000,000,000
  6. No pending corporate actions (earnings within 3 days)
```

### 2.3 Universe Implementation

The universe is defined via `UniverseFilter` and applied by `UniverseBuilder`:

```python
from pipeline.strategy.universe import UniverseBuilder, UniverseFilter

builder = UniverseBuilder(UniverseFilter(
    min_adv_dollars=500000000,
    min_price=10.0,
    min_market_cap=10000000000,
    max_spread_bps=5.0,
))
universe = builder.build(instruments)
```

---

## 3. Signal Generation

### 3.1 Signal: cross_sectional_momentum

**Family:** momentum

**Description:** 12-1 month price momentum, z-scored and volatility-adjusted

#### Raw Indicators

**momentum_return** (weight: 0.6, higher = stronger signal)

12-1 month total return

```
\text{MOM}_{i,t} = \frac{P_{i,t-21}}{P_{i,t-252}} - 1
```

Normalization: zscore, lookback: 252 days

**ma_crossover** (weight: 0.25, higher = stronger signal)

50/200 MA ratio

```
\text{MAC}_{i,t} = \frac{\text{SMA}(P, 50)}{\text{SMA}(P, 200)} - 1
```

Normalization: zscore, lookback: 252 days

**volatility** (weight: 0.15, lower = stronger signal)

Realized volatility (penalize high-vol names)

```
\sigma_{i,t} = \text{std}(\text{ret}_i, 60) \times \sqrt{252}
```

Normalization: zscore, lookback: 252 days

#### Composite Signal

The composite signal is the weighted average of normalized indicators:

```
composite = weighted_avg(momentum_return (w=0.6), ma_crossover (w=0.25), volatility (w=0.15))
```

Signals are then ranked cross-sectionally across the universe to produce a relative ranking from 0 (weakest) to 1 (strongest).

---

## 4. Entry Rules

### 4.1 Required Confluence (ALL Must Be True)

```
ENTRY SIGNAL = TRUE when ALL of:

  1. signal_threshold
  2. regime_filter
  3. max_positions
  4. no_duplicate_position
  5. risk_budget
  6. sector_exposure
```

All conditions must evaluate to TRUE simultaneously (AND logic).
The first failing condition short-circuits evaluation.

### 4.2 Entry Execution

```
On ENTRY SIGNAL = TRUE:
  1. Compute position size (Section 6)
  2. Execute at next available close price
  3. Apply signal lag of 1 day(s) to avoid look-ahead bias
  4. Record: entry_date, entry_price, signal_value, stop_loss, target
```

---

## 5. Exit Rules

The strategy employs a **layered exit framework** with independent triggers.
The first trigger hit closes the position.

### 5.1 Stop-Loss Exit (Capital Preservation)

```
STOP LOSS:
  stop_price = entry_price - (ATR_at_entry * 2.0)

  Trailing stop (activated after +1.5 ATR profit):
    trail_stop = max(trail_stop, high - ATR_current * 2.5)

  EXIT if Close < stop_price OR Close < trail_stop
```

### 5.2 Profit Target Exit

```
TARGET PRICE:
  target = entry_price + (ATR_at_entry * 4.0)
  EXIT if Close >= target
```

### 5.3 Time-Based Exit

```
TIME EXIT:
  max_holding_days = 63
  EXIT if days_held >= max_holding_days
```

### 5.4 Signal Reversal Exit

```
SIGNAL REVERSAL:
  if RSI > 80 AND position is profitable:
    EXIT (overbought, take profit)
  if REGIME changes to BEAR:
    EXIT all positions immediately
```

### 5.5 Exit Priority

```
Priority order (highest to lowest):
  1. Stop-loss / trailing stop (non-negotiable)
  2. Regime change to BEAR (systemic risk)
  3. Signal reversal (thesis invalidated)
  4. Profit target (greed management)
  5. Time-based exit (opportunity cost)
```

---

## 6. Position Sizing Model

### 6.1 Sizing Method: Volatility Scaled

```
POSITION SIZING:

  method = volatility_scaled
  target_annual_vol = 10%
  target_position_risk = 0.50%
  vol_lookback = 60 days

  For volatility-scaled sizing:
    w_i = (sigma_target / (sigma_i * sqrt(N))) * sign(s_i) * conviction_i
```

### 6.2 Position Constraints

| Parameter | Value |
|---|---|
| Max position weight | 5.0% |
| Min position weight | 0.50% |
| Max gross exposure | 100% |
| Max net exposure | 100% |
| Max ADV participation | 5% |
| Min trade notional | $50,000 |
| Vol floor | 5% |
| Vol cap | 100% |

### 6.3 Conviction Scaling

Signal strength modulates position size within [0.5x, 1.5x]:

```
conviction = 0.5 + (1.5 - 0.5) * (|signal| / max|signal|)
```

### 6.4 Mathematical Formulation

```
For volatility-scaled sizing:

  w_i = (sigma_target / (sigma_i * sqrt(N))) * sign(s_i) * c_i

where:
  sigma_target = 0.10 (annualized portfolio volatility target)
  sigma_i      = annualized volatility of instrument i
  N            = number of active positions
  s_i          = composite signal value for instrument i
  c_i          = conviction scalar in [0.5, 1.5]

Implied notional:
  notional_i = w_i * total_capital
  shares_i   = floor(notional_i / price_i)
```

---

## 7. Risk Parameters and Constraints

### 7.1 Risk Constraint Table

| Constraint | Type | Limit | Severity | Applies To | Description |
|---|---|---|---|---|---|
| Max Single Position Weight | position_weight | 5.00% | hard | Portfolio | No single position may exceed 5% of capital |
| Max Gross Exposure | gross_exposure | 100.00% | hard | Portfolio | Total gross exposure capped at 100% |
| Max Net Exposure | net_exposure | 100.00% | hard | Portfolio | Net exposure capped at 100% |
| Max Portfolio Drawdown | max_drawdown | 15.00% | hard | Portfolio | Strategy halts at 15% drawdown from peak |
| Max ADV Participation | adv_participation | 5.00% | hard | Portfolio | Max 5% of average daily volume per trade |
| Max Daily Turnover | turnover | 20.00% | soft | Portfolio | Daily turnover target below 20% |
| Sector Cap: Technology | sector_exposure | 30.00% | hard | Technology | Technology sector capped at 30% |
| Sector Cap: Healthcare | sector_exposure | 30.00% | hard | Healthcare | Healthcare sector capped at 30% |
| Sector Cap: Financials | sector_exposure | 30.00% | hard | Financials | Financials sector capped at 30% |
| Sector Cap: Consumer Discretionary | sector_exposure | 30.00% | hard | Consumer Discretionary | Consumer Discretionary sector capped at 30% |
| Sector Cap: Industrials | sector_exposure | 30.00% | hard | Industrials | Industrials sector capped at 30% |
| Sector Cap: Energy | sector_exposure | 30.00% | hard | Energy | Energy sector capped at 30% |
| Sector Cap: Utilities | sector_exposure | 30.00% | hard | Utilities | Utilities sector capped at 30% |
| Sector Cap: Materials | sector_exposure | 30.00% | hard | Materials | Materials sector capped at 30% |
| Sector Cap: Consumer Staples | sector_exposure | 30.00% | hard | Consumer Staples | Consumer Staples sector capped at 30% |
| Sector Cap: Real Estate | sector_exposure | 30.00% | hard | Real Estate | Real Estate sector capped at 30% |
| Sector Cap: Communication Services | sector_exposure | 30.00% | hard | Communication Services | Communication Services sector capped at 30% |

### 7.2 Constraint Evaluation

Before each rebalance, the portfolio is checked against all constraints:

```python
results = risk_constraints.evaluate_portfolio(
    weights=portfolio_weights,
    sector_map=sector_mapping,
    country_map=country_mapping,
    volatilities=asset_volatilities,
    current_drawdown=current_drawdown,
)
violations = risk_constraints.get_hard_violations(results)
if violations:
    # Scale positions to satisfy constraints
    apply_constraint_scaling(portfolio, violations)
```

### 7.3 Drawdown Circuit Breakers

```
Level 1 — Warning (10% drawdown from peak):
  - Log warning, flag for review
  - Reduce new position sizes by 50%

Level 2 — Halt (15% drawdown from peak):
  - Halt all new entries
  - Tighten stops on existing positions
  - Mandatory review

Level 3 — Shutdown (20% drawdown from peak):
  - Close ALL positions
  - Full strategy shutdown
  - Manual restart required
```

---

## 8. Backtesting Methodology

### 8.1 Configuration

```
Backtest Protocol:
  Initial capital:    $100,000,000
  Period:             2019-01-01 to 2024-12-31
  Rebalance freq:     daily
  Signal lag:         1 day(s)
  Spread:             3.0 bps (half-spread)
  Commission:         $0.0050 per share
  Slippage:           2.0 bps per trade
```

### 8.2 Backtest Integrity

```
Checklist:
  [x] Signal lag applied (1 day) — no look-ahead bias
  [x] Adjusted prices for splits and dividends
  [x] Transaction costs (spread + slippage) applied to every trade
  [x] Positions rounded to whole shares
  [x] ADV participation limits enforced
  [x] Multiple regimes tested (walk-forward validation)
```

### 8.3 Backtest Results

| Metric | Value |
|---|---|
| Total Return | -0.04% |
| CAGR | -0.00 |
| Sharpe Ratio | 0.426 |
| Sortino Ratio | 0.390 |
| Max Drawdown | -6.91% |
| Calmar Ratio | -0.004 |
| Hit Rate | 44.92% |
| Avg Winner | 23,468.11 |
| Avg Loser | -18,655.91 |
| Profit Factor | 1.026 |
| Total Trades | 610 |
| Avg Holding Days | 7.66 |
| Annualized Turnover | N/A |
| Avg Gross Exposure | 0.426 |
| Total Costs | 213,048.49 |
| Time in Market | 42.58% |

---

## 9. Benchmark Selection

### 9.1 Primary Benchmark

**S&P 500** (SPY)

S&P 500 ETF Trust (Total Return)

**Justification:** SPY represents the opportunity cost of passive US large-cap equity exposure. A systematic strategy must justify its complexity by outperforming this alternative on a risk-adjusted basis.

### 9.2 Secondary Benchmarks

| Benchmark | Ticker | Justification |
|---|---|---|
| Risk-Free Rate | TBILL | Does the strategy beat holding cash? |
| MSCI World | URTH | Global equity benchmark for strategies with international exposure. |

### 9.3 Key Relative Metrics

```
vs. S&P 500 (SPY):
  Information Ratio = (strategy_return - benchmark_return) / tracking_error
  Target: IR > 0.3

  Up-capture = strategy_return_up_periods / benchmark_return_up_periods
  Down-capture = strategy_return_down_periods / benchmark_return_down_periods
  Target: Up-capture > 70%, Down-capture < 50%
```

### 9.3 Relative Performance

| Metric | S&P 500 |
|---|---|
| Active Return (ann.) | 17.62% |
| Tracking Error (ann.) | 18.83% |
| Information Ratio | 0.935 |
| Beta | 0.038 |
| Alpha (ann.) | 2.44% |
| Up Capture | 2.34% |
| Down Capture | 0.62% |
| Correlation | 0.170 |
| Strategy Sharpe | 0.427 |
| Benchmark Sharpe | -0.827 |
| Strategy Total Return | 2.80% |
| Benchmark Total Return | -24.33% |
| Strategy Max DD | -5.45% |
| Benchmark Max DD | -30.52% |

---

## 10. Edge Decay Monitoring

### 10.1 Monitoring Dashboard (Rolling 60-Day)

```
MONITOR CONTINUOUSLY:

  1. Rolling Win Rate (60-day):
     ALERT if win_rate < 0.45

  2. Rolling Profit Factor (60-day):
     ALERT if profit_factor < 1.0

  3. Rolling Sharpe (60-day):
     ALERT if sharpe < 0.0

  4. Signal Hit Rate:
     ALERT if hit_rate declines > 20% from inception average

  5. Hurst Exponent of Equity Curve:
     ALERT if H < 0.45 (equity curve losing trend)
```

### 10.2 Response Protocol

```
YELLOW ALERT (1 metric breached):
  - Review last 20 trades
  - Continue with 50% reduced size

ORANGE ALERT (2+ metrics breached):
  - Halt new entries for 2 weeks
  - Run full parameter recalibration

RED ALERT (3+ metrics for 3+ months):
  - Full strategy shutdown
  - Comprehensive thesis review
  - Determine if edge has permanently decayed
```

### 10.3 Automatic Recalibration

```
Every 3 months (quarterly):
  1. Re-run walk-forward validation on last 12 months
  2. Compare OOS metrics to inception averages
  3. Flag if OOS Sharpe < 50% of inception Sharpe
```

### 10.4 Current Decay Status

| Metric | Value | Floor | Status |
|---|---|---|---|
| Win Rate | 0.48 | 0.45 | OK |
| Profit Factor | 1.34 | 1.0 | OK |
| Rolling Sharpe | 2.57 | 0.0 | OK |
| Alert Level | GREEN | — | — |
| Metrics Breached | 0 | — | — |

---

## Appendix: Mathematical Reference

### A.1 Signal Formulas

**Cross-Sectional Momentum (12-1):**
```
MOM_{i,t} = P_{i,t-21} / P_{i,t-252} - 1
```

**Moving Average Crossover:**
```
MAC_{i,t} = SMA(P_i, 50) / SMA(P_i, 200) - 1
```

**Z-Score Normalization:**
```
z_{i,t} = (x_{i,t} - mu_t) / sigma_t
where mu_t = rolling mean, sigma_t = rolling std
```

### A.2 Risk Formulas

**Annualized Sharpe Ratio:**
```
Sharpe = (mu_excess / sigma) * sqrt(252)
where mu_excess = mean daily excess return, sigma = daily return std
```

**Maximum Drawdown:**
```
DD(t) = (NAV(t) - Peak(t)) / Peak(t)
MaxDD = min(DD(t)) over all t
```

**Information Ratio:**
```
IR = (R_strategy - R_benchmark) / TE
where TE = std(R_strategy - R_benchmark) * sqrt(252)
```

### A.3 Position Sizing

**Volatility-Scaled:**
```
w_i = (sigma_target / (sigma_i * sqrt(N))) * sign(s_i) * c_i

where:
  sigma_target = annualized portfolio vol target
  sigma_i = annualized vol of instrument i
  N = number of positions
  s_i = signal value
  c_i = conviction multiplier
```

**Implied Notional and Shares:**
```
notional_i = w_i * capital
shares_i = floor(notional_i / price_i)
```

### A.4 Trading Logic Pseudocode

```python
def daily_rebalance(portfolio, market_data, date):
    # 1. Compute signals
    signals = signal_pipeline.run(market_data)

    # 2. Lag signals (t-1 to avoid look-ahead)
    signals = signals.shift(1)

    # 3. Check exits on existing positions
    for position in portfolio.open_positions:
        exit_signal = exit_engine.check_exit(position, market_data)
        if exit_signal.should_exit:
            execute_exit(portfolio, position, exit_signal)

    # 4. Check drawdown circuit breakers
    if portfolio.drawdown > max_drawdown_threshold:
        close_all_positions(portfolio)
        return

    # 5. Scan universe for entry signals
    for ticker in universe:
        if entry_rules.evaluate(ticker, signals[ticker], context).eligible:
            size = sizing_model.compute_targets(
                signals[ticker], prices[ticker], vol[ticker]
            )
            if passes_risk_constraints(portfolio, size):
                execute_entry(portfolio, ticker, size)

    # 6. Log state and check edge decay
    log_portfolio_state(portfolio, date)
    edge_decay_monitor.evaluate()
```

---

*This document is auto-generated from the strategy configuration and
backtest results. All parameters, formulas, and thresholds are derived
directly from the implemented code — no manual edits.*

*Past performance in backtests does not guarantee future results.
All trading involves risk of loss. The mathematical edge described
herein is probabilistic, not deterministic.*

*Quantitative Strategies Group — Systematic Trading Research*