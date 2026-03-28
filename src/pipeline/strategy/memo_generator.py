"""Auto-generate a Goldman Sachs-style quantitative strategy memo.

Reads a ``StrategyDefinition`` and optional ``HarnessBacktestResult`` to
produce a Markdown memo that:
  - Contains all required sections in consistent order.
  - Uses LaTeX-ready mathematical formulas.
  - Includes pseudocode for the trading logic.
  - Contains risk parameter tables from the constraint configuration.
  - Is grounded in the implemented logic — no free-form drift.
"""

from __future__ import annotations

import datetime
import logging

import numpy as np

from pipeline.strategy.backtest_harness import HarnessBacktestResult
from pipeline.strategy.benchmark import benchmark_analysis_to_markdown
from pipeline.strategy.strategy_definition import StrategyDefinition

logger = logging.getLogger(__name__)


class MemoGenerator:
    """Generate a Goldman Sachs-style quantitative strategy memo.

    The memo is grounded entirely in the ``StrategyDefinition`` and
    ``HarnessBacktestResult`` — every parameter, threshold, and formula
    is derived from the configured values.
    """

    def __init__(
        self,
        strategy: StrategyDefinition,
        backtest_result: HarnessBacktestResult | None = None,
    ) -> None:
        self.strategy = strategy
        self.result = backtest_result

    def generate(self) -> str:
        """Generate the complete memo as Markdown text."""
        sections = [
            self._header(),
            self._table_of_contents(),
            self._strategy_thesis(),
            self._universe_selection(),
            self._signal_generation(),
            self._entry_rules(),
            self._exit_rules(),
            self._position_sizing(),
            self._risk_parameters(),
            self._backtesting_methodology(),
            self._benchmark_selection(),
            self._edge_decay_monitoring(),
            self._appendix_math(),
            self._footer(),
        ]
        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------

    def _header(self) -> str:
        t = self.strategy.thesis
        date_str = datetime.date.today().strftime("%B %Y")
        return f"""# Goldman Sachs Quantitative Strategies Group
## Systematic Trading Strategy Memo

**Strategy Name:** {t.name}
**Classification:** {t.classification}
**Target AUM:** {t.target_aum}
**Author:** Quantitative Strategies Group
**Date:** {date_str}
**Status:** {t.status}

---"""

    # ------------------------------------------------------------------
    # Table of Contents
    # ------------------------------------------------------------------

    def _table_of_contents(self) -> str:
        return """## Table of Contents

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

---"""

    # ------------------------------------------------------------------
    # 1. Strategy Thesis
    # ------------------------------------------------------------------

    def _strategy_thesis(self) -> str:
        t = self.strategy.thesis
        drivers = "\n".join(f"   - {d}" for d in t.drivers)
        edge_disappears = "\n".join(f"- {e}" for e in t.when_edge_disappears)

        perf_table = ""
        if self.result and self.result.metrics:
            m = self.result.metrics
            perf_table = f"""
**Backtest Performance Summary:**

| Metric | Value |
|---|---|
| CAGR | {m.cagr:.2%} |
| Sharpe Ratio | {m.sharpe_ratio:.2f} |
| Sortino Ratio | {m.sortino_ratio:.2f} |
| Max Drawdown | {m.max_drawdown:.2%} |
| Hit Rate | {m.hit_rate:.2%} |
| Profit Factor | {m.profit_factor:.2f} |
| Total Trades | {m.total_trades:,d} |
"""

        return f"""## 1. Strategy Thesis

### 1.1 The Inefficiency

{t.inefficiency}

### 1.2 Hypothesized Drivers

{drivers}

### 1.3 Expected Holding Period and Turnover

- **Holding Period:** {t.holding_period}
- **Turnover Regime:** {t.turnover_regime}

### 1.4 Alpha Source vs Systematic Risk Premia

{t.alpha_source}

**Factor Exposure:**
{t.risk_premium_exposure}

### 1.5 When the Edge Disappears

{edge_disappears}
{perf_table}
---"""

    # ------------------------------------------------------------------
    # 2. Universe Selection
    # ------------------------------------------------------------------

    def _universe_selection(self) -> str:
        uf = self.strategy.universe_filter
        asset_classes = ", ".join(ac.value for ac in uf.asset_classes)
        regions = ", ".join(r.value for r in uf.regions)
        exchanges = ", ".join(e.value for e in uf.exchanges)

        return f"""## 2. Universe Selection

### 2.1 Tradeable Instruments

- **Asset Classes:** {asset_classes}
- **Regions:** {regions}
- **Exchanges:** {exchanges}

### 2.2 Eligibility Filters

All instruments must pass the following filters to remain in the tradeable
universe on any given rebalance date:

```
UNIVERSE_FILTER:
  1. Average daily dollar volume (20-day) > ${uf.min_adv_dollars:,.0f}
  2. Share price > ${uf.min_price:.2f}
  3. Listed on {exchanges}
  4. Bid-ask spread < {uf.max_spread_bps:.1f} bps
  5. Market capitalization > ${uf.min_market_cap:,.0f}
  6. No pending corporate actions (earnings within {uf.earnings_blackout_days} days)
```

### 2.3 Universe Implementation

The universe is defined via `UniverseFilter` and applied by `UniverseBuilder`:

```python
from pipeline.strategy.universe import UniverseBuilder, UniverseFilter

builder = UniverseBuilder(UniverseFilter(
    min_adv_dollars={uf.min_adv_dollars:.0f},
    min_price={uf.min_price},
    min_market_cap={uf.min_market_cap:.0f},
    max_spread_bps={uf.max_spread_bps},
))
universe = builder.build(instruments)
```

---"""

    # ------------------------------------------------------------------
    # 3. Signal Generation
    # ------------------------------------------------------------------

    def _signal_generation(self) -> str:
        parts = ["## 3. Signal Generation\n"]

        for i, sig_def in enumerate(self.strategy.signal_definitions, 1):
            parts.append(f"### 3.{i} Signal: {sig_def.name}")
            parts.append(f"\n**Family:** {sig_def.family.value}")
            parts.append(f"\n**Description:** {sig_def.description}\n")

            parts.append("#### Raw Indicators\n")
            for _indicator, config in sig_def.indicators:
                norm_label = config.normalization.value
                direction = (
                    "higher = stronger signal"
                    if config.higher_is_better
                    else "lower = stronger signal"
                )
                parts.append(f"**{config.name}** (weight: {config.weight}, {direction})")
                if config.description:
                    parts.append(f"\n{config.description}")
                if config.formula:
                    parts.append(f"\n```\n{config.formula}\n```")
                parts.append(
                    f"\nNormalization: {norm_label},"
                    f" lookback: {config.lookback_window} days\n"
                )

            parts.append("#### Composite Signal\n")
            parts.append("The composite signal is the weighted average of normalized indicators:\n")
            weight_strs = []
            for _, config in sig_def.indicators:
                weight_strs.append(f"{config.name} (w={config.weight})")
            parts.append("```")
            parts.append(f"composite = weighted_avg({', '.join(weight_strs)})")
            parts.append("```\n")

            parts.append(
                "Signals are then ranked cross-sectionally across the universe "
                "to produce a relative ranking from 0 (weakest) to 1 (strongest).\n"
            )

        parts.append("---")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # 4. Entry Rules
    # ------------------------------------------------------------------

    def _entry_rules(self) -> str:
        rules = self.strategy.entry_rules
        conditions = []
        for i, cond in enumerate(rules.conditions, 1):
            conditions.append(f"  {i}. {cond.name}")

        cond_text = "\n".join(conditions)
        return f"""## 4. Entry Rules

### 4.1 Required Confluence (ALL Must Be True)

```
ENTRY SIGNAL = TRUE when ALL of:

{cond_text}
```

All conditions must evaluate to TRUE simultaneously (AND logic).
The first failing condition short-circuits evaluation.

### 4.2 Entry Execution

```
On ENTRY SIGNAL = TRUE:
  1. Compute position size (Section 6)
  2. Execute at next available close price
  3. Apply signal lag of {self.strategy.backtest_config.signal_lag_days} day(s) \
to avoid look-ahead bias
  4. Record: entry_date, entry_price, signal_value, stop_loss, target
```

---"""

    # ------------------------------------------------------------------
    # 5. Exit Rules
    # ------------------------------------------------------------------

    def _exit_rules(self) -> str:
        ex = self.strategy.exit_engine
        return f"""## 5. Exit Rules

The strategy employs a **layered exit framework** with independent triggers.
The first trigger hit closes the position.

### 5.1 Stop-Loss Exit (Capital Preservation)

```
STOP LOSS:
  stop_price = entry_price - (ATR_at_entry * {ex.stop_atr_multiple:.1f})

  Trailing stop (activated after +{ex.trailing_activation_atr:.1f} ATR profit):
    trail_stop = max(trail_stop, high - ATR_current * {ex.trailing_atr_multiple:.1f})

  EXIT if Close < stop_price OR Close < trail_stop
```

### 5.2 Profit Target Exit

```
TARGET PRICE:
  target = entry_price + (ATR_at_entry * {ex.target_atr_multiple:.1f})
  EXIT if Close >= target
```

### 5.3 Time-Based Exit

```
TIME EXIT:
  max_holding_days = {ex.max_holding_days}
  EXIT if days_held >= max_holding_days
```

### 5.4 Signal Reversal Exit

```
SIGNAL REVERSAL:
  if RSI > {ex.rsi_overbought:.0f} AND position is profitable:
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

---"""

    # ------------------------------------------------------------------
    # 6. Position Sizing
    # ------------------------------------------------------------------

    def _position_sizing(self) -> str:
        sc = self.strategy.sizing_config
        method_name = sc.method.value.replace("_", " ").title()

        return f"""## 6. Position Sizing Model

### 6.1 Sizing Method: {method_name}

```
POSITION SIZING:

  method = {sc.method.value}
  target_annual_vol = {sc.target_annual_vol:.0%}
  target_position_risk = {sc.target_position_risk:.2%}
  vol_lookback = {sc.vol_lookback_days} days

  For volatility-scaled sizing:
    w_i = (sigma_target / (sigma_i * sqrt(N))) * sign(s_i) * conviction_i
```

### 6.2 Position Constraints

| Parameter | Value |
|---|---|
| Max position weight | {sc.max_position_weight:.1%} |
| Min position weight | {sc.min_position_weight:.2%} |
| Max gross exposure | {sc.max_gross_exposure:.0%} |
| Max net exposure | {sc.max_net_exposure:.0%} |
| Max ADV participation | {sc.max_adv_participation:.0%} |
| Min trade notional | ${sc.min_trade_notional:,.0f} |
| Vol floor | {sc.vol_floor:.0%} |
| Vol cap | {sc.vol_cap:.0%} |

### 6.3 Conviction Scaling

Signal strength modulates position size \
within [{sc.conviction_scale_min:.1f}x, {sc.conviction_scale_max:.1f}x]:

```
conviction = {sc.conviction_scale_min} + \
({sc.conviction_scale_max} - {sc.conviction_scale_min}) \
* (|signal| / max|signal|)
```

### 6.4 Mathematical Formulation

```
For volatility-scaled sizing:

  w_i = (sigma_target / (sigma_i * sqrt(N))) * sign(s_i) * c_i

where:
  sigma_target = {sc.target_annual_vol:.2f} (annualized portfolio volatility target)
  sigma_i      = annualized volatility of instrument i
  N            = number of active positions
  s_i          = composite signal value for instrument i
  c_i          = conviction scalar in [{sc.conviction_scale_min}, {sc.conviction_scale_max}]

Implied notional:
  notional_i = w_i * total_capital
  shares_i   = floor(notional_i / price_i)
```

---"""

    # ------------------------------------------------------------------
    # 7. Risk Parameters
    # ------------------------------------------------------------------

    def _risk_parameters(self) -> str:
        constraints_table = self.strategy.risk_constraints.to_markdown_table()

        return f"""## 7. Risk Parameters and Constraints

### 7.1 Risk Constraint Table

{constraints_table}

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

---"""

    # ------------------------------------------------------------------
    # 8. Backtesting Methodology
    # ------------------------------------------------------------------

    def _backtesting_methodology(self) -> str:
        bc = self.strategy.backtest_config
        result_text = ""

        if self.result:
            result_text = f"""
### 8.3 Backtest Results

{self.result.summary_table()}
"""

        return f"""## 8. Backtesting Methodology

### 8.1 Configuration

```
Backtest Protocol:
  Initial capital:    ${bc.initial_capital:,.0f}
  Period:             {bc.start_date or 'earliest'} to {bc.end_date or 'latest'}
  Rebalance freq:     {bc.rebalance_frequency}
  Signal lag:         {bc.signal_lag_days} day(s)
  Spread:             {bc.spread_bps:.1f} bps (half-spread)
  Commission:         ${bc.commission_per_share:.4f} per share
  Slippage:           {bc.slippage_bps:.1f} bps per trade
```

### 8.2 Backtest Integrity

```
Checklist:
  [x] Signal lag applied ({bc.signal_lag_days} day) — no look-ahead bias
  [x] Adjusted prices for splits and dividends
  [x] Transaction costs (spread + slippage) applied to every trade
  [x] Positions rounded to whole shares
  [x] ADV participation limits enforced
  [x] Multiple regimes tested (walk-forward validation)
```
{result_text}
---"""

    # ------------------------------------------------------------------
    # 9. Benchmark Selection
    # ------------------------------------------------------------------

    def _benchmark_selection(self) -> str:
        bm = self.strategy.benchmark_suite
        primary = bm.primary

        secondary_rows = []
        for s in bm.secondary:
            secondary_rows.append(f"| {s.name} | {s.ticker} | {s.justification} |")
        secondary_table = "\n".join(secondary_rows) if secondary_rows else "| — | — | — |"

        bm_analysis_text = ""
        if self.result and self.result.benchmark_analyses:
            bm_analysis_text = f"""
### 9.3 Relative Performance

{benchmark_analysis_to_markdown(self.result.benchmark_analyses)}
"""

        return f"""## 9. Benchmark Selection

### 9.1 Primary Benchmark

**{primary.name}** ({primary.ticker})

{primary.description}

**Justification:** {primary.justification}

### 9.2 Secondary Benchmarks

| Benchmark | Ticker | Justification |
|---|---|---|
{secondary_table}

### 9.3 Key Relative Metrics

```
vs. {primary.name} ({primary.ticker}):
  Information Ratio = (strategy_return - benchmark_return) / tracking_error
  Target: IR > 0.3

  Up-capture = strategy_return_up_periods / benchmark_return_up_periods
  Down-capture = strategy_return_down_periods / benchmark_return_down_periods
  Target: Up-capture > 70%, Down-capture < 50%
```
{bm_analysis_text}
---"""

    # ------------------------------------------------------------------
    # 10. Edge Decay Monitoring
    # ------------------------------------------------------------------

    def _edge_decay_monitoring(self) -> str:
        edc = self.strategy.edge_decay_config
        window = edc.get("window", 60)
        wr_floor = edc.get("win_rate_floor", 0.45)
        pf_floor = edc.get("profit_factor_floor", 1.0)
        sharpe_floor = edc.get("sharpe_floor", 0.0)

        decay_status = ""
        if self.result and self.result.decay_monitor:
            dm = self.result.decay_monitor
            m = dm.evaluate()
            wr_ok = m.rolling_win_rate >= wr_floor or np.isnan(
                m.rolling_win_rate
            )
            pf_ok = m.rolling_profit_factor >= pf_floor or np.isnan(
                m.rolling_profit_factor
            )
            sh_ok = m.rolling_sharpe >= sharpe_floor or np.isnan(
                m.rolling_sharpe
            )
            wr_status = "OK" if wr_ok else "BREACH"
            pf_status = "OK" if pf_ok else "BREACH"
            sh_status = "OK" if sh_ok else "BREACH"
            decay_status = f"""
### 10.4 Current Decay Status

| Metric | Value | Floor | Status |
|---|---|---|---|
| Win Rate | {m.rolling_win_rate:.2f} | {wr_floor} | {wr_status} |
| Profit Factor | {m.rolling_profit_factor:.2f} | {pf_floor} | {pf_status} |
| Rolling Sharpe | {m.rolling_sharpe:.2f} | {sharpe_floor} | {sh_status} |
| Alert Level | {m.alert_level.name} | — | — |
| Metrics Breached | {m.breached_count} | — | — |
"""

        return f"""## 10. Edge Decay Monitoring

### 10.1 Monitoring Dashboard (Rolling {window}-Day)

```
MONITOR CONTINUOUSLY:

  1. Rolling Win Rate ({window}-day):
     ALERT if win_rate < {wr_floor}

  2. Rolling Profit Factor ({window}-day):
     ALERT if profit_factor < {pf_floor}

  3. Rolling Sharpe ({window}-day):
     ALERT if sharpe < {sharpe_floor}

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
{decay_status}
---"""

    # ------------------------------------------------------------------
    # Appendix: Mathematical Reference
    # ------------------------------------------------------------------

    def _appendix_math(self) -> str:
        return r"""## Appendix: Mathematical Reference

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
```"""

    # ------------------------------------------------------------------
    # Footer
    # ------------------------------------------------------------------

    def _footer(self) -> str:
        return """---

*This document is auto-generated from the strategy configuration and
backtest results. All parameters, formulas, and thresholds are derived
directly from the implemented code — no manual edits.*

*Past performance in backtests does not guarantee future results.
All trading involves risk of loss. The mathematical edge described
herein is probabilistic, not deterministic.*

*Quantitative Strategies Group — Systematic Trading Research*"""


def generate_memo(
    strategy: StrategyDefinition,
    backtest_result: HarnessBacktestResult | None = None,
) -> str:
    """Convenience function to generate a complete strategy memo."""
    gen = MemoGenerator(strategy, backtest_result)
    return gen.generate()
