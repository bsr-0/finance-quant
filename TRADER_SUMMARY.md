# Trader Summary

**For:** Discretionary or systematic trader evaluating this system
**Read time:** ~10 minutes
**Date:** 2026-03-04

---

## What This System Does

This is a **research-grade quantitative trading platform** that generates daily trading signals for liquid US equities and ETFs. It scores the universe of instruments on a 0–100 composite signal, applies risk management controls, and outputs a standardized trade blotter.

**It is NOT connected to any broker.** Signals must be executed manually or integrated with a broker API before live trading.

---

## Strategy 1: QSG-MICRO-SWING-001 — Pullback Reversion

### The Bet

Buy liquid stocks (SPY, QQQ, AAPL, MSFT, etc.) when they pull back within an uptrend. Hold 5–15 days for mean reversion.

### Why It Should Work

Retail investors overreact to short-term dips in trending stocks. When RSI drops below 35 and price touches the lower Bollinger Band while the 50/200-day moving averages confirm an uptrend, the selling is typically excessive. The strategy buys this pullback and captures the reversion.

### Entry Signal

A composite score (0–100) from four categories:
- **Trend** (40 pts): SMA alignment confirms uptrend
- **Pullback** (30 pts): RSI, Bollinger, Stochastic all confirm oversold
- **Volume** (15 pts): Low volume on pullback = sellers exhausted
- **Momentum** (15 pts): MACD improving, volatility in sweet spot

Entry requires score ≥ 60 in BULL regime, ≥ 70 in NEUTRAL. No entries in BEAR.

### Risk Controls

| Control | Threshold | Action |
|---------|-----------|--------|
| Hard stop-loss | -1.5× ATR | Mandatory exit |
| Trailing stop | 2× ATR from high | Locks in gains |
| Daily loss limit | 2% of equity | Blocks new entries |
| Drawdown (YELLOW) | 5% | Reduce size 50%, require score ≥ 75 |
| Drawdown (ORANGE) | 10% | No new entries |
| Drawdown (RED) | 15% | Close everything. 30-day cooldown. |
| 4 consecutive losses | — | Halt until a winner |

### Target Performance

| Metric | Target |
|--------|--------|
| CAGR | 8–15% net |
| Max Drawdown | < 10% |
| Win Rate | > 55% |
| Sharpe Ratio | > 0.8 |
| Holding Period | 5–12 days |

### When It Works

- Bull markets with moderate volatility (VIX 15–25)
- Range-bound markets with clear support/resistance
- When retail panic creates buyable dips in strong stocks

### When It Fails

- Sustained bear markets (regime filter blocks entries, which is the correct behavior)
- Flash crashes or gap-down opens beyond the stop level
- Low-volatility grind-up markets with no pullbacks (no signals generated)

---

## Strategy 2: QSG-SYSTEMATIC-MOM-001 — Cross-Sectional Momentum

### Status: UNDERPERFORMING

| Metric | Value |
|--------|-------|
| CAGR | -0.02% |
| Sharpe | 0.43 |
| Max DD | -6.91% |

**Do not trade this strategy in its current form.** The momentum strategy has near-zero alpha after costs. It needs redesign (add short leg, shorter lookback, crash protection) before it merits capital.

---

## How to Run Daily

### 1. Update Prices

Download latest OHLCV data for the universe to `data/prices/`:

```bash
python -m pipeline.cli extract prices --start 2024-01-01 --end 2024-12-31
```

### 2. Generate Signals

```bash
python -m pipeline.cli generate-signals \
    --prices-dir data/prices/ \
    --output data/signals/ \
    --threshold 60
```

This produces `data/signals/signals_YYYYMMDD.csv` with columns:

| Column | Description |
|--------|-------------|
| ticker | Instrument |
| direction | LONG |
| score | 0–100 composite |
| confidence | HIGH / MEDIUM / LOW |
| entry_price | Suggested entry (close) |
| stop_price | Hard stop-loss level |
| target_1 | First profit target (+2× ATR) |
| target_2 | Second profit target (+3× ATR) |
| regime | BULL / NEUTRAL / BEAR |

### 3. Review the Blotter

- **HIGH confidence (score ≥ 80):** Strongest signals. Full position size.
- **MEDIUM confidence (score 70–79):** Good signals. 75% position size.
- **LOW confidence (score 60–69):** Marginal signals. 50% position size.

Only trade signals where you agree with the thesis. The system is a **tool**, not an oracle.

### 4. Execute

Place orders manually via your broker. Use market-on-close or limit orders near the suggested entry price. Set stops and targets as indicated.

### 5. Monitor

After entering, track:
- Has the stop been hit? → Exit immediately
- Has the trailing stop activated (1 ATR gain from entry)? → Watch for trailing stop trigger
- Is RSI > 70 with profit? → Consider taking profit
- Has 15 days elapsed? → Exit regardless

---

## What to Do If Something Breaks

| Situation | Action |
|-----------|--------|
| No signals generated | Check that price data is fresh and not empty. Check regime — BEAR = no signals by design. |
| Data download fails | Yahoo Finance rate limit or outage. Wait and retry. Do not trade with stale data. |
| Circuit breaker triggers | YELLOW: reduce size. ORANGE: stop entering. RED: close everything and take 30 days off. |
| Edge decay warning | YELLOW: review last 20 trades for pattern. ORANGE: halt for 2 weeks. RED: stop using the strategy. |
| Signal seems wrong | Check the individual score components (trend, pullback, volume, volatility). If any component seems stale or incorrect, skip the signal. |

---

## What You Must Accept Before Using This System

1. **No live trading integration.** You must execute signals manually or build your own broker connection.
2. **Backtest ≠ live performance.** Transaction costs, slippage, and timing differences will reduce live returns vs backtest.
3. **The stop-loss is not guaranteed.** Overnight gaps can blow through stops. The system models exits at close prices.
4. **Past signals do not predict future performance.** The strategy is based on a behavioral hypothesis that may weaken over time.
5. **You are responsible for your own risk.** This system provides signals, not financial advice. Size positions according to your own risk tolerance.
6. **The momentum strategy does not work.** QSG-SYSTEMATIC-MOM-001 has negative expected returns in its current form. Do not trade it.
7. **Single data source risk.** All price data comes from Yahoo Finance. API failures mean no signals (fail-safe, but inconvenient).

---

## Key Files

| File | Purpose |
|------|---------|
| `src/pipeline/cli.py` | CLI entry point (all commands) |
| `src/pipeline/strategy/engine.py` | Swing strategy backtest engine |
| `src/pipeline/strategy/signals.py` | Signal computation (0–100 score) |
| `src/pipeline/strategy/exits.py` | 7-trigger exit framework |
| `src/pipeline/strategy/sizing.py` | Risk-based position sizing |
| `src/pipeline/strategy/risk.py` | Drawdown circuit breakers |
| `src/pipeline/strategy/signal_output.py` | Standardized signal CSV output |
| `src/pipeline/strategy/pre_trade_checks.py` | Pre-trade validation |
| `config.yaml` | Configuration (universe, parameters) |
| `STRATEGY_SPEC.md` | Full strategy formalization |
| `RISK_FRAMEWORK.md` | Risk management details |
