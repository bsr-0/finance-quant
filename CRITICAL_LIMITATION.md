# Critical Limitation #1: No Persistent Position State Across Daily Runs

**Date:** 2026-04-07
**Severity:** CRITICAL — live multi-day trading is broken; positions lose all exit metadata overnight

---

## Summary

The `PositionMonitor` holds all tracked position state (entry price, stop-loss, trailing stop, profit targets, ATR, signal score) **exclusively in memory**. When the daily trading runner exits, this state is discarded. The next day's run starts with an empty position register — the system forgets every open position.

This is fatal for the QSG-MICRO-SWING-001 strategy, which holds positions for **5-21 days** (`max_holding_days=15`). A position opened on Day 1 becomes invisible to the exit engine on Day 2+.

---

## Evidence

### 1. Position state is in-memory only

**File:** `src/pipeline/execution/position_monitor.py:140`

```python
self._tracked: dict[str, TrackedPosition] = {}
```

No `save()`, `load()`, or `persist()` method exists on `PositionMonitor`. The `_tracked` dict is created empty at construction and lost when the object is garbage collected.

### 2. ATR is hardcoded to zero on registration

**File:** `src/pipeline/execution/runner.py:205`

```python
atr_at_entry=0.0,  # Set from signal if available
```

The signal executor outputs ATR in its result details, but the runner never extracts it. The position monitor needs ATR for exit calculations (stop-loss = entry - 1.5x ATR, trailing = 2x ATR).

### 3. Runner creates a fresh monitor every session

**File:** `src/pipeline/execution/runner.py` — `TradingRunner.__init__()` creates a new `PositionMonitor()`. No mechanism loads yesterday's positions.

### 4. Strategy requires multi-day holding

**File:** `src/pipeline/strategy/engine.py` — `StrategyConfig.max_holding_days=15`

The swing strategy holds positions for up to 15 trading days. Every day after Day 1, the exit engine should check stop-loss, trailing stop, profit targets, and time exit — but it can't, because the position metadata is gone.

---

## What Breaks

| Day | What Happens | Problem |
|-----|-------------|---------|
| Day 1 | Signal fires, position opened, stop=$95, target=$115, ATR=2.1 | Works correctly |
| Day 2 | Runner restarts, `_tracked` is empty | Position exists at broker but system doesn't know about it |
| Day 2+ | Stock drops to $93 | Stop-loss at $95 should have fired — but monitor has no record of the position or its stop |
| Day 5 | Stock gaps down 10% | No trailing stop, no circuit breaker awareness for this "unknown" position |
| Day 15 | Time exit should fire | System has no idea position has been held for 15 days |

### Reconciler sees the mismatch but can't fix it

The `Reconciler` (`execution/reconciler.py`) compares system positions vs broker positions and flags discrepancies. But it only reports — it doesn't reconstruct `TrackedPosition` metadata (stop price, targets, ATR). So it knows a position *exists* but can't manage it.

---

## Why This Is #1

The previous #1 limitation (no signal alpha validation) was about whether signals *should* trade. This limitation is about whether the system can *safely manage* trades it has already entered. A position without exit monitoring is an unmanaged risk — exactly the scenario circuit breakers and stop-losses are designed to prevent.

| Previous Limitations | Status |
|---------------------|--------|
| No IC testing | **Fixed** — `walk_forward_ic()`, `ic_decay_analysis()`, CLI command |
| No multiple-testing correction | **Fixed** — `SignalTrialRegistry` with BH FDR |
| Deflated Sharpe unused | **Fixed** — wired into evaluator + signal selection |
| Walk-forward embargo | **Fixed** — `label_horizon` parameter |
| Signal weights hardcoded | **Fixed** — `optimize_weights()` via CV ridge |
| **Position state not persisted** | **OPEN — this document** |

---

## Recommended Remediation

### Phase 1: Position register (blocking for live deployment)

1. **`src/pipeline/execution/position_register.py`** (new file) — JSON-backed persistent store for `TrackedPosition` objects. Save after every state change (open, partial close, stop update, trailing activation). Load on runner startup.

2. **Fix ATR extraction** in `runner.py:205` — change `atr_at_entry=0.0` to `atr_at_entry=detail.get("atr", 0.0)`.

3. **Load broker positions on startup** — `TradingRunner.__init__()` should query the broker for open positions and cross-reference against the persistent register. Positions found at the broker but missing from the register should trigger a WARNING and be registered with conservative defaults.

### Phase 2: Multi-day integration test

4. **End-to-end multi-day test** — simulate Day 1 (open positions), Day 2 (verify positions loaded, exits checked), Day 3 (stop-loss fires). This test does not exist in the current 876-test suite.

---

## Conclusion

The system has strong signal validation (now with IC testing and FDR correction), solid backtesting infrastructure, and a well-designed exit engine. But the exit engine is disconnected from multi-day reality. It can evaluate exit conditions perfectly — for positions it knows about. After a daily restart, it knows about none of them. Until position state is persisted, live trading with holding periods >1 day will run without stop-losses, trailing stops, or profit targets — the exact scenario the risk management framework was built to prevent.
