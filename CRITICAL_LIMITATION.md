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

## Remediation Status

All items have been implemented:

| Item | Status | Implementation |
|------|--------|----------------|
| Position register | **Done** | `PositionRegister` in `execution/position_register.py` — JSON-backed, atomic writes, fcntl locking |
| Monitor persistence | **Done** | `PositionMonitor` loads from register on init, persists after every state change |
| ATR/target/score extraction | **Done** | `runner.py` now extracts `atr`, `target_1`, `target_2`, `score` from executor details |
| Executor detail fields | **Done** | Both ORDER_SUBMITTED and DRY_RUN paths now include all signal fields |
| Multi-day flow test | **Done** | `TestMultiDayFlow.test_day1_open_day2_exit` verifies Day 1 → Day 2 stop-loss |
| Backward compatibility | **Done** | `position_register=None` keeps old behavior (no persistence) |

---

## Conclusion

With position state now persisted across daily runs, the system can safely hold positions for the strategy's intended 5-21 day holding period. Stop-losses, trailing stops, profit targets, and time exits all survive across runner restarts.
