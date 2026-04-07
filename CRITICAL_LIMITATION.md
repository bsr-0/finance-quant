# Critical Limitation #1: No Statistical Validation of Signal Alpha

**Date:** 2026-04-07
**Severity:** CRITICAL — renders all backtest results unreliable for capital allocation decisions

---

## Summary

The system's signals are constructed from **8 signal families** combined with **hardcoded weights** (trend=40%, pullback=30%, momentum=20%, volume=10%) and **arbitrary entry thresholds** (60, 70) that have never been statistically validated for predictive power. There is no framework to determine whether any signal — individually or in combination — produces returns distinguishable from noise.

This single limitation cascades through every downstream component: backtests report performance that may be entirely attributable to overfitting, risk models size positions based on unvalidated edge assumptions, and the execution layer would deploy real capital on signals with unknown alpha.

---

## Why This Is #1

The repo has 47 documented deficiencies (see `GAP_ANALYSIS.md`). Many have been remediated — confidence intervals via Lo (2002), Holm-Bonferroni corrections for factor correlations, proper p-values in bias checks, block bootstrap, Cornish-Fisher VaR, etc. These are real improvements.

But they all share a common dependency: **they assume the underlying signal has alpha to measure**. If the signal is noise, then:

- Sharpe confidence intervals correctly bound a meaningless number
- Walk-forward validation faithfully evaluates an overfit model
- Transaction cost models accurately price trades that shouldn't happen
- Risk controls carefully size positions in a losing strategy

The signal is the foundation. Without validating it, every other component is precisely wrong.

---

## Evidence in the Codebase

### 1. Signal weights are hardcoded, not learned or validated

**File:** `src/pipeline/strategy/signals.py`

The `SignalEngine` combines sub-signals with fixed weights defined at construction time. These weights were chosen by the developer, not by any optimization or cross-validated selection process. There is no record of how these weights were determined or whether alternatives were tested.

### 2. Entry thresholds are arbitrary

**File:** `src/pipeline/strategy/backtest_harness.py`

The composite signal is compared against thresholds (e.g., 60 for entry) that are constants in the strategy configuration. No threshold sensitivity analysis exists. No walk-forward optimization of thresholds has been performed.

### 3. No Information Coefficient (IC) testing

The repo has no function to compute:
- Rolling IC (rank correlation between signal and forward returns)
- IC information ratio (mean IC / std IC)
- IC decay curves (predictive power vs. holding horizon)

These are the standard metrics for determining whether a signal has alpha. Their absence means the system literally cannot answer: "Does this signal predict returns?"

### 4. No multiple-testing correction for signal discovery

**File:** `src/pipeline/strategy/signal_library.py`

Eight signal families are defined and combined. The `holm_bonferroni()` function exists in `eval/robustness.py` and was wired into factor analysis — but it is **never applied to signal selection itself**. With 8 families, each with multiple parameterizations, the probability of finding a spuriously significant signal by chance is substantial.

The `GAP_ANALYSIS.md` acknowledges this explicitly:

> *"This requires a signal research framework (walk-forward IC testing) that is beyond the scope of a single fix. Flagged as critical follow-up."*

That follow-up has not been completed.

### 5. Walk-forward validation lacks embargo

**File:** `src/pipeline/backtesting/walk_forward.py:118`

Default `step_size = test_size` creates zero-gap folds. When labels overlap (e.g., 5-day forward returns), information leaks from test into subsequent training. This inflates apparent out-of-sample performance, masking whether the signal generalizes.

### 6. No deflated Sharpe integration into signal selection

The `deflated_sharpe_ratio()` function exists in `eval/robustness.py` but is a standalone metric. It is not integrated into any signal selection or strategy validation pipeline. The deflated Sharpe explicitly adjusts for the number of strategies tried — exactly the correction needed — but it sits unused.

---

## Impact Assessment

| Component | How This Limitation Affects It |
|-----------|-------------------------------|
| **Backtesting** | Reports returns from potentially overfit signals; no way to distinguish alpha from noise |
| **Risk Management** | Sizes positions assuming edge exists; if edge = 0, optimal position = 0 |
| **Execution** | Would deploy real capital on unvalidated signals |
| **Walk-Forward** | Validates an unvalidated signal; leaky embargo compounds the problem |
| **Performance Reports** | Sharpe/Sortino/IR metrics are meaningless if the signal has no predictive power |
| **Live Readiness** | The checklist scores Signal Generation at 12/15, but signal *validity* is 0/15 |

---

## Recommended Remediation

### Phase 1: Signal Alpha Validation (blocking for any live deployment)

1. **IC analysis framework** — Compute rolling Spearman IC between each signal and N-day forward returns. Require IC > 0.02 with t-stat > 2.0 over walk-forward folds.

2. **Deflated Sharpe gate** — Integrate `deflated_sharpe_ratio()` into the strategy evaluation pipeline. Reject strategies where DSR p-value > 0.05 after accounting for all strategies tested.

3. **Walk-forward embargo** — Change default embargo to `max(label_horizon, 5)` trading days. This is a one-line fix with outsized impact.

4. **Signal weight optimization** — Replace hardcoded weights with cross-validated optimization (ridge regression of sub-signals on forward returns) within walk-forward folds.

### Phase 2: Ongoing signal monitoring (required before scaling capital)

5. **IC decay monitoring** — Track live IC vs. backtest IC. Alert when rolling IC drops below 50% of historical average.

6. **Multiple-testing registry** — Log every signal variant tested. Apply Holm-Bonferroni or Benjamini-Hochberg correction to the full set of tested signals.

---

## Conclusion

This is a well-engineered system with strong infrastructure (data pipelines, risk controls, execution layer, monitoring). The gap is not in engineering — it's in the scientific foundation. The system can execute trades precisely, manage risk carefully, and report results beautifully. It just can't tell you whether the trades should happen in the first place.

Until signal alpha is statistically validated through proper out-of-sample testing with multiple-comparison corrections, the system is — as the `GAP_ANALYSIS.md` itself states — **not ready for live capital deployment**.
