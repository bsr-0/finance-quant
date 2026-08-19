"""Validation harness for the QSG-MICRO-SWING-001 signal score.

Orchestrates existing machinery (nothing here reimplements a statistic) to
answer the question the strategy has never had answered: does the composite
score have any monotone relationship with forward returns, on real 2010-2026
data rather than the 26 distinct dates the live-tracked history offers?

Every variant tested -- the aggregate score, its four component buckets, and
each of its ten underlying boolean conditions -- is registered in a
``SignalTrialRegistry`` and screened together via Benjamini-Hochberg. That
registry is what keeps this from being the curve-fitting exercise a validation
harness is supposed to rule out: nothing gets excluded after the fact.

See ``docs`` / the signal-validation plan for the full five-diagnostic design
(D1-D5). D5 (the same-bar stop/target resolution bracket) applies to
trade-level barrier outcomes, not to this panel's continuous forward-return IC
test -- that bracket is already implemented at the execution layer in
``pipeline.web.outcome_resolution.ResolutionPolicy``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pipeline.eval.robustness import block_bootstrap_ci, probability_of_backtest_overfitting
from pipeline.eval.signal_alpha import (
    ICDecayResult,
    SignalAlphaResult,
    SignalTrialRegistry,
    compute_forward_returns,
    ic_decay_analysis,
    rolling_ic,
    walk_forward_ic,
)
from pipeline.strategy.signal_panel import ScorePanel
from pipeline.strategy.signals import SignalEngine, compute_indicators

logger = logging.getLogger(__name__)

DEFAULT_HORIZONS = [1, 5, 10, 21, 63]
COLLINEARITY_OVERSOLD_CONDITIONS = (
    "rsi_oversold",
    "below_bb_lower",
    "stoch_oversold",
    "williams_oversold",
)


@dataclass
class MonotonicityReport:
    """D1: does score rank predict forward-return rank at all?"""

    decay: ICDecayResult
    daily_ic_ci: tuple[float, float]
    """Block-bootstrap 95% CI on the daily IC series at the best horizon."""
    decile_returns: pd.DataFrame
    """Mean forward return per score decile, at the best horizon."""


@dataclass
class ComponentDecomposition:
    """D2: which of the 10 underlying boolean conditions (if any) carry the
    score's information, and how redundant are they with each other?"""

    trials: list[SignalAlphaResult]
    screened: list[tuple[SignalAlphaResult, bool]]
    """(result, significant) after BH correction across every trial in this
    decomposition -- the 10 conditions, 4 bucket totals, and the score."""
    collinearity: pd.DataFrame
    """Phi-coefficient matrix among the four oversold conditions."""
    first_pc_variance_ratio: float
    """Share of variance the first principal component explains across the
    four oversold conditions. > 0.70 means they're one construct counted
    roughly four times, not four independent confirmations."""


@dataclass
class SignalValidationReport:
    score_result: SignalAlphaResult
    monotonicity: MonotonicityReport
    decomposition: ComponentDecomposition
    pbo: float
    """Probability of backtest overfitting between in-sample and OOS folds."""
    verdict: str
    """PASS | INVERTED | INCONCLUSIVE -- see docstring on run_validation."""
    reasoning: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# D1: monotonicity
# ---------------------------------------------------------------------------


def _decile_returns(signal: pd.DataFrame, forward_returns: pd.DataFrame) -> pd.DataFrame:
    """Mean forward return by score decile, pooled across all dates."""
    sig_long = signal.stack()
    ret_long = forward_returns.stack()
    aligned = pd.DataFrame({"score": sig_long, "ret": ret_long}).dropna()
    if aligned.empty:
        return pd.DataFrame(columns=["decile", "mean_return", "n"])

    try:
        aligned["decile"] = pd.qcut(aligned["score"], 10, labels=False, duplicates="drop")
    except ValueError:
        aligned["decile"] = pd.cut(aligned["score"], 10, labels=False)

    out = aligned.groupby("decile")["ret"].agg(mean_return="mean", n="count").reset_index()
    return out


def analyze_monotonicity(
    panel: ScorePanel,
    price_panel: pd.DataFrame,
    horizons: list[int] | None = None,
    signal_name: str = "score",
) -> MonotonicityReport:
    """D1: score-decile bucket returns plus a block-bootstrap CI on daily IC."""
    horizons = horizons or DEFAULT_HORIZONS
    decay = ic_decay_analysis(panel.score, price_panel, signal_name=signal_name, horizons=horizons)

    best_h = decay.best_horizon
    fwd = compute_forward_returns(price_panel, horizon=best_h)

    # window=1 disables rolling_ic's smoothing, yielding the raw per-date IC
    # series -- the thing a block bootstrap needs, not a pre-averaged one.
    daily_ic = rolling_ic(panel.score, fwd, window=1).dropna()
    block = max(21, 2 * best_h)
    ci = block_bootstrap_ci(daily_ic, lambda s: float(s.mean()), block_size=block)

    deciles = _decile_returns(panel.score, fwd)

    return MonotonicityReport(decay=decay, daily_ic_ci=ci, decile_returns=deciles)


# ---------------------------------------------------------------------------
# D2: component decomposition
# ---------------------------------------------------------------------------


def build_component_signal_panels(
    price_frames: dict[str, pd.DataFrame],
    engine: SignalEngine,
) -> dict[str, pd.DataFrame]:
    """Wide date x ticker panels for each of the score's 10 boolean conditions
    plus its 4 bucket totals -- everything ``score_frame`` sums into the
    aggregate score, tested individually rather than only in aggregate."""
    per_ticker_conditions: dict[str, dict[str, pd.Series]] = {}
    per_ticker_buckets: dict[str, pd.DataFrame] = {}

    for ticker, df in price_frames.items():
        if df.empty or len(df) < 5:
            continue
        try:
            indicators = compute_indicators(df)
            conditions = engine.component_conditions(indicators)
            per_ticker_conditions[ticker] = {
                name: mask.astype(int) for name, (mask, _points, _bucket) in conditions.items()
            }
            per_ticker_buckets[ticker] = engine.score_frame(indicators)
        except Exception:
            logger.exception("Failed to build component signals for %s", ticker)

    if not per_ticker_conditions:
        return {}

    condition_names = next(iter(per_ticker_conditions.values())).keys()
    panels: dict[str, pd.DataFrame] = {
        name: pd.DataFrame({t: c[name] for t, c in per_ticker_conditions.items()}).sort_index()
        for name in condition_names
    }
    for field_name in ("trend_pts", "pullback_pts", "volume_pts", "volatility_pts", "score"):
        panels[field_name] = pd.DataFrame(
            {t: b[field_name] for t, b in per_ticker_buckets.items()}
        ).sort_index()

    return panels


def _phi_coefficient(a: pd.Series, b: pd.Series) -> float:
    """Matthews correlation coefficient for two binary series (the phi
    coefficient is the same statistic under a different name)."""
    aligned = pd.DataFrame({"a": a, "b": b}).dropna()
    if aligned.empty or aligned["a"].nunique() < 2 or aligned["b"].nunique() < 2:
        return np.nan
    return float(np.corrcoef(aligned["a"], aligned["b"])[0, 1])


def analyze_component_decomposition(
    panel_close: pd.DataFrame,
    component_panels: dict[str, pd.DataFrame],
    train_size: int = 504,
    test_size: int = 63,
    embargo_size: int = 15,
    label_horizon: int = 15,
    alpha: float = 0.05,
) -> ComponentDecomposition:
    """D2: test each condition, bucket total, and the score itself as its own
    trial, then BH-screen the whole set together."""
    fwd = compute_forward_returns(panel_close, horizon=label_horizon)

    registry = SignalTrialRegistry()
    for name, panel in component_panels.items():
        result = walk_forward_ic(
            panel,
            fwd,
            signal_name=name,
            train_size=train_size,
            test_size=test_size,
            embargo_size=embargo_size,
            label_horizon=label_horizon,
        )
        registry.record_trial(result)

    screened = registry.screen(alpha=alpha)

    oversold = [c for c in COLLINEARITY_OVERSOLD_CONDITIONS if c in component_panels]
    collinearity = pd.DataFrame(index=oversold, columns=oversold, dtype=float)
    stacked = {name: component_panels[name].stack() for name in oversold}
    for a in oversold:
        for b in oversold:
            collinearity.loc[a, b] = _phi_coefficient(stacked[a], stacked[b])

    first_pc_ratio = np.nan
    if len(oversold) >= 2:
        combined = pd.DataFrame(stacked).dropna()
        if len(combined) > len(oversold) and combined.nunique().min() > 1:
            cov = np.cov(combined.T.to_numpy())
            eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
            total = eigvals.sum()
            first_pc_ratio = float(eigvals[0] / total) if total > 0 else np.nan

    return ComponentDecomposition(
        trials=registry.trials,
        screened=screened,
        collinearity=collinearity,
        first_pc_variance_ratio=first_pc_ratio,
    )


# ---------------------------------------------------------------------------
# Orchestration and the G3 decision
# ---------------------------------------------------------------------------


def run_validation(
    panel: ScorePanel,
    price_frames: dict[str, pd.DataFrame],
    engine: SignalEngine | None = None,
    train_size: int = 504,
    test_size: int = 63,
    embargo_size: int = 15,
    significance_threshold: float = 0.95,
    alpha: float = 0.05,
) -> SignalValidationReport:
    """Run the full validation harness and render the G3 verdict.

    Verdicts:
      PASS         -- score_result.passed, ic_mean > 0, survives BH across the
                       full trial registry, and the daily-IC bootstrap CI
                       excludes 0. The signal works; the live strategy's
                       problem is execution (R:R), not the score.
      INVERTED      -- same criteria with ic_mean < 0. Genuine negative alpha;
                       flip the signal's sign and re-validate as a NEW trial
                       before shipping, per the trial-registry discipline.
      INCONCLUSIVE  -- anything else, which is the expected outcome given the
                       score's construction was never validated. Do not tune
                       _score_row on the strength of this alone.
    """
    engine = engine or SignalEngine()
    price_panel = pd.DataFrame({t: df["close"] for t, df in price_frames.items() if not df.empty})

    monotonicity = analyze_monotonicity(panel, price_panel)
    best_h = monotonicity.decay.best_horizon

    score_result = walk_forward_ic(
        panel.score,
        compute_forward_returns(price_panel, horizon=best_h),
        signal_name="score",
        train_size=train_size,
        test_size=test_size,
        embargo_size=embargo_size,
        significance_threshold=significance_threshold,
        label_horizon=best_h,
    )

    component_panels = build_component_signal_panels(price_frames, engine)
    decomposition = analyze_component_decomposition(
        price_panel,
        component_panels,
        train_size=train_size,
        test_size=test_size,
        embargo_size=embargo_size,
        label_horizon=best_h,
        alpha=alpha,
    )

    # PBO needs a (train_score, test_score) pair per trial; approximate with
    # each component trial's per-fold IC split at the midpoint as a train/test
    # proxy, since the strategy has no separate hyperparameter sweep to draw
    # train/test scores from.
    pbo = np.nan
    if decomposition.trials:
        mid = max(1, len(decomposition.trials) // 2)
        train_scores = pd.Series(
            [t.ic_mean for t in decomposition.trials[:mid] if np.isfinite(t.ic_mean)]
        )
        test_scores = pd.Series(
            [t.ic_mean for t in decomposition.trials[mid:] if np.isfinite(t.ic_mean)]
        )
        if len(train_scores) and len(test_scores):
            train_scores = train_scores.reset_index(drop=True)
            test_scores = test_scores.reset_index(drop=True)
            pbo = probability_of_backtest_overfitting(train_scores, test_scores)

    score_screened = next(
        (sig for r, sig in decomposition.screened if r.signal_name == "score"), False
    )
    ci_lo, ci_hi = monotonicity.daily_ic_ci
    ci_excludes_zero = np.isfinite(ci_lo) and np.isfinite(ci_hi) and (ci_lo > 0 or ci_hi < 0)

    reasoning: list[str] = []
    verdict = "INCONCLUSIVE"

    if not score_result.passed:
        reasoning.append(
            f"score walk-forward IC did not pass the deflated-Sharpe gate "
            f"(DSR={score_result.deflated_sharpe_prob:.3f} <= {significance_threshold})"
        )
    if not score_screened:
        reasoning.append("score did not survive BH correction across the full trial registry")
    if not ci_excludes_zero:
        reasoning.append(f"daily-IC bootstrap CI [{ci_lo:.4f}, {ci_hi:.4f}] does not exclude zero")

    if score_result.passed and score_screened and ci_excludes_zero:
        verdict = "PASS" if score_result.ic_mean > 0 else "INVERTED"
        reasoning = [
            f"score_result.passed=True, ic_mean={score_result.ic_mean:.4f}, "
            f"survived BH, CI excludes zero"
        ]

    return SignalValidationReport(
        score_result=score_result,
        monotonicity=monotonicity,
        decomposition=decomposition,
        pbo=pbo,
        verdict=verdict,
        reasoning=reasoning,
    )
