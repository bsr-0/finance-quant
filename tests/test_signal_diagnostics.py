"""Tests for the signal validation harness (pipeline.eval.signal_diagnostics).

The critical property of a validation harness is that it doesn't just always
say yes: it must find a signal planted with real predictive power AND fail to
find one in pure noise. Both are tested directly, since a harness that passes
everything is worse than no harness.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.eval.signal_diagnostics import (
    analyze_component_decomposition,
    analyze_monotonicity,
    build_component_signal_panels,
    run_validation,
)
from pipeline.strategy.signal_panel import build_score_panel
from pipeline.strategy.signals import SignalEngine


def _make_price_frame(seed: int, n: int, drift: float = 0.0002) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2005-01-01", periods=n)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(drift, 0.015, n))), index=idx)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(100_000, 900_000, n).astype(float),
        }
    )


def _planted_signal_panel(n: int, n_symbols: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A synthetic (signal, forward_returns) pair with a genuine, strong
    monotone relationship: signal[t] literally causes return[t+1]."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2005-01-01", periods=n)
    symbols = [f"S{i}" for i in range(n_symbols)]

    signal = pd.DataFrame(rng.uniform(0, 100, (n, n_symbols)), index=idx, columns=symbols)
    noise = rng.normal(0, 0.02, (n, n_symbols))
    # Forward return strongly increasing in the current signal value.
    fwd_return = 0.001 * (signal.to_numpy() - 50) + noise
    forward_returns = pd.DataFrame(fwd_return, index=idx, columns=symbols)
    return signal, forward_returns


def _null_signal_panel(n: int, n_symbols: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2005-01-01", periods=n)
    symbols = [f"S{i}" for i in range(n_symbols)]
    signal = pd.DataFrame(rng.uniform(0, 100, (n, n_symbols)), index=idx, columns=symbols)
    forward_returns = pd.DataFrame(rng.normal(0, 0.02, (n, n_symbols)), index=idx, columns=symbols)
    return signal, forward_returns


# ---------------------------------------------------------------------------
# D1: monotonicity
# ---------------------------------------------------------------------------


def test_analyze_monotonicity_detects_a_planted_signal():
    signal, fwd = _planted_signal_panel(n=800, n_symbols=15, seed=1)

    class _Panel:
        score = signal

    report = analyze_monotonicity(_Panel(), _prices_from_returns(fwd), horizons=[1, 5])

    assert report.decay.ic_by_horizon[report.decay.best_horizon] > 0.1
    lo, hi = report.daily_ic_ci
    assert lo > 0, "CI on a strongly planted signal should exclude zero"


def test_analyze_monotonicity_finds_nothing_in_pure_noise():
    """A 95% CI has a genuine ~5% chance of excluding zero on pure noise by
    construction, so this checks the rate across several seeds rather than
    asserting on one -- a single unlucky seed failing here would be the CI
    behaving correctly, not a harness bug."""
    excludes_zero = 0
    n_seeds = 8
    for seed in range(n_seeds):
        signal, fwd = _null_signal_panel(n=500, n_symbols=15, seed=seed)

        class _Panel:
            score = signal

        report = analyze_monotonicity(_Panel(), _prices_from_returns(fwd), horizons=[1, 5])
        lo, hi = report.daily_ic_ci
        if not (lo <= 0 <= hi):
            excludes_zero += 1

    assert excludes_zero <= 2, (
        f"{excludes_zero}/{n_seeds} null trials falsely excluded zero -- "
        "expected roughly 1/20 at a 95% CI, not this many"
    )


def test_decile_returns_are_monotone_for_planted_signal():
    signal, fwd = _planted_signal_panel(n=500, n_symbols=15, seed=3)

    class _Panel:
        score = signal

    report = analyze_monotonicity(_Panel(), _prices_from_returns(fwd), horizons=[1])
    deciles = report.decile_returns.sort_values("decile")
    # Top decile should clearly beat bottom decile given the planted relationship.
    assert deciles.iloc[-1]["mean_return"] > deciles.iloc[0]["mean_return"]


def _prices_from_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Build a price panel such that compute_forward_returns(horizon=1)
    recovers *returns* exactly: compute_forward_returns computes
    pct_change(1).shift(-1), so day t's price move must equal returns[t-1]
    (a one-day shift) for that composition to land back on returns[t]."""
    shifted = returns.shift(1).fillna(0)
    return (1 + shifted).cumprod() * 100


# ---------------------------------------------------------------------------
# D2: component decomposition
# ---------------------------------------------------------------------------


def test_build_component_signal_panels_shape():
    frames = {"A": _make_price_frame(1, 300), "B": _make_price_frame(2, 300)}
    engine = SignalEngine()
    panels = build_component_signal_panels(frames, engine)

    expected_keys = {
        "trend_stacked", "above_sma_200", "sma_50_rising", "rsi_oversold",
        "below_bb_lower", "stoch_oversold", "volume_below_sma", "obv_rising",
        "atr_pct_in_range", "macd_rising", "williams_oversold",
        "trend_pts", "pullback_pts", "volume_pts", "volatility_pts", "score",
    }  # fmt: skip
    assert expected_keys.issubset(panels.keys())
    for name, panel in panels.items():
        assert sorted(panel.columns) == ["A", "B"], name


def test_component_decomposition_registers_every_condition_as_a_trial():
    frames = {f"S{i}": _make_price_frame(i, 700) for i in range(6)}
    engine = SignalEngine()
    panels = build_component_signal_panels(frames, engine)
    price_panel = pd.DataFrame({t: df["close"] for t, df in frames.items()})

    decomposition = analyze_component_decomposition(
        price_panel, panels, train_size=252, test_size=63, embargo_size=5, label_horizon=5
    )

    assert len(decomposition.trials) == len(panels)
    assert len(decomposition.screened) == len(panels)


def test_collinearity_matrix_flags_near_duplicate_conditions():
    """Two conditions defined to fire together almost always should show a
    high phi coefficient and dominate the first principal component."""
    from pipeline.eval.signal_diagnostics import COLLINEARITY_OVERSOLD_CONDITIONS, _phi_coefficient

    idx = pd.bdate_range("2020-01-01", periods=500)
    rng = np.random.default_rng(4)
    base = rng.random(500) < 0.3
    almost_same = base.copy()
    flip = rng.random(500) < 0.02
    almost_same[flip] = ~almost_same[flip]

    phi = _phi_coefficient(
        pd.Series(base.astype(int), index=idx), pd.Series(almost_same.astype(int), index=idx)
    )
    assert phi > 0.85
    assert len(COLLINEARITY_OVERSOLD_CONDITIONS) == 4


# ---------------------------------------------------------------------------
# Full orchestration: the harness itself must discriminate signal from noise
# ---------------------------------------------------------------------------


def test_run_validation_end_to_end_on_synthetic_universe():
    """Smoke test the full orchestration path (monotonicity + component
    decomposition + PBO + verdict) against a small synthetic universe, with
    walk-forward windows small enough to actually produce folds."""
    frames = {f"S{i}": _make_price_frame(100 + i, 900) for i in range(8)}
    spy = _make_price_frame(999, 900)["close"]
    engine = SignalEngine()

    panel = build_score_panel(frames, engine, spy_prices=spy)
    report = run_validation(panel, frames, engine, train_size=252, test_size=63, embargo_size=5)

    assert report.verdict in ("PASS", "INVERTED", "INCONCLUSIVE")
    assert report.reasoning
    assert len(report.decomposition.trials) > 0
    assert report.score_result.n_folds >= 0


def test_run_validation_does_not_pass_on_pure_noise_score():
    """The strongest guard against a harness that passes everything: feed it
    a score with no relationship to price and confirm it does not say PASS."""
    frames = {f"S{i}": _make_price_frame(200 + i, 900) for i in range(8)}
    spy = _make_price_frame(998, 900)["close"]
    engine = SignalEngine()

    panel = build_score_panel(frames, engine, spy_prices=spy)
    # Overwrite the real score with pure noise, independent of price.
    rng = np.random.default_rng(0)
    panel.score = pd.DataFrame(
        rng.uniform(0, 100, panel.score.shape), index=panel.score.index, columns=panel.score.columns
    )

    report = run_validation(panel, frames, engine, train_size=252, test_size=63, embargo_size=5)
    assert report.verdict != "PASS"
