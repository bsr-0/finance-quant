"""Tests for pipeline.eval.robustness: bootstrap CIs, deflated Sharpe, and
multiple-testing corrections.

No prior coverage existed for this module before the bootstrap CI percentile
fix -- both bootstrap_ci and block_bootstrap_ci used alpha and (1-alpha) as the
percentile cut points directly, which puts alpha in EACH tail rather than
alpha/2, so the default alpha=0.05 produced a 90% interval while every caller
(and the docstring) expects (1-alpha) = 95%.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.eval.robustness import (
    benjamini_hochberg,
    block_bootstrap_ci,
    bootstrap_ci,
    deflated_sharpe_ratio,
    holm_bonferroni,
    probability_of_backtest_overfitting,
)


def _mean(s: pd.Series) -> float:
    return float(s.mean())


# --- bootstrap coverage: the actual bug -----------------------------------


def test_bootstrap_ci_achieves_stated_coverage():
    """A (1-alpha) CI on the sample mean of a known distribution should
    contain the true mean roughly (1-alpha) of the time, not (1-2*alpha) --
    which is what the old alpha-in-each-tail formula produced (alpha=0.10 gave
    an 80% interval, not 90%)."""
    true_mean = 0.0
    rng = np.random.default_rng(0)
    alpha = 0.10
    n_trials = 200
    hits = 0
    for i in range(n_trials):
        sample = pd.Series(rng.normal(true_mean, 1.0, 100))
        lo, hi = bootstrap_ci(sample, _mean, n_boot=200, alpha=alpha, seed=i)
        hits += lo <= true_mean <= hi

    coverage = hits / n_trials
    assert coverage > 0.82, f"coverage {coverage:.2f} suggests the old alpha-in-each-tail bug"


def test_block_bootstrap_ci_achieves_stated_coverage_with_many_blocks():
    """Same coverage check for block_bootstrap_ci, but sized so there are
    enough blocks (~50) that boundary effects don't swamp the percentile-fix
    signal -- with few blocks (e.g. n=100, block_size=21 -> 5 blocks) block
    bootstrap is known to undercover regardless of the percentile formula,
    which isn't what this test is checking."""
    true_mean = 0.0
    rng = np.random.default_rng(0)
    alpha = 0.10
    n_trials = 200
    hits = 0
    for i in range(n_trials):
        sample = pd.Series(rng.normal(true_mean, 1.0, 500))
        lo, hi = block_bootstrap_ci(sample, _mean, block_size=10, n_boot=200, alpha=alpha, seed=i)
        hits += lo <= true_mean <= hi

    coverage = hits / n_trials
    assert coverage > 0.82, f"coverage {coverage:.2f} suggests the old alpha-in-each-tail bug"


def test_block_bootstrap_ci_widens_as_alpha_shrinks():
    """Sanity check independent of nominal coverage: a smaller alpha (higher
    confidence) must always widen the interval."""
    data = pd.Series(np.random.default_rng(4).normal(0, 1, 300))
    lo_90, hi_90 = block_bootstrap_ci(data, _mean, block_size=10, n_boot=300, alpha=0.10, seed=7)
    lo_99, hi_99 = block_bootstrap_ci(data, _mean, block_size=10, n_boot=300, alpha=0.01, seed=7)
    assert (hi_99 - lo_99) > (hi_90 - lo_90)


def test_bootstrap_ci_is_symmetric_around_the_mean_for_symmetric_data():
    data = pd.Series(np.random.default_rng(1).normal(5.0, 2.0, 500))
    lo, hi = bootstrap_ci(data, _mean, n_boot=1000, alpha=0.05, seed=1)
    center = (lo + hi) / 2
    assert center == pytest.approx(data.mean(), abs=0.2)


def test_bootstrap_ci_empty_series_returns_nan():
    lo, hi = bootstrap_ci(pd.Series(dtype=float), _mean)
    assert np.isnan(lo) and np.isnan(hi)


def test_block_bootstrap_falls_back_when_series_shorter_than_block():
    data = pd.Series([1.0, 2.0, 3.0])
    lo, hi = block_bootstrap_ci(data, _mean, block_size=21, n_boot=200, seed=0)
    assert not np.isnan(lo)
    assert lo <= hi


def test_block_bootstrap_preserves_autocorrelation_structure():
    """A block bootstrap on strongly autocorrelated data should give a wider
    CI than an IID bootstrap on the same data, since IID resampling destroys
    the serial dependence that inflates the true sampling variance."""
    rng = np.random.default_rng(2)
    n = 500
    ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = 0.9 * ar[i - 1] + rng.normal(0, 1)
    series = pd.Series(ar)

    iid_lo, iid_hi = bootstrap_ci(series, _mean, n_boot=500, alpha=0.05, seed=3)
    block_lo, block_hi = block_bootstrap_ci(
        series, _mean, block_size=21, n_boot=500, alpha=0.05, seed=3
    )

    assert (block_hi - block_lo) > (iid_hi - iid_lo)


# --- deflated Sharpe --------------------------------------------------------


def test_deflated_sharpe_ratio_high_sharpe_many_obs_is_confident():
    prob = deflated_sharpe_ratio(sharpe=2.0, n_obs=500)
    assert prob > 0.95


def test_deflated_sharpe_ratio_zero_sharpe_is_uncertain():
    prob = deflated_sharpe_ratio(sharpe=0.0, n_obs=100)
    assert prob == pytest.approx(0.5, abs=0.05)


def test_deflated_sharpe_ratio_more_trials_same_sharpe_is_not_more_confident():
    """More observations at the same Sharpe should increase confidence; this
    just pins the monotonicity direction so a future edit can't invert it."""
    low_n = deflated_sharpe_ratio(sharpe=1.0, n_obs=30)
    high_n = deflated_sharpe_ratio(sharpe=1.0, n_obs=300)
    assert high_n > low_n


def test_deflated_sharpe_ratio_degenerate_inputs():
    assert np.isnan(deflated_sharpe_ratio(sharpe=1.0, n_obs=1))
    assert np.isnan(deflated_sharpe_ratio(sharpe=float("nan"), n_obs=100))


# --- multiple testing corrections ------------------------------------------


def test_benjamini_hochberg_rejects_strong_signals_among_noise():
    pvals = [0.001, 0.002, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    rejected = benjamini_hochberg(pvals, alpha=0.05)
    assert rejected[0] and rejected[1]
    assert not any(rejected[2:])


def test_benjamini_hochberg_more_lenient_than_holm_bonferroni():
    """BH controls FDR (more power); Holm-Bonferroni controls FWER (more
    conservative). BH should never reject fewer hypotheses than Holm here."""
    pvals = [0.001, 0.01, 0.02, 0.03, 0.04, 0.2, 0.3, 0.5]
    bh_rejected = sum(benjamini_hochberg(pvals, alpha=0.05))
    holm_rejected = sum(p <= 0.05 for p in holm_bonferroni(pvals))
    assert bh_rejected >= holm_rejected


def test_holm_bonferroni_empty_and_single():
    assert holm_bonferroni([]) == []
    assert holm_bonferroni([0.03]) == [0.03]


def test_benjamini_hochberg_empty():
    assert benjamini_hochberg([]) == []


# --- probability of backtest overfitting -----------------------------------


def test_pbo_high_when_in_sample_winner_is_out_of_sample_loser():
    idx = pd.RangeIndex(20)
    train = pd.Series(np.arange(20), index=idx)  # top decile = last 2 configs
    # Those same configs are the worst performers out-of-sample.
    test = pd.Series(np.arange(20)[::-1], index=idx)
    pbo = probability_of_backtest_overfitting(train, test)
    assert pbo > 0.5


def test_pbo_low_when_in_sample_and_out_of_sample_agree():
    idx = pd.RangeIndex(20)
    train = pd.Series(np.arange(20), index=idx)
    test = pd.Series(np.arange(20), index=idx)
    pbo = probability_of_backtest_overfitting(train, test)
    assert pbo < 0.5
