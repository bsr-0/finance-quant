"""Tests for signal alpha significance framework and FDR control."""

import numpy as np
import pandas as pd
import pytest

from pipeline.eval.robustness import benjamini_hochberg
from pipeline.eval.signal_alpha import (
    SignalAlphaResult,
    rank_ic,
    signal_fdr_screen,
    walk_forward_ic,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dates():
    return pd.bdate_range("2020-01-01", periods=600)


@pytest.fixture
def symbols():
    return [f"SYM_{i}" for i in range(20)]


@pytest.fixture
def random_signals(dates, symbols):
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        rng.standard_normal((len(dates), len(symbols))), index=dates, columns=symbols
    )


@pytest.fixture
def random_returns(dates, symbols):
    rng = np.random.default_rng(99)
    return pd.DataFrame(
        rng.standard_normal((len(dates), len(symbols))) * 0.01, index=dates, columns=symbols
    )


@pytest.fixture
def predictive_signals(dates, symbols):
    """Signals that are correlated with forward returns."""
    rng = np.random.default_rng(7)
    base = rng.standard_normal((len(dates), len(symbols)))
    noise = rng.standard_normal((len(dates), len(symbols))) * 0.3
    # Returns are signal + noise → signal predicts returns
    returns = pd.DataFrame(base + noise, index=dates, columns=symbols) * 0.01
    signals = pd.DataFrame(base, index=dates, columns=symbols)
    return signals, returns


# ---------------------------------------------------------------------------
# rank_ic tests
# ---------------------------------------------------------------------------


class TestRankIC:
    def test_perfect_signal(self):
        sig = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        ret = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05])
        ic = rank_ic(sig, ret)
        assert ic == pytest.approx(1.0, abs=0.01)

    def test_inverse_signal(self):
        sig = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0])
        ret = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05])
        ic = rank_ic(sig, ret)
        assert ic == pytest.approx(-1.0, abs=0.01)

    def test_random_signal_near_zero(self):
        rng = np.random.default_rng(42)
        n = 100
        sig = pd.Series(rng.standard_normal(n))
        ret = pd.Series(rng.standard_normal(n))
        ic = rank_ic(sig, ret)
        assert abs(ic) < 0.3  # Random: IC close to zero

    def test_insufficient_data(self):
        sig = pd.Series([1.0, 2.0])
        ret = pd.Series([0.01, 0.02])
        ic = rank_ic(sig, ret)
        assert np.isnan(ic)

    def test_handles_nans(self):
        sig = pd.Series([1.0, np.nan, 3.0, 4.0, 5.0])
        ret = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05])
        ic = rank_ic(sig, ret)
        assert np.isfinite(ic)


# ---------------------------------------------------------------------------
# walk_forward_ic tests
# ---------------------------------------------------------------------------


class TestWalkForwardIC:
    def test_noise_fails_gate(self, random_signals, random_returns):
        result = walk_forward_ic(
            random_signals,
            random_returns,
            signal_name="noise",
            train_size=100,
            test_size=50,
            embargo_size=5,
        )
        assert isinstance(result, SignalAlphaResult)
        assert result.n_folds >= 2
        assert not result.passed  # Pure noise should not pass

    def test_predictive_signal_passes(self, predictive_signals):
        signals, returns = predictive_signals
        result = walk_forward_ic(
            signals,
            returns,
            signal_name="predictive",
            train_size=100,
            test_size=50,
            embargo_size=5,
        )
        assert result.n_folds >= 2
        assert result.ic_mean > 0  # Positive IC expected
        # With strong signal + 20 symbols + 600 dates, should pass
        assert result.passed

    def test_result_fields(self, random_signals, random_returns):
        result = walk_forward_ic(
            random_signals,
            random_returns,
            signal_name="test",
            train_size=100,
            test_size=50,
        )
        assert result.signal_name == "test"
        assert np.isfinite(result.ic_mean)
        assert np.isfinite(result.ic_std)
        assert np.isfinite(result.ic_t_stat)
        assert np.isfinite(result.ic_p_value)
        assert 0 <= result.ic_p_value <= 1
        assert len(result.per_fold_ic) == result.n_folds

    def test_insufficient_data(self, symbols):
        dates = pd.bdate_range("2020-01-01", periods=50)
        rng = np.random.default_rng(42)
        signals = pd.DataFrame(
            rng.standard_normal((50, len(symbols))), index=dates, columns=symbols
        )
        returns = pd.DataFrame(
            rng.standard_normal((50, len(symbols))) * 0.01, index=dates, columns=symbols
        )
        result = walk_forward_ic(
            signals,
            returns,
            signal_name="short",
            train_size=252,
            test_size=63,
        )
        assert result.n_folds == 0
        assert not result.passed


# ---------------------------------------------------------------------------
# Benjamini-Hochberg FDR tests
# ---------------------------------------------------------------------------


class TestBenjaminiHochberg:
    def test_all_significant(self):
        pvals = [0.001, 0.002, 0.003, 0.004]
        rejected = benjamini_hochberg(pvals, alpha=0.05)
        assert all(rejected)

    def test_none_significant(self):
        pvals = [0.5, 0.6, 0.7, 0.8]
        rejected = benjamini_hochberg(pvals, alpha=0.05)
        assert not any(rejected)

    def test_partial_rejection(self):
        # p-values: 0.01, 0.04, 0.06, 0.50
        # Sorted: 0.01(rank1), 0.04(rank2), 0.06(rank3), 0.50(rank4)
        # BH threshold: rank/m * alpha = [0.0125, 0.025, 0.0375, 0.05]
        # 0.01 <= 0.0125 ✓, 0.04 > 0.025 ✗
        pvals = [0.04, 0.01, 0.50, 0.06]
        rejected = benjamini_hochberg(pvals, alpha=0.05)
        assert rejected[1]  # p=0.01 should be rejected
        assert not rejected[2]  # p=0.50 should not

    def test_empty(self):
        assert benjamini_hochberg([], alpha=0.05) == []

    def test_single_significant(self):
        rejected = benjamini_hochberg([0.01], alpha=0.05)
        assert rejected == [True]

    def test_single_not_significant(self):
        rejected = benjamini_hochberg([0.10], alpha=0.05)
        assert rejected == [False]


# ---------------------------------------------------------------------------
# signal_fdr_screen tests
# ---------------------------------------------------------------------------


class TestSignalFDRScreen:
    def _make_result(self, name: str, p_value: float) -> SignalAlphaResult:
        return SignalAlphaResult(
            signal_name=name,
            ic_mean=0.05,
            ic_std=0.02,
            ic_t_stat=2.5,
            ic_p_value=p_value,
            deflated_sharpe_prob=0.9,
            n_folds=5,
            passed=True,
        )

    def test_screen_returns_pairs(self):
        results = [
            self._make_result("sig_a", 0.001),
            self._make_result("sig_b", 0.50),
        ]
        screened = signal_fdr_screen(results, alpha=0.05)
        assert len(screened) == 2
        assert screened[0][1] is True  # sig_a significant
        assert screened[1][1] is False  # sig_b not significant

    def test_nan_pvalue_treated_as_not_significant(self):
        results = [
            self._make_result("sig_a", 0.001),
            self._make_result("sig_nan", np.nan),
        ]
        screened = signal_fdr_screen(results, alpha=0.05)
        assert screened[1][1] is False  # NaN p-value → not significant
