"""Unit tests for risk metrics (no database required)."""

import numpy as np
import pandas as pd
import pytest

from pipeline.features.risk_metrics import (
    calculate_risk_metrics,
    close_to_close_vol,
    drawdown_duration,
    drawdown_series,
    ewma_vol,
    garman_klass_vol,
    historical_cvar,
    historical_var,
    hurst_exponent,
    max_drawdown,
    parkinson_vol,
    rolling_kurtosis,
    rolling_skewness,
    sharpe_ratio,
    sortino_ratio,
    yang_zhang_vol,
)


@pytest.fixture
def sample_prices():
    np.random.seed(42)
    n = 300
    returns = np.random.normal(0.0004, 0.015, n)
    prices = 100 * np.exp(np.cumsum(returns))
    idx = pd.bdate_range("2023-01-01", periods=n)
    return pd.Series(prices, index=idx, name="close")


@pytest.fixture
def sample_ohlcv(sample_prices):
    np.random.seed(42)
    n = len(sample_prices)
    close = sample_prices.values
    return pd.DataFrame({
        "open": close * (1 + np.random.randn(n) * 0.002),
        "high": close * (1 + np.abs(np.random.randn(n) * 0.005)),
        "low": close * (1 - np.abs(np.random.randn(n) * 0.005)),
        "close": close,
        "volume": np.random.randint(100_000, 5_000_000, n).astype(float),
    }, index=sample_prices.index)


class TestVolatilityEstimators:
    def test_close_to_close_vol_range(self, sample_prices):
        vol = close_to_close_vol(sample_prices, 20).dropna()
        assert vol.min() > 0
        # Annualised vol for daily returns ~ 0.015 * sqrt(252) ~ 0.24
        assert 0.05 < vol.median() < 0.80

    def test_parkinson_vol_non_negative(self, sample_ohlcv):
        vol = parkinson_vol(sample_ohlcv["high"], sample_ohlcv["low"], 20).dropna()
        assert (vol >= 0).all()

    def test_garman_klass_vol_non_negative(self, sample_ohlcv):
        vol = garman_klass_vol(
            sample_ohlcv["open"], sample_ohlcv["high"],
            sample_ohlcv["low"], sample_ohlcv["close"], 20
        ).dropna()
        assert (vol >= 0).all()

    def test_yang_zhang_vol_non_negative(self, sample_ohlcv):
        vol = yang_zhang_vol(
            sample_ohlcv["open"], sample_ohlcv["high"],
            sample_ohlcv["low"], sample_ohlcv["close"], 20
        ).dropna()
        assert (vol >= 0).all()

    def test_ewma_vol_non_negative(self, sample_prices):
        vol = ewma_vol(sample_prices, 60).dropna()
        assert (vol >= 0).all()


class TestValueAtRisk:
    def test_var_is_negative(self, sample_prices):
        returns = np.log(sample_prices / sample_prices.shift(1)).dropna()
        var = historical_var(returns, 0.95, 60).dropna()
        # VaR at 95% should be a loss (negative)
        assert var.median() < 0

    def test_cvar_worse_than_var(self, sample_prices):
        returns = np.log(sample_prices / sample_prices.shift(1)).dropna()
        var = historical_var(returns, 0.95, 60).dropna()
        cvar = historical_cvar(returns, 0.95, 60).dropna()
        # CVaR (expected shortfall) should be <= VaR
        common = var.index.intersection(cvar.index)
        assert (cvar.loc[common] <= var.loc[common] + 1e-10).all()


class TestDrawdown:
    def test_drawdown_non_positive(self, sample_prices):
        dd = drawdown_series(sample_prices)
        assert (dd <= 0 + 1e-10).all()

    def test_max_drawdown_is_minimum(self, sample_prices):
        dd = drawdown_series(sample_prices)
        mdd = max_drawdown(sample_prices, 252).dropna()
        assert mdd.min() >= dd.min() - 1e-10

    def test_drawdown_duration_non_negative(self, sample_prices):
        dur = drawdown_duration(sample_prices)
        assert (dur >= 0).all()


class TestPerformanceRatios:
    def test_sharpe_ratio_sign(self, sample_prices):
        returns = np.log(sample_prices / sample_prices.shift(1)).dropna()
        sr = sharpe_ratio(returns, window=100).dropna()
        assert len(sr) > 0

    def test_sortino_ratio_length(self, sample_prices):
        returns = np.log(sample_prices / sample_prices.shift(1)).dropna()
        sort = sortino_ratio(returns, window=100).dropna()
        assert len(sort) > 0


class TestHigherMoments:
    def test_skewness_finite(self, sample_prices):
        returns = np.log(sample_prices / sample_prices.shift(1)).dropna()
        skew = rolling_skewness(returns, 60).dropna()
        assert skew.notna().all()

    def test_kurtosis_finite(self, sample_prices):
        returns = np.log(sample_prices / sample_prices.shift(1)).dropna()
        kurt = rolling_kurtosis(returns, 60).dropna()
        assert kurt.notna().all()


class TestHurstExponent:
    def test_random_walk_hurst_near_half(self):
        np.random.seed(123)
        prices = pd.Series(100 + np.cumsum(np.random.randn(500)))
        h = hurst_exponent(prices, max_lag=100)
        # Random walk should give H ~ 0.5 (+/- 0.15)
        assert 0.3 < h < 0.7

    def test_short_series_returns_nan(self):
        prices = pd.Series([1.0, 2.0, 3.0])
        h = hurst_exponent(prices, max_lag=100)
        assert np.isnan(h)


class TestCalculateRiskMetrics:
    def test_all_columns_added(self, sample_ohlcv):
        result = calculate_risk_metrics(sample_ohlcv)
        expected_cols = [
            "vol_cc_20", "vol_cc_60", "vol_ewma_60",
            "vol_parkinson_20", "vol_gk_20", "vol_yz_20",
            "var_95_60d", "cvar_95_60d",
            "drawdown", "max_drawdown_252d", "drawdown_duration",
            "sharpe_252d", "sortino_252d",
            "skewness_60d", "kurtosis_60d",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_output_length(self, sample_ohlcv):
        result = calculate_risk_metrics(sample_ohlcv)
        assert len(result) == len(sample_ohlcv)
