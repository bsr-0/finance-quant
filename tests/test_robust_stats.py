"""Unit tests for robust statistics (no database required)."""

import numpy as np
import pandas as pd
import pytest

from pipeline.features.robust_stats import (
    clean_returns,
    detect_outliers_iqr,
    detect_outliers_mad,
    detect_outliers_zscore,
    iqr,
    ledoit_wolf_shrinkage,
    mad,
    mad_zscore,
    robust_mean,
    robust_std,
    rolling_mad,
    rolling_winsorize,
    winsorize,
)


@pytest.fixture
def normal_series():
    np.random.seed(42)
    return pd.Series(np.random.randn(200))


@pytest.fixture
def series_with_outliers():
    np.random.seed(42)
    data = np.random.randn(200)
    data[50] = 50.0  # extreme positive outlier
    data[100] = -40.0  # extreme negative outlier
    return pd.Series(data)


@pytest.fixture
def return_df():
    np.random.seed(42)
    n = 200
    return pd.DataFrame(
        {
            "A": np.random.randn(n) * 0.01,
            "B": np.random.randn(n) * 0.015,
            "C": np.random.randn(n) * 0.02,
        }
    )


class TestWinsorize:
    def test_clips_extremes(self, series_with_outliers):
        w = winsorize(series_with_outliers, 0.01, 0.99)
        assert w.max() < 50.0
        assert w.min() > -40.0

    def test_preserves_normal_values(self, normal_series):
        w = winsorize(normal_series, 0.05, 0.95)
        # Most values should be unchanged
        unchanged = (w == normal_series).sum()
        assert unchanged > len(normal_series) * 0.85

    def test_rolling_winsorize(self, series_with_outliers):
        w = rolling_winsorize(series_with_outliers, window=50)
        # Outlier at index 50 should be clipped
        assert w.iloc[50] < 50.0


class TestMAD:
    def test_mad_positive(self, normal_series):
        m = mad(normal_series)
        assert m > 0

    def test_mad_of_constant_is_zero(self):
        s = pd.Series([5.0] * 100)
        assert mad(s) == 0.0

    def test_rolling_mad_length(self, normal_series):
        rm = rolling_mad(normal_series, 30)
        assert len(rm) == len(normal_series)

    def test_mad_zscore_detects_outlier(self, series_with_outliers):
        z = mad_zscore(series_with_outliers, 40)
        # The extreme outlier at index 50 should have a very high score
        assert abs(z.iloc[50]) > 5


class TestRobustRolling:
    def test_robust_mean_close_to_mean(self, normal_series):
        rm = robust_mean(normal_series, 60, 0.05).dropna()
        standard_mean = normal_series.rolling(60).mean().dropna()
        # For normal data, trimmed mean should be close to regular mean
        diff = (rm - standard_mean).abs().median()
        assert diff < 0.1

    def test_robust_std_positive(self, normal_series):
        rs = robust_std(normal_series, 60).dropna()
        assert (rs > 0).all()

    def test_iqr_positive(self, normal_series):
        i = iqr(normal_series, 60).dropna()
        assert (i > 0).all()


class TestOutlierDetection:
    def test_zscore_finds_outliers(self, series_with_outliers):
        mask = detect_outliers_zscore(series_with_outliers, 40, 3.0)
        assert mask.iloc[50] == True  # noqa: E712
        assert mask.iloc[100] == True  # noqa: E712

    def test_mad_finds_outliers(self, series_with_outliers):
        mask = detect_outliers_mad(series_with_outliers, 40, 3.0)
        assert mask.iloc[50] == True  # noqa: E712

    def test_iqr_finds_outliers(self, series_with_outliers):
        mask = detect_outliers_iqr(series_with_outliers, 40, 1.5)
        assert mask.iloc[50] == True  # noqa: E712

    def test_no_false_positives_on_clean_data(self, normal_series):
        mask = detect_outliers_mad(normal_series, 60, 5.0)
        # Very few false positives expected at 5-sigma
        assert mask.sum() < len(normal_series) * 0.05


class TestLedoitWolfShrinkage:
    def test_returns_positive_definite(self, return_df):
        cov, intensity = ledoit_wolf_shrinkage(return_df)
        eigenvalues = np.linalg.eigvals(cov.values)
        assert (eigenvalues > -1e-10).all()

    def test_shrinkage_between_0_and_1(self, return_df):
        _, intensity = ledoit_wolf_shrinkage(return_df)
        assert 0 <= intensity <= 1

    def test_symmetric(self, return_df):
        cov, _ = ledoit_wolf_shrinkage(return_df)
        np.testing.assert_allclose(cov.values, cov.values.T, atol=1e-10)

    def test_shape(self, return_df):
        cov, _ = ledoit_wolf_shrinkage(return_df)
        assert cov.shape == (3, 3)
        assert list(cov.columns) == ["A", "B", "C"]


class TestCleanReturns:
    def test_outliers_become_nan(self, series_with_outliers):
        cleaned = clean_returns(series_with_outliers, window=40, outlier_threshold=3.0)
        # At least the known outliers should be NaN
        assert pd.isna(cleaned.iloc[50])

    def test_most_values_preserved(self, normal_series):
        cleaned = clean_returns(normal_series, window=40, outlier_threshold=5.0)
        non_null = cleaned.notna().sum()
        assert non_null > len(normal_series) * 0.90
