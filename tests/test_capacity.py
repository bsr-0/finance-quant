"""Unit tests for capacity analysis and sensitivity analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.backtesting.capacity import (
    CapacityResult,
    SensitivityResult,
    capacity_analysis,
    multi_param_sensitivity,
    sensitivity_analysis,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_returns():
    """Synthetic daily return series with positive expected value."""
    np.random.seed(42)
    n = 504  # ~2 years
    returns = np.random.normal(0.0006, 0.012, n)  # ~15 % annual, 19 % vol
    idx = pd.bdate_range("2022-01-01", periods=n)
    return pd.Series(returns, index=idx)


@pytest.fixture
def sample_df(sample_returns):
    """Feature DataFrame for sensitivity tests."""
    np.random.seed(42)
    n = len(sample_returns)
    return pd.DataFrame(
        {
            "feature_a": np.random.randn(n),
            "target": sample_returns.values,
        },
        index=sample_returns.index,
    )


# ---------------------------------------------------------------------------
# Capacity Analysis Tests
# ---------------------------------------------------------------------------

class TestCapacityAnalysis:
    def test_returns_capacity_result(self, sample_returns):
        result = capacity_analysis(
            returns=sample_returns,
            trades_per_year=50,
            avg_price=100.0,
            adv=1_000_000,
        )
        assert isinstance(result, CapacityResult)

    def test_capital_levels_match(self, sample_returns):
        levels = [100_000, 500_000, 1_000_000]
        result = capacity_analysis(
            returns=sample_returns,
            trades_per_year=50,
            avg_price=100.0,
            adv=1_000_000,
            capital_levels=levels,
        )
        assert result.capital_levels == levels
        assert len(result.net_sharpes) == 3
        assert len(result.cost_drags) == 3

    def test_cost_drag_increases_with_capital(self, sample_returns):
        levels = [10_000, 100_000, 1_000_000, 10_000_000]
        result = capacity_analysis(
            returns=sample_returns,
            trades_per_year=100,
            avg_price=50.0,
            adv=500_000,
            capital_levels=levels,
        )
        # Cost drag should be non-decreasing with capital (more $ → more impact)
        for i in range(1, len(result.cost_drags)):
            assert result.cost_drags[i] >= result.cost_drags[i - 1] - 1e-10

    def test_net_sharpe_decreases_with_capital(self, sample_returns):
        levels = [10_000, 100_000, 1_000_000, 10_000_000]
        result = capacity_analysis(
            returns=sample_returns,
            trades_per_year=100,
            avg_price=50.0,
            adv=500_000,
            capital_levels=levels,
        )
        # Net Sharpe should be non-increasing with capital
        for i in range(1, len(result.net_sharpes)):
            assert result.net_sharpes[i] <= result.net_sharpes[i - 1] + 1e-10

    def test_capacity_estimate_positive(self, sample_returns):
        result = capacity_analysis(
            returns=sample_returns,
            trades_per_year=20,
            avg_price=100.0,
            adv=10_000_000,
            min_sharpe=0.5,
        )
        # With high ADV and few trades, capacity should be positive
        assert result.capacity_estimate > 0

    def test_capacity_estimate_zero_for_bad_strategy(self):
        # Returns that are deeply negative → capacity = 0
        np.random.seed(0)
        bad_returns = pd.Series(np.random.normal(-0.005, 0.02, 252))
        result = capacity_analysis(
            returns=bad_returns,
            trades_per_year=100,
            avg_price=50.0,
            adv=100_000,
            min_sharpe=0.5,
        )
        assert result.capacity_estimate == 0.0

    def test_summary_dataframe_shape(self, sample_returns):
        levels = [1e5, 5e5, 1e6]
        result = capacity_analysis(
            returns=sample_returns,
            trades_per_year=50,
            avg_price=100.0,
            adv=1_000_000,
            capital_levels=levels,
        )
        df = result.summary()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "capital" in df.columns
        assert "net_sharpe" in df.columns

    def test_raises_on_empty_returns(self):
        with pytest.raises(ValueError, match="empty"):
            capacity_analysis(
                returns=pd.Series(dtype=float),
                trades_per_year=50,
                avg_price=100.0,
                adv=1_000_000,
            )

    def test_zero_adv_no_crash(self, sample_returns):
        result = capacity_analysis(
            returns=sample_returns,
            trades_per_year=50,
            avg_price=100.0,
            adv=0,
            capital_levels=[1_000_000],
        )
        # Impact is zero when ADV = 0 (no market depth assumption)
        assert result.cost_drags[0] >= 0


# ---------------------------------------------------------------------------
# Sensitivity Analysis Tests
# ---------------------------------------------------------------------------

class TestSensitivityAnalysis:
    def test_returns_sensitivity_result(self, sample_df):
        def sharpe_fn(df: pd.DataFrame, window: float) -> float:
            w = int(window)
            returns = df["target"]
            mu = returns.rolling(w, min_periods=5).mean().dropna().mean()
            sigma = returns.rolling(w, min_periods=5).std().dropna().mean()
            return float(mu / sigma * np.sqrt(252)) if sigma > 0 else np.nan

        result = sensitivity_analysis(
            sample_df, "window", [20, 40, 60, 80], sharpe_fn, "sharpe"
        )
        assert isinstance(result, SensitivityResult)
        assert result.param_name == "window"
        assert len(result.metric_values) == 4

    def test_is_robust_positive_metrics(self, sample_df):
        # All-positive metrics should be robust
        result = SensitivityResult(
            param_name="p",
            param_values=[1, 2, 3],
            metric_values=[0.5, 0.7, 0.6],
            metric_name="sharpe",
            baseline_value=0.6,
        )
        assert result.is_robust is True

    def test_is_not_robust_with_negatives(self):
        result = SensitivityResult(
            param_name="p",
            param_values=[1, 2, 3],
            metric_values=[0.5, -0.2, 0.4],
            metric_name="sharpe",
            baseline_value=0.5,
        )
        assert result.is_robust is False

    def test_variation_coefficient(self):
        result = SensitivityResult(
            param_name="p",
            param_values=[1, 2, 3],
            metric_values=[1.0, 1.0, 1.0],
            metric_name="metric",
            baseline_value=1.0,
        )
        # Constant values → zero CV
        assert result.variation_coefficient == pytest.approx(0.0)

    def test_summary_dataframe(self, sample_df):
        result = sensitivity_analysis(
            sample_df,
            "threshold",
            [0.01, 0.02, 0.03],
            lambda df, v: float(df["target"].mean() / df["target"].std()),
            "ir",
        )
        df = result.summary()
        assert isinstance(df, pd.DataFrame)
        assert "threshold" in df.columns
        assert "ir" in df.columns
        assert len(df) == 3

    def test_handles_function_errors_gracefully(self, sample_df):
        def bad_fn(df: pd.DataFrame, v: float) -> float:
            if v == 2.0:
                raise RuntimeError("deliberate error")
            return 1.0

        result = sensitivity_analysis(sample_df, "p", [1.0, 2.0, 3.0], bad_fn)
        assert np.isnan(result.metric_values[1])
        assert result.metric_values[0] == pytest.approx(1.0)

    def test_baseline_uses_midpoint_by_default(self, sample_df):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = sensitivity_analysis(
            sample_df,
            "p",
            values,
            lambda df, v: v,  # metric equals param value
        )
        # Midpoint index = 2, so baseline = 3.0
        assert result.baseline_value == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Multi-Parameter Sensitivity Tests
# ---------------------------------------------------------------------------

class TestMultiParamSensitivity:
    def test_full_grid_evaluated(self, sample_df):
        grid = {"window": [10, 20], "threshold": [0.01, 0.02]}
        result = multi_param_sensitivity(
            sample_df,
            grid,
            lambda df, params: params["window"] * params["threshold"],
            metric_name="score",
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4  # 2 × 2 grid
        assert "window" in result.columns
        assert "threshold" in result.columns
        assert "score" in result.columns

    def test_error_becomes_nan(self, sample_df):
        def bad_fn(df: pd.DataFrame, params: dict) -> float:
            if params["p"] == 2:
                raise ValueError("error")
            return 1.0

        result = multi_param_sensitivity(sample_df, {"p": [1, 2, 3]}, bad_fn)
        nan_rows = result[result["metric"].isna()]
        assert len(nan_rows) == 1
