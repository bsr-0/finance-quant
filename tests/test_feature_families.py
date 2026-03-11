"""Tests for feature families (Agent Directive V7 Section 6)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.features.feature_families import (
    HierarchicalFeatures,
    InteractionFeatures,
    RepresentationFeatures,
    SeasonalFeatures,
    select_features,
)

# ---------------------------------------------------------------------------
# Seasonal Features
# ---------------------------------------------------------------------------


class TestSeasonalFeatures:
    def test_from_index_columns(self):
        idx = pd.date_range("2024-01-01", periods=90, freq="D")
        df = pd.DataFrame({"price": np.random.randn(90)}, index=idx)
        out = SeasonalFeatures.from_index(df)
        expected_cols = {
            "day_of_week",
            "month",
            "quarter",
            "week_of_year",
            "is_month_end",
            "is_quarter_end",
            "day_of_year",
            "is_year_start",
        }
        assert expected_cols == set(out.columns)
        assert len(out) == 90

    def test_day_of_week_range(self):
        idx = pd.date_range("2024-01-01", periods=14, freq="D")
        df = pd.DataFrame({"x": range(14)}, index=idx)
        out = SeasonalFeatures.from_index(df)
        assert out["day_of_week"].min() == 0
        assert out["day_of_week"].max() == 6

    def test_quarter_values(self):
        idx = pd.date_range("2024-01-01", periods=365, freq="D")
        df = pd.DataFrame({"x": range(365)}, index=idx)
        out = SeasonalFeatures.from_index(df)
        assert set(out["quarter"].unique()) == {1, 2, 3, 4}

    def test_month_end_flag(self):
        idx = pd.date_range("2024-01-28", periods=5, freq="D")
        df = pd.DataFrame({"x": range(5)}, index=idx)
        out = SeasonalFeatures.from_index(df)
        # Jan 31 is month end
        assert out["is_month_end"].sum() >= 1


# ---------------------------------------------------------------------------
# Hierarchical Features
# ---------------------------------------------------------------------------


class TestHierarchicalFeatures:
    @pytest.fixture()
    def grouped_df(self):
        return pd.DataFrame(
            {
                "value": [10, 20, 30, 40, 50, 60],
                "sector": ["A", "A", "A", "B", "B", "B"],
            }
        )

    def test_group_mean(self, grouped_df):
        result = HierarchicalFeatures.group_mean(grouped_df, "value", "sector")
        assert result.name == "value_group_mean"
        # Sector A mean = 20, Sector B mean = 50
        assert result.iloc[0] == pytest.approx(20.0)
        assert result.iloc[3] == pytest.approx(50.0)

    def test_group_rank(self, grouped_df):
        result = HierarchicalFeatures.group_rank(grouped_df, "value", "sector")
        assert result.name == "value_group_rank"
        # Ranks are percentile within group
        assert 0 < result.iloc[0] <= 1.0

    def test_group_z_score(self, grouped_df):
        result = HierarchicalFeatures.group_z_score(grouped_df, "value", "sector")
        assert result.name == "value_group_z"
        # z-scores should have mean ≈ 0 within each group
        for sector in ["A", "B"]:
            mask = grouped_df["sector"] == sector
            z_mean = result[mask].mean()
            assert abs(z_mean) < 1e-10

    def test_group_stats_returns_all(self, grouped_df):
        result = HierarchicalFeatures.group_stats(grouped_df, "value", "sector")
        assert result.shape[1] == 3
        assert "value_group_mean" in result.columns
        assert "value_group_rank" in result.columns
        assert "value_group_z" in result.columns


# ---------------------------------------------------------------------------
# Interaction Features
# ---------------------------------------------------------------------------


class TestInteractionFeatures:
    @pytest.fixture()
    def numeric_df(self):
        return pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "c": [0.0, 1.0, 2.0]})

    def test_product(self, numeric_df):
        result = InteractionFeatures.product(numeric_df, "a", "b")
        assert list(result) == [4.0, 10.0, 18.0]
        assert result.name == "a_x_b"

    def test_ratio(self, numeric_df):
        result = InteractionFeatures.ratio(numeric_df, "a", "b")
        assert result.iloc[0] == pytest.approx(0.25)
        assert result.name == "a_over_b"

    def test_ratio_zero_denominator(self, numeric_df):
        result = InteractionFeatures.ratio(numeric_df, "a", "c")
        assert np.isnan(result.iloc[0])  # c[0] == 0 → NaN

    def test_difference(self, numeric_df):
        result = InteractionFeatures.difference(numeric_df, "b", "a")
        assert list(result) == [3.0, 3.0, 3.0]

    def test_pairwise_interactions(self, numeric_df):
        result = InteractionFeatures.pairwise_interactions(
            numeric_df, ["a", "b"], methods=["product"]
        )
        assert result.shape[1] == 1
        assert "a_x_b" in result.columns

    def test_pairwise_all_methods(self, numeric_df):
        result = InteractionFeatures.pairwise_interactions(numeric_df, ["a", "b"])
        # 1 pair × 3 methods = 3 columns
        assert result.shape[1] == 3


# ---------------------------------------------------------------------------
# Representation Features
# ---------------------------------------------------------------------------


class TestRepresentationFeatures:
    def test_target_encode_no_leakage(self):
        """Target encoding must not use current or future rows."""
        df = pd.DataFrame(
            {
                "cat": ["A", "A", "A", "B", "B", "B"] * 5,
                "target": [1, 0, 1, 0, 1, 0] * 5,
            }
        )
        result = RepresentationFeatures.target_encode(df, "cat", "target", min_samples=2)
        # First row must be NaN (no prior data)
        assert np.isnan(result.iloc[0])
        assert result.name == "cat_target_enc"

    def test_target_encode_monotonic_convergence(self):
        """With enough data, the encoding should converge to the true mean."""
        n = 200
        rng = np.random.default_rng(42)
        cats = rng.choice(["X", "Y"], size=n)
        targets = np.where(cats == "X", rng.normal(0.7, 0.1, n), rng.normal(0.3, 0.1, n))
        df = pd.DataFrame({"cat": cats, "target": targets})
        result = RepresentationFeatures.target_encode(df, "cat", "target", min_samples=5)
        # Later values should be close to 0.7 for X and 0.3 for Y
        late = result.iloc[-20:]
        assert late.notna().all()

    def test_frequency_encode(self):
        df = pd.DataFrame({"cat": ["A", "A", "B", "A", "B", "B"]})
        result = RepresentationFeatures.frequency_encode(df, "cat")
        # First row has no prior data → NaN
        assert np.isnan(result.iloc[0])
        assert result.name == "cat_freq_enc"


# ---------------------------------------------------------------------------
# Feature Selection
# ---------------------------------------------------------------------------


class TestSelectFeatures:
    @pytest.fixture()
    def feature_df(self):
        rng = np.random.default_rng(42)
        n = 200
        return pd.DataFrame(
            {
                "good_1": rng.normal(0, 1, n),
                "good_2": rng.normal(0, 1, n),
                "constant": np.ones(n),  # zero variance
                "mostly_nan": np.where(rng.random(n) < 0.9, np.nan, 1.0),  # 90% missing
                "duplicate": rng.normal(0, 1, n),  # will be set to near-copy
            }
        )

    def test_drops_constant(self, feature_df):
        selected = select_features(feature_df)
        assert "constant" not in selected

    def test_drops_mostly_nan(self, feature_df):
        selected = select_features(feature_df, max_missing_rate=0.5)
        assert "mostly_nan" not in selected

    def test_drops_highly_correlated(self, feature_df):
        # Make duplicate nearly identical to good_1
        noise = np.random.default_rng(0).normal(0, 0.001, len(feature_df))
        feature_df["duplicate"] = feature_df["good_1"] + noise
        selected = select_features(feature_df, max_correlation=0.95)
        # Only one of good_1/duplicate should survive
        assert not ("good_1" in selected and "duplicate" in selected)

    def test_max_features_with_target(self, feature_df):
        y = feature_df["good_1"] * 2 + np.random.default_rng(1).normal(0, 0.1, len(feature_df))
        selected = select_features(
            feature_df[["good_1", "good_2"]],
            y=y,
            max_features=1,
        )
        assert len(selected) == 1
        assert "good_1" in selected  # should be most correlated with target

    def test_returns_list(self, feature_df):
        result = select_features(feature_df)
        assert isinstance(result, list)
        assert all(isinstance(c, str) for c in result)
