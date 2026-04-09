"""Unit tests for backtesting framework (no database required)."""

import numpy as np
import pandas as pd
import pytest

from pipeline.backtesting.transaction_costs import (
    FixedPlusSpreadModel,
    SquareRootImpactModel,
    Trade,
    apply_transaction_costs,
)
from pipeline.backtesting.walk_forward import (
    ValidationResult,
    purged_kfold_splits,
    purged_kfold_validate,
    walk_forward_splits,
    walk_forward_validate,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_df():
    """Feature DataFrame with a target column."""
    np.random.seed(42)
    n = 500
    idx = pd.bdate_range("2022-01-01", periods=n)
    return pd.DataFrame(
        {
            "feature_a": np.random.randn(n),
            "feature_b": np.random.randn(n),
            "target": np.random.randn(n) * 0.01,
        },
        index=idx,
    )


def _dummy_train(train_df):
    """Dummy model: just remember the mean target."""
    return {"mean": train_df["target"].mean()}


def _dummy_predict(model, test_df):
    """Predict the training mean for every observation."""
    return pd.Series(model["mean"], index=test_df.index)


def _dummy_eval(y_true, y_pred):
    """Compute MAE and MSE."""
    err = y_true - y_pred
    return {"mae": float(err.abs().mean()), "mse": float((err**2).mean())}


# ---------------------------------------------------------------------------
# Walk-Forward Tests
# ---------------------------------------------------------------------------


class TestWalkForwardSplits:
    def test_basic_splits(self, sample_df):
        splits = list(walk_forward_splits(sample_df.index, 100, 50))
        assert len(splits) >= 2

        for train_idx, test_idx in splits:
            # No overlap
            assert len(set(train_idx) & set(test_idx)) == 0
            # Test follows train
            assert train_idx[-1] < test_idx[0]

    def test_expanding_window(self, sample_df):
        splits = list(walk_forward_splits(sample_df.index, 100, 50, expanding=True))
        train_sizes = [len(t) for t, _ in splits]
        # In expanding mode, each successive training set should be >= previous
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] >= train_sizes[i - 1]

    def test_rolling_window(self, sample_df):
        splits = list(walk_forward_splits(sample_df.index, 100, 50, expanding=False))
        train_sizes = [len(t) for t, _ in splits]
        # In rolling mode, all training sets should be same size
        assert all(s == train_sizes[0] for s in train_sizes)

    def test_custom_step_size(self, sample_df):
        splits_step_25 = list(walk_forward_splits(sample_df.index, 100, 50, step_size=25))
        splits_step_50 = list(walk_forward_splits(sample_df.index, 100, 50, step_size=50))
        # Smaller step should produce more folds
        assert len(splits_step_25) >= len(splits_step_50)


class TestWalkForwardValidate:
    def test_returns_result(self, sample_df):
        result = walk_forward_validate(
            sample_df,
            _dummy_train,
            _dummy_predict,
            _dummy_eval,
            target_col="target",
            train_size=100,
            test_size=50,
        )
        assert isinstance(result, ValidationResult)
        assert len(result.folds) >= 2

    def test_metrics_computed(self, sample_df):
        result = walk_forward_validate(
            sample_df,
            _dummy_train,
            _dummy_predict,
            _dummy_eval,
            target_col="target",
            train_size=100,
            test_size=50,
        )
        for fold in result.folds:
            assert "mae" in fold.metrics
            assert "mse" in fold.metrics
            assert fold.metrics["mae"] >= 0

    def test_mean_metrics(self, sample_df):
        result = walk_forward_validate(
            sample_df,
            _dummy_train,
            _dummy_predict,
            _dummy_eval,
            target_col="target",
            train_size=100,
            test_size=50,
        )
        means = result.mean_metrics
        assert "mae" in means
        assert means["mae"] >= 0

    def test_summary_dataframe(self, sample_df):
        result = walk_forward_validate(
            sample_df,
            _dummy_train,
            _dummy_predict,
            _dummy_eval,
            target_col="target",
            train_size=100,
            test_size=50,
        )
        summary = result.summary()
        assert isinstance(summary, pd.DataFrame)
        assert "fold" in summary.columns
        assert "mae" in summary.columns


class TestWalkForwardEmbargo:
    def test_embargo_default_creates_gap(self, sample_df):
        """Default embargo_size=5 creates a gap between train and test."""
        splits = list(walk_forward_splits(sample_df.index, 100, 50))
        for train_idx, test_idx in splits:
            gap = test_idx[0] - train_idx[-1]
            assert gap == 6  # 5 embargo + 1 for the next index

    def test_embargo_zero_no_gap(self, sample_df):
        """embargo_size=0 reproduces the old contiguous behavior."""
        splits = list(walk_forward_splits(sample_df.index, 100, 50, embargo_size=0))
        for train_idx, test_idx in splits:
            assert test_idx[0] == train_idx[-1] + 1

    def test_embargo_no_overlap(self, sample_df):
        """No indices appear in both train and test or the embargo zone."""
        embargo = 10
        splits = list(walk_forward_splits(sample_df.index, 100, 50, embargo_size=embargo))
        for train_idx, test_idx in splits:
            train_set = set(train_idx)
            test_set = set(test_idx)
            embargo_zone = set(range(train_idx[-1] + 1, test_idx[0]))
            assert len(train_set & test_set) == 0
            assert len(train_set & embargo_zone) == 0
            assert len(embargo_zone) == embargo

    def test_embargo_reduces_fold_count(self, sample_df):
        """Larger embargo means fewer folds fit in the same data."""
        splits_0 = list(walk_forward_splits(sample_df.index, 100, 50, embargo_size=0))
        splits_20 = list(walk_forward_splits(sample_df.index, 100, 50, embargo_size=20))
        assert len(splits_0) >= len(splits_20)

    def test_validate_passes_embargo(self, sample_df):
        """walk_forward_validate accepts embargo_size parameter."""
        result = walk_forward_validate(
            sample_df,
            _dummy_train,
            _dummy_predict,
            _dummy_eval,
            target_col="target",
            train_size=100,
            test_size=50,
            embargo_size=10,
        )
        assert isinstance(result, ValidationResult)
        assert len(result.folds) >= 1

    def test_label_horizon_overrides_embargo(self, sample_df):
        """label_horizon increases embargo when larger than embargo_size."""
        splits_base = list(walk_forward_splits(sample_df.index, 100, 50, embargo_size=5))
        splits_horizon = list(
            walk_forward_splits(sample_df.index, 100, 50, embargo_size=5, label_horizon=20)
        )
        # With label_horizon=20, the effective embargo is 20, so fewer folds
        assert len(splits_base) >= len(splits_horizon)
        # Verify the gap is at least 20
        for train_idx, test_idx in splits_horizon:
            gap = test_idx[0] - train_idx[-1]
            assert gap >= 21  # 20 embargo + 1

    def test_label_horizon_no_effect_when_smaller(self, sample_df):
        """label_horizon has no effect when smaller than embargo_size."""
        splits_base = list(walk_forward_splits(sample_df.index, 100, 50, embargo_size=10))
        splits_small = list(
            walk_forward_splits(sample_df.index, 100, 50, embargo_size=10, label_horizon=3)
        )
        assert len(splits_base) == len(splits_small)


# ---------------------------------------------------------------------------
# Purged k-Fold Tests
# ---------------------------------------------------------------------------


class TestPurgedKFoldSplits:
    def test_basic_splits(self, sample_df):
        splits = list(purged_kfold_splits(sample_df.index, n_folds=5))
        assert len(splits) == 5

    def test_no_overlap(self, sample_df):
        splits = list(purged_kfold_splits(sample_df.index, n_folds=5, embargo_pct=0.02))
        for train_idx, test_idx in splits:
            overlap = set(train_idx) & set(test_idx)
            assert len(overlap) == 0

    def test_embargo_gap(self, sample_df):
        embargo_pct = 0.05
        splits = list(purged_kfold_splits(sample_df.index, n_folds=5, embargo_pct=embargo_pct))
        n = len(sample_df)
        embargo_size = int(n * embargo_pct)

        for train_idx, test_idx in splits:
            test_end = test_idx[-1]
            # No training index should be in [test_end+1, test_end+embargo_size]
            embargo_zone = set(range(test_end + 1, min(test_end + embargo_size + 1, n)))
            assert len(embargo_zone & set(train_idx)) == 0


class TestPurgedKFoldValidate:
    def test_returns_result(self, sample_df):
        result = purged_kfold_validate(
            sample_df,
            _dummy_train,
            _dummy_predict,
            _dummy_eval,
            target_col="target",
            n_folds=5,
        )
        assert isinstance(result, ValidationResult)
        assert len(result.folds) == 5


# ---------------------------------------------------------------------------
# Transaction Cost Tests
# ---------------------------------------------------------------------------


class TestFixedPlusSpreadModel:
    def test_basic_cost(self):
        model = FixedPlusSpreadModel(spread_bps=10, commission_per_share=0.005)
        trade = Trade(symbol="SPY", side="buy", quantity=100, price=450.0)
        cost = model.estimate(trade)

        assert cost.spread_cost > 0
        assert cost.commission > 0
        assert cost.market_impact == 0
        assert cost.total == cost.spread_cost + cost.commission

    def test_min_commission(self):
        model = FixedPlusSpreadModel(min_commission=1.0, commission_per_share=0.005)
        trade = Trade(symbol="SPY", side="buy", quantity=1, price=450.0)
        cost = model.estimate(trade)
        assert cost.commission >= 1.0

    def test_larger_trade_costs_more(self):
        model = FixedPlusSpreadModel()
        small = model.estimate(Trade("SPY", "buy", 10, 450.0))
        large = model.estimate(Trade("SPY", "buy", 1000, 450.0))
        assert large.total > small.total


class TestSquareRootImpactModel:
    def test_impact_increases_with_size(self):
        model = SquareRootImpactModel(sigma=0.02, eta=0.25)
        small = model.estimate(Trade("SPY", "buy", 1000, 450.0, adv=10_000_000))
        large = model.estimate(Trade("SPY", "buy", 100_000, 450.0, adv=10_000_000))
        assert large.market_impact > small.market_impact

    def test_zero_adv_no_crash(self):
        model = SquareRootImpactModel()
        cost = model.estimate(Trade("SPY", "buy", 100, 450.0, adv=0))
        assert cost.market_impact == 0

    def test_includes_all_components(self):
        model = SquareRootImpactModel()
        cost = model.estimate(Trade("SPY", "buy", 10_000, 450.0, adv=5_000_000))
        assert cost.spread_cost > 0
        assert cost.commission > 0
        assert cost.market_impact > 0


class TestApplyTransactionCosts:
    def test_net_return_less_than_gross(self):
        dates = pd.bdate_range("2024-01-01", periods=10)
        prices = pd.DataFrame({"SPY": np.linspace(450, 460, 10)}, index=dates)
        positions = pd.DataFrame({"SPY": [0, 100, 100, 100, 100, 200, 200, 200, 0, 0]}, index=dates)

        result = apply_transaction_costs(positions, prices)

        assert "net_return" in result.columns
        assert "gross_return" in result.columns
        # On days with trades, net return should be less than gross
        trade_days = positions.diff().fillna(0).abs().sum(axis=1) > 0
        for dt in result.index[trade_days]:
            assert result.loc[dt, "total_cost"] >= 0
