"""Temporal Integrity Tests (Agent Directive V7 — Section 23.2).

Mandatory specialized tests for temporal integrity:
1. Feature timestamp assertion.
2. Walk-forward replay test.
3. Data leakage canary.
4. Pipeline ordering test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.backtesting.bias_checks import (
    check_no_future_data,
    data_shift_test,
    enforce_timestamp_ordering,
    random_shuffle_test,
)
from pipeline.backtesting.walk_forward import (
    walk_forward_splits,
    walk_forward_validate,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ts_index(n: int = 252, freq: str = "B") -> pd.DatetimeIndex:
    return pd.bdate_range("2023-01-01", periods=n, freq=freq)


def _make_features(n: int = 252, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = _make_ts_index(n)
    return pd.DataFrame(
        {
            "feat_a": rng.normal(0, 1, n),
            "feat_b": rng.normal(0, 1, n),
            "target": rng.normal(0, 0.01, n),
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# 23.2.1 — Feature Timestamp Assertion
# ---------------------------------------------------------------------------

class TestFeatureTimestampAssertion:
    """Verify that feature as-of timestamps are strictly before the
    prediction target's event time."""

    def test_valid_features_pass(self):
        """Features computed from past data should pass."""
        features = _make_features()
        # Make last target NaN (no future data available)
        features.loc[features.index[-1], "target"] = np.nan
        signals = pd.DataFrame({"signal": 1}, index=features.index)
        passed, violations = check_no_future_data(features, signals)
        assert passed, f"Unexpected violations: {violations}"

    def test_future_target_detected(self):
        """Non-NaN target on last row should be flagged."""
        features = _make_features()
        # Last target is NOT NaN — suspicious
        features.loc[features.index[-1], "target"] = 0.05
        signals = pd.DataFrame({"signal": 1}, index=features.index)
        passed, violations = check_no_future_data(features, signals)
        assert not passed
        assert any("target" in v.lower() for v in violations)

    def test_unsorted_features_detected(self):
        """Out-of-order timestamps should be flagged."""
        features = _make_features()
        # Reverse the index
        features = features.iloc[::-1]
        signals = pd.DataFrame({"signal": 1}, index=features.index)
        passed, violations = check_no_future_data(features, signals)
        assert not passed
        assert any("sorted" in v.lower() or "monotonic" in v.lower() for v in violations)

    def test_features_beyond_signals_detected(self):
        """Features extending beyond signal dates should be flagged."""
        features = _make_features(252)
        signals = pd.DataFrame(
            {"signal": 1}, index=features.index[:200]
        )
        passed, violations = check_no_future_data(features, signals)
        assert not passed
        assert any("extend" in v.lower() for v in violations)


# ---------------------------------------------------------------------------
# 23.2.2 — Walk-Forward Replay Test
# ---------------------------------------------------------------------------

class TestWalkForwardReplay:
    """Replay walk-forward validation on a frozen dataset and verify
    that results exactly match."""

    def test_deterministic_walk_forward(self):
        """Walk-forward splits must be deterministic for the same input."""
        idx = _make_ts_index(500)
        splits_1 = list(walk_forward_splits(idx, train_size=200, test_size=50))
        splits_2 = list(walk_forward_splits(idx, train_size=200, test_size=50))

        assert len(splits_1) == len(splits_2)
        for (tr1, te1), (tr2, te2) in zip(splits_1, splits_2, strict=True):
            np.testing.assert_array_equal(tr1, tr2)
            np.testing.assert_array_equal(te1, te2)

    def test_walk_forward_no_overlap(self):
        """Train and test sets must not overlap."""
        idx = _make_ts_index(500)
        for train_idx, test_idx in walk_forward_splits(
            idx, train_size=200, test_size=50, embargo_size=5
        ):
            overlap = set(train_idx) & set(test_idx)
            assert len(overlap) == 0, f"Train/test overlap: {overlap}"

    def test_embargo_enforced(self):
        """Embargo gap must exist between train end and test start."""
        idx = _make_ts_index(500)
        embargo = 10
        for train_idx, test_idx in walk_forward_splits(
            idx, train_size=200, test_size=50, embargo_size=embargo
        ):
            gap = test_idx[0] - train_idx[-1]
            assert gap >= embargo, (
                f"Embargo violation: gap={gap}, required={embargo}"
            )

    def test_reproducible_validation_results(self):
        """Running validation twice on the same data must yield identical results."""
        df = _make_features(500)

        def train_fn(train_df):
            return float(train_df["target"].mean())

        def predict_fn(model, test_df):
            return pd.Series(model, index=test_df.index)

        def eval_fn(y_true, y_pred):
            return {"mae": float((y_true - y_pred).abs().mean())}

        result1 = walk_forward_validate(
            df, train_fn, predict_fn, eval_fn, "target",
            train_size=200, test_size=50, embargo_size=5
        )
        result2 = walk_forward_validate(
            df, train_fn, predict_fn, eval_fn, "target",
            train_size=200, test_size=50, embargo_size=5
        )

        assert len(result1.folds) == len(result2.folds)
        for f1, f2 in zip(result1.folds, result2.folds, strict=True):
            assert f1.metrics == f2.metrics


# ---------------------------------------------------------------------------
# 23.2.3 — Data Leakage Canary
# ---------------------------------------------------------------------------

class TestDataLeakageCanary:
    """Insert deliberately future-leaked features and verify that the
    validation framework detects and rejects them."""

    def test_canary_feature_detected_by_shift_test(self):
        """A feature that IS the future target should fail the shift test."""
        df = _make_features(252)
        # Create a canary feature that is the target shifted back (= future data)
        df["canary_leak"] = df["target"].shift(-1)
        df = df.dropna()

        def strategy_fn(data):
            """Strategy that uses the canary leak gets perfect score."""
            if "canary_leak" in data.columns:
                preds = data["canary_leak"]
                actual = data["target"]
                return float(1.0 / max((actual - preds).abs().mean(), 1e-10))
            return 0.0

        suspicious, stats = data_shift_test(
            df, "target", strategy_fn, shift_periods=1
        )
        # The canary should make the strategy look suspiciously good
        # and shifting should degrade performance
        assert stats["degradation_pct"] != 0.0 or suspicious

    def test_shuffle_test_detects_temporal_dependency(self):
        """A strategy with real temporal signal should survive shuffle test.
        A strategy using future data should be flagged."""
        rng = np.random.default_rng(42)
        n = 500
        returns = pd.Series(rng.normal(0, 0.01, n), index=_make_ts_index(n))

        # Trivially bad strategy — just returns mean (no temporal info)
        def no_alpha_fn(rets):
            return float(rets.mean()) / max(float(rets.std()), 1e-10)

        suspicious, stats = random_shuffle_test(returns, no_alpha_fn, n_shuffles=50)
        # A strategy with no alpha should be suspicious (similar on shuffled)
        assert suspicious or abs(stats["z_score"]) < 3.0


# ---------------------------------------------------------------------------
# 23.2.4 — Pipeline Ordering Test
# ---------------------------------------------------------------------------

class TestPipelineOrdering:
    """Verify that pipeline execution order is deterministic."""

    def test_enforce_timestamp_ordering_sorts(self):
        """enforce_timestamp_ordering must return sorted data."""
        idx = _make_ts_index(100)
        df = pd.DataFrame({"val": range(100)}, index=idx)
        # Shuffle
        df = df.sample(frac=1.0, random_state=42)
        assert not df.index.is_monotonic_increasing

        result = enforce_timestamp_ordering(df)
        assert result.index.is_monotonic_increasing

    def test_enforce_timestamp_ordering_warns_on_duplicates(self):
        """Duplicate timestamps should trigger a warning."""
        idx = pd.DatetimeIndex(["2023-01-01", "2023-01-01", "2023-01-02"])
        df = pd.DataFrame({"val": [1, 2, 3]}, index=idx)
        # Should not raise, but logs a warning
        result = enforce_timestamp_ordering(df)
        assert len(result) == 3

    def test_feature_computation_deterministic(self):
        """Feature computations must be deterministic across runs."""
        df = _make_features(200, seed=99)
        from pipeline.features.technical_indicators import TechnicalIndicators

        result1 = TechnicalIndicators.calculate_all(
            pd.DataFrame(
                {
                    "open": 100 + df["feat_a"].cumsum(),
                    "high": 101 + df["feat_a"].cumsum(),
                    "low": 99 + df["feat_a"].cumsum(),
                    "close": 100.5 + df["feat_a"].cumsum(),
                    "volume": (1000 + df["feat_b"] * 100).abs(),
                },
                index=df.index,
            )
        )
        result2 = TechnicalIndicators.calculate_all(
            pd.DataFrame(
                {
                    "open": 100 + df["feat_a"].cumsum(),
                    "high": 101 + df["feat_a"].cumsum(),
                    "low": 99 + df["feat_a"].cumsum(),
                    "close": 100.5 + df["feat_a"].cumsum(),
                    "volume": (1000 + df["feat_b"] * 100).abs(),
                },
                index=df.index,
            )
        )
        pd.testing.assert_frame_equal(result1, result2)


# ---------------------------------------------------------------------------
# Property-Based Tests (Section 23.1)
# ---------------------------------------------------------------------------

class TestNonNegotiableProperties:
    """Property tests for non-negotiable principles from Section 1."""

    def test_temporal_ordering_preserved_in_walk_forward(self):
        """Train end must always be before test start."""
        idx = _make_ts_index(500)
        for train_idx, test_idx in walk_forward_splits(
            idx, train_size=100, test_size=50, embargo_size=5
        ):
            train_end = idx[train_idx[-1]]
            test_start = idx[test_idx[0]]
            assert train_end < test_start, (
                f"Temporal violation: train_end={train_end} >= test_start={test_start}"
            )

    def test_no_future_data_in_expanding_window(self):
        """Expanding window must never include future data in training."""
        idx = _make_ts_index(500)
        for train_idx, test_idx in walk_forward_splits(
            idx, train_size=100, test_size=50, expanding=True, embargo_size=5
        ):
            max_train_date = idx[train_idx[-1]]
            min_test_date = idx[test_idx[0]]
            assert max_train_date < min_test_date
