"""Tests for calibration module (Agent Directive V7 — Section 8)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.calibration import (
    CalibrationMethod,
    CalibratedModelWrapper,
    Calibrator,
    generate_calibration_comparison,
)


class TestCalibrator:
    """Tests for the Calibrator class."""

    def _make_binary_data(self, n: int = 200, seed: int = 42) -> tuple[pd.Series, pd.Series]:
        rng = np.random.RandomState(seed)
        y_true = pd.Series(rng.randint(0, 2, size=n), name="target")
        # Miscalibrated probabilities: shift toward overconfidence
        raw = y_true.astype(float) * 0.6 + rng.uniform(0, 0.4, size=n)
        y_prob = pd.Series(np.clip(raw, 0.01, 0.99), name="prob")
        return y_true, y_prob

    def test_platt_fit_transform(self) -> None:
        y_true, y_prob = self._make_binary_data()
        cal = Calibrator(method=CalibrationMethod.PLATT)
        result = cal.fit_transform(y_true, y_prob)

        assert len(result) == len(y_prob)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_isotonic_fit_transform(self) -> None:
        y_true, y_prob = self._make_binary_data()
        cal = Calibrator(method=CalibrationMethod.ISOTONIC)
        result = cal.fit_transform(y_true, y_prob)

        assert len(result) == len(y_prob)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_isotonic_monotonic(self) -> None:
        y_true, y_prob = self._make_binary_data(n=500)
        cal = Calibrator(method=CalibrationMethod.ISOTONIC)
        cal.fit(y_true, y_prob)

        # Transform a sorted input — output should be non-decreasing
        sorted_input = pd.Series(np.linspace(0.01, 0.99, 100))
        result = cal.transform(sorted_input)
        diffs = np.diff(result.values)
        assert np.all(diffs >= -1e-10), "Isotonic output should be non-decreasing"

    def test_none_method_passthrough(self) -> None:
        y_true, y_prob = self._make_binary_data()
        cal = Calibrator(method=CalibrationMethod.NONE)
        result = cal.fit_transform(y_true, y_prob)
        pd.testing.assert_series_equal(result, y_prob)

    def test_calibrator_improves_ece(self) -> None:
        from pipeline.eval.metrics import calibration_error

        rng = np.random.RandomState(99)
        n = 500
        y_true = pd.Series(rng.randint(0, 2, size=n))
        # Heavily miscalibrated: predict ~0.8 for everything
        y_prob = pd.Series(np.clip(0.8 + rng.normal(0, 0.05, size=n), 0.01, 0.99))

        cal = Calibrator(method=CalibrationMethod.ISOTONIC)
        calibrated = cal.fit_transform(y_true, y_prob)

        raw_ece = calibration_error(y_true, y_prob)
        cal_ece = calibration_error(y_true, calibrated)
        assert cal_ece < raw_ece, f"Calibration should reduce ECE: {cal_ece} >= {raw_ece}"

    def test_transform_before_fit_raises(self) -> None:
        cal = Calibrator()
        with pytest.raises(RuntimeError, match="not been fit"):
            cal.transform(pd.Series([0.5, 0.6]))

    def test_small_sample_fallback(self) -> None:
        y_true = pd.Series([0, 1, 0, 1, 0])
        y_prob = pd.Series([0.3, 0.7, 0.4, 0.6, 0.5])
        cal = Calibrator(method=CalibrationMethod.ISOTONIC)
        cal.fit(y_true, y_prob)
        assert cal.method == CalibrationMethod.NONE


class TestCalibratedModelWrapper:
    """Tests for the CalibratedModelWrapper class."""

    def _make_dataset(self, n: int = 400, seed: int = 42) -> pd.DataFrame:
        rng = np.random.RandomState(seed)
        dates = pd.bdate_range("2020-01-01", periods=n)
        df = pd.DataFrame(
            {
                "feat1": rng.randn(n),
                "feat2": rng.randn(n),
                "target": rng.randint(0, 2, size=n).astype(float),
            },
            index=dates,
        )
        return df

    def test_wrapper_produces_valid_callables(self) -> None:
        from pipeline.backtesting.walk_forward import walk_forward_validate

        df = self._make_dataset()

        def train_fn(train_df: pd.DataFrame) -> object:
            from sklearn.linear_model import LogisticRegression

            X = train_df[["feat1", "feat2"]].values
            y = train_df["target"].values
            model = LogisticRegression(max_iter=200)
            model.fit(X, y)
            return model

        def predict_fn(model: object, test_df: pd.DataFrame) -> pd.Series:
            X = test_df[["feat1", "feat2"]].values
            probs = model.predict_proba(X)[:, 1]  # type: ignore[union-attr]
            return pd.Series(probs, index=test_df.index)

        def eval_fn(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
            return {"hit_rate": float((y_true == (y_pred > 0.5).astype(float)).mean())}

        wrapper = CalibratedModelWrapper(
            train_fn, predict_fn, target_col="target"
        )

        result = walk_forward_validate(
            df,
            wrapper.calibrated_train_fn,
            wrapper.calibrated_predict_fn,
            eval_fn,
            target_col="target",
            train_size=200,
            test_size=50,
        )

        assert len(result.folds) > 0
        for fold in result.folds:
            assert "hit_rate" in fold.metrics
            assert fold.predictions is not None
            assert fold.predictions.min() >= 0.0
            assert fold.predictions.max() <= 1.0

    def test_wrapper_temporal_safety(self) -> None:
        """Verify calibrator is fit on later portion of training data."""
        df = self._make_dataset(n=300)
        train_df = df.iloc[:200]

        call_log: list[tuple[str, int, int]] = []

        def train_fn(train_subset: pd.DataFrame) -> str:
            call_log.append(("model_train", len(train_subset), 0))
            return "mock_model"

        def predict_fn(model: str, test_df: pd.DataFrame) -> pd.Series:
            call_log.append(("predict", len(test_df), 0))
            return pd.Series(0.5, index=test_df.index)

        wrapper = CalibratedModelWrapper(
            train_fn, predict_fn, calibration_fraction=0.2, target_col="target"
        )
        wrapper.calibrated_train_fn(train_df)

        # Model should be trained on 80% of data
        model_train_size = call_log[0][1]
        assert model_train_size == 160, f"Expected 160, got {model_train_size}"

        # Predict called on calibration portion (40 rows)
        predict_size = call_log[1][1]
        assert predict_size == 40, f"Expected 40, got {predict_size}"


class TestCalibrationComparison:
    """Tests for calibration comparison utility."""

    def test_comparison_structure(self) -> None:
        rng = np.random.RandomState(42)
        n = 200
        y_true = pd.Series(rng.randint(0, 2, size=n))
        y_raw = pd.Series(np.clip(rng.uniform(0, 1, size=n), 0.01, 0.99))
        y_cal = pd.Series(np.clip(rng.uniform(0, 1, size=n), 0.01, 0.99))

        result = generate_calibration_comparison(y_true, y_raw, y_cal, method="isotonic")

        assert result.n_samples == n
        assert result.method == "isotonic"
        assert 0 <= result.raw_ece <= 1
        assert 0 <= result.calibrated_ece <= 1
        assert result.raw_brier >= 0
        assert result.calibrated_brier >= 0
