"""Probability calibration methods (Agent Directive V7 — Section 8).

Provides calibration transformations that map raw model outputs to
well-calibrated probabilities, plus a wrapper that integrates calibration
into the walk-forward validation callable interface.

Temporal safety: calibration is always fit on a held-out portion of the
training window, never on data the model was trained on.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CalibrationMethod(str, Enum):
    """Supported calibration methods."""

    PLATT = "platt"
    ISOTONIC = "isotonic"
    NONE = "none"


@dataclass
class CalibrationComparison:
    """Side-by-side comparison of raw vs calibrated outputs."""

    raw_ece: float
    raw_brier: float
    raw_log_loss: float
    calibrated_ece: float
    calibrated_brier: float
    calibrated_log_loss: float
    method: str
    n_samples: int


class Calibrator:
    """Probability calibrator using Platt scaling or isotonic regression.

    Usage::

        cal = Calibrator(method=CalibrationMethod.ISOTONIC)
        cal.fit(y_true, y_prob)
        calibrated = cal.transform(y_prob_new)
    """

    def __init__(self, method: CalibrationMethod = CalibrationMethod.ISOTONIC) -> None:
        self.method = method
        self._fitted = False
        self._model: Any = None

    def fit(self, y_true: pd.Series, y_prob: pd.Series) -> "Calibrator":
        """Fit calibrator on held-out validation data."""
        if self.method == CalibrationMethod.NONE:
            self._fitted = True
            return self

        y_t = np.asarray(y_true, dtype=float)
        y_p = np.asarray(y_prob, dtype=float)

        mask = np.isfinite(y_t) & np.isfinite(y_p)
        y_t, y_p = y_t[mask], y_p[mask]

        if len(y_t) < 10:
            logger.warning("Too few samples (%d) for calibration, using identity", len(y_t))
            self.method = CalibrationMethod.NONE
            self._fitted = True
            return self

        if self.method == CalibrationMethod.PLATT:
            from sklearn.linear_model import LogisticRegression

            self._model = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
            self._model.fit(y_p.reshape(-1, 1), y_t)

        elif self.method == CalibrationMethod.ISOTONIC:
            from sklearn.isotonic import IsotonicRegression

            self._model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            self._model.fit(y_p, y_t)

        self._fitted = True
        return self

    def transform(self, y_prob: pd.Series) -> pd.Series:
        """Apply calibration to raw probabilities."""
        if not self._fitted:
            raise RuntimeError("Calibrator has not been fit yet")

        if self.method == CalibrationMethod.NONE:
            return y_prob

        y_p = np.asarray(y_prob, dtype=float)
        index = y_prob.index if isinstance(y_prob, pd.Series) else None

        if self.method == CalibrationMethod.PLATT:
            calibrated = self._model.predict_proba(y_p.reshape(-1, 1))[:, 1]
        elif self.method == CalibrationMethod.ISOTONIC:
            calibrated = self._model.predict(y_p)
        else:
            calibrated = y_p

        result = np.clip(calibrated, 0.0, 1.0)
        if index is not None:
            return pd.Series(result, index=index, name="calibrated_prob")
        return pd.Series(result, name="calibrated_prob")

    def fit_transform(self, y_true: pd.Series, y_prob: pd.Series) -> pd.Series:
        """Fit and transform in one step."""
        self.fit(y_true, y_prob)
        return self.transform(y_prob)


class CalibratedModelWrapper:
    """Wraps model train_fn/predict_fn to add post-hoc calibration.

    Splits each training window into model-training and calibration-fitting
    subsets to prevent leakage. The calibrator is always fit on the *later*
    portion of the training data (temporally safe).

    The returned callables are compatible with ``walk_forward_validate()``.

    Usage::

        wrapper = CalibratedModelWrapper(train_fn, predict_fn)
        result = walk_forward_validate(
            df, wrapper.calibrated_train_fn,
            wrapper.calibrated_predict_fn, eval_fn, target_col,
        )
    """

    def __init__(
        self,
        train_fn: Callable[[pd.DataFrame], Any],
        predict_fn: Callable[[Any, pd.DataFrame], pd.Series],
        method: CalibrationMethod = CalibrationMethod.ISOTONIC,
        calibration_fraction: float = 0.2,
        target_col: str = "target",
    ) -> None:
        self.train_fn = train_fn
        self.predict_fn = predict_fn
        self.method = method
        self.calibration_fraction = calibration_fraction
        self.target_col = target_col

    def calibrated_train_fn(self, train_df: pd.DataFrame) -> Any:
        """Train model on first portion, fit calibrator on second portion."""
        n = len(train_df)
        cal_size = max(10, int(n * self.calibration_fraction))
        model_end = n - cal_size

        model_df = train_df.iloc[:model_end]
        cal_df = train_df.iloc[model_end:]

        model = self.train_fn(model_df)

        raw_preds = self.predict_fn(model, cal_df)
        y_true_cal = cal_df[self.target_col]

        calibrator = Calibrator(method=self.method)
        calibrator.fit(y_true_cal, raw_preds)

        return {"model": model, "calibrator": calibrator}

    def calibrated_predict_fn(self, model_bundle: Any, test_df: pd.DataFrame) -> pd.Series:
        """Predict with model, then apply calibration."""
        model = model_bundle["model"]
        calibrator = model_bundle["calibrator"]

        raw_preds = self.predict_fn(model, test_df)
        return calibrator.transform(raw_preds)


def generate_calibration_comparison(
    y_true: pd.Series,
    raw_prob: pd.Series,
    calibrated_prob: pd.Series,
    method: str = "isotonic",
) -> CalibrationComparison:
    """Compare raw vs calibrated outputs using standard metrics.

    Uses existing metric functions from eval/metrics.py.
    """
    from pipeline.eval.metrics import brier_score, calibration_error, log_loss

    y_t, y_r = y_true.align(raw_prob, join="inner")
    y_t_c, y_c = y_true.align(calibrated_prob, join="inner")

    return CalibrationComparison(
        raw_ece=calibration_error(y_t, y_r),
        raw_brier=brier_score(y_t, y_r),
        raw_log_loss=log_loss(y_t, y_r),
        calibrated_ece=calibration_error(y_t_c, y_c),
        calibrated_brier=brier_score(y_t_c, y_c),
        calibrated_log_loss=log_loss(y_t_c, y_c),
        method=method,
        n_samples=len(y_t),
    )
