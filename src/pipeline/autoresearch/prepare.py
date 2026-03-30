"""Immutable evaluator — the neutral judge.

This file defines the evaluation function that the LLM agent CANNOT modify.
It wraps the existing walk-forward validation infrastructure and returns a
single scalar metric (lower is better) that drives the keep/revert decision.

The agent's sandbox (``train_config.py``) controls *what* is trained.
This file controls *how* it is judged.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, mean_squared_error

from pipeline.backtesting.walk_forward import ValidationResult, walk_forward_validate
from pipeline.model_search import ModelSpec, _create_estimator

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Immutable evaluation outcome."""

    primary_metric: float  # The single scalar (lower = better)
    secondary_metrics: dict[str, float] = field(default_factory=dict)
    validation_result: ValidationResult | None = None
    elapsed_seconds: float = 0.0
    error: str | None = None


def _eval_fn(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Per-fold evaluation metrics.

    Returns a dict consumed by ``walk_forward_validate``.
    The primary metric for AutoResearch is *negative Sharpe* so that
    lower = better (consistent with loss minimisation).
    """
    # Simple return-based Sharpe approximation:
    # treat predictions as position signals, compute realised PnL
    aligned = pd.DataFrame({"y": y_true, "pred": y_pred}).dropna()
    if aligned.empty:
        return {"neg_sharpe": 0.0, "hit_rate": 0.0, "mse": 1.0}

    y = aligned["y"]
    pred = aligned["pred"]

    # Directional hit rate
    direction_correct = (np.sign(pred) == np.sign(y)).mean()

    # Signal-weighted returns (position sizing proportional to prediction)
    pnl = pred * y
    sharpe = pnl.mean() / (pnl.std() + 1e-9) * np.sqrt(252)

    mse = float(mean_squared_error(y, pred))

    return {
        "neg_sharpe": -sharpe,  # lower is better
        "hit_rate": float(direction_correct),
        "mse": mse,
        "mean_pnl": float(pnl.mean()),
    }


def evaluate(
    df: pd.DataFrame,
    model_spec: ModelSpec,
    target_col: str = "fwd_return_1d",
    train_size: int = 252,
    test_size: int = 63,
    embargo_size: int = 5,
    expanding: bool = True,
) -> EvalResult:
    """Run walk-forward validation and return a single-scalar result.

    This function is the **immutable judge** of the AutoResearch loop.
    The agent proposes changes to the model spec; this function scores them.

    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix with a DatetimeIndex and ``target_col``.
    model_spec : ModelSpec
        Model configuration (family, hyperparameters, features).
    target_col : str
        Column name for the prediction target.
    train_size, test_size, embargo_size, expanding
        Walk-forward parameters.

    Returns
    -------
    EvalResult
        ``.primary_metric`` is *negative Sharpe* (lower = better).
    """
    t0 = time.perf_counter()

    try:
        estimator = _create_estimator(model_spec)
    except (ValueError, ImportError) as exc:
        return EvalResult(
            primary_metric=0.0,
            error=f"Model creation failed: {exc}",
            elapsed_seconds=time.perf_counter() - t0,
        )

    feature_cols = model_spec.feature_cols
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]

    def train_fn(train_df: pd.DataFrame) -> Any:
        x = train_df[feature_cols].fillna(0.0)
        y = train_df[target_col]
        estimator.fit(x, y)
        return estimator

    def predict_fn(model: Any, test_df: pd.DataFrame) -> pd.Series:
        x = test_df[feature_cols].fillna(0.0)
        return pd.Series(model.predict(x), index=test_df.index)

    try:
        vr = walk_forward_validate(
            df=df,
            train_fn=train_fn,
            predict_fn=predict_fn,
            eval_fn=_eval_fn,
            target_col=target_col,
            train_size=train_size,
            test_size=test_size,
            expanding=expanding,
            embargo_size=embargo_size,
        )
    except Exception as exc:
        return EvalResult(
            primary_metric=0.0,
            error=f"Validation failed: {exc}",
            elapsed_seconds=time.perf_counter() - t0,
        )

    elapsed = time.perf_counter() - t0
    mean = vr.mean_metrics

    return EvalResult(
        primary_metric=mean.get("neg_sharpe", 0.0),
        secondary_metrics=mean,
        validation_result=vr,
        elapsed_seconds=elapsed,
    )
