"""Immutable evaluator — the neutral judge.

This file defines the evaluation function that the LLM agent CANNOT modify.
It wraps the existing walk-forward validation infrastructure and returns a
single scalar metric (lower is better) that drives the keep/revert decision.

Evaluation includes:
- Annualised Sharpe ratio (primary, negated so lower = better)
- Sortino ratio (downside-only volatility)
- Maximum drawdown
- Hit rate (directional accuracy)
- Mean squared error
- Drawdown-penalised composite score

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
from sklearn.metrics import mean_squared_error

from pipeline.backtesting.walk_forward import ValidationResult, walk_forward_validate
from pipeline.model_search import ModelSpec, _create_estimator

logger = logging.getLogger(__name__)

ANNUALISATION_FACTOR = np.sqrt(252)


@dataclass
class EvalResult:
    """Immutable evaluation outcome."""

    primary_metric: float  # The single scalar (lower = better)
    secondary_metrics: dict[str, float] = field(default_factory=dict)
    validation_result: ValidationResult | None = None
    fold_metrics: list[dict[str, float]] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    error: str | None = None


def _eval_fn(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Per-fold evaluation metrics.

    Returns a dict consumed by ``walk_forward_validate``.
    All metrics are computed from signal-weighted PnL where the prediction
    magnitude acts as position size.
    """
    aligned = pd.DataFrame({"y": y_true, "pred": y_pred}).dropna()
    if len(aligned) < 10:
        return {
            "neg_sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "hit_rate": 0.0,
            "mse": 1.0,
            "mean_pnl": 0.0,
            "pnl_std": 0.0,
            "n_obs": float(len(aligned)),
        }

    y = aligned["y"]
    pred = aligned["pred"]

    # Signal-weighted returns (position sizing proportional to prediction)
    pnl = pred * y

    # -- Sharpe (annualised) --
    sharpe = pnl.mean() / (pnl.std() + 1e-9) * ANNUALISATION_FACTOR

    # -- Sortino (downside deviation only) --
    downside = pnl[pnl < 0]
    downside_std = downside.std() if len(downside) > 1 else 1e-9
    sortino = pnl.mean() / (downside_std + 1e-9) * ANNUALISATION_FACTOR

    # -- Maximum drawdown on cumulative PnL curve --
    cum_pnl = pnl.cumsum()
    running_max = cum_pnl.cummax()
    drawdowns = cum_pnl - running_max
    max_drawdown = float(drawdowns.min()) if len(drawdowns) > 0 else 0.0

    # -- Hit rate (directional accuracy) --
    nonzero = (y != 0) & (pred != 0)
    if nonzero.sum() > 0:
        hit_rate = float((np.sign(pred[nonzero]) == np.sign(y[nonzero])).mean())
    else:
        hit_rate = 0.0

    mse = float(mean_squared_error(y, pred))

    return {
        "neg_sharpe": -sharpe,
        "sortino": float(sortino),
        "max_drawdown": float(max_drawdown),
        "hit_rate": hit_rate,
        "mse": mse,
        "mean_pnl": float(pnl.mean()),
        "pnl_std": float(pnl.std()),
        "n_obs": float(len(aligned)),
    }


def _composite_score(mean_metrics: dict[str, float]) -> float:
    """Compute a drawdown-penalised composite (lower = better).

    composite = neg_sharpe + 2.0 * max_drawdown_penalty

    The max_drawdown term is negative (losses), so subtracting it penalises
    strategies with deep drawdowns even if their Sharpe is decent.
    """
    neg_sharpe = mean_metrics.get("neg_sharpe", 0.0)
    max_dd = mean_metrics.get("max_drawdown", 0.0)  # negative number
    # max_dd is already negative; multiplying by 2 and adding increases score (worse)
    return neg_sharpe + 2.0 * max_dd


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
        ``.primary_metric`` is the composite score (lower = better).
        ``.secondary_metrics`` contains neg_sharpe, sortino, max_drawdown,
        hit_rate, mse, mean_pnl, pnl_std.
    """
    t0 = time.perf_counter()

    # Validate feature columns exist in the dataframe
    feature_cols = model_spec.feature_cols
    if feature_cols is not None:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            return EvalResult(
                primary_metric=0.0,
                error=f"Missing feature columns: {missing[:10]}",
                elapsed_seconds=time.perf_counter() - t0,
            )
    else:
        feature_cols = [c for c in df.columns if c != target_col]

    if target_col not in df.columns:
        return EvalResult(
            primary_metric=0.0,
            error=f"Target column '{target_col}' not found",
            elapsed_seconds=time.perf_counter() - t0,
        )

    # Check minimum data requirements
    min_rows = train_size + test_size + embargo_size
    if len(df) < min_rows:
        return EvalResult(
            primary_metric=0.0,
            error=f"Dataset too small: {len(df)} rows < {min_rows} required",
            elapsed_seconds=time.perf_counter() - t0,
        )

    try:
        estimator = _create_estimator(model_spec)
    except (ValueError, ImportError) as exc:
        return EvalResult(
            primary_metric=0.0,
            error=f"Model creation failed: {exc}",
            elapsed_seconds=time.perf_counter() - t0,
        )

    # Freeze feature_cols for closures
    _feature_cols = list(feature_cols)

    def train_fn(train_df: pd.DataFrame) -> Any:
        x = train_df[_feature_cols].fillna(0.0)
        y = train_df[target_col]
        estimator.fit(x, y)
        return estimator

    def predict_fn(model: Any, test_df: pd.DataFrame) -> pd.Series:
        x = test_df[_feature_cols].fillna(0.0)
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
    fold_metrics = [f.metrics for f in vr.folds]

    primary = _composite_score(mean)

    return EvalResult(
        primary_metric=primary,
        secondary_metrics=mean,
        validation_result=vr,
        fold_metrics=fold_metrics,
        elapsed_seconds=elapsed,
    )
