"""Walk-forward and purged cross-validation for time-series models.

Implements validation methodologies that prevent look-ahead bias:

* **Walk-forward**: expanding or rolling window, train → predict → slide.
* **Purged k-fold**: k-fold split with an embargo buffer between train and
  test to prevent information leakage from overlapping labels.

Reference: Marcos Lopez de Prado, *Advances in Financial Machine Learning*,
Chapter 7 (Cross-Validation in Finance).
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Protocol,
    Tuple,
)

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    """Result of a single train/test fold."""
    fold_index: int
    train_start: Any
    train_end: Any
    test_start: Any
    test_end: Any
    train_size: int
    test_size: int
    metrics: Dict[str, float] = field(default_factory=dict)
    predictions: Optional[pd.Series] = None


@dataclass
class ValidationResult:
    """Aggregate result across all folds."""
    folds: List[FoldResult]
    strategy_name: str = ""

    @property
    def mean_metrics(self) -> Dict[str, float]:
        """Mean of each metric across folds."""
        if not self.folds:
            return {}
        keys = self.folds[0].metrics.keys()
        return {
            k: float(np.mean([f.metrics[k] for f in self.folds if k in f.metrics]))
            for k in keys
        }

    @property
    def std_metrics(self) -> Dict[str, float]:
        """Std of each metric across folds."""
        if not self.folds:
            return {}
        keys = self.folds[0].metrics.keys()
        return {
            k: float(np.std([f.metrics[k] for f in self.folds if k in f.metrics]))
            for k in keys
        }

    def summary(self) -> pd.DataFrame:
        rows = []
        for f in self.folds:
            row = {
                "fold": f.fold_index,
                "train_start": f.train_start,
                "train_end": f.train_end,
                "test_start": f.test_start,
                "test_end": f.test_end,
                "train_size": f.train_size,
                "test_size": f.test_size,
            }
            row.update(f.metrics)
            rows.append(row)
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Walk-Forward Validation
# ---------------------------------------------------------------------------

def walk_forward_splits(
    index: pd.DatetimeIndex,
    train_size: int,
    test_size: int,
    step_size: Optional[int] = None,
    expanding: bool = True,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """Generate (train_indices, test_indices) for walk-forward validation.

    Args:
        index: DatetimeIndex of the dataset.
        train_size: Minimum training window size (number of observations).
        test_size: Test window size.
        step_size: Slide step; defaults to *test_size* (non-overlapping).
        expanding: If True, training window grows; if False, it rolls.
    """
    n = len(index)
    step = step_size or test_size

    start = 0
    while True:
        if expanding:
            train_start = 0
        else:
            train_start = start

        train_end = start + train_size
        test_end = train_end + test_size

        if test_end > n:
            break

        train_idx = np.arange(train_start, train_end)
        test_idx = np.arange(train_end, test_end)
        yield train_idx, test_idx

        start += step


def walk_forward_validate(
    df: pd.DataFrame,
    train_fn: Callable[[pd.DataFrame], Any],
    predict_fn: Callable[[Any, pd.DataFrame], pd.Series],
    eval_fn: Callable[[pd.Series, pd.Series], Dict[str, float]],
    target_col: str,
    train_size: int = 252,
    test_size: int = 63,
    step_size: Optional[int] = None,
    expanding: bool = True,
) -> ValidationResult:
    """Run walk-forward validation.

    Args:
        df: Feature DataFrame with a DatetimeIndex.
        train_fn: ``model = train_fn(train_df)``
        predict_fn: ``predictions = predict_fn(model, test_df)``
        eval_fn: ``metrics = eval_fn(y_true, y_pred)`` → dict of scores
        target_col: Column name of the target variable.
        train_size: Minimum training observations.
        test_size: Test observations per fold.
        step_size: Slide step (defaults to *test_size*).
        expanding: Expanding or rolling window.

    Returns:
        ValidationResult with per-fold and aggregate metrics.
    """
    folds: List[FoldResult] = []

    for fold_i, (train_idx, test_idx) in enumerate(
        walk_forward_splits(df.index, train_size, test_size, step_size, expanding)
    ):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        model = train_fn(train_df)
        preds = predict_fn(model, test_df)
        y_true = test_df[target_col]
        metrics = eval_fn(y_true, preds)

        folds.append(FoldResult(
            fold_index=fold_i,
            train_start=df.index[train_idx[0]],
            train_end=df.index[train_idx[-1]],
            test_start=df.index[test_idx[0]],
            test_end=df.index[test_idx[-1]],
            train_size=len(train_idx),
            test_size=len(test_idx),
            metrics=metrics,
            predictions=preds,
        ))

        logger.info(
            f"Fold {fold_i}: train {df.index[train_idx[0]].date()} → "
            f"{df.index[train_idx[-1]].date()}, "
            f"test {df.index[test_idx[0]].date()} → "
            f"{df.index[test_idx[-1]].date()}, "
            f"metrics={metrics}"
        )

    return ValidationResult(folds=folds)


# ---------------------------------------------------------------------------
# Purged k-Fold Cross-Validation
# ---------------------------------------------------------------------------

def purged_kfold_splits(
    index: pd.DatetimeIndex,
    n_folds: int = 5,
    embargo_pct: float = 0.01,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """Generate purged k-fold splits for time-series data.

    Each fold is a contiguous time block.  An *embargo* buffer after each
    test set is excluded from the training set to prevent leakage from
    labels that span multiple observations.

    Args:
        index: DatetimeIndex of the dataset (must be sorted).
        n_folds: Number of folds.
        embargo_pct: Fraction of dataset to embargo after each test set.
    """
    n = len(index)
    embargo_size = int(n * embargo_pct)
    fold_size = n // n_folds

    for i in range(n_folds):
        test_start = i * fold_size
        test_end = min((i + 1) * fold_size, n)

        embargo_end = min(test_end + embargo_size, n)

        train_before = np.arange(0, test_start)
        train_after = np.arange(embargo_end, n)
        train_idx = np.concatenate([train_before, train_after])
        test_idx = np.arange(test_start, test_end)

        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        yield train_idx, test_idx


def purged_kfold_validate(
    df: pd.DataFrame,
    train_fn: Callable[[pd.DataFrame], Any],
    predict_fn: Callable[[Any, pd.DataFrame], pd.Series],
    eval_fn: Callable[[pd.Series, pd.Series], Dict[str, float]],
    target_col: str,
    n_folds: int = 5,
    embargo_pct: float = 0.01,
) -> ValidationResult:
    """Run purged k-fold cross-validation."""
    folds: List[FoldResult] = []

    for fold_i, (train_idx, test_idx) in enumerate(
        purged_kfold_splits(df.index, n_folds, embargo_pct)
    ):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        model = train_fn(train_df)
        preds = predict_fn(model, test_df)
        y_true = test_df[target_col]
        metrics = eval_fn(y_true, preds)

        folds.append(FoldResult(
            fold_index=fold_i,
            train_start=df.index[train_idx[0]],
            train_end=df.index[train_idx[-1]],
            test_start=df.index[test_idx[0]],
            test_end=df.index[test_idx[-1]],
            train_size=len(train_idx),
            test_size=len(test_idx),
            metrics=metrics,
            predictions=preds,
        ))

    return ValidationResult(folds=folds)
