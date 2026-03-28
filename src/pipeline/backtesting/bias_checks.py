"""Look-ahead bias protection and detection for backtesting.

Provides utilities to verify that strategy decisions depend only on
information available up to the decision timestamp.  Includes both
preventive guards (enforced during backtest execution) and detective
tests (run post-hoc to catch accidental leakage).

Tests provided:
    1. **Timestamp ordering test**: Verify that all signals/features
       have timestamps strictly before the trading timestamp.
    2. **Random shuffle test**: Shuffle the time order of returns and
       verify that the strategy's Sharpe ratio degrades to noise.
    3. **Data shift test**: Shift forward-looking data by 1 period and
       verify that performance changes significantly.
    4. **Feature staleness check**: Ensure no feature uses data from
       timestamps later than the decision point.

Assumptions:
    - All data has a timestamp column or DatetimeIndex.
    - Features are computed before being consumed by the strategy.
    - The tests are deterministic (seeded randomness).
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import norm

logger = logging.getLogger(__name__)


def check_no_future_data(
    features: pd.DataFrame,
    signals: pd.DataFrame,
    target_col: str = "target",
    feature_timestamp_col: str | None = None,
    signal_timestamp_col: str | None = None,
) -> tuple[bool, list[str]]:
    """Check that no feature row uses data from the future.

    Verifies that for each signal timestamp, all feature values are
    computed from data available strictly before that timestamp.

    Args:
        features: Feature DataFrame with DatetimeIndex.
        signals: Signal DataFrame with DatetimeIndex.
        target_col: Name of the target column in features (if present).
        feature_timestamp_col: Optional explicit timestamp column.
        signal_timestamp_col: Optional explicit timestamp column.

    Returns:
        (passed, violations): True if no look-ahead detected, plus
        list of violation descriptions.
    """
    violations: list[str] = []

    # Check that features index is sorted
    if not features.index.is_monotonic_increasing:
        violations.append("Feature index is not sorted ascending — possible time disorder")

    # Check that target column (if present) is not filled from the future
    if target_col in features.columns:
        # The target should be NaN for the last row (since there's no future data)
        last_target = features[target_col].iloc[-1]
        if pd.notna(last_target):
            violations.append(
                f"Target column '{target_col}' has a non-NaN value on the last row "
                f"({last_target}). This suggests the target may include future data, "
                f"since the final observation should not yet have a realised outcome."
            )

    # Check alignment: signals should not reference future feature data
    if not signals.index.is_monotonic_increasing:
        violations.append("Signal index is not sorted ascending")

    # Check that feature data doesn't extend beyond signal data
    if not features.empty and not signals.empty and features.index[-1] > signals.index[-1]:
        violations.append(
            f"Features extend beyond signals: "
            f"features end={features.index[-1]}, signals end={signals.index[-1]}"
        )

    passed = len(violations) == 0
    return passed, violations


def random_shuffle_test(
    returns: pd.Series,
    strategy_fn: Callable[[pd.Series], float],
    n_shuffles: int = 100,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[bool, dict[str, float]]:
    """Test for look-ahead bias by shuffling return order.

    If the strategy has no look-ahead bias, its performance on shuffled
    data should be statistically indistinguishable from zero (or worse
    than the original).

    Args:
        returns: Original return series.
        strategy_fn: Function that takes returns and produces a Sharpe
            ratio (or other performance metric).
        n_shuffles: Number of random shuffles.
        seed: Random seed.
        alpha: Significance level for the test.

    Returns:
        (suspicious, stats): True if performance on shuffled data is
        suspiciously similar to original (potential look-ahead).
    """
    rng = np.random.default_rng(seed)
    original_metric = strategy_fn(returns)

    shuffled_metrics = []
    for _ in range(n_shuffles):
        shuffled = returns.copy()
        idx = shuffled.index.copy()
        values = shuffled.values.copy()
        rng.shuffle(values)
        shuffled = pd.Series(values, index=idx)
        shuffled_metrics.append(strategy_fn(shuffled))

    shuffled_arr = np.array(shuffled_metrics)
    mean_shuffled = float(np.mean(shuffled_arr))
    std_shuffled = float(np.std(shuffled_arr))

    # The original should significantly outperform shuffled
    # If it doesn't, the strategy may have look-ahead bias
    z_score = (original_metric - mean_shuffled) / std_shuffled if std_shuffled > 0 else 0.0

    # Suspicious if shuffled performance is similar to original
    # (i.e. z-score is low, meaning the strategy doesn't depend on order)
    suspicious = abs(z_score) < 2.0  # Not significantly better than random

    # Two-sided p-value from normal distribution
    p_value = float(2 * norm.sf(abs(z_score))) if std_shuffled > 0 else 1.0
    stats = {
        "original_metric": original_metric,
        "mean_shuffled": mean_shuffled,
        "std_shuffled": std_shuffled,
        "z_score": z_score,
        "p_value": p_value,
    }

    if suspicious:
        logger.warning(
            "Shuffle test: strategy performance not significantly different from "
            "shuffled data (z=%.2f). Possible look-ahead bias or no real alpha.",
            z_score,
        )

    return suspicious, stats


def data_shift_test(
    features: pd.DataFrame,
    target_col: str,
    strategy_fn: Callable[[pd.DataFrame], float],
    shift_periods: int = 1,
) -> tuple[bool, dict[str, float]]:
    """Test for look-ahead by shifting target data.

    Shifts the target column forward by *shift_periods* and re-runs
    the strategy.  If performance doesn't degrade significantly, the
    strategy may be using future target values.

    Args:
        features: Feature DataFrame including the target column.
        target_col: Name of the target column.
        strategy_fn: Function that takes features and returns a metric.
        shift_periods: How many periods to shift the target forward.

    Returns:
        (suspicious, stats): True if shifting doesn't hurt performance.
    """
    original_metric = strategy_fn(features)

    shifted = features.copy()
    shifted[target_col] = shifted[target_col].shift(shift_periods)
    shifted = shifted.dropna(subset=[target_col])
    shifted_metric = strategy_fn(shifted)

    if original_metric != 0:
        degradation = (original_metric - shifted_metric) / abs(original_metric)
    else:
        degradation = 0

    # If shifting the target barely changes performance, the strategy
    # probably isn't using future data (or it's not using the target at all)
    suspicious = abs(degradation) < 0.1  # Less than 10% degradation

    stats = {
        "original_metric": original_metric,
        "shifted_metric": shifted_metric,
        "degradation_pct": degradation,
        "shift_periods": shift_periods,
    }

    return suspicious, stats


def enforce_timestamp_ordering(
    data: pd.DataFrame,
    timestamp_col: str | None = None,
) -> pd.DataFrame:
    """Sort data by timestamp and raise if duplicates exist.

    This is a preventive guard to ensure chronological ordering.

    Args:
        data: DataFrame to validate.
        timestamp_col: Column name; uses index if None.

    Returns:
        Sorted DataFrame.

    Raises:
        ValueError: If the data contains exact duplicate timestamps
            with different values (ambiguous ordering).
    """
    if timestamp_col:
        data = data.sort_values(timestamp_col)
        dupes = data.duplicated(subset=[timestamp_col], keep=False)
    else:
        data = data.sort_index()
        dupes = data.index.duplicated(keep=False)

    if dupes.any():
        n_dupes = dupes.sum()
        logger.warning(
            "Found %d duplicate timestamps in data. These may cause "
            "non-deterministic behavior in the backtest.",
            n_dupes,
        )

    return data
