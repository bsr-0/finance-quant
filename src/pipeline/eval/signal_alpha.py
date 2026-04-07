"""Walk-forward signal alpha significance testing.

Validates that a signal has genuine predictive power by computing rank
Information Coefficient (IC) out-of-sample across walk-forward folds,
then applying a deflated Sharpe gate to the IC series.

Usage::

    result = walk_forward_ic(signals_df, returns_df, signal_name="momentum")
    if result.passed:
        print("Signal has statistically significant alpha")

    # Full workflow from price data:
    fwd = compute_forward_returns(price_panel, horizon=5)
    result = walk_forward_ic(signals_df, fwd, signal_name="momentum")
    decay = ic_decay_analysis(signals_df, price_panel, horizons=[1, 5, 10, 21])
    rolling = rolling_ic(signals_df, fwd, window=63)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.stats import t as t_dist

from pipeline.backtesting.walk_forward import walk_forward_splits
from pipeline.eval.robustness import benjamini_hochberg, deflated_sharpe_ratio

logger = logging.getLogger(__name__)


@dataclass
class SignalAlphaResult:
    """Result of walk-forward IC significance test for a single signal."""

    signal_name: str
    ic_mean: float
    """Mean rank IC across out-of-sample folds."""
    ic_std: float
    """Standard deviation of per-fold IC values."""
    ic_t_stat: float
    """t-statistic: ic_mean / (ic_std / sqrt(n_folds))."""
    ic_p_value: float
    """Two-sided p-value from t-distribution."""
    deflated_sharpe_prob: float
    """Probability that IC Sharpe exceeds zero after deflation."""
    n_folds: int
    per_fold_ic: list[float] = field(default_factory=list)
    passed: bool = False
    """True if deflated_sharpe_prob > significance_threshold."""


@dataclass
class ICDecayResult:
    """IC measured at multiple forward-return horizons."""

    signal_name: str
    horizons: list[int]
    ic_by_horizon: dict[int, float]
    """Mean IC for each horizon (days)."""
    ic_std_by_horizon: dict[int, float]
    """IC standard deviation for each horizon."""
    ic_ir_by_horizon: dict[int, float]
    """IC information ratio (mean/std) per horizon."""
    best_horizon: int
    """Horizon with highest IC IR."""


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def compute_forward_returns(
    price_panel: pd.DataFrame,
    horizon: int = 5,
) -> pd.DataFrame:
    """Compute forward returns from a price panel for IC testing.

    Args:
        price_panel: DataFrame with DatetimeIndex rows and symbol columns,
            containing close prices.
        horizon: Number of trading days forward (e.g. 5 = 1 week).

    Returns:
        DataFrame of same shape with N-day forward returns.  The last
        ``horizon`` rows will be NaN (no future data available).
    """
    if price_panel.empty:
        return price_panel.copy()
    return price_panel.pct_change(horizon).shift(-horizon)


def rank_ic(signal: pd.Series, forward_returns: pd.Series) -> float:
    """Spearman rank correlation between signal values and forward returns.

    Args:
        signal: Cross-sectional signal values for one date.
        forward_returns: Corresponding forward returns.

    Returns:
        Spearman rank correlation coefficient, or NaN if insufficient data.
    """
    aligned_sig, aligned_ret = signal.align(forward_returns, join="inner")
    valid = aligned_sig.notna() & aligned_ret.notna()
    aligned_sig = aligned_sig[valid]
    aligned_ret = aligned_ret[valid]
    if len(aligned_sig) < 3:
        return np.nan
    corr, _ = spearmanr(aligned_sig.values, aligned_ret.values)
    return float(corr)


def rolling_ic(
    signals: pd.DataFrame,
    forward_returns: pd.DataFrame,
    window: int = 63,
) -> pd.Series:
    """Compute rolling cross-sectional rank IC over a sliding window.

    For each date, computes rank_ic across symbols, then smooths with
    a rolling mean.  Useful for monitoring whether a signal's predictive
    power is stable, improving, or decaying over time.

    Args:
        signals: Signal values, DatetimeIndex x symbols.
        forward_returns: Forward returns, same shape.
        window: Rolling window for smoothing daily IC values.

    Returns:
        Series of rolling mean IC indexed by date.
    """
    common_dates = signals.index.intersection(forward_returns.index).sort_values()
    daily_ics = []
    for dt in common_dates:
        ic_val = rank_ic(signals.loc[dt], forward_returns.loc[dt])
        daily_ics.append(ic_val)

    ic_series = pd.Series(daily_ics, index=common_dates, name="rolling_ic")
    return ic_series.rolling(window, min_periods=max(window // 4, 1)).mean()


def ic_decay_analysis(
    signals: pd.DataFrame,
    price_panel: pd.DataFrame,
    signal_name: str = "",
    horizons: list[int] | None = None,
) -> ICDecayResult:
    """Measure IC at multiple forward-return horizons to find optimal holding period.

    A signal with IC that peaks at 5 days and decays by 21 days is a
    short-term alpha.  One that peaks at 63 days is a slower factor.
    Knowing the decay curve is essential for choosing rebalance frequency
    and position sizing.

    Args:
        signals: Signal values, DatetimeIndex x symbols.
        price_panel: Close prices, DatetimeIndex x symbols (same symbols).
        signal_name: Human-readable identifier.
        horizons: List of forward-return horizons in trading days.
            Default: ``[1, 5, 10, 21, 63]``.

    Returns:
        ICDecayResult with IC statistics per horizon.
    """
    if horizons is None:
        horizons = [1, 5, 10, 21, 63]

    ic_by_horizon: dict[int, float] = {}
    ic_std_by_horizon: dict[int, float] = {}
    ic_ir_by_horizon: dict[int, float] = {}

    for h in horizons:
        fwd = compute_forward_returns(price_panel, horizon=h)
        common_dates = signals.index.intersection(fwd.index).sort_values()

        daily_ics: list[float] = []
        for dt in common_dates:
            ic_val = rank_ic(signals.loc[dt], fwd.loc[dt])
            if np.isfinite(ic_val):
                daily_ics.append(ic_val)

        if len(daily_ics) < 2:
            ic_by_horizon[h] = np.nan
            ic_std_by_horizon[h] = np.nan
            ic_ir_by_horizon[h] = np.nan
            continue

        ic_arr = np.array(daily_ics)
        mean_ic = float(ic_arr.mean())
        std_ic = float(ic_arr.std(ddof=1))
        ic_by_horizon[h] = mean_ic
        ic_std_by_horizon[h] = std_ic
        ic_ir_by_horizon[h] = mean_ic / std_ic if std_ic > 0 else 0.0

    # Best horizon = highest IC IR (most consistent predictive power)
    valid = {h: ir for h, ir in ic_ir_by_horizon.items() if np.isfinite(ir)}
    best_horizon = max(valid, key=valid.get) if valid else horizons[0]

    logger.info(
        "IC decay for '%s': %s  (best horizon=%dd)",
        signal_name,
        {h: f"{ic:.4f}" for h, ic in ic_by_horizon.items()},
        best_horizon,
    )

    return ICDecayResult(
        signal_name=signal_name,
        horizons=horizons,
        ic_by_horizon=ic_by_horizon,
        ic_std_by_horizon=ic_std_by_horizon,
        ic_ir_by_horizon=ic_ir_by_horizon,
        best_horizon=best_horizon,
    )


def walk_forward_ic(
    signals: pd.DataFrame,
    returns: pd.DataFrame,
    signal_name: str = "",
    train_size: int = 252,
    test_size: int = 63,
    embargo_size: int = 5,
    expanding: bool = True,
    significance_threshold: float = 0.95,
) -> SignalAlphaResult:
    """Compute rank IC across walk-forward OOS folds with deflated Sharpe gate.

    For each fold, computes the cross-sectional rank IC on each test date
    (Spearman correlation between signal rank and forward return across
    symbols), then averages to produce one IC value per fold.

    The per-fold IC series is treated as a "return stream" and tested via
    the deflated Sharpe ratio framework.

    Args:
        signals: Signal scores, DatetimeIndex x symbols.
        returns: Forward returns, same shape as signals.
        signal_name: Human-readable signal identifier.
        train_size: Training window size (observations).
        test_size: Test window size per fold.
        embargo_size: Gap between train and test (default 5).
        expanding: Expanding (True) or rolling (False) window.
        significance_threshold: Deflated Sharpe probability threshold for
            the signal to ``pass`` (default 0.95).

    Returns:
        SignalAlphaResult with IC statistics and pass/fail verdict.
    """
    common_dates = signals.index.intersection(returns.index)
    if len(common_dates) < train_size + test_size + embargo_size:
        logger.warning(
            "Insufficient data for walk-forward IC: %d dates, need %d",
            len(common_dates),
            train_size + test_size + embargo_size,
        )
        return SignalAlphaResult(
            signal_name=signal_name,
            ic_mean=np.nan,
            ic_std=np.nan,
            ic_t_stat=np.nan,
            ic_p_value=np.nan,
            deflated_sharpe_prob=np.nan,
            n_folds=0,
            passed=False,
        )

    signals_aligned = signals.loc[common_dates]
    returns_aligned = returns.loc[common_dates]

    per_fold_ic: list[float] = []
    for _train_idx, test_idx in walk_forward_splits(
        common_dates,
        train_size,
        test_size,
        embargo_size=embargo_size,
        expanding=expanding,
    ):
        test_dates = common_dates[test_idx]
        daily_ics: list[float] = []
        for dt in test_dates:
            sig_row = signals_aligned.loc[dt]
            ret_row = returns_aligned.loc[dt]
            ic_val = rank_ic(sig_row, ret_row)
            if np.isfinite(ic_val):
                daily_ics.append(ic_val)

        if daily_ics:
            per_fold_ic.append(float(np.mean(daily_ics)))

    n_folds = len(per_fold_ic)
    if n_folds < 2:
        return SignalAlphaResult(
            signal_name=signal_name,
            ic_mean=np.nan,
            ic_std=np.nan,
            ic_t_stat=np.nan,
            ic_p_value=np.nan,
            deflated_sharpe_prob=np.nan,
            n_folds=n_folds,
            per_fold_ic=per_fold_ic,
            passed=False,
        )

    ic_arr = np.array(per_fold_ic)
    ic_mean = float(ic_arr.mean())
    ic_std = float(ic_arr.std(ddof=1))

    if ic_std == 0:
        ic_t_stat = np.inf if ic_mean > 0 else -np.inf if ic_mean < 0 else 0.0
        ic_p_value = 0.0 if ic_mean != 0 else 1.0
    else:
        ic_t_stat = float(ic_mean / (ic_std / np.sqrt(n_folds)))
        ic_p_value = float(2 * t_dist.sf(abs(ic_t_stat), df=n_folds - 1))

    # Treat IC series as a "return stream" and apply deflated Sharpe
    ic_sharpe = ic_mean / ic_std if ic_std > 0 else 0.0
    skew = float(pd.Series(ic_arr).skew()) if n_folds >= 3 else 0.0
    excess_kurt = float(pd.Series(ic_arr).kurtosis()) if n_folds >= 4 else 0.0

    dsr_prob = deflated_sharpe_ratio(
        sharpe=ic_sharpe,
        n_obs=n_folds,
        skew=skew,
        excess_kurtosis=excess_kurt,
    )

    passed = bool(np.isfinite(dsr_prob) and dsr_prob > significance_threshold)

    logger.info(
        "Signal '%s': IC mean=%.4f, std=%.4f, t=%.2f, p=%.4f, DSR=%.3f, passed=%s",
        signal_name,
        ic_mean,
        ic_std,
        ic_t_stat,
        ic_p_value,
        dsr_prob,
        passed,
    )

    return SignalAlphaResult(
        signal_name=signal_name,
        ic_mean=ic_mean,
        ic_std=ic_std,
        ic_t_stat=ic_t_stat,
        ic_p_value=ic_p_value,
        deflated_sharpe_prob=float(dsr_prob),
        n_folds=n_folds,
        per_fold_ic=per_fold_ic,
        passed=passed,
    )


def signal_fdr_screen(
    alpha_results: list[SignalAlphaResult],
    alpha: float = 0.05,
) -> list[tuple[SignalAlphaResult, bool]]:
    """Screen multiple signal candidates using Benjamini-Hochberg FDR control.

    Args:
        alpha_results: Results from :func:`walk_forward_ic` for each signal.
        alpha: Target false discovery rate.

    Returns:
        List of ``(result, is_significant)`` tuples.
    """
    pvals = [r.ic_p_value for r in alpha_results]
    # Replace NaN p-values with 1.0 (not significant)
    pvals = [p if np.isfinite(p) else 1.0 for p in pvals]
    rejected = benjamini_hochberg(pvals, alpha)
    return list(zip(alpha_results, rejected, strict=True))
