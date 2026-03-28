"""Statistical robustness utilities for backtest evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def deflated_sharpe_ratio(
    sharpe: float,
    n_obs: int,
    skew: float = 0.0,
    excess_kurtosis: float = 0.0,
    benchmark_sr: float = 0.0,
    *,
    kurtosis: float | None = None,
) -> float:
    """Deflated Sharpe ratio probability (Bailey & Lopez de Prado).

    Returns probability that Sharpe exceeds benchmark_sr.

    Args:
        sharpe: Annualized Sharpe ratio.
        n_obs: Number of observations.
        skew: Sample skewness.
        excess_kurtosis: Excess kurtosis (pandas default). 0.0 = normal.
        benchmark_sr: Benchmark Sharpe to test against.
        kurtosis: Deprecated — use *excess_kurtosis*. If provided and
            excess_kurtosis is at default, raw kurtosis is converted.
    """
    if n_obs <= 1 or not np.isfinite(sharpe):
        return np.nan
    # Handle legacy callers passing raw kurtosis via the old parameter
    if kurtosis is not None and excess_kurtosis == 0.0:
        excess_kurtosis = kurtosis - 3.0
    raw_kurtosis = excess_kurtosis + 3.0
    denom = np.sqrt(1 - skew * sharpe + ((raw_kurtosis - 1) / 4) * sharpe**2)
    if denom == 0:
        return np.nan
    z = (sharpe - benchmark_sr) * np.sqrt(n_obs - 1) / denom
    return float(norm.cdf(z))


def bootstrap_ci(
    series: pd.Series,
    metric_fn,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """IID bootstrap confidence interval for a metric computed on *series*.

    .. warning:: This uses IID resampling. For time-series data with
       serial correlation (e.g. financial returns), use
       :func:`block_bootstrap_ci` instead.
    """
    series = series.dropna()
    if series.empty:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    values = series.values
    stats = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        stats.append(metric_fn(pd.Series(sample)))
    lo = np.percentile(stats, alpha * 100)
    hi = np.percentile(stats, (1 - alpha) * 100)
    return float(lo), float(hi)


def block_bootstrap_ci(
    series: pd.Series,
    metric_fn,
    block_size: int = 21,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Block bootstrap confidence interval preserving serial dependence.

    Resamples contiguous blocks of *block_size* observations to maintain
    the autocorrelation structure of the original series.  This is the
    appropriate bootstrap method for financial return series.

    Args:
        series: Time-series data (e.g. daily returns).
        metric_fn: Function that takes a pd.Series and returns a scalar.
        block_size: Number of contiguous observations per block.
        n_boot: Number of bootstrap replications.
        alpha: Significance level (two-sided).
        seed: Random seed.

    Returns:
        (lower_ci, upper_ci) at the (1 - alpha) confidence level.
    """
    series = series.dropna()
    n = len(series)
    if n < block_size:
        return bootstrap_ci(series, metric_fn, n_boot, alpha, seed)
    rng = np.random.default_rng(seed)
    values = series.values
    n_blocks = (n + block_size - 1) // block_size
    stats = []
    for _ in range(n_boot):
        blocks = []
        for _ in range(n_blocks):
            start = rng.integers(0, n - block_size + 1)
            blocks.append(values[start : start + block_size])
        sample = np.concatenate(blocks)[:n]
        stats.append(metric_fn(pd.Series(sample)))
    lo = np.percentile(stats, alpha * 100)
    hi = np.percentile(stats, (1 - alpha) * 100)
    return float(lo), float(hi)


def holm_bonferroni(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni correction for multiple testing."""
    m = len(pvals)
    if m == 0:
        return []
    sorted_idx = np.argsort(pvals)
    adjusted = [0.0] * m
    for i, idx in enumerate(sorted_idx):
        adjusted[idx] = min(1.0, (m - i) * pvals[idx])
    return adjusted


def benjamini_hochberg(pvals: list[float], alpha: float = 0.05) -> list[bool]:
    """Benjamini-Hochberg FDR control for multiple hypothesis testing.

    Controls the expected proportion of false discoveries among rejected
    hypotheses.  More powerful than Holm-Bonferroni (FWER) when many
    hypotheses are tested simultaneously.

    Args:
        pvals: Raw p-values, one per hypothesis.
        alpha: Target false discovery rate.

    Returns:
        List of booleans: ``True`` = reject null (signal is significant).
    """
    m = len(pvals)
    if m == 0:
        return []
    sorted_idx = list(np.argsort(pvals))
    rejected = [False] * m
    # Find largest k such that p_(k) <= k/m * alpha
    max_k = -1
    for i, idx in enumerate(sorted_idx):
        rank = i + 1
        if pvals[idx] <= rank / m * alpha:
            max_k = i
    # Reject all hypotheses up to and including max_k
    if max_k >= 0:
        for i in range(max_k + 1):
            rejected[sorted_idx[i]] = True
    return rejected


def probability_of_backtest_overfitting(train_scores: pd.Series, test_scores: pd.Series) -> float:
    """Estimate probability of backtest overfitting (PBO).

    Simple proxy: fraction where top-quantile in-sample underperforms median out-of-sample.
    """
    train_scores, test_scores = train_scores.align(test_scores, join="inner")
    if train_scores.empty:
        return np.nan
    threshold = train_scores.quantile(0.9)
    top = test_scores[train_scores >= threshold]
    if top.empty:
        return np.nan
    return float((top < test_scores.median()).mean())
