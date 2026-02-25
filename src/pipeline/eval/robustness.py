"""Statistical robustness utilities for backtest evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def deflated_sharpe_ratio(
    sharpe: float,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    benchmark_sr: float = 0.0,
) -> float:
    """Deflated Sharpe ratio probability (Bailey & Lopez de Prado).

    Returns probability that Sharpe exceeds benchmark_sr.
    """
    if n_obs <= 1 or not np.isfinite(sharpe):
        return np.nan
    denom = np.sqrt(1 - skew * sharpe + ((kurtosis - 1) / 4) * sharpe ** 2)
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
    """Bootstrap confidence interval for a metric computed on *series*."""
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
