"""Market regime classification utilities."""

from __future__ import annotations

import pandas as pd
from scipy.stats import kruskal


def classify_regimes(prices: pd.Series) -> pd.Series:
    """Classify regimes based on 200D MA and drawdown thresholds."""
    prices = prices.dropna()
    ma_200 = prices.rolling(200, min_periods=50).mean()
    peak = prices.cummax()
    drawdown = (prices - peak) / peak.replace(0, float("nan"))

    regime = pd.Series(index=prices.index, dtype="object")
    regime[(prices > ma_200) & (drawdown > -0.10)] = "bull"
    regime[(prices < ma_200) | (drawdown < -0.20)] = "bear"
    regime[regime.isna()] = "flat"
    return regime


def regime_performance(returns: pd.Series, regimes: pd.Series) -> dict[str, dict[str, float]]:
    """Compute mean/vol/sharpe per regime with significance test.

    Uses the Kruskal-Wallis H-test (non-parametric) to determine
    whether returns across regimes are drawn from statistically
    distinct distributions.  This avoids the normality assumption
    that would be required by ANOVA.
    """
    out: dict[str, dict[str, float]] = {}
    aligned_returns, aligned_regimes = returns.align(regimes, join="inner")

    # Collect regime subsets for significance test
    regime_samples = []
    for name in ["bull", "bear", "flat"]:
        subset = aligned_returns[aligned_regimes == name].dropna()
        if subset.empty:
            out[name] = {"mean": float("nan"), "vol": float("nan"), "sharpe": float("nan")}
            continue
        mean = subset.mean() * 252
        vol = subset.std() * (252 ** 0.5)
        sharpe = mean / vol if vol != 0 else float("nan")
        out[name] = {"mean": float(mean), "vol": float(vol), "sharpe": float(sharpe)}
        if len(subset) >= 5:
            regime_samples.append(subset.values)

    # Test whether regimes are statistically distinct
    if len(regime_samples) >= 2:
        stat, p_value = kruskal(*regime_samples)
        out["significance"] = {
            "kruskal_wallis_stat": float(stat),
            "p_value": float(p_value),
            "regimes_are_distinct": float(p_value < 0.05),
        }
    else:
        out["significance"] = {
            "kruskal_wallis_stat": float("nan"),
            "p_value": float("nan"),
            "regimes_are_distinct": 0.0,
        }

    return out
