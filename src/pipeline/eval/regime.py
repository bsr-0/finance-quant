"""Market regime classification utilities."""

from __future__ import annotations

import pandas as pd


def classify_regimes(prices: pd.Series) -> pd.Series:
    """Classify regimes based on 200D MA and drawdown thresholds."""
    prices = prices.dropna()
    ma_200 = prices.rolling(200, min_periods=50).mean()
    peak = prices.cummax()
    drawdown = (prices - peak) / peak

    regime = pd.Series(index=prices.index, dtype="object")
    regime[(prices > ma_200) & (drawdown > -0.10)] = "bull"
    regime[(prices < ma_200) | (drawdown < -0.20)] = "bear"
    regime[regime.isna()] = "flat"
    return regime


def regime_performance(returns: pd.Series, regimes: pd.Series) -> dict[str, dict[str, float]]:
    """Compute mean/vol/sharpe per regime."""
    out: dict[str, dict[str, float]] = {}
    aligned_returns, aligned_regimes = returns.align(regimes, join="inner")
    for name in ["bull", "bear", "flat"]:
        subset = aligned_returns[aligned_regimes == name].dropna()
        if subset.empty:
            out[name] = {"mean": float("nan"), "vol": float("nan"), "sharpe": float("nan")}
            continue
        mean = subset.mean() * 252
        vol = subset.std() * (252 ** 0.5)
        sharpe = mean / vol if vol != 0 else float("nan")
        out[name] = {"mean": float(mean), "vol": float(vol), "sharpe": float(sharpe)}
    return out
