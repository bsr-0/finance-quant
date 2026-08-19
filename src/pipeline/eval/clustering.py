"""Correlated-signal clustering for trade-level statistics.

A signal universe of sector ETFs is not 17 independent draws: XLU, XLP, and
XLRE move together, as do SPY, QQQ, and XLK. Nominal sample size overstates
statistical power for any trade-level statistic (win rate, mean P&L) that
doesn't already collapse to one observation per date the way ``rank_ic`` does.
This module estimates how much smaller the *effective* sample really is and
provides a cluster-aware bootstrap so confidence intervals reflect it.

The 0.70 correlation threshold mirrors ``StrategyConfig.max_correlation``
(``src/pipeline/strategy/engine.py:89``), so evaluation and the (currently
unused) production risk constraint share one definition of "too correlated to
treat as independent."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_CORRELATION_THRESHOLD = 0.70


def cluster_by_correlation(
    returns: pd.DataFrame,
    threshold: float = DEFAULT_CORRELATION_THRESHOLD,
    lookback: int | None = None,
) -> dict[str, int]:
    """Group symbols into clusters via single-linkage on return correlation.

    Args:
        returns: Daily returns, DatetimeIndex x symbols.
        threshold: Symbols are merged into a cluster while the correlation
            between their cluster's max-linkage distance is at least this.
        lookback: If given, only the last *lookback* rows are used.

    Returns:
        ``{symbol: cluster_id}``. Symbols with no return data of their own
        (e.g. absent from *returns*) are not included.
    """
    data = returns.tail(lookback) if lookback else returns
    data = data.dropna(axis=1, how="all")
    symbols = list(data.columns)
    if len(symbols) <= 1:
        return {s: i for i, s in enumerate(symbols)}

    corr = data.corr(min_periods=max(5, len(data) // 4))

    # Single-linkage via union-find: merge any pair above threshold, then any
    # pair connected through a chain of such merges lands in the same cluster.
    parent = {s: s for s in symbols}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, a in enumerate(symbols):
        for b in symbols[i + 1 :]:
            c = corr.loc[a, b]
            if pd.notna(c) and c >= threshold:
                union(a, b)

    roots = {s: find(s) for s in symbols}
    root_to_id = {root: i for i, root in enumerate(sorted(set(roots.values())))}
    return {s: root_to_id[root] for s, root in roots.items()}


@dataclass(frozen=True)
class EffectiveSampleSize:
    """Two independent estimates of how many *independent* observations a
    trade-level sample really contains, given date and cross-sectional
    clustering."""

    n_nominal: int
    n_clusters_used: int
    design_effect_n: float
    """n / (1 + (mean_group_size - 1) * icc): shrinks n by the average
    correlation within (date, cluster) groups."""
    declustered_n: int
    """Number of (date, cluster) groups after collapsing each to one
    observation. The conservative headline number."""


def _intraclass_correlation(values: pd.Series, groups: pd.Series) -> float:
    """One-way random-effects ICC: between-group variance share of total."""
    df = pd.DataFrame({"v": values, "g": groups}).dropna()
    if df["g"].nunique() < 2 or len(df) < 3:
        return 0.0
    grand_mean = df["v"].mean()
    group_stats = df.groupby("g")["v"].agg(["mean", "count"])
    k = df["g"].nunique()
    n = len(df)
    mean_group_size = n / k

    ss_between = float((group_stats["count"] * (group_stats["mean"] - grand_mean) ** 2).sum())
    ss_within = float(df.groupby("g")["v"].apply(lambda g: ((g - g.mean()) ** 2).sum()).sum())
    df_between = k - 1
    df_within = n - k
    if df_between <= 0 or df_within <= 0:
        return 0.0

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within if df_within > 0 else 0.0
    if mean_group_size <= 1 or ms_between <= 0:
        return 0.0

    icc = (ms_between - ms_within) / (ms_between + (mean_group_size - 1) * ms_within)
    return float(np.clip(icc, 0.0, 1.0))


def effective_sample_size(
    records: pd.DataFrame,
    cluster_map: dict[str, int],
    date_col: str = "signal_date",
    symbol_col: str = "ticker",
    value_col: str = "pnl_pct",
) -> EffectiveSampleSize:
    """Estimate the effective independent sample size of trade-level records.

    Args:
        records: One row per trade, with a date, a symbol, and a numeric
            outcome (e.g. pnl_pct).
        cluster_map: ``{symbol: cluster_id}`` from ``cluster_by_correlation``.
        date_col, symbol_col, value_col: Column names.
    """
    df = records[[date_col, symbol_col, value_col]].dropna().copy()
    n_nominal = len(df)
    if n_nominal == 0:
        return EffectiveSampleSize(0, 0, 0.0, 0)

    df["cluster"] = df[symbol_col].map(cluster_map)
    df["cluster"] = df["cluster"].fillna(-1).astype(int)
    df["group"] = list(zip(df[date_col], df["cluster"], strict=True))

    n_groups = df["group"].nunique()
    group_sizes = df.groupby("group").size()
    mean_group_size = float(group_sizes.mean())

    icc = _intraclass_correlation(df[value_col], df["group"].astype(str))
    design_effect = 1 + (mean_group_size - 1) * icc
    design_effect_n = n_nominal / design_effect if design_effect > 0 else float(n_nominal)

    return EffectiveSampleSize(
        n_nominal=n_nominal,
        n_clusters_used=df["cluster"].nunique(),
        design_effect_n=float(design_effect_n),
        declustered_n=int(n_groups),
    )


def declustered_replicate(
    records: pd.DataFrame,
    cluster_map: dict[str, int],
    date_col: str = "signal_date",
    symbol_col: str = "ticker",
    value_col: str = "pnl_pct",
) -> pd.Series:
    """Collapse each (date, cluster) group to its mean value.

    This is the conservative statistic to headline: it treats correlated,
    same-day signals as one observation rather than several.
    """
    df = records[[date_col, symbol_col, value_col]].dropna().copy()
    df["cluster"] = df[symbol_col].map(cluster_map).fillna(-1).astype(int)
    return df.groupby([date_col, "cluster"])[value_col].mean()


def cluster_bootstrap(
    records: pd.DataFrame,
    cluster_map: dict[str, int],
    statistic,
    date_col: str = "signal_date",
    symbol_col: str = "ticker",
    value_col: str = "pnl_pct",
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap CI resampling whole (date, cluster) blocks rather than rows.

    Generalizes a plain date-clustered bootstrap to also absorb cross-sectional
    correlation: resampling individual trades would treat six same-day,
    same-sector signals as six independent draws when they're closer to one.

    Args:
        statistic: Function taking a 1-D numpy array of values, returning a
            scalar (e.g. ``np.mean``).
    """
    df = records[[date_col, symbol_col, value_col]].dropna().copy()
    df["cluster"] = df[symbol_col].map(cluster_map).fillna(-1).astype(int)
    df["group"] = list(zip(df[date_col], df["cluster"], strict=True))

    groups = df.groupby("group")[value_col].apply(lambda g: g.to_numpy())
    group_keys = list(groups.index)
    if not group_keys:
        return (np.nan, np.nan)

    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        chosen = rng.choice(len(group_keys), size=len(group_keys), replace=True)
        sample = np.concatenate([groups.iloc[i] for i in chosen])
        stats.append(statistic(sample))

    lo = np.percentile(stats, alpha / 2 * 100)
    hi = np.percentile(stats, (1 - alpha / 2) * 100)
    return float(lo), float(hi)
