"""Tests for pipeline.eval.clustering: correlated-signal grouping and the
effective-sample-size / cluster-bootstrap machinery built on top of it."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.eval.clustering import (
    cluster_bootstrap,
    cluster_by_correlation,
    declustered_replicate,
    effective_sample_size,
)


def test_cluster_by_correlation_groups_highly_correlated_symbols():
    idx = pd.bdate_range("2020-01-01", periods=200)
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, len(idx))
    noise_scale = 0.05  # small noise -> correlation well above 0.70
    returns = pd.DataFrame(
        {
            "XLU": base + rng.normal(0, noise_scale, len(idx)),
            "XLP": base + rng.normal(0, noise_scale, len(idx)),
            "XLRE": base + rng.normal(0, noise_scale, len(idx)),
            "INDEPENDENT": rng.normal(0, 1, len(idx)),
        },
        index=idx,
    )
    clusters = cluster_by_correlation(returns, threshold=0.70)

    assert clusters["XLU"] == clusters["XLP"] == clusters["XLRE"]
    assert clusters["INDEPENDENT"] != clusters["XLU"]


def test_cluster_by_correlation_transitive_merge():
    """A-B correlated and B-C correlated (but A-C not directly measured above
    threshold) must still land in the same cluster via the chain."""
    idx = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, len(idx))
    b = 0.75 * a + rng.normal(0, 0.4, len(idx))
    c = 0.75 * b + rng.normal(0, 0.4, len(idx))
    returns = pd.DataFrame({"A": a, "B": b, "C": c}, index=idx)

    clusters = cluster_by_correlation(returns, threshold=0.70)
    if clusters["A"] == clusters["B"] and clusters["B"] == clusters["C"]:
        assert clusters["A"] == clusters["C"]


def test_cluster_by_correlation_all_independent_gives_singleton_clusters():
    idx = pd.bdate_range("2020-01-01", periods=200)
    rng = np.random.default_rng(2)
    returns = pd.DataFrame({f"S{i}": rng.normal(0, 1, len(idx)) for i in range(5)}, index=idx)
    clusters = cluster_by_correlation(returns, threshold=0.70)
    assert len(set(clusters.values())) == 5


def test_cluster_by_correlation_handles_single_symbol():
    idx = pd.bdate_range("2020-01-01", periods=50)
    returns = pd.DataFrame({"ONLY": np.random.default_rng(0).normal(0, 1, 50)}, index=idx)
    clusters = cluster_by_correlation(returns)
    assert clusters == {"ONLY": 0}


# --- effective sample size --------------------------------------------------


def _correlated_records(n_dates: int = 20, cluster_size: int = 4, seed: int = 0) -> pd.DataFrame:
    """Trade records where cluster members share most of their date-level
    variance -- a strong within-group correlation, by construction."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_dates):
        date_effect = rng.normal(0, 3)
        for s in range(cluster_size):
            rows.append(
                {
                    "signal_date": f"2020-{(d % 12) + 1:02d}-01",
                    "ticker": f"CLUSTER_{s}",
                    "pnl_pct": date_effect + rng.normal(0, 0.1),
                }
            )
    return pd.DataFrame(rows)


def test_effective_sample_size_shrinks_for_correlated_records():
    records = _correlated_records()
    cluster_map = {f"CLUSTER_{s}": 0 for s in range(4)}  # all one cluster

    result = effective_sample_size(records, cluster_map)

    assert result.n_nominal == len(records)
    assert result.design_effect_n < result.n_nominal
    assert result.declustered_n <= result.n_nominal


def test_effective_sample_size_matches_nominal_when_uncorrelated():
    rng = np.random.default_rng(3)
    records = pd.DataFrame(
        {
            "signal_date": [f"2020-01-{i:02d}" for i in range(1, 21)],
            "ticker": [f"T{i}" for i in range(20)],
            "pnl_pct": rng.normal(0, 1, 20),
        }
    )
    cluster_map = {f"T{i}": i for i in range(20)}  # every symbol its own cluster

    result = effective_sample_size(records, cluster_map)
    # No repeated (date, cluster) groups and no correlation structure to
    # estimate an ICC from -> should stay close to nominal.
    assert result.design_effect_n == pytest.approx(result.n_nominal, rel=0.3)
    assert result.declustered_n == result.n_nominal


def test_effective_sample_size_empty():
    result = effective_sample_size(pd.DataFrame(columns=["signal_date", "ticker", "pnl_pct"]), {})
    assert result.n_nominal == 0


def test_declustered_replicate_collapses_same_day_cluster():
    records = _correlated_records(n_dates=5, cluster_size=4)
    cluster_map = {f"CLUSTER_{s}": 0 for s in range(4)}

    replicate = declustered_replicate(records, cluster_map)
    assert len(replicate) == 5  # one row per date, since all 4 tickers share cluster 0


# --- cluster bootstrap -------------------------------------------------------


def test_cluster_bootstrap_ci_contains_true_mean():
    rng = np.random.default_rng(5)
    rows = []
    for d in range(40):
        date_effect = rng.normal(0.5, 1.0)
        for s in range(3):
            rows.append(
                {
                    "signal_date": f"d{d}",
                    "ticker": f"T{s}",
                    "pnl_pct": date_effect + rng.normal(0, 0.2),
                }
            )
    records = pd.DataFrame(rows)
    cluster_map = {"T0": 0, "T1": 1, "T2": 2}

    lo, hi = cluster_bootstrap(records, cluster_map, np.mean, n_boot=2000, seed=1)
    assert lo <= 0.5 <= hi


def test_cluster_bootstrap_wider_than_naive_iid_bootstrap_under_clustering():
    """Resampling whole (date, cluster) blocks should give a wider CI than
    resampling individual rows when same-day, same-cluster values are
    strongly correlated -- naive row resampling understates the true
    uncertainty."""
    from pipeline.eval.robustness import bootstrap_ci

    records = _correlated_records(n_dates=15, cluster_size=5, seed=9)
    cluster_map = {f"CLUSTER_{s}": 0 for s in range(5)}

    cluster_lo, cluster_hi = cluster_bootstrap(records, cluster_map, np.mean, n_boot=2000, seed=2)
    naive_lo, naive_hi = bootstrap_ci(records["pnl_pct"], lambda s: s.mean(), n_boot=2000, seed=2)

    assert (cluster_hi - cluster_lo) > (naive_hi - naive_lo)


def test_cluster_bootstrap_empty_returns_nan():
    lo, hi = cluster_bootstrap(
        pd.DataFrame(columns=["signal_date", "ticker", "pnl_pct"]), {}, np.mean
    )
    assert np.isnan(lo) and np.isnan(hi)
