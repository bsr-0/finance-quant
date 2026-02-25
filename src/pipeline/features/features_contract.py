"""Feature builders for prediction market contracts with strict as-of semantics."""

from __future__ import annotations

import pandas as pd


def build_contract_snapshot_asof(
    snapshots: pd.DataFrame,
    contract_id: str,
    asof_ts,
) -> pd.Series | None:
    """Return the latest snapshot for *contract_id* available as of *asof_ts*."""
    df = snapshots.copy()
    df = df[(df["contract_id"] == contract_id) & (df["available_time"] <= asof_ts)]
    if df.empty:
        return None
    df = df.sort_values("asof_ts")
    return df.iloc[-1]


def build_feature_matrix(
    snapshots: pd.DataFrame,
    asof_col: str = "asof_ts",
    available_col: str = "available_time",
) -> pd.DataFrame:
    """Build a leakage-safe feature matrix from snapshots.

    Drops rows where available_time is after asof_ts.
    """
    df = snapshots.copy()
    df = df[df[available_col] <= df[asof_col]]
    return df
