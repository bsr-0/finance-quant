"""Utilities to align features/labels with as-of semantics."""

from __future__ import annotations

import pandas as pd


def feature_asof(
    df: pd.DataFrame,
    horizon: int = 1,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Shift features forward by *horizon* to prevent leakage.

    Returns a DataFrame with features lagged by horizon periods.
    """
    feature_cols = feature_cols or list(df.columns)
    features = df[feature_cols].copy()
    return features.shift(horizon)


def align_features_labels(
    df: pd.DataFrame,
    label_col: str,
    horizon: int = 1,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Return a leakage-safe DataFrame with lagged features and aligned labels."""
    features = feature_asof(df, horizon=horizon, feature_cols=feature_cols)
    out = pd.concat([features, df[[label_col]]], axis=1)
    return out.dropna()


def filter_available_asof(
    df: pd.DataFrame,
    asof_ts,
    available_col: str = "available_time",
) -> pd.DataFrame:
    """Filter rows with available_time <= asof_ts."""
    return df[df[available_col] <= asof_ts]
