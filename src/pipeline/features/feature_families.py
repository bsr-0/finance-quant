"""Feature families for Agent Directive V7 Section 6.

Implements the four missing feature families required by the directive:
- Seasonal/calendar features
- Hierarchical features (group-level aggregates)
- Interaction features (cross-feature products and ratios)
- Representation features (leakage-safe target encoding)

Also provides automated feature elimination via ``select_features()``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Seasonal / Calendar Features (V7 Section 6 — temporal/seasonal family)
# ---------------------------------------------------------------------------


class SeasonalFeatures:
    """Generate calendar and seasonal features from a datetime index."""

    @staticmethod
    def from_index(df: pd.DataFrame) -> pd.DataFrame:
        """Extract all seasonal features from the DataFrame's DatetimeIndex.

        Parameters
        ----------
        df : pd.DataFrame
            Must have a DatetimeIndex (or a column castable to datetime).

        Returns
        -------
        pd.DataFrame
            Seasonal feature columns aligned to the input index.
        """
        idx = pd.DatetimeIndex(df.index)
        out = pd.DataFrame(index=df.index)
        out["day_of_week"] = idx.dayofweek
        out["month"] = idx.month
        out["quarter"] = idx.quarter
        out["week_of_year"] = idx.isocalendar().week.astype(int).values
        out["is_month_end"] = idx.is_month_end.astype(int)
        out["is_quarter_end"] = idx.is_quarter_end.astype(int)
        out["day_of_year"] = idx.dayofyear
        out["is_year_start"] = (idx.month == 1).astype(int) & (idx.day <= 5).astype(int)
        return out

    @staticmethod
    def day_of_week(idx: pd.DatetimeIndex) -> pd.Series:
        return pd.Series(idx.dayofweek, index=idx, name="day_of_week")

    @staticmethod
    def month(idx: pd.DatetimeIndex) -> pd.Series:
        return pd.Series(idx.month, index=idx, name="month")

    @staticmethod
    def quarter(idx: pd.DatetimeIndex) -> pd.Series:
        return pd.Series(idx.quarter, index=idx, name="quarter")


# ---------------------------------------------------------------------------
# Hierarchical Features (V7 Section 6 — hierarchical family)
# ---------------------------------------------------------------------------


class HierarchicalFeatures:
    """Aggregate entity-level metrics within a group (sector, league, etc.)."""

    @staticmethod
    def group_mean(df: pd.DataFrame, value_col: str, group_col: str) -> pd.Series:
        """Mean of *value_col* within each *group_col* group."""
        return df.groupby(group_col)[value_col].transform("mean").rename(f"{value_col}_group_mean")

    @staticmethod
    def group_rank(df: pd.DataFrame, value_col: str, group_col: str) -> pd.Series:
        """Percentile rank of *value_col* within each *group_col* group."""
        return (
            df.groupby(group_col)[value_col]
            .transform(lambda x: x.rank(pct=True))
            .rename(f"{value_col}_group_rank")
        )

    @staticmethod
    def group_z_score(df: pd.DataFrame, value_col: str, group_col: str) -> pd.Series:
        """Z-score of *value_col* within each *group_col* group."""
        grp = df.groupby(group_col)[value_col]
        mean = grp.transform("mean")
        std = grp.transform("std")
        z = (df[value_col] - mean) / std.replace(0, np.nan)
        return z.rename(f"{value_col}_group_z")

    @staticmethod
    def group_stats(df: pd.DataFrame, value_col: str, group_col: str) -> pd.DataFrame:
        """All hierarchical features for *value_col* grouped by *group_col*."""
        return pd.DataFrame(
            {
                f"{value_col}_group_mean": HierarchicalFeatures.group_mean(
                    df, value_col, group_col
                ),
                f"{value_col}_group_rank": HierarchicalFeatures.group_rank(
                    df, value_col, group_col
                ),
                f"{value_col}_group_z": HierarchicalFeatures.group_z_score(
                    df, value_col, group_col
                ),
            }
        )


# ---------------------------------------------------------------------------
# Interaction Features (V7 Section 6 — interaction family)
# ---------------------------------------------------------------------------


class InteractionFeatures:
    """Generate cross-feature interaction terms."""

    @staticmethod
    def product(df: pd.DataFrame, col_a: str, col_b: str) -> pd.Series:
        """Element-wise product of two columns."""
        return (df[col_a] * df[col_b]).rename(f"{col_a}_x_{col_b}")

    @staticmethod
    def ratio(df: pd.DataFrame, col_a: str, col_b: str) -> pd.Series:
        """Safe ratio of col_a / col_b (NaN when denominator is zero)."""
        denom = df[col_b].replace(0, np.nan)
        return (df[col_a] / denom).rename(f"{col_a}_over_{col_b}")

    @staticmethod
    def difference(df: pd.DataFrame, col_a: str, col_b: str) -> pd.Series:
        """Difference col_a - col_b."""
        return (df[col_a] - df[col_b]).rename(f"{col_a}_minus_{col_b}")

    @staticmethod
    def pairwise_interactions(
        df: pd.DataFrame, columns: list[str], methods: list[str] | None = None
    ) -> pd.DataFrame:
        """Generate pairwise interactions for all column pairs.

        Parameters
        ----------
        columns : list[str]
            Columns to interact.
        methods : list[str] | None
            Subset of ["product", "ratio", "difference"].  Defaults to all.
        """
        methods = methods or ["product", "ratio", "difference"]
        dispatch = {
            "product": InteractionFeatures.product,
            "ratio": InteractionFeatures.ratio,
            "difference": InteractionFeatures.difference,
        }
        parts: list[pd.Series] = []
        for i, a in enumerate(columns):
            for b in columns[i + 1 :]:
                for m in methods:
                    if m in dispatch:
                        parts.append(dispatch[m](df, a, b))
        if not parts:
            return pd.DataFrame(index=df.index)
        return pd.concat(parts, axis=1)


# ---------------------------------------------------------------------------
# Representation Features (V7 Section 6 — representation family)
# ---------------------------------------------------------------------------


class RepresentationFeatures:
    """Leakage-safe representation features."""

    @staticmethod
    def target_encode(
        df: pd.DataFrame,
        cat_col: str,
        target_col: str,
        min_samples: int = 10,
    ) -> pd.Series:
        """Expanding-window target encoding (no future leakage).

        For each row, the encoding is the mean of the target for all
        *previous* rows with the same category value.  If fewer than
        ``min_samples`` prior observations exist, the global expanding mean
        is used as a fallback.

        Parameters
        ----------
        df : pd.DataFrame
            Must be sorted in temporal order.
        cat_col : str
            Categorical column to encode.
        target_col : str
            Numeric target column.
        min_samples : int
            Minimum prior observations before using the group mean.
        """
        result = pd.Series(np.nan, index=df.index, name=f"{cat_col}_target_enc")
        global_expanding = df[target_col].expanding().mean().shift(1)

        for cat_value in df[cat_col].unique():
            mask = df[cat_col] == cat_value
            cat_expanding = df.loc[mask, target_col].expanding().mean().shift(1)
            cat_count = df.loc[mask, target_col].expanding().count().shift(1)
            # Use group mean when enough samples, else fallback to global
            use_group = cat_count >= min_samples
            result.loc[mask] = np.where(use_group, cat_expanding, global_expanding.loc[mask])
        return result

    @staticmethod
    def frequency_encode(df: pd.DataFrame, cat_col: str) -> pd.Series:
        """Expanding frequency encoding (proportion of each category so far)."""
        result = pd.Series(np.nan, index=df.index, name=f"{cat_col}_freq_enc")
        total_count = pd.Series(0.0, index=df.index)
        for cat_value in df[cat_col].unique():
            mask = df[cat_col] == cat_value
            cat_count = mask.astype(int).expanding().sum().shift(1)
            total = pd.Series(1, index=df.index).expanding().sum().shift(1)
            result.loc[mask] = (cat_count / total).loc[mask]
            total_count += cat_count.fillna(0)
        return result


# ---------------------------------------------------------------------------
# Automated Feature Selection / Elimination (V7 Section 6)
# ---------------------------------------------------------------------------


def select_features(
    x: pd.DataFrame,
    y: pd.Series | None = None,
    max_missing_rate: float = 0.5,
    min_variance: float = 1e-8,
    max_correlation: float = 0.95,
    max_features: int | None = None,
) -> list[str]:
    """Select features by eliminating low-quality columns.

    Elimination steps (in order):
    1. Drop columns with missing rate > ``max_missing_rate``.
    2. Drop columns with variance < ``min_variance``.
    3. Drop one of each pair with Pearson correlation > ``max_correlation``.
    4. If *y* is provided and ``max_features`` is set, keep the top
       features ranked by absolute correlation with the target.

    Parameters
    ----------
    x : pd.DataFrame
        Feature matrix (numeric columns).
    y : pd.Series | None
        Target variable for importance-based filtering.
    max_missing_rate : float
        Maximum fraction of NaN values allowed.
    min_variance : float
        Minimum variance threshold.
    max_correlation : float
        Maximum pairwise correlation allowed.
    max_features : int | None
        If set, keep only this many features (requires *y*).

    Returns
    -------
    list[str]
        Names of the surviving feature columns.
    """
    cols = list(x.columns)
    logger.info("Feature selection starting with %d columns", len(cols))

    # Step 1: missing rate filter
    missing = x[cols].isnull().mean()
    cols = [c for c in cols if missing[c] <= max_missing_rate]
    logger.info("After missing-rate filter: %d columns", len(cols))

    # Step 2: variance filter
    var = x[cols].var()
    cols = [c for c in cols if var[c] >= min_variance]
    logger.info("After variance filter: %d columns", len(cols))

    # Step 3: correlation filter — drop the later column in each high pair
    if len(cols) > 1:
        corr = x[cols].corr().abs()
        drop = set()
        for i in range(len(cols)):
            if cols[i] in drop:
                continue
            for j in range(i + 1, len(cols)):
                if cols[j] in drop:
                    continue
                if corr.iloc[i, j] > max_correlation:
                    drop.add(cols[j])
        cols = [c for c in cols if c not in drop]
        logger.info("After correlation filter: %d columns", len(cols))

    # Step 4: importance-based top-k (optional)
    if y is not None and max_features is not None and len(cols) > max_features:
        abs_corr = x[cols].corrwith(y).abs().sort_values(ascending=False)
        cols = abs_corr.head(max_features).index.tolist()
        logger.info("After importance filter: %d columns", len(cols))

    return cols
