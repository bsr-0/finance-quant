"""Build a ready-to-use feature matrix for AutoResearch.

Chains together:
1. Query OHLCV prices from ``cur_prices_ohlcv_daily``
2. Compute technical indicators (SMA, RSI, MACD, Bollinger, etc.)
3. Query snapshot features from ``snap_symbol_features`` (fundamentals,
   options, sentiment, events)
4. Flatten JSON columns (macro_panel, news_counts)
5. Add seasonal/calendar features
6. Compute forward return target (``fwd_return_1d``)
7. Align features to prevent look-ahead bias
8. Export to parquet

Usage::

    from pipeline.autoresearch.build_dataset import build_autoresearch_dataset
    path = build_autoresearch_dataset(symbol="SPY")
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.db import get_db_manager
from pipeline.features.feature_families import SeasonalFeatures, select_features
from pipeline.features.technical_indicators import TechnicalIndicators

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/autoresearch")


def _query_prices(
    symbol: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch OHLCV data for a single symbol from curated prices."""
    db = get_db_manager()
    rows = db.run_query(
        """
        SELECT p.date, p.open, p.high, p.low, p.close, p.adj_close, p.volume
        FROM cur_prices_ohlcv_daily p
        JOIN dim_symbol s ON p.symbol_id = s.symbol_id
        WHERE s.ticker = :ticker
          AND p.date >= :start_date
          AND p.date <= :end_date
        ORDER BY p.date
        """,
        {"ticker": symbol, "start_date": start_date, "end_date": end_date},
    )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    # Convert numeric columns
    for col in ["open", "high", "low", "close", "adj_close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _query_snapshots(
    symbol: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch snapshot features for a single symbol."""
    db = get_db_manager()
    rows = db.run_query(
        """
        SELECT sf.*
        FROM snap_symbol_features sf
        JOIN dim_symbol s ON sf.symbol_id = s.symbol_id
        WHERE s.ticker = :ticker
          AND sf.asof_ts >= :start_date
          AND sf.asof_ts <= :end_date
        ORDER BY sf.asof_ts
        """,
        {"ticker": symbol, "start_date": start_date, "end_date": end_date},
    )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "asof_ts" in df.columns:
        df["asof_ts"] = pd.to_datetime(df["asof_ts"])
        df = df.set_index("asof_ts")
    return df


def _flatten_json_column(df: pd.DataFrame, col: str, prefix: str = "") -> pd.DataFrame:
    """Expand a JSON/dict column into separate numeric columns."""
    if col not in df.columns:
        return df

    def safe_parse(val):
        if val is None:
            return {}
        if isinstance(val, dict):
            return val
        if isinstance(val, str):
            try:
                return json.loads(val)
            except (json.JSONDecodeError, TypeError):
                return {}
        return {}

    expanded = df[col].apply(safe_parse).apply(pd.Series)
    if expanded.empty:
        return df.drop(columns=[col])

    if prefix:
        expanded = expanded.add_prefix(prefix)

    # Convert to numeric where possible
    for c in expanded.columns:
        expanded[c] = pd.to_numeric(expanded[c], errors="coerce")

    return pd.concat([df.drop(columns=[col]), expanded], axis=1)


def _query_macro_latest(start_date: str, end_date: str) -> pd.DataFrame:
    """Query macro observations and pivot to wide format (date x series)."""
    db = get_db_manager()
    rows = db.run_query(
        """
        SELECT ms.series_code, mo.period_end AS date, mo.value
        FROM cur_macro_observations mo
        JOIN dim_macro_series ms ON mo.series_id = ms.series_id
        WHERE mo.period_end >= :start_date
          AND mo.period_end <= :end_date
        ORDER BY mo.period_end
        """,
        {"start_date": start_date, "end_date": end_date},
    )
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Pivot: rows=date, columns=series_code
    wide = df.pivot_table(index="date", columns="series_code", values="value", aggfunc="last")
    # Forward-fill macro data (released monthly/quarterly)
    wide = wide.sort_index().ffill()
    return wide


def build_autoresearch_dataset(
    symbol: str = "SPY",
    start_date: str = "2020-01-01",
    end_date: str = "2024-12-31",
    output_dir: Path | None = None,
    include_snapshots: bool = True,
    include_macro: bool = True,
    max_missing_rate: float = 0.5,
) -> Path:
    """Build a complete feature matrix for AutoResearch.

    Parameters
    ----------
    symbol : str
        Ticker symbol (default SPY).
    start_date, end_date : str
        Date range for the dataset.
    output_dir : Path, optional
        Output directory (default ``data/autoresearch/``).
    include_snapshots : bool
        Whether to join snapshot features (fundamentals, options, etc.).
    include_macro : bool
        Whether to join macro observations.
    max_missing_rate : float
        Drop columns with more than this fraction of NaN values.

    Returns
    -------
    Path
        Path to the exported parquet file.
    """
    output_dir = output_dir or OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Prices ---
    logger.info("Querying prices for %s (%s to %s)", symbol, start_date, end_date)
    prices = _query_prices(symbol, start_date, end_date)

    if prices.empty:
        raise ValueError(
            f"No price data found for {symbol} between {start_date} and {end_date}. "
            f"Run `mdw extract prices` and `mdw load-raw prices` first."
        )
    logger.info("Loaded %d price rows", len(prices))

    # --- Step 2: Technical indicators ---
    logger.info("Computing technical indicators")
    df = TechnicalIndicators.calculate_all(
        prices,
        price_col="close",
        high_col="high",
        low_col="low",
        volume_col="volume",
    )

    # --- Step 3: Seasonal features ---
    logger.info("Adding seasonal features")
    seasonal = SeasonalFeatures.from_index(df)
    df = pd.concat([df, seasonal], axis=1)

    # --- Step 4: Snapshot features ---
    if include_snapshots:
        logger.info("Querying snapshot features")
        snaps = _query_snapshots(symbol, start_date, end_date)
        if not snaps.empty:
            # Drop metadata columns
            drop_cols = [
                c for c in ["symbol_id", "event_time", "available_time", "created_at", "updated_at"]
                if c in snaps.columns
            ]
            snaps = snaps.drop(columns=drop_cols, errors="ignore")

            # Flatten JSON columns
            snaps = _flatten_json_column(snaps, "macro_panel", prefix="")
            snaps = _flatten_json_column(snaps, "news_counts", prefix="news_")

            # Align snapshot dates with price dates
            snaps.index = snaps.index.normalize()
            df.index = pd.to_datetime(df.index)
            snaps = snaps[~snaps.index.duplicated(keep="last")]
            df = df.join(snaps, how="left", rsuffix="_snap")
            logger.info("Joined %d snapshot rows (%d new columns)", len(snaps), len(snaps.columns))
        else:
            logger.warning("No snapshot features found — skipping")

    # --- Step 5: Macro features (if not already in snapshots) ---
    if include_macro:
        logger.info("Querying macro observations")
        macro = _query_macro_latest(start_date, end_date)
        if not macro.empty:
            # Only join macro columns not already present
            existing = set(df.columns)
            new_macro_cols = [c for c in macro.columns if c not in existing]
            if new_macro_cols:
                macro = macro[new_macro_cols]
                macro.index = pd.to_datetime(macro.index)
                df = df.join(macro, how="left")
                # Forward-fill macro (released infrequently)
                df[new_macro_cols] = df[new_macro_cols].ffill()
                logger.info("Joined %d macro series", len(new_macro_cols))
        else:
            logger.warning("No macro data found — skipping")

    # --- Step 6: Compute forward return target ---
    logger.info("Computing fwd_return_1d target")
    df["fwd_return_1d"] = df["close"].pct_change().shift(-1)

    # --- Step 7: Drop metadata and low-quality columns ---
    # Remove raw OHLCV (already captured in technicals)
    meta_cols = ["open", "high", "low", "adj_close"]
    df = df.drop(columns=[c for c in meta_cols if c in df.columns], errors="ignore")

    # Drop columns with too many NaNs
    before_cols = len(df.columns)
    missing_rates = df.isnull().mean()
    keep_cols = missing_rates[missing_rates <= max_missing_rate].index.tolist()
    # Always keep target and close
    for must_keep in ["fwd_return_1d", "close", "volume"]:
        if must_keep in df.columns and must_keep not in keep_cols:
            keep_cols.append(must_keep)
    df = df[keep_cols]
    dropped = before_cols - len(df.columns)
    if dropped > 0:
        logger.info("Dropped %d columns with >%.0f%% missing", dropped, max_missing_rate * 100)

    # Drop rows where target is NaN (last row + any gaps)
    df = df.dropna(subset=["fwd_return_1d"])

    # --- Step 8: Export ---
    output_path = output_dir / f"{symbol.lower()}_features.parquet"
    df.to_parquet(output_path)

    # Summary
    feature_cols = [c for c in df.columns if c != "fwd_return_1d"]
    logger.info(
        "Dataset saved to %s: %d rows x %d features + target",
        output_path,
        len(df),
        len(feature_cols),
    )

    return output_path
