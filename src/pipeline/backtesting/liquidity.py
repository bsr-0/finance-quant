"""Historical liquidity utilities for realistic backtesting constraints.

Provides rolling ADV (Average Daily Volume) computation so that trade
size limits and market impact estimates use the actual liquidity available
on each historical date, rather than a single static average.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def compute_historical_adv(
    volume: pd.DataFrame,
    window: int = 20,
    min_periods: int = 5,
) -> pd.DataFrame:
    """Compute rolling average daily volume from raw daily volume data.

    Args:
        volume: DataFrame with DatetimeIndex rows and symbol columns,
            containing daily trading volume.
        window: Rolling window size in trading days (default 20 ≈ 1 month).
        min_periods: Minimum observations required for a valid ADV value.

    Returns:
        DataFrame of same shape with rolling mean volume per symbol per date.
    """
    return volume.rolling(window, min_periods=min_periods).mean()
