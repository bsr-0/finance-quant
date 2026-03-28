"""Standardized signal output for trader consumption.

Formats signal scores into a CSV-ready DataFrame with all information
a discretionary or systematic trader needs to act on signals:
ticker, direction, score, entry/stop/target prices, risk metrics,
regime, and confidence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from pipeline.strategy.signals import SignalScore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SignalRow:
    """Single row of the signal output blotter."""

    date: pd.Timestamp
    ticker: str
    direction: str  # LONG / FLAT
    score: int
    trend_pts: int
    pullback_pts: int
    volume_pts: int
    volatility_pts: int
    entry_price: float
    stop_price: float
    target_1: float
    target_2: float
    atr: float
    atr_pct: float
    regime: str
    confidence: str  # HIGH / MEDIUM / LOW
    strategy_id: str


def _confidence_label(score: int) -> str:
    if score >= 80:
        return "HIGH"
    if score >= 70:
        return "MEDIUM"
    return "LOW"


def format_signals(
    scores: list[SignalScore],
    price_data: dict[str, pd.DataFrame],
    date: pd.Timestamp,
    stop_atr_multiple: float = 1.5,
    target_1_atr_multiple: float = 2.0,
    target_2_atr_multiple: float = 3.0,
    strategy_id: str = "QSG-MICRO-SWING-001",
) -> pd.DataFrame:
    """Convert signal scores into a trader-consumable DataFrame.

    Only includes signals where ``entry_eligible`` is True.

    Args:
        scores: Signal scores from ``SignalEngine.score_universe``.
        price_data: ``{ticker: DataFrame}`` with OHLCV and indicators.
        date: Signal generation date.
        stop_atr_multiple: ATR multiplier for stop-loss distance.
        target_1_atr_multiple: ATR multiplier for first profit target.
        target_2_atr_multiple: ATR multiplier for second profit target.
        strategy_id: Identifier for the originating strategy.

    Returns:
        DataFrame with one row per eligible signal, sorted by score descending.
    """
    rows: list[dict] = []

    for sig in scores:
        if not sig.entry_eligible:
            continue

        ticker = sig.symbol
        df = price_data.get(ticker)
        if df is None or df.empty:
            continue

        # Get latest available row
        row = df.loc[date] if date in df.index else df.iloc[-1]

        close = float(row["close"])
        atr = float(row.get("atr_14", 0))
        atr_pct = float(row.get("atr_pct", 0))

        if atr <= 0:
            continue

        stop_price = close - atr * stop_atr_multiple
        target_1 = close + atr * target_1_atr_multiple
        target_2 = close + atr * target_2_atr_multiple

        rows.append(
            {
                "date": date,
                "ticker": ticker,
                "direction": "LONG",
                "score": sig.score,
                "trend_pts": sig.trend_pts,
                "pullback_pts": sig.pullback_pts,
                "volume_pts": sig.volume_pts,
                "volatility_pts": sig.volatility_pts,
                "entry_price": round(close, 4),
                "stop_price": round(stop_price, 4),
                "target_1": round(target_1, 4),
                "target_2": round(target_2, 4),
                "atr": round(atr, 4),
                "atr_pct": round(atr_pct, 2),
                "regime": sig.regime,
                "confidence": _confidence_label(sig.score),
                "strategy_id": strategy_id,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "direction",
                "score",
                "trend_pts",
                "pullback_pts",
                "volume_pts",
                "volatility_pts",
                "entry_price",
                "stop_price",
                "target_1",
                "target_2",
                "atr",
                "atr_pct",
                "regime",
                "confidence",
                "strategy_id",
            ]
        )

    result = pd.DataFrame(rows)
    result = result.sort_values("score", ascending=False).reset_index(drop=True)
    return result


def write_signal_csv(
    signals_df: pd.DataFrame,
    output_dir: str | Path,
    date: pd.Timestamp | None = None,
) -> Path:
    """Write signals DataFrame to a dated CSV file.

    Args:
        signals_df: Output from ``format_signals``.
        output_dir: Directory to write the CSV file.
        date: Date for the filename.  If None, uses today.

    Returns:
        Path to the written CSV file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if date is None:
        date_str = datetime.now().strftime("%Y%m%d")
    else:
        date_str = pd.Timestamp(date).strftime("%Y%m%d")

    filepath = output_dir / f"signals_{date_str}.csv"
    signals_df.to_csv(filepath, index=False)
    logger.info("Wrote %d signals to %s", len(signals_df), filepath)
    return filepath
