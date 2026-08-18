"""Load and repair historical OHLCV panels from the raw price lake.

The raw lake stores many overlapping snapshots per ticker
(``{TICKER}_{start}_{end}.parquet``), so a single "latest" file is not the full
history -- the newest snapshot is often a short incremental extract.  Frames are
assembled by concatenating every snapshot for a ticker and de-duplicating dates
by extraction time.

It also repairs a real defect in the lake: the ``close`` column is back-adjusted
using ``split_ratio`` records that are sometimes spurious, which injects
fabricated ~2x discontinuities.  For example XLU carries a ``2:1`` split on
2025-12-05 that never happened -- the underlying price moved -0.9% that day
while the adjusted ``close`` jumped +98%.  Sampling 150 tickers over 2010-2026
found 48 such jumps across 36 tickers in ``close`` versus 12 across 5 in
``unadjusted_close``, so the latter is used to corroborate and correct.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

RAW_PRICES_DIR = Path("data/raw/prices")
OHLCV = ["open", "high", "low", "close", "volume"]

# A one-day move this large is treated as a candidate artifact...
JUMP_THRESHOLD = 0.45
# ...and confirmed spurious only if the corroborating series disagrees this much.
DISAGREEMENT_THRESHOLD = 0.25


@dataclass(frozen=True)
class Discontinuity:
    """A price break in the adjusted series unsupported by the raw series."""

    ticker: str
    date: pd.Timestamp
    adjusted_return: float
    raw_return: float

    @property
    def factor(self) -> float:
        """Multiplicative size of the spurious step."""
        return (1.0 + self.adjusted_return) / (1.0 + self.raw_return)


def detect_price_discontinuities(
    df: pd.DataFrame,
    ticker: str = "",
    jump_threshold: float = JUMP_THRESHOLD,
    disagreement_threshold: float = DISAGREEMENT_THRESHOLD,
) -> list[Discontinuity]:
    """Find breaks in ``close`` that ``unadjusted_close`` does not corroborate.

    A genuine split or a genuine large move shows up in both series.  A bogus
    adjustment shows up only in ``close``.
    """
    if "unadjusted_close" not in df.columns or len(df) < 2:
        return []

    adj = df["close"].pct_change()
    raw = df["unadjusted_close"].pct_change()
    suspect = (adj.abs() > jump_threshold) & ((adj - raw).abs() > disagreement_threshold)

    return [
        Discontinuity(
            ticker=ticker,
            date=date,
            adjusted_return=float(adj.loc[date]),
            raw_return=float(raw.loc[date]),
        )
        for date in df.index[suspect.fillna(False)]
    ]


def repair_price_discontinuities(
    df: pd.DataFrame, ticker: str = ""
) -> tuple[pd.DataFrame, list[Discontinuity]]:
    """Remove spurious adjustment steps from OHLC while keeping the series adjusted.

    A bogus split back-adjusts every bar *before* the split date, so it is the
    earlier segment that is wrong and the most recent segment that matches the
    true traded price.  Repair therefore anchors on the latest segment and
    scales earlier bars up by the break factor, leaving current prices -- and so
    any entry/stop/target recorded against them -- untouched.  Rescaling forward
    instead would silently move every recent price onto a fictional scale.

    Working from ``close`` rather than substituting ``unadjusted_close``
    preserves the legitimate dividend adjustment.
    """
    breaks = detect_price_discontinuities(df, ticker)
    if not breaks:
        return df, []

    out = df.copy()
    # Cumulative correction: bars before a break are scaled up by that break's
    # factor, compounding across multiple breaks.
    scale = pd.Series(1.0, index=out.index)
    for brk in breaks:
        scale.loc[: brk.date] *= brk.factor
        # The break bar itself belongs to the corrected (later) segment.
        scale.loc[brk.date] /= brk.factor

    for col in ("open", "high", "low", "close"):
        if col in out.columns:
            out[col] = out[col] * scale

    logger.warning(
        "Repaired %d spurious price discontinuit%s in %s: %s",
        len(breaks),
        "y" if len(breaks) == 1 else "ies",
        ticker or "<unknown>",
        ", ".join(f"{b.date.date()} ({b.adjusted_return * 100:+.0f}%)" for b in breaks),
    )
    return out, breaks


def _snapshot_paths(tickers: set[str], raw_dir: Path) -> dict[str, list[Path]]:
    """Group every snapshot file in *raw_dir* by ticker."""
    grouped: dict[str, list[Path]] = {}
    for path in raw_dir.iterdir():
        if path.suffix.lower() not in {".parquet", ".pq", ".csv"}:
            continue
        ticker = path.stem.split("_")[0].upper()
        if tickers and ticker not in tickers:
            continue
        grouped.setdefault(ticker, []).append(path)
    return grouped


def load_ticker_frames(
    tickers: set[str] | list[str],
    raw_dir: Path | str = RAW_PRICES_DIR,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    repair: bool = True,
    min_bars: int = 1,
) -> tuple[dict[str, pd.DataFrame], list[Discontinuity]]:
    """Assemble a date-indexed OHLCV frame per ticker from the raw lake.

    Args:
        tickers: Tickers to load.  Empty means every ticker present.
        raw_dir: Raw price directory.
        start: Optional inclusive lower date bound.
        end: Optional inclusive upper date bound.
        repair: Correct spurious adjustment discontinuities.
        min_bars: Drop tickers with fewer than this many bars.

    Returns:
        ``(frames, discontinuities)`` where ``frames`` maps ticker to an
        ascending, date-indexed OHLCV frame.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        logger.warning("Raw price directory %s does not exist", raw_dir)
        return {}, []

    wanted = {t.upper() for t in tickers}
    grouped = _snapshot_paths(wanted, raw_dir)

    frames: dict[str, pd.DataFrame] = {}
    all_breaks: list[Discontinuity] = []

    for ticker, paths in grouped.items():
        parts = [
            pd.read_csv(p) if p.suffix.lower() == ".csv" else pd.read_parquet(p) for p in paths
        ]
        df = pd.concat(parts, ignore_index=True)
        if "date" not in df.columns:
            continue

        df["date"] = pd.to_datetime(df["date"])
        # Overlapping snapshots disagree only where a later extract revised a
        # bar, so the most recent extraction wins.
        sort_cols = ["date", "extracted_at"] if "extracted_at" in df.columns else ["date"]
        df = df.sort_values(sort_cols).drop_duplicates("date", keep="last")
        df = df.set_index("date").sort_index()

        if repair:
            df, breaks = repair_price_discontinuities(df, ticker)
            all_breaks.extend(breaks)

        if start is not None:
            df = df[df.index >= pd.Timestamp(start)]
        if end is not None:
            df = df[df.index <= pd.Timestamp(end)]

        if len(df) >= min_bars:
            frames[ticker] = df

    missing = wanted - frames.keys()
    if missing:
        logger.warning("No usable price data for %d ticker(s): %s", len(missing), sorted(missing))

    return frames, all_breaks


def load_price_panel(
    tickers: set[str] | list[str],
    raw_dir: Path | str = RAW_PRICES_DIR,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    field: str = "close",
    repair: bool = True,
) -> pd.DataFrame:
    """Build a wide ``DatetimeIndex x ticker`` panel of one OHLCV field.

    This is the shape consumed by ``pipeline.eval.signal_alpha`` (no MultiIndex).
    """
    frames, _ = load_ticker_frames(tickers, raw_dir, start, end, repair=repair)
    if not frames:
        return pd.DataFrame()
    return pd.DataFrame(
        {t: df[field] for t, df in frames.items() if field in df.columns}
    ).sort_index()
