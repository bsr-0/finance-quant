"""Repair double-applied split adjustments in stored price tables.

The Yahoo chart endpoint returns ``quote.close`` already back-adjusted for
splits while still listing the split under ``events.splits``.  The extractor
applied the adjustment a second time, so every bar before a split was scaled by
the split ratio again.  That injects a step change into an otherwise continuous
series -- XLU's close jumps +98% on 2025-12-05 while the traded price moved
-0.9% -- which corrupts any indicator whose window spans the break.  A 200-day
SMA is 40% of the swing strategy's score, so this reaches the signals directly.

``adjust_for_corporate_actions`` no longer re-applies these, but data already
written to ``raw_prices_ohlcv`` and ``cur_prices_ohlcv_daily`` still carries the
defect and re-running the transform would only copy it forward.  This module
repairs both in place.

The correction is exact rather than heuristic: for a double-applied split of
ratio ``R``, bars before the split date were divided by ``R`` one time too many,
so multiplying them by ``R`` restores them.  This holds for reverse splits too
(``R < 1``).  Anchoring on the post-split segment keeps current prices fixed,
which matters because recorded entry/stop/target prices live on that scale.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
from sqlalchemy import text

logger = logging.getLogger(__name__)

# Ignore ratios within ~10% of 1.0: the double-adjustment signature is not
# separable from ordinary daily volatility at that scale, and the correction
# would be immaterial anyway.
MIN_ABS_LOG_RATIO = 0.1

# The ratio hypothesis must fit at least this much better than the no-step
# hypothesis. A double adjustment is a pure multiplication, so the observed step
# matches the ratio almost exactly; requiring only "closer to ratio than to 1"
# also catches genuine corporate actions that happen to move the price part of
# the way, and repairing those over-corrects. Demanding a factor-of-two better
# fit separates the two cleanly and makes the repair converge.
FIT_MARGIN = 0.5


@dataclass(frozen=True)
class DoubleAdjustedSplit:
    """A split whose adjustment was applied twice."""

    ticker: str
    date: pd.Timestamp
    ratio: float
    observed_step: float

    def __str__(self) -> str:
        return (
            f"{self.ticker} {self.date:%Y-%m-%d} ratio={self.ratio:g} "
            f"observed step x{self.observed_step:.3f}"
        )


# A split bar in a correctly-adjusted series shows no step. If instead the step
# matches the split ratio, the adjustment was applied on top of already-adjusted
# data. Comparing both hypotheses in log space needs no tolerance constant.
_FIND_SQL = """
WITH px AS (
    SELECT symbol_id, date, close,
           LAG(close) OVER (PARTITION BY symbol_id ORDER BY date) AS prev_close
    FROM cur_prices_ohlcv_daily
)
SELECT s.ticker,
       p.symbol_id,
       p.date,
       ca.ratio,
       p.close / p.prev_close AS observed_step
FROM px p
JOIN cur_corporate_actions ca
  ON ca.symbol_id = p.symbol_id
 AND ca.action_date = p.date
 AND ca.action_type = 'split'
JOIN dim_symbol s ON s.symbol_id = p.symbol_id
WHERE p.prev_close > 0
  AND p.close > 0
  AND ca.ratio IS NOT NULL
  AND ca.ratio > 0
  AND abs(ln(ca.ratio)) >= :min_log_ratio
  AND abs(ln(p.close / p.prev_close) - ln(ca.ratio)) < :fit_margin * abs(ln(p.close / p.prev_close))
ORDER BY s.ticker, p.date
"""


def find_double_adjusted_splits(
    conn,
    min_log_ratio: float = MIN_ABS_LOG_RATIO,
    fit_margin: float = FIT_MARGIN,
) -> list[DoubleAdjustedSplit]:
    """Find splits whose adjustment was applied to already-adjusted prices."""
    rows = conn.execute(
        text(_FIND_SQL), {"min_log_ratio": min_log_ratio, "fit_margin": fit_margin}
    ).fetchall()
    return [
        DoubleAdjustedSplit(
            ticker=r.ticker,
            date=pd.Timestamp(r.date),
            ratio=float(r.ratio),
            observed_step=float(r.observed_step),
        )
        for r in rows
    ]


# Bars before a break are scaled by that break's ratio; overlapping breaks
# compound, hence the product over all later break dates.
_CORRECTIONS_SQL = """
CREATE OR REPLACE TEMP TABLE _price_corrections AS
WITH px AS (
    SELECT symbol_id, date, close,
           LAG(close) OVER (PARTITION BY symbol_id ORDER BY date) AS prev_close
    FROM cur_prices_ohlcv_daily
),
bad AS (
    SELECT p.symbol_id, p.date, ca.ratio
    FROM px p
    JOIN cur_corporate_actions ca
      ON ca.symbol_id = p.symbol_id
     AND ca.action_date = p.date
     AND ca.action_type = 'split'
    WHERE p.prev_close > 0
      AND p.close > 0
      AND ca.ratio IS NOT NULL
      AND ca.ratio > 0
      AND abs(ln(ca.ratio)) >= :min_log_ratio
      AND abs(ln(p.close / p.prev_close) - ln(ca.ratio))
          < :fit_margin * abs(ln(p.close / p.prev_close))
)
SELECT t.symbol_id,
       s.ticker,
       t.date,
       EXP(SUM(LN(b.ratio))) AS factor
FROM cur_prices_ohlcv_daily t
JOIN bad b ON b.symbol_id = t.symbol_id AND t.date < b.date
JOIN dim_symbol s ON s.symbol_id = t.symbol_id
GROUP BY t.symbol_id, s.ticker, t.date
"""

_UPDATE_CURATED_SQL = """
UPDATE cur_prices_ohlcv_daily AS p
SET open = p.open * c.factor,
    high = p.high * c.factor,
    low = p.low * c.factor,
    close = p.close * c.factor,
    adj_close = p.adj_close * c.factor,
    data_quality_flag = 'split_readjusted',
    updated_at = NOW()
FROM _price_corrections c
WHERE c.symbol_id = p.symbol_id AND c.date = p.date
"""

_UPDATE_RAW_SQL = """
UPDATE raw_prices_ohlcv AS r
SET open = r.open * c.factor,
    high = r.high * c.factor,
    low = r.low * c.factor,
    close = r.close * c.factor,
    adj_close = r.adj_close * c.factor
FROM _price_corrections c
WHERE c.ticker = r.ticker AND c.date = r.date
"""


def repair_double_adjusted_splits(
    db,
    min_log_ratio: float = MIN_ABS_LOG_RATIO,
    fit_margin: float = FIT_MARGIN,
    dry_run: bool = True,
) -> dict:
    """Undo double-applied split adjustments in the raw and curated price tables.

    Both tables are repaired: the curated table is a straight passthrough of the
    raw one, so fixing only the former would be undone by the next transform.

    Args:
        db: Database manager exposing ``.engine``.
        min_log_ratio: Ignore splits closer to 1.0 than this in log space.
        fit_margin: How much better the ratio hypothesis must fit than no-step.
        dry_run: Report what would change without writing.

    Returns:
        Summary with ``splits``, ``tickers``, ``bars``, ``curated_rows``,
        ``raw_rows`` and ``dry_run``.
    """
    with db.engine.connect() as conn:
        breaks = find_double_adjusted_splits(conn, min_log_ratio, fit_margin)
        if not breaks:
            logger.info("No double-adjusted splits found")
            return {
                "splits": 0,
                "tickers": 0,
                "bars": 0,
                "curated_rows": 0,
                "raw_rows": 0,
                "dry_run": dry_run,
                "breaks": [],
            }

        conn.execute(
            text(_CORRECTIONS_SQL),
            {"min_log_ratio": min_log_ratio, "fit_margin": fit_margin},
        )
        bars = conn.execute(text("SELECT count(*) FROM _price_corrections")).scalar() or 0

        summary = {
            "splits": len(breaks),
            "tickers": len({b.ticker for b in breaks}),
            "bars": int(bars),
            "curated_rows": 0,
            "raw_rows": 0,
            "dry_run": dry_run,
            "breaks": breaks,
        }

        if dry_run:
            logger.info(
                "Dry run: %d double-adjusted splits across %d tickers would rescale %d bars",
                summary["splits"],
                summary["tickers"],
                summary["bars"],
            )
            return summary

        curated = conn.execute(text(_UPDATE_CURATED_SQL))
        raw = conn.execute(text(_UPDATE_RAW_SQL))
        conn.commit()

        summary["curated_rows"] = curated.rowcount if curated.rowcount is not None else int(bars)
        summary["raw_rows"] = raw.rowcount if raw.rowcount is not None else 0

        logger.info(
            "Repaired %d double-adjusted splits across %d tickers (%d bars rescaled)",
            summary["splits"],
            summary["tickers"],
            summary["bars"],
        )
        return summary
