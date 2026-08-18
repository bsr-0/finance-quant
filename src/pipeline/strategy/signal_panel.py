"""Build a historical, multi-symbol panel of signal scores.

``SignalEngine.scan_history`` (``signals.py:238``) is unusable for this: it's
per-symbol, recomputes indicators on every call, and loops ``_score_row`` once
per row in plain Python. Scoring 17 ETFs over 2010-2026 that way is slow;
scoring the full 1,160-ticker universe is worse. This module vectorizes across
both symbols and dates using ``SignalEngine.score_frame``, so the whole panel
is built from a handful of DataFrame operations per symbol instead of one
Python call per row.

The eligibility rule replicated here matches ``scan_history`` exactly
(``signals.py:262-268``), including its regime default of ``"bull"`` when no
SPY series is supplied or a date's regime is missing. That default is a
separate, known issue -- see the signal-validation plan's Phase 4 -- and is
deliberately left alone here so this panel measures the score the pipeline
actually produces today, not a hypothetically corrected one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from pipeline.eval.regime import classify_regimes
from pipeline.strategy.signals import SignalEngine, compute_indicators

logger = logging.getLogger(__name__)


@dataclass
class ScorePanel:
    """Wide ``DatetimeIndex x ticker`` frames, one column per field.

    This is the shape ``pipeline.eval.signal_alpha`` consumes (no MultiIndex):
    ``compute_forward_returns``, ``rank_ic``, ``walk_forward_ic``, and the
    already-existing ``test-signal-alpha`` CLI command all expect a date-index,
    symbol-column frame.
    """

    score: pd.DataFrame
    trend_pts: pd.DataFrame
    pullback_pts: pd.DataFrame
    volume_pts: pd.DataFrame
    volatility_pts: pd.DataFrame
    entry_eligible: pd.DataFrame
    close: pd.DataFrame

    def to_parquet_dict(self) -> dict[str, pd.DataFrame]:
        return {
            "score": self.score,
            "trend_pts": self.trend_pts,
            "pullback_pts": self.pullback_pts,
            "volume_pts": self.volume_pts,
            "volatility_pts": self.volatility_pts,
            "entry_eligible": self.entry_eligible,
            "close": self.close,
        }


def _regime_for_dates(spy_prices: pd.Series | None, dates: pd.DatetimeIndex) -> pd.Series:
    """Reproduce the regime lookup in ``scan_history`` for an arbitrary date index."""
    if spy_prices is None or len(spy_prices) < 50:
        return pd.Series("BULL", index=dates)

    regimes = classify_regimes(spy_prices)
    aligned = regimes.reindex(dates)
    # scan_history:262 defaults a missing date's regime to "bull" before the
    # isinstance/upper() call below, rather than propagating NaN.
    aligned = aligned.fillna("bull").astype(str).str.upper()
    aligned = aligned.replace("FLAT", "NEUTRAL")
    return aligned


def score_ticker(
    df: pd.DataFrame,
    engine: SignalEngine,
    spy_prices: pd.Series | None = None,
) -> pd.DataFrame:
    """Score every bar of one ticker's OHLCV history.

    Mirrors ``SignalEngine.scan_history`` field-for-field but computes the score
    components with ``score_frame`` instead of a per-row Python loop.
    """
    indicators = compute_indicators(df)
    scored = engine.score_frame(indicators)

    regime = _regime_for_dates(spy_prices, indicators.index)
    threshold = regime.map({"NEUTRAL": engine.neutral_threshold}).fillna(engine.entry_threshold)

    eligible = (
        (regime != "BEAR")
        & (scored["trend_pts"] >= 25)
        & (scored["pullback_pts"] > 0)
        & (scored["score"] >= threshold)
    )

    out = scored.copy()
    out["regime"] = regime
    out["entry_eligible"] = eligible
    out["close"] = indicators["close"]
    return out


def build_score_panel(
    price_frames: dict[str, pd.DataFrame],
    engine: SignalEngine | None = None,
    spy_prices: pd.Series | None = None,
) -> ScorePanel:
    """Score every ticker in *price_frames* and assemble a wide panel.

    Args:
        price_frames: ``{ticker: OHLCV DataFrame}``, e.g. from
            ``pipeline.strategy.price_panel.load_ticker_frames``.
        engine: Scoring engine. Defaults constructed if omitted.
        spy_prices: SPY close series for regime classification, ideally spanning
            a longer window than any individual ticker's history so the regime
            gate isn't blind to drawdowns that started before the panel begins.
    """
    engine = engine or SignalEngine()

    per_ticker: dict[str, pd.DataFrame] = {}
    for ticker, df in price_frames.items():
        if df.empty or len(df) < 5:
            logger.warning("Skipping %s: only %d bars", ticker, len(df))
            continue
        try:
            per_ticker[ticker] = score_ticker(df, engine, spy_prices)
        except Exception:
            logger.exception("Failed to score %s", ticker)

    if not per_ticker:
        empty = pd.DataFrame()
        return ScorePanel(empty, empty, empty, empty, empty, empty, empty)

    def wide(field: str, dtype=None) -> pd.DataFrame:
        cols = {t: d[field] for t, d in per_ticker.items()}
        panel = pd.DataFrame(cols).sort_index()
        return panel.astype(dtype) if dtype is not None else panel

    return ScorePanel(
        score=wide("score"),
        trend_pts=wide("trend_pts"),
        pullback_pts=wide("pullback_pts"),
        volume_pts=wide("volume_pts"),
        volatility_pts=wide("volatility_pts"),
        entry_eligible=wide("entry_eligible", dtype=bool),
        close=wide("close"),
    )
