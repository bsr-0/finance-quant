"""Signal generation for the Trend-Aligned Pullback Reversion strategy.

Computes a composite signal score (0-100) from four independent categories:
  - Trend alignment (40 pts)
  - Pullback depth  (30 pts)
  - Volume confirmation (15 pts)
  - Volatility / momentum context (15 pts)

An entry signal fires only when the score reaches the configured threshold
(default 60) and all regime filters pass.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pipeline.eval.regime import classify_regimes
from pipeline.features.technical_indicators import TechnicalIndicators

logger = logging.getLogger(__name__)

TI = TechnicalIndicators


@dataclass(frozen=True)
class SignalScore:
    """Result of signal computation for a single symbol on a single date."""

    symbol: str
    date: pd.Timestamp
    score: int
    trend_pts: int
    pullback_pts: int
    volume_pts: int
    volatility_pts: int
    regime: str
    entry_eligible: bool


def _slope(series: pd.Series, window: int = 5) -> pd.Series:
    """Simple linear slope over *window* periods."""
    return series.diff(window) / window


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all indicators required by the signal engine.

    Expects a DataFrame with columns: open, high, low, close, volume
    indexed by date (ascending).  Returns a copy with indicator columns
    appended.
    """
    out = df.copy()
    close = df["close"]

    # Moving averages
    out["sma_20"] = TI.sma(close, 20)
    out["sma_50"] = TI.sma(close, 50)
    out["sma_200"] = TI.sma(close, 200)
    out["sma_50_slope"] = _slope(out["sma_50"], 5)

    # RSI
    out["rsi_14"] = TI.rsi(close, 14)

    # Bollinger Bands
    bb_upper, bb_mid, bb_lower = TI.bollinger_bands(close, 20, 2.0)
    out["bb_upper"] = bb_upper
    out["bb_lower"] = bb_lower

    # Stochastic
    k, d = TI.stochastic(df["high"], df["low"], close, 14, 3)
    out["stoch_k"] = k
    out["stoch_d"] = d

    # ATR
    out["atr_14"] = TI.atr(df["high"], df["low"], close, 14)
    out["atr_pct"] = out["atr_14"] / close * 100

    # Volume
    out["volume_sma_20"] = TI.sma(df["volume"], 20)

    # OBV
    out["obv"] = TI.obv(close, df["volume"])
    out["obv_slope"] = _slope(out["obv"], 5)

    # MACD
    macd_line, signal_line, histogram = TI.macd(close)
    out["macd_hist"] = histogram
    out["macd_hist_prev"] = histogram.shift(1)

    # Williams %R
    out["williams_r"] = TI.williams_r(df["high"], df["low"], close)

    return out


class SignalEngine:
    """Generate entry signals for the pullback-reversion strategy."""

    def __init__(
        self,
        entry_threshold: int = 60,
        neutral_threshold: int = 70,
        atr_pct_min: float = 0.5,
        atr_pct_max: float = 4.0,
    ) -> None:
        self.entry_threshold = entry_threshold
        self.neutral_threshold = neutral_threshold
        self.atr_pct_min = atr_pct_min
        self.atr_pct_max = atr_pct_max

    def _score_row(self, row: pd.Series) -> tuple[int, int, int, int, int]:
        """Compute composite score for a single row of indicator data."""
        trend = 0
        pullback = 0
        volume = 0
        volatility = 0

        close = row["close"]
        sma_50 = row.get("sma_50", np.nan)
        sma_200 = row.get("sma_200", np.nan)

        # --- Trend alignment (max 40) ---
        if not np.isnan(sma_50) and not np.isnan(sma_200):
            if close > sma_50 and sma_50 > sma_200:
                trend += 25
            if close > sma_200:
                trend += 10
        slope_50 = row.get("sma_50_slope", 0)
        if not np.isnan(slope_50) and slope_50 > 0:
            trend += 5

        # --- Pullback depth (max 30) ---
        rsi = row.get("rsi_14", 50)
        if not np.isnan(rsi) and rsi < 35:
            pullback += 15
        bb_lower = row.get("bb_lower", np.nan)
        if not np.isnan(bb_lower) and close <= bb_lower:
            pullback += 10
        stoch_k = row.get("stoch_k", 50)
        if not np.isnan(stoch_k) and stoch_k < 20:
            pullback += 5

        # --- Volume confirmation (max 15) ---
        vol = row.get("volume", 0)
        vol_sma = row.get("volume_sma_20", 0)
        if vol_sma > 0 and not np.isnan(vol_sma) and vol < vol_sma * 0.8:
            volume += 10
        obv_sl = row.get("obv_slope", 0)
        if not np.isnan(obv_sl) and obv_sl > 0:
            volume += 5

        # --- Volatility / momentum (max 15) ---
        atr_pct = row.get("atr_pct", 0)
        if self.atr_pct_min < atr_pct < self.atr_pct_max:
            volatility += 5
        macd_hist = row.get("macd_hist", 0)
        macd_prev = row.get("macd_hist_prev", 0)
        if not np.isnan(macd_hist) and not np.isnan(macd_prev) and macd_hist > macd_prev:
            volatility += 5
        wr = row.get("williams_r", -50)
        if not np.isnan(wr) and wr < -80:
            volatility += 5

        total = trend + pullback + volume + volatility
        return total, trend, pullback, volume, volatility

    def score_universe(
        self,
        indicator_data: dict[str, pd.DataFrame],
        spy_prices: pd.Series | None = None,
        date: pd.Timestamp | None = None,
    ) -> list[SignalScore]:
        """Score all symbols in the universe for a given date.

        Args:
            indicator_data: ``{symbol: DataFrame}`` where each DataFrame has
                been processed by ``compute_indicators``.
            spy_prices: SPY close prices for regime classification.
            date: The evaluation date.  If ``None``, uses the last available
                row in each DataFrame.

        Returns:
            List of ``SignalScore`` objects, sorted by score descending.
        """
        # Determine regime
        regime = "BULL"
        if spy_prices is not None and len(spy_prices) >= 50:
            regimes = classify_regimes(spy_prices)
            if len(regimes) > 0:
                if date is None:
                    regime = regimes.iloc[-1].upper()
                else:
                    regime = regimes.get(date, "BULL").upper()
                if regime == "FLAT":
                    regime = "NEUTRAL"

        results: list[SignalScore] = []
        for symbol, df in indicator_data.items():
            if df.empty:
                continue

            row = df.iloc[-1] if date is None else df.loc[date] if date in df.index else None
            if row is None:
                continue

            total, trend, pb, vol, volat = self._score_row(row)

            # Determine eligibility
            threshold = self.neutral_threshold if regime == "NEUTRAL" else self.entry_threshold
            eligible = (
                regime != "BEAR"
                and trend >= 25  # Must have primary uptrend
                and pb > 0      # Must have some pullback signal
                and total >= threshold
            )

            results.append(
                SignalScore(
                    symbol=symbol,
                    date=row.name if hasattr(row, "name") else date,
                    score=total,
                    trend_pts=trend,
                    pullback_pts=pb,
                    volume_pts=vol,
                    volatility_pts=volat,
                    regime=regime,
                    entry_eligible=eligible,
                )
            )

        results.sort(key=lambda s: s.score, reverse=True)
        return results

    def scan_history(
        self,
        df: pd.DataFrame,
        symbol: str,
        spy_prices: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Score every row in a historical DataFrame for a single symbol.

        Returns a DataFrame with columns: score, trend_pts, pullback_pts,
        volume_pts, volatility_pts, regime, entry_eligible.
        """
        indicator_df = compute_indicators(df)

        # Regime series
        if spy_prices is not None and len(spy_prices) >= 50:
            regimes = classify_regimes(spy_prices)
        else:
            regimes = pd.Series("bull", index=df.index)

        records = []
        for idx in indicator_df.index:
            row = indicator_df.loc[idx]
            total, trend, pb, vol, volat = self._score_row(row)

            regime_val = regimes.get(idx, "bull")
            regime = regime_val.upper() if isinstance(regime_val, str) else "BULL"
            if regime == "FLAT":
                regime = "NEUTRAL"

            threshold = self.neutral_threshold if regime == "NEUTRAL" else self.entry_threshold
            eligible = (
                regime != "BEAR"
                and trend >= 25
                and pb > 0
                and total >= threshold
            )

            records.append(
                {
                    "date": idx,
                    "score": total,
                    "trend_pts": trend,
                    "pullback_pts": pb,
                    "volume_pts": vol,
                    "volatility_pts": volat,
                    "regime": regime,
                    "entry_eligible": eligible,
                }
            )

        return pd.DataFrame(records).set_index("date")
