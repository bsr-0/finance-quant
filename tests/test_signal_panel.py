"""Parity and panel-construction tests for vectorized signal scoring.

``SignalEngine.score_frame`` is a vectorized reimplementation of
``SignalEngine._score_row``, built because scoring a multi-decade, many-symbol
panel one Python-level row at a time is too slow.  The entire safety margin for
that reimplementation is exact agreement with the row-wise original, including
on the NaN-heavy warmup rows every real indicator frame starts with -- so the
parity test here is deliberately adversarial rather than a happy-path check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.strategy.signals import SignalEngine, compute_indicators


def _random_indicator_frame(rng: np.random.Generator, n: int = 60) -> pd.DataFrame:
    """A frame shaped like ``compute_indicators`` output, but with prices random
    enough that comparisons land on both sides of every threshold, plus randomly
    injected NaNs simulating warmup periods and missing data."""
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, n))), index=idx)
    high = close * (1 + abs(rng.normal(0, 0.01, n)))
    low = close * (1 - abs(rng.normal(0, 0.01, n)))
    volume = pd.Series(rng.integers(10_000, 1_000_000, n).astype(float), index=idx)

    df = pd.DataFrame({"open": close, "high": high, "low": low, "close": close, "volume": volume})
    df = compute_indicators(df)

    # Overwrite each derived indicator with fresh random values so scores land
    # on both sides of every threshold, independent of what the small synthetic
    # OHLCV happened to produce.
    for col, lo, hi in [
        ("sma_50", 80, 120),
        ("sma_200", 80, 120),
        ("sma_50_slope", -2, 2),
        ("rsi_14", 0, 100),
        ("bb_lower", 80, 120),
        ("stoch_k", 0, 100),
        ("volume_sma_20", 10_000, 1_000_000),
        ("obv_slope", -1000, 1000),
        ("atr_pct", 0, 6),
        ("macd_hist", -3, 3),
        ("macd_hist_prev", -3, 3),
        ("williams_r", -100, 0),
    ]:
        df[col] = rng.uniform(lo, hi, n)

    # Inject NaNs simulating indicator warmup and sparse data, at a rate high
    # enough that the NaN-gating logic is exercised on most rows.
    for col in df.columns:
        if col in ("open", "high", "low", "close", "volume"):
            continue
        mask = rng.random(n) < 0.3
        df.loc[mask, col] = np.nan

    return df


@pytest.mark.parametrize("seed", range(10))
def test_score_frame_matches_score_row_row_by_row(seed):
    rng = np.random.default_rng(seed)
    df = _random_indicator_frame(rng)
    engine = SignalEngine()

    vectorized = engine.score_frame(df)

    for date, row in df.iterrows():
        total, trend, pullback, volume, volatility = engine._score_row(row)
        got = vectorized.loc[date]
        assert got["score"] == total, f"score mismatch at {date}"
        assert got["trend_pts"] == trend, f"trend_pts mismatch at {date}"
        assert got["pullback_pts"] == pullback, f"pullback_pts mismatch at {date}"
        assert got["volume_pts"] == volume, f"volume_pts mismatch at {date}"
        assert got["volatility_pts"] == volatility, f"volatility_pts mismatch at {date}"


def test_score_frame_matches_on_real_indicator_warmup_rows():
    """The first ~200 rows of a real indicator frame are NaN-heavy in a
    structured way (progressively fewer NaNs as each rolling window fills),
    which is a different NaN pattern than the random injection above."""
    idx = pd.bdate_range("2020-01-01", periods=260)
    rng = np.random.default_rng(0)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, 260))), index=idx)
    df = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(100_000, 900_000, 260).astype(float),
        }
    )
    df = compute_indicators(df)
    engine = SignalEngine()

    vectorized = engine.score_frame(df)
    for date, row in df.iterrows():
        total, trend, pullback, volume, volatility = engine._score_row(row)
        got = vectorized.loc[date]
        assert got["score"] == total
        assert got["trend_pts"] == trend
        assert got["pullback_pts"] == pullback
        assert got["volume_pts"] == volume
        assert got["volatility_pts"] == volatility


def test_score_frame_handles_missing_columns():
    """A frame missing derived indicator columns entirely (not just NaN values)
    must fall back to the same defaults as ``row.get(col, default)``."""
    idx = pd.bdate_range("2020-01-01", periods=10)
    df = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 500_000.0,
        },
        index=idx,
    )
    engine = SignalEngine()

    vectorized = engine.score_frame(df)
    for date, row in df.iterrows():
        total, trend, pullback, volume, volatility = engine._score_row(row)
        got = vectorized.loc[date]
        assert got["score"] == total
        assert got["trend_pts"] == trend
        assert got["pullback_pts"] == pullback
        assert got["volume_pts"] == volume
        assert got["volatility_pts"] == volatility


def test_trend_bonus_requires_both_moving_averages_present():
    """Regression pin: the original nests the close>sma_200 bonus inside the
    same NaN guard as the close>sma_50>sma_200 bonus, so it requires sma_50 to
    be present even though sma_50 isn't part of that comparison."""
    idx = pd.bdate_range("2020-01-01", periods=1)
    df = pd.DataFrame(
        {
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.0],
            "volume": [500_000.0],
            "sma_50": [np.nan],  # missing
            "sma_200": [90.0],  # present, and close > sma_200
        },
        index=idx,
    )
    engine = SignalEngine()

    vectorized = engine.score_frame(df)
    total, trend, *_ = engine._score_row(df.iloc[0])

    assert trend == 0, "sma_50 missing must suppress the close>sma_200 bonus too"
    assert vectorized.iloc[0]["trend_pts"] == 0
    assert vectorized.iloc[0]["trend_pts"] == trend
