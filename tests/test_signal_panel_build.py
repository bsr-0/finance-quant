"""Tests for pipeline.strategy.signal_panel: building a wide, multi-symbol,
multi-date score panel from per-ticker OHLCV history."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.strategy.signal_panel import build_score_panel, score_ticker
from pipeline.strategy.signals import SignalEngine, compute_indicators


def make_ohlcv(seed: int, n: int = 300, start: str = "2020-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0002, 0.015, n))), index=idx)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(100_000, 900_000, n).astype(float),
        }
    )


def test_score_ticker_matches_row_wise_score_row():
    df = make_ohlcv(seed=7)
    engine = SignalEngine()
    scored = score_ticker(df, engine)

    indicators = compute_indicators(df)
    for date in indicators.index[-30:]:
        total, trend, pb, vol, volat = engine._score_row(indicators.loc[date])
        row = scored.loc[date]
        assert row["score"] == total
        assert row["trend_pts"] == trend
        assert row["pullback_pts"] == pb
        assert row["volume_pts"] == vol
        assert row["volatility_pts"] == volat


def test_score_ticker_eligibility_matches_scan_history_rule():
    """entry_eligible must reproduce SignalEngine.scan_history's rule exactly:
    regime != BEAR, trend >= 25, pullback > 0, score >= threshold."""
    df = make_ohlcv(seed=3)
    engine = SignalEngine()
    scored = score_ticker(df, engine)

    for date, row in scored.iterrows():
        threshold = (
            engine.neutral_threshold if row["regime"] == "NEUTRAL" else engine.entry_threshold
        )
        expected = (
            row["regime"] != "BEAR"
            and row["trend_pts"] >= 25
            and row["pullback_pts"] > 0
            and row["score"] >= threshold
        )
        assert bool(row["entry_eligible"]) == expected, date


def test_score_ticker_defaults_to_bull_without_spy():
    df = make_ohlcv(seed=1)
    scored = score_ticker(df, SignalEngine(), spy_prices=None)
    assert (scored["regime"] == "BULL").all()


def test_build_score_panel_produces_wide_frames_no_multiindex():
    frames = {"A": make_ohlcv(1), "B": make_ohlcv(2), "C": make_ohlcv(3)}
    panel = build_score_panel(frames, SignalEngine())

    assert sorted(panel.score.columns) == ["A", "B", "C"]
    assert not isinstance(panel.score.columns, pd.MultiIndex)
    assert not isinstance(panel.score.index, pd.MultiIndex)
    assert isinstance(panel.score.index, pd.DatetimeIndex)
    assert panel.entry_eligible.dtypes.unique().tolist() == [np.dtype(bool)]


def test_build_score_panel_field_values_match_score_ticker():
    frames = {"A": make_ohlcv(1), "B": make_ohlcv(2)}
    engine = SignalEngine()
    panel = build_score_panel(frames, engine)

    for ticker, df in frames.items():
        individual = score_ticker(df, engine)
        pd.testing.assert_series_equal(panel.score[ticker], individual["score"], check_names=False)
        pd.testing.assert_series_equal(panel.close[ticker], individual["close"], check_names=False)


def test_build_score_panel_skips_short_history_without_raising():
    frames = {"A": make_ohlcv(1), "TOOSHORT": make_ohlcv(2, n=3)}
    panel = build_score_panel(frames, SignalEngine())

    assert "A" in panel.score.columns
    assert "TOOSHORT" not in panel.score.columns


def test_build_score_panel_handles_empty_input():
    panel = build_score_panel({}, SignalEngine())
    assert panel.score.empty


def test_build_score_panel_survives_one_bad_ticker():
    """A single malformed frame must not take down the whole panel build."""
    bad = pd.DataFrame({"close": [1.0]})  # missing open/high/low/volume
    frames = {"GOOD": make_ohlcv(1), "BAD": bad}
    panel = build_score_panel(frames, SignalEngine())

    assert "GOOD" in panel.score.columns
    assert "BAD" not in panel.score.columns


def test_regime_gate_blocks_bear_regardless_of_score():
    """A ticker forced into a BEAR regime must never be eligible."""
    df = make_ohlcv(seed=9)
    engine = SignalEngine()

    # A SPY series that is deeply underwater the whole time -> persistent BEAR.
    spy_idx = pd.bdate_range(df.index[0] - pd.Timedelta(days=400), periods=len(df) + 300)
    spy = pd.Series(np.linspace(200, 50, len(spy_idx)), index=spy_idx)

    scored = score_ticker(df, engine, spy_prices=spy)
    bear_rows = scored[scored["regime"] == "BEAR"]
    assert not bear_rows.empty, "fixture should actually produce some BEAR regime rows"
    assert not bear_rows["entry_eligible"].any()
