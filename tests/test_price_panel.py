"""Tests for raw-lake price loading and adjustment-artifact repair."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.strategy.price_panel import (
    detect_price_discontinuities,
    load_price_panel,
    load_ticker_frames,
    repair_price_discontinuities,
)


def make_frame(closes, raw=None, start="2020-01-01"):
    """OHLCV frame with an optional independent unadjusted series."""
    idx = pd.bdate_range(start, periods=len(closes))
    closes = np.asarray(closes, dtype=float)
    raw = closes.copy() if raw is None else np.asarray(raw, dtype=float)
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes * 1.01,
            "low": closes * 0.99,
            "close": closes,
            "volume": 1_000_000,
            "unadjusted_close": raw,
        },
        index=idx,
    )


def test_detects_break_unsupported_by_raw_series():
    # Adjusted close doubles; the true price does not move.
    df = make_frame([10, 10, 20, 20, 20], raw=[20, 20, 20, 20, 20])
    breaks = detect_price_discontinuities(df, "XLU")

    assert len(breaks) == 1
    assert breaks[0].date == df.index[2]
    assert breaks[0].factor == pytest.approx(2.0)


def test_ignores_genuine_large_move_corroborated_by_raw():
    # Both series move together: a real crash, not an adjustment artifact.
    df = make_frame([100, 100, 45, 45], raw=[100, 100, 45, 45])
    assert detect_price_discontinuities(df, "REAL") == []


def test_ignores_ordinary_volatility():
    df = make_frame([100, 103, 99, 105], raw=[100, 103, 99, 105])
    assert detect_price_discontinuities(df, "CALM") == []


def test_repair_removes_the_step():
    df = make_frame([10, 10, 20, 20, 20], raw=[20, 20, 20, 20, 20])
    fixed, breaks = repair_price_discontinuities(df, "XLU")

    assert len(breaks) == 1
    assert fixed["close"].pct_change().abs().max() < 0.01


def test_repair_anchors_on_the_recent_segment():
    """Recent prices must not move: recorded entry/stop/target live on that scale."""
    df = make_frame([10, 10, 20, 20, 20], raw=[20, 20, 20, 20, 20])
    fixed, _ = repair_price_discontinuities(df, "XLU")

    # Post-break bars unchanged, pre-break bars scaled up onto the true scale.
    assert fixed["close"].iloc[-1] == pytest.approx(20.0)
    assert fixed["close"].iloc[2] == pytest.approx(20.0)
    assert fixed["close"].iloc[0] == pytest.approx(20.0)


def test_repair_scales_ohlc_consistently():
    df = make_frame([10, 10, 20, 20], raw=[20, 20, 20, 20])
    fixed, _ = repair_price_discontinuities(df, "XLU")

    row = fixed.iloc[0]
    assert row["low"] < row["close"] < row["high"]
    assert row["high"] / row["close"] == pytest.approx(1.01)


def test_repair_is_a_noop_on_clean_data():
    df = make_frame([100, 101, 102, 103], raw=[100, 101, 102, 103])
    fixed, breaks = repair_price_discontinuities(df, "CLEAN")

    assert breaks == []
    pd.testing.assert_frame_equal(fixed, df)


def test_repair_handles_multiple_breaks():
    df = make_frame([5, 5, 10, 10, 20, 20], raw=[20] * 6)
    fixed, breaks = repair_price_discontinuities(df, "TWICE")

    assert len(breaks) == 2
    assert fixed["close"].pct_change().abs().max() < 0.01
    assert fixed["close"].iloc[-1] == pytest.approx(20.0)


# --- loading from the raw lake -------------------------------------------


def write_snapshot(raw_dir, ticker, start, end, closes, extracted):
    idx = pd.bdate_range(start, periods=len(closes))
    pd.DataFrame(
        {
            "date": idx,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": 1000,
            "unadjusted_close": closes,
            "extracted_at": pd.Timestamp(extracted, tz="UTC"),
        }
    ).to_parquet(raw_dir / f"{ticker}_{start}_{end}.parquet")


def test_loader_concatenates_all_snapshots(tmp_path):
    """The newest file is often a short incremental extract, not the history."""
    write_snapshot(tmp_path, "SPY", "2020-01-01", "2020-01-10", [100] * 8, "2020-01-11")
    write_snapshot(tmp_path, "SPY", "2026-04-01", "2026-04-03", [200] * 3, "2026-04-04")

    frames, _ = load_ticker_frames(["SPY"], tmp_path)

    assert len(frames["SPY"]) == 11, "must span both snapshots, not just the newest"
    assert frames["SPY"].index.min() == pd.Timestamp("2020-01-01")


def test_loader_prefers_the_latest_extraction_for_a_revised_bar(tmp_path):
    write_snapshot(tmp_path, "SPY", "2020-01-01", "2020-01-03", [100, 100, 100], "2020-01-04")
    write_snapshot(tmp_path, "SPY", "2020-01-01", "2020-01-03", [111, 111, 111], "2020-06-01")

    frames, _ = load_ticker_frames(["SPY"], tmp_path)

    assert len(frames["SPY"]) == 3
    assert (frames["SPY"]["close"] == 111).all()


def test_loader_respects_date_bounds(tmp_path):
    write_snapshot(tmp_path, "SPY", "2020-01-01", "2020-01-20", [100] * 14, "2020-02-01")

    frames, _ = load_ticker_frames(["SPY"], tmp_path, start="2020-01-08", end="2020-01-14")
    idx = frames["SPY"].index

    assert idx.min() >= pd.Timestamp("2020-01-08")
    assert idx.max() <= pd.Timestamp("2020-01-14")


def test_loader_returns_empty_for_missing_directory(tmp_path):
    frames, breaks = load_ticker_frames(["SPY"], tmp_path / "nope")
    assert frames == {} and breaks == []


def test_price_panel_is_wide_and_has_no_multiindex(tmp_path):
    write_snapshot(tmp_path, "SPY", "2020-01-01", "2020-01-08", [100] * 6, "2020-02-01")
    write_snapshot(tmp_path, "QQQ", "2020-01-01", "2020-01-08", [200] * 6, "2020-02-01")

    panel = load_price_panel(["SPY", "QQQ"], tmp_path)

    assert sorted(panel.columns) == ["QQQ", "SPY"]
    assert isinstance(panel.index, pd.DatetimeIndex)
    assert not isinstance(panel.columns, pd.MultiIndex)
