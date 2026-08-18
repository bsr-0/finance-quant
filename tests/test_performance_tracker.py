"""Tests for barrier-outcome resolution and prediction performance tracking.

Covers ``pipeline.web.outcome_resolution.resolve_one`` and the
``PerformanceTracker`` wrapper.  All fixtures are synthetic OHLCV frames -- no
network, no database.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from pipeline.web.outcome_resolution import ResolutionPolicy, resolve_one
from pipeline.web.performance_tracker import PerformanceTracker

ENTRY = 100.0
STOP = 97.0
TARGET = 104.0

# Most assertions are about barrier geometry, so costs are off by default and
# exercised explicitly in test_cost_bps_applied.
FREE = ResolutionPolicy(cost_bps=0.0)


def make_bars(
    rows: list[tuple[float, float, float, float]], start: str = "2026-01-05"
) -> pd.DataFrame:
    """Build a business-day-indexed OHLCV frame from (open, high, low, close) tuples."""
    idx = pd.bdate_range(start=start, periods=len(rows))
    return pd.DataFrame(
        [{"open": o, "high": h, "low": lo, "close": c} for o, h, lo, c in rows],
        index=idx,
    )


def flat_bar(price: float = 100.0) -> tuple[float, float, float, float]:
    """A bar that touches neither barrier."""
    return (price, price + 0.5, price - 0.5, price)


# --- resolve_one: clean barrier hits -------------------------------------


def test_clean_target_hit():
    bars = make_bars([flat_bar(), flat_bar(), (100.0, 104.5, 99.5, 104.2)])
    res = resolve_one(bars, ENTRY, STOP, TARGET, FREE)

    assert res is not None
    assert res.outcome == "hit_target"
    assert res.fill_price == TARGET
    assert res.bars_held == 3
    assert res.pnl_pct == pytest.approx(4.0)
    assert not res.gapped and not res.same_bar_ambiguous


def test_clean_stop_hit():
    bars = make_bars([flat_bar(), (100.0, 100.5, 96.5, 97.2)])
    res = resolve_one(bars, ENTRY, STOP, TARGET, FREE)

    assert res is not None
    assert res.outcome == "stopped_out"
    assert res.fill_price == STOP
    assert res.bars_held == 2
    assert res.pnl_pct == pytest.approx(-3.0)


# --- resolve_one: same-bar ambiguity -------------------------------------


def test_same_bar_both_barriers_resolves_conservatively():
    # One wide bar spanning both barriers.
    bars = make_bars([(100.0, 105.0, 96.0, 101.0)])
    res = resolve_one(bars, ENTRY, STOP, TARGET, FREE)

    assert res is not None
    assert res.outcome == "stopped_out"
    assert res.same_bar_ambiguous is True


def test_same_bar_open_near_target_still_stops_out():
    """Regression pin: resolution must not depend on where the bar opened.

    The previous rule compared abs(open-stop) to abs(open-target), so a bar that
    opened near the target resolved as a win.  That made the bias a function of
    the stop/target asymmetry rather than of what happened intrabar.
    """
    bars = make_bars([(103.8, 105.0, 96.0, 101.0)])  # opens just under target
    res = resolve_one(bars, ENTRY, STOP, TARGET, FREE)

    assert res is not None
    assert res.outcome == "stopped_out"
    assert res.same_bar_ambiguous is True


def test_same_bar_policy_target_first():
    bars = make_bars([(100.0, 105.0, 96.0, 101.0)])
    res = resolve_one(
        bars, ENTRY, STOP, TARGET, ResolutionPolicy(cost_bps=0.0, same_bar_policy="target_first")
    )

    assert res is not None
    assert res.outcome == "hit_target"
    assert res.same_bar_ambiguous is True


# --- resolve_one: gap fills ----------------------------------------------


def test_gap_down_fills_worse_than_stop():
    bars = make_bars([(94.0, 95.0, 93.0, 94.5)])  # gaps straight through the stop
    res = resolve_one(bars, ENTRY, STOP, TARGET, FREE)

    assert res is not None
    assert res.outcome == "stopped_out"
    assert res.gapped is True
    assert res.fill_price == 94.0
    # Strictly worse than assuming an exact fill at the stop.
    assert res.pnl_pct < (STOP - ENTRY) / ENTRY * 100


def test_gap_up_fills_better_than_target():
    bars = make_bars([(106.0, 107.0, 105.5, 106.5)])
    res = resolve_one(bars, ENTRY, STOP, TARGET, FREE)

    assert res is not None
    assert res.outcome == "hit_target"
    assert res.gapped is True
    assert res.fill_price == 106.0
    assert res.pnl_pct > (TARGET - ENTRY) / ENTRY * 100


# --- resolve_one: holding window (the blocking bug) ----------------------


def test_holding_window_bounds_the_scan():
    """A stop beyond the holding window must not resolve the trade.

    Previously the bar scan ran to ``as_of`` with no holding bound, so a stale
    history file resolved a 15-bar trade against months of price action.
    """
    bars = make_bars([flat_bar() for _ in range(19)] + [(100.0, 100.5, 90.0, 91.0)])
    assert len(bars) == 20

    res = resolve_one(bars, ENTRY, STOP, TARGET, FREE)

    assert res is not None
    assert res.outcome == "expired", "stop at bar 20 must not resolve a 15-bar trade"
    assert res.bars_held == 15
    assert res.resolved_date == bars.index[14]


def test_still_active_when_window_incomplete():
    bars = make_bars([flat_bar() for _ in range(5)])
    assert resolve_one(bars, ENTRY, STOP, TARGET, FREE) is None


def test_bars_held_counts_bars_not_calendar_days():
    # Starts on a Thursday so the window spans a weekend: 5 bars, 7 calendar days.
    bars = make_bars(
        [flat_bar(), flat_bar(), flat_bar(), flat_bar(), (100.0, 104.5, 99.5, 104.2)],
        start="2026-01-08",
    )
    res = resolve_one(bars, ENTRY, STOP, TARGET, FREE)

    assert res is not None
    assert res.bars_held == 5
    calendar_days = (res.resolved_date - (bars.index[0] - pd.Timedelta(days=1))).days
    assert calendar_days == 7
    assert calendar_days > res.bars_held


def test_cost_bps_applied():
    bars = make_bars([(100.0, 104.5, 99.5, 104.2)])
    gross = resolve_one(bars, ENTRY, STOP, TARGET, FREE)
    net = resolve_one(bars, ENTRY, STOP, TARGET, ResolutionPolicy(cost_bps=3.0))

    assert gross is not None and net is not None
    assert net.pnl_pct == pytest.approx(gross.pnl_pct - 0.03)


# --- PerformanceTracker integration --------------------------------------


def build_tracker(tmp_path, preds: list[dict], policy: ResolutionPolicy | None = None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "history.json"
    path.write_text(json.dumps({"predictions": preds, "last_updated": ""}))
    return PerformanceTracker(path, policy=policy or FREE)


def make_pred(ticker: str = "XLP", signal_date: str = "2026-01-02") -> dict:
    return {
        "signal_date": signal_date,
        "ticker": ticker,
        "score": 80,
        "confidence": "UNRATED",
        "entry_price": ENTRY,
        "stop_price": STOP,
        "target_price": TARGET,
        "regime": "BULL",
        "direction": "long",
        "outcome": "active",
        "resolved_date": None,
        "resolved_price": None,
        "pnl_pct": None,
        "days_held": None,
    }


def test_missing_ticker_is_unresolvable(tmp_path):
    tracker = build_tracker(tmp_path, [make_pred(ticker="DELISTED")])
    summary = tracker.resolve_outcomes({"XLP": make_bars([flat_bar()])}, as_of="2026-02-01")

    assert summary["unresolvable"] == 1
    assert tracker.history.predictions[0]["outcome"] == "active"
    assert tracker.history.predictions[0]["unresolvable_reason"] == "no_price_data"
    assert tracker.get_stats()["unresolvable"] == 1


def test_resolution_is_idempotent(tmp_path):
    bars = make_bars([flat_bar(), (100.0, 104.5, 99.5, 104.2)])
    tracker = build_tracker(tmp_path, [make_pred()])

    tracker.resolve_outcomes({"XLP": bars}, as_of="2026-02-01")
    first = json.dumps(tracker.history.predictions, sort_keys=True)
    tracker.resolve_outcomes({"XLP": bars}, as_of="2026-03-01")
    second = json.dumps(tracker.history.predictions, sort_keys=True)

    assert first == second


def test_bars_after_as_of_are_not_used(tmp_path):
    # Target is hit on the third bar, but as_of stops at the second.
    bars = make_bars([flat_bar(), flat_bar(), (100.0, 104.5, 99.5, 104.2)], start="2026-01-05")
    tracker = build_tracker(tmp_path, [make_pred()])

    summary = tracker.resolve_outcomes({"XLP": bars}, as_of=str(bars.index[1].date()))

    assert summary["hit_target"] == 0
    assert summary["still_active"] == 1
    assert tracker.history.predictions[0]["outcome"] == "active"


def test_profitable_expiry_splits_the_two_win_definitions(tmp_path):
    # Drifts up to +2% without ever touching the +4% target, then expires.
    bars = make_bars([flat_bar(102.0) for _ in range(15)])
    tracker = build_tracker(tmp_path, [make_pred()])
    tracker.resolve_outcomes({"XLP": bars}, as_of="2026-03-01")

    stats = tracker.get_stats()
    assert stats["expired"] == 1
    assert stats["target_hit_rate"] == 0.0, "an expiry never reached the target"
    assert stats["profitable_rate"] == 100.0, "but it did make money"
    assert stats["win_rate"] == stats["profitable_rate"]


def test_stop_first_is_no_more_optimistic_than_target_first(tmp_path):
    ambiguous = make_bars([(100.0, 105.0, 96.0, 101.0)])
    preds = [make_pred(ticker=f"T{i}") for i in range(4)]
    price_data = {f"T{i}": ambiguous for i in range(4)}

    rates = {}
    for policy_name in ("stop_first", "target_first"):
        tracker = build_tracker(
            tmp_path / policy_name,
            [dict(p) for p in preds],
            policy=ResolutionPolicy(cost_bps=0.0, same_bar_policy=policy_name),
        )
        tracker.resolve_outcomes(price_data, as_of="2026-03-01")
        rates[policy_name] = tracker.get_stats()["profitable_rate"]

    assert rates["stop_first"] <= rates["target_first"]


def test_stats_report_ambiguity_and_gap_counts(tmp_path):
    tracker = build_tracker(
        tmp_path,
        [make_pred(ticker="AMB"), make_pred(ticker="GAP")],
    )
    tracker.resolve_outcomes(
        {
            "AMB": make_bars([(100.0, 105.0, 96.0, 101.0)]),
            "GAP": make_bars([(94.0, 95.0, 93.0, 94.5)]),
        },
        as_of="2026-03-01",
    )

    stats = tracker.get_stats()
    assert stats["n_ambiguous"] == 1
    assert stats["n_gapped"] == 1
    assert stats["mean_bars_held"] == 1.0
