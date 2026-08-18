"""Barrier-outcome resolution for tracked signal predictions.

Pure functions with no I/O, extracted from ``PerformanceTracker`` so the
resolution rules are testable in isolation.

A prediction is resolved against three barriers: a stop, a profit target, and a
holding-period limit measured in *bars* (not calendar days).  The bar scan is
bounded by ``ResolutionPolicy.max_holding_bars`` before it starts, so a
prediction can never be resolved against price action beyond its intended
holding period regardless of how stale the history file is.

Fills are gap-aware: when a bar opens beyond a barrier the fill is taken at the
open, which is worse than the stop on a gap down and better than the target on a
gap up.  Assuming an exact fill at the barrier price systematically overstates
performance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)

SAME_BAR_POLICIES = ("stop_first", "target_first")


@dataclass(frozen=True)
class ResolutionPolicy:
    """Rules governing how a prediction is resolved against price bars.

    Attributes:
        max_holding_bars: Holding limit in trading bars.  The bar scan is
            truncated to this length before any barrier check runs.
        same_bar_policy: How to resolve a bar that touches both barriers.
            ``stop_first`` is the conservative default -- see
            ``_resolve_ambiguous`` for why an unconditional rule is preferred
            over a price-dependent one.
        model_gaps: When True, fill at the bar open if it gaps through a
            barrier.  When False, always fill at the barrier price (the legacy
            optimistic behaviour).
        cost_bps: Round-trip transaction cost in basis points, subtracted once
            from the realised P&L.
    """

    max_holding_bars: int = 15
    same_bar_policy: str = "stop_first"
    model_gaps: bool = True
    cost_bps: float = 3.0

    def __post_init__(self) -> None:
        if self.same_bar_policy not in SAME_BAR_POLICIES:
            raise ValueError(
                f"same_bar_policy must be one of {SAME_BAR_POLICIES}, got {self.same_bar_policy!r}"
            )
        if self.max_holding_bars < 1:
            raise ValueError(f"max_holding_bars must be >= 1, got {self.max_holding_bars}")


@dataclass(frozen=True)
class Resolution:
    """Outcome of resolving a single prediction."""

    outcome: str  # hit_target | stopped_out | expired
    resolved_date: pd.Timestamp
    fill_price: float
    pnl_pct: float
    bars_held: int
    same_bar_ambiguous: bool = False
    gapped: bool = False


def _pnl_pct(entry: float, exit_price: float, direction: str, cost_bps: float) -> float:
    """Realised P&L in percent, net of a one-off cost charge."""
    gross = (exit_price - entry) / entry * 100.0
    if direction == "short":
        gross = -gross
    return gross - cost_bps / 100.0


def _resolve_ambiguous(policy: ResolutionPolicy) -> str:
    """Decide a bar that touched both barriers.

    Deliberately ignores the bar's open.  The previous implementation compared
    ``abs(open - stop)`` to ``abs(open - target)``, which makes the bias a
    function of the stop/target asymmetry: with a 1.5-ATR stop and a 2.0-ATR
    target the open is closer to the stop most of the time, so ambiguous bars
    resolved as stops at a rate that had nothing to do with what actually
    happened intrabar.  A fixed rule has a known, reportable direction; callers
    bracket their statistics by running both policies.
    """
    return "stopped_out" if policy.same_bar_policy == "stop_first" else "hit_target"


def resolve_one(
    bars: pd.DataFrame,
    entry: float,
    stop: float,
    target: float,
    policy: ResolutionPolicy,
    direction: str = "long",
) -> Resolution | None:
    """Resolve one prediction against forward price bars.

    Args:
        bars: Forward bars strictly after the signal date, ascending, already
            filtered to those visible as of the evaluation date.  Must carry
            ``open``/``high``/``low``/``close`` columns; ``close`` is used as a
            fallback when the others are absent.
        entry: Entry price.
        stop: Stop price (below entry for longs, above for shorts).
        target: Profit target (above entry for longs, below for shorts).
        policy: Resolution rules.
        direction: ``long`` or ``short``.

    Returns:
        A ``Resolution``, or ``None`` if the prediction is still open -- that is,
        neither barrier was touched and fewer than ``max_holding_bars`` bars are
        available yet.
    """
    if bars.empty:
        return None

    is_long = direction.lower() != "short"

    # Bound the scan *before* checking any barrier.  Without this a stale
    # history file resolves a 15-bar trade against months of price action.
    window = bars.iloc[: policy.max_holding_bars]

    for i, (bar_date, bar) in enumerate(window.iterrows(), start=1):
        close = float(bar.get("close", 0.0))
        bar_open = float(bar.get("open", close))
        high = float(bar.get("high", close))
        low = float(bar.get("low", close))

        if is_long:
            gap_stop = bar_open <= stop
            gap_target = bar_open >= target
            hit_stop = low <= stop
            hit_target = high >= target
        else:
            gap_stop = bar_open >= stop
            gap_target = bar_open <= target
            hit_stop = high >= stop
            hit_target = low <= target

        # A gap through a barrier fills at the open, which is strictly worse
        # than the stop and strictly better than the target.  The two gap cases
        # are mutually exclusive because stop and target straddle the entry.
        if policy.model_gaps and gap_stop:
            return Resolution(
                outcome="stopped_out",
                resolved_date=bar_date,
                fill_price=bar_open,
                pnl_pct=_pnl_pct(entry, bar_open, direction, policy.cost_bps),
                bars_held=i,
                gapped=True,
            )

        if policy.model_gaps and gap_target:
            return Resolution(
                outcome="hit_target",
                resolved_date=bar_date,
                fill_price=bar_open,
                pnl_pct=_pnl_pct(entry, bar_open, direction, policy.cost_bps),
                bars_held=i,
                gapped=True,
            )

        if hit_stop and hit_target:
            outcome = _resolve_ambiguous(policy)
            fill = stop if outcome == "stopped_out" else target
            return Resolution(
                outcome=outcome,
                resolved_date=bar_date,
                fill_price=fill,
                pnl_pct=_pnl_pct(entry, fill, direction, policy.cost_bps),
                bars_held=i,
                same_bar_ambiguous=True,
            )

        if hit_stop:
            return Resolution(
                outcome="stopped_out",
                resolved_date=bar_date,
                fill_price=stop,
                pnl_pct=_pnl_pct(entry, stop, direction, policy.cost_bps),
                bars_held=i,
            )

        if hit_target:
            return Resolution(
                outcome="hit_target",
                resolved_date=bar_date,
                fill_price=target,
                pnl_pct=_pnl_pct(entry, target, direction, policy.cost_bps),
                bars_held=i,
            )

    # Neither barrier touched.  Expire only once the full holding window has
    # actually elapsed; otherwise the prediction is still open.
    if len(window) >= policy.max_holding_bars:
        last_date = window.index[-1]
        last_close = float(window.iloc[-1]["close"])
        return Resolution(
            outcome="expired",
            resolved_date=last_date,
            fill_price=last_close,
            pnl_pct=_pnl_pct(entry, last_close, direction, policy.cost_bps),
            bars_held=len(window),
        )

    return None
