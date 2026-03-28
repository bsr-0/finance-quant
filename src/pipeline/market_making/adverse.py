"""Adverse selection / toxic flow detection for market-making.

Detects when fills are systematically followed by unfavorable price moves
(indicating informed order flow) and adjusts quoting behavior to reduce
losses to toxic counterparties.

Metrics tracked per symbol:
    - **Post-fill PnL**: Mark-to-market PnL at T+1, T+5, T+10 ticks/events
      after each fill.  Persistently negative values indicate adverse
      selection.
    - **Fill-side imbalance**: Whether fills are concentrated on one side
      (e.g. always getting lifted on the ask), suggesting one-directional
      informed flow.
    - **Spread capture rate**: Fraction of the quoted spread actually
      captured (net of post-fill price moves).
    - **Fill-to-cancel ratio**: Ratio of fills to quote cancellations.
      Very low ratios may indicate the MM is being picked off.

Assumptions:
    - The detector is fed fills and subsequent price observations in
      chronological order.
    - Price observations after fills can be at arbitrary intervals;
      the detector uses event counts, not wall-clock time.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AdverseConfig:
    """Configuration for adverse selection detection.

    Attributes:
        lookback_fills: Number of recent fills to consider in rolling
            toxicity metrics.
        horizons: List of event-count horizons for post-fill PnL measurement.
        toxicity_threshold: If average post-fill PnL at the shortest
            horizon is worse than this fraction of the spread, flag as toxic.
        widen_multiplier: Spread multiplier applied when toxicity is detected.
        size_reduction: Quote size multiplier when toxicity is detected.
        min_fills_for_signal: Minimum fills needed before toxicity metrics
            are considered reliable.
    """

    lookback_fills: int = 100
    horizons: list[int] = field(default_factory=lambda: [1, 5, 10])
    toxicity_threshold: float = -0.5
    widen_multiplier: float = 1.5
    size_reduction: float = 0.5
    min_fills_for_signal: int = 20


@dataclass
class FillRecord:
    """Record of a single fill for adverse selection analysis."""

    symbol: str
    side: str           # "buy" or "sell"
    fill_price: float
    fill_size: float
    mid_at_fill: float
    timestamp_ns: int
    spread_at_fill: float
    # Filled in later as post-fill prices arrive
    post_fill_mids: list[float] = field(default_factory=list)


@dataclass
class ToxicityMetrics:
    """Summary of adverse selection metrics for a symbol.

    Attributes:
        symbol: Instrument identifier.
        avg_post_fill_pnl: Dict of {horizon: avg signed PnL per unit}.
            Negative means the market moved against the MM after fills.
        spread_capture_rate: Fraction of quoted spread actually captured.
        fill_side_ratio: Ratio of buy fills to total fills (0.5 = balanced).
        is_toxic: Whether the current flow is flagged as toxic.
        recommended_widen: Suggested spread multiplier (1.0 if not toxic).
        recommended_size_mult: Suggested size multiplier (1.0 if not toxic).
        num_fills_analyzed: Number of fills used in the calculation.
    """

    symbol: str = ""
    avg_post_fill_pnl: dict[int, float] = field(default_factory=dict)
    spread_capture_rate: float = 0.0
    fill_side_ratio: float = 0.5
    is_toxic: bool = False
    recommended_widen: float = 1.0
    recommended_size_mult: float = 1.0
    num_fills_analyzed: int = 0


class AdverseSelectionDetector:
    """Detect toxic order flow and recommend quoting adjustments.

    Usage::

        detector = AdverseSelectionDetector()

        # When a fill occurs:
        detector.record_fill(fill_record)

        # After subsequent price observations:
        detector.record_post_fill_price(symbol, mid_price)

        # Check toxicity:
        metrics = detector.evaluate(symbol)
        if metrics.is_toxic:
            # Widen spreads or reduce size
    """

    def __init__(self, config: AdverseConfig | None = None) -> None:
        self.config = config or AdverseConfig()
        self._fills: dict[str, deque[FillRecord]] = {}
        self._pending_fills: dict[str, list[FillRecord]] = {}
        self._price_tick_count: dict[str, int] = {}

    def record_fill(self, fill: FillRecord) -> None:
        """Record a new fill for adverse selection analysis."""
        sym = fill.symbol
        if sym not in self._fills:
            self._fills[sym] = deque(maxlen=self.config.lookback_fills)
        self._fills[sym].append(fill)

        # Track this fill as pending (needs post-fill prices)
        if sym not in self._pending_fills:
            self._pending_fills[sym] = []
        self._pending_fills[sym].append(fill)

    def record_post_fill_price(self, symbol: str, mid_price: float) -> None:
        """Record a subsequent price observation after fills.

        Each call increments the tick counter for *symbol*.  Pending fills
        that haven't yet accumulated enough post-fill observations will
        receive this price.
        """
        count = self._price_tick_count.get(symbol, 0) + 1
        self._price_tick_count[symbol] = count

        max_horizon = max(self.config.horizons) if self.config.horizons else 10
        pending = self._pending_fills.get(symbol, [])
        still_pending = []
        for fill in pending:
            fill.post_fill_mids.append(mid_price)
            if len(fill.post_fill_mids) < max_horizon:
                still_pending.append(fill)
        self._pending_fills[symbol] = still_pending

    def evaluate(self, symbol: str) -> ToxicityMetrics:
        """Compute toxicity metrics for a symbol.

        Returns:
            ``ToxicityMetrics`` with the current assessment.
        """
        cfg = self.config
        fills_deque = self._fills.get(symbol)
        if not fills_deque or len(fills_deque) < cfg.min_fills_for_signal:
            return ToxicityMetrics(
                symbol=symbol,
                num_fills_analyzed=len(fills_deque) if fills_deque else 0,
            )

        fills = list(fills_deque)

        # Post-fill PnL at each horizon
        avg_pnl: dict[int, float] = {}
        for h in cfg.horizons:
            pnls = []
            for f in fills:
                if len(f.post_fill_mids) >= h:
                    future_mid = f.post_fill_mids[h - 1]
                    if f.side == "buy":
                        pnl = future_mid - f.fill_price
                    else:
                        pnl = f.fill_price - future_mid
                    # Normalize by spread at fill
                    if f.spread_at_fill > 0:
                        pnl /= f.spread_at_fill
                    pnls.append(pnl)
            avg_pnl[h] = float(np.mean(pnls)) if pnls else 0.0

        # Spread capture rate
        captures = []
        for f in fills:
            if f.spread_at_fill > 0 and len(f.post_fill_mids) >= 1:
                if f.side == "buy":
                    capture = (f.post_fill_mids[0] - f.fill_price) / f.spread_at_fill
                else:
                    capture = (f.fill_price - f.post_fill_mids[0]) / f.spread_at_fill
                captures.append(capture)
        spread_capture = float(np.mean(captures)) if captures else 0.0

        # Fill-side ratio
        buy_count = sum(1 for f in fills if f.side == "buy")
        side_ratio = buy_count / len(fills) if fills else 0.5

        # Toxicity determination
        shortest_h = min(cfg.horizons) if cfg.horizons else 1
        is_toxic = avg_pnl.get(shortest_h, 0.0) < cfg.toxicity_threshold

        recommended_widen = cfg.widen_multiplier if is_toxic else 1.0
        recommended_size = cfg.size_reduction if is_toxic else 1.0

        if is_toxic:
            logger.warning(
                "TOXIC FLOW %s: post-fill PnL=%s, capture=%.2f, side_ratio=%.2f",
                symbol,
                {k: f"{v:.3f}" for k, v in avg_pnl.items()},
                spread_capture,
                side_ratio,
            )

        return ToxicityMetrics(
            symbol=symbol,
            avg_post_fill_pnl=avg_pnl,
            spread_capture_rate=spread_capture,
            fill_side_ratio=side_ratio,
            is_toxic=is_toxic,
            recommended_widen=recommended_widen,
            recommended_size_mult=recommended_size,
            num_fills_analyzed=len(fills),
        )

    def per_side_toxicity(self, symbol: str) -> dict[str, float]:
        """Return average post-fill PnL separately for buy and sell fills.

        Useful for detecting one-directional informed flow.
        """
        fills_deque = self._fills.get(symbol)
        if not fills_deque:
            return {"buy": 0.0, "sell": 0.0}

        shortest_h = min(self.config.horizons) if self.config.horizons else 1
        sides: dict[str, list[float]] = {"buy": [], "sell": []}

        for f in fills_deque:
            if len(f.post_fill_mids) >= shortest_h:
                future_mid = f.post_fill_mids[shortest_h - 1]
                pnl = future_mid - f.fill_price if f.side == "buy" else f.fill_price - future_mid
                sides[f.side].append(pnl)

        return {
            side: float(np.mean(vals)) if vals else 0.0
            for side, vals in sides.items()
        }

    def summary(self, symbol: str) -> dict:
        """Return a serializable summary of toxicity metrics."""
        m = self.evaluate(symbol)
        return {
            "symbol": m.symbol,
            "is_toxic": m.is_toxic,
            "post_fill_pnl": m.avg_post_fill_pnl,
            "spread_capture": m.spread_capture_rate,
            "fill_side_ratio": m.fill_side_ratio,
            "recommended_widen": m.recommended_widen,
            "recommended_size_mult": m.recommended_size_mult,
            "num_fills": m.num_fills_analyzed,
        }
