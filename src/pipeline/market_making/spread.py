"""Dynamic spread calculation for market-making.

Computes quoted bid/ask prices around a fair-value estimate.  Spreads adapt
to realized volatility, order-book imbalance, and competitive pressure
rather than using static tick offsets.

Theory:
    The quoted half-spread has three components:
    1. **Base spread** — minimum compensation for adverse selection risk,
       proportional to short-term volatility (Avellaneda-Stoikov insight).
    2. **Inventory penalty** — widens the side where the market-maker is
       already over-exposed, incentivizing mean-reversion of inventory.
    3. **Regime/competition adjustment** — multiplicative factor that
       tightens in calm/competitive markets and widens under stress.

Assumptions:
    - Fair value (mid or micro-price) is supplied externally.
    - Volatility is realized (not implied) and can be estimated from
      recent trade/quote data.
    - Order-book imbalance is a scalar in [-1, 1] where +1 means
      all resting interest is on the bid side (buy pressure).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SpreadConfig:
    """Configuration for the spread calculator.

    Attributes:
        min_spread_bps: Hard floor on the total spread in basis points.
        max_spread_bps: Hard ceiling on the total spread in basis points.
        vol_scale: Multiplier on realized volatility to set the base spread
            (higher → wider spreads per unit vol).
        imbalance_scale: How much order-book imbalance shifts the mid
            (0 = ignore imbalance, 1 = full shift).
        inventory_skew_scale: How aggressively to skew the spread as
            inventory grows (bps per unit of normalized inventory).
        stress_multiplier_max: Maximum multiplier applied under extreme
            stress conditions (e.g. 3.0 means spreads can triple).
        competitive_tightening: Factor to tighten spreads in competitive
            environments (0 = no tightening, 1 = halve the spread).
        tick_size: Minimum price increment for rounding quotes.
    """

    min_spread_bps: float = 2.0
    max_spread_bps: float = 200.0
    vol_scale: float = 1.5
    imbalance_scale: float = 0.3
    inventory_skew_scale: float = 5.0
    stress_multiplier_max: float = 3.0
    competitive_tightening: float = 0.0
    tick_size: float = 0.01


@dataclass
class QuotedSpread:
    """Result of spread computation.

    Attributes:
        bid: Quoted bid price.
        ask: Quoted ask price.
        mid: Fair-value mid price used in the calculation.
        half_spread: Half of the total spread (in price units).
        spread_bps: Total spread in basis points of the mid price.
        inventory_skew: Price-shift applied due to inventory (positive
            means mid shifted up, i.e. the MM wants to sell).
    """

    bid: float
    ask: float
    mid: float
    half_spread: float
    spread_bps: float
    inventory_skew: float


class SpreadCalculator:
    """Compute dynamic bid/ask spreads around a fair value.

    Inputs are real-time scalars (volatility, imbalance, inventory).
    The calculator is stateless — all state lives in the caller.
    """

    def __init__(self, config: SpreadConfig | None = None) -> None:
        self.config = config or SpreadConfig()

    def compute(
        self,
        fair_value: float,
        volatility: float,
        inventory_normalized: float = 0.0,
        order_book_imbalance: float = 0.0,
        stress_level: float = 0.0,
        competition_factor: float = 0.0,
    ) -> QuotedSpread:
        """Compute bid/ask prices.

        Args:
            fair_value: Estimated fair mid-price.
            volatility: Short-term realized volatility (annualized or
                per-period — must be consistent with ``vol_scale``).
            inventory_normalized: Current inventory divided by the
                maximum allowed inventory.  In [-1, 1], where +1 means
                max long, -1 means max short.
            order_book_imbalance: Bid-side weight minus ask-side weight,
                in [-1, 1].
            stress_level: Continuous stress indicator in [0, 1].
                0 = normal, 1 = extreme stress.
            competition_factor: How tight competition is, in [0, 1].
                0 = no competition, 1 = maximum competition.

        Returns:
            ``QuotedSpread`` with the resulting bid/ask and diagnostics.
        """
        cfg = self.config

        # 1. Base half-spread from volatility
        base_half_bps = cfg.vol_scale * volatility * 10_000
        base_half_bps = max(base_half_bps, cfg.min_spread_bps / 2)

        # 2. Stress widening (linear interpolation toward max multiplier)
        stress_mult = 1.0 + stress_level * (cfg.stress_multiplier_max - 1.0)
        half_bps = base_half_bps * stress_mult

        # 3. Competitive tightening
        tighten = 1.0 - competition_factor * cfg.competitive_tightening
        half_bps *= max(tighten, 0.5)  # never tighten more than 50%

        # 4. Enforce min/max bounds
        total_bps = np.clip(half_bps * 2, cfg.min_spread_bps, cfg.max_spread_bps)
        half_bps = total_bps / 2

        half_spread_price = fair_value * half_bps / 10_000

        # 5. Inventory skew: shift the mid so that the overweight side
        #    is quoted more aggressively (tighter) to encourage fills
        #    that reduce inventory.
        skew_bps = cfg.inventory_skew_scale * inventory_normalized
        inventory_skew = fair_value * skew_bps / 10_000

        # 6. Order-book imbalance adjustment (micro-price)
        imbalance_shift = fair_value * cfg.imbalance_scale * order_book_imbalance / 10_000

        adjusted_mid = fair_value + inventory_skew + imbalance_shift

        # 7. Final quotes
        raw_bid = adjusted_mid - half_spread_price
        raw_ask = adjusted_mid + half_spread_price

        # Round to tick
        bid = self._round_down(raw_bid, cfg.tick_size)
        ask = self._round_up(raw_ask, cfg.tick_size)

        # Ensure bid < ask after rounding
        if bid >= ask:
            ask = bid + cfg.tick_size

        return QuotedSpread(
            bid=bid,
            ask=ask,
            mid=fair_value,
            half_spread=half_spread_price,
            spread_bps=total_bps,
            inventory_skew=inventory_skew,
        )

    @staticmethod
    def _round_down(price: float, tick: float) -> float:
        return np.floor(price / tick) * tick

    @staticmethod
    def _round_up(price: float, tick: float) -> float:
        return np.ceil(price / tick) * tick
