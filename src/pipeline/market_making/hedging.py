"""Hedging strategies for market-making inventory risk.

Provides cost-aware hedging logic that reduces portfolio risk without
systematically eliminating the market-maker's spread alpha.

Design principles:
    1. **Risk-driven, not fill-driven**: Hedge decisions are triggered by
       risk limits (delta, inventory, sector), not by every individual fill.
    2. **Cost-aware sizing**: Hedge trades are sized to balance risk
       reduction against the transaction costs of hedging.
    3. **No over-hedging**: Small fills within normal inventory bounds
       are NOT hedged — the spread revenue depends on carrying inventory
       temporarily.
    4. **Multi-instrument**: Supports hedging with correlated instruments
       (e.g. futures vs spot, ETF vs basket).

Assumptions:
    - Hedge instruments are liquid enough that impact costs are small.
    - Delta/sensitivity mappings (e.g. symbol → hedge instrument) are
      provided externally.
    - The hedging engine is advisory: it produces recommended trades,
      not executed ones.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class HedgeConfig:
    """Configuration for the hedge manager.

    Attributes:
        hedge_threshold_pct: Don't hedge until inventory exceeds this
            fraction of the maximum position (avoids over-hedging
            small fills).
        target_hedge_ratio: Fraction of excess inventory to hedge
            (1.0 = full hedge, 0.5 = half-hedge).
        max_hedge_cost_bps: Maximum acceptable cost for a hedge trade
            (in bps of notional).  If estimated cost exceeds this,
            the hedge is deferred.
        min_hedge_notional: Minimum notional value for a hedge trade
            (below this, skip to avoid nuisance trades).
        hedge_instruments: Mapping from symbol to its hedge instrument
            (e.g. {"AAPL": "QQQ", "MSFT": "QQQ"}).
        hedge_ratios: Beta / delta ratio between symbol and hedge
            instrument (e.g. {"AAPL": 1.2} means hedge 1.2 units of
            QQQ per unit of AAPL).
        sector_exposure_limit: Maximum net notional per sector before
            sector-level hedging kicks in.
        rebalance_interval_s: Minimum seconds between hedge rebalances.
    """

    hedge_threshold_pct: float = 0.30
    target_hedge_ratio: float = 0.7
    max_hedge_cost_bps: float = 5.0
    min_hedge_notional: float = 10_000.0
    hedge_instruments: dict[str, str] = field(default_factory=dict)
    hedge_ratios: dict[str, float] = field(default_factory=dict)
    sector_exposure_limit: float = 5_000_000.0
    rebalance_interval_s: float = 5.0


@dataclass
class HedgeTrade:
    """A recommended hedge trade."""

    symbol: str
    hedge_instrument: str
    side: str  # "buy" or "sell"
    quantity: float
    estimated_cost_bps: float
    reason: str
    urgency: str = "normal"  # "normal", "urgent", "critical"


@dataclass
class HedgeState:
    """Current state of hedging activity for a symbol."""

    symbol: str
    current_hedge_position: float = 0.0
    target_hedge_position: float = 0.0
    unhedged_exposure: float = 0.0
    last_rebalance_ns: int = 0


class HedgeManager:
    """Compute and manage hedging decisions for market-making inventory.

    Usage::

        mgr = HedgeManager(config)

        # After inventory changes:
        trades = mgr.compute_hedges(
            inventories={"AAPL": 5000, "MSFT": -3000},
            prices={"AAPL": 190.0, "MSFT": 420.0, "QQQ": 450.0},
            max_positions={"AAPL": 10000, "MSFT": 10000},
        )
        for trade in trades:
            execute(trade)
    """

    def __init__(self, config: HedgeConfig | None = None) -> None:
        self.config = config or HedgeConfig()
        self._hedge_state: dict[str, HedgeState] = {}
        self._current_hedges: dict[str, float] = {}  # hedge_instrument → position

    def compute_hedges(
        self,
        inventories: dict[str, float],
        prices: dict[str, float],
        max_positions: dict[str, float],
        timestamp_ns: int = 0,
        cost_estimate_bps: float = 2.0,
    ) -> list[HedgeTrade]:
        """Compute recommended hedge trades.

        Args:
            inventories: Current inventory per symbol (signed).
            prices: Current prices for all instruments (symbols + hedges).
            max_positions: Maximum allowed position per symbol.
            timestamp_ns: Current timestamp for rebalance throttling.
            cost_estimate_bps: Estimated round-trip cost for hedge trades.

        Returns:
            List of ``HedgeTrade`` recommendations.
        """
        cfg = self.config
        trades: list[HedgeTrade] = []

        # Aggregate exposure by hedge instrument
        hedge_needs: dict[str, float] = {}  # hedge_instrument → target delta

        for symbol, position in inventories.items():
            max_pos = max_positions.get(symbol, 10_000)
            utilization = abs(position) / max_pos if max_pos > 0 else 0
            price = prices.get(symbol, 0)

            if utilization < cfg.hedge_threshold_pct:
                continue  # Don't hedge small inventories

            hedge_instr = cfg.hedge_instruments.get(symbol)
            if not hedge_instr:
                continue  # No hedge instrument configured

            beta = cfg.hedge_ratios.get(symbol, 1.0)
            excess = position - np.sign(position) * max_pos * cfg.hedge_threshold_pct
            target_hedge = -excess * beta * cfg.target_hedge_ratio

            # Throttle: don't rebalance too frequently
            state = self._hedge_state.get(symbol)
            if state and timestamp_ns > 0:
                elapsed_s = (timestamp_ns - state.last_rebalance_ns) / 1e9
                if elapsed_s < cfg.rebalance_interval_s:
                    continue

            # Accumulate by hedge instrument
            hedge_needs[hedge_instr] = hedge_needs.get(hedge_instr, 0) + target_hedge

            # Update state
            if symbol not in self._hedge_state:
                self._hedge_state[symbol] = HedgeState(symbol=symbol)
            self._hedge_state[symbol].target_hedge_position = target_hedge
            self._hedge_state[symbol].unhedged_exposure = position * price
            self._hedge_state[symbol].last_rebalance_ns = timestamp_ns

        # Convert net hedge needs into trades
        for hedge_instr, target_delta in hedge_needs.items():
            current = self._current_hedges.get(hedge_instr, 0)
            trade_qty = target_delta - current

            if abs(trade_qty) < 1:
                continue

            hedge_price = prices.get(hedge_instr, 0)
            if hedge_price <= 0:
                continue

            notional = abs(trade_qty) * hedge_price
            if notional < cfg.min_hedge_notional:
                continue

            if cost_estimate_bps > cfg.max_hedge_cost_bps:
                logger.info(
                    "DEFER hedge %s: cost %.1f bps > limit %.1f bps",
                    hedge_instr,
                    cost_estimate_bps,
                    cfg.max_hedge_cost_bps,
                )
                continue

            side = "buy" if trade_qty > 0 else "sell"
            urgency = "urgent" if abs(trade_qty) > abs(current) * 2 else "normal"

            trades.append(
                HedgeTrade(
                    symbol="",  # Aggregate across symbols
                    hedge_instrument=hedge_instr,
                    side=side,
                    quantity=abs(trade_qty),
                    estimated_cost_bps=cost_estimate_bps,
                    reason=f"delta_rebalance: need={target_delta:.0f}, current={current:.0f}",
                    urgency=urgency,
                )
            )

            self._current_hedges[hedge_instr] = target_delta

        return trades

    def get_unhedged_exposure(
        self,
        inventories: dict[str, float],
        prices: dict[str, float],
    ) -> dict[str, float]:
        """Return the unhedged notional exposure per symbol.

        Factors in existing hedge positions to show the residual risk.
        """
        result = {}
        for symbol, position in inventories.items():
            price = prices.get(symbol, 0)
            gross = position * price
            hedge_instr = self.config.hedge_instruments.get(symbol)
            if hedge_instr:
                beta = self.config.hedge_ratios.get(symbol, 1.0)
                hedge_pos = self._current_hedges.get(hedge_instr, 0)
                hedged_notional = hedge_pos * prices.get(hedge_instr, 0) / beta
                result[symbol] = gross + hedged_notional
            else:
                result[symbol] = gross
        return result

    def reset(self) -> None:
        """Clear all hedge state (e.g. on session restart)."""
        self._hedge_state.clear()
        self._current_hedges.clear()
