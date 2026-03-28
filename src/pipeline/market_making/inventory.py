"""Inventory management for market-making.

Tracks per-symbol and portfolio-level inventory, computes risk penalties
for quoting logic, and enforces hard inventory limits with automatic
de-risking behavior.

Design:
    Inventory risk is modeled as a quadratic penalty:
        penalty(q) = gamma * q^2
    where q is the inventory position and gamma is the risk-aversion
    coefficient.  This is consistent with the Avellaneda-Stoikov framework
    and penalizes large inventories super-linearly.

Assumptions:
    - Positions are tracked in units of the base asset (shares/contracts).
    - PnL is computed mark-to-market using the latest fair value.
    - De-risking behavior is deterministic: once a threshold is crossed,
      the response (skew, size reduction, or hard flat) is immediate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

logger = logging.getLogger(__name__)


class InventoryLevel(IntEnum):
    """De-risking severity levels based on inventory utilization."""

    NORMAL = 0  # |inventory| < 50% of limit
    ELEVATED = 1  # |inventory| >= 50% of limit — skew quotes
    CRITICAL = 2  # |inventory| >= 80% of limit — reduce size + skew
    BREACH = 3  # |inventory| >= 100% of limit — hard flat trigger


@dataclass
class InventoryConfig:
    """Configuration for inventory management.

    Attributes:
        max_position: Hard limit on absolute position per symbol (shares).
        max_notional: Hard limit on total notional exposure per symbol.
        max_portfolio_notional: Hard limit on aggregate portfolio notional.
        gamma: Risk-aversion coefficient for quadratic inventory penalty.
        target_position: Target inventory level (usually 0 for market-makers).
        elevated_threshold: Fraction of max_position triggering ELEVATED.
        critical_threshold: Fraction of max_position triggering CRITICAL.
        mean_revert_speed: How fast to revert to target (0 = slow, 1 = fast).
        size_reduction_at_critical: Multiplier on quote size at CRITICAL level.
    """

    max_position: float = 10_000
    max_notional: float = 5_000_000.0
    max_portfolio_notional: float = 50_000_000.0
    gamma: float = 0.001
    target_position: float = 0.0
    elevated_threshold: float = 0.50
    critical_threshold: float = 0.80
    mean_revert_speed: float = 0.3
    size_reduction_at_critical: float = 0.25


@dataclass
class SymbolInventory:
    """Per-symbol inventory state."""

    symbol: str
    position: float = 0.0
    avg_cost: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    num_fills: int = 0
    total_volume: float = 0.0

    def update_mark(self, fair_value: float) -> None:
        """Update unrealized PnL to the latest fair value."""
        if self.position != 0 and self.avg_cost > 0:
            self.unrealized_pnl = (fair_value - self.avg_cost) * self.position
        else:
            self.unrealized_pnl = 0.0


@dataclass
class InventorySnapshot:
    """Summary of the current inventory state across all symbols."""

    total_long_notional: float
    total_short_notional: float
    gross_notional: float
    net_notional: float
    total_realized_pnl: float
    total_unrealized_pnl: float
    symbol_count: int
    worst_symbol: str
    worst_utilization: float


class InventoryManager:
    """Track and manage market-making inventory across symbols.

    Provides:
    - Fill recording with average-cost accounting.
    - Inventory risk penalty for the spread calculator.
    - De-risking level computation for the quoting engine.
    - Portfolio-level aggregation and limit checks.
    """

    def __init__(self, config: InventoryConfig | None = None) -> None:
        self.config = config or InventoryConfig()
        self._inventories: dict[str, SymbolInventory] = {}

    def get_or_create(self, symbol: str) -> SymbolInventory:
        """Return the inventory record for *symbol*, creating if needed."""
        if symbol not in self._inventories:
            self._inventories[symbol] = SymbolInventory(symbol=symbol)
        return self._inventories[symbol]

    def record_fill(
        self,
        symbol: str,
        quantity: float,
        price: float,
    ) -> None:
        """Record a fill and update average cost.

        Args:
            symbol: Instrument identifier.
            quantity: Signed quantity (positive = buy, negative = sell).
            price: Execution price.
        """
        inv = self.get_or_create(symbol)
        old_pos = inv.position
        new_pos = old_pos + quantity

        if old_pos == 0:
            inv.avg_cost = price
        elif np.sign(old_pos) == np.sign(quantity):
            # Adding to the position — weighted average cost
            total_cost = inv.avg_cost * abs(old_pos) + price * abs(quantity)
            inv.avg_cost = total_cost / abs(new_pos) if new_pos != 0 else 0.0
        else:
            # Reducing or flipping — realize PnL on the closed portion
            closed_qty = min(abs(old_pos), abs(quantity))
            pnl = (price - inv.avg_cost) * closed_qty * np.sign(old_pos)
            inv.realized_pnl += pnl
            if abs(new_pos) > 0 and np.sign(new_pos) != np.sign(old_pos):
                # Position flipped — new cost basis is the fill price
                inv.avg_cost = price
            # If partially reduced, avg_cost stays the same

        inv.position = new_pos
        inv.num_fills += 1
        inv.total_volume += abs(quantity) * price

        logger.debug(
            "FILL %s: qty=%.1f @ %.4f → pos=%.1f, avg_cost=%.4f, rpnl=%.2f",
            symbol,
            quantity,
            price,
            new_pos,
            inv.avg_cost,
            inv.realized_pnl,
        )

    def normalized_inventory(self, symbol: str) -> float:
        """Return inventory as a fraction of the maximum position.

        Result is in [-1, 1].  Used as input to the spread calculator's
        inventory skew.
        """
        inv = self.get_or_create(symbol)
        if self.config.max_position <= 0:
            return 0.0
        return np.clip(inv.position / self.config.max_position, -1.0, 1.0)

    def inventory_risk_penalty(self, symbol: str) -> float:
        """Quadratic risk penalty for current inventory.

        Returns gamma * (q - target)^2, which can be subtracted from the
        fair value to bias quotes toward inventory reduction.
        """
        inv = self.get_or_create(symbol)
        deviation = inv.position - self.config.target_position
        return self.config.gamma * deviation**2

    def get_inventory_level(self, symbol: str) -> InventoryLevel:
        """Determine the de-risking severity level for a symbol."""
        utilization = abs(self.normalized_inventory(symbol))
        if utilization >= 1.0:
            return InventoryLevel.BREACH
        if utilization >= self.config.critical_threshold:
            return InventoryLevel.CRITICAL
        if utilization >= self.config.elevated_threshold:
            return InventoryLevel.ELEVATED
        return InventoryLevel.NORMAL

    def quote_size_multiplier(self, symbol: str) -> float:
        """Return a multiplier for the quote size based on inventory level.

        At NORMAL: 1.0 (full size).
        At ELEVATED: linear ramp-down.
        At CRITICAL: ``size_reduction_at_critical``.
        At BREACH: 0.0 on the side that would increase exposure.
        """
        level = self.get_inventory_level(symbol)
        if level == InventoryLevel.BREACH:
            return 0.0
        if level == InventoryLevel.CRITICAL:
            return self.config.size_reduction_at_critical
        if level == InventoryLevel.ELEVATED:
            utilization = abs(self.normalized_inventory(symbol))
            # Linear ramp from 1.0 at elevated threshold to
            # size_reduction at critical threshold
            t = (utilization - self.config.elevated_threshold) / (
                self.config.critical_threshold - self.config.elevated_threshold
            )
            t = np.clip(t, 0.0, 1.0)
            return 1.0 - t * (1.0 - self.config.size_reduction_at_critical)
        return 1.0

    def should_send_aggressor(self, symbol: str) -> bool:
        """Return True if inventory is at BREACH and an aggressive order
        should be sent to flatten the position immediately."""
        return self.get_inventory_level(symbol) == InventoryLevel.BREACH

    def update_marks(self, fair_values: dict[str, float]) -> None:
        """Mark all positions to their current fair values."""
        for symbol, fv in fair_values.items():
            if symbol in self._inventories:
                self._inventories[symbol].update_mark(fv)

    def snapshot(self) -> InventorySnapshot:
        """Aggregate portfolio-level inventory statistics."""
        total_long = 0.0
        total_short = 0.0
        total_rpnl = 0.0
        total_upnl = 0.0
        worst_symbol = ""
        worst_util = 0.0

        for symbol, inv in self._inventories.items():
            notional = abs(inv.position * inv.avg_cost) if inv.avg_cost > 0 else 0
            if inv.position > 0:
                total_long += notional
            elif inv.position < 0:
                total_short += notional
            total_rpnl += inv.realized_pnl
            total_upnl += inv.unrealized_pnl

            util = abs(self.normalized_inventory(symbol))
            if util > worst_util:
                worst_util = util
                worst_symbol = symbol

        return InventorySnapshot(
            total_long_notional=total_long,
            total_short_notional=total_short,
            gross_notional=total_long + total_short,
            net_notional=abs(total_long - total_short),
            total_realized_pnl=total_rpnl,
            total_unrealized_pnl=total_upnl,
            symbol_count=len(self._inventories),
            worst_symbol=worst_symbol,
            worst_utilization=worst_util,
        )

    def check_portfolio_limits(self) -> tuple[bool, str]:
        """Check if aggregate portfolio notional exceeds the hard limit.

        Returns:
            ``(passes, reason)`` — True if within limits.
        """
        snap = self.snapshot()
        if snap.gross_notional > self.config.max_portfolio_notional:
            return False, (
                f"Gross notional {snap.gross_notional:,.0f} exceeds "
                f"portfolio limit {self.config.max_portfolio_notional:,.0f}"
            )
        return True, "OK"
