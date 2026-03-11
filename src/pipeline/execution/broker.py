"""Abstract broker interface and order lifecycle types.

Defines the contract that any broker implementation (Alpaca, IBKR, etc.) must
fulfill.  The interface is deliberately minimal — just enough for the swing
strategy's needs (market/limit orders, position queries, account info).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from pipeline.execution.capital_guard import AccountSnapshot

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Order types
# ---------------------------------------------------------------------------

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial_fill"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class Order:
    """Represents an order through its lifecycle."""

    symbol: str
    side: OrderSide
    order_type: OrderType
    qty: float
    limit_price: float | None = None
    stop_price: float | None = None

    # Filled by broker
    order_id: str = ""
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: float = 0.0
    filled_avg_price: float = 0.0
    submitted_at: datetime | None = None
    filled_at: datetime | None = None
    reject_reason: str = ""
    broker_details: dict[str, Any] = field(default_factory=dict)

    @property
    def notional(self) -> float:
        """Estimated notional value (uses limit price or filled price)."""
        price = self.filled_avg_price or self.limit_price or 0.0
        return abs(self.qty * price)


@dataclass(frozen=True)
class Position:
    """A currently held position as reported by the broker."""

    symbol: str
    qty: float
    market_value: float
    avg_entry_price: float
    current_price: float
    unrealised_pnl: float
    side: str  # "long" or "short"


# ---------------------------------------------------------------------------
# Abstract broker
# ---------------------------------------------------------------------------

class BaseBroker(ABC):
    """Abstract broker interface.

    Implementations must provide account snapshots, order submission,
    order status queries, and position queries.
    """

    @abstractmethod
    def get_account_snapshot(self) -> AccountSnapshot:
        """Fetch current account state from the broker.

        This MUST query the broker's API directly — never return cached
        or internally-tracked values.
        """

    @abstractmethod
    def submit_order(self, order: Order) -> Order:
        """Submit an order to the broker.

        Args:
            order: The order to submit.

        Returns:
            The same order object with broker-assigned fields populated
            (order_id, status, submitted_at).

        Raises:
            BrokerError: If the broker rejects the order or the API call fails.
        """

    @abstractmethod
    def get_order_status(self, order_id: str) -> Order:
        """Query the current status of a submitted order."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order.  Returns True if successfully cancelled."""

    @abstractmethod
    def get_positions(self) -> list[Position]:
        """Get all currently open positions from the broker."""

    @abstractmethod
    def close_position(self, symbol: str) -> Order:
        """Close an entire position in a symbol (market sell/cover)."""

    @abstractmethod
    def close_all_positions(self) -> list[Order]:
        """Close ALL open positions.  Used by RED circuit breaker."""


class BrokerError(Exception):
    """Raised when a broker operation fails."""

    def __init__(self, message: str, broker_code: str = "", details: dict[str, Any] | None = None):
        super().__init__(message)
        self.broker_code = broker_code
        self.details = details or {}
