"""Alpaca broker implementation.

Wraps the Alpaca Trading API to implement the BaseBroker interface.
Designed for cash accounts with $100-$500 capital.

Setup::

    # In .env:
    ALPACA_API_KEY=your_key
    ALPACA_SECRET_KEY=your_secret
    ALPACA_BASE_URL=https://paper-api.alpaca.markets  # paper trading
    # ALPACA_BASE_URL=https://api.alpaca.markets      # live trading

    # Usage:
    broker = AlpacaBroker.from_env()
    snapshot = broker.get_account_snapshot()

Note: This module imports ``alpaca-py`` (``pip install alpaca-py``).
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone

from pipeline.execution.broker import (
    BaseBroker,
    BrokerError,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)
from pipeline.execution.capital_guard import AccountSnapshot

logger = logging.getLogger(__name__)

# Map Alpaca order status strings to our enum
_STATUS_MAP = {
    "new": OrderStatus.SUBMITTED,
    "accepted": OrderStatus.SUBMITTED,
    "pending_new": OrderStatus.PENDING,
    "accepted_for_bidding": OrderStatus.SUBMITTED,
    "partially_filled": OrderStatus.PARTIAL,
    "filled": OrderStatus.FILLED,
    "canceled": OrderStatus.CANCELLED,
    "cancelled": OrderStatus.CANCELLED,
    "expired": OrderStatus.EXPIRED,
    "rejected": OrderStatus.REJECTED,
    "pending_cancel": OrderStatus.SUBMITTED,
    "pending_replace": OrderStatus.SUBMITTED,
    "stopped": OrderStatus.FILLED,
    "suspended": OrderStatus.REJECTED,
    "done_for_day": OrderStatus.CANCELLED,
    "replaced": OrderStatus.SUBMITTED,
}

_SENSITIVE_RE = re.compile(
    r"(api[_-]?key|secret|token|authorization|password|credential)[=:]\s*\S+(\s+\S+)?",
    re.IGNORECASE,
)


def _sanitize_error(e: Exception) -> str:
    """Strip potential auth details from exception messages."""
    msg = str(e)
    msg = _SENSITIVE_RE.sub(r"\1=***", msg)
    if len(msg) > 200:
        msg = msg[:200] + "..."
    return msg


class AlpacaBroker(BaseBroker):
    """Alpaca Trading API broker implementation.

    Supports both paper and live trading via the base URL.
    Fractional shares are supported for micro-capital accounts.
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        base_url: str = "https://paper-api.alpaca.markets",
    ) -> None:
        try:
            from alpaca.trading.client import TradingClient
        except ImportError:
            raise ImportError(
                "alpaca-py is required for Alpaca broker. "
                "Install with: pip install alpaca-py"
            )

        self._is_paper = "paper" in base_url.lower()
        self._client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=self._is_paper,
        )
        logger.info(
            "Alpaca broker initialized (mode=%s)",
            "PAPER" if self._is_paper else "LIVE",
        )

    @classmethod
    def from_env(cls) -> AlpacaBroker:
        """Create an AlpacaBroker from environment variables."""
        api_key = os.environ.get("ALPACA_API_KEY", "")
        secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
        base_url = os.environ.get(
            "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
        )

        if not api_key or not secret_key:
            raise BrokerError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in environment. "
                "Get keys at https://app.alpaca.markets/brokerage/dashboard/overview"
            )

        return cls(api_key=api_key, secret_key=secret_key, base_url=base_url)

    def get_account_snapshot(self) -> AccountSnapshot:
        """Fetch live account state from Alpaca."""
        try:
            acct = self._client.get_account()
        except Exception as e:
            raise BrokerError(f"Failed to fetch Alpaca account: {_sanitize_error(e)}")

        equity = float(acct.equity)
        cash = float(acct.cash)
        buying_power = float(acct.buying_power)
        long_value = float(acct.long_market_value)
        short_value = float(acct.short_market_value)
        positions_value = long_value + abs(short_value)

        # Determine margin status
        # Alpaca: multiplier "1" = cash account, "2"/"4" = margin
        is_margin = str(getattr(acct, "multiplier", "1")) != "1"

        # Count positions
        try:
            positions = self._client.get_all_positions()
            position_count = len(positions)
        except Exception:
            position_count = 0

        snapshot = AccountSnapshot(
            equity=equity,
            cash=cash,
            buying_power=buying_power,
            positions_market_value=positions_value,
            position_count=position_count,
            is_margin_account=is_margin,
        )

        logger.debug(
            "Account snapshot: equity=$%.2f cash=$%.2f buying_power=$%.2f "
            "positions=$%.2f margin=%s",
            equity, cash, buying_power, positions_value, is_margin,
        )

        return snapshot

    def submit_order(self, order: Order) -> Order:
        """Submit an order to Alpaca."""
        from alpaca.trading.requests import (
            LimitOrderRequest,
            MarketOrderRequest,
            OrderSide as AlpacaSide,
        )
        from alpaca.trading.enums import TimeInForce

        try:
            side = AlpacaSide.BUY if order.side == OrderSide.BUY else AlpacaSide.SELL

            if order.order_type == OrderType.MARKET:
                request = MarketOrderRequest(
                    symbol=order.symbol,
                    qty=order.qty,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                )
            elif order.order_type == OrderType.LIMIT:
                if order.limit_price is None:
                    raise BrokerError("Limit price required for limit orders")
                request = LimitOrderRequest(
                    symbol=order.symbol,
                    qty=order.qty,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=order.limit_price,
                )
            else:
                raise BrokerError(f"Unsupported order type: {order.order_type}")

            result = self._client.submit_order(request)

            order.order_id = str(result.id)
            order.status = _STATUS_MAP.get(str(result.status), OrderStatus.SUBMITTED)
            order.submitted_at = datetime.now(timezone.utc)

            if result.filled_qty:
                order.filled_qty = float(result.filled_qty)
            if result.filled_avg_price:
                order.filled_avg_price = float(result.filled_avg_price)

            logger.info(
                "Order submitted: %s %s %.4f %s @ %s → id=%s status=%s",
                order.side.value, order.symbol, order.qty,
                order.order_type.value,
                f"${order.limit_price:.2f}" if order.limit_price else "MKT",
                order.order_id, order.status.value,
            )

            return order

        except BrokerError:
            raise
        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.reject_reason = _sanitize_error(e)
            raise BrokerError(f"Order submission failed: {_sanitize_error(e)}")

    def get_order_status(self, order_id: str) -> Order:
        """Query order status from Alpaca."""
        try:
            result = self._client.get_order_by_id(order_id)

            order = Order(
                symbol=result.symbol,
                side=OrderSide.BUY if str(result.side) == "buy" else OrderSide.SELL,
                order_type=OrderType.LIMIT if result.limit_price else OrderType.MARKET,
                qty=float(result.qty),
                limit_price=float(result.limit_price) if result.limit_price else None,
                order_id=str(result.id),
                status=_STATUS_MAP.get(str(result.status), OrderStatus.SUBMITTED),
                filled_qty=float(result.filled_qty) if result.filled_qty else 0.0,
                filled_avg_price=float(result.filled_avg_price) if result.filled_avg_price else 0.0,
            )
            return order
        except Exception as e:
            raise BrokerError(f"Failed to get order {order_id}: {_sanitize_error(e)}")

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order on Alpaca."""
        try:
            self._client.cancel_order_by_id(order_id)
            logger.info("Order %s cancelled", order_id)
            return True
        except Exception as e:
            logger.warning("Failed to cancel order %s: %s", order_id, e)
            return False

    def get_positions(self) -> list[Position]:
        """Get all open positions from Alpaca."""
        try:
            alpaca_positions = self._client.get_all_positions()
        except Exception as e:
            raise BrokerError(f"Failed to fetch positions: {_sanitize_error(e)}")

        positions = []
        for p in alpaca_positions:
            qty = float(p.qty)
            positions.append(Position(
                symbol=p.symbol,
                qty=qty,
                market_value=float(p.market_value),
                avg_entry_price=float(p.avg_entry_price),
                current_price=float(p.current_price),
                unrealised_pnl=float(p.unrealized_pl),
                side="long" if qty > 0 else "short",
            ))

        return positions

    def close_position(self, symbol: str) -> Order:
        """Close an entire position in a symbol."""
        try:
            result = self._client.close_position(symbol)
            order = Order(
                symbol=symbol,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                qty=float(result.qty) if result.qty else 0.0,
                order_id=str(result.id),
                status=_STATUS_MAP.get(str(result.status), OrderStatus.SUBMITTED),
            )
            logger.info("Close position submitted for %s → order %s", symbol, order.order_id)
            return order
        except Exception as e:
            raise BrokerError(f"Failed to close position {symbol}: {_sanitize_error(e)}")

    def close_all_positions(self) -> list[Order]:
        """Close ALL positions (RED circuit breaker action)."""
        logger.critical("CLOSING ALL POSITIONS — RED circuit breaker")
        try:
            results = self._client.close_all_positions(cancel_orders=True)
        except Exception as e:
            raise BrokerError(f"Failed to close all positions: {_sanitize_error(e)}")

        orders = []
        for r in results:
            if hasattr(r, "body") and hasattr(r.body, "id"):
                body = r.body
                orders.append(Order(
                    symbol=body.symbol if hasattr(body, "symbol") else "UNKNOWN",
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    qty=float(body.qty) if hasattr(body, "qty") and body.qty else 0.0,
                    order_id=str(body.id),
                    status=OrderStatus.SUBMITTED,
                ))

        logger.critical("Submitted %d close-all orders", len(orders))
        return orders
