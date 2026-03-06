"""Tests for the Alpaca broker implementation with mocked alpaca-py.

Validates order submission, status mapping, position queries, and
account snapshots without requiring real API keys or network access.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from pipeline.execution.broker import (
    BrokerError,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)
from pipeline.execution.capital_guard import AccountSnapshot


# ---------------------------------------------------------------------------
# Helpers: mock Alpaca SDK objects
# ---------------------------------------------------------------------------

def _mock_alpaca_account(
    equity="500.00",
    cash="300.00",
    buying_power="300.00",
    long_market_value="200.00",
    short_market_value="0.00",
    multiplier="1",
):
    acct = MagicMock()
    acct.equity = equity
    acct.cash = cash
    acct.buying_power = buying_power
    acct.long_market_value = long_market_value
    acct.short_market_value = short_market_value
    acct.multiplier = multiplier
    return acct


def _mock_alpaca_order(
    order_id="abc-123",
    status="new",
    symbol="AAPL",
    side="buy",
    qty="2.0",
    filled_qty=None,
    filled_avg_price=None,
    limit_price="150.00",
):
    order = MagicMock()
    order.id = order_id
    order.status = status
    order.symbol = symbol
    order.side = side
    order.qty = qty
    order.filled_qty = filled_qty
    order.filled_avg_price = filled_avg_price
    order.limit_price = limit_price
    return order


def _mock_alpaca_position(
    symbol="AAPL",
    qty="2.0",
    market_value="310.00",
    avg_entry_price="150.00",
    current_price="155.00",
    unrealized_pl="10.00",
):
    pos = MagicMock()
    pos.symbol = symbol
    pos.qty = qty
    pos.market_value = market_value
    pos.avg_entry_price = avg_entry_price
    pos.current_price = current_price
    pos.unrealized_pl = unrealized_pl
    return pos


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAlpacaBrokerInit:
    @patch.dict("os.environ", {
        "ALPACA_API_KEY": "test-key",
        "ALPACA_SECRET_KEY": "test-secret",
        "ALPACA_BASE_URL": "https://paper-api.alpaca.markets",
    })
    @patch("pipeline.execution.alpaca_broker.TradingClient", create=True)
    def test_from_env_paper_mode(self, mock_client_cls):
        """from_env() with paper URL creates a paper broker."""
        # Patch the import inside alpaca_broker
        with patch.dict("sys.modules", {
            "alpaca": MagicMock(),
            "alpaca.trading": MagicMock(),
            "alpaca.trading.client": MagicMock(TradingClient=mock_client_cls),
        }):
            from pipeline.execution.alpaca_broker import AlpacaBroker
            broker = AlpacaBroker(
                api_key="test-key",
                secret_key="test-secret",
                base_url="https://paper-api.alpaca.markets",
            )
            assert broker._is_paper is True

    def test_from_env_missing_keys_raises(self):
        """from_env() with no keys raises BrokerError."""
        with patch.dict("os.environ", {}, clear=True):
            # Remove any existing env vars
            import os
            os.environ.pop("ALPACA_API_KEY", None)
            os.environ.pop("ALPACA_SECRET_KEY", None)

            from pipeline.execution.alpaca_broker import AlpacaBroker
            with pytest.raises(BrokerError, match="ALPACA_API_KEY"):
                AlpacaBroker.from_env()


class TestAlpacaBrokerAccountSnapshot:
    def test_get_account_snapshot(self):
        """Account snapshot fetches live data and returns typed result."""
        from pipeline.execution.alpaca_broker import AlpacaBroker

        mock_client = MagicMock()
        mock_client.get_account.return_value = _mock_alpaca_account(
            equity="500.00", cash="300.00", buying_power="300.00",
            long_market_value="200.00", short_market_value="0.00",
            multiplier="1",
        )
        mock_client.get_all_positions.return_value = [
            _mock_alpaca_position(),
        ]

        # Create broker with mocked client
        with patch("pipeline.execution.alpaca_broker.AlpacaBroker.__init__", return_value=None):
            broker = AlpacaBroker.__new__(AlpacaBroker)
            broker._client = mock_client
            broker._is_paper = True

        snapshot = broker.get_account_snapshot()

        assert isinstance(snapshot, AccountSnapshot)
        assert snapshot.equity == 500.0
        assert snapshot.cash == 300.0
        assert snapshot.buying_power == 300.0
        assert snapshot.positions_market_value == 200.0
        assert snapshot.position_count == 1
        assert snapshot.is_margin_account is False

    def test_margin_account_detected(self):
        """Multiplier > 1 indicates margin account."""
        from pipeline.execution.alpaca_broker import AlpacaBroker

        mock_client = MagicMock()
        mock_client.get_account.return_value = _mock_alpaca_account(multiplier="2")
        mock_client.get_all_positions.return_value = []

        with patch("pipeline.execution.alpaca_broker.AlpacaBroker.__init__", return_value=None):
            broker = AlpacaBroker.__new__(AlpacaBroker)
            broker._client = mock_client
            broker._is_paper = True

        snapshot = broker.get_account_snapshot()
        assert snapshot.is_margin_account is True

    def test_account_fetch_failure_raises(self):
        """BrokerError raised if account fetch fails."""
        from pipeline.execution.alpaca_broker import AlpacaBroker

        mock_client = MagicMock()
        mock_client.get_account.side_effect = Exception("Connection refused")

        with patch("pipeline.execution.alpaca_broker.AlpacaBroker.__init__", return_value=None):
            broker = AlpacaBroker.__new__(AlpacaBroker)
            broker._client = mock_client
            broker._is_paper = True

        with pytest.raises(BrokerError, match="Failed to fetch"):
            broker.get_account_snapshot()


class TestAlpacaBrokerOrderSubmission:
    def _make_broker(self):
        from pipeline.execution.alpaca_broker import AlpacaBroker
        mock_client = MagicMock()
        with patch("pipeline.execution.alpaca_broker.AlpacaBroker.__init__", return_value=None):
            broker = AlpacaBroker.__new__(AlpacaBroker)
            broker._client = mock_client
            broker._is_paper = True
        return broker, mock_client

    def test_submit_market_order(self):
        broker, mock_client = self._make_broker()
        mock_client.submit_order.return_value = _mock_alpaca_order(
            status="filled", filled_qty="2.0", filled_avg_price="150.50",
        )

        order = Order(
            symbol="AAPL", side=OrderSide.BUY,
            order_type=OrderType.MARKET, qty=2.0,
        )
        result = broker.submit_order(order)

        assert result.order_id == "abc-123"
        assert result.status == OrderStatus.FILLED
        assert result.filled_qty == 2.0
        assert result.filled_avg_price == 150.50
        mock_client.submit_order.assert_called_once()

    def test_submit_limit_order(self):
        broker, mock_client = self._make_broker()
        mock_client.submit_order.return_value = _mock_alpaca_order(
            status="new", filled_qty=None, filled_avg_price=None,
        )

        order = Order(
            symbol="AAPL", side=OrderSide.BUY,
            order_type=OrderType.LIMIT, qty=2.0, limit_price=150.0,
        )
        result = broker.submit_order(order)

        assert result.status == OrderStatus.SUBMITTED
        assert result.order_id == "abc-123"

    def test_limit_order_missing_price_raises(self):
        broker, _ = self._make_broker()

        order = Order(
            symbol="AAPL", side=OrderSide.BUY,
            order_type=OrderType.LIMIT, qty=2.0, limit_price=None,
        )
        with pytest.raises(BrokerError, match="Limit price required"):
            broker.submit_order(order)

    def test_broker_api_failure_raises(self):
        broker, mock_client = self._make_broker()
        mock_client.submit_order.side_effect = Exception("Insufficient funds")

        order = Order(
            symbol="AAPL", side=OrderSide.BUY,
            order_type=OrderType.MARKET, qty=2.0,
        )
        with pytest.raises(BrokerError, match="Order submission failed"):
            broker.submit_order(order)
        assert order.status == OrderStatus.REJECTED


class TestAlpacaBrokerPositions:
    def _make_broker(self):
        from pipeline.execution.alpaca_broker import AlpacaBroker
        mock_client = MagicMock()
        with patch("pipeline.execution.alpaca_broker.AlpacaBroker.__init__", return_value=None):
            broker = AlpacaBroker.__new__(AlpacaBroker)
            broker._client = mock_client
            broker._is_paper = True
        return broker, mock_client

    def test_get_positions(self):
        broker, mock_client = self._make_broker()
        mock_client.get_all_positions.return_value = [
            _mock_alpaca_position("AAPL", "2.0", "310.00", "150.00", "155.00", "10.00"),
            _mock_alpaca_position("MSFT", "-1.0", "300.00", "305.00", "300.00", "-5.00"),
        ]

        positions = broker.get_positions()
        assert len(positions) == 2
        assert positions[0].symbol == "AAPL"
        assert positions[0].qty == 2.0
        assert positions[0].side == "long"
        assert positions[1].qty == -1.0
        assert positions[1].side == "short"

    def test_close_position(self):
        broker, mock_client = self._make_broker()
        mock_client.close_position.return_value = _mock_alpaca_order(
            order_id="close-1", status="new", symbol="AAPL", qty="2.0",
        )

        order = broker.close_position("AAPL")
        assert order.order_id == "close-1"
        assert order.symbol == "AAPL"

    def test_close_all_positions(self):
        broker, mock_client = self._make_broker()

        mock_body = MagicMock()
        mock_body.id = "close-all-1"
        mock_body.symbol = "AAPL"
        mock_body.qty = "2.0"
        mock_body.status = "new"

        mock_result = MagicMock()
        mock_result.body = mock_body

        mock_client.close_all_positions.return_value = [mock_result]

        orders = broker.close_all_positions()
        assert len(orders) == 1
        assert orders[0].order_id == "close-all-1"


class TestAlpacaBrokerOrderStatusMapping:
    """Verify all Alpaca status strings map correctly."""

    def test_status_map_coverage(self):
        from pipeline.execution.alpaca_broker import _STATUS_MAP

        # Key statuses from Alpaca docs
        assert _STATUS_MAP["new"] == OrderStatus.SUBMITTED
        assert _STATUS_MAP["filled"] == OrderStatus.FILLED
        assert _STATUS_MAP["canceled"] == OrderStatus.CANCELLED
        assert _STATUS_MAP["cancelled"] == OrderStatus.CANCELLED
        assert _STATUS_MAP["expired"] == OrderStatus.EXPIRED
        assert _STATUS_MAP["rejected"] == OrderStatus.REJECTED
        assert _STATUS_MAP["partially_filled"] == OrderStatus.PARTIAL
        assert _STATUS_MAP["pending_new"] == OrderStatus.PENDING

    def test_get_order_status(self):
        from pipeline.execution.alpaca_broker import AlpacaBroker
        mock_client = MagicMock()
        with patch("pipeline.execution.alpaca_broker.AlpacaBroker.__init__", return_value=None):
            broker = AlpacaBroker.__new__(AlpacaBroker)
            broker._client = mock_client
            broker._is_paper = True

        mock_client.get_order_by_id.return_value = _mock_alpaca_order(
            order_id="abc-123", status="filled",
            filled_qty="2.0", filled_avg_price="150.50",
        )

        order = broker.get_order_status("abc-123")
        assert order.status == OrderStatus.FILLED
        assert order.filled_qty == 2.0

    def test_cancel_order(self):
        from pipeline.execution.alpaca_broker import AlpacaBroker
        mock_client = MagicMock()
        with patch("pipeline.execution.alpaca_broker.AlpacaBroker.__init__", return_value=None):
            broker = AlpacaBroker.__new__(AlpacaBroker)
            broker._client = mock_client
            broker._is_paper = True

        assert broker.cancel_order("abc-123") is True
        mock_client.cancel_order_by_id.assert_called_once_with("abc-123")

    def test_cancel_order_failure(self):
        from pipeline.execution.alpaca_broker import AlpacaBroker
        mock_client = MagicMock()
        mock_client.cancel_order_by_id.side_effect = Exception("Not found")
        with patch("pipeline.execution.alpaca_broker.AlpacaBroker.__init__", return_value=None):
            broker = AlpacaBroker.__new__(AlpacaBroker)
            broker._client = mock_client
            broker._is_paper = True

        assert broker.cancel_order("abc-123") is False
