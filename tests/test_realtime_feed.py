"""Tests for the real-time price feed module.

Covers PriceQuote properties, feed lifecycle, WebSocket message parsing,
polling backend, and PositionMonitor integration with realtime prices.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest

from pipeline.execution.realtime_feed import PriceQuote, RealtimePriceFeed


# ---------------------------------------------------------------------------
# PriceQuote unit tests
# ---------------------------------------------------------------------------

class TestPriceQuote:
    def test_mid_with_bid_ask(self):
        q = PriceQuote(symbol="AAPL", price=150.0, bid=149.90, ask=150.10)
        assert q.mid == pytest.approx(150.0)

    def test_mid_without_bid_ask(self):
        q = PriceQuote(symbol="AAPL", price=150.0)
        assert q.mid == 150.0

    def test_spread(self):
        q = PriceQuote(symbol="AAPL", price=150.0, bid=149.90, ask=150.10)
        assert q.spread == pytest.approx(0.20)

    def test_spread_no_bid_ask(self):
        q = PriceQuote(symbol="AAPL", price=150.0)
        assert q.spread == 0.0

    def test_age_seconds(self):
        old_ts = datetime.now(timezone.utc) - timedelta(seconds=30)
        q = PriceQuote(symbol="AAPL", price=150.0, timestamp=old_ts)
        assert q.age_seconds >= 29.0

    def test_fresh_quote_age(self):
        q = PriceQuote(symbol="AAPL", price=150.0)
        assert q.age_seconds < 2.0


# ---------------------------------------------------------------------------
# RealtimePriceFeed unit tests
# ---------------------------------------------------------------------------

class TestRealtimePriceFeed:
    def _make_feed(self, **kwargs):
        defaults = dict(
            symbols=["AAPL", "MSFT"],
            api_key="test-key",
            secret_key="test-secret",
            mode="polling",
            poll_interval=1,
            stale_threshold=120,
        )
        defaults.update(kwargs)
        return RealtimePriceFeed(**defaults)

    def test_initial_state(self):
        feed = self._make_feed()
        assert not feed.is_running
        assert not feed.is_connected
        assert feed.get_latest("AAPL") is None
        assert feed.get_price("AAPL") is None
        assert feed.is_stale("AAPL") is True

    def test_symbols_uppercased(self):
        feed = self._make_feed(symbols=["aapl", "msft"])
        assert feed.symbols == ["AAPL", "MSFT"]

    def test_add_symbols(self):
        feed = self._make_feed()
        feed.add_symbols(["GOOGL", "aapl"])  # AAPL already present
        assert "GOOGL" in feed.symbols
        assert feed.symbols.count("AAPL") == 1

    def test_update_price(self):
        feed = self._make_feed()
        quote = PriceQuote(symbol="AAPL", price=155.0, source="test")
        feed._update_price(quote)

        result = feed.get_latest("AAPL")
        assert result is not None
        assert result.price == 155.0
        assert result.source == "test"

    def test_get_price_shortcut(self):
        feed = self._make_feed()
        feed._update_price(PriceQuote(symbol="AAPL", price=155.0))
        assert feed.get_price("AAPL") == 155.0

    def test_get_all_latest(self):
        feed = self._make_feed()
        feed._update_price(PriceQuote(symbol="AAPL", price=155.0))
        feed._update_price(PriceQuote(symbol="MSFT", price=380.0))

        all_quotes = feed.get_all_latest()
        assert len(all_quotes) == 2
        assert all_quotes["AAPL"].price == 155.0
        assert all_quotes["MSFT"].price == 380.0

    def test_stale_detection(self):
        feed = self._make_feed(stale_threshold=5)
        old_ts = datetime.now(timezone.utc) - timedelta(seconds=10)
        feed._update_price(PriceQuote(symbol="AAPL", price=155.0, timestamp=old_ts))
        assert feed.is_stale("AAPL") is True

    def test_fresh_not_stale(self):
        feed = self._make_feed(stale_threshold=120)
        feed._update_price(PriceQuote(symbol="AAPL", price=155.0))
        assert feed.is_stale("AAPL") is False

    def test_callback_invoked(self):
        callback = MagicMock()
        feed = self._make_feed()
        feed._on_price = callback

        quote = PriceQuote(symbol="AAPL", price=155.0)
        feed._update_price(quote)

        callback.assert_called_once_with(quote)

    def test_callback_exception_doesnt_crash(self):
        def bad_callback(q):
            raise ValueError("boom")

        feed = self._make_feed()
        feed._on_price = bad_callback

        # Should not raise
        feed._update_price(PriceQuote(symbol="AAPL", price=155.0))
        assert feed.get_price("AAPL") == 155.0

    def test_thread_safety(self):
        """Concurrent reads and writes don't cause errors."""
        feed = self._make_feed()
        errors = []

        def writer():
            for i in range(100):
                feed._update_price(
                    PriceQuote(symbol="AAPL", price=150.0 + i * 0.01)
                )

        def reader():
            for _ in range(100):
                q = feed.get_latest("AAPL")
                if q and q.price < 100:
                    errors.append("bad price")

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(errors) == 0


# ---------------------------------------------------------------------------
# WebSocket message parsing
# ---------------------------------------------------------------------------

class TestWSMessageParsing:
    def _make_feed(self):
        return RealtimePriceFeed(
            symbols=["AAPL"],
            api_key="test",
            secret_key="test",
            mode="websocket",
        )

    def test_parse_trade_message(self):
        feed = self._make_feed()
        msg = {
            "T": "t",
            "S": "AAPL",
            "p": 155.25,
            "s": 100,
            "t": "2024-06-15T14:30:00.123456Z",
        }
        feed._handle_ws_message(msg)

        q = feed.get_latest("AAPL")
        assert q is not None
        assert q.price == 155.25
        assert q.volume == 100
        assert q.high == 155.25
        assert q.low == 155.25
        assert q.source == "alpaca_ws_trade"

    def test_trade_updates_high_low(self):
        feed = self._make_feed()

        # First trade
        feed._handle_ws_message({"T": "t", "S": "AAPL", "p": 150.0, "s": 50, "t": ""})
        # Higher trade
        feed._handle_ws_message({"T": "t", "S": "AAPL", "p": 155.0, "s": 50, "t": ""})
        # Lower trade
        feed._handle_ws_message({"T": "t", "S": "AAPL", "p": 148.0, "s": 50, "t": ""})

        q = feed.get_latest("AAPL")
        assert q.high == 155.0
        assert q.low == 148.0
        assert q.price == 148.0

    def test_parse_quote_message(self):
        feed = self._make_feed()
        msg = {
            "T": "q",
            "S": "AAPL",
            "bp": 155.00,
            "ap": 155.10,
            "t": "2024-06-15T14:30:00Z",
        }
        feed._handle_ws_message(msg)

        q = feed.get_latest("AAPL")
        assert q is not None
        assert q.bid == 155.00
        assert q.ask == 155.10
        assert q.price == pytest.approx(155.05)  # mid price
        assert q.source == "alpaca_ws_quote"

    def test_quote_preserves_trade_price(self):
        feed = self._make_feed()

        # Trade first
        feed._handle_ws_message({"T": "t", "S": "AAPL", "p": 155.25, "s": 100, "t": ""})
        # Then quote
        feed._handle_ws_message({"T": "q", "S": "AAPL", "bp": 155.20, "ap": 155.30, "t": ""})

        q = feed.get_latest("AAPL")
        assert q.price == 155.25  # keeps trade price
        assert q.bid == 155.20
        assert q.ask == 155.30

    def test_unknown_message_type_ignored(self):
        feed = self._make_feed()
        feed._handle_ws_message({"T": "x", "S": "AAPL"})
        assert feed.get_latest("AAPL") is None


# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------

class TestTimestampParsing:
    def test_rfc3339_with_nanos(self):
        ts = RealtimePriceFeed._parse_alpaca_timestamp("2024-06-15T14:30:00.123456789Z")
        assert ts.year == 2024
        assert ts.month == 6
        assert ts.hour == 14

    def test_rfc3339_without_frac(self):
        ts = RealtimePriceFeed._parse_alpaca_timestamp("2024-06-15T14:30:00Z")
        assert ts.year == 2024

    def test_empty_string_returns_now(self):
        ts = RealtimePriceFeed._parse_alpaca_timestamp("")
        assert ts.year >= 2024

    def test_invalid_string_returns_now(self):
        ts = RealtimePriceFeed._parse_alpaca_timestamp("not-a-timestamp")
        assert ts.year >= 2024


# ---------------------------------------------------------------------------
# PositionMonitor integration
# ---------------------------------------------------------------------------

class TestPositionMonitorRealtimeIntegration:
    """Test that PositionMonitor uses realtime prices when available."""

    def test_monitor_uses_realtime_price(self):
        from pipeline.execution.position_monitor import PositionMonitor, TrackedPosition
        from pipeline.execution.capital_guard import CapitalGuardConfig, AccountSnapshot

        # Mock broker
        broker = MagicMock()
        broker.get_account_snapshot.return_value = AccountSnapshot(
            equity=500.0, cash=300.0, buying_power=300.0,
            positions_market_value=200.0, position_count=1,
            is_margin_account=False,
        )

        from pipeline.execution.broker import Position
        broker.get_positions.return_value = [
            Position(
                symbol="AAPL", qty=1.0, market_value=150.0,
                avg_entry_price=145.0, current_price=150.0,
                unrealised_pnl=5.0, side="long",
            )
        ]

        # Mock realtime feed with a specific price
        rt_feed = MagicMock(spec=RealtimePriceFeed)
        rt_feed.is_running = True
        rt_feed.get_latest.return_value = PriceQuote(
            symbol="AAPL", price=140.0, high=152.0,  # Below stop
        )
        rt_feed.is_stale.return_value = False

        guard_config = CapitalGuardConfig(max_capital=500.0)
        monitor = PositionMonitor(
            broker=broker, guard_config=guard_config, realtime_feed=rt_feed,
        )
        monitor.initialize()

        # Register a position with stop at $143
        tracked = TrackedPosition(
            symbol="AAPL",
            entry_date=datetime.now(timezone.utc) - timedelta(days=1),
            entry_price=145.0,
            shares=1.0,
            stop_price=143.0,
            atr_at_entry=2.0,
        )
        monitor.register_position(tracked)

        result = monitor.check_and_exit()

        # The realtime price $140 is below stop $143, so exit should trigger
        assert result.exits_triggered == 1
        assert result.actions[0].reason.value == "stop_loss"

    def test_monitor_falls_back_to_broker_when_stale(self):
        from pipeline.execution.position_monitor import PositionMonitor, TrackedPosition
        from pipeline.execution.capital_guard import CapitalGuardConfig, AccountSnapshot
        from pipeline.execution.broker import Position

        broker = MagicMock()
        broker.get_account_snapshot.return_value = AccountSnapshot(
            equity=500.0, cash=300.0, buying_power=300.0,
            positions_market_value=200.0, position_count=1,
            is_margin_account=False,
        )
        # Broker price $147: above stop ($143), below profit target ($149)
        broker.get_positions.return_value = [
            Position(
                symbol="AAPL", qty=1.0, market_value=147.0,
                avg_entry_price=145.0, current_price=147.0,
                unrealised_pnl=2.0, side="long",
            )
        ]

        rt_feed = MagicMock(spec=RealtimePriceFeed)
        rt_feed.is_running = True
        old_quote = PriceQuote(
            symbol="AAPL", price=140.0,
            timestamp=datetime.now(timezone.utc) - timedelta(seconds=300),
        )
        rt_feed.get_latest.return_value = old_quote
        rt_feed.is_stale.return_value = True  # Stale!

        guard_config = CapitalGuardConfig(max_capital=500.0)
        monitor = PositionMonitor(
            broker=broker, guard_config=guard_config, realtime_feed=rt_feed,
        )
        monitor.initialize()

        tracked = TrackedPosition(
            symbol="AAPL",
            entry_date=datetime.now(timezone.utc) - timedelta(days=1),
            entry_price=145.0,
            shares=1.0,
            stop_price=143.0,
            atr_at_entry=2.0,
        )
        monitor.register_position(tracked)

        result = monitor.check_and_exit()

        # Broker price $147: above stop, below target — no exit
        assert result.exits_triggered == 0

    def test_monitor_works_without_feed(self):
        from pipeline.execution.position_monitor import PositionMonitor, TrackedPosition
        from pipeline.execution.capital_guard import CapitalGuardConfig, AccountSnapshot
        from pipeline.execution.broker import Position

        broker = MagicMock()
        broker.get_account_snapshot.return_value = AccountSnapshot(
            equity=500.0, cash=300.0, buying_power=300.0,
            positions_market_value=200.0, position_count=1,
            is_margin_account=False,
        )
        broker.get_positions.return_value = [
            Position(
                symbol="AAPL", qty=1.0, market_value=150.0,
                avg_entry_price=145.0, current_price=150.0,
                unrealised_pnl=5.0, side="long",
            )
        ]

        guard_config = CapitalGuardConfig(max_capital=500.0)
        # No realtime feed
        monitor = PositionMonitor(broker=broker, guard_config=guard_config)
        monitor.initialize()

        tracked = TrackedPosition(
            symbol="AAPL",
            entry_date=datetime.now(timezone.utc) - timedelta(days=1),
            entry_price=145.0,
            shares=1.0,
            stop_price=143.0,
            atr_at_entry=2.0,
        )
        monitor.register_position(tracked)

        # Should work fine using broker price
        result = monitor.check_and_exit()
        assert result.positions_checked == 1
