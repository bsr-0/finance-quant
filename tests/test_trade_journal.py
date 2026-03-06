"""Tests for the trade journal and fill polling."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import pytest

from pipeline.execution.broker import (
    BaseBroker,
    BrokerError,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)
from pipeline.execution.capital_guard import AccountSnapshot, CapitalGuardConfig
from pipeline.execution.signal_executor import SignalExecutor
from pipeline.execution.trade_journal import TradeJournal


# ---------------------------------------------------------------------------
# Trade journal tests
# ---------------------------------------------------------------------------

class TestTradeJournal:
    def test_record_order(self, tmp_path):
        journal = TradeJournal(journal_dir=tmp_path)
        order = Order(
            symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.LIMIT,
            qty=2.0, limit_price=150.0, order_id="ord-1",
            status=OrderStatus.SUBMITTED,
        )
        journal.record_order(order, signal_score=75, signal_regime="BULL")

        entries = journal.read_journal()
        assert len(entries) == 1
        assert entries[0]["event_type"] == "ORDER_SUBMITTED"
        assert entries[0]["symbol"] == "AAPL"
        assert entries[0]["signal_score"] == "75"
        assert entries[0]["signal_regime"] == "BULL"

    def test_record_fill(self, tmp_path):
        journal = TradeJournal(journal_dir=tmp_path)
        order = Order(
            symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.LIMIT,
            qty=2.0, limit_price=150.0, order_id="ord-1",
            status=OrderStatus.FILLED, filled_qty=2.0, filled_avg_price=149.95,
        )
        journal.record_fill(order)

        entries = journal.read_journal()
        assert len(entries) == 1
        assert entries[0]["event_type"] == "ORDER_FILLED"
        assert entries[0]["filled_avg_price"] == "149.95"

    def test_record_exit(self, tmp_path):
        journal = TradeJournal(journal_dir=tmp_path)
        journal.record_exit(
            symbol="AAPL", reason="stop_loss",
            exit_price=145.0, pnl=-10.0, shares=2, order_id="exit-1",
        )

        entries = journal.read_journal()
        assert len(entries) == 1
        assert entries[0]["event_type"] == "POSITION_EXIT"
        assert entries[0]["exit_reason"] == "stop_loss"
        assert entries[0]["pnl"] == "-10.0"

    def test_record_guard_rejection(self, tmp_path):
        journal = TradeJournal(journal_dir=tmp_path)
        journal.record_guard_rejection(
            symbol="TSLA", checks_failed=["max_capital_check", "buying_power_check"],
        )

        entries = journal.read_journal()
        assert len(entries) == 1
        assert entries[0]["event_type"] == "GUARD_REJECTED"
        assert "max_capital_check" in entries[0]["reject_reason"]

    def test_record_rejection(self, tmp_path):
        journal = TradeJournal(journal_dir=tmp_path)
        order = Order(
            symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET,
            qty=1.0, order_id="rej-1", status=OrderStatus.REJECTED,
            reject_reason="Insufficient funds",
        )
        journal.record_rejection(order)

        entries = journal.read_journal()
        assert len(entries) == 1
        assert entries[0]["event_type"] == "ORDER_REJECTED"
        assert entries[0]["reject_reason"] == "Insufficient funds"

    def test_multiple_entries_in_one_day(self, tmp_path):
        journal = TradeJournal(journal_dir=tmp_path)
        for i in range(5):
            order = Order(
                symbol=f"SYM{i}", side=OrderSide.BUY,
                order_type=OrderType.MARKET, qty=1.0,
                order_id=f"ord-{i}", status=OrderStatus.SUBMITTED,
            )
            journal.record_order(order)

        entries = journal.read_journal()
        assert len(entries) == 5

    def test_read_nonexistent_date(self, tmp_path):
        journal = TradeJournal(journal_dir=tmp_path)
        entries = journal.read_journal("20200101")
        assert entries == []

    def test_journal_dir_created(self, tmp_path):
        new_dir = tmp_path / "nested" / "journal"
        journal = TradeJournal(journal_dir=new_dir)
        assert new_dir.exists()


# ---------------------------------------------------------------------------
# Fill polling tests (uses MockBroker)
# ---------------------------------------------------------------------------

class MockBroker(BaseBroker):
    """Mock broker with configurable fill behavior."""

    def __init__(
        self,
        equity: float = 500.0,
        fill_on_poll: int = 0,
    ):
        self._equity = equity
        self._fill_on_poll = fill_on_poll
        self._poll_count = 0
        self.submitted_orders: list[Order] = []

    def get_account_snapshot(self) -> AccountSnapshot:
        return AccountSnapshot(
            equity=self._equity, cash=self._equity, buying_power=self._equity,
            positions_market_value=0, position_count=0, is_margin_account=False,
        )

    def submit_order(self, order: Order) -> Order:
        order.order_id = f"mock-{len(self.submitted_orders)}"
        order.status = OrderStatus.SUBMITTED
        order.submitted_at = datetime.now(timezone.utc)
        self.submitted_orders.append(order)
        return order

    def get_order_status(self, order_id: str) -> Order:
        self._poll_count += 1
        for o in self.submitted_orders:
            if o.order_id == order_id:
                # Fill after N polls
                if self._poll_count >= self._fill_on_poll:
                    o.status = OrderStatus.FILLED
                    o.filled_qty = o.qty
                    o.filled_avg_price = o.limit_price or 100.0
                return o
        raise BrokerError(f"Order {order_id} not found")

    def cancel_order(self, order_id: str) -> bool:
        for o in self.submitted_orders:
            if o.order_id == order_id:
                o.status = OrderStatus.CANCELLED
        return True

    def get_positions(self) -> list[Position]:
        return []

    def close_position(self, symbol): raise NotImplementedError
    def close_all_positions(self): return []


class TestFillPolling:
    def test_poll_until_filled(self):
        broker = MockBroker(fill_on_poll=2)
        config = CapitalGuardConfig(max_capital=400)
        executor = SignalExecutor(
            broker=broker, guard_config=config,
            fill_poll_interval=0,  # No sleep in tests
            fill_poll_timeout=10,
        )

        # Manually submit an order
        order = Order(
            symbol="AAPL", side=OrderSide.BUY,
            order_type=OrderType.LIMIT, qty=1.0, limit_price=150.0,
        )
        submitted = broker.submit_order(order)

        results = executor.poll_pending_orders([submitted.order_id])

        assert submitted.order_id in results
        assert results[submitted.order_id].status == OrderStatus.FILLED

    def test_poll_timeout_cancels(self):
        broker = MockBroker(fill_on_poll=999)  # Never fills
        config = CapitalGuardConfig(max_capital=400)
        executor = SignalExecutor(
            broker=broker, guard_config=config,
            fill_poll_interval=0,
            fill_poll_timeout=0,  # Immediate timeout
        )

        order = Order(
            symbol="AAPL", side=OrderSide.BUY,
            order_type=OrderType.LIMIT, qty=1.0, limit_price=150.0,
        )
        submitted = broker.submit_order(order)

        results = executor.poll_pending_orders([submitted.order_id])

        assert submitted.order_id in results
        # Should be cancelled after timeout
        assert results[submitted.order_id].status == OrderStatus.CANCELLED

    def test_poll_empty_list(self):
        broker = MockBroker()
        config = CapitalGuardConfig(max_capital=400)
        executor = SignalExecutor(broker=broker, guard_config=config)

        results = executor.poll_pending_orders([])
        assert results == {}


# ---------------------------------------------------------------------------
# ExecutionSettings tests
# ---------------------------------------------------------------------------

class TestExecutionSettings:
    def test_default_values(self):
        from pipeline.settings import ExecutionSettings
        settings = ExecutionSettings()
        assert settings.broker == "alpaca"
        assert settings.mode == "paper"
        assert settings.is_paper is True
        assert settings.max_capital == 300.0
        assert settings.max_positions == 2
        assert settings.fill_poll_interval_seconds == 5

    def test_live_mode(self):
        from pipeline.settings import ExecutionSettings
        settings = ExecutionSettings(mode="live", base_url="https://api.alpaca.markets")
        assert settings.is_paper is False

    def test_paper_detection_from_url(self):
        from pipeline.settings import ExecutionSettings
        settings = ExecutionSettings(mode="live", base_url="https://paper-api.alpaca.markets")
        assert settings.is_paper is True  # URL overrides mode

    def test_settings_in_pipeline_settings(self):
        from pipeline.settings import PipelineSettings
        settings = PipelineSettings()
        assert hasattr(settings, "execution")
        assert settings.execution.broker == "alpaca"
        assert settings.execution.max_capital == 300.0
