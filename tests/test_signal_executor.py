"""Tests for the signal executor and position monitor."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

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
from pipeline.execution.position_monitor import (
    PositionMonitor,
    TrackedPosition,
)


# ---------------------------------------------------------------------------
# Mock broker
# ---------------------------------------------------------------------------

class MockBroker(BaseBroker):
    """Broker mock that tracks submitted orders."""

    def __init__(
        self,
        equity: float = 500.0,
        cash: float = 500.0,
        positions: list[Position] | None = None,
        reject_orders: bool = False,
        is_margin: bool = False,
    ):
        self._equity = equity
        self._cash = cash
        self._positions = positions or []
        self._reject_orders = reject_orders
        self._is_margin = is_margin
        self.submitted_orders: list[Order] = []
        self._closed_symbols: list[str] = []

    def get_account_snapshot(self) -> AccountSnapshot:
        pos_value = sum(p.market_value for p in self._positions)
        return AccountSnapshot(
            equity=self._equity,
            cash=self._cash,
            buying_power=self._cash,
            positions_market_value=pos_value,
            position_count=len(self._positions),
            is_margin_account=self._is_margin,
        )

    def submit_order(self, order: Order) -> Order:
        if self._reject_orders:
            order.status = OrderStatus.REJECTED
            order.reject_reason = "Mock rejection"
            raise BrokerError("Mock broker rejection")
        order.order_id = f"mock-{len(self.submitted_orders)}"
        order.status = OrderStatus.FILLED
        order.filled_qty = order.qty
        order.filled_avg_price = order.limit_price or 100.0
        order.submitted_at = datetime.now(timezone.utc)
        self.submitted_orders.append(order)
        return order

    def get_order_status(self, order_id: str) -> Order:
        for o in self.submitted_orders:
            if o.order_id == order_id:
                return o
        raise BrokerError(f"Order {order_id} not found")

    def cancel_order(self, order_id: str) -> bool:
        return True

    def get_positions(self) -> list[Position]:
        return self._positions

    def close_position(self, symbol: str) -> Order:
        self._closed_symbols.append(symbol)
        return Order(
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            qty=1.0,
            order_id=f"close-{symbol}",
            status=OrderStatus.FILLED,
        )

    def close_all_positions(self) -> list[Order]:
        orders = []
        for p in self._positions:
            orders.append(self.close_position(p.symbol))
        self._positions.clear()
        return orders


# ---------------------------------------------------------------------------
# Signal DataFrame helper
# ---------------------------------------------------------------------------

def make_signal_df(
    tickers: list[str] | None = None,
    scores: list[int] | None = None,
    prices: list[float] | None = None,
) -> pd.DataFrame:
    """Create a minimal signal DataFrame."""
    tickers = tickers or ["AAPL"]
    scores = scores or [75]
    prices = prices or [150.0]

    rows = []
    for ticker, score, price in zip(tickers, scores, prices):
        atr = price * 0.02  # 2% ATR
        rows.append({
            "date": "2025-03-06",
            "ticker": ticker,
            "direction": "LONG",
            "score": score,
            "trend_pts": 30,
            "pullback_pts": 20,
            "volume_pts": 10,
            "volatility_pts": 10,
            "entry_price": price,
            "stop_price": price - atr * 1.5,
            "target_1": price + atr * 2.0,
            "target_2": price + atr * 3.0,
            "atr": atr,
            "atr_pct": 2.0,
            "regime": "BULL",
            "confidence": "HIGH" if score >= 80 else "MEDIUM",
            "strategy_id": "QSG-MICRO-SWING-001",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests: Signal Executor
# ---------------------------------------------------------------------------

class TestSignalExecutorBasic:
    def test_execute_single_signal(self):
        broker = MockBroker(equity=500, cash=500)
        config = CapitalGuardConfig(max_capital=400)
        executor = SignalExecutor(broker=broker, guard_config=config)

        df = make_signal_df(["AAPL"], [75], [150.0])
        result = executor.execute_signal_df(df)

        assert result.signals_parsed == 1
        assert result.orders_submitted == 1
        assert result.orders_filled == 1
        assert len(broker.submitted_orders) == 1
        assert broker.submitted_orders[0].symbol == "AAPL"

    def test_empty_signals(self):
        broker = MockBroker()
        config = CapitalGuardConfig(max_capital=400)
        executor = SignalExecutor(broker=broker, guard_config=config)

        df = pd.DataFrame()
        result = executor.execute_signal_df(df)

        assert result.signals_parsed == 0
        assert result.orders_submitted == 0

    def test_bear_regime_skipped(self):
        broker = MockBroker()
        config = CapitalGuardConfig(max_capital=400)
        executor = SignalExecutor(broker=broker, guard_config=config)

        df = make_signal_df(["AAPL"], [80], [150.0])
        df["regime"] = "BEAR"
        result = executor.execute_signal_df(df)

        assert result.orders_submitted == 0
        assert any(d.get("action") == "SKIP_BEAR_REGIME" for d in result.details)

    def test_already_held_skipped(self):
        broker = MockBroker(
            equity=500, cash=300,
            positions=[Position("AAPL", 2, 300, 150, 155, 10, "long")],
        )
        config = CapitalGuardConfig(max_capital=400)
        executor = SignalExecutor(broker=broker, guard_config=config)

        df = make_signal_df(["AAPL"], [80], [150.0])
        result = executor.execute_signal_df(df)

        assert result.orders_submitted == 0
        assert any(d.get("action") == "SKIP_ALREADY_HELD" for d in result.details)


class TestSignalExecutorGuard:
    def test_guard_rejects_exceeding_max_capital(self):
        broker = MockBroker(
            equity=500, cash=100,
            positions=[Position("SPY", 5, 400, 80, 82, 10, "long")],
        )
        config = CapitalGuardConfig(max_capital=300)
        executor = SignalExecutor(broker=broker, guard_config=config)

        # $150 order + $400 existing = $550 > $300 max
        df = make_signal_df(["AAPL"], [80], [150.0])
        result = executor.execute_signal_df(df)

        assert result.guard_rejections >= 1
        assert result.orders_submitted == 0

    def test_margin_account_rejected(self):
        broker = MockBroker(equity=500, cash=500, is_margin=True)
        config = CapitalGuardConfig(max_capital=400, require_cash_account=True)
        executor = SignalExecutor(broker=broker, guard_config=config)

        df = make_signal_df(["AAPL"], [80], [50.0])
        result = executor.execute_signal_df(df)

        assert result.guard_rejections >= 1


class TestSignalExecutorDryRun:
    def test_dry_run_no_orders_submitted(self):
        broker = MockBroker(equity=500, cash=500)
        config = CapitalGuardConfig(max_capital=400)
        executor = SignalExecutor(
            broker=broker, guard_config=config, dry_run=True,
        )

        df = make_signal_df(["AAPL"], [80], [150.0])
        result = executor.execute_signal_df(df)

        assert result.orders_submitted == 1  # counted as "would submit"
        assert len(broker.submitted_orders) == 0  # but nothing actually sent
        assert any(d.get("action") == "DRY_RUN_APPROVED" for d in result.details)


class TestSignalExecutorBrokerFailure:
    def test_broker_rejection_handled(self):
        broker = MockBroker(equity=500, cash=500, reject_orders=True)
        config = CapitalGuardConfig(max_capital=400)
        executor = SignalExecutor(broker=broker, guard_config=config)

        df = make_signal_df(["AAPL"], [80], [150.0])
        result = executor.execute_signal_df(df)

        assert result.orders_rejected >= 1
        assert any(d.get("action") == "BROKER_REJECTED" for d in result.details)


class TestSignalExecutorMultiple:
    def test_multiple_signals_ordered_by_score(self):
        broker = MockBroker(equity=1000, cash=1000)
        config = CapitalGuardConfig(max_capital=800, max_positions=4)
        executor = SignalExecutor(broker=broker, guard_config=config)

        df = make_signal_df(
            ["AAPL", "MSFT", "GOOGL"],
            [65, 85, 75],
            [50.0, 50.0, 50.0],
        )
        result = executor.execute_signal_df(df)

        # All 3 should pass (small orders, large account)
        assert result.signals_parsed == 3
        # First submitted order should be highest score (MSFT=85)
        if broker.submitted_orders:
            assert broker.submitted_orders[0].symbol == "MSFT"


# ---------------------------------------------------------------------------
# Tests: Position Monitor
# ---------------------------------------------------------------------------

class TestPositionMonitor:
    def test_no_exits_when_price_above_stop(self):
        broker = MockBroker(
            equity=500, cash=300,
            positions=[Position("AAPL", 2, 310, 150, 155, 10, "long")],
        )
        config = CapitalGuardConfig(max_capital=400)
        monitor = PositionMonitor(broker=broker, guard_config=config)
        # Use a recent entry date to avoid time-exit (max 15 days)
        recent_entry = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        monitor.register_position(TrackedPosition(
            symbol="AAPL",
            entry_date=recent_entry,
            entry_price=150.0,
            shares=2,
            stop_price=145.5,
            atr_at_entry=3.0,
        ))

        result = monitor.check_and_exit()
        assert result.exits_triggered == 0
        assert result.positions_checked == 1

    def test_stop_loss_triggers_exit(self):
        # Current price ($140) below stop ($145.5)
        broker = MockBroker(
            equity=500, cash=220,
            positions=[Position("AAPL", 2, 280, 150, 140, -20, "long")],
        )
        config = CapitalGuardConfig(max_capital=400)
        monitor = PositionMonitor(broker=broker, guard_config=config)
        monitor.register_position(TrackedPosition(
            symbol="AAPL",
            entry_date=datetime(2025, 3, 1, tzinfo=timezone.utc),
            entry_price=150.0,
            shares=2,
            stop_price=145.5,
            atr_at_entry=3.0,
        ))

        result = monitor.check_and_exit()
        assert result.exits_triggered == 1
        assert result.exits_executed == 1
        assert "AAPL" in broker._closed_symbols

    def test_red_circuit_breaker_closes_all(self):
        # Equity $400 with peak $500 = 20% drawdown > 15% RED threshold
        broker = MockBroker(
            equity=400, cash=100,
            positions=[
                Position("AAPL", 2, 150, 150, 75, -150, "long"),
                Position("MSFT", 1, 150, 150, 75, -75, "long"),
            ],
        )
        config = CapitalGuardConfig(max_capital=500)
        risk_mgr = SwingRiskManager()
        risk_mgr.initialize(500)  # Peak = $500

        monitor = PositionMonitor(
            broker=broker, guard_config=config, risk_manager=risk_mgr,
        )
        monitor.register_position(TrackedPosition(
            symbol="AAPL", entry_date=datetime(2025, 3, 1, tzinfo=timezone.utc),
            entry_price=150, shares=2, stop_price=140, atr_at_entry=3,
        ))
        monitor.register_position(TrackedPosition(
            symbol="MSFT", entry_date=datetime(2025, 3, 1, tzinfo=timezone.utc),
            entry_price=150, shares=1, stop_price=140, atr_at_entry=3,
        ))

        result = monitor.check_and_exit()
        assert result.circuit_breaker_level == "RED"
        assert result.exits_executed == 2

    def test_regime_change_triggers_exit(self):
        broker = MockBroker(
            equity=500, cash=300,
            positions=[Position("AAPL", 2, 310, 150, 155, 10, "long")],
        )
        config = CapitalGuardConfig(max_capital=400)
        monitor = PositionMonitor(
            broker=broker, guard_config=config, regime="BEAR",
        )
        monitor.register_position(TrackedPosition(
            symbol="AAPL",
            entry_date=datetime(2025, 3, 1, tzinfo=timezone.utc),
            entry_price=150.0,
            shares=2,
            stop_price=145.5,
            atr_at_entry=3.0,
        ))

        result = monitor.check_and_exit()
        assert result.exits_triggered == 1

    def test_untracked_position_removed(self):
        """Positions in broker but not tracked are ignored (not closed)."""
        broker = MockBroker(
            equity=500, cash=300,
            positions=[Position("AAPL", 2, 310, 150, 155, 10, "long")],
        )
        config = CapitalGuardConfig(max_capital=400)
        monitor = PositionMonitor(broker=broker, guard_config=config)
        # Don't register the position

        result = monitor.check_and_exit()
        assert result.exits_triggered == 0
        assert result.positions_checked == 1


# Need this import for the RED circuit breaker test
from pipeline.strategy.risk import SwingRiskManager
