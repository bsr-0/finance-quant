"""Tests for position register persistence and multi-day position tracking."""

from datetime import UTC, datetime
from unittest.mock import MagicMock

from pipeline.execution.broker import (
    AccountSnapshot,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)
from pipeline.execution.capital_guard import CapitalGuardConfig
from pipeline.execution.position_monitor import (
    PositionMonitor,
    TrackedPosition,
)
from pipeline.execution.position_register import PositionRegister

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tracked(
    symbol: str = "AAPL",
    entry_price: float = 150.0,
    shares: float = 2.0,
    stop_price: float = 145.0,
    atr: float = 3.0,
    target_1: float = 160.0,
    target_2: float = 170.0,
    signal_score: int = 75,
    trailing_stop: float = 0.0,
    trailing_activated: bool = False,
    highest_price: float = 0.0,
) -> TrackedPosition:
    return TrackedPosition(
        symbol=symbol,
        entry_date=datetime(2025, 6, 15, 14, 30, 0, tzinfo=UTC),
        entry_price=entry_price,
        shares=shares,
        stop_price=stop_price,
        atr_at_entry=atr,
        trailing_stop=trailing_stop,
        trailing_activated=trailing_activated,
        highest_price=highest_price,
        target_1=target_1,
        target_2=target_2,
        signal_score=signal_score,
    )


def _make_mock_broker(
    equity: float = 10000.0,
    positions: list[Position] | None = None,
) -> MagicMock:
    broker = MagicMock()
    broker.get_account_snapshot.return_value = AccountSnapshot(
        equity=equity,
        cash=equity,
        buying_power=equity,
        positions_market_value=0.0,
        position_count=len(positions or []),
        is_margin_account=False,
    )
    broker.get_positions.return_value = positions or []
    broker.close_position.return_value = Order(
        symbol="AAPL",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        qty=2,
        order_id="exit-001",
        status=OrderStatus.FILLED,
    )
    return broker


# ---------------------------------------------------------------------------
# PositionRegister unit tests
# ---------------------------------------------------------------------------


class TestPositionRegister:
    def test_save_and_load_roundtrip(self, tmp_path):
        reg = PositionRegister(path=tmp_path / "positions.json")
        positions = {
            "AAPL": _make_tracked("AAPL", entry_price=150.0, atr=3.0),
            "MSFT": _make_tracked("MSFT", entry_price=350.0, atr=5.0, signal_score=80),
        }
        reg.save(positions)
        loaded = reg.load()

        assert set(loaded.keys()) == {"AAPL", "MSFT"}
        for sym in ("AAPL", "MSFT"):
            orig = positions[sym]
            reloaded = loaded[sym]
            assert reloaded.symbol == orig.symbol
            assert reloaded.entry_price == orig.entry_price
            assert reloaded.shares == orig.shares
            assert reloaded.stop_price == orig.stop_price
            assert reloaded.atr_at_entry == orig.atr_at_entry
            assert reloaded.trailing_stop == orig.trailing_stop
            assert reloaded.trailing_activated == orig.trailing_activated
            assert reloaded.highest_price == orig.highest_price
            assert reloaded.target_1 == orig.target_1
            assert reloaded.target_2 == orig.target_2
            assert reloaded.signal_score == orig.signal_score

    def test_load_empty_file(self, tmp_path):
        reg = PositionRegister(path=tmp_path / "missing.json")
        loaded = reg.load()
        assert loaded == {}

    def test_clear(self, tmp_path):
        reg = PositionRegister(path=tmp_path / "positions.json")
        reg.save({"AAPL": _make_tracked()})
        assert reg.path.exists()
        reg.clear()
        assert not reg.path.exists()
        assert reg.load() == {}

    def test_atomic_write(self, tmp_path):
        reg = PositionRegister(path=tmp_path / "positions.json")
        reg.save({"AAPL": _make_tracked()})
        assert reg.path.exists()
        content = reg.path.read_text()
        assert "AAPL" in content

    def test_datetime_serialization(self, tmp_path):
        reg = PositionRegister(path=tmp_path / "positions.json")
        original = _make_tracked()
        reg.save({"AAPL": original})
        loaded = reg.load()
        assert loaded["AAPL"].entry_date.tzinfo is not None
        assert loaded["AAPL"].entry_date.year == 2025
        assert loaded["AAPL"].entry_date.month == 6

    def test_trailing_stop_state_preserved(self, tmp_path):
        reg = PositionRegister(path=tmp_path / "positions.json")
        pos = _make_tracked(trailing_stop=148.0, trailing_activated=True, highest_price=155.0)
        reg.save({"AAPL": pos})
        loaded = reg.load()
        assert loaded["AAPL"].trailing_stop == 148.0
        assert loaded["AAPL"].trailing_activated is True
        assert loaded["AAPL"].highest_price == 155.0


# ---------------------------------------------------------------------------
# PositionMonitor persistence integration tests
# ---------------------------------------------------------------------------


class TestPositionMonitorPersistence:
    def test_register_persists(self, tmp_path):
        reg = PositionRegister(path=tmp_path / "positions.json")
        broker = _make_mock_broker()
        monitor = PositionMonitor(
            broker=broker,
            guard_config=CapitalGuardConfig(max_capital=10000),
            position_register=reg,
        )
        tracked = _make_tracked()
        monitor.register_position(tracked)

        # Verify persisted to disk
        loaded = reg.load()
        assert "AAPL" in loaded
        assert loaded["AAPL"].entry_price == 150.0

    def test_load_on_init(self, tmp_path):
        reg = PositionRegister(path=tmp_path / "positions.json")
        # Pre-populate the register
        reg.save({"AAPL": _make_tracked(), "MSFT": _make_tracked("MSFT")})

        broker = _make_mock_broker()
        monitor = PositionMonitor(
            broker=broker,
            guard_config=CapitalGuardConfig(max_capital=10000),
            position_register=reg,
        )

        assert "AAPL" in monitor.tracked_positions
        assert "MSFT" in monitor.tracked_positions

    def test_no_register_backward_compatible(self):
        broker = _make_mock_broker()
        monitor = PositionMonitor(
            broker=broker,
            guard_config=CapitalGuardConfig(max_capital=10000),
        )
        # No register — should work exactly as before
        tracked = _make_tracked()
        monitor.register_position(tracked)
        assert "AAPL" in monitor.tracked_positions

    def test_exit_removes_from_register(self, tmp_path):
        reg = PositionRegister(path=tmp_path / "positions.json")
        reg.save({"AAPL": _make_tracked()})

        broker = _make_mock_broker(
            positions=[Position("AAPL", 2, 300, 150, 140, -20, "long")],
        )
        monitor = PositionMonitor(
            broker=broker,
            guard_config=CapitalGuardConfig(max_capital=10000),
            position_register=reg,
        )

        # Position loaded from register
        assert "AAPL" in monitor.tracked_positions

        # Stop should trigger — current price 140 < stop 145
        monitor.check_and_exit()

        # Position should be removed from register after exit
        loaded = reg.load()
        assert "AAPL" not in loaded


# ---------------------------------------------------------------------------
# Multi-day flow test
# ---------------------------------------------------------------------------


class TestMultiDayFlow:
    def test_day1_open_day2_exit(self, tmp_path):
        """Simulate Day 1 open → Day 2 exit via stop-loss across runner restarts."""
        register_path = tmp_path / "positions.json"

        # --- Day 1: Open a position ---
        broker_day1 = _make_mock_broker(
            positions=[Position("AAPL", 2, 310, 155, 155, 0, "long")],
        )
        monitor_day1 = PositionMonitor(
            broker=broker_day1,
            guard_config=CapitalGuardConfig(max_capital=10000),
            position_register=PositionRegister(path=register_path),
        )
        monitor_day1.register_position(
            _make_tracked("AAPL", entry_price=155.0, stop_price=150.0, atr=3.0)
        )

        # Verify register has position
        day1_loaded = PositionRegister(path=register_path).load()
        assert "AAPL" in day1_loaded
        assert day1_loaded["AAPL"].stop_price == 150.0
        assert day1_loaded["AAPL"].atr_at_entry == 3.0

        # --- Day 2: New monitor (simulating restart) ---
        broker_day2 = _make_mock_broker(
            positions=[Position("AAPL", 2, 290, 155, 145, -20, "long")],
        )
        monitor_day2 = PositionMonitor(
            broker=broker_day2,
            guard_config=CapitalGuardConfig(max_capital=10000),
            position_register=PositionRegister(path=register_path),
        )

        # Position should be loaded from register
        assert "AAPL" in monitor_day2.tracked_positions
        assert monitor_day2.tracked_positions["AAPL"].stop_price == 150.0

        # Price is 145 < stop 150 → stop-loss should trigger
        result = monitor_day2.check_and_exit()
        assert result.exits_triggered >= 1

        # Position should be gone from register
        day2_loaded = PositionRegister(path=register_path).load()
        assert "AAPL" not in day2_loaded
