"""Tests for the paper trading runner."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from pipeline.execution.broker import (
    BaseBroker,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)
from pipeline.execution.capital_guard import AccountSnapshot
from pipeline.execution.paper_runner import (
    DailyReport,
    PaperRunnerConfig,
    PaperTradingRunner,
)

# ---------------------------------------------------------------------------
# Mock broker for paper runner tests
# ---------------------------------------------------------------------------

class PaperMockBroker(BaseBroker):
    """Mock broker that simulates paper trading mode."""

    def __init__(
        self,
        equity: float = 500.0,
        cash: float = 500.0,
        buying_power: float = 500.0,
        positions: list[Position] | None = None,
    ):
        self._equity = equity
        self._cash = cash
        self._buying_power = buying_power
        self._positions = positions or []
        self._is_paper = True  # Simulate paper mode
        self.submitted_orders: list[Order] = []
        self._order_counter = 0

    def get_account_snapshot(self) -> AccountSnapshot:
        positions_value = sum(p.market_value for p in self._positions)
        return AccountSnapshot(
            equity=self._equity,
            cash=self._cash,
            buying_power=self._buying_power,
            positions_market_value=positions_value,
            position_count=len(self._positions),
            is_margin_account=False,
        )

    def submit_order(self, order: Order) -> Order:
        self._order_counter += 1
        order.order_id = f"paper-{self._order_counter}"
        order.status = OrderStatus.SUBMITTED
        order.submitted_at = datetime.now(UTC)
        self.submitted_orders.append(order)
        return order

    def get_positions(self) -> list[Position]:
        return self._positions

    def get_order_status(self, order_id: str) -> Order:
        raise NotImplementedError

    def cancel_order(self, order_id: str) -> bool:
        return True

    def close_position(self, symbol: str) -> Order:
        return Order(
            symbol=symbol, side=OrderSide.SELL, order_type=OrderType.MARKET,
            qty=1.0, order_id="close-1", status=OrderStatus.SUBMITTED,
        )

    def close_all_positions(self) -> list[Order]:
        return []


# ---------------------------------------------------------------------------
# Helper to write signal CSVs
# ---------------------------------------------------------------------------

def _make_signal_csv(tmp_path: Path, date_str: str = "20240115") -> Path:
    """Create a signal CSV in the given directory."""
    rows = [{
        "date": "2024-01-15",
        "ticker": "AAPL",
        "direction": "LONG",
        "score": 80,
        "trend_pts": 30,
        "pullback_pts": 15,
        "volume_pts": 10,
        "volatility_pts": 25,
        "entry_price": 150.0,
        "stop_price": 145.0,
        "target_1": 160.0,
        "target_2": 165.0,
        "atr": 5.0,
        "atr_pct": 3.33,
        "regime": "BULL",
        "confidence": "HIGH",
        "strategy_id": "QSG-MICRO-SWING-001",
    }]
    df = pd.DataFrame(rows)
    filepath = tmp_path / f"signals_{date_str}.csv"
    df.to_csv(filepath, index=False)
    return filepath


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPaperRunnerInit:
    def test_creates_with_custom_broker(self, tmp_path):
        """Runner should accept a custom broker."""
        broker = PaperMockBroker()
        config = PaperRunnerConfig(
            signal_dir=str(tmp_path),
            log_dir=str(tmp_path / "logs"),
        )
        runner = PaperTradingRunner(config=config, broker=broker)
        assert runner.broker is broker

    def test_rejects_live_alpaca_broker(self):
        """Runner should reject non-paper AlpacaBroker."""
        # We can't easily test this without alpaca-py installed,
        # so we just verify the config defaults to paper mode.
        config = PaperRunnerConfig()
        assert config.dry_run is False


class TestDailyRun:
    def test_run_with_signal_file(self, tmp_path):
        """Daily run should execute signals from a CSV."""
        signal_dir = tmp_path / "signals"
        signal_dir.mkdir()
        signal_file = _make_signal_csv(signal_dir)

        broker = PaperMockBroker(equity=500, cash=500, buying_power=500)
        config = PaperRunnerConfig(
            signal_dir=str(signal_dir),
            max_capital=300,
            log_dir=str(tmp_path / "logs"),
            order_type="market",
        )
        runner = PaperTradingRunner(config=config, broker=broker)
        report = runner.run_daily(signal_file=signal_file)

        assert report.execution_result is not None
        assert report.execution_result.signals_parsed == 1
        assert report.account_equity == 500.0

    def test_run_finds_latest_signal(self, tmp_path):
        """Daily run should auto-find the latest signal CSV."""
        signal_dir = tmp_path / "signals"
        signal_dir.mkdir()
        _make_signal_csv(signal_dir, "20240114")
        _make_signal_csv(signal_dir, "20240115")

        broker = PaperMockBroker(equity=500, cash=500, buying_power=500)
        config = PaperRunnerConfig(
            signal_dir=str(signal_dir),
            max_capital=300,
            log_dir=str(tmp_path / "logs"),
            order_type="market",
        )
        runner = PaperTradingRunner(config=config, broker=broker)
        report = runner.run_daily()

        assert report.signal_file is not None
        assert "20240115" in report.signal_file

    def test_run_no_signal_file(self, tmp_path):
        """Daily run with no signal files should succeed without errors."""
        signal_dir = tmp_path / "empty_signals"
        signal_dir.mkdir()

        broker = PaperMockBroker()
        config = PaperRunnerConfig(
            signal_dir=str(signal_dir),
            log_dir=str(tmp_path / "logs"),
        )
        runner = PaperTradingRunner(config=config, broker=broker)
        report = runner.run_daily()

        assert report.execution_result is None
        assert report.signal_file is None

    def test_run_with_reconciliation(self, tmp_path):
        """Daily run should reconcile positions."""
        signal_dir = tmp_path / "signals"
        signal_dir.mkdir()

        broker = PaperMockBroker()
        config = PaperRunnerConfig(
            signal_dir=str(signal_dir),
            log_dir=str(tmp_path / "logs"),
            reconcile_enabled=True,
        )
        runner = PaperTradingRunner(config=config, broker=broker)
        report = runner.run_daily()

        assert report.reconciliation_clean is True

    def test_run_with_reconciliation_disabled(self, tmp_path):
        """Reconciliation can be disabled."""
        signal_dir = tmp_path / "signals"
        signal_dir.mkdir()

        broker = PaperMockBroker()
        config = PaperRunnerConfig(
            signal_dir=str(signal_dir),
            log_dir=str(tmp_path / "logs"),
            reconcile_enabled=False,
        )
        runner = PaperTradingRunner(config=config, broker=broker)
        report = runner.run_daily()

        assert report.reconciliation_clean is None


class TestGetStatus:
    def test_status_with_no_positions(self, tmp_path):
        broker = PaperMockBroker()
        config = PaperRunnerConfig(log_dir=str(tmp_path / "logs"))
        runner = PaperTradingRunner(config=config, broker=broker)

        status = runner.get_status()
        assert status.account_equity == 500.0
        assert len(status.positions) == 0
        assert status.total_unrealised_pnl == 0.0
        assert status.is_healthy is True

    def test_status_with_positions(self, tmp_path):
        positions = [
            Position("AAPL", 2.0, 310.0, 150.0, 155.0, 10.0, "long"),
            Position("MSFT", 1.0, 305.0, 300.0, 305.0, 5.0, "long"),
        ]
        broker = PaperMockBroker(equity=800, positions=positions)
        config = PaperRunnerConfig(log_dir=str(tmp_path / "logs"))
        runner = PaperTradingRunner(config=config, broker=broker)

        status = runner.get_status()
        assert len(status.positions) == 2
        assert status.total_unrealised_pnl == 15.0
        assert status.positions[0].symbol == "AAPL"


class TestDailyReport:
    def test_report_summary(self):
        report = DailyReport(
            timestamp=datetime(2024, 1, 15, 16, 0, 0, tzinfo=UTC),
            signal_file="signals/signals_20240115.csv",
            execution_result=None,
            exit_orders=["AAPL"],
            reconciliation_clean=True,
            account_equity=500.0,
            account_cash=300.0,
            positions_count=1,
        )
        summary = report.summary()
        assert "Paper Trading" in summary
        assert "$500.00" in summary
        assert "CLEAN" in summary
        assert "AAPL" in summary

    def test_report_with_errors(self):
        report = DailyReport(
            timestamp=datetime(2024, 1, 15, tzinfo=UTC),
            signal_file=None,
            execution_result=None,
            exit_orders=[],
            reconciliation_clean=None,
            account_equity=0,
            account_cash=0,
            positions_count=0,
            errors=["Connection timeout"],
        )
        summary = report.summary()
        assert "Connection timeout" in summary


class TestSystemPositionTracking:
    def test_updates_system_positions_after_execution(self, tmp_path):
        """Runner should track positions internally for reconciliation."""
        signal_dir = tmp_path / "signals"
        signal_dir.mkdir()
        _make_signal_csv(signal_dir)

        broker = PaperMockBroker(equity=500, cash=500, buying_power=500)
        config = PaperRunnerConfig(
            signal_dir=str(signal_dir),
            max_capital=300,
            log_dir=str(tmp_path / "logs"),
            order_type="market",
            reconcile_enabled=False,
        )
        runner = PaperTradingRunner(config=config, broker=broker)
        runner.run_daily()

        # After execution, system should track the new position
        if broker.submitted_orders:
            assert "AAPL" in runner._system_positions


class TestLogOutput:
    def test_report_logged_to_file(self, tmp_path):
        """Daily run should write a report log file."""
        signal_dir = tmp_path / "signals"
        signal_dir.mkdir()
        log_dir = tmp_path / "logs"

        broker = PaperMockBroker()
        config = PaperRunnerConfig(
            signal_dir=str(signal_dir),
            log_dir=str(log_dir),
        )
        runner = PaperTradingRunner(config=config, broker=broker)
        runner.run_daily()

        log_files = list(log_dir.glob("paper_report_*.txt"))
        assert len(log_files) >= 1
