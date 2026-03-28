"""Tests for the position reconciler."""

from __future__ import annotations

from pipeline.execution.broker import BaseBroker, Position
from pipeline.execution.capital_guard import AccountSnapshot
from pipeline.execution.reconciler import (
    DiscrepancyType,
    PositionReconciler,
    Severity,
    SystemPosition,
)

# ---------------------------------------------------------------------------
# Mock broker
# ---------------------------------------------------------------------------

class MockBroker(BaseBroker):
    """Minimal broker mock for reconciler tests."""

    def __init__(self, positions: list[Position] | None = None):
        self._positions = positions or []

    def get_account_snapshot(self) -> AccountSnapshot:
        return AccountSnapshot(
            equity=1000, cash=500, buying_power=500,
            positions_market_value=500, position_count=len(self._positions),
            is_margin_account=False,
        )

    def get_positions(self) -> list[Position]:
        return self._positions

    def submit_order(self, order): raise NotImplementedError
    def get_order_status(self, order_id): raise NotImplementedError
    def cancel_order(self, order_id): raise NotImplementedError
    def close_position(self, symbol): raise NotImplementedError
    def close_all_positions(self): raise NotImplementedError


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCleanReconciliation:
    def test_both_empty(self):
        recon = PositionReconciler(broker=MockBroker())
        result = recon.reconcile({})
        assert result.is_clean
        assert len(result.discrepancies) == 0

    def test_matching_positions(self):
        broker = MockBroker(positions=[
            Position("AAPL", 2.0, 300.0, 150.0, 155.0, 10.0, "long"),
        ])
        recon = PositionReconciler(broker=broker)
        result = recon.reconcile({
            "AAPL": SystemPosition("AAPL", 2.0, 150.0, "long"),
        })
        assert result.is_clean


class TestMissingInSystem:
    def test_broker_has_unknown_position(self):
        broker = MockBroker(positions=[
            Position("AAPL", 5.0, 750.0, 150.0, 155.0, 25.0, "long"),
        ])
        recon = PositionReconciler(broker=broker)
        result = recon.reconcile({})  # system has nothing

        assert not result.is_clean
        assert len(result.discrepancies) == 1
        d = result.discrepancies[0]
        assert d.type == DiscrepancyType.MISSING_IN_SYSTEM
        assert d.severity == Severity.CRITICAL
        assert d.symbol == "AAPL"
        assert d.broker_qty == 5.0


class TestMissingInBroker:
    def test_system_has_phantom_position(self):
        broker = MockBroker(positions=[])  # broker has nothing
        recon = PositionReconciler(broker=broker)
        result = recon.reconcile({
            "MSFT": SystemPosition("MSFT", 3.0, 400.0, "long"),
        })

        assert not result.is_clean
        d = result.discrepancies[0]
        assert d.type == DiscrepancyType.MISSING_IN_BROKER
        assert d.severity == Severity.CRITICAL


class TestQtyMismatch:
    def test_different_quantities(self):
        broker = MockBroker(positions=[
            Position("AAPL", 5.0, 750.0, 150.0, 155.0, 25.0, "long"),
        ])
        recon = PositionReconciler(broker=broker)
        result = recon.reconcile({
            "AAPL": SystemPosition("AAPL", 3.0, 150.0, "long"),
        })

        assert not result.is_clean
        d = result.discrepancies[0]
        assert d.type == DiscrepancyType.QTY_MISMATCH
        assert d.system_qty == 3.0
        assert d.broker_qty == 5.0

    def test_tiny_rounding_diff_ignored(self):
        """Fractional share rounding within tolerance."""
        broker = MockBroker(positions=[
            Position("AAPL", 2.005, 300.0, 150.0, 155.0, 10.0, "long"),
        ])
        recon = PositionReconciler(broker=broker, qty_tolerance=0.01)
        result = recon.reconcile({
            "AAPL": SystemPosition("AAPL", 2.0, 150.0, "long"),
        })
        assert result.is_clean


class TestSideMismatch:
    def test_long_vs_short(self):
        broker = MockBroker(positions=[
            Position("SPY", 10.0, 4500.0, 450.0, 455.0, 50.0, "short"),
        ])
        recon = PositionReconciler(broker=broker)
        result = recon.reconcile({
            "SPY": SystemPosition("SPY", 10.0, 450.0, "long"),
        })

        assert not result.is_clean
        side_d = [d for d in result.discrepancies if d.type == DiscrepancyType.SIDE_MISMATCH]
        assert len(side_d) == 1


class TestConsecutiveClean:
    def test_tracks_consecutive_clean(self):
        broker = MockBroker()
        recon = PositionReconciler(broker=broker)

        recon.reconcile({})
        recon.reconcile({})
        recon.reconcile({})
        assert recon.consecutive_clean_count == 3

    def test_resets_on_dirty(self):
        recon = PositionReconciler(broker=MockBroker())
        recon.reconcile({})
        recon.reconcile({})

        # Now a dirty one
        dirty_broker = MockBroker(positions=[
            Position("AAPL", 1.0, 150.0, 150.0, 155.0, 5.0, "long"),
        ])
        recon.broker = dirty_broker
        recon.reconcile({})
        assert recon.consecutive_clean_count == 0
