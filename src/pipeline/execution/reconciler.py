"""Position reconciler: post-trade QAQC.

Compares the system's internal view of positions against what the broker
actually reports.  Any discrepancy triggers alerts and can optionally
halt trading until resolved.

This is the third QAQC layer (after capital guard and broker-side
enforcement) and catches bugs, race conditions, or partial fills that
could cause the system to drift from reality.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum

from pipeline.execution.broker import BaseBroker, Position
from pipeline.infrastructure.notifier import AlertSeverity, notify

logger = logging.getLogger(__name__)


class DiscrepancyType(Enum):
    """Types of position discrepancy."""

    MISSING_IN_SYSTEM = "missing_in_system"
    """Broker has a position the system doesn't know about."""

    MISSING_IN_BROKER = "missing_in_broker"
    """System thinks it has a position the broker doesn't have."""

    QTY_MISMATCH = "qty_mismatch"
    """Both have the position but quantities differ."""

    SIDE_MISMATCH = "side_mismatch"
    """System and broker disagree on long/short."""


class Severity(Enum):
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Discrepancy:
    """A single discrepancy between system and broker state."""

    symbol: str
    type: DiscrepancyType
    severity: Severity
    system_qty: float
    broker_qty: float
    system_value: float
    broker_value: float
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __str__(self) -> str:
        return (
            f"[{self.severity.value.upper()}] {self.symbol}: {self.type.value} — "
            f"{self.message} (system={self.system_qty}, broker={self.broker_qty})"
        )


@dataclass
class SystemPosition:
    """What the system thinks it holds (from internal tracking)."""

    symbol: str
    qty: float
    avg_entry_price: float
    side: str  # "long" or "short"


@dataclass
class ReconciliationResult:
    """Result of a reconciliation run."""

    timestamp: datetime
    discrepancies: list[Discrepancy]
    system_positions: dict[str, SystemPosition]
    broker_positions: dict[str, Position]
    system_total_value: float
    broker_total_value: float
    is_clean: bool

    @property
    def value_discrepancy(self) -> float:
        return abs(self.system_total_value - self.broker_total_value)

    def summary(self) -> str:
        if self.is_clean:
            return (
                f"CLEAN: {len(self.broker_positions)} positions reconciled, "
                f"broker value=${self.broker_total_value:.2f}"
            )
        msgs = [str(d) for d in self.discrepancies]
        return (
            f"DISCREPANCIES FOUND ({len(self.discrepancies)}):\n"
            + "\n".join(f"  {m}" for m in msgs)
            + f"\n  Value diff: ${self.value_discrepancy:.2f}"
        )


class PositionReconciler:
    """Compare system state vs broker state and report discrepancies.

    Usage::

        reconciler = PositionReconciler(broker=alpaca_broker)

        # After each fill or at end of day:
        result = reconciler.reconcile(system_positions={
            "AAPL": SystemPosition("AAPL", 2, 150.0, "long"),
        })

        if not result.is_clean:
            for d in result.discrepancies:
                send_alert(str(d))
    """

    def __init__(
        self,
        broker: BaseBroker,
        qty_tolerance: float = 0.01,
        halt_on_critical: bool = True,
    ) -> None:
        """
        Args:
            broker: Broker to query for real positions.
            qty_tolerance: Fractional tolerance for qty comparisons
                (handles fractional share rounding).
            halt_on_critical: If True, recommend trading halt on critical
                discrepancies.
        """
        self.broker = broker
        self.qty_tolerance = qty_tolerance
        self.halt_on_critical = halt_on_critical
        self._history: list[ReconciliationResult] = []

    def reconcile(
        self,
        system_positions: dict[str, SystemPosition],
    ) -> ReconciliationResult:
        """Run a full reconciliation.

        Args:
            system_positions: What the system thinks it holds (symbol → position).

        Returns:
            ReconciliationResult with any discrepancies found.
        """
        broker_positions_list = self.broker.get_positions()
        broker_positions = {p.symbol: p for p in broker_positions_list}

        discrepancies: list[Discrepancy] = []
        now = datetime.now(UTC)

        all_symbols = set(system_positions.keys()) | set(broker_positions.keys())

        for symbol in sorted(all_symbols):
            sys_pos = system_positions.get(symbol)
            brk_pos = broker_positions.get(symbol)

            if sys_pos is None and brk_pos is not None:
                # Broker has it, system doesn't
                discrepancies.append(Discrepancy(
                    symbol=symbol,
                    type=DiscrepancyType.MISSING_IN_SYSTEM,
                    severity=Severity.CRITICAL,
                    system_qty=0.0,
                    broker_qty=brk_pos.qty,
                    system_value=0.0,
                    broker_value=brk_pos.market_value,
                    message=(
                        f"Broker holds {brk_pos.qty} shares "
                        f"(${brk_pos.market_value:.2f}) but system has no record"
                    ),
                    timestamp=now,
                ))

            elif sys_pos is not None and brk_pos is None:
                # System thinks it has it, broker doesn't
                discrepancies.append(Discrepancy(
                    symbol=symbol,
                    type=DiscrepancyType.MISSING_IN_BROKER,
                    severity=Severity.CRITICAL,
                    system_qty=sys_pos.qty,
                    broker_qty=0.0,
                    system_value=sys_pos.qty * sys_pos.avg_entry_price,
                    broker_value=0.0,
                    message=(
                        f"System thinks it holds {sys_pos.qty} shares "
                        f"but broker has no position"
                    ),
                    timestamp=now,
                ))

            elif sys_pos is not None and brk_pos is not None:
                # Both have it — check quantities
                qty_diff = abs(sys_pos.qty - brk_pos.qty)
                if qty_diff > self.qty_tolerance:
                    severity = (
                        Severity.CRITICAL if qty_diff > 1.0
                        else Severity.WARNING
                    )
                    discrepancies.append(Discrepancy(
                        symbol=symbol,
                        type=DiscrepancyType.QTY_MISMATCH,
                        severity=severity,
                        system_qty=sys_pos.qty,
                        broker_qty=brk_pos.qty,
                        system_value=sys_pos.qty * sys_pos.avg_entry_price,
                        broker_value=brk_pos.market_value,
                        message=(
                            f"Qty mismatch: system={sys_pos.qty}, "
                            f"broker={brk_pos.qty} (diff={qty_diff:.4f})"
                        ),
                        timestamp=now,
                    ))

                # Check side
                if sys_pos.side != brk_pos.side:
                    discrepancies.append(Discrepancy(
                        symbol=symbol,
                        type=DiscrepancyType.SIDE_MISMATCH,
                        severity=Severity.CRITICAL,
                        system_qty=sys_pos.qty,
                        broker_qty=brk_pos.qty,
                        system_value=sys_pos.qty * sys_pos.avg_entry_price,
                        broker_value=brk_pos.market_value,
                        message=(
                            f"Side mismatch: system={sys_pos.side}, "
                            f"broker={brk_pos.side}"
                        ),
                        timestamp=now,
                    ))

        system_total = sum(
            sp.qty * sp.avg_entry_price for sp in system_positions.values()
        )
        broker_total = sum(bp.market_value for bp in broker_positions.values())

        is_clean = len(discrepancies) == 0

        result = ReconciliationResult(
            timestamp=now,
            discrepancies=discrepancies,
            system_positions=system_positions,
            broker_positions=broker_positions,
            system_total_value=system_total,
            broker_total_value=broker_total,
            is_clean=is_clean,
        )

        self._history.append(result)

        if is_clean:
            logger.info("Reconciliation CLEAN: %s", result.summary())
        else:
            for d in discrepancies:
                if d.severity == Severity.CRITICAL:
                    logger.critical("Reconciliation: %s", d)
                else:
                    logger.warning("Reconciliation: %s", d)

            if self.halt_on_critical:
                critical = [d for d in discrepancies if d.severity == Severity.CRITICAL]
                if critical:
                    logger.critical(
                        "RECOMMEND TRADING HALT: %d critical discrepancies found. "
                        "Resolve before placing new orders.",
                        len(critical),
                    )
                    notify(
                        AlertSeverity.CRITICAL,
                        "Reconciliation — Trading Halt Recommended",
                        f"{len(critical)} critical discrepancies between system and broker.",
                        {
                            "discrepancies": [str(d) for d in critical],
                            "value_diff": round(result.value_discrepancy, 2),
                        },
                    )

        return result

    @property
    def last_result(self) -> ReconciliationResult | None:
        return self._history[-1] if self._history else None

    @property
    def consecutive_clean_count(self) -> int:
        """Number of consecutive clean reconciliations (confidence measure)."""
        count = 0
        for r in reversed(self._history):
            if r.is_clean:
                count += 1
            else:
                break
        return count
