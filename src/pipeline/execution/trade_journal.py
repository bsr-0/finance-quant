"""Trade journal: persists execution results to disk for audit and analysis.

Every order submission, fill, rejection, and exit is logged as a CSV row
in a daily journal file.  This provides a durable record of all trading
activity independent of broker statements.

Usage::

    journal = TradeJournal(journal_dir="logs/trade_journal")

    # After order submission:
    journal.record_order(order, signal_score=75, signal_regime="BULL")

    # After fill confirmation:
    journal.record_fill(order)

    # After exit:
    journal.record_exit(symbol="AAPL", reason="stop_loss", exit_price=145.0, pnl=-5.0)

    # Read back:
    df = journal.read_journal("2025-03-06")
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline.execution.broker import Order

logger = logging.getLogger(__name__)

# Column order for the journal CSV
_COLUMNS = [
    "timestamp",
    "event_type",
    "symbol",
    "side",
    "order_type",
    "qty",
    "limit_price",
    "stop_price",
    "filled_qty",
    "filled_avg_price",
    "order_id",
    "status",
    "reject_reason",
    "signal_score",
    "signal_regime",
    "signal_confidence",
    "exit_reason",
    "pnl",
    "notes",
]


class TradeJournal:
    """Append-only trade journal persisted as daily CSV files.

    Thread-safe: each write opens/closes the file independently.
    """

    def __init__(self, journal_dir: str | Path = "logs/trade_journal") -> None:
        self._dir = Path(journal_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _journal_path(self, date: datetime | None = None) -> Path:
        dt = date or datetime.now(timezone.utc)
        return self._dir / f"journal_{dt.strftime('%Y%m%d')}.csv"

    def _append_row(self, row: dict[str, Any]) -> None:
        """Append a single row to today's journal file."""
        path = self._journal_path()
        write_header = not path.exists()

        try:
            with open(path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=_COLUMNS, extrasaction="ignore")
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
        except OSError as e:
            logger.warning("Failed to write journal entry: %s", e)

    def record_order(
        self,
        order: Order,
        signal_score: int = 0,
        signal_regime: str = "",
        signal_confidence: str = "",
        notes: str = "",
    ) -> None:
        """Record an order submission event."""
        self._append_row({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "ORDER_SUBMITTED",
            "symbol": order.symbol,
            "side": order.side.value,
            "order_type": order.order_type.value,
            "qty": order.qty,
            "limit_price": order.limit_price or "",
            "stop_price": order.stop_price or "",
            "filled_qty": order.filled_qty,
            "filled_avg_price": order.filled_avg_price or "",
            "order_id": order.order_id,
            "status": order.status.value,
            "reject_reason": "",
            "signal_score": signal_score,
            "signal_regime": signal_regime,
            "signal_confidence": signal_confidence,
            "exit_reason": "",
            "pnl": "",
            "notes": notes,
        })
        logger.debug("Journal: ORDER_SUBMITTED %s %s", order.symbol, order.order_id)

    def record_fill(self, order: Order, notes: str = "") -> None:
        """Record an order fill event."""
        self._append_row({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "ORDER_FILLED",
            "symbol": order.symbol,
            "side": order.side.value,
            "order_type": order.order_type.value,
            "qty": order.qty,
            "limit_price": order.limit_price or "",
            "stop_price": order.stop_price or "",
            "filled_qty": order.filled_qty,
            "filled_avg_price": order.filled_avg_price,
            "order_id": order.order_id,
            "status": order.status.value,
            "reject_reason": "",
            "signal_score": "",
            "signal_regime": "",
            "signal_confidence": "",
            "exit_reason": "",
            "pnl": "",
            "notes": notes,
        })
        logger.debug("Journal: ORDER_FILLED %s %s", order.symbol, order.order_id)

    def record_rejection(self, order: Order, reason: str = "", notes: str = "") -> None:
        """Record an order rejection event."""
        self._append_row({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "ORDER_REJECTED",
            "symbol": order.symbol,
            "side": order.side.value if hasattr(order.side, "value") else str(order.side),
            "order_type": order.order_type.value if hasattr(order.order_type, "value") else str(order.order_type),
            "qty": order.qty,
            "limit_price": order.limit_price or "",
            "stop_price": order.stop_price or "",
            "filled_qty": 0,
            "filled_avg_price": "",
            "order_id": order.order_id,
            "status": "rejected",
            "reject_reason": reason or order.reject_reason,
            "signal_score": "",
            "signal_regime": "",
            "signal_confidence": "",
            "exit_reason": "",
            "pnl": "",
            "notes": notes,
        })

    def record_exit(
        self,
        symbol: str,
        reason: str,
        exit_price: float,
        pnl: float,
        shares: float = 0,
        order_id: str = "",
        notes: str = "",
    ) -> None:
        """Record a position exit event."""
        self._append_row({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "POSITION_EXIT",
            "symbol": symbol,
            "side": "sell",
            "order_type": "market",
            "qty": shares,
            "limit_price": "",
            "stop_price": "",
            "filled_qty": shares,
            "filled_avg_price": exit_price,
            "order_id": order_id,
            "status": "filled",
            "reject_reason": "",
            "signal_score": "",
            "signal_regime": "",
            "signal_confidence": "",
            "exit_reason": reason,
            "pnl": round(pnl, 2),
            "notes": notes,
        })
        logger.debug("Journal: POSITION_EXIT %s reason=%s pnl=$%.2f", symbol, reason, pnl)

    def record_guard_rejection(
        self,
        symbol: str,
        checks_failed: list[str],
        notes: str = "",
    ) -> None:
        """Record a capital guard rejection."""
        self._append_row({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "GUARD_REJECTED",
            "symbol": symbol,
            "side": "buy",
            "order_type": "",
            "qty": "",
            "limit_price": "",
            "stop_price": "",
            "filled_qty": "",
            "filled_avg_price": "",
            "order_id": "",
            "status": "rejected",
            "reject_reason": "; ".join(checks_failed),
            "signal_score": "",
            "signal_regime": "",
            "signal_confidence": "",
            "exit_reason": "",
            "pnl": "",
            "notes": notes,
        })

    def read_journal(self, date_str: str | None = None) -> list[dict[str, str]]:
        """Read journal entries for a given date (YYYY-MM-DD or YYYYMMDD).

        Returns list of dicts (one per row).  Returns empty list if no
        journal file exists for the date.
        """
        if date_str is None:
            date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        date_str = date_str.replace("-", "")
        path = self._dir / f"journal_{date_str}.csv"
        if not path.exists():
            return []
        with open(path, newline="") as f:
            return list(csv.DictReader(f))

    @property
    def journal_dir(self) -> Path:
        return self._dir
