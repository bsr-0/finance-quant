"""Persistent position register for multi-day position tracking.

Stores TrackedPosition state to disk so that stop-losses, trailing stops,
profit targets, and time exits survive across daily runner restarts.

Uses atomic writes (temp file + os.replace) with file locking for
concurrency safety, following the same pattern as experiment_registry.py.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from pipeline.execution.position_monitor import TrackedPosition

logger = logging.getLogger(__name__)


def _tracked_to_dict(pos: TrackedPosition) -> dict:
    return {
        "symbol": pos.symbol,
        "entry_date": pos.entry_date.isoformat(),
        "entry_price": pos.entry_price,
        "shares": pos.shares,
        "stop_price": pos.stop_price,
        "atr_at_entry": pos.atr_at_entry,
        "trailing_stop": pos.trailing_stop,
        "trailing_activated": pos.trailing_activated,
        "highest_price": pos.highest_price,
        "target_1": pos.target_1,
        "target_2": pos.target_2,
        "signal_score": pos.signal_score,
    }


def _dict_to_tracked(d: dict) -> TrackedPosition:
    entry_date = d["entry_date"]
    if isinstance(entry_date, str):
        dt = datetime.fromisoformat(entry_date)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        entry_date = dt
    return TrackedPosition(
        symbol=d["symbol"],
        entry_date=entry_date,
        entry_price=d["entry_price"],
        shares=d["shares"],
        stop_price=d["stop_price"],
        atr_at_entry=d.get("atr_at_entry", 0.0),
        trailing_stop=d.get("trailing_stop", 0.0),
        trailing_activated=d.get("trailing_activated", False),
        highest_price=d.get("highest_price", 0.0),
        target_1=d.get("target_1", 0.0),
        target_2=d.get("target_2", 0.0),
        signal_score=d.get("signal_score", 0),
    )


class PositionRegister:
    """JSON-backed persistent store for open TrackedPosition objects.

    Usage::

        register = PositionRegister("data/open_positions.json")

        # Save current positions
        register.save(monitor.tracked_positions)

        # Load on next startup
        positions = register.load()
    """

    def __init__(self, path: str | Path = "data/open_positions.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, positions: dict[str, TrackedPosition]) -> None:
        """Atomic write of all tracked positions to disk."""
        data = {sym: _tracked_to_dict(pos) for sym, pos in positions.items()}
        content = json.dumps(data, indent=2)

        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.path.parent), suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    f.write(content)
                    f.flush()
                    os.fsync(f.fileno())
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
            os.replace(tmp_path, str(self.path))
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise

        logger.debug("Position register saved: %d positions", len(positions))

    def load(self) -> dict[str, TrackedPosition]:
        """Load tracked positions from disk. Returns empty dict if no file."""
        if not self.path.exists():
            return {}
        try:
            with open(self.path) as f:
                fcntl.flock(f, fcntl.LOCK_SH)
                try:
                    data = json.load(f)
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
            positions = {sym: _dict_to_tracked(d) for sym, d in data.items()}
            logger.info(
                "Position register loaded: %d open positions", len(positions)
            )
            return positions
        except (json.JSONDecodeError, KeyError):
            logger.warning(
                "Corrupt position register at %s — returning empty",
                self.path,
                exc_info=True,
            )
            return {}

    def clear(self) -> None:
        """Remove the register file (all positions closed)."""
        with contextlib.suppress(FileNotFoundError):
            self.path.unlink()
        logger.info("Position register cleared")
