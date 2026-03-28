"""Handle corrupted or malformed data files and records.

Provides a quarantine mechanism that logs corrupt files and records so they
can be investigated later without crashing the pipeline.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CorruptionRecord:
    """A single corruption event."""

    source: str
    file_path: str | None
    record_index: int | None
    error_type: str
    error_message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    details: dict[str, Any] = field(default_factory=dict)


class CorruptionHandler:
    """Track and quarantine corrupted files and records.

    Accumulates corruption events during a pipeline run and persists them
    to a JSONL quarantine log.  The handler distinguishes between two kinds
    of corruption:

    * **File-level** – the parquet file itself cannot be read (truncated,
      wrong format, permission error, etc.).
    * **Record-level** – the file is readable but individual rows have
      missing required fields, invalid types, or values that violate
      constraints.

    Usage::

        handler = CorruptionHandler("prices")
        handler.record_corrupt_file(path, error)
        handler.record_corrupt_record(path, idx, error, raw_record)
        handler.flush()  # persist to quarantine log
    """

    def __init__(
        self,
        source: str,
        quarantine_dir: Path | None = None,
    ):
        self.source = source
        self._quarantine_dir = quarantine_dir or Path("data/quarantine")
        self._events: list[CorruptionRecord] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def corrupt_file_count(self) -> int:
        return sum(1 for e in self._events if e.record_index is None)

    @property
    def corrupt_record_count(self) -> int:
        return sum(1 for e in self._events if e.record_index is not None)

    @property
    def has_events(self) -> bool:
        return len(self._events) > 0

    def record_corrupt_file(
        self,
        file_path: Path | str,
        error: Exception,
    ) -> None:
        """Log a file that could not be read at all."""
        event = CorruptionRecord(
            source=self.source,
            file_path=str(file_path),
            record_index=None,
            error_type=type(error).__name__,
            error_message=str(error),
        )
        self._events.append(event)
        logger.error(
            "Corrupt file quarantined: %s (%s: %s)",
            file_path,
            type(error).__name__,
            error,
        )

    def record_corrupt_record(
        self,
        file_path: Path | str | None,
        record_index: int,
        error: Exception | str,
        raw_record: dict[str, Any] | None = None,
    ) -> None:
        """Log a single record that failed validation."""
        if isinstance(error, Exception):
            error_type = type(error).__name__
            error_message = str(error)
        else:
            error_type = "ValidationError"
            error_message = error

        details: dict[str, Any] = {}
        if raw_record is not None:
            # Store a truncated snapshot for debugging
            try:
                snapshot = json.dumps(raw_record, default=str)
                if len(snapshot) > 2000:
                    snapshot = snapshot[:2000] + "..."
                details["raw_record"] = snapshot
            except (TypeError, ValueError):
                details["raw_record"] = "<unserializable>"

        event = CorruptionRecord(
            source=self.source,
            file_path=str(file_path) if file_path else None,
            record_index=record_index,
            error_type=error_type,
            error_message=error_message,
            details=details,
        )
        self._events.append(event)
        logger.warning(
            "Corrupt record quarantined: %s row %d (%s: %s)",
            file_path or "<unknown>",
            record_index,
            error_type,
            error_message,
        )

    def flush(self) -> Path | None:
        """Persist accumulated events to a JSONL quarantine log.

        Returns the path to the log file, or ``None`` if there were no events.
        """
        if not self._events:
            return None

        self._quarantine_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        log_path = self._quarantine_dir / f"{self.source}_{ts}.jsonl"

        with open(log_path, "w") as fh:
            for event in self._events:
                line = {
                    "source": event.source,
                    "file_path": event.file_path,
                    "record_index": event.record_index,
                    "error_type": event.error_type,
                    "error_message": event.error_message,
                    "timestamp": event.timestamp.isoformat(),
                    "details": event.details,
                }
                fh.write(json.dumps(line, default=str) + "\n")

        logger.info(
            "Quarantine log written: %s (%d file errors, %d record errors)",
            log_path,
            self.corrupt_file_count,
            self.corrupt_record_count,
        )
        return log_path

    def summary(self) -> dict[str, Any]:
        """Return a summary dict suitable for pipeline reporting."""
        return {
            "source": self.source,
            "corrupt_files": self.corrupt_file_count,
            "corrupt_records": self.corrupt_record_count,
            "total_events": len(self._events),
        }


def read_parquet_safe(file_path: Path) -> tuple[Any | None, Exception | None]:
    """Read a parquet file, returning (dataframe, None) on success or (None, error) on failure.

    Handles corrupted, truncated, and invalid files gracefully.
    """
    import pandas as pd
    import pyarrow  # noqa: F401 – import to surface ArrowInvalid early

    try:
        df = pd.read_parquet(file_path)
        return df, None
    except FileNotFoundError as e:
        return None, e
    except (
        # pyarrow raises these for corrupt/truncated files
        Exception
    ) as e:
        return None, e


def validate_required_fields(
    record: dict[str, Any],
    required_fields: list[str],
) -> str | None:
    """Check that all required fields are present and non-null.

    Returns an error message string if validation fails, or ``None`` if OK.
    """
    missing = [f for f in required_fields if record.get(f) is None]
    if missing:
        return f"Missing required fields: {', '.join(missing)}"
    return None
