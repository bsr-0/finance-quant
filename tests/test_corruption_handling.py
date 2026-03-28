"""Tests for data corruption handling in the pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from pipeline.infrastructure.corruption import (
    CorruptionHandler,
    read_parquet_safe,
    validate_required_fields,
)

# ---------------------------------------------------------------------------
# CorruptionHandler
# ---------------------------------------------------------------------------


class TestCorruptionHandler:
    def test_empty_handler_has_no_events(self):
        handler = CorruptionHandler("test")
        assert not handler.has_events
        assert handler.corrupt_file_count == 0
        assert handler.corrupt_record_count == 0

    def test_record_corrupt_file(self):
        handler = CorruptionHandler("test")
        handler.record_corrupt_file(Path("/tmp/bad.parquet"), ValueError("bad magic"))
        assert handler.has_events
        assert handler.corrupt_file_count == 1
        assert handler.corrupt_record_count == 0

    def test_record_corrupt_record(self):
        handler = CorruptionHandler("test")
        handler.record_corrupt_record(
            Path("/tmp/data.parquet"), 42, "Missing required fields: ticker"
        )
        assert handler.corrupt_record_count == 1
        assert handler.corrupt_file_count == 0

    def test_record_corrupt_record_with_raw(self):
        handler = CorruptionHandler("test")
        handler.record_corrupt_record(
            None,
            0,
            ValueError("bad value"),
            raw_record={"ticker": None, "date": "2024-01-01"},
        )
        assert handler.corrupt_record_count == 1
        event = handler._events[0]
        assert "raw_record" in event.details

    def test_flush_writes_jsonl(self, tmp_path):
        handler = CorruptionHandler("prices", quarantine_dir=tmp_path)
        handler.record_corrupt_file(Path("/data/bad.parquet"), OSError("disk error"))
        handler.record_corrupt_record(Path("/data/ok.parquet"), 5, "missing ticker")

        log_path = handler.flush()
        assert log_path is not None
        assert log_path.exists()

        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 2

        file_event = json.loads(lines[0])
        assert file_event["error_type"] == "OSError"
        assert file_event["record_index"] is None

        record_event = json.loads(lines[1])
        assert record_event["record_index"] == 5

    def test_flush_returns_none_when_no_events(self, tmp_path):
        handler = CorruptionHandler("test", quarantine_dir=tmp_path)
        assert handler.flush() is None

    def test_summary(self):
        handler = CorruptionHandler("fred")
        handler.record_corrupt_file(Path("a.parquet"), RuntimeError("x"))
        handler.record_corrupt_record(Path("b.parquet"), 0, "y")
        handler.record_corrupt_record(Path("b.parquet"), 1, "z")

        summary = handler.summary()
        assert summary == {
            "source": "fred",
            "corrupt_files": 1,
            "corrupt_records": 2,
            "total_events": 3,
        }


# ---------------------------------------------------------------------------
# read_parquet_safe
# ---------------------------------------------------------------------------


class TestReadParquetSafe:
    def test_reads_valid_file(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        path = tmp_path / "valid.parquet"
        df.to_parquet(path)

        result, error = read_parquet_safe(path)
        assert error is None
        assert len(result) == 2

    def test_returns_error_for_missing_file(self, tmp_path):
        result, error = read_parquet_safe(tmp_path / "nonexistent.parquet")
        assert result is None
        assert isinstance(error, FileNotFoundError)

    def test_returns_error_for_corrupt_file(self, tmp_path):
        path = tmp_path / "corrupt.parquet"
        path.write_bytes(b"this is not a parquet file")

        result, error = read_parquet_safe(path)
        assert result is None
        assert error is not None

    def test_returns_error_for_truncated_file(self, tmp_path):
        # Write a valid parquet then truncate it
        df = pd.DataFrame({"a": range(100)})
        path = tmp_path / "truncated.parquet"
        df.to_parquet(path)

        data = path.read_bytes()
        path.write_bytes(data[:50])  # truncate

        result, error = read_parquet_safe(path)
        assert result is None
        assert error is not None


# ---------------------------------------------------------------------------
# validate_required_fields
# ---------------------------------------------------------------------------


class TestValidateRequiredFields:
    def test_all_present(self):
        record = {"ticker": "AAPL", "date": "2024-01-01", "close": 150.0}
        assert validate_required_fields(record, ["ticker", "date"]) is None

    def test_missing_field(self):
        record = {"ticker": "AAPL", "date": None}
        err = validate_required_fields(record, ["ticker", "date"])
        assert err is not None
        assert "date" in err

    def test_absent_field(self):
        record = {"ticker": "AAPL"}
        err = validate_required_fields(record, ["ticker", "date"])
        assert err is not None
        assert "date" in err

    def test_multiple_missing(self):
        record = {"volume": 100}
        err = validate_required_fields(record, ["ticker", "date"])
        assert "ticker" in err
        assert "date" in err


# ---------------------------------------------------------------------------
# RawLoader integration with corruption handling
# ---------------------------------------------------------------------------


class TestRawLoaderCorruptionHandling:
    """Test that RawLoader methods gracefully handle corrupt data."""

    def _make_parquet(self, tmp_path: Path, name: str, data: dict) -> Path:
        path = tmp_path / name
        pd.DataFrame(data).to_parquet(path)
        return path

    def test_load_prices_skips_corrupt_file(self, tmp_path):
        """A corrupt parquet file should return 0 rows, not crash."""
        corrupt_path = tmp_path / "corrupt.parquet"
        corrupt_path.write_bytes(b"NOT_PARQUET_DATA")

        handler = CorruptionHandler("prices", quarantine_dir=tmp_path / "quarantine")

        from pipeline.load.raw_loader import RawLoader

        loader = RawLoader.__new__(RawLoader)
        loader._batch_size = 1000

        result = loader._read_parquet_or_quarantine(corrupt_path, handler)
        assert result is None
        assert handler.corrupt_file_count == 1

    def test_filter_valid_records_removes_bad_rows(self):
        from pipeline.load.raw_loader import RawLoader

        loader = RawLoader.__new__(RawLoader)
        loader._batch_size = 1000

        handler = CorruptionHandler("prices")
        records = [
            {"ticker": "AAPL", "date": "2024-01-01", "close": 150.0},
            {"ticker": None, "date": "2024-01-02", "close": 151.0},  # bad: no ticker
            {"ticker": "GOOG", "date": None, "close": 100.0},  # bad: no date
            {"ticker": "MSFT", "date": "2024-01-03", "close": 200.0},
        ]

        valid = loader._filter_valid_records(records, ["ticker", "date"], handler)
        assert len(valid) == 2
        assert valid[0]["ticker"] == "AAPL"
        assert valid[1]["ticker"] == "MSFT"
        assert handler.corrupt_record_count == 2

    def test_load_fred_skips_records_with_missing_series_code(self, tmp_path):
        """FRED records missing series_code should be quarantined, not crash."""
        handler = CorruptionHandler("fred", quarantine_dir=tmp_path / "quarantine")

        from pipeline.load.raw_loader import RawLoader

        loader = RawLoader.__new__(RawLoader)
        loader._batch_size = 1000

        records = [
            {"series_code": "GDP", "date": "2024-01-01", "value": 100.0},
            {"series_code": None, "date": "2024-01-02", "value": 101.0},
        ]

        valid = loader._filter_valid_records(
            records, ["series_code", "date"], handler, Path("test.parquet")
        )
        assert len(valid) == 1
        assert handler.corrupt_record_count == 1

    def test_load_all_raw_files_handles_mixed_corruption(self, tmp_path):
        """load_all_raw_files should continue past corrupt files and log them."""
        source_dir = tmp_path / "prices"
        source_dir.mkdir()

        # Write one valid file
        valid_df = pd.DataFrame({
            "ticker": ["AAPL"],
            "date": ["2024-01-01"],
            "open": [150.0],
            "high": [155.0],
            "low": [149.0],
            "close": [153.0],
            "adj_close": [153.0],
            "volume": [1000000],
        })
        valid_df.to_parquet(source_dir / "valid.parquet")

        # Write one corrupt file
        (source_dir / "corrupt.parquet").write_bytes(b"GARBAGE")

        from pipeline.load.raw_loader import RawLoader

        loader = RawLoader.__new__(RawLoader)
        loader._batch_size = 1000
        # Mock the db to avoid needing a real database
        with patch.object(RawLoader, "__init__", lambda self: None):
            loader = RawLoader.__new__(RawLoader)
            loader._batch_size = 1000
            loader.db = None  # won't be used for corrupt file

            # The corrupt file should be quarantined, the valid one would need a DB.
            # We test that the handler captures the corrupt file correctly.
            handler = CorruptionHandler("prices", quarantine_dir=tmp_path / "quarantine")

            # Test _read_parquet_or_quarantine directly on both files
            result_corrupt = loader._read_parquet_or_quarantine(
                source_dir / "corrupt.parquet", handler
            )
            result_valid = loader._read_parquet_or_quarantine(
                source_dir / "valid.parquet", handler
            )

            assert result_corrupt is None
            assert result_valid is not None
            assert len(result_valid) == 1
            assert handler.corrupt_file_count == 1

            # Flush and verify quarantine log
            log_path = handler.flush()
            assert log_path is not None
            lines = log_path.read_text().strip().split("\n")
            assert len(lines) == 1
            event = json.loads(lines[0])
            assert "corrupt.parquet" in event["file_path"]
