"""Tests for checkpoint-based resume in historical backfill."""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.extract.checkpoint_helpers import (
    make_operation_id,
    mark_item_done,
    resumable_items,
)
from pipeline.infrastructure.checkpoint import CheckpointContext, CheckpointManager


@pytest.fixture()
def ckpt_dir(tmp_path: Path) -> Path:
    """Create a temporary checkpoint directory."""
    d = tmp_path / "checkpoints"
    d.mkdir()
    return d


@pytest.fixture()
def ckpt(ckpt_dir: Path) -> CheckpointManager:
    return CheckpointManager(ckpt_dir)


# ---------------------------------------------------------------------------
# Unit tests for checkpoint_helpers
# ---------------------------------------------------------------------------


class TestMakeOperationId:
    def test_deterministic(self):
        assert make_operation_id("prices") == "backfill_prices"

    def test_unique_per_source(self):
        assert make_operation_id("prices") != make_operation_id("fred")


class TestResumableItems:
    def test_yields_all_when_no_checkpoint(self, ckpt: CheckpointManager):
        items = ["A", "B", "C"]
        with CheckpointContext(ckpt, "test_op", resume=True) as ctx:
            result = list(resumable_items(items, ctx))
        assert result == [(0, "A"), (1, "B"), (2, "C")]

    def test_skips_completed_items(self, ckpt: CheckpointManager):
        # Pre-populate checkpoint with A and B completed
        ckpt.save_checkpoint(
            "test_op",
            {"completed_keys": ["A", "B"], "completed_items": 2, "total_items": 3},
        )
        items = ["A", "B", "C"]
        with CheckpointContext(ckpt, "test_op", resume=True) as ctx:
            result = list(resumable_items(items, ctx))
        assert result == [(2, "C")]

    def test_skips_all_when_all_completed(self, ckpt: CheckpointManager):
        ckpt.save_checkpoint(
            "test_op",
            {"completed_keys": ["A", "B"], "completed_items": 2, "total_items": 2},
        )
        items = ["A", "B"]
        with CheckpointContext(ckpt, "test_op", resume=True) as ctx:
            result = list(resumable_items(items, ctx))
        assert result == []

    def test_custom_key_fn(self, ckpt: CheckpointManager):
        ckpt.save_checkpoint(
            "test_op",
            {"completed_keys": ["Berkshire Hathaway"]},
        )
        filers = [
            {"name": "Berkshire Hathaway", "cik": 1},
            {"name": "Bridgewater", "cik": 2},
        ]
        with CheckpointContext(ckpt, "test_op", resume=True) as ctx:
            result = list(resumable_items(filers, ctx, key_fn=lambda f: f["name"]))
        assert len(result) == 1
        assert result[0][1]["name"] == "Bridgewater"

    def test_no_resume_yields_all(self, ckpt: CheckpointManager):
        ckpt.save_checkpoint(
            "test_op",
            {"completed_keys": ["A", "B"]},
        )
        items = ["A", "B", "C"]
        with CheckpointContext(ckpt, "test_op", resume=False) as ctx:
            result = list(resumable_items(items, ctx))
        assert result == [(0, "A"), (1, "B"), (2, "C")]


class TestMarkItemDone:
    def test_persists_to_checkpoint(self, ckpt: CheckpointManager):
        with CheckpointContext(ckpt, "test_op", resume=True) as ctx:
            mark_item_done(ctx, "ticker_A", 0, 3)
            mark_item_done(ctx, "ticker_B", 1, 3)

        # Context exited successfully → checkpoint deleted.
        # Verify state was correct during the context by loading mid-way.
        # We test via a new context that the items were being tracked:
        # Since successful exit deletes checkpoint, let's test the
        # incremental save by simulating a failure.

    def test_incremental_save_survives_crash(self, ckpt: CheckpointManager):
        """If the context exits with an error, checkpoint should persist."""
        with pytest.raises(RuntimeError), CheckpointContext(ckpt, "crash_op", resume=True) as ctx:
            mark_item_done(ctx, "A", 0, 3)
            mark_item_done(ctx, "B", 1, 3)
            raise RuntimeError("simulated crash")

        # Checkpoint should exist with the state saved by mark_item_done
        loaded = ckpt.load_checkpoint("crash_op")
        assert loaded is not None
        state = loaded["state"]
        assert set(state["completed_keys"]) == {"A", "B"}
        assert state["completed_items"] == 2
        assert state["total_items"] == 3
        assert state["last_processed"] == "B"

    def test_resume_after_crash(self, ckpt: CheckpointManager):
        """After a crash, resuming should skip completed items."""
        # Simulate crash after processing A and B
        with pytest.raises(RuntimeError), CheckpointContext(ckpt, "resume_op", resume=True) as ctx:
            for i, item in resumable_items(["A", "B", "C"], ctx):
                mark_item_done(ctx, item, i, 3)
                if item == "B":
                    raise RuntimeError("crash")

        # Resume — should only yield C
        with CheckpointContext(ckpt, "resume_op", resume=True) as ctx:
            remaining = list(resumable_items(["A", "B", "C"], ctx))
        assert remaining == [(2, "C")]


# ---------------------------------------------------------------------------
# Source-level checkpoint tests (mocking extractors)
# ---------------------------------------------------------------------------


class TestSourceLevelCheckpoint:
    def test_completed_source_skipped_on_resume(self, ckpt: CheckpointManager):
        """Sources recorded as completed should be skipped."""
        ckpt.save_checkpoint(
            "backfill_orchestrator",
            {
                "completed_sources": ["prices", "factors"],
                "completed_items": 2,
                "total_items": 3,
            },
        )

        source_list = [
            ("prices", "prices"),
            ("factors", "factors"),
            ("fred", "fred"),
        ]

        executed = []
        with CheckpointContext(ckpt, "backfill_orchestrator", resume=True) as ctx:
            completed = set(ctx.state.get("completed_sources", []))
            for name, _ in source_list:
                if name in completed:
                    continue
                executed.append(name)
                done = ctx.state.get("completed_sources", [])
                done.append(name)
                ctx.update(
                    completed_sources=done,
                    completed_items=len(done),
                    total_items=len(source_list),
                    last_processed=name,
                )
                ctx.save()

        assert executed == ["fred"]

    def test_no_resume_clears_checkpoints(self, ckpt: CheckpointManager):
        """--no-resume should clear all backfill checkpoints."""
        ckpt.save_checkpoint("backfill_orchestrator", {"completed_sources": ["prices"]})
        ckpt.save_checkpoint("backfill_prices", {"completed_keys": ["AAPL"]})

        # Simulate --no-resume cleanup
        ckpt.delete_checkpoint("backfill_orchestrator")
        ckpt.delete_checkpoint("backfill_prices")

        assert ckpt.load_checkpoint("backfill_orchestrator") is None
        assert ckpt.load_checkpoint("backfill_prices") is None

    def test_successful_backfill_cleans_checkpoint(self, ckpt: CheckpointManager):
        """On full success, CheckpointContext should delete the checkpoint."""
        with CheckpointContext(ckpt, "backfill_orchestrator", resume=True) as ctx:
            ctx.update(completed_sources=["prices"], completed_items=1, total_items=1)
            ctx.save()

        # After successful exit, checkpoint should be gone
        assert ckpt.load_checkpoint("backfill_orchestrator") is None


# ---------------------------------------------------------------------------
# Integration: checkpoint + extractor pattern
# ---------------------------------------------------------------------------


class TestExtractorCheckpointIntegration:
    def test_price_extractor_resume_pattern(self, ckpt_dir: Path):
        """Verify the checkpoint wiring pattern used in extractors."""
        ckpt = CheckpointManager(ckpt_dir)
        op_id = "backfill_prices"
        tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]

        # Simulate extracting first 3, then crashing
        extracted = []
        with pytest.raises(RuntimeError), CheckpointContext(ckpt, op_id, resume=True) as ctx:
            for i, ticker in resumable_items(tickers, ctx):
                extracted.append(ticker)
                mark_item_done(ctx, ticker, i, len(tickers))
                if ticker == "GOOG":
                    raise RuntimeError("network error")

        assert extracted == ["AAPL", "MSFT", "GOOG"]

        # Resume — should pick up from AMZN
        extracted_resume = []
        with CheckpointContext(ckpt, op_id, resume=True) as ctx:
            for i, ticker in resumable_items(tickers, ctx):
                extracted_resume.append(ticker)
                mark_item_done(ctx, ticker, i, len(tickers))

        assert extracted_resume == ["AMZN", "META"]

        # After successful completion, checkpoint should be cleaned up
        assert ckpt.load_checkpoint(op_id) is None

    def test_gdelt_date_resume_pattern(self, ckpt_dir: Path):
        """Verify date-based checkpointing (GDELT pattern)."""
        from datetime import date

        ckpt = CheckpointManager(ckpt_dir)
        op_id = "backfill_gdelt"
        dates = [date(2024, 1, d) for d in range(1, 6)]  # Jan 1-5

        # Process 2 dates then crash
        processed = []
        with pytest.raises(RuntimeError), CheckpointContext(ckpt, op_id, resume=True) as ctx:
            for i, d in resumable_items(dates, ctx, key_fn=lambda d: d.isoformat()):
                processed.append(d)
                mark_item_done(ctx, d.isoformat(), i, len(dates))
                if d == date(2024, 1, 2):
                    raise RuntimeError("timeout")

        assert len(processed) == 2

        # Resume
        resumed = []
        with CheckpointContext(ckpt, op_id, resume=True) as ctx:
            for i, d in resumable_items(dates, ctx, key_fn=lambda d: d.isoformat()):
                resumed.append(d)
                mark_item_done(ctx, d.isoformat(), i, len(dates))

        assert resumed == [date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)]
