"""Unit tests for infrastructure modules (no database required)."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from pipeline.infrastructure.checkpoint import CheckpointContext, CheckpointManager
from pipeline.infrastructure.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitState,
)
from pipeline.infrastructure.metrics import MetricsCollector, PipelineMetrics


class TestCircuitBreaker:
    """Tests for circuit breaker pattern."""

    def test_initial_state_is_closed(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        assert cb.state == CircuitState.CLOSED

    def test_opens_after_failure_threshold(self):
        cb = CircuitBreaker("test", failure_threshold=3)

        for _ in range(3):
            cb._on_failure()

        assert cb.state == CircuitState.OPEN

    def test_rejects_calls_when_open(self):
        cb = CircuitBreaker("test", failure_threshold=1)
        cb._on_failure()

        with pytest.raises(CircuitBreakerOpenError):
            cb.call(lambda: "result")

    def test_success_decrements_failure_count(self):
        cb = CircuitBreaker("test", failure_threshold=5)
        cb._on_failure()
        cb._on_failure()
        assert cb._failure_count == 2

        cb._on_success()
        assert cb._failure_count == 1

    def test_half_open_after_recovery_timeout(self):
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.01)
        cb._on_failure()
        assert cb._state == CircuitState.OPEN

        time.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN

    def test_closes_after_successes_in_half_open(self):
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.01)
        cb._on_failure()
        time.sleep(0.02)

        # Should be half open now
        assert cb.state == CircuitState.HALF_OPEN

        # 3 successes should close it
        cb._state = CircuitState.HALF_OPEN
        for _ in range(3):
            cb._on_success()

        assert cb._state == CircuitState.CLOSED

    def test_decorator_usage(self):
        cb = CircuitBreaker("test", failure_threshold=5)

        @cb
        def success_fn():
            return 42

        assert success_fn() == 42


class TestCheckpointManager:
    """Tests for checkpoint manager."""

    def test_save_and_load_checkpoint(self, tmp_path):
        mgr = CheckpointManager(tmp_path)
        mgr.save_checkpoint("op1", {"step": 5, "total": 10})

        checkpoint = mgr.load_checkpoint("op1")

        assert checkpoint is not None
        assert checkpoint["state"]["step"] == 5
        assert checkpoint["operation_id"] == "op1"

    def test_load_nonexistent_checkpoint(self, tmp_path):
        mgr = CheckpointManager(tmp_path)
        assert mgr.load_checkpoint("nonexistent") is None

    def test_delete_checkpoint(self, tmp_path):
        mgr = CheckpointManager(tmp_path)
        mgr.save_checkpoint("op1", {"step": 1})

        assert mgr.delete_checkpoint("op1") is True
        assert mgr.load_checkpoint("op1") is None

    def test_list_checkpoints(self, tmp_path):
        mgr = CheckpointManager(tmp_path)
        mgr.save_checkpoint("op_a", {"step": 1})
        mgr.save_checkpoint("op_b", {"step": 2})

        ids = mgr.list_checkpoints()
        assert set(ids) == {"op_a", "op_b"}

    def test_get_progress(self, tmp_path):
        mgr = CheckpointManager(tmp_path)
        mgr.save_checkpoint("op1", {
            "completed_items": 5,
            "total_items": 10,
            "last_processed": "item_5",
        })

        progress = mgr.get_progress("op1")
        assert progress["completed_items"] == 5
        assert progress["total_items"] == 10

    def test_checkpoint_context_success(self, tmp_path):
        mgr = CheckpointManager(tmp_path)

        with CheckpointContext(mgr, "op_ctx") as ctx:
            ctx.update(step=1)
            ctx.update(step=2)

        # Checkpoint should be cleaned up on success
        assert mgr.load_checkpoint("op_ctx") is None

    def test_checkpoint_context_failure_saves(self, tmp_path):
        mgr = CheckpointManager(tmp_path)

        with pytest.raises(ValueError):
            with CheckpointContext(mgr, "op_fail") as ctx:
                ctx.update(step=3)
                raise ValueError("boom")

        checkpoint = mgr.load_checkpoint("op_fail")
        assert checkpoint is not None
        assert checkpoint["state"]["step"] == 3


class TestMetricsCollector:
    """Tests for metrics collector."""

    def test_counter_increments(self):
        mc = MetricsCollector(namespace="test")
        mc.counter("requests", 1)
        mc.counter("requests", 1)

        assert mc._counters["test.requests"] == 2

    def test_gauge_sets(self):
        mc = MetricsCollector(namespace="test")
        mc.gauge("temperature", 42.0)

        assert mc._gauges["test.temperature"] == 42.0

    def test_histogram_records(self):
        mc = MetricsCollector(namespace="test")
        mc.histogram("latency", 10.0)
        mc.histogram("latency", 20.0)

        assert mc._histograms["test.latency"] == [10.0, 20.0]

    def test_timer_context(self):
        mc = MetricsCollector(namespace="test")

        with mc.timer_context("op"):
            time.sleep(0.01)

        assert len(mc._timers["test.op"]) == 1
        assert mc._timers["test.op"][0] >= 10  # at least 10ms

    def test_get_summary(self):
        mc = MetricsCollector(namespace="test")
        mc.counter("req", 5)
        mc.gauge("temp", 42)
        mc.histogram("lat", 100)
        mc.histogram("lat", 200)
        mc.timer("op", 50.0)

        summary = mc.get_summary()

        assert summary["counters"]["test.req"] == 5
        assert summary["gauges"]["test.temp"] == 42
        assert summary["histograms"]["test.lat"]["count"] == 2
        assert summary["timers"]["test.op"]["count"] == 1

    def test_export_to_file(self, tmp_path):
        mc = MetricsCollector(namespace="test")
        mc.counter("req", 3)

        path = tmp_path / "metrics.json"
        mc.export_to_file(path)

        with open(path) as f:
            data = json.load(f)

        assert "counters" in data
        assert data["counters"]["test.req"] == 3

    def test_reset(self):
        mc = MetricsCollector(namespace="test")
        mc.counter("req", 5)
        mc.gauge("temp", 42)

        mc.reset()

        assert len(mc._counters) == 0
        assert len(mc._gauges) == 0

    def test_pipeline_metrics_record_extracted(self):
        import pipeline.infrastructure.metrics as metrics_mod

        metrics_mod._metrics = None
        pm = PipelineMetrics("test_pipeline")
        pm.record_extracted("fred", 100)

        summary = pm.metrics.get_summary()
        assert summary["counters"]["mdw.records_extracted"] == 100

    def test_pipeline_metrics_record_error(self):
        import pipeline.infrastructure.metrics as metrics_mod

        metrics_mod._metrics = None
        pm = PipelineMetrics("test_pipeline")
        pm.record_error("ConnectionError")

        summary = pm.metrics.get_summary()
        assert summary["counters"]["mdw.errors"] == 1
