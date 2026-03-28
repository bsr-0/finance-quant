"""Unit tests for infrastructure modules (no database required)."""

import contextlib
import json
import threading
import time

import pytest

from pipeline.infrastructure.checkpoint import CheckpointContext, CheckpointManager
from pipeline.infrastructure.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitState,
    get_circuit_breaker,
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

    def test_thread_safety_concurrent_failures(self):
        """Concurrent failures don't corrupt state or lose counts."""
        cb = CircuitBreaker("thread_test_failures", failure_threshold=1000)
        errors = []
        num_threads = 8
        calls_per_thread = 50

        def hammer_failures():
            for _ in range(calls_per_thread):
                try:
                    cb.call(lambda: (_ for _ in ()).throw(ValueError("boom")))
                except (ValueError, CircuitBreakerOpenError):
                    pass
                except Exception as e:
                    errors.append(str(e))

        threads = [threading.Thread(target=hammer_failures) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert cb._failure_count == num_threads * calls_per_thread

    def test_thread_safety_concurrent_successes(self):
        """Concurrent successes don't corrupt state."""
        cb = CircuitBreaker("thread_test_success", failure_threshold=100)
        errors = []
        num_threads = 8
        calls_per_thread = 50

        def hammer_successes():
            for _ in range(calls_per_thread):
                try:
                    cb.call(lambda: 42)
                except Exception as e:
                    errors.append(str(e))

        threads = [threading.Thread(target=hammer_successes) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert cb.state == CircuitState.CLOSED
        assert cb._failure_count == 0

    def test_thread_safety_mixed_success_and_failure(self):
        """Mixed concurrent calls maintain consistent state."""
        cb = CircuitBreaker("thread_test_mixed", failure_threshold=1000)
        errors = []

        def do_successes():
            for _ in range(100):
                try:
                    cb.call(lambda: "ok")
                except Exception as e:
                    errors.append(str(e))

        def do_failures():
            for _ in range(100):
                try:
                    cb.call(lambda: (_ for _ in ()).throw(ValueError))
                except (ValueError, CircuitBreakerOpenError):
                    pass
                except Exception as e:
                    errors.append(str(e))

        threads = []
        for _ in range(4):
            threads.append(threading.Thread(target=do_successes))
            threads.append(threading.Thread(target=do_failures))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # State should be valid (one of the three states)
        assert cb.state in (CircuitState.CLOSED, CircuitState.OPEN, CircuitState.HALF_OPEN)

    def test_thread_safety_state_transition(self):
        """Circuit opens correctly under concurrent pressure."""
        cb = CircuitBreaker("thread_test_transition", failure_threshold=10)

        def do_failures():
            for _ in range(20):
                with contextlib.suppress(ValueError, CircuitBreakerOpenError):
                    cb.call(lambda: (_ for _ in ()).throw(ValueError))

        threads = [threading.Thread(target=do_failures) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert cb.state == CircuitState.OPEN

    def test_get_circuit_breaker_thread_safety(self):
        """Concurrent registry access returns the same instance."""
        import pipeline.infrastructure.circuit_breaker as cb_mod

        # Clear registry for this test
        cb_mod._circuit_breakers.clear()

        results = []

        def get_breaker():
            breaker = get_circuit_breaker("shared_test", failure_threshold=5)
            results.append(id(breaker))

        threads = [threading.Thread(target=get_breaker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should get the same instance
        assert len(set(results)) == 1

        # Cleanup
        cb_mod._circuit_breakers.pop("shared_test", None)


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
        mgr.save_checkpoint(
            "op1",
            {
                "completed_items": 5,
                "total_items": 10,
                "last_processed": "item_5",
            },
        )

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

        with pytest.raises(ValueError), CheckpointContext(mgr, "op_fail") as ctx:
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


class TestValidateRange:
    """Tests for CLI _validate_range callback factory."""

    def test_returns_value_within_bounds(self):
        from pipeline.cli import _validate_range

        check = _validate_range(min_val=0, max_val=100)
        assert check(50) == 50

    def test_returns_min_boundary(self):
        from pipeline.cli import _validate_range

        check = _validate_range(min_val=0)
        assert check(0) == 0

    def test_returns_max_boundary(self):
        from pipeline.cli import _validate_range

        check = _validate_range(max_val=100)
        assert check(100) == 100

    def test_none_passthrough(self):
        from pipeline.cli import _validate_range

        check = _validate_range(min_val=1, max_val=100)
        assert check(None) is None

    def test_raises_below_min(self):
        import typer

        from pipeline.cli import _validate_range

        check = _validate_range(min_val=1)
        with pytest.raises(typer.BadParameter, match="Must be >= 1"):
            check(0)

    def test_raises_above_max(self):
        import typer

        from pipeline.cli import _validate_range

        check = _validate_range(max_val=100)
        with pytest.raises(typer.BadParameter, match="Must be <= 100"):
            check(101)

    def test_raises_negative(self):
        import typer

        from pipeline.cli import _validate_range

        check = _validate_range(min_val=0)
        with pytest.raises(typer.BadParameter, match="Must be >= 0"):
            check(-1)

    def test_min_only(self):
        from pipeline.cli import _validate_range

        check = _validate_range(min_val=5)
        assert check(5) == 5
        assert check(100) == 100

    def test_max_only(self):
        from pipeline.cli import _validate_range

        check = _validate_range(max_val=10)
        assert check(10) == 10
        assert check(-5) == -5
