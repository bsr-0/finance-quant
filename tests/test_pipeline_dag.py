"""Tests for pipeline DAG and freshness SLA (V7 Section 19)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from pipeline.infrastructure.freshness_sla import (
    DataCategory,
    FreshnessMonitor,
    FreshnessSLA,
    StalenessAction,
)
from pipeline.infrastructure.pipeline_dag import (
    PipelineDAG,
    PipelineTask,
    TaskState,
)

# ---------------------------------------------------------------------------
# Pipeline DAG
# ---------------------------------------------------------------------------


class TestPipelineDAG:
    def test_topological_sort_simple(self):
        dag = PipelineDAG()
        dag.add_task(PipelineTask("a", lambda: 1))
        dag.add_task(PipelineTask("b", lambda: 2, upstream=["a"]))
        dag.add_task(PipelineTask("c", lambda: 3, upstream=["b"]))
        order = dag.topological_sort()
        assert order.index("a") < order.index("b") < order.index("c")

    def test_topological_sort_diamond(self):
        dag = PipelineDAG()
        dag.add_task(PipelineTask("a", lambda: None))
        dag.add_task(PipelineTask("b", lambda: None, upstream=["a"]))
        dag.add_task(PipelineTask("c", lambda: None, upstream=["a"]))
        dag.add_task(PipelineTask("d", lambda: None, upstream=["b", "c"]))
        order = dag.topological_sort()
        assert order[0] == "a"
        assert order[-1] == "d"

    def test_cycle_detection(self):
        dag = PipelineDAG()
        dag.add_task(PipelineTask("a", lambda: None, upstream=["b"]))
        dag.add_task(PipelineTask("b", lambda: None, upstream=["a"]))
        with pytest.raises(ValueError, match="Cycle"):
            dag.topological_sort()

    def test_unknown_dependency(self):
        dag = PipelineDAG()
        dag.add_task(PipelineTask("a", lambda: None, upstream=["missing"]))
        with pytest.raises(ValueError, match="unknown task"):
            dag.topological_sort()

    def test_duplicate_task_name(self):
        dag = PipelineDAG()
        dag.add_task(PipelineTask("a", lambda: None))
        with pytest.raises(ValueError, match="Duplicate"):
            dag.add_task(PipelineTask("a", lambda: None))

    def test_execute_success(self):
        call_order = []
        dag = PipelineDAG()
        dag.add_task(PipelineTask("step1", lambda: call_order.append("step1")))
        dag.add_task(PipelineTask("step2", lambda: call_order.append("step2"), upstream=["step1"]))
        results = dag.execute()
        assert all(r.state == TaskState.COMPLETED for r in results)
        assert call_order == ["step1", "step2"]

    def test_execute_failure_blocks_downstream(self):
        def fail():
            raise RuntimeError("boom")

        dag = PipelineDAG()
        dag.add_task(PipelineTask("bad", fail, max_retries=0))
        dag.add_task(PipelineTask("after", lambda: None, upstream=["bad"]))
        results = dag.execute()
        assert results[0].state == TaskState.FAILED
        assert results[1].state == TaskState.BLOCKED

    def test_execute_retry(self):
        counter = {"n": 0}

        def flaky():
            counter["n"] += 1
            if counter["n"] < 2:
                raise RuntimeError("transient")
            return "ok"

        dag = PipelineDAG()
        dag.add_task(PipelineTask("flaky", flaky, max_retries=2, retry_backoff=0.01))
        results = dag.execute()
        assert results[0].state == TaskState.COMPLETED
        assert results[0].retries_used == 1

    def test_export_spec(self):
        dag = PipelineDAG()
        dag.add_task(PipelineTask("a", lambda: None))
        dag.add_task(PipelineTask("b", lambda: None, upstream=["a"]))
        spec = dag.export_spec()
        assert spec["report_type"] == "pipeline_dag_spec"
        assert len(spec["tasks"]) == 2
        assert spec["execution_order"] == ["a", "b"]

    def test_validate_empty(self):
        dag = PipelineDAG()
        assert dag.validate() == []

    def test_validate_with_issues(self):
        dag = PipelineDAG()
        dag.add_task(PipelineTask("a", lambda: None, upstream=["missing"]))
        issues = dag.validate()
        assert len(issues) > 0


# ---------------------------------------------------------------------------
# Freshness SLA
# ---------------------------------------------------------------------------


class TestFreshnessMonitor:
    def test_fresh_source_no_violation(self):
        monitor = FreshnessMonitor()
        now = datetime.now(UTC)
        result = monitor.check("prices_daily", now - timedelta(hours=1), now=now)
        assert result is None

    def test_stale_source_violation(self):
        monitor = FreshnessMonitor()
        now = datetime.now(UTC)
        result = monitor.check("prices_daily", now - timedelta(hours=48), now=now)
        assert result is not None
        assert result.source_name == "prices_daily"
        assert result.action == "serve_stale_with_flag"

    def test_realtime_source_violation(self):
        monitor = FreshnessMonitor()
        now = datetime.now(UTC)
        result = monitor.check("polymarket_odds", now - timedelta(minutes=30), now=now)
        assert result is not None
        assert result.action == "suspend_decisions"

    def test_unknown_source_returns_none(self):
        monitor = FreshnessMonitor()
        now = datetime.now(UTC)
        result = monitor.check("unknown_source", now, now=now)
        assert result is None

    def test_check_all(self):
        monitor = FreshnessMonitor()
        now = datetime.now(UTC)
        last_updated = {
            "prices_daily": now - timedelta(hours=1),  # fresh
            "polymarket_odds": now - timedelta(hours=2),  # stale
        }
        violations = monitor.check_all(last_updated)
        assert len(violations) == 1
        assert violations[0].source_name == "polymarket_odds"

    def test_export_sla_registry(self):
        monitor = FreshnessMonitor()
        registry = monitor.export_sla_registry()
        assert registry["report_type"] == "freshness_sla_registry"
        assert registry["total_sources"] >= 10

    def test_custom_sla(self):
        sla = FreshnessSLA(
            source_name="custom",
            category=DataCategory.DAILY_BATCH,
            max_staleness_seconds=3600,
            staleness_action=StalenessAction.ALERT_ONLY,
        )
        monitor = FreshnessMonitor(slas=[sla])
        now = datetime.now(UTC)
        result = monitor.check("custom", now - timedelta(hours=2), now=now)
        assert result is not None
        assert result.action == "alert_only"

    def test_clear_violations(self):
        monitor = FreshnessMonitor()
        now = datetime.now(UTC)
        monitor.check("polymarket_odds", now - timedelta(hours=2), now=now)
        assert len(monitor.get_violations()) == 1
        monitor.clear_violations()
        assert len(monitor.get_violations()) == 0
