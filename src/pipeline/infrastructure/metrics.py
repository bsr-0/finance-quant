"""Metrics collection and monitoring for the pipeline."""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class Metric:
    """Single metric data point."""

    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    tags: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "type": self.metric_type.value,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
        }


class MetricsCollector:
    """Collect and store pipeline metrics."""

    def __init__(self, namespace: str = "mdw"):
        self.namespace = namespace
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._timers: dict[str, list[float]] = defaultdict(list)
        self._metrics_log: list[Metric] = []

    def counter(self, name: str, value: float = 1, tags: dict[str, str] | None = None):
        """Increment counter."""
        full_name = f"{self.namespace}.{name}"
        self._counters[full_name] += value
        self._metrics_log.append(
            Metric(
                name=full_name,
                value=self._counters[full_name],
                metric_type=MetricType.COUNTER,
                tags=tags or {},
            )
        )

    def gauge(self, name: str, value: float, tags: dict[str, str] | None = None):
        """Set gauge value."""
        full_name = f"{self.namespace}.{name}"
        self._gauges[full_name] = value
        self._metrics_log.append(
            Metric(name=full_name, value=value, metric_type=MetricType.GAUGE, tags=tags or {})
        )

    def histogram(self, name: str, value: float, tags: dict[str, str] | None = None):
        """Record histogram value."""
        full_name = f"{self.namespace}.{name}"
        self._histograms[full_name].append(value)
        self._metrics_log.append(
            Metric(name=full_name, value=value, metric_type=MetricType.HISTOGRAM, tags=tags or {})
        )

    def timer(self, name: str, value_ms: float, tags: dict[str, str] | None = None):
        """Record timer value."""
        full_name = f"{self.namespace}.{name}"
        self._timers[full_name].append(value_ms)
        self._metrics_log.append(
            Metric(name=full_name, value=value_ms, metric_type=MetricType.TIMER, tags=tags or {})
        )

    @contextmanager
    def timer_context(self, name: str, tags: dict[str, str] | None = None):
        """Context manager for timing operations."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.timer(name, elapsed_ms, tags)

    def get_summary(self) -> dict:
        """Get metrics summary."""
        summary = {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": {},
            "timers": {},
        }

        for name, values in self._histograms.items():
            if values:
                summary["histograms"][name] = {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                }

        for name, values in self._timers.items():
            if values:
                summary["timers"][name] = {
                    "count": len(values),
                    "min_ms": min(values),
                    "max_ms": max(values),
                    "avg_ms": sum(values) / len(values),
                    "p95_ms": sorted(values)[int(len(values) * 0.95)]
                    if len(values) > 20
                    else max(values),
                }

        return summary

    def export_to_file(self, path: Path) -> None:
        """Export metrics to JSON file."""
        with open(path, "w") as f:
            json.dump(self.get_summary(), f, indent=2)
        logger.info(f"Metrics exported to {path}")

    def reset(self):
        """Reset all metrics."""
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()
        self._timers.clear()
        self._metrics_log.clear()


# Global metrics collector
_metrics: MetricsCollector | None = None


def get_metrics(namespace: str = "mdw") -> MetricsCollector:
    """Get or create global metrics collector."""
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector(namespace)
    return _metrics


class PipelineMetrics:
    """Convenience class for pipeline-specific metrics."""

    def __init__(self, pipeline_name: str):
        self.metrics = get_metrics()
        self.pipeline_name = pipeline_name
        self.base_tags = {"pipeline": pipeline_name}

    def record_extracted(self, source: str, count: int):
        """Record extracted records."""
        self.metrics.counter("records_extracted", count, {**self.base_tags, "source": source})

    def record_loaded(self, table: str, count: int):
        """Record loaded records."""
        self.metrics.counter("records_loaded", count, {**self.base_tags, "table": table})

    def record_transformed(self, table: str, count: int):
        """Record transformed records."""
        self.metrics.counter("records_transformed", count, {**self.base_tags, "table": table})

    def record_error(self, error_type: str):
        """Record error."""
        self.metrics.counter("errors", 1, {**self.base_tags, "error_type": error_type})

    def record_dq_failure(self, test_name: str):
        """Record DQ test failure."""
        self.metrics.counter("dq_failures", 1, {**self.base_tags, "test": test_name})

    def time_operation(self, operation: str):
        """Time an operation."""
        return self.metrics.timer_context(
            "operation_duration_ms", {**self.base_tags, "operation": operation}
        )


@contextmanager
def track_operation(operation_name: str, pipeline: str = "default"):
    """Context manager to track operation metrics."""
    metrics = PipelineMetrics(pipeline)

    try:
        with metrics.time_operation(operation_name):
            yield metrics
    except Exception as e:
        metrics.record_error(type(e).__name__)
        raise


def log_pipeline_summary(pipeline_name: str) -> None:
    """Log summary of pipeline metrics."""
    metrics = get_metrics()
    summary = metrics.get_summary()

    logger.info(f"=== Pipeline Summary: {pipeline_name} ===")

    if summary["counters"]:
        logger.info("Counters:")
        for name, value in summary["counters"].items():
            logger.info(f"  {name}: {value}")

    if summary["timers"]:
        logger.info("Timers:")
        for name, stats in summary["timers"].items():
            logger.info(f"  {name}: avg={stats['avg_ms']:.2f}ms, p95={stats['p95_ms']:.2f}ms")
