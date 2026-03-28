"""Lightweight DAG executor for pipeline orchestration.

Agent Directive V7 Section 19.1 requires DAG-based scheduling with
explicit dependency declarations, idempotent tasks, and deterministic
outputs.  This module provides a zero-external-dependency DAG runner
that satisfies these requirements without Airflow/Prefect/Dagster.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class TaskState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class TaskResult:
    """Outcome of a single task execution."""

    task_name: str
    state: TaskState
    duration_seconds: float = 0.0
    error: str = ""
    retries_used: int = 0
    output: Any = None
    completed_at: str = ""


@dataclass
class PipelineTask:
    """A single task in the pipeline DAG.

    Parameters
    ----------
    name : str
        Unique task identifier.
    fn : Callable
        The callable to execute.  Must be idempotent if ``idempotent``
        is True.
    upstream : list[str]
        Names of tasks that must complete before this one runs.
    max_retries : int
        Number of retry attempts on failure.
    retry_backoff : float
        Initial backoff in seconds (doubles each retry).
    idempotent : bool
        Whether the task is safe to re-run.
    """

    name: str
    fn: Callable[..., Any]
    upstream: list[str] = field(default_factory=list)
    max_retries: int = 3
    retry_backoff: float = 1.0
    idempotent: bool = True


class PipelineDAG:
    """DAG-based pipeline executor with topological ordering.

    Usage::

        dag = PipelineDAG()
        dag.add_task(PipelineTask("extract", extract_fn))
        dag.add_task(PipelineTask("transform", transform_fn, upstream=["extract"]))
        dag.add_task(PipelineTask("load", load_fn, upstream=["transform"]))
        results = dag.execute()
    """

    def __init__(self) -> None:
        self._tasks: dict[str, PipelineTask] = {}

    def add_task(self, task: PipelineTask) -> None:
        if task.name in self._tasks:
            raise ValueError(f"Duplicate task name: {task.name}")
        self._tasks[task.name] = task

    @property
    def task_names(self) -> list[str]:
        return list(self._tasks.keys())

    # ------------------------------------------------------------------
    # Topological sort (Kahn's algorithm)
    # ------------------------------------------------------------------

    def topological_sort(self) -> list[str]:
        """Return task names in dependency-safe execution order.

        Raises ``ValueError`` on cycles.
        """
        in_degree: dict[str, int] = dict.fromkeys(self._tasks, 0)
        for task in self._tasks.values():
            for dep in task.upstream:
                if dep not in self._tasks:
                    raise ValueError(f"Task '{task.name}' depends on unknown task '{dep}'")
                in_degree[task.name] += 1

        queue = [n for n, d in in_degree.items() if d == 0]
        order: list[str] = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for task in self._tasks.values():
                if node in task.upstream:
                    in_degree[task.name] -= 1
                    if in_degree[task.name] == 0:
                        queue.append(task.name)

        if len(order) != len(self._tasks):
            raise ValueError("Cycle detected in pipeline DAG")
        return order

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, **kwargs: Any) -> list[TaskResult]:
        """Execute all tasks in topological order.

        Returns a list of ``TaskResult`` objects.  If a task fails after
        all retries, downstream tasks are marked BLOCKED.
        """
        order = self.topological_sort()
        results: dict[str, TaskResult] = {}
        failed_ancestors: set[str] = set()

        for name in order:
            task = self._tasks[name]

            # Check if any upstream task failed
            if any(dep in failed_ancestors for dep in task.upstream):
                result = TaskResult(
                    task_name=name,
                    state=TaskState.BLOCKED,
                    error="Upstream task failed",
                )
                results[name] = result
                failed_ancestors.add(name)
                logger.warning("Task '%s' BLOCKED (upstream failure)", name)
                continue

            result = self._run_with_retries(task, **kwargs)
            results[name] = result

            if result.state == TaskState.FAILED:
                failed_ancestors.add(name)

        return [results[n] for n in order]

    def _run_with_retries(self, task: PipelineTask, **kwargs: Any) -> TaskResult:
        """Run a task with retry logic and exponential backoff."""
        backoff = task.retry_backoff
        last_error = ""

        for attempt in range(task.max_retries + 1):
            start = time.monotonic()
            try:
                output = task.fn(**kwargs)
                elapsed = time.monotonic() - start
                logger.info(
                    "Task '%s' completed in %.2fs (attempt %d)",
                    task.name,
                    elapsed,
                    attempt + 1,
                )
                return TaskResult(
                    task_name=task.name,
                    state=TaskState.COMPLETED,
                    duration_seconds=elapsed,
                    retries_used=attempt,
                    output=output,
                    completed_at=datetime.now(UTC).isoformat(),
                )
            except Exception as exc:
                elapsed = time.monotonic() - start
                last_error = str(exc)
                logger.warning(
                    "Task '%s' failed (attempt %d/%d): %s",
                    task.name,
                    attempt + 1,
                    task.max_retries + 1,
                    last_error,
                )
                if attempt < task.max_retries:
                    time.sleep(backoff)
                    backoff *= 2

        return TaskResult(
            task_name=task.name,
            state=TaskState.FAILED,
            duration_seconds=time.monotonic() - start,
            retries_used=task.max_retries,
            error=last_error,
        )

    def validate(self) -> list[str]:
        """Return a list of validation issues (empty if valid)."""
        issues: list[str] = []
        try:
            self.topological_sort()
        except ValueError as exc:
            issues.append(str(exc))
        for task in self._tasks.values():
            for dep in task.upstream:
                if dep not in self._tasks:
                    issues.append(f"Task '{task.name}' depends on unknown task '{dep}'")
        return issues

    def export_spec(self) -> dict[str, Any]:
        """<pipeline_dag_spec> — Section 19.4 required output."""
        return {
            "report_type": "pipeline_dag_spec",
            "tasks": [
                {
                    "name": t.name,
                    "upstream": t.upstream,
                    "max_retries": t.max_retries,
                    "idempotent": t.idempotent,
                }
                for t in self._tasks.values()
            ],
            "execution_order": self.topological_sort(),
            "generated_at": datetime.now(UTC).isoformat(),
        }
