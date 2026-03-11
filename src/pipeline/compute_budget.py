"""Compute Budget and Resource Prioritization (Agent Directive V7 — Section 20).

Defines a budget-aware search protocol that maximizes discovery per unit
of compute. Tracks budget consumption, enforces phase allocations, and
implements search termination rules.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PhaseAllocation:
    """Budget allocation for a research phase."""

    phase: str
    pct: float
    budget_seconds: float = 0.0
    consumed_seconds: float = 0.0

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.budget_seconds - self.consumed_seconds)

    @property
    def utilization_pct(self) -> float:
        if self.budget_seconds == 0:
            return 0.0
        return self.consumed_seconds / self.budget_seconds


DEFAULT_PHASE_ALLOCATIONS = {
    "data": 0.10,
    "features": 0.15,
    "model_search": 0.35,
    "ensemble_calibration": 0.15,
    "decision_optimization": 0.10,
    "audit": 0.10,
    "reserve": 0.05,
}


@dataclass
class ExperimentCost:
    """Cost record for a single experiment."""

    experiment_id: str
    phase: str
    start_time: float
    end_time: float = 0.0
    cost_seconds: float = 0.0
    primary_metric_value: float | None = None
    improvement_over_baseline: float | None = None

    @property
    def elapsed(self) -> float:
        if self.end_time > 0:
            return self.end_time - self.start_time
        return time.time() - self.start_time


@dataclass
class BudgetReport:
    """Budget tracking report (Section 20.3)."""

    total_budget_seconds: float
    total_consumed_seconds: float
    phase_breakdown: list[dict[str, Any]]
    cost_per_improvement: float | None
    projected_remaining_cost: float | None
    should_terminate: bool
    termination_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ComputeBudget:
    """Budget-aware compute tracker per Section 20.

    Usage::

        budget = ComputeBudget(total_budget_hours=72)
        with budget.track_experiment("exp_1", phase="model_search") as tracker:
            # ... run experiment ...
            tracker.primary_metric_value = 1.42
        report = budget.get_report()
        if report.should_terminate:
            print("Search should stop:", report.termination_reason)
    """

    def __init__(
        self,
        total_budget_hours: float = 72.0,
        phase_allocations: dict[str, float] | None = None,
        max_single_experiment_pct: float = 0.05,
        termination_n: int = 10,
        termination_min_improvement_pct: float = 0.001,
        storage_path: str | Path = "data/compute_budget.json",
    ):
        self.total_budget_seconds = total_budget_hours * 3600
        self.max_single_experiment_seconds = (
            self.total_budget_seconds * max_single_experiment_pct
        )
        self.termination_n = termination_n
        self.termination_min_improvement_pct = termination_min_improvement_pct
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        allocs = phase_allocations or DEFAULT_PHASE_ALLOCATIONS
        self.phases: dict[str, PhaseAllocation] = {}
        for phase, pct in allocs.items():
            self.phases[phase] = PhaseAllocation(
                phase=phase,
                pct=pct,
                budget_seconds=self.total_budget_seconds * pct,
            )

        self._costs: list[ExperimentCost] = []
        self._active: dict[str, ExperimentCost] = {}
        self._load()

    def _load(self) -> None:
        if self.storage_path.exists():
            with open(self.storage_path) as f:
                data = json.load(f)
            for entry in data.get("costs", []):
                cost = ExperimentCost(**entry)
                self._costs.append(cost)
                phase = self.phases.get(cost.phase)
                if phase:
                    phase.consumed_seconds += cost.cost_seconds

    def _save(self) -> None:
        with open(self.storage_path, "w") as f:
            json.dump(
                {
                    "total_budget_seconds": self.total_budget_seconds,
                    "costs": [asdict(c) for c in self._costs],
                },
                f,
                indent=2,
            )

    @property
    def total_consumed(self) -> float:
        return sum(p.consumed_seconds for p in self.phases.values())

    @property
    def total_remaining(self) -> float:
        return max(0.0, self.total_budget_seconds - self.total_consumed)

    def track_experiment(
        self, experiment_id: str, phase: str
    ) -> _ExperimentTracker:
        """Context manager to track compute cost of an experiment."""
        return _ExperimentTracker(self, experiment_id, phase)

    def _start_tracking(self, experiment_id: str, phase: str) -> ExperimentCost:
        cost = ExperimentCost(
            experiment_id=experiment_id,
            phase=phase,
            start_time=time.time(),
        )
        self._active[experiment_id] = cost
        return cost

    def _stop_tracking(self, experiment_id: str) -> ExperimentCost:
        cost = self._active.pop(experiment_id)
        cost.end_time = time.time()
        cost.cost_seconds = cost.end_time - cost.start_time
        self._costs.append(cost)

        phase = self.phases.get(cost.phase)
        if phase:
            phase.consumed_seconds += cost.cost_seconds

        if cost.cost_seconds > self.max_single_experiment_seconds:
            logger.warning(
                "Experiment %s exceeded single-experiment budget cap: %.1fs > %.1fs",
                experiment_id,
                cost.cost_seconds,
                self.max_single_experiment_seconds,
            )

        self._save()
        return cost

    def check_budget_available(self, phase: str) -> bool:
        """Check if budget remains for a phase."""
        p = self.phases.get(phase)
        if not p:
            return True
        return p.remaining_seconds > 0

    def check_search_termination(self) -> tuple[bool, str]:
        """Check if search should terminate (Section 20.1).

        Stop if the last N experiments have not improved the best
        candidate by more than 0.1% on the primary metric.
        """
        completed = [c for c in self._costs if c.primary_metric_value is not None]
        if len(completed) < self.termination_n:
            return False, ""

        recent = completed[-self.termination_n :]
        older = completed[: -self.termination_n]

        if not older:
            return False, ""

        best_older = max(c.primary_metric_value or 0.0 for c in older)
        best_recent = max(c.primary_metric_value or 0.0 for c in recent)

        if best_older == 0:
            return False, ""

        improvement = (best_recent - best_older) / abs(best_older)
        if improvement < self.termination_min_improvement_pct:
            reason = (
                f"Last {self.termination_n} experiments improved by only "
                f"{improvement:.4%} (threshold: {self.termination_min_improvement_pct:.4%})"
            )
            return True, reason
        return False, ""

    def get_report(self) -> BudgetReport:
        """Generate budget tracking report (Section 20.3)."""
        should_terminate, reason = self.check_search_termination()

        # Cost per unit improvement
        completed = [c for c in self._costs if c.primary_metric_value is not None]
        cost_per_improvement = None
        if len(completed) >= 2:
            sorted_by_time = sorted(completed, key=lambda c: c.start_time)
            total_cost = sum(c.cost_seconds for c in sorted_by_time)
            improvement = (sorted_by_time[-1].primary_metric_value or 0.0) - (
                sorted_by_time[0].primary_metric_value or 0.0
            )
            if improvement > 0:
                cost_per_improvement = total_cost / improvement

        return BudgetReport(
            total_budget_seconds=self.total_budget_seconds,
            total_consumed_seconds=self.total_consumed,
            phase_breakdown=[
                {
                    "phase": p.phase,
                    "budget_seconds": p.budget_seconds,
                    "consumed_seconds": p.consumed_seconds,
                    "remaining_seconds": p.remaining_seconds,
                    "utilization_pct": round(p.utilization_pct, 4),
                }
                for p in self.phases.values()
            ],
            cost_per_improvement=cost_per_improvement,
            projected_remaining_cost=None,
            should_terminate=should_terminate,
            termination_reason=reason,
        )


    def generate_pareto_frontier(self) -> dict[str, Any]:
        """<pareto_frontier_analysis> — Section 20.4 required output.

        Shows the trade-off between compute invested and performance achieved.
        """
        completed = [c for c in self._costs if c.primary_metric_value is not None]
        if not completed:
            return {
                "report_type": "pareto_frontier_analysis",
                "frontier_points": [],
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }

        sorted_by_time = sorted(completed, key=lambda c: c.start_time)
        cumulative_cost = 0.0
        best_so_far = float("-inf")
        frontier_points = []

        for cost in sorted_by_time:
            cumulative_cost += cost.cost_seconds
            val = cost.primary_metric_value or 0.0
            if val > best_so_far:
                best_so_far = val
                frontier_points.append({
                    "experiment_id": cost.experiment_id,
                    "cumulative_cost_seconds": round(cumulative_cost, 2),
                    "best_metric": round(best_so_far, 6),
                })

        return {
            "report_type": "pareto_frontier_analysis",
            "frontier_points": frontier_points,
            "total_experiments": len(completed),
            "total_cost_seconds": round(cumulative_cost, 2),
            "final_best_metric": round(best_so_far, 6),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def export_search_termination_justification(self) -> dict[str, Any]:
        """<search_termination_justification> — Section 20.4 required output."""
        should_terminate, reason = self.check_search_termination()
        report = self.get_report()
        return {
            "report_type": "search_termination_justification",
            "should_terminate": should_terminate,
            "reason": reason,
            "total_budget_seconds": self.total_budget_seconds,
            "total_consumed_seconds": report.total_consumed_seconds,
            "budget_utilization_pct": round(
                report.total_consumed_seconds / max(self.total_budget_seconds, 1), 4
            ),
            "cost_per_improvement": report.cost_per_improvement,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }


class _ExperimentTracker:
    """Context manager for tracking experiment compute cost."""

    def __init__(self, budget: ComputeBudget, experiment_id: str, phase: str):
        self._budget = budget
        self._experiment_id = experiment_id
        self._phase = phase
        self._cost: ExperimentCost | None = None
        self.primary_metric_value: float | None = None

    def __enter__(self) -> _ExperimentTracker:
        self._cost = self._budget._start_tracking(self._experiment_id, self._phase)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        if self._cost is not None:
            self._cost.primary_metric_value = self.primary_metric_value
            self._budget._stop_tracking(self._experiment_id)
        return False
