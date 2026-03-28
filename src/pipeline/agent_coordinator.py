"""Multi-Agent System Architecture (Agent Directive V7 — Section 2).

Implements a coordinated lab of specialized agents that share a common
experiment registry and common validation rules. Each agent has a primary
responsibility and key deliverables as defined in the directive.

Agent roles:
- **Research Orchestrator**: Problem framing, priorities, roadmap.
- **Data Agent**: Point-in-time datasets, lineage, quality.
- **Feature Agent**: Feature generation, filtering, stress-testing.
- **Model Agent**: Model search, hyperparameter tuning.
- **Ensemble Agent**: Model combination, calibration.
- **Decision Agent**: Probabilities → actions, sizing, allocations.
- **Audit Agent**: Leakage, drift, overfitting, validation audit.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


class AgentRole(StrEnum):
    """Agent roles per Section 2."""

    RESEARCH_ORCHESTRATOR = "research_orchestrator"
    DATA_AGENT = "data_agent"
    FEATURE_AGENT = "feature_agent"
    MODEL_AGENT = "model_agent"
    ENSEMBLE_AGENT = "ensemble_agent"
    DECISION_AGENT = "decision_agent"
    AUDIT_AGENT = "audit_agent"


class TaskStatus(StrEnum):
    """Status of an agent task."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"


@dataclass
class AgentSpec:
    """Specification for a specialized agent."""

    role: AgentRole
    primary_responsibility: str
    deliverables: list[str]
    can_veto: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["role"] = self.role.value
        return d


@dataclass
class AgentTask:
    """A unit of work assigned to an agent."""

    task_id: str = field(default_factory=lambda: str(uuid4()))
    assigned_to: str = ""  # AgentRole value
    description: str = ""
    priority: int = 0  # lower = higher priority
    status: TaskStatus = TaskStatus.PENDING
    depends_on: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    completed_at: str = ""
    result: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AgentTask:
        d = dict(d)
        if "status" in d:
            d["status"] = TaskStatus(d["status"])
        return cls(**d)


@dataclass
class ResearchRoadmap:
    """Roadmap maintained by the Research Orchestrator."""

    problem_id: str = ""
    objective: str = ""
    constraints: list[str] = field(default_factory=list)
    phases: list[str] = field(default_factory=list)
    current_phase: str = ""
    failure_register: list[str] = field(default_factory=list)
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Default agent definitions per Section 2
# ---------------------------------------------------------------------------


def _default_agents() -> dict[AgentRole, AgentSpec]:
    return {
        AgentRole.RESEARCH_ORCHESTRATOR: AgentSpec(
            role=AgentRole.RESEARCH_ORCHESTRATOR,
            primary_responsibility=(
                "Sets problem framing, constraints, output schema, " "experiment priorities."
            ),
            deliverables=[
                "problem_summary",
                "roadmap",
                "failure_register",
            ],
        ),
        AgentRole.DATA_AGENT: AgentSpec(
            role=AgentRole.DATA_AGENT,
            primary_responsibility=(
                "Builds point-in-time datasets and lineage; searches " "for new historical data."
            ),
            deliverables=[
                "dataset_catalog",
                "data_lineage",
                "dataset_quality_report",
            ],
        ),
        AgentRole.FEATURE_AGENT: AgentSpec(
            role=AgentRole.FEATURE_AGENT,
            primary_responsibility=("Generates, filters, and stress-tests feature families."),
            deliverables=[
                "feature_catalog",
                "feature_stability_report",
            ],
        ),
        AgentRole.MODEL_AGENT: AgentSpec(
            role=AgentRole.MODEL_AGENT,
            primary_responsibility=(
                "Searches model families and hyperparameters under " "temporal validation."
            ),
            deliverables=[
                "model_search_report",
                "model_registry",
            ],
        ),
        AgentRole.ENSEMBLE_AGENT: AgentSpec(
            role=AgentRole.ENSEMBLE_AGENT,
            primary_responsibility=(
                "Combines models, calibrates outputs, and tests " "stacked systems."
            ),
            deliverables=[
                "ensemble_report",
                "calibration_report",
            ],
        ),
        AgentRole.DECISION_AGENT: AgentSpec(
            role=AgentRole.DECISION_AGENT,
            primary_responsibility=(
                "Turns probabilities/forecasts into choices, sizing, " "rankings, allocations."
            ),
            deliverables=[
                "decision_policy_report",
                "profitability_analysis",
            ],
        ),
        AgentRole.AUDIT_AGENT: AgentSpec(
            role=AgentRole.AUDIT_AGENT,
            primary_responsibility=(
                "Attempts to break everything: leakage, drift, " "overfitting, invalid backtests."
            ),
            deliverables=[
                "leakage_audit",
                "validation_audit",
                "reproducibility_report",
            ],
            can_veto=True,
        ),
    }


class AgentCoordinator:
    """Coordinates the multi-agent system per Section 2.

    The coordinator manages agent roles, task assignment, dependency
    tracking, and shared state. It integrates with the experiment
    registry for a common experiment ledger.

    Usage::

        coordinator = AgentCoordinator()

        # Set up the research roadmap
        coordinator.set_roadmap(
            problem_id="sp500_direction",
            objective="Predict next-day S&P 500 direction",
            phases=["data_discovery", "feature_engineering", "model_search",
                    "ensemble", "decision_optimization", "audit"],
        )

        # Assign tasks to agents
        data_task = coordinator.assign_task(
            role=AgentRole.DATA_AGENT,
            description="Build point-in-time dataset from FRED + prices",
            priority=1,
        )

        # Complete tasks
        coordinator.complete_task(
            data_task.task_id,
            result={"rows": 50000, "features": 14},
        )
    """

    def __init__(
        self,
        storage_path: str | Path = "data/agent_coordinator.json",
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        self._agents = _default_agents()
        self._tasks: dict[str, AgentTask] = {}
        self._roadmap: ResearchRoadmap | None = None
        self._cycle_count: int = 0
        self._load()

    # ---- Persistence -------------------------------------------------------

    def _load(self) -> None:
        if self.storage_path.exists():
            with open(self.storage_path) as f:
                data = json.load(f)
            for t in data.get("tasks", []):
                task = AgentTask.from_dict(t)
                self._tasks[task.task_id] = task
            if data.get("roadmap"):
                self._roadmap = ResearchRoadmap(**data["roadmap"])
            self._cycle_count = data.get("cycle_count", 0)

    def _save(self) -> None:
        with open(self.storage_path, "w") as f:
            json.dump(
                {
                    "agents": {k.value: v.to_dict() for k, v in self._agents.items()},
                    "tasks": [t.to_dict() for t in self._tasks.values()],
                    "roadmap": self._roadmap.to_dict() if self._roadmap else None,
                    "cycle_count": self._cycle_count,
                },
                f,
                indent=2,
                default=str,
            )

    # ---- Agent registry ----------------------------------------------------

    def get_agent(self, role: AgentRole) -> AgentSpec:
        """Get the specification for an agent role."""
        return self._agents[role]

    def list_agents(self) -> list[AgentSpec]:
        """List all registered agents."""
        return list(self._agents.values())

    # ---- Roadmap -----------------------------------------------------------

    def set_roadmap(
        self,
        problem_id: str,
        objective: str,
        phases: list[str] | None = None,
        constraints: list[str] | None = None,
    ) -> ResearchRoadmap:
        """Set or update the research roadmap (Orchestrator responsibility)."""
        if phases is None:
            phases = [
                "problem_definition",
                "data_discovery",
                "feature_engineering",
                "model_search",
                "ensemble_calibration",
                "decision_optimization",
                "audit",
            ]
        self._roadmap = ResearchRoadmap(
            problem_id=problem_id,
            objective=objective,
            phases=phases,
            constraints=constraints or [],
            current_phase=phases[0] if phases else "",
        )
        self._save()
        logger.info("Roadmap set for problem %s", problem_id)
        return self._roadmap

    def get_roadmap(self) -> ResearchRoadmap | None:
        return self._roadmap

    def advance_phase(self) -> str:
        """Advance to the next phase in the roadmap."""
        if not self._roadmap or not self._roadmap.phases:
            raise ValueError("No roadmap configured")
        phases = self._roadmap.phases
        current_idx = (
            phases.index(self._roadmap.current_phase)
            if self._roadmap.current_phase in phases
            else -1
        )
        if current_idx >= len(phases) - 1:
            raise ValueError("Already at the final phase")
        self._roadmap.current_phase = phases[current_idx + 1]
        self._roadmap.updated_at = datetime.now(UTC).isoformat()
        self._save()
        logger.info("Advanced to phase: %s", self._roadmap.current_phase)
        return self._roadmap.current_phase

    def register_failure(self, description: str) -> None:
        """Add a failure to the failure register."""
        if self._roadmap is None:
            raise ValueError("No roadmap configured")
        self._roadmap.failure_register.append(description)
        self._roadmap.updated_at = datetime.now(UTC).isoformat()
        self._save()
        logger.warning("Failure registered: %s", description)

    # ---- Task management ---------------------------------------------------

    def assign_task(
        self,
        role: AgentRole,
        description: str,
        priority: int = 5,
        depends_on: list[str] | None = None,
    ) -> AgentTask:
        """Assign a task to a specific agent role."""
        task = AgentTask(
            assigned_to=role.value,
            description=description,
            priority=priority,
            depends_on=depends_on or [],
        )
        self._tasks[task.task_id] = task
        self._save()
        logger.info("Task %s assigned to %s: %s", task.task_id, role.value, description)
        return task

    def start_task(self, task_id: str) -> AgentTask:
        """Mark a task as in progress."""
        task = self._tasks[task_id]
        # Check dependencies
        for dep_id in task.depends_on:
            dep = self._tasks.get(dep_id)
            if dep and dep.status != TaskStatus.COMPLETED:
                raise ValueError(f"Dependency {dep_id} not completed (status: {dep.status.value})")
        task.status = TaskStatus.IN_PROGRESS
        self._save()
        return task

    def complete_task(
        self,
        task_id: str,
        result: dict[str, Any] | None = None,
        notes: str = "",
    ) -> AgentTask:
        """Mark a task as completed with results."""
        task = self._tasks[task_id]
        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.now(UTC).isoformat()
        task.result = result or {}
        task.notes = notes
        self._save()
        logger.info("Task %s completed", task_id)
        return task

    def fail_task(self, task_id: str, reason: str = "") -> AgentTask:
        """Mark a task as failed."""
        task = self._tasks[task_id]
        task.status = TaskStatus.FAILED
        task.notes = reason
        self._save()
        logger.warning("Task %s failed: %s", task_id, reason)
        return task

    def block_task(self, task_id: str, reason: str = "") -> AgentTask:
        """Mark a task as blocked."""
        task = self._tasks[task_id]
        task.status = TaskStatus.BLOCKED
        task.notes = reason
        self._save()
        return task

    def get_task(self, task_id: str) -> AgentTask | None:
        return self._tasks.get(task_id)

    def list_tasks(
        self,
        role: AgentRole | None = None,
        status: TaskStatus | None = None,
    ) -> list[AgentTask]:
        """List tasks with optional filters."""
        results = list(self._tasks.values())
        if role is not None:
            results = [t for t in results if t.assigned_to == role.value]
        if status is not None:
            results = [t for t in results if t.status == status]
        return sorted(results, key=lambda t: t.priority)

    def get_ready_tasks(self) -> list[AgentTask]:
        """Get tasks whose dependencies are all completed."""
        ready = []
        for task in self._tasks.values():
            if task.status != TaskStatus.PENDING:
                continue
            deps_met = all(
                self._tasks.get(dep_id) is not None
                and self._tasks[dep_id].status == TaskStatus.COMPLETED
                for dep_id in task.depends_on
            )
            if deps_met:
                ready.append(task)
        return sorted(ready, key=lambda t: t.priority)

    # ---- Research cycle ----------------------------------------------------

    def start_research_cycle(self) -> int:
        """Begin a new research cycle (Section 14)."""
        self._cycle_count += 1
        self._save()
        logger.info("Starting research cycle %d", self._cycle_count)
        return self._cycle_count

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    # ---- Hypothesis generation (Section 14) ----------------------------------

    def generate_hypotheses(
        self,
        knowledge_findings: list[dict[str, Any]] | None = None,
        current_metrics: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        """Auto-generate experiment hypotheses from knowledge and metrics.

        Per Section 14, the research loop should propose new data sources,
        features, architectures, or policy changes.

        Parameters
        ----------
        knowledge_findings : list[dict] | None
            Past findings from KnowledgeStore.export().
        current_metrics : dict[str, float] | None
            Current system metrics for gap identification.

        Returns
        -------
        list[dict]
            Hypothesis objects with type, description, rationale, and
            suggested agent assignment.
        """
        hypotheses: list[dict[str, Any]] = []
        findings = knowledge_findings or []

        # Generate hypotheses from failed approaches (try alternatives)
        failed = [f for f in findings if not f.get("works", True)]
        for f in failed[:3]:
            hypotheses.append(
                {
                    "type": "alternative_approach",
                    "description": f"Revisit failed approach: {f.get('finding', '')}",
                    "rationale": "Previous attempt failed; try with modified parameters or data",
                    "assigned_to": AgentRole.MODEL_AGENT.value,
                    "priority": 3,
                }
            )

        # Suggest feature expansion if metrics are stagnant
        if current_metrics:
            sharpe = current_metrics.get("sharpe", 0)
            if sharpe < 1.0:
                hypotheses.append(
                    {
                        "type": "feature_expansion",
                        "description": "Expand feature set with interaction and seasonal features",
                        "rationale": f"Current Sharpe ({sharpe:.2f}) suggests room for improvement",
                        "assigned_to": AgentRole.FEATURE_AGENT.value,
                        "priority": 1,
                    }
                )

            hit_rate = current_metrics.get("hit_rate", 0)
            if hit_rate < 0.55:
                hypotheses.append(
                    {
                        "type": "model_architecture",
                        "description": "Try ensemble stacking with diverse base models",
                        "rationale": f"Hit rate ({hit_rate:.2f}) may benefit from model diversity",
                        "assigned_to": AgentRole.ENSEMBLE_AGENT.value,
                        "priority": 2,
                    }
                )

        # Standard hypotheses for each cycle
        hypotheses.append(
            {
                "type": "data_source",
                "description": "Search for new data sources not yet integrated",
                "rationale": "Broader data universe may contain untapped signal",
                "assigned_to": AgentRole.DATA_AGENT.value,
                "priority": 4,
            }
        )
        hypotheses.append(
            {
                "type": "calibration",
                "description": "Re-calibrate probability outputs with latest data",
                "rationale": "Calibration can drift over time; periodic refresh needed",
                "assigned_to": AgentRole.ENSEMBLE_AGENT.value,
                "priority": 5,
            }
        )

        return hypotheses

    def schedule_research_cycle(self) -> list[AgentTask]:
        """Schedule the standard research cycle as tasks (Section 14).

        Creates a full cycle: data refresh → feature search → model search
        → ensemble → decision optimization → audit → promote/reject.
        """
        if not self._roadmap:
            raise ValueError("No roadmap configured; call set_roadmap() first")

        cycle_num = self._cycle_count + 1
        prefix = f"cycle_{cycle_num}"

        phases: list[tuple[AgentRole, str, list[str]]] = [
            (AgentRole.DATA_AGENT, "Refresh datasets and validate lineage", []),
            (AgentRole.FEATURE_AGENT, "Search and filter feature families", []),
            (AgentRole.MODEL_AGENT, "Search model architectures and hyperparameters", []),
            (AgentRole.ENSEMBLE_AGENT, "Optimize ensemble and calibrate outputs", []),
            (AgentRole.DECISION_AGENT, "Optimize decision policy and thresholds", []),
            (AgentRole.AUDIT_AGENT, "Audit for leakage, drift, and overfitting", []),
        ]

        tasks: list[AgentTask] = []
        prev_id: str | None = None
        for i, (role, desc, _) in enumerate(phases):
            depends = [prev_id] if prev_id else []
            task = self.assign_task(
                role=role,
                description=f"[{prefix}] {desc}",
                priority=i + 1,
                depends_on=depends,
            )
            tasks.append(task)
            prev_id = task.task_id

        return tasks

    def review_dissents(
        self,
        dissent_registry: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Review open dissents before starting a new cycle (Section 22.4).

        Returns dissents that should be revisited.
        """
        if not dissent_registry:
            return []
        open_dissents = [d for d in dissent_registry if d.get("status", "open") == "open"]
        return open_dissents

    # ---- Export -------------------------------------------------------------

    def export_state(self) -> dict[str, Any]:
        """Export the full coordinator state for reporting."""
        return {
            "agents": {k.value: v.to_dict() for k, v in self._agents.items()},
            "roadmap": self._roadmap.to_dict() if self._roadmap else None,
            "tasks": {
                "total": len(self._tasks),
                "pending": len(self.list_tasks(status=TaskStatus.PENDING)),
                "in_progress": len(self.list_tasks(status=TaskStatus.IN_PROGRESS)),
                "completed": len(self.list_tasks(status=TaskStatus.COMPLETED)),
                "failed": len(self.list_tasks(status=TaskStatus.FAILED)),
                "blocked": len(self.list_tasks(status=TaskStatus.BLOCKED)),
            },
            "cycle_count": self._cycle_count,
        }
