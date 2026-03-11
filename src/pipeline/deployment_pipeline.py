"""Deployment Pipeline Configuration (Agent Directive V7 — Section 18.1).

Implements the staged deployment pipeline required before any system
receives live traffic or makes real decisions:

1. Shadow mode — parallel run, no real actions.
2. Canary deployment — 5-10% live traffic.
3. Graduated rollout — 25% → 50% → 100%.
4. Full production — incumbent retained as warm standby.

Also includes monitoring alert thresholds (Section 18.2) and
retraining trigger definitions (Section 18.4).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DeploymentStage(str, Enum):
    RESEARCH = "research"
    SHADOW = "shadow"
    CANARY = "canary"
    GRADUATED_25 = "graduated_25"
    GRADUATED_50 = "graduated_50"
    PRODUCTION = "production"
    ROLLED_BACK = "rolled_back"


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class RollbackTrigger:
    """Condition that triggers automatic rollback."""

    name: str
    description: str
    metric: str
    threshold: float
    comparison: str = ">"  # ">", "<", ">=", "<="

    def evaluate(self, value: float) -> bool:
        """Returns True if rollback should be triggered."""
        ops = {
            ">": lambda v, t: v > t,
            "<": lambda v, t: v < t,
            ">=": lambda v, t: v >= t,
            "<=": lambda v, t: v <= t,
        }
        return ops.get(self.comparison, lambda v, t: False)(value, self.threshold)


@dataclass
class StageConfig:
    """Configuration for a deployment stage."""

    stage: DeploymentStage
    min_duration_cycles: int
    traffic_pct: float
    rollback_triggers: list[RollbackTrigger] = field(default_factory=list)
    gate_criteria: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["stage"] = self.stage.value
        return d


@dataclass
class AlertThreshold:
    """Monitoring alert threshold (Section 18.2)."""

    signal_family: str
    metric: str
    warning_threshold: float
    critical_threshold: float
    sustained_windows: int = 3
    description: str = ""

    def evaluate(self, value: float) -> AlertSeverity:
        if abs(value) >= abs(self.critical_threshold):
            return AlertSeverity.CRITICAL
        if abs(value) >= abs(self.warning_threshold):
            return AlertSeverity.WARNING
        return AlertSeverity.INFO


@dataclass
class RetrainingTrigger:
    """Retraining trigger definition (Section 18.4)."""

    trigger_type: str  # scheduled, performance, drift, data_event
    condition: str
    response: str
    active: bool = True


@dataclass
class DeploymentRecord:
    """Record of a deployment transition."""

    deployment_id: str
    experiment_id: str
    from_stage: DeploymentStage
    to_stage: DeploymentStage
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    approved_by: str = ""
    notes: str = ""
    metrics_at_transition: dict[str, float] = field(default_factory=dict)


def default_stage_configs() -> list[StageConfig]:
    """Default stage configurations per Section 18.1."""
    return [
        StageConfig(
            stage=DeploymentStage.SHADOW,
            min_duration_cycles=50,
            traffic_pct=0.0,
            gate_criteria=(
                "System runs in parallel with incumbent. Predictions logged "
                "but not acted upon. Minimum 50 decision cycles."
            ),
            rollback_triggers=[
                RollbackTrigger(
                    name="schema_divergence",
                    description="Output schema or latency divergence",
                    metric="schema_errors",
                    threshold=0,
                    comparison=">",
                ),
            ],
        ),
        StageConfig(
            stage=DeploymentStage.CANARY,
            min_duration_cycles=100,
            traffic_pct=0.10,
            gate_criteria=(
                "Route 5-10% of live decisions to candidate. Compare "
                "realized performance on matched cohort."
            ),
            rollback_triggers=[
                RollbackTrigger(
                    name="primary_metric_degradation",
                    description="Statistically significant degradation (p < 0.05)",
                    metric="primary_metric_pvalue",
                    threshold=0.05,
                    comparison="<",
                ),
                RollbackTrigger(
                    name="risk_metric_breach",
                    description="Any risk metric exceeds limits",
                    metric="max_drawdown",
                    threshold=-0.15,
                    comparison="<",
                ),
            ],
        ),
        StageConfig(
            stage=DeploymentStage.GRADUATED_25,
            min_duration_cycles=50,
            traffic_pct=0.25,
            gate_criteria="25% allocation with hold period for comparison.",
            rollback_triggers=[
                RollbackTrigger(
                    name="drawdown_breach",
                    description="Drawdown exceeds 1.5x backtest pessimistic",
                    metric="drawdown_ratio",
                    threshold=1.5,
                    comparison=">",
                ),
            ],
        ),
        StageConfig(
            stage=DeploymentStage.GRADUATED_50,
            min_duration_cycles=50,
            traffic_pct=0.50,
            gate_criteria="50% allocation with hold period.",
            rollback_triggers=[
                RollbackTrigger(
                    name="drawdown_breach",
                    description="Drawdown exceeds 1.5x backtest pessimistic",
                    metric="drawdown_ratio",
                    threshold=1.5,
                    comparison=">",
                ),
                RollbackTrigger(
                    name="latency_breach",
                    description="p99 latency exceeds SLA",
                    metric="latency_p99_ms",
                    threshold=500,
                    comparison=">",
                ),
            ],
        ),
        StageConfig(
            stage=DeploymentStage.PRODUCTION,
            min_duration_cycles=0,
            traffic_pct=1.0,
            gate_criteria=(
                "Full production. Incumbent retained as warm standby " "for instant rollback."
            ),
            rollback_triggers=[
                RollbackTrigger(
                    name="circuit_breaker",
                    description="Manual trigger or automated circuit-breaker",
                    metric="circuit_breaker_active",
                    threshold=0,
                    comparison=">",
                ),
            ],
        ),
    ]


def default_alert_thresholds() -> list[AlertThreshold]:
    """Default monitoring thresholds per Section 18.2."""
    return [
        AlertThreshold(
            signal_family="prediction_quality",
            metric="rolling_brier_score_zscore",
            warning_threshold=1.5,
            critical_threshold=2.0,
            sustained_windows=3,
            description="Brier score degrades beyond 2 std from 30-cycle baseline",
        ),
        AlertThreshold(
            signal_family="calibration_health",
            metric="ece",
            warning_threshold=0.06,
            critical_threshold=0.08,
            description="ECE exceeds 0.08",
        ),
        AlertThreshold(
            signal_family="calibration_health",
            metric="reliability_slope_deviation",
            warning_threshold=0.10,
            critical_threshold=0.15,
            description="Reliability slope deviates from 1.0 by more than 0.15",
        ),
        AlertThreshold(
            signal_family="decision_quality",
            metric="rolling_sharpe",
            warning_threshold=0.75,
            critical_threshold=0.50,
            description="Rolling Sharpe drops below 0.5 for 20+ cycles",
        ),
        AlertThreshold(
            signal_family="decision_quality",
            metric="drawdown_vs_max_pct",
            warning_threshold=0.60,
            critical_threshold=0.80,
            description="Current drawdown exceeds 80% of backtest max",
        ),
        AlertThreshold(
            signal_family="feature_health",
            metric="psi",
            warning_threshold=0.15,
            critical_threshold=0.20,
            description="Top feature shows PSI > 0.2",
        ),
        AlertThreshold(
            signal_family="feature_health",
            metric="missing_rate",
            warning_threshold=0.02,
            critical_threshold=0.05,
            description="Missing rate exceeds 5%",
        ),
        AlertThreshold(
            signal_family="infrastructure_health",
            metric="error_rate",
            warning_threshold=0.05,
            critical_threshold=0.10,
            description="Error rate exceeds 0.1%",
        ),
    ]


def default_retraining_triggers() -> list[RetrainingTrigger]:
    """Default retraining triggers per Section 18.4."""
    return [
        RetrainingTrigger(
            trigger_type="scheduled",
            condition="Calendar-based cadence appropriate to domain volatility",
            response=(
                "Execute full pipeline: data refresh, feature recomputation, "
                "model retrain, ensemble recalibration, walk-forward validation, "
                "shadow deployment."
            ),
        ),
        RetrainingTrigger(
            trigger_type="performance",
            condition="Primary metric degrades beyond alert threshold sustained",
            response=(
                "Trigger expedited retraining with priority investigation. "
                "If retrained model also degrades, escalate to human review."
            ),
        ),
        RetrainingTrigger(
            trigger_type="drift",
            condition="Two or more drift axes fire simultaneously",
            response=(
                "Trigger retraining with expanded feature search. "
                "Feature Agent should re-evaluate feature families."
            ),
        ),
        RetrainingTrigger(
            trigger_type="data_event",
            condition="Known structural break (rule change, market structure change)",
            response=(
                "Trigger full pipeline rebuild with updated data lineage. "
                "Invalidate cached feature statistics."
            ),
        ),
    ]


class DeploymentPipeline:
    """Manage staged deployments per Section 18.1.

    Usage::

        pipeline = DeploymentPipeline()
        pipeline.start_shadow("exp_123")
        # ... after 50+ cycles ...
        pipeline.advance("exp_123")  # shadow → canary
        # ... after validation ...
        pipeline.advance("exp_123")  # canary → graduated_25
    """

    STAGE_ORDER = [
        DeploymentStage.RESEARCH,
        DeploymentStage.SHADOW,
        DeploymentStage.CANARY,
        DeploymentStage.GRADUATED_25,
        DeploymentStage.GRADUATED_50,
        DeploymentStage.PRODUCTION,
    ]

    def __init__(
        self,
        storage_path: str | Path = "data/deployment_log.json",
        stage_configs: list[StageConfig] | None = None,
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.stage_configs = {c.stage: c for c in (stage_configs or default_stage_configs())}
        self._deployments: dict[str, list[DeploymentRecord]] = {}
        self._current_stages: dict[str, DeploymentStage] = {}
        self._load()

    def _load(self) -> None:
        if self.storage_path.exists():
            with open(self.storage_path) as f:
                data = json.load(f)
            for entry in data:
                exp_id = entry["experiment_id"]
                rec = DeploymentRecord(
                    deployment_id=entry["deployment_id"],
                    experiment_id=exp_id,
                    from_stage=DeploymentStage(entry["from_stage"]),
                    to_stage=DeploymentStage(entry["to_stage"]),
                    timestamp=entry["timestamp"],
                    approved_by=entry.get("approved_by", ""),
                    notes=entry.get("notes", ""),
                    metrics_at_transition=entry.get("metrics_at_transition", {}),
                )
                self._deployments.setdefault(exp_id, []).append(rec)
                self._current_stages[exp_id] = rec.to_stage

    def _save(self) -> None:
        all_records = []
        for records in self._deployments.values():
            for r in records:
                all_records.append(asdict(r))
        for rec in all_records:
            rec["from_stage"] = (
                rec["from_stage"].value
                if hasattr(rec["from_stage"], "value")
                else rec["from_stage"]
            )
            rec["to_stage"] = (
                rec["to_stage"].value if hasattr(rec["to_stage"], "value") else rec["to_stage"]
            )
        with open(self.storage_path, "w") as f:
            json.dump(all_records, f, indent=2, default=str)

    def get_current_stage(self, experiment_id: str) -> DeploymentStage:
        return self._current_stages.get(experiment_id, DeploymentStage.RESEARCH)

    def start_shadow(self, experiment_id: str, approved_by: str = "system") -> DeploymentRecord:
        """Start shadow deployment for an experiment."""
        return self._transition(
            experiment_id,
            DeploymentStage.RESEARCH,
            DeploymentStage.SHADOW,
            approved_by=approved_by,
        )

    def advance(
        self,
        experiment_id: str,
        approved_by: str = "system",
        metrics: dict[str, float] | None = None,
        notes: str = "",
    ) -> DeploymentRecord:
        """Advance to the next deployment stage."""
        current = self.get_current_stage(experiment_id)
        idx = self.STAGE_ORDER.index(current)
        if idx >= len(self.STAGE_ORDER) - 1:
            raise ValueError(f"Already at final stage: {current.value}")
        next_stage = self.STAGE_ORDER[idx + 1]
        return self._transition(experiment_id, current, next_stage, approved_by, metrics, notes)

    def rollback(
        self,
        experiment_id: str,
        reason: str = "",
        approved_by: str = "system",
    ) -> DeploymentRecord:
        """Roll back to research stage."""
        current = self.get_current_stage(experiment_id)
        return self._transition(
            experiment_id,
            current,
            DeploymentStage.ROLLED_BACK,
            approved_by=approved_by,
            notes=reason,
        )

    def check_rollback_triggers(
        self, experiment_id: str, current_metrics: dict[str, float]
    ) -> list[RollbackTrigger]:
        """Check if any rollback triggers fire for current metrics."""
        stage = self.get_current_stage(experiment_id)
        config = self.stage_configs.get(stage)
        if not config:
            return []
        fired = []
        for trigger in config.rollback_triggers:
            if trigger.metric in current_metrics:
                if trigger.evaluate(current_metrics[trigger.metric]):
                    fired.append(trigger)
                    logger.warning(
                        "Rollback trigger fired for %s: %s (value=%s, threshold=%s)",
                        experiment_id,
                        trigger.name,
                        current_metrics[trigger.metric],
                        trigger.threshold,
                    )
        return fired

    def auto_rollback_if_triggered(
        self, experiment_id: str, current_metrics: dict[str, float]
    ) -> DeploymentRecord | None:
        """Automatically roll back if any trigger fires.

        Per Section 18.1, rollback triggers should be enforced
        automatically without requiring manual intervention.

        Returns the rollback DeploymentRecord if triggered, else None.
        """
        fired = self.check_rollback_triggers(experiment_id, current_metrics)
        if not fired:
            return None
        trigger_names = [t.name for t in fired]
        reason = f"Auto-rollback: triggers fired: {', '.join(trigger_names)}"
        logger.warning("Auto-rollback for %s: %s", experiment_id, reason)
        return self.rollback(experiment_id, reason=reason, approved_by="auto_rollback_system")

    def _transition(
        self,
        experiment_id: str,
        from_stage: DeploymentStage,
        to_stage: DeploymentStage,
        approved_by: str = "",
        metrics: dict[str, float] | None = None,
        notes: str = "",
    ) -> DeploymentRecord:
        from uuid import uuid4

        record = DeploymentRecord(
            deployment_id=str(uuid4()),
            experiment_id=experiment_id,
            from_stage=from_stage,
            to_stage=to_stage,
            approved_by=approved_by,
            notes=notes,
            metrics_at_transition=metrics or {},
        )
        self._deployments.setdefault(experiment_id, []).append(record)
        self._current_stages[experiment_id] = to_stage
        self._save()
        logger.info(
            "Deployment %s: %s → %s (approved by %s)",
            experiment_id,
            from_stage.value,
            to_stage.value,
            approved_by,
        )
        return record

    def get_deployment_history(self, experiment_id: str) -> list[DeploymentRecord]:
        return self._deployments.get(experiment_id, [])

    def export_config(self) -> dict[str, Any]:
        """Export full deployment pipeline config (Section 18.6)."""
        return {
            "stage_configs": [c.to_dict() for c in self.stage_configs.values()],
            "alert_thresholds": [asdict(t) for t in default_alert_thresholds()],
            "retraining_triggers": [asdict(t) for t in default_retraining_triggers()],
        }

    def export_drift_report(
        self,
        experiment_id: str,
        drift_results: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """<drift_detection_report> — Section 18.6 required output."""
        return {
            "report_type": "drift_detection_report",
            "experiment_id": experiment_id,
            "current_stage": self.get_current_stage(experiment_id).value,
            "drift_results": drift_results or {},
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def export_retraining_log(self) -> dict[str, Any]:
        """<retraining_trigger_log> — Section 18.6 required output."""
        triggers = default_retraining_triggers()
        return {
            "report_type": "retraining_trigger_log",
            "configured_triggers": [asdict(t) for t in triggers],
            "deployment_history": {
                exp_id: [
                    {
                        "from": r.from_stage.value,
                        "to": r.to_stage.value,
                        "timestamp": r.timestamp,
                        "notes": r.notes,
                    }
                    for r in records
                ]
                for exp_id, records in self._deployments.items()
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
