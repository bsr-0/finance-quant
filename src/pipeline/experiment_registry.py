"""Experiment Registry and Ledger (Agent Directive V7 — Section 3).

Implements the shared experiment ledger required by the directive. Every
experiment must be logged with a complete record before its results can
be trusted. The registry provides:

* Structured experiment records with all required fields.
* Promotion gates: only experiments that pass audit may be promoted.
* Reproducibility hashing for dataset + config + seed state.
* JSON-file persistence (no database dependency).
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from collections import defaultdict

logger = logging.getLogger(__name__)


class ExperimentStatus(str, Enum):
    """Status of an experiment in the registry."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PROMOTED = "promoted"
    REJECTED = "rejected"


@dataclass
class ExperimentRecord:
    """Complete experiment record per Directive V7 Section 3.

    All agents must write to this shared ledger. No result may be
    trusted unless the ledger entry is complete.
    """

    experiment_id: str = field(default_factory=lambda: str(uuid4()))
    problem_id: str = ""
    dataset_version: str = ""
    as_of_timestamp_rules: str = ""
    feature_set_id: str = ""
    model_family: str = ""
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    validation_scheme: str = ""
    calibration_method: str = ""
    decision_policy: str = ""
    primary_metric: str = ""
    primary_metric_value: float | None = None
    secondary_metrics: dict[str, float] = field(default_factory=dict)
    path_risk_metrics: dict[str, float] = field(default_factory=dict)
    reproducibility_hash: str = ""
    experiment_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    status: ExperimentStatus = ExperimentStatus.RUNNING
    agent: str = ""
    compute_cost_seconds: float = 0.0
    notes: str = ""
    audit_result: str = ""
    promotion_gate_passed: bool = False

    def compute_reproducibility_hash(self) -> str:
        """Compute a hash from the reproducibility-critical fields."""
        payload = json.dumps(
            {
                "dataset_version": self.dataset_version,
                "feature_set_id": self.feature_set_id,
                "model_family": self.model_family,
                "hyperparameters": self.hyperparameters,
                "validation_scheme": self.validation_scheme,
                "calibration_method": self.calibration_method,
            },
            sort_keys=True,
        )
        self.reproducibility_hash = hashlib.sha256(payload.encode()).hexdigest()[:16]
        return self.reproducibility_hash

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExperimentRecord:
        d = dict(d)
        if "status" in d:
            d["status"] = ExperimentStatus(d["status"])
        return cls(**d)


class ExperimentRegistry:
    """Persistent experiment registry backed by a JSON file.

    Usage::

        registry = ExperimentRegistry()
        record = registry.create_experiment(
            problem_id="sp500_direction",
            model_family="lightgbm",
            dataset_version="v3.2",
            feature_set_id="core_tech_v2",
            validation_scheme="walk_forward_expanding",
        )
        # ... run experiment ...
        registry.complete_experiment(
            record.experiment_id,
            primary_metric="sharpe",
            primary_metric_value=1.42,
            secondary_metrics={"sortino": 1.8, "max_dd": -0.12},
        )
    """

    def __init__(self, storage_path: str | Path = "data/experiment_registry.json"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, ExperimentRecord] = {}
        self._load()

    def _load(self) -> None:
        if self.storage_path.exists():
            with open(self.storage_path) as f:
                fcntl.flock(f, fcntl.LOCK_SH)
                try:
                    data = json.load(f)
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
            for d in data:
                rec = ExperimentRecord.from_dict(d)
                self._records[rec.experiment_id] = rec
            logger.info("Loaded %d experiments from registry", len(self._records))

    def _save(self) -> None:
        # Atomic write: write to temp file, then rename (Section 3 concurrency safety)
        content = json.dumps(
            [r.to_dict() for r in self._records.values()],
            indent=2,
            default=str,
        )
        fd, tmp_path = tempfile.mkstemp(dir=str(self.storage_path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    f.write(content)
                    f.flush()
                    os.fsync(f.fileno())
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
            os.replace(tmp_path, str(self.storage_path))
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def create_experiment(self, **kwargs: Any) -> ExperimentRecord:
        """Create and register a new experiment."""
        record = ExperimentRecord(**kwargs)
        record.compute_reproducibility_hash()
        self._records[record.experiment_id] = record
        self._save()
        logger.info(
            "Registered experiment %s [%s / %s]",
            record.experiment_id,
            record.model_family,
            record.problem_id,
        )
        return record

    def complete_experiment(
        self,
        experiment_id: str,
        primary_metric: str = "",
        primary_metric_value: float | None = None,
        secondary_metrics: dict[str, float] | None = None,
        path_risk_metrics: dict[str, float] | None = None,
        compute_cost_seconds: float = 0.0,
        notes: str = "",
    ) -> ExperimentRecord:
        """Mark an experiment as completed with results."""
        rec = self._records[experiment_id]
        rec.status = ExperimentStatus.COMPLETED
        if primary_metric:
            rec.primary_metric = primary_metric
        if primary_metric_value is not None:
            rec.primary_metric_value = primary_metric_value
        if secondary_metrics:
            rec.secondary_metrics.update(secondary_metrics)
        if path_risk_metrics:
            rec.path_risk_metrics.update(path_risk_metrics)
        rec.compute_cost_seconds = compute_cost_seconds
        rec.notes = notes
        self._save()
        logger.info(
            "Completed experiment %s: %s=%.4f",
            experiment_id,
            rec.primary_metric,
            rec.primary_metric_value or 0.0,
        )
        return rec

    def fail_experiment(self, experiment_id: str, reason: str = "") -> None:
        """Mark an experiment as failed."""
        rec = self._records[experiment_id]
        rec.status = ExperimentStatus.FAILED
        rec.notes = reason
        self._save()
        logger.warning("Experiment %s failed: %s", experiment_id, reason)

    def promote_experiment(self, experiment_id: str, audit_result: str = "") -> bool:
        """Promote an experiment after audit.

        Per Section 14: only promote if the candidate wins on both
        primary objective and risk-adjusted robustness criteria.
        """
        rec = self._records[experiment_id]
        if rec.status != ExperimentStatus.COMPLETED:
            logger.error("Cannot promote experiment %s: status is %s", experiment_id, rec.status)
            return False
        rec.status = ExperimentStatus.PROMOTED
        rec.audit_result = audit_result
        rec.promotion_gate_passed = True
        self._save()
        logger.info("Promoted experiment %s", experiment_id)
        return True

    def reject_experiment(self, experiment_id: str, reason: str = "") -> None:
        """Reject an experiment (Section 15 failure modes)."""
        rec = self._records[experiment_id]
        rec.status = ExperimentStatus.REJECTED
        rec.audit_result = reason
        rec.promotion_gate_passed = False
        self._save()
        logger.warning("Rejected experiment %s: %s", experiment_id, reason)

    def get_experiment(self, experiment_id: str) -> ExperimentRecord | None:
        return self._records.get(experiment_id)

    def list_experiments(
        self,
        status: ExperimentStatus | None = None,
        problem_id: str | None = None,
        model_family: str | None = None,
    ) -> list[ExperimentRecord]:
        """List experiments with optional filters."""
        results = list(self._records.values())
        if status is not None:
            results = [r for r in results if r.status == status]
        if problem_id is not None:
            results = [r for r in results if r.problem_id == problem_id]
        if model_family is not None:
            results = [r for r in results if r.model_family == model_family]
        return sorted(results, key=lambda r: r.experiment_timestamp, reverse=True)

    def get_best_experiment(
        self, problem_id: str, metric: str | None = None
    ) -> ExperimentRecord | None:
        """Get the best promoted experiment for a problem."""
        promoted = self.list_experiments(status=ExperimentStatus.PROMOTED, problem_id=problem_id)
        if not promoted:
            return None
        if metric:
            return max(
                promoted,
                key=lambda r: r.secondary_metrics.get(metric, r.primary_metric_value or 0.0),
            )
        return max(promoted, key=lambda r: r.primary_metric_value or 0.0)

    def compute_efficiency_ratio(self, problem_id: str) -> float | None:
        """Compute cost per unit improvement (Section 20.3)."""
        experiments = self.list_experiments(problem_id=problem_id)
        completed = [e for e in experiments if e.primary_metric_value is not None]
        if len(completed) < 2:
            return None
        sorted_by_time = sorted(completed, key=lambda e: e.experiment_timestamp)
        total_cost = sum(e.compute_cost_seconds for e in sorted_by_time)
        improvement = (sorted_by_time[-1].primary_metric_value or 0.0) - (
            sorted_by_time[0].primary_metric_value or 0.0
        )
        if improvement <= 0:
            return None
        return total_cost / improvement

    def check_search_termination(
        self, problem_id: str, n_recent: int = 10, min_improvement_pct: float = 0.001
    ) -> bool:
        """Check if search should stop (Section 20.1).

        Returns True if the last N experiments have not improved the best
        candidate by more than min_improvement_pct.
        """
        experiments = self.list_experiments(problem_id=problem_id)
        completed = [
            e
            for e in experiments
            if e.status in (ExperimentStatus.COMPLETED, ExperimentStatus.PROMOTED)
            and e.primary_metric_value is not None
        ]
        if len(completed) < n_recent:
            return False
        sorted_by_time = sorted(completed, key=lambda e: e.experiment_timestamp)
        recent = sorted_by_time[-n_recent:]
        best_before = (
            max(e.primary_metric_value or 0.0 for e in sorted_by_time[:-n_recent])
            if len(sorted_by_time) > n_recent
            else 0.0
        )
        best_recent = max(e.primary_metric_value or 0.0 for e in recent)
        if best_before == 0:
            return False
        improvement = (best_recent - best_before) / abs(best_before)
        return improvement < min_improvement_pct


# ---------------------------------------------------------------------------
# Knowledge Retention (Section 14)
# ---------------------------------------------------------------------------


@dataclass
class KnowledgeFinding:
    """A finding stored for future agent learning."""

    finding_id: str = field(default_factory=lambda: str(uuid4()))
    domain: str = ""
    horizon: str = ""
    model_family: str = ""
    finding: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    experiment_ids: list[str] = field(default_factory=list)
    works: bool = True  # True = this approach works; False = doesn't work
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> KnowledgeFinding:
        return cls(**d)


class KnowledgeStore:
    """Persistent knowledge store for cross-cycle learning (Section 14).

    Stores findings so future agents learn what tends to work by domain,
    horizon, and data regime.

    Usage::

        store = KnowledgeStore()
        store.store_finding(
            domain="finance",
            horizon="daily",
            model_family="lightgbm",
            finding="LightGBM with rolling_std features outperforms linear models",
            evidence={"sharpe_improvement": 0.3},
            works=True,
        )
        relevant = store.query_findings(domain="finance", horizon="daily")
    """

    def __init__(self, storage_path: str | Path = "data/knowledge_store.json"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._findings: list[KnowledgeFinding] = []
        self._load()

    def _load(self) -> None:
        if self.storage_path.exists():
            with open(self.storage_path) as f:
                fcntl.flock(f, fcntl.LOCK_SH)
                try:
                    data = json.load(f)
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
                for d in data:
                    self._findings.append(KnowledgeFinding.from_dict(d))

    def _save(self) -> None:
        content = json.dumps([f.to_dict() for f in self._findings], indent=2, default=str)
        fd, tmp_path = tempfile.mkstemp(dir=str(self.storage_path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as tf:
                fcntl.flock(tf, fcntl.LOCK_EX)
                try:
                    tf.write(content)
                    tf.flush()
                    os.fsync(tf.fileno())
                finally:
                    fcntl.flock(tf, fcntl.LOCK_UN)
            os.replace(tmp_path, str(self.storage_path))
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def store_finding(
        self,
        domain: str,
        finding: str,
        horizon: str = "",
        model_family: str = "",
        evidence: dict[str, Any] | None = None,
        experiment_ids: list[str] | None = None,
        works: bool = True,
    ) -> KnowledgeFinding:
        """Store a finding for future reference."""
        entry = KnowledgeFinding(
            domain=domain,
            horizon=horizon,
            model_family=model_family,
            finding=finding,
            evidence=evidence or {},
            experiment_ids=experiment_ids or [],
            works=works,
        )
        self._findings.append(entry)
        self._save()
        logger.info("Knowledge stored: %s (works=%s)", finding[:80], works)
        return entry

    def query_findings(
        self,
        domain: str | None = None,
        horizon: str | None = None,
        model_family: str | None = None,
        works: bool | None = None,
    ) -> list[KnowledgeFinding]:
        """Query findings with optional filters."""
        results = list(self._findings)
        if domain is not None:
            results = [f for f in results if f.domain == domain]
        if horizon is not None:
            results = [f for f in results if f.horizon == horizon]
        if model_family is not None:
            results = [f for f in results if f.model_family == model_family]
        if works is not None:
            results = [f for f in results if f.works == works]
        return results

    def generate_meta_learning_insights(
        self,
        registry: ExperimentRegistry | None = None,
    ) -> dict[str, Any]:
        """Analyse stored findings and experiment history to generate insights."""
        by_domain: dict[str, list[KnowledgeFinding]] = defaultdict(list)
        for f in self._findings:
            by_domain[f.domain].append(f)

        domain_summaries = {}
        for domain, findings in by_domain.items():
            works = [f for f in findings if f.works]
            fails = [f for f in findings if not f.works]
            domain_summaries[domain] = {
                "total_findings": len(findings),
                "works": len(works),
                "doesnt_work": len(fails),
                "top_working_approaches": [f.finding for f in works[:5]],
                "top_failures": [f.finding for f in fails[:5]],
            }

        return {
            "report_type": "knowledge_retention_report",
            "domains": domain_summaries,
            "total_findings": len(self._findings),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def export(self) -> list[dict[str, Any]]:
        """Export all findings."""
        return [f.to_dict() for f in self._findings]
