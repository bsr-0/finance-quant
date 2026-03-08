"""Multi-Agent Conflict Resolution Protocol (Agent Directive V7 — Section 22).

Implements the formal conflict resolution protocol so that disagreements
between agents are surfaced, documented, and resolved systematically.

Conflict categories:
- **Factual**: Contradictory empirical claims about experiments/data.
- **Priority**: Disagreement on what to work on next.
- **Safety**: Audit Agent flags a concern another agent disputes.
- **Resource**: Agents compete for limited compute or data budget.

Resolution hierarchy:
1. Evidence duel — both sides submit reproducible evidence.
2. Audit arbitration — Audit Agent runs independent reproduction.
3. Orchestrator decision — for non-safety priority/resource conflicts.
4. Human escalation — final fallback via governance framework.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


class ConflictCategory(str, Enum):
    """Categories of inter-agent conflict (Section 22.1)."""

    FACTUAL = "factual"
    PRIORITY = "priority"
    SAFETY = "safety"
    RESOURCE = "resource"


class ConflictStatus(str, Enum):
    """Resolution status of a conflict."""

    OPEN = "open"
    EVIDENCE_DUEL = "evidence_duel"
    AUDIT_ARBITRATION = "audit_arbitration"
    ORCHESTRATOR_DECISION = "orchestrator_decision"
    ESCALATED_TO_HUMAN = "escalated_to_human"
    RESOLVED = "resolved"


@dataclass
class EvidenceSubmission:
    """Evidence submitted by an agent during an evidence duel."""

    submission_id: str = field(default_factory=lambda: str(uuid4()))
    agent: str = ""
    claim: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    experiment_ids: list[str] = field(default_factory=list)
    code_references: list[str] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Dissent:
    """A dissent filed by an agent that disagrees with a resolution (Section 22.4)."""

    dissent_id: str = field(default_factory=lambda: str(uuid4()))
    conflict_id: str = ""
    agent: str = ""
    reasoning: str = ""
    insufficiently_weighted_evidence: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    reviewed: bool = False
    review_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Dissent:
        return cls(**d)


@dataclass
class ConflictRecord:
    """A conflict record in the conflict log."""

    conflict_id: str = field(default_factory=lambda: str(uuid4()))
    category: ConflictCategory = ConflictCategory.FACTUAL
    status: ConflictStatus = ConflictStatus.OPEN
    description: str = ""
    agents_involved: list[str] = field(default_factory=list)
    evidence_submissions: list[dict[str, Any]] = field(default_factory=list)
    audit_arbitration_result: str = ""
    resolution: str = ""
    resolved_by: str = ""
    dissents: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    resolved_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["category"] = self.category.value
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ConflictRecord:
        d = dict(d)
        d["category"] = ConflictCategory(d["category"])
        d["status"] = ConflictStatus(d["status"])
        return cls(**d)


class ConflictResolver:
    """Conflict resolution protocol per Section 22.

    Usage::

        resolver = ConflictResolver()

        # Raise a conflict
        conflict = resolver.raise_conflict(
            category=ConflictCategory.FACTUAL,
            description="Model Agent reports 3% Brier improvement; "
                        "Audit Agent reproduction shows no improvement.",
            agents_involved=["model_agent", "audit_agent"],
        )

        # Submit evidence from each side
        resolver.submit_evidence(
            conflict.conflict_id,
            agent="model_agent",
            claim="Improvement is real",
            evidence={"experiment_id": "exp_42", "brier_delta": -0.03},
        )
        resolver.submit_evidence(
            conflict.conflict_id,
            agent="audit_agent",
            claim="Improvement disappears after leakage correction",
            evidence={"experiment_id": "exp_42_rerun", "brier_delta": 0.0},
        )

        # Audit arbitration (Audit Agent runs independent reproduction)
        resolver.audit_arbitrate(
            conflict.conflict_id,
            result="Audit reproduction confirms no improvement after "
                   "correcting lookahead bias in feature F7.",
        )

        # File a dissent if an agent disagrees
        resolver.file_dissent(
            conflict.conflict_id,
            agent="model_agent",
            reasoning="Feature F7 uses published data available before event",
        )
    """

    def __init__(
        self,
        storage_path: str | Path = "data/conflict_log.json",
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        self._conflicts: dict[str, ConflictRecord] = {}
        self._dissents: list[Dissent] = []
        self._load()

    # ---- Persistence -------------------------------------------------------

    def _load(self) -> None:
        if self.storage_path.exists():
            with open(self.storage_path) as f:
                data = json.load(f)
            for c in data.get("conflicts", []):
                rec = ConflictRecord.from_dict(c)
                self._conflicts[rec.conflict_id] = rec
            for d in data.get("dissents", []):
                self._dissents.append(Dissent.from_dict(d))

    def _save(self) -> None:
        with open(self.storage_path, "w") as f:
            json.dump(
                {
                    "conflicts": [c.to_dict() for c in self._conflicts.values()],
                    "dissents": [d.to_dict() for d in self._dissents],
                },
                f,
                indent=2,
                default=str,
            )

    # ---- Conflict lifecycle ------------------------------------------------

    def raise_conflict(
        self,
        category: ConflictCategory,
        description: str,
        agents_involved: list[str],
    ) -> ConflictRecord:
        """Raise a new conflict between agents."""
        conflict = ConflictRecord(
            category=category,
            description=description,
            agents_involved=agents_involved,
        )
        self._conflicts[conflict.conflict_id] = conflict
        self._save()
        logger.info(
            "Conflict raised [%s]: %s (agents: %s)",
            category.value,
            description,
            ", ".join(agents_involved),
        )
        return conflict

    def submit_evidence(
        self,
        conflict_id: str,
        agent: str,
        claim: str,
        evidence: dict[str, Any] | None = None,
        experiment_ids: list[str] | None = None,
        code_references: list[str] | None = None,
    ) -> EvidenceSubmission:
        """Submit evidence for an evidence duel (Step 1, Section 22.2)."""
        conflict = self._conflicts[conflict_id]
        submission = EvidenceSubmission(
            agent=agent,
            claim=claim,
            evidence=evidence or {},
            experiment_ids=experiment_ids or [],
            code_references=code_references or [],
        )
        conflict.evidence_submissions.append(submission.to_dict())
        conflict.status = ConflictStatus.EVIDENCE_DUEL
        self._save()
        logger.info(
            "Evidence submitted for conflict %s by %s", conflict_id, agent
        )
        return submission

    def audit_arbitrate(
        self,
        conflict_id: str,
        result: str,
    ) -> ConflictRecord:
        """Audit Agent runs independent reproduction (Step 2, Section 22.2).

        The Audit Agent's finding is binding on factual matters.
        For safety disagreements, the Audit Agent always has veto power.
        """
        conflict = self._conflicts[conflict_id]
        conflict.status = ConflictStatus.AUDIT_ARBITRATION
        conflict.audit_arbitration_result = result

        # For factual and safety conflicts, audit result is binding
        if conflict.category in (ConflictCategory.FACTUAL, ConflictCategory.SAFETY):
            conflict.status = ConflictStatus.RESOLVED
            conflict.resolution = f"Audit arbitration (binding): {result}"
            conflict.resolved_by = "audit_agent"
            conflict.resolved_at = datetime.now(timezone.utc).isoformat()
            logger.info(
                "Conflict %s resolved by audit arbitration: %s",
                conflict_id,
                result,
            )
        self._save()
        return conflict

    def orchestrator_decide(
        self,
        conflict_id: str,
        decision: str,
        justification: str,
    ) -> ConflictRecord:
        """Orchestrator makes final call on priority/resource conflicts (Step 3).

        Only applies to non-safety conflicts.
        """
        conflict = self._conflicts[conflict_id]
        if conflict.category == ConflictCategory.SAFETY:
            raise ValueError(
                "Orchestrator cannot override safety conflicts. "
                "Use audit_arbitrate or escalate_to_human."
            )
        conflict.status = ConflictStatus.RESOLVED
        conflict.resolution = f"Orchestrator decision: {decision}. Justification: {justification}"
        conflict.resolved_by = "research_orchestrator"
        conflict.resolved_at = datetime.now(timezone.utc).isoformat()
        self._save()
        logger.info("Conflict %s resolved by orchestrator: %s", conflict_id, decision)
        return conflict

    def escalate_to_human(
        self,
        conflict_id: str,
        reason: str = "",
    ) -> ConflictRecord:
        """Escalate conflict to human operator (Step 4, Section 22.2)."""
        conflict = self._conflicts[conflict_id]
        conflict.status = ConflictStatus.ESCALATED_TO_HUMAN
        conflict.resolution = f"Escalated to human: {reason}"
        self._save()
        logger.warning(
            "Conflict %s escalated to human: %s", conflict_id, reason
        )
        return conflict

    def resolve_human_escalation(
        self,
        conflict_id: str,
        decision: str,
        decided_by: str,
    ) -> ConflictRecord:
        """Record the human operator's final decision."""
        conflict = self._conflicts[conflict_id]
        conflict.status = ConflictStatus.RESOLVED
        conflict.resolution = f"Human decision by {decided_by}: {decision}"
        conflict.resolved_by = decided_by
        conflict.resolved_at = datetime.now(timezone.utc).isoformat()
        self._save()
        logger.info(
            "Conflict %s resolved by human %s: %s",
            conflict_id,
            decided_by,
            decision,
        )
        return conflict

    # ---- Audit Agent veto (Section 22.3) -----------------------------------

    def audit_veto(
        self,
        conflict_id: str,
        reason: str,
    ) -> ConflictRecord:
        """Exercise the Audit Agent's veto power on safety matters.

        Per Section 22.3: If the Audit Agent identifies confirmed temporal
        leakage, validation contamination, or any Section 15 issue, no
        other agent may override. The only path forward is to fix the
        underlying issue and resubmit for audit.
        """
        conflict = self._conflicts[conflict_id]
        if conflict.category != ConflictCategory.SAFETY:
            raise ValueError(
                "Audit veto is reserved exclusively for safety concerns "
                "(Section 22.3). Use orchestrator_decide for priority/resource."
            )
        conflict.status = ConflictStatus.RESOLVED
        conflict.resolution = f"AUDIT VETO: {reason}. Fix required before resubmission."
        conflict.resolved_by = "audit_agent"
        conflict.resolved_at = datetime.now(timezone.utc).isoformat()
        self._save()
        logger.warning("Audit Agent VETOED conflict %s: %s", conflict_id, reason)
        return conflict

    # ---- Dissent registry (Section 22.4) -----------------------------------

    def file_dissent(
        self,
        conflict_id: str,
        agent: str,
        reasoning: str,
        insufficiently_weighted_evidence: str = "",
    ) -> Dissent:
        """File a dissent against a resolution.

        Dissents do not block action but create a record that can be
        revisited if future evidence vindicates the dissenting position.
        """
        dissent = Dissent(
            conflict_id=conflict_id,
            agent=agent,
            reasoning=reasoning,
            insufficiently_weighted_evidence=insufficiently_weighted_evidence,
        )
        self._dissents.append(dissent)
        conflict = self._conflicts[conflict_id]
        conflict.dissents.append(dissent.to_dict())
        self._save()
        logger.info(
            "Dissent filed by %s on conflict %s", agent, conflict_id
        )
        return dissent

    def review_dissent(
        self,
        dissent_id: str,
        notes: str,
    ) -> Dissent | None:
        """Mark a dissent as reviewed (Orchestrator reviews at cycle start)."""
        for dissent in self._dissents:
            if dissent.dissent_id == dissent_id:
                dissent.reviewed = True
                dissent.review_notes = notes
                self._save()
                return dissent
        return None

    def get_open_dissents(self) -> list[Dissent]:
        """Get all unreviewed dissents for cycle-start review."""
        return [d for d in self._dissents if not d.reviewed]

    # ---- Query -------------------------------------------------------------

    def get_conflict(self, conflict_id: str) -> ConflictRecord | None:
        return self._conflicts.get(conflict_id)

    def list_conflicts(
        self,
        status: ConflictStatus | None = None,
        category: ConflictCategory | None = None,
    ) -> list[ConflictRecord]:
        """List conflicts with optional filters."""
        results = list(self._conflicts.values())
        if status is not None:
            results = [c for c in results if c.status == status]
        if category is not None:
            results = [c for c in results if c.category == category]
        return sorted(results, key=lambda c: c.created_at, reverse=True)

    # ---- Required outputs (Section 22.5) -----------------------------------

    def export_conflict_log(self) -> list[dict[str, Any]]:
        """Export the full conflict log."""
        return [c.to_dict() for c in self._conflicts.values()]

    def export_dissent_registry(self) -> list[dict[str, Any]]:
        """Export the full dissent registry."""
        return [d.to_dict() for d in self._dissents]

    def export_summary(self) -> dict[str, Any]:
        """Export a conflict resolution summary."""
        conflicts = list(self._conflicts.values())
        return {
            "total_conflicts": len(conflicts),
            "open": len([c for c in conflicts if c.status == ConflictStatus.OPEN]),
            "resolved": len([c for c in conflicts if c.status == ConflictStatus.RESOLVED]),
            "escalated": len(
                [c for c in conflicts if c.status == ConflictStatus.ESCALATED_TO_HUMAN]
            ),
            "by_category": {
                cat.value: len([c for c in conflicts if c.category == cat])
                for cat in ConflictCategory
            },
            "total_dissents": len(self._dissents),
            "unreviewed_dissents": len(self.get_open_dissents()),
        }
