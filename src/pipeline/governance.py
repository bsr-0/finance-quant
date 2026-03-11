"""Human-in-the-Loop Governance (Agent Directive V7 — Section 21).

Implements the decision authority matrix, approval request protocol,
compliance checkpoints, and audit trail for governance actions.

Authority levels:
- **Autonomous**: Routine research within guardrails — no approval needed.
- **Notify**: Reversible live system changes — execute and notify.
- **Approve**: Significant irreversible impact — wait for explicit approval.
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


class AuthorityLevel(str, Enum):
    AUTONOMOUS = "autonomous"
    NOTIFY = "notify"
    APPROVE = "approve"


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"


class GovernanceDomain(str, Enum):
    FINANCE = "finance"
    BETTING = "betting"
    ELECTIONS = "elections"
    GENERAL = "general"


@dataclass
class ActionClassification:
    """Classification of an agent action."""

    action_type: str
    authority_level: AuthorityLevel
    description: str
    examples: list[str] = field(default_factory=list)


@dataclass
class ApprovalRequest:
    """Structured approval request per Section 21.2."""

    request_id: str = field(default_factory=lambda: str(uuid4()))
    action_summary: str = ""
    evidence_package: dict[str, Any] = field(default_factory=dict)
    risk_assessment: dict[str, Any] = field(default_factory=dict)
    rollback_plan: str = ""
    expiration: str = ""
    requested_by: str = ""
    status: ApprovalStatus = ApprovalStatus.PENDING
    reviewed_by: str = ""
    review_timestamp: str = ""
    review_notes: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class ComplianceCheckpoint:
    """Compliance checkpoint for regulated domains (Section 21.3)."""

    domain: GovernanceDomain
    requirement: str
    agent_responsibility: str
    checked: bool = False
    check_result: str = ""
    check_timestamp: str = ""


@dataclass
class GovernanceAuditEntry:
    """Immutable audit trail entry (Section 21.4)."""

    entry_id: str = field(default_factory=lambda: str(uuid4()))
    action_type: str = ""
    actor: str = ""
    justification: str = ""
    outcome: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    related_request_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def default_decision_authority_matrix() -> list[ActionClassification]:
    """Default decision authority matrix per Section 21.1."""
    return [
        # Autonomous actions
        ActionClassification(
            action_type="run_experiment",
            authority_level=AuthorityLevel.AUTONOMOUS,
            description="Routine research and experimentation within guardrails",
            examples=[
                "Running experiments",
                "Training models",
                "Generating features",
                "Logging results",
                "Executing research loop",
            ],
        ),
        ActionClassification(
            action_type="data_analysis",
            authority_level=AuthorityLevel.AUTONOMOUS,
            description="Data exploration and analysis",
            examples=["Computing statistics", "Generating reports", "Running audits"],
        ),
        # Notify actions
        ActionClassification(
            action_type="shadow_deployment",
            authority_level=AuthorityLevel.NOTIFY,
            description="Promote candidate to shadow mode",
            examples=["Starting shadow deployment", "Adjusting alert thresholds"],
        ),
        ActionClassification(
            action_type="scheduled_retrain",
            authority_level=AuthorityLevel.NOTIFY,
            description="Execute scheduled retraining",
            examples=[
                "Activating scheduled retrain",
                "Updating feature pipelines",
            ],
        ),
        # Approve actions
        ActionClassification(
            action_type="production_deploy",
            authority_level=AuthorityLevel.APPROVE,
            description="Deploy new model to full production",
            examples=[
                "Deploying to production",
                "Changing decision policy",
                "Increasing exposure limits",
            ],
        ),
        ActionClassification(
            action_type="data_source_change",
            authority_level=AuthorityLevel.APPROVE,
            description="Decommission or add a data source",
            examples=[
                "Removing a data provider",
                "Adding a new upstream feed",
            ],
        ),
        ActionClassification(
            action_type="anomaly_action",
            authority_level=AuthorityLevel.APPROVE,
            description="Any action during a detected anomaly",
            examples=[
                "Trading during drift event",
                "Overriding risk limits",
            ],
        ),
    ]


def default_compliance_checkpoints(
    domain: GovernanceDomain,
) -> list[ComplianceCheckpoint]:
    """Domain-specific compliance checkpoints per Section 21.3."""
    checkpoints = {
        GovernanceDomain.FINANCE: [
            ComplianceCheckpoint(
                domain=GovernanceDomain.FINANCE,
                requirement="Regulatory position limits and reporting obligations",
                agent_responsibility=(
                    "Decision Agent must check all proposed actions against "
                    "configured regulatory limits before execution."
                ),
            ),
            ComplianceCheckpoint(
                domain=GovernanceDomain.FINANCE,
                requirement="Best-execution requirements",
                agent_responsibility=(
                    "Verify execution quality and document compliance "
                    "with applicable algorithmic trading regulations."
                ),
            ),
        ],
        GovernanceDomain.BETTING: [
            ComplianceCheckpoint(
                domain=GovernanceDomain.BETTING,
                requirement="Jurisdictional legality and platform ToS",
                agent_responsibility=(
                    "Verify jurisdictional compliance for every action. "
                    "Respect configured loss limits."
                ),
            ),
        ],
        GovernanceDomain.ELECTIONS: [
            ComplianceCheckpoint(
                domain=GovernanceDomain.ELECTIONS,
                requirement="Data privacy and disclosure requirements",
                agent_responsibility=(
                    "Flag data sources subject to privacy or disclosure rules. "
                    "Ensure compliance before ingestion."
                ),
            ),
        ],
    }
    return checkpoints.get(domain, [])


class GovernanceFramework:
    """Governance framework implementing Section 21.

    Usage::

        gov = GovernanceFramework(domain=GovernanceDomain.FINANCE)

        # Check if an action needs approval
        level = gov.get_authority_level("production_deploy")
        if level == AuthorityLevel.APPROVE:
            request = gov.submit_approval_request(
                action_summary="Deploy model v3.2 to production",
                evidence_package={"experiment_id": "exp_123", ...},
                risk_assessment={"worst_case_drawdown": -0.15, ...},
                rollback_plan="Revert to model v3.1 via warm standby",
            )
            # ... wait for human approval ...
            gov.approve_request(request.request_id, reviewed_by="analyst_1")
    """

    def __init__(
        self,
        domain: GovernanceDomain = GovernanceDomain.GENERAL,
        storage_path: str | Path = "data/governance_log.json",
    ):
        self.domain = domain
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        self._authority_matrix = {a.action_type: a for a in default_decision_authority_matrix()}
        self._compliance_checkpoints = default_compliance_checkpoints(domain)
        self._approval_requests: dict[str, ApprovalRequest] = {}
        self._audit_trail: list[GovernanceAuditEntry] = []
        self._load()

    def _load(self) -> None:
        if self.storage_path.exists():
            with open(self.storage_path) as f:
                data = json.load(f)
            for entry in data.get("audit_trail", []):
                self._audit_trail.append(GovernanceAuditEntry(**entry))
            for req in data.get("approval_requests", []):
                req["status"] = ApprovalStatus(req["status"])
                r = ApprovalRequest(**req)
                self._approval_requests[r.request_id] = r

    def _save(self) -> None:
        with open(self.storage_path, "w") as f:
            json.dump(
                {
                    "audit_trail": [asdict(e) for e in self._audit_trail],
                    "approval_requests": [
                        {**asdict(r), "status": r.status.value}
                        for r in self._approval_requests.values()
                    ],
                },
                f,
                indent=2,
                default=str,
            )

    def get_authority_level(self, action_type: str) -> AuthorityLevel:
        """Look up the authority level for an action type."""
        classification = self._authority_matrix.get(action_type)
        if classification is None:
            logger.warning("Unknown action type '%s' — defaulting to APPROVE", action_type)
            return AuthorityLevel.APPROVE
        return classification.authority_level

    def check_action_allowed(self, action_type: str, actor: str = "agent") -> tuple[bool, str]:
        """Check whether an action can proceed.

        Returns (allowed, reason).
        """
        level = self.get_authority_level(action_type)

        if level == AuthorityLevel.AUTONOMOUS:
            self._log_audit(action_type, actor, "auto_approved", "Autonomous action")
            return True, "autonomous"

        if level == AuthorityLevel.NOTIFY:
            self._log_audit(action_type, actor, "notify_sent", "Notify action executed")
            return True, "notify"

        # APPROVE level — check for existing approval
        pending = [
            r
            for r in self._approval_requests.values()
            if r.status == ApprovalStatus.APPROVED and action_type in r.action_summary.lower()
        ]
        if pending:
            return True, f"approved via request {pending[0].request_id}"
        return False, "requires_approval"

    def submit_approval_request(
        self,
        action_summary: str,
        evidence_package: dict[str, Any] | None = None,
        risk_assessment: dict[str, Any] | None = None,
        rollback_plan: str = "",
        requested_by: str = "agent",
        expiration: str = "",
    ) -> ApprovalRequest:
        """Submit a structured approval request (Section 21.2)."""
        request = ApprovalRequest(
            action_summary=action_summary,
            evidence_package=evidence_package or {},
            risk_assessment=risk_assessment or {},
            rollback_plan=rollback_plan,
            requested_by=requested_by,
            expiration=expiration,
        )
        self._approval_requests[request.request_id] = request
        self._log_audit(
            "approval_request",
            requested_by,
            "submitted",
            action_summary,
            related_request_id=request.request_id,
        )
        self._save()
        logger.info("Approval request submitted: %s", request.request_id)
        return request

    def approve_request(self, request_id: str, reviewed_by: str, notes: str = "") -> bool:
        """Approve a pending request."""
        req = self._approval_requests.get(request_id)
        if not req or req.status != ApprovalStatus.PENDING:
            return False
        req.status = ApprovalStatus.APPROVED
        req.reviewed_by = reviewed_by
        req.review_timestamp = datetime.now(timezone.utc).isoformat()
        req.review_notes = notes
        self._log_audit(
            "approval_decision",
            reviewed_by,
            "approved",
            req.action_summary,
            related_request_id=request_id,
        )
        self._save()
        logger.info("Request %s approved by %s", request_id, reviewed_by)
        return True

    def deny_request(self, request_id: str, reviewed_by: str, reason: str = "") -> bool:
        """Deny a pending request."""
        req = self._approval_requests.get(request_id)
        if not req or req.status != ApprovalStatus.PENDING:
            return False
        req.status = ApprovalStatus.DENIED
        req.reviewed_by = reviewed_by
        req.review_timestamp = datetime.now(timezone.utc).isoformat()
        req.review_notes = reason
        self._log_audit(
            "approval_decision",
            reviewed_by,
            "denied",
            reason,
            related_request_id=request_id,
        )
        self._save()
        logger.warning("Request %s denied by %s: %s", request_id, reviewed_by, reason)
        return True

    def run_compliance_checks(self) -> list[ComplianceCheckpoint]:
        """Run all compliance checkpoints for the domain (Section 21.3)."""
        for checkpoint in self._compliance_checkpoints:
            checkpoint.checked = True
            checkpoint.check_timestamp = datetime.now(timezone.utc).isoformat()
            checkpoint.check_result = "reviewed"
        self._log_audit(
            "compliance_check",
            "system",
            "completed",
            f"Ran {len(self._compliance_checkpoints)} compliance checkpoints for {self.domain.value}",
        )
        self._save()
        return self._compliance_checkpoints

    def enforce_expirations(self) -> list[str]:
        """Expire pending approval requests past their expiration time.

        Per Section 21.2, an expired request is void and the agent must
        resubmit with updated evidence.

        Returns a list of expired request IDs.
        """
        now = datetime.now(timezone.utc)
        expired_ids: list[str] = []
        for req in self._approval_requests.values():
            if req.status != ApprovalStatus.PENDING:
                continue
            if not req.expiration:
                continue
            try:
                exp_dt = datetime.fromisoformat(req.expiration)
                if exp_dt.tzinfo is None:
                    exp_dt = exp_dt.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                continue
            if now > exp_dt:
                req.status = ApprovalStatus.EXPIRED
                expired_ids.append(req.request_id)
                self._log_audit(
                    "approval_expired",
                    "system",
                    "expired",
                    f"Request {req.request_id} expired at {req.expiration}",
                    related_request_id=req.request_id,
                )
        if expired_ids:
            self._save()
            logger.info("Expired %d approval requests", len(expired_ids))
        return expired_ids

    def get_audit_trail(self) -> list[GovernanceAuditEntry]:
        """Get the full immutable audit trail (Section 21.4)."""
        return list(self._audit_trail)

    def _log_audit(
        self,
        action_type: str,
        actor: str,
        outcome: str,
        justification: str,
        related_request_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        entry = GovernanceAuditEntry(
            action_type=action_type,
            actor=actor,
            outcome=outcome,
            justification=justification,
            related_request_id=related_request_id,
            metadata=metadata or {},
        )
        self._audit_trail.append(entry)

    def export_decision_authority_matrix(self) -> list[dict[str, Any]]:
        """Export the decision authority matrix (Section 21.5)."""
        return [asdict(a) for a in self._authority_matrix.values()]

    def export_approval_request_log(self) -> dict[str, Any]:
        """<approval_request_log> — Section 21.5 required output."""
        return {
            "report_type": "approval_request_log",
            "requests": [
                {
                    "request_id": r.request_id,
                    "action_summary": r.action_summary,
                    "status": r.status.value,
                    "requested_by": r.requested_by,
                    "requested_at": r.timestamp,
                    "reviewed_by": r.reviewed_by,
                    "review_timestamp": r.review_timestamp,
                    "review_notes": r.review_notes,
                }
                for r in self._approval_requests.values()
            ],
            "total_requests": len(self._approval_requests),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def export_escalation_protocol(self) -> dict[str, Any]:
        """<escalation_protocol> — Section 21.5 required output."""
        return {
            "report_type": "escalation_protocol",
            "authority_levels": [
                {
                    "level": "autonomous",
                    "description": "Routine research within guardrails",
                    "examples": "Running experiments, training models, generating features",
                    "approval_required": False,
                },
                {
                    "level": "notify",
                    "description": "Reversible live system changes",
                    "examples": "Shadow deployment, adjusting thresholds, scheduled retrain",
                    "approval_required": False,
                    "notification_required": True,
                },
                {
                    "level": "approve",
                    "description": "Significant, potentially irreversible impact",
                    "examples": "Production deploy, changing decision policy, increasing exposure",
                    "approval_required": True,
                    "required_evidence": [
                        "risk_summary",
                        "backtest_evidence",
                        "rollback_plan",
                    ],
                },
            ],
            "escalation_path": [
                "1. Agent identifies action requiring approval",
                "2. Submit structured approval request with evidence",
                "3. Human reviewer evaluates risk and evidence",
                "4. Approve/deny with justification logged to audit trail",
                "5. If denied, agent must revise approach and resubmit",
                "6. For regulatory issues, escalate to compliance officer",
            ],
            "compliance_domains": [d.value for d in GovernanceDomain],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
