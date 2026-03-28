"""Extended Failure Mode Checks (Agent Directive V7 — Section 25.1).

Implements the 8 additional failure modes that must trigger immediate
rejection, beyond those in Section 15.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class FailureCheck:
    """Result of a single failure mode check."""

    check_name: str
    passed: bool
    reason: str = ""
    severity: str = "critical"  # critical = blocks deployment

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FailureModeReport:
    """Aggregated failure mode report."""

    checks: list[FailureCheck]
    all_passed: bool = True
    failed_checks: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_type": "failure_mode_report",
            "all_passed": self.all_passed,
            "checks": [c.to_dict() for c in self.checks],
            "failed_checks": self.failed_checks or [],
            "generated_at": datetime.now(UTC).isoformat(),
        }


class FailureModeChecker:
    """Validates the 8 extended failure modes from Section 25.1.

    Usage::

        checker = FailureModeChecker()
        report = checker.run_all_checks(
            deployment_history=["shadow", "canary", "production"],
            monitoring_active=True,
            pipeline_idempotent=True,
            budget_overrun=False,
            all_actions_authorized=True,
            no_silent_overrides=True,
            ci_gates_passed=True,
            compliance_complete=True,
        )
        if not report.all_passed:
            print("REJECTION:", report.failed_checks)
    """

    @staticmethod
    def check_shadow_bypass(
        deployment_history: list[str],
        human_approved_bypass: bool = False,
    ) -> FailureCheck:
        """25.1.1: A deployment that bypasses shadow mode or canary testing
        without explicit human approval.
        """
        has_shadow = "shadow" in deployment_history
        has_canary = "canary" in deployment_history

        if not has_shadow or not has_canary:
            if human_approved_bypass:
                return FailureCheck(
                    check_name="shadow_bypass",
                    passed=True,
                    reason="Shadow/canary bypassed with human approval",
                )
            return FailureCheck(
                check_name="shadow_bypass",
                passed=False,
                reason="Deployment bypassed shadow/canary without human approval",
            )
        return FailureCheck(
            check_name="shadow_bypass",
            passed=True,
            reason="Shadow and canary stages completed",
        )

    @staticmethod
    def check_monitoring_active(monitoring_active: bool) -> FailureCheck:
        """25.1.2: A production system without functioning monitoring."""
        return FailureCheck(
            check_name="monitoring_active",
            passed=monitoring_active,
            reason=(
                "" if monitoring_active else "Production system lacks active monitoring dashboard"
            ),
        )

    @staticmethod
    def check_pipeline_idempotency(pipeline_idempotent: bool) -> FailureCheck:
        """25.1.3: A data pipeline lacking idempotency, schema validation,
        or fault tolerance.
        """
        return FailureCheck(
            check_name="pipeline_idempotency",
            passed=pipeline_idempotent,
            reason="" if pipeline_idempotent else "Pipeline lacks idempotency or fault tolerance",
        )

    @staticmethod
    def check_budget_overrun(
        budget_exceeded: bool,
        overrun_approved: bool = False,
    ) -> FailureCheck:
        """25.1.4: A compute budget overrun not flagged and approved."""
        if budget_exceeded and not overrun_approved:
            return FailureCheck(
                check_name="budget_overrun",
                passed=False,
                reason="Compute budget exceeded without approval",
            )
        return FailureCheck(
            check_name="budget_overrun",
            passed=True,
            reason="" if not budget_exceeded else "Budget overrun approved",
        )

    @staticmethod
    def check_unauthorized_action(all_actions_authorized: bool) -> FailureCheck:
        """25.1.5: An action requiring human approval executed without approval."""
        return FailureCheck(
            check_name="unauthorized_action",
            passed=all_actions_authorized,
            reason="" if all_actions_authorized else "Action executed without required approval",
        )

    @staticmethod
    def check_silent_conflict_override(no_silent_overrides: bool) -> FailureCheck:
        """25.1.6: A conflict resolved by silent override rather than
        the documented protocol.
        """
        return FailureCheck(
            check_name="silent_conflict_override",
            passed=no_silent_overrides,
            reason="" if no_silent_overrides else "Conflict resolved by silent override",
        )

    @staticmethod
    def check_ci_gates_passed(ci_passed: bool) -> FailureCheck:
        """25.1.7: A code change merged without passing CI/CD gates."""
        return FailureCheck(
            check_name="ci_gates_passed",
            passed=ci_passed,
            reason="" if ci_passed else "Code merged without passing CI/CD gates",
        )

    @staticmethod
    def check_compliance_complete(compliance_done: bool) -> FailureCheck:
        """25.1.8: A system deployed in a regulated domain without
        completing the applicable compliance checklist.
        """
        return FailureCheck(
            check_name="compliance_complete",
            passed=compliance_done,
            reason="" if compliance_done else "Deployed without completing compliance checklist",
        )

    def run_all_checks(
        self,
        deployment_history: list[str] | None = None,
        human_approved_bypass: bool = False,
        monitoring_active: bool = True,
        pipeline_idempotent: bool = True,
        budget_exceeded: bool = False,
        overrun_approved: bool = False,
        all_actions_authorized: bool = True,
        no_silent_overrides: bool = True,
        ci_passed: bool = True,
        compliance_done: bool = True,
    ) -> FailureModeReport:
        """Run all 8 extended failure mode checks."""
        checks = [
            self.check_shadow_bypass(deployment_history or [], human_approved_bypass),
            self.check_monitoring_active(monitoring_active),
            self.check_pipeline_idempotency(pipeline_idempotent),
            self.check_budget_overrun(budget_exceeded, overrun_approved),
            self.check_unauthorized_action(all_actions_authorized),
            self.check_silent_conflict_override(no_silent_overrides),
            self.check_ci_gates_passed(ci_passed),
            self.check_compliance_complete(compliance_done),
        ]

        failed = [c.check_name for c in checks if not c.passed]
        all_passed = len(failed) == 0

        if not all_passed:
            logger.warning(
                "FAILURE MODE REJECTION: %d checks failed: %s",
                len(failed),
                ", ".join(failed),
            )

        return FailureModeReport(
            checks=checks,
            all_passed=all_passed,
            failed_checks=failed if failed else None,
        )
