"""Tests for Agent Directive V7 modules.

Covers:
- Experiment Registry (Section 3)
- Drift Detection (Section 18.3)
- Deployment Pipeline (Section 18.1)
- Compute Budget (Section 20)
- Governance Framework (Section 21)
- Agent Coordinator (Section 2)
- Conflict Resolution (Section 22)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Experiment Registry (Section 3)
# ---------------------------------------------------------------------------


class TestExperimentRegistry:

    def _make_registry(self, tmp_path: Path):
        from pipeline.experiment_registry import ExperimentRegistry

        return ExperimentRegistry(storage_path=tmp_path / "registry.json")

    def test_create_experiment(self, tmp_path):
        reg = self._make_registry(tmp_path)
        rec = reg.create_experiment(
            problem_id="sp500_direction",
            model_family="lightgbm",
            dataset_version="v3.2",
            feature_set_id="core_tech_v2",
            validation_scheme="walk_forward",
        )
        assert rec.experiment_id
        assert rec.problem_id == "sp500_direction"
        assert rec.reproducibility_hash  # auto-computed

    def test_complete_experiment(self, tmp_path):
        reg = self._make_registry(tmp_path)
        rec = reg.create_experiment(problem_id="test", model_family="lr")
        reg.complete_experiment(
            rec.experiment_id,
            primary_metric="sharpe",
            primary_metric_value=1.42,
            secondary_metrics={"sortino": 1.8},
        )
        updated = reg.get_experiment(rec.experiment_id)
        assert updated.status.value == "completed"
        assert updated.primary_metric_value == 1.42

    def test_promote_requires_completed(self, tmp_path):
        reg = self._make_registry(tmp_path)
        rec = reg.create_experiment(problem_id="test", model_family="lr")
        # Cannot promote a running experiment
        assert not reg.promote_experiment(rec.experiment_id)

    def test_promote_after_complete(self, tmp_path):
        reg = self._make_registry(tmp_path)
        rec = reg.create_experiment(problem_id="test", model_family="lr")
        reg.complete_experiment(rec.experiment_id, primary_metric_value=1.5)
        assert reg.promote_experiment(rec.experiment_id, audit_result="passed")
        updated = reg.get_experiment(rec.experiment_id)
        assert updated.status.value == "promoted"

    def test_reject_experiment(self, tmp_path):
        reg = self._make_registry(tmp_path)
        rec = reg.create_experiment(problem_id="test", model_family="lr")
        reg.complete_experiment(rec.experiment_id, primary_metric_value=0.3)
        reg.reject_experiment(rec.experiment_id, reason="leakage detected")
        updated = reg.get_experiment(rec.experiment_id)
        assert updated.status.value == "rejected"

    def test_list_experiments_with_filters(self, tmp_path):
        reg = self._make_registry(tmp_path)
        reg.create_experiment(problem_id="a", model_family="lr")
        reg.create_experiment(problem_id="b", model_family="lgbm")
        reg.create_experiment(problem_id="a", model_family="lgbm")

        assert len(reg.list_experiments(problem_id="a")) == 2
        assert len(reg.list_experiments(model_family="lgbm")) == 2

    def test_persistence(self, tmp_path):
        path = tmp_path / "registry.json"
        from pipeline.experiment_registry import ExperimentRegistry

        reg1 = ExperimentRegistry(storage_path=path)
        rec = reg1.create_experiment(problem_id="persist", model_family="lr")
        reg1.complete_experiment(rec.experiment_id, primary_metric_value=1.0)

        # Load fresh
        reg2 = ExperimentRegistry(storage_path=path)
        loaded = reg2.get_experiment(rec.experiment_id)
        assert loaded is not None
        assert loaded.primary_metric_value == 1.0

    def test_get_best_experiment(self, tmp_path):
        reg = self._make_registry(tmp_path)
        r1 = reg.create_experiment(problem_id="p", model_family="lr")
        reg.complete_experiment(r1.experiment_id, primary_metric_value=1.0)
        reg.promote_experiment(r1.experiment_id)

        r2 = reg.create_experiment(problem_id="p", model_family="lgbm")
        reg.complete_experiment(r2.experiment_id, primary_metric_value=2.0)
        reg.promote_experiment(r2.experiment_id)

        best = reg.get_best_experiment("p")
        assert best.primary_metric_value == 2.0

    def test_search_termination(self, tmp_path):
        reg = self._make_registry(tmp_path)
        # Create 15 experiments with no improvement
        for _i in range(15):
            r = reg.create_experiment(problem_id="stale", model_family="lr")
            reg.complete_experiment(r.experiment_id, primary_metric_value=1.0)
        assert reg.check_search_termination("stale", n_recent=10)


# ---------------------------------------------------------------------------
# Drift Detection (Section 18.3)
# ---------------------------------------------------------------------------


class TestDriftDetection:

    def _make_data(self, n=200, seed=42):
        rng = np.random.default_rng(seed)
        features = pd.DataFrame(
            {"f1": rng.normal(0, 1, n), "f2": rng.normal(0, 1, n)},
            index=pd.bdate_range("2023-01-01", periods=n),
        )
        target = pd.Series(rng.normal(0, 1, n), index=features.index, name="target")
        return features, target

    def test_no_drift_detected(self):
        from pipeline.drift_detection import DriftDetector

        ref_feat, ref_target = self._make_data(200, seed=42)
        cur_feat, cur_target = self._make_data(200, seed=43)

        detector = DriftDetector(ref_feat, ref_target)
        report = detector.run_all_checks(cur_feat, cur_target)
        # Same distribution, should not trigger
        assert not report.requires_retraining

    def test_data_drift_detected(self):
        from pipeline.drift_detection import DriftDetector

        ref_feat, ref_target = self._make_data(200, seed=42)
        # Shifted distribution
        rng = np.random.default_rng(99)
        cur_feat = pd.DataFrame(
            {"f1": rng.normal(5, 1, 200), "f2": rng.normal(5, 1, 200)},
            index=pd.bdate_range("2024-01-01", periods=200),
        )
        cur_target = pd.Series(rng.normal(0, 1, 200), index=cur_feat.index)

        detector = DriftDetector(ref_feat, ref_target)
        report = detector.run_all_checks(cur_feat, cur_target)
        triggered = [r.axis.value for r in report.results if r.triggered]
        assert "data_drift" in triggered

    def test_label_drift_detected(self):
        from pipeline.drift_detection import DriftDetector

        ref_feat, ref_target = self._make_data(200, seed=42)
        cur_feat, _ = self._make_data(200, seed=43)
        # Massively shifted target
        cur_target = pd.Series(np.ones(200) * 10, index=cur_feat.index)

        detector = DriftDetector(ref_feat, ref_target)
        results = detector.check_label_drift(cur_target)
        assert any(r.triggered for r in results)

    def test_psi_computation(self):
        from pipeline.drift_detection import population_stability_index

        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 1000)
        same = rng.normal(0, 1, 1000)
        shifted = rng.normal(3, 1, 1000)

        psi_same = population_stability_index(ref, same)
        psi_shifted = population_stability_index(ref, shifted)
        assert psi_same < 0.1
        assert psi_shifted > 0.2

    def test_requires_retraining_two_axes(self):
        from pipeline.drift_detection import DriftDetector

        ref_feat, ref_target = self._make_data(200, seed=42)
        rng = np.random.default_rng(99)
        # Both features and target shifted
        cur_feat = pd.DataFrame(
            {"f1": rng.normal(5, 1, 200), "f2": rng.normal(5, 1, 200)},
            index=pd.bdate_range("2024-01-01", periods=200),
        )
        cur_target = pd.Series(np.ones(200) * 10, index=cur_feat.index)

        detector = DriftDetector(ref_feat, ref_target)
        report = detector.run_all_checks(cur_feat, cur_target)
        assert report.requires_retraining


# ---------------------------------------------------------------------------
# Deployment Pipeline (Section 18.1)
# ---------------------------------------------------------------------------


class TestDeploymentPipeline:

    def test_stage_progression(self, tmp_path):
        from pipeline.deployment_pipeline import DeploymentPipeline, DeploymentStage

        dp = DeploymentPipeline(storage_path=tmp_path / "deploy.json")

        dp.start_shadow("exp_1")
        assert dp.get_current_stage("exp_1") == DeploymentStage.SHADOW

        dp.advance("exp_1")
        assert dp.get_current_stage("exp_1") == DeploymentStage.CANARY

        dp.advance("exp_1")
        assert dp.get_current_stage("exp_1") == DeploymentStage.GRADUATED_25

    def test_rollback(self, tmp_path):
        from pipeline.deployment_pipeline import DeploymentPipeline, DeploymentStage

        dp = DeploymentPipeline(storage_path=tmp_path / "deploy.json")

        dp.start_shadow("exp_1")
        dp.advance("exp_1")
        dp.rollback("exp_1", reason="performance degradation")
        assert dp.get_current_stage("exp_1") == DeploymentStage.ROLLED_BACK

    def test_cannot_advance_past_production(self, tmp_path):
        from pipeline.deployment_pipeline import DeploymentPipeline

        dp = DeploymentPipeline(storage_path=tmp_path / "deploy.json")

        dp.start_shadow("exp_1")
        dp.advance("exp_1")  # canary
        dp.advance("exp_1")  # grad 25
        dp.advance("exp_1")  # grad 50
        dp.advance("exp_1")  # production

        with pytest.raises(ValueError):
            dp.advance("exp_1")

    def test_rollback_trigger_check(self, tmp_path):
        from pipeline.deployment_pipeline import DeploymentPipeline

        dp = DeploymentPipeline(storage_path=tmp_path / "deploy.json")
        dp.start_shadow("exp_1")
        dp.advance("exp_1")  # canary

        # Check rollback triggers with bad metrics
        fired = dp.check_rollback_triggers("exp_1", {"max_drawdown": -0.20})
        assert len(fired) > 0

    def test_persistence(self, tmp_path):
        from pipeline.deployment_pipeline import DeploymentPipeline, DeploymentStage

        path = tmp_path / "deploy.json"
        dp1 = DeploymentPipeline(storage_path=path)
        dp1.start_shadow("exp_1")
        dp1.advance("exp_1")

        dp2 = DeploymentPipeline(storage_path=path)
        assert dp2.get_current_stage("exp_1") == DeploymentStage.CANARY

    def test_export_config(self, tmp_path):
        from pipeline.deployment_pipeline import DeploymentPipeline

        dp = DeploymentPipeline(storage_path=tmp_path / "deploy.json")
        config = dp.export_config()
        assert "stage_configs" in config
        assert "alert_thresholds" in config
        assert "retraining_triggers" in config


# ---------------------------------------------------------------------------
# Compute Budget (Section 20)
# ---------------------------------------------------------------------------


class TestComputeBudget:

    def test_budget_tracking(self, tmp_path):
        from pipeline.compute_budget import ComputeBudget

        budget = ComputeBudget(
            total_budget_hours=1.0,
            storage_path=tmp_path / "budget.json",
        )
        with budget.track_experiment("exp_1", phase="model_search") as t:
            t.primary_metric_value = 1.0

        assert budget.total_consumed > 0
        report = budget.get_report()
        assert report.total_consumed_seconds > 0

    def test_phase_allocation(self, tmp_path):
        from pipeline.compute_budget import ComputeBudget

        budget = ComputeBudget(
            total_budget_hours=10.0,
            storage_path=tmp_path / "budget.json",
        )
        # model_search gets 35%
        assert budget.phases["model_search"].pct == 0.35
        assert budget.phases["model_search"].budget_seconds == 10 * 3600 * 0.35

    def test_budget_available_check(self, tmp_path):
        from pipeline.compute_budget import ComputeBudget

        budget = ComputeBudget(
            total_budget_hours=0.001,  # tiny budget
            storage_path=tmp_path / "budget.json",
        )
        assert budget.check_budget_available("model_search")

    def test_report_generation(self, tmp_path):
        from pipeline.compute_budget import ComputeBudget

        budget = ComputeBudget(
            total_budget_hours=1.0,
            storage_path=tmp_path / "budget.json",
        )
        report = budget.get_report()
        assert len(report.phase_breakdown) == 7  # 7 default phases
        assert report.total_budget_seconds == 3600.0


# ---------------------------------------------------------------------------
# Governance Framework (Section 21)
# ---------------------------------------------------------------------------


class TestGovernanceFramework:

    def test_authority_levels(self, tmp_path):
        from pipeline.governance import AuthorityLevel, GovernanceFramework

        gov = GovernanceFramework(storage_path=tmp_path / "gov.json")

        assert gov.get_authority_level("run_experiment") == AuthorityLevel.AUTONOMOUS
        assert gov.get_authority_level("shadow_deployment") == AuthorityLevel.NOTIFY
        assert gov.get_authority_level("production_deploy") == AuthorityLevel.APPROVE
        # Unknown defaults to APPROVE
        assert gov.get_authority_level("unknown_action") == AuthorityLevel.APPROVE

    def test_autonomous_action_allowed(self, tmp_path):
        from pipeline.governance import GovernanceFramework

        gov = GovernanceFramework(storage_path=tmp_path / "gov.json")
        allowed, reason = gov.check_action_allowed("run_experiment")
        assert allowed
        assert reason == "autonomous"

    def test_approve_action_blocked_without_request(self, tmp_path):
        from pipeline.governance import GovernanceFramework

        gov = GovernanceFramework(storage_path=tmp_path / "gov.json")
        allowed, reason = gov.check_action_allowed("production_deploy")
        assert not allowed
        assert reason == "requires_approval"

    def test_approval_workflow(self, tmp_path):
        from pipeline.governance import GovernanceFramework

        gov = GovernanceFramework(storage_path=tmp_path / "gov.json")

        request = gov.submit_approval_request(
            action_summary="Deploy model v3 to production_deploy",
            risk_assessment={"worst_case_drawdown": -0.15},
            rollback_plan="Revert to model v2",
        )
        assert request.status.value == "pending"

        gov.approve_request(request.request_id, reviewed_by="analyst_1")
        allowed, reason = gov.check_action_allowed("production_deploy")
        assert allowed

    def test_deny_request(self, tmp_path):
        from pipeline.governance import GovernanceFramework

        gov = GovernanceFramework(storage_path=tmp_path / "gov.json")

        request = gov.submit_approval_request(
            action_summary="Deploy risky model",
        )
        gov.deny_request(request.request_id, reviewed_by="risk_mgr", reason="Too risky")

        req = gov._approval_requests[request.request_id]
        assert req.status.value == "denied"

    def test_audit_trail(self, tmp_path):
        from pipeline.governance import GovernanceFramework

        gov = GovernanceFramework(storage_path=tmp_path / "gov.json")
        gov.check_action_allowed("run_experiment", actor="model_agent")
        trail = gov.get_audit_trail()
        assert len(trail) > 0
        assert trail[0].actor == "model_agent"

    def test_compliance_checkpoints(self, tmp_path):
        from pipeline.governance import GovernanceDomain, GovernanceFramework

        gov = GovernanceFramework(
            domain=GovernanceDomain.FINANCE,
            storage_path=tmp_path / "gov.json",
        )
        checkpoints = gov.run_compliance_checks()
        assert len(checkpoints) > 0
        assert all(c.checked for c in checkpoints)

    def test_persistence(self, tmp_path):
        from pipeline.governance import GovernanceFramework

        path = tmp_path / "gov.json"
        gov1 = GovernanceFramework(storage_path=path)
        gov1.check_action_allowed("run_experiment")
        gov1.submit_approval_request(action_summary="test action")

        gov2 = GovernanceFramework(storage_path=path)
        assert len(gov2.get_audit_trail()) > 0
        assert len(gov2._approval_requests) > 0


# ---------------------------------------------------------------------------
# Agent Coordinator (Section 2)
# ---------------------------------------------------------------------------


class TestAgentCoordinator:

    def _make_coordinator(self, tmp_path: Path):
        from pipeline.agent_coordinator import AgentCoordinator

        return AgentCoordinator(storage_path=tmp_path / "coordinator.json")

    def test_default_agents(self, tmp_path):
        from pipeline.agent_coordinator import AgentRole

        coord = self._make_coordinator(tmp_path)
        agents = coord.list_agents()
        assert len(agents) == 7
        roles = {a.role for a in agents}
        assert AgentRole.RESEARCH_ORCHESTRATOR in roles
        assert AgentRole.AUDIT_AGENT in roles

    def test_audit_agent_has_veto(self, tmp_path):
        from pipeline.agent_coordinator import AgentRole

        coord = self._make_coordinator(tmp_path)
        audit = coord.get_agent(AgentRole.AUDIT_AGENT)
        assert audit.can_veto is True
        # Other agents should not have veto
        data = coord.get_agent(AgentRole.DATA_AGENT)
        assert data.can_veto is False

    def test_set_and_get_roadmap(self, tmp_path):
        coord = self._make_coordinator(tmp_path)
        roadmap = coord.set_roadmap(
            problem_id="sp500",
            objective="Predict direction",
            phases=["data", "features", "model", "audit"],
        )
        assert roadmap.problem_id == "sp500"
        assert roadmap.current_phase == "data"
        assert coord.get_roadmap() is not None

    def test_advance_phase(self, tmp_path):
        coord = self._make_coordinator(tmp_path)
        coord.set_roadmap(
            problem_id="test",
            objective="test",
            phases=["a", "b", "c"],
        )
        assert coord.advance_phase() == "b"
        assert coord.advance_phase() == "c"
        with pytest.raises(ValueError):
            coord.advance_phase()

    def test_register_failure(self, tmp_path):
        coord = self._make_coordinator(tmp_path)
        coord.set_roadmap(problem_id="test", objective="test")
        coord.register_failure("Leakage detected in feature F7")
        roadmap = coord.get_roadmap()
        assert len(roadmap.failure_register) == 1
        assert "F7" in roadmap.failure_register[0]

    def test_assign_and_complete_task(self, tmp_path):
        from pipeline.agent_coordinator import AgentRole, TaskStatus

        coord = self._make_coordinator(tmp_path)
        task = coord.assign_task(
            role=AgentRole.DATA_AGENT,
            description="Build dataset",
            priority=1,
        )
        assert task.status == TaskStatus.PENDING

        coord.start_task(task.task_id)
        assert coord.get_task(task.task_id).status == TaskStatus.IN_PROGRESS

        coord.complete_task(task.task_id, result={"rows": 1000})
        assert coord.get_task(task.task_id).status == TaskStatus.COMPLETED

    def test_task_dependency_enforcement(self, tmp_path):
        from pipeline.agent_coordinator import AgentRole

        coord = self._make_coordinator(tmp_path)
        t1 = coord.assign_task(
            role=AgentRole.DATA_AGENT,
            description="Build dataset",
        )
        t2 = coord.assign_task(
            role=AgentRole.FEATURE_AGENT,
            description="Generate features",
            depends_on=[t1.task_id],
        )
        # t2 depends on t1 — cannot start while t1 is pending
        with pytest.raises(ValueError):
            coord.start_task(t2.task_id)

        # Complete t1, then t2 should be startable
        coord.start_task(t1.task_id)
        coord.complete_task(t1.task_id)
        coord.start_task(t2.task_id)  # should not raise

    def test_get_ready_tasks(self, tmp_path):
        from pipeline.agent_coordinator import AgentRole

        coord = self._make_coordinator(tmp_path)
        t1 = coord.assign_task(role=AgentRole.DATA_AGENT, description="t1")
        coord.assign_task(
            role=AgentRole.FEATURE_AGENT,
            description="t2",
            depends_on=[t1.task_id],
        )

        ready = coord.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].task_id == t1.task_id

    def test_list_tasks_by_role(self, tmp_path):
        from pipeline.agent_coordinator import AgentRole

        coord = self._make_coordinator(tmp_path)
        coord.assign_task(role=AgentRole.DATA_AGENT, description="d1")
        coord.assign_task(role=AgentRole.DATA_AGENT, description="d2")
        coord.assign_task(role=AgentRole.MODEL_AGENT, description="m1")

        data_tasks = coord.list_tasks(role=AgentRole.DATA_AGENT)
        assert len(data_tasks) == 2

    def test_research_cycle(self, tmp_path):
        coord = self._make_coordinator(tmp_path)
        assert coord.cycle_count == 0
        assert coord.start_research_cycle() == 1
        assert coord.start_research_cycle() == 2

    def test_persistence(self, tmp_path):
        from pipeline.agent_coordinator import AgentCoordinator, AgentRole

        path = tmp_path / "coord.json"
        c1 = AgentCoordinator(storage_path=path)
        c1.set_roadmap(problem_id="persist", objective="test")
        t = c1.assign_task(role=AgentRole.DATA_AGENT, description="task")
        c1.start_research_cycle()

        c2 = AgentCoordinator(storage_path=path)
        assert c2.get_roadmap().problem_id == "persist"
        assert c2.get_task(t.task_id) is not None
        assert c2.cycle_count == 1

    def test_export_state(self, tmp_path):
        from pipeline.agent_coordinator import AgentRole

        coord = self._make_coordinator(tmp_path)
        coord.set_roadmap(problem_id="test", objective="test")
        coord.assign_task(role=AgentRole.DATA_AGENT, description="t1")
        state = coord.export_state()
        assert "agents" in state
        assert state["tasks"]["total"] == 1
        assert state["tasks"]["pending"] == 1


# ---------------------------------------------------------------------------
# Conflict Resolution (Section 22)
# ---------------------------------------------------------------------------


class TestConflictResolution:

    def _make_resolver(self, tmp_path: Path):
        from pipeline.conflict_resolution import ConflictResolver

        return ConflictResolver(storage_path=tmp_path / "conflicts.json")

    def test_raise_conflict(self, tmp_path):
        from pipeline.conflict_resolution import ConflictCategory, ConflictStatus

        resolver = self._make_resolver(tmp_path)
        conflict = resolver.raise_conflict(
            category=ConflictCategory.FACTUAL,
            description="Model vs Audit disagreement",
            agents_involved=["model_agent", "audit_agent"],
        )
        assert conflict.status == ConflictStatus.OPEN
        assert len(conflict.agents_involved) == 2

    def test_evidence_duel(self, tmp_path):
        from pipeline.conflict_resolution import ConflictCategory, ConflictStatus

        resolver = self._make_resolver(tmp_path)
        conflict = resolver.raise_conflict(
            category=ConflictCategory.FACTUAL,
            description="Brier score dispute",
            agents_involved=["model_agent", "audit_agent"],
        )
        resolver.submit_evidence(
            conflict.conflict_id,
            agent="model_agent",
            claim="3% improvement is real",
            evidence={"brier_delta": -0.03},
        )
        resolver.submit_evidence(
            conflict.conflict_id,
            agent="audit_agent",
            claim="No improvement after leakage fix",
            evidence={"brier_delta": 0.0},
        )
        updated = resolver.get_conflict(conflict.conflict_id)
        assert updated.status == ConflictStatus.EVIDENCE_DUEL
        assert len(updated.evidence_submissions) == 2

    def test_audit_arbitration_resolves_factual(self, tmp_path):
        from pipeline.conflict_resolution import ConflictCategory, ConflictStatus

        resolver = self._make_resolver(tmp_path)
        conflict = resolver.raise_conflict(
            category=ConflictCategory.FACTUAL,
            description="Factual disagreement",
            agents_involved=["model_agent", "audit_agent"],
        )
        resolver.audit_arbitrate(
            conflict.conflict_id,
            result="No improvement confirmed",
        )
        updated = resolver.get_conflict(conflict.conflict_id)
        assert updated.status == ConflictStatus.RESOLVED
        assert updated.resolved_by == "audit_agent"

    def test_audit_arbitration_resolves_safety(self, tmp_path):
        from pipeline.conflict_resolution import ConflictCategory, ConflictStatus

        resolver = self._make_resolver(tmp_path)
        conflict = resolver.raise_conflict(
            category=ConflictCategory.SAFETY,
            description="Temporal leakage dispute",
            agents_involved=["feature_agent", "audit_agent"],
        )
        resolver.audit_arbitrate(
            conflict.conflict_id,
            result="Leakage confirmed in feature F7",
        )
        updated = resolver.get_conflict(conflict.conflict_id)
        assert updated.status == ConflictStatus.RESOLVED

    def test_orchestrator_decides_priority(self, tmp_path):
        from pipeline.conflict_resolution import ConflictCategory, ConflictStatus

        resolver = self._make_resolver(tmp_path)
        conflict = resolver.raise_conflict(
            category=ConflictCategory.PRIORITY,
            description="Feature expansion vs stabilization",
            agents_involved=["feature_agent", "research_orchestrator"],
        )
        resolver.orchestrator_decide(
            conflict.conflict_id,
            decision="Stabilize first",
            justification="Current feature set needs validation before expansion",
        )
        updated = resolver.get_conflict(conflict.conflict_id)
        assert updated.status == ConflictStatus.RESOLVED
        assert updated.resolved_by == "research_orchestrator"

    def test_orchestrator_cannot_override_safety(self, tmp_path):
        from pipeline.conflict_resolution import ConflictCategory

        resolver = self._make_resolver(tmp_path)
        conflict = resolver.raise_conflict(
            category=ConflictCategory.SAFETY,
            description="Safety issue",
            agents_involved=["audit_agent"],
        )
        with pytest.raises(ValueError, match="safety"):
            resolver.orchestrator_decide(
                conflict.conflict_id,
                decision="Override",
                justification="I want to",
            )

    def test_audit_veto(self, tmp_path):
        from pipeline.conflict_resolution import ConflictCategory, ConflictStatus

        resolver = self._make_resolver(tmp_path)
        conflict = resolver.raise_conflict(
            category=ConflictCategory.SAFETY,
            description="Validation contamination",
            agents_involved=["model_agent", "audit_agent"],
        )
        resolver.audit_veto(
            conflict.conflict_id,
            reason="Confirmed validation contamination — test data leaked into training",
        )
        updated = resolver.get_conflict(conflict.conflict_id)
        assert updated.status == ConflictStatus.RESOLVED
        assert "VETO" in updated.resolution

    def test_audit_veto_only_for_safety(self, tmp_path):
        from pipeline.conflict_resolution import ConflictCategory

        resolver = self._make_resolver(tmp_path)
        conflict = resolver.raise_conflict(
            category=ConflictCategory.RESOURCE,
            description="Compute allocation dispute",
            agents_involved=["model_agent", "ensemble_agent"],
        )
        with pytest.raises(ValueError, match="safety"):
            resolver.audit_veto(conflict.conflict_id, reason="Not safety")

    def test_human_escalation(self, tmp_path):
        from pipeline.conflict_resolution import ConflictCategory, ConflictStatus

        resolver = self._make_resolver(tmp_path)
        conflict = resolver.raise_conflict(
            category=ConflictCategory.SAFETY,
            description="Regulatory compliance issue",
            agents_involved=["audit_agent", "decision_agent"],
        )
        resolver.escalate_to_human(conflict.conflict_id, reason="Regulatory")
        updated = resolver.get_conflict(conflict.conflict_id)
        assert updated.status == ConflictStatus.ESCALATED_TO_HUMAN

        resolver.resolve_human_escalation(
            conflict.conflict_id,
            decision="Proceed with additional safeguards",
            decided_by="compliance_officer",
        )
        updated = resolver.get_conflict(conflict.conflict_id)
        assert updated.status == ConflictStatus.RESOLVED

    def test_file_and_review_dissent(self, tmp_path):
        from pipeline.conflict_resolution import ConflictCategory

        resolver = self._make_resolver(tmp_path)
        conflict = resolver.raise_conflict(
            category=ConflictCategory.FACTUAL,
            description="Disagreement",
            agents_involved=["model_agent", "audit_agent"],
        )
        resolver.audit_arbitrate(conflict.conflict_id, result="No improvement")

        dissent = resolver.file_dissent(
            conflict.conflict_id,
            agent="model_agent",
            reasoning="Feature uses published data",
            insufficiently_weighted_evidence="Publication timestamps",
        )
        assert len(resolver.get_open_dissents()) == 1

        resolver.review_dissent(dissent.dissent_id, notes="Will revisit next cycle")
        assert len(resolver.get_open_dissents()) == 0

    def test_list_conflicts_by_category(self, tmp_path):
        from pipeline.conflict_resolution import ConflictCategory

        resolver = self._make_resolver(tmp_path)
        resolver.raise_conflict(
            category=ConflictCategory.FACTUAL,
            description="f1",
            agents_involved=["a"],
        )
        resolver.raise_conflict(
            category=ConflictCategory.RESOURCE,
            description="r1",
            agents_involved=["b"],
        )
        factual = resolver.list_conflicts(category=ConflictCategory.FACTUAL)
        assert len(factual) == 1

    def test_persistence(self, tmp_path):
        from pipeline.conflict_resolution import ConflictCategory, ConflictResolver

        path = tmp_path / "conflicts.json"
        r1 = ConflictResolver(storage_path=path)
        c = r1.raise_conflict(
            category=ConflictCategory.FACTUAL,
            description="test",
            agents_involved=["a", "b"],
        )
        r1.file_dissent(c.conflict_id, agent="a", reasoning="disagree")

        r2 = ConflictResolver(storage_path=path)
        assert r2.get_conflict(c.conflict_id) is not None
        assert len(r2.get_open_dissents()) == 1

    def test_export_summary(self, tmp_path):
        from pipeline.conflict_resolution import ConflictCategory

        resolver = self._make_resolver(tmp_path)
        resolver.raise_conflict(
            category=ConflictCategory.FACTUAL,
            description="test",
            agents_involved=["a"],
        )
        summary = resolver.export_summary()
        assert summary["total_conflicts"] == 1
        assert summary["open"] == 1


# ---------------------------------------------------------------------------
# Report Generators (Sections 4-12)
# ---------------------------------------------------------------------------


class TestReportGenerators:

    def test_problem_summary(self):
        from pipeline.report_generators import generate_problem_summary

        report = generate_problem_summary(
            problem_id="sp500",
            objective="Predict direction",
            prediction_target="next_day_return",
            horizon="1d",
        )
        assert report["report_type"] == "problem_summary"
        assert report["problem_id"] == "sp500"

    def test_objective_verification(self):
        from pipeline.report_generators import generate_objective_verification

        report = generate_objective_verification(
            problem_id="sp500",
            decision_metric="sharpe",
            decision_metric_value=1.5,
        )
        assert report["verified"] is True

    def test_constraints_register(self):
        from pipeline.report_generators import generate_constraints_register

        report = generate_constraints_register(
            problem_id="sp500",
            constraints=[
                {"name": "max_leverage", "type": "risk", "value": "1.0", "source": "config"},
            ],
        )
        assert report["total"] == 1

    def test_availability_matrix(self):
        from pipeline.report_generators import generate_availability_matrix

        report = generate_availability_matrix(
            sources=[{"name": "FRED", "fields": ["GDP", "CPI"], "start_date": "2000-01-01"}],
        )
        assert report["total_sources"] == 1

    def test_feature_catalog(self):
        from pipeline.report_generators import generate_feature_catalog

        report = generate_feature_catalog(
            features=[
                {"name": "rsi_14", "family": "momentum"},
                {"name": "sma_20", "family": "trend"},
            ],
        )
        assert report["total_features"] == 2
        assert "momentum" in report["families"]

    def test_feature_importance_report(self):
        from pipeline.report_generators import generate_feature_importance_report

        report = generate_feature_importance_report(
            importances={"rsi_14": 0.3, "sma_20": 0.2, "volume": 0.5},
        )
        assert list(report["top_10"].keys())[0] == "volume"

    def test_feature_stability_report(self):
        from pipeline.report_generators import generate_feature_stability_report

        report = generate_feature_stability_report(
            stability_scores={"rsi_14": 0.8, "bad_feature": 0.2},
        )
        assert "rsi_14" in report["stable_features"]
        assert "bad_feature" in report["unstable_features"]

    def test_feature_retirement_log(self, tmp_path):
        from pipeline.report_generators import FeatureRetirementLog

        log = FeatureRetirementLog(storage_path=tmp_path / "retirement.json")
        log.retire("old_feature", reason="Unstable importance")
        export = log.export()
        assert export["total_retired"] == 1

    def test_meta_learning_report(self):
        from pipeline.report_generators import generate_meta_learning_report

        experiments = [
            {"model_family": "lgbm", "problem_id": "p1", "primary_metric_value": 1.5},
            {"model_family": "lr", "problem_id": "p1", "primary_metric_value": 0.8},
            {"model_family": "lgbm", "problem_id": "p1", "primary_metric_value": 1.2},
        ]
        report = generate_meta_learning_report(experiments)
        assert report["best_by_problem"]["p1"]["best_family"] == "lgbm"

    def test_probability_diagnostics(self):
        from pipeline.report_generators import generate_probability_diagnostics

        rng = np.random.default_rng(42)
        y_true = pd.Series(rng.integers(0, 2, 200).astype(float))
        y_prob = pd.Series(rng.uniform(0, 1, 200))
        report = generate_probability_diagnostics(y_true, y_prob)
        assert "brier_score" in report
        assert "ece" in report
        assert "brier_decomposition" in report

    def test_threshold_sweep(self):
        from pipeline.report_generators import generate_threshold_sweep

        rng = np.random.default_rng(42)
        y_true = pd.Series(rng.integers(0, 2, 100).astype(float))
        y_score = pd.Series(rng.uniform(0, 1, 100))
        report = generate_threshold_sweep(y_true, y_score)
        assert len(report["results"]) > 0
        assert "precision" in report["results"][0]

    def test_abstention_report(self):
        from pipeline.report_generators import generate_abstention_report

        rng = np.random.default_rng(42)
        y_true = pd.Series(rng.integers(0, 2, 100).astype(float))
        y_score = pd.Series(rng.uniform(0, 1, 100))
        report = generate_abstention_report(y_true, y_score)
        assert len(report["results"]) > 0

    def test_simulation_assumptions(self):
        from pipeline.report_generators import generate_simulation_assumptions

        report = generate_simulation_assumptions(spread_bps=15.0)
        assert report["spread_bps"] == 15.0

    def test_risk_path_report(self):
        from pipeline.report_generators import generate_risk_path_report

        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0.001, 0.02, 252))
        report = generate_risk_path_report(returns)
        assert "max_drawdown" in report
        assert "max_losing_streak" in report

    def test_robustness_report(self):
        from pipeline.report_generators import generate_robustness_report

        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0.001, 0.02, 252))
        report = generate_robustness_report(
            returns=returns,
            sharpe=1.5,
            n_obs=252,
        )
        assert "deflated_sharpe_probability" in report

    def test_reproducibility_report(self):
        from pipeline.report_generators import generate_reproducibility_report

        experiments = [
            {
                "reproducibility_hash": "abc123",
                "dataset_version": "v1",
                "hyperparameters": {"lr": 0.1},
            },
            {
                "reproducibility_hash": "def456",
                "dataset_version": "v1",
                "hyperparameters": {"lr": 0.2},
            },
        ]
        report = generate_reproducibility_report(experiments)
        assert report["total_experiments"] == 2
        assert report["pct_captured"] == 1.0

    def test_architecture_review(self):
        from pipeline.report_generators import generate_architecture_review

        report = generate_architecture_review(
            modules=[{"name": "pipeline.extract", "type": "package"}],
            issues=[{"location": "db.py", "severity": "low", "description": "Unused import"}],
        )
        assert report["total_modules"] == 1
        assert report["total_issues"] == 1


# ---------------------------------------------------------------------------
# Evaluation Matrix (Section 13)
# ---------------------------------------------------------------------------


class TestEvaluationMatrix:

    def test_evaluate_with_returns(self):
        from pipeline.evaluation_matrix import EvaluationMatrix

        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0.001, 0.02, 252))
        matrix = EvaluationMatrix()
        entry = matrix.evaluate(candidate_id="lgbm_v1", returns=returns)
        assert entry.decision_utility.get("sharpe") is not None
        assert entry.risk.get("max_drawdown") is not None

    def test_evaluate_with_predictions(self):
        from pipeline.evaluation_matrix import EvaluationMatrix

        rng = np.random.default_rng(42)
        y_true = pd.Series(rng.integers(0, 2, 200).astype(float))
        y_prob = pd.Series(rng.uniform(0, 1, 200))
        y_pred = (y_prob > 0.5).astype(float)
        matrix = EvaluationMatrix()
        entry = matrix.evaluate(
            candidate_id="model_a",
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
        )
        assert "brier" in entry.predictive_accuracy
        assert "ece" in entry.calibration

    def test_compare_multiple(self):
        from pipeline.evaluation_matrix import EvaluationMatrix

        rng = np.random.default_rng(42)
        matrix = EvaluationMatrix()
        for i in range(3):
            returns = pd.Series(rng.normal(0.001 * (i + 1), 0.02, 252))
            matrix.evaluate(candidate_id=f"model_{i}", returns=returns)
        comparison = matrix.compare()
        assert comparison["total_candidates"] == 3

    def test_get_best(self):
        from pipeline.evaluation_matrix import EvaluationMatrix

        rng = np.random.default_rng(42)
        matrix = EvaluationMatrix()
        matrix.evaluate(
            candidate_id="weak",
            returns=pd.Series(rng.normal(-0.001, 0.02, 252)),
        )
        matrix.evaluate(
            candidate_id="strong",
            returns=pd.Series(rng.normal(0.005, 0.02, 252)),
        )
        best = matrix.get_best("decision_utility.sharpe")
        assert best == "strong"


# ---------------------------------------------------------------------------
# A/B Testing (Section 18.5)
# ---------------------------------------------------------------------------


class TestABTesting:

    def test_power_analysis(self):
        from pipeline.ab_testing import PowerAnalysis

        n = PowerAnalysis.compute_sample_size(effect_size=0.5, power=0.80)
        assert n > 0
        assert n < 200  # ~63 for d=0.5

    def test_power_computation(self):
        from pipeline.ab_testing import PowerAnalysis

        power = PowerAnalysis.compute_power(n=100, effect_size=0.5)
        assert power > 0.8

    def test_sequential_boundaries(self):
        from pipeline.ab_testing import SequentialTestBoundary

        boundary = SequentialTestBoundary(n_looks=4, alpha=0.05)
        bounds = boundary.get_boundaries()
        assert len(bounds) == 4
        # Early looks should have stricter boundaries
        assert bounds[0]["z_boundary"] > bounds[-1]["z_boundary"]

    def test_sequential_should_stop(self):
        from pipeline.ab_testing import SequentialTestBoundary

        boundary = SequentialTestBoundary(n_looks=4, alpha=0.05)
        # Very large z should trigger stop
        assert boundary.should_stop(look=4, z_statistic=5.0)
        # Small z at first look should not
        assert not boundary.should_stop(look=1, z_statistic=1.0)

    def test_design_test(self, tmp_path):
        from pipeline.ab_testing import ABTestManager

        manager = ABTestManager(storage_path=tmp_path / "ab.json")
        config = manager.design_test(
            candidate_id="v3",
            incumbent_id="v2",
            effect_size=0.3,
        )
        assert config.minimum_sample_size > 0
        assert config.candidate_id == "v3"

    def test_test_lifecycle(self, tmp_path):
        from pipeline.ab_testing import ABTestManager

        manager = ABTestManager(storage_path=tmp_path / "ab.json")
        config = manager.design_test(
            candidate_id="v3",
            incumbent_id="v2",
            effect_size=0.5,
        )
        manager.start_test(config.test_id)

        rng = np.random.default_rng(42)
        for i in range(50):
            manager.record_observation(
                config.test_id,
                group="candidate",
                primary_value=rng.normal(0.05, 0.1),
                cycle=i,
            )
            manager.record_observation(
                config.test_id,
                group="incumbent",
                primary_value=rng.normal(0.0, 0.1),
                cycle=i,
            )

        result = manager.complete_test(config.test_id)
        assert result.winner in ("candidate", "incumbent", "inconclusive")
        assert result.p_value >= 0

    def test_export_protocol(self, tmp_path):
        from pipeline.ab_testing import ABTestManager

        manager = ABTestManager(storage_path=tmp_path / "ab.json")
        config = manager.design_test(
            candidate_id="v3",
            incumbent_id="v2",
            effect_size=0.3,
        )
        protocol = manager.export_protocol(config.test_id)
        assert protocol["report_type"] == "ab_test_protocol"
        assert "boundaries" in protocol

    def test_persistence(self, tmp_path):
        from pipeline.ab_testing import ABTestManager

        path = tmp_path / "ab.json"
        m1 = ABTestManager(storage_path=path)
        config = m1.design_test(candidate_id="v3", incumbent_id="v2")

        m2 = ABTestManager(storage_path=path)
        assert m2.get_test(config.test_id) is not None


# ---------------------------------------------------------------------------
# Knowledge Store (Section 14)
# ---------------------------------------------------------------------------


class TestKnowledgeStore:

    def _make_store(self, tmp_path):
        from pipeline.experiment_registry import KnowledgeStore

        return KnowledgeStore(storage_path=tmp_path / "knowledge.json")

    def test_store_and_query(self, tmp_path):
        store = self._make_store(tmp_path)
        store.store_finding(
            domain="finance",
            horizon="daily",
            model_family="lgbm",
            finding="LightGBM outperforms linear models on daily data",
            works=True,
        )
        store.store_finding(
            domain="finance",
            horizon="daily",
            model_family="lstm",
            finding="LSTM overfits on small datasets",
            works=False,
        )
        results = store.query_findings(domain="finance")
        assert len(results) == 2
        working = store.query_findings(domain="finance", works=True)
        assert len(working) == 1

    def test_persistence(self, tmp_path):
        from pipeline.experiment_registry import KnowledgeStore

        path = tmp_path / "knowledge.json"
        s1 = KnowledgeStore(storage_path=path)
        s1.store_finding(domain="test", finding="Something works")

        s2 = KnowledgeStore(storage_path=path)
        assert len(s2.query_findings()) == 1

    def test_meta_learning_insights(self, tmp_path):
        store = self._make_store(tmp_path)
        store.store_finding(domain="finance", finding="A works", works=True)
        store.store_finding(domain="finance", finding="B fails", works=False)
        insights = store.generate_meta_learning_insights()
        assert insights["report_type"] == "knowledge_retention_report"
        assert insights["domains"]["finance"]["works"] == 1


# ---------------------------------------------------------------------------
# Failure Mode Checks (Section 25.1)
# ---------------------------------------------------------------------------


class TestFailureModeChecks:

    def test_all_pass(self):
        from pipeline.failure_mode_checks import FailureModeChecker

        checker = FailureModeChecker()
        report = checker.run_all_checks(
            deployment_history=["shadow", "canary", "graduated_25", "production"],
        )
        assert report.all_passed
        assert len(report.checks) == 8

    def test_shadow_bypass_fails(self):
        from pipeline.failure_mode_checks import FailureModeChecker

        checker = FailureModeChecker()
        report = checker.run_all_checks(
            deployment_history=["production"],  # skipped shadow/canary
        )
        assert not report.all_passed
        assert "shadow_bypass" in report.failed_checks

    def test_shadow_bypass_with_approval(self):
        from pipeline.failure_mode_checks import FailureModeChecker

        checker = FailureModeChecker()
        report = checker.run_all_checks(
            deployment_history=["production"],
            human_approved_bypass=True,
        )
        # shadow_bypass should pass with human approval
        shadow_check = [c for c in report.checks if c.check_name == "shadow_bypass"][0]
        assert shadow_check.passed

    def test_monitoring_failure(self):
        from pipeline.failure_mode_checks import FailureModeChecker

        check = FailureModeChecker.check_monitoring_active(False)
        assert not check.passed

    def test_budget_overrun(self):
        from pipeline.failure_mode_checks import FailureModeChecker

        check = FailureModeChecker.check_budget_overrun(
            budget_exceeded=True,
            overrun_approved=False,
        )
        assert not check.passed
        check2 = FailureModeChecker.check_budget_overrun(
            budget_exceeded=True,
            overrun_approved=True,
        )
        assert check2.passed

    def test_multiple_failures(self):
        from pipeline.failure_mode_checks import FailureModeChecker

        checker = FailureModeChecker()
        report = checker.run_all_checks(
            deployment_history=[],
            monitoring_active=False,
            ci_passed=False,
        )
        assert not report.all_passed
        assert len(report.failed_checks) >= 3


# ---------------------------------------------------------------------------
# Domain Checklists (Section 24.5)
# ---------------------------------------------------------------------------


class TestDomainChecklists:

    def test_risk_register(self):
        from pipeline.domain_checklist import generate_domain_risk_register

        report = generate_domain_risk_register("finance")
        assert report["total_risks"] > 0
        assert report["high_severity"] > 0

    def test_data_quirks(self):
        from pipeline.domain_checklist import generate_domain_data_quirks

        report = generate_domain_data_quirks("finance")
        assert report["total_quirks"] > 0

    def test_regulatory_checklist(self):
        from pipeline.domain_checklist import generate_regulatory_checklist

        report = generate_regulatory_checklist("finance")
        assert report["total_requirements"] > 0
        assert report["compliant"] > 0

    def test_unknown_domain(self):
        from pipeline.domain_checklist import generate_domain_risk_register

        report = generate_domain_risk_register("unknown")
        assert report["total_risks"] == 0


# ---------------------------------------------------------------------------
# Deployment Pipeline exports (Section 18.6)
# ---------------------------------------------------------------------------


class TestDeploymentPipelineExports:

    def test_drift_report(self, tmp_path):
        from pipeline.deployment_pipeline import DeploymentPipeline

        dp = DeploymentPipeline(storage_path=tmp_path / "deploy.json")
        dp.start_shadow("exp_1")
        report = dp.export_drift_report("exp_1", drift_results={"concept": False})
        assert report["report_type"] == "drift_detection_report"

    def test_retraining_log(self, tmp_path):
        from pipeline.deployment_pipeline import DeploymentPipeline

        dp = DeploymentPipeline(storage_path=tmp_path / "deploy.json")
        dp.start_shadow("exp_1")
        report = dp.export_retraining_log()
        assert report["report_type"] == "retraining_trigger_log"
        assert len(report["configured_triggers"]) == 4


# ---------------------------------------------------------------------------
# Governance exports (Section 21.5)
# ---------------------------------------------------------------------------


class TestGovernanceExports:

    def test_approval_request_log(self, tmp_path):
        from pipeline.governance import GovernanceFramework

        gov = GovernanceFramework(storage_path=tmp_path / "gov.json")
        gov.submit_approval_request(action_summary="Deploy model v3")
        log = gov.export_approval_request_log()
        assert log["report_type"] == "approval_request_log"
        assert log["total_requests"] == 1

    def test_escalation_protocol(self, tmp_path):
        from pipeline.governance import GovernanceFramework

        gov = GovernanceFramework(storage_path=tmp_path / "gov.json")
        protocol = gov.export_escalation_protocol()
        assert protocol["report_type"] == "escalation_protocol"
        assert len(protocol["authority_levels"]) == 3
        assert len(protocol["escalation_path"]) > 0


# ---------------------------------------------------------------------------
# Compute Budget exports (Section 20.4)
# ---------------------------------------------------------------------------


class TestComputeBudgetExports:

    def test_pareto_frontier(self, tmp_path):
        from pipeline.compute_budget import ComputeBudget

        budget = ComputeBudget(total_budget_hours=1.0, storage_path=tmp_path / "budget.json")
        for i in range(5):
            with budget.track_experiment(f"exp_{i}", phase="model_search") as t:
                t.primary_metric_value = 1.0 + i * 0.1
        frontier = budget.generate_pareto_frontier()
        assert frontier["report_type"] == "pareto_frontier_analysis"
        assert len(frontier["frontier_points"]) > 0

    def test_search_termination_justification(self, tmp_path):
        from pipeline.compute_budget import ComputeBudget

        budget = ComputeBudget(total_budget_hours=1.0, storage_path=tmp_path / "budget.json")
        justification = budget.export_search_termination_justification()
        assert justification["report_type"] == "search_termination_justification"
