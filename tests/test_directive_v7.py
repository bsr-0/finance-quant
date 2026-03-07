"""Tests for Agent Directive V7 modules.

Covers:
- Experiment Registry (Section 3)
- Drift Detection (Section 18.3)
- Deployment Pipeline (Section 18.1)
- Compute Budget (Section 20)
- Governance Framework (Section 21)
"""

from __future__ import annotations

import json
import tempfile
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
        for i in range(15):
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
        target = pd.Series(
            rng.normal(0, 1, n), index=features.index, name="target"
        )
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
        cur_target = pd.Series(
            rng.normal(0, 1, 200), index=cur_feat.index
        )

        detector = DriftDetector(ref_feat, ref_target)
        report = detector.run_all_checks(cur_feat, cur_target)
        triggered = [r.axis.value for r in report.results if r.triggered]
        assert "data_drift" in triggered

    def test_label_drift_detected(self):
        from pipeline.drift_detection import DriftDetector
        ref_feat, ref_target = self._make_data(200, seed=42)
        cur_feat, _ = self._make_data(200, seed=43)
        # Massively shifted target
        cur_target = pd.Series(
            np.ones(200) * 10, index=cur_feat.index
        )

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
        cur_target = pd.Series(
            np.ones(200) * 10, index=cur_feat.index
        )

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
        fired = dp.check_rollback_triggers(
            "exp_1", {"max_drawdown": -0.20}
        )
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
