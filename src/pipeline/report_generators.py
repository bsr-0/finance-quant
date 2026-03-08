"""Report generators for Agent Directive V7 required outputs (Sections 4-12).

Each function produces a structured dict matching a directive-mandated
output artifact.  These wrap existing pipeline functionality so the
outputs can be serialized (JSON) or rendered into reports.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Section 4 — Phase 0: Problem Definition outputs
# ---------------------------------------------------------------------------

def generate_problem_summary(
    problem_id: str,
    objective: str,
    prediction_target: str = "",
    horizon: str = "",
    granularity: str = "",
    entity: str = "",
    action_layer: str = "",
    constraints: list[str] | None = None,
) -> dict[str, Any]:
    """<problem_summary> — Section 4 required output."""
    return {
        "report_type": "problem_summary",
        "problem_id": problem_id,
        "objective": objective,
        "prediction_target": prediction_target,
        "horizon": horizon,
        "granularity": granularity,
        "entity": entity,
        "action_layer": action_layer,
        "constraints": constraints or [],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def generate_objective_verification(
    problem_id: str,
    decision_metric: str,
    decision_metric_value: float,
    predictive_metric: str = "",
    predictive_metric_value: float | None = None,
    alignment_notes: str = "",
) -> dict[str, Any]:
    """<objective_verification_report> — Section 4 required output.

    Verifies that the system optimises the real decision objective,
    not merely predictive accuracy.
    """
    return {
        "report_type": "objective_verification_report",
        "problem_id": problem_id,
        "decision_metric": decision_metric,
        "decision_metric_value": decision_metric_value,
        "predictive_metric": predictive_metric,
        "predictive_metric_value": predictive_metric_value,
        "alignment_notes": alignment_notes,
        "verified": decision_metric_value is not None,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def generate_constraints_register(
    problem_id: str,
    constraints: list[dict[str, str]],
) -> dict[str, Any]:
    """<constraints_register> — Section 4 required output.

    Each constraint is a dict with keys: name, type, value, source.
    """
    return {
        "report_type": "constraints_register",
        "problem_id": problem_id,
        "constraints": constraints,
        "total": len(constraints),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Section 5 — Phase 1: Dataset Discovery outputs
# ---------------------------------------------------------------------------

def generate_availability_matrix(
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    """<availability_matrix> — Section 5 required output.

    Each source entry should contain: name, fields, start_date, end_date,
    update_frequency, availability_lag.
    """
    return {
        "report_type": "availability_matrix",
        "sources": sources,
        "total_sources": len(sources),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def generate_dataset_expansion_report(
    current_sources: list[str],
    candidate_sources: list[dict[str, str]] | None = None,
    rejected_sources: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """<dataset_expansion_report> — Section 5 required output."""
    return {
        "report_type": "dataset_expansion_report",
        "current_sources": current_sources,
        "candidate_sources": candidate_sources or [],
        "rejected_sources": rejected_sources or [],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Section 6 — Phase 2: Feature Discovery outputs
# ---------------------------------------------------------------------------

def generate_feature_catalog(
    features: list[dict[str, Any]],
) -> dict[str, Any]:
    """<feature_catalog> — Section 6 required output.

    Each feature entry: name, family, computation, availability_lag,
    production_available, stability_score.
    """
    return {
        "report_type": "feature_catalog",
        "features": features,
        "total_features": len(features),
        "families": list({f.get("family", "unknown") for f in features}),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def generate_feature_importance_report(
    importances: dict[str, float],
    method: str = "model_importance",
    model_family: str = "",
) -> dict[str, Any]:
    """<feature_importance_report> — Section 6 required output."""
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    return {
        "report_type": "feature_importance_report",
        "method": method,
        "model_family": model_family,
        "importances": dict(sorted_imp),
        "top_10": dict(sorted_imp[:10]),
        "total_features": len(importances),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def generate_feature_stability_report(
    stability_scores: dict[str, float],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """<feature_stability_report> — Section 6 required output.

    stability_scores: feature_name → stability metric (e.g. rank
    correlation of importance across walk-forward folds).
    """
    stable = {k: v for k, v in stability_scores.items() if v >= threshold}
    unstable = {k: v for k, v in stability_scores.items() if v < threshold}
    return {
        "report_type": "feature_stability_report",
        "threshold": threshold,
        "stable_features": stable,
        "unstable_features": unstable,
        "pct_stable": len(stable) / max(len(stability_scores), 1),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


@dataclass
class FeatureRetirementEntry:
    feature_name: str
    reason: str
    retired_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    replacement: str = ""


class FeatureRetirementLog:
    """<feature_retirement_log> — Section 6 required output."""

    def __init__(self, storage_path: str | Path = "data/feature_retirement_log.json"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: list[FeatureRetirementEntry] = []
        self._load()

    def _load(self) -> None:
        if self.storage_path.exists():
            with open(self.storage_path) as f:
                for e in json.load(f):
                    self._entries.append(FeatureRetirementEntry(**e))

    def _save(self) -> None:
        with open(self.storage_path, "w") as f:
            json.dump([asdict(e) for e in self._entries], f, indent=2)

    def retire(self, feature_name: str, reason: str, replacement: str = "") -> None:
        entry = FeatureRetirementEntry(
            feature_name=feature_name,
            reason=reason,
            replacement=replacement,
        )
        self._entries.append(entry)
        self._save()

    def list_retired(self) -> list[dict[str, Any]]:
        return [asdict(e) for e in self._entries]

    def export(self) -> dict[str, Any]:
        return {
            "report_type": "feature_retirement_log",
            "entries": self.list_retired(),
            "total_retired": len(self._entries),
        }


# ---------------------------------------------------------------------------
# Section 7 — Phase 3: Model Search outputs
# ---------------------------------------------------------------------------

def generate_meta_learning_report(
    experiments: list[dict[str, Any]],
) -> dict[str, Any]:
    """<meta_learning_report> — Section 7 required output.

    Analyses which model families, feature sets, and validation schemes
    perform best by problem type and data regime.
    """
    if not experiments:
        return {
            "report_type": "meta_learning_report",
            "insights": [],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    by_family: dict[str, list[float]] = {}
    by_problem: dict[str, dict[str, list[float]]] = {}

    for exp in experiments:
        family = exp.get("model_family", "unknown")
        problem = exp.get("problem_id", "unknown")
        val = exp.get("primary_metric_value")
        if val is not None:
            by_family.setdefault(family, []).append(val)
            by_problem.setdefault(problem, {}).setdefault(family, []).append(val)

    insights = []
    for family, values in by_family.items():
        insights.append({
            "model_family": family,
            "n_experiments": len(values),
            "mean_metric": float(np.mean(values)),
            "std_metric": float(np.std(values)) if len(values) > 1 else 0.0,
            "best_metric": float(max(values)),
        })

    best_by_problem = {}
    for problem, families in by_problem.items():
        best_family = max(families.items(), key=lambda x: max(x[1]))
        best_by_problem[problem] = {
            "best_family": best_family[0],
            "best_metric": float(max(best_family[1])),
        }

    return {
        "report_type": "meta_learning_report",
        "insights": sorted(insights, key=lambda x: x["best_metric"], reverse=True),
        "best_by_problem": best_by_problem,
        "total_experiments_analysed": len(experiments),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Section 8 — Phase 4: Ensemble & Calibration outputs
# ---------------------------------------------------------------------------

def generate_probability_diagnostics(
    y_true: pd.Series,
    y_prob: pd.Series,
    bins: int = 10,
) -> dict[str, Any]:
    """<probability_diagnostics> — Section 8 required output.

    Produces calibration diagnostics: reliability curve data, ECE,
    Brier decomposition, log loss by bucket.
    """
    from pipeline.eval.metrics import brier_score, calibration_error, log_loss

    y_true, y_prob = y_true.align(y_prob, join="inner")
    y_true = y_true.dropna()
    y_prob = y_prob.loc[y_true.index].dropna()
    y_true = y_true.loc[y_prob.index]

    # Reliability curve
    df = pd.DataFrame({"y": y_true, "p": y_prob})
    df["bin"] = pd.cut(df["p"], bins=bins, labels=False, include_lowest=True)
    reliability = []
    for _, grp in df.groupby("bin"):
        if grp.empty:
            continue
        reliability.append({
            "mean_predicted": float(grp["p"].mean()),
            "mean_observed": float(grp["y"].mean()),
            "count": len(grp),
        })

    # Brier decomposition: reliability, resolution, uncertainty
    overall_base_rate = float(y_true.mean())
    uncertainty = overall_base_rate * (1 - overall_base_rate)
    rel_term = 0.0
    res_term = 0.0
    n = len(df)
    for bucket in reliability:
        nk = bucket["count"]
        rel_term += nk * (bucket["mean_predicted"] - bucket["mean_observed"]) ** 2
        res_term += nk * (bucket["mean_observed"] - overall_base_rate) ** 2
    rel_term /= n
    res_term /= n

    return {
        "report_type": "probability_diagnostics",
        "brier_score": brier_score(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob),
        "ece": calibration_error(y_true, y_prob, bins=bins),
        "brier_decomposition": {
            "reliability": float(rel_term),
            "resolution": float(res_term),
            "uncertainty": float(uncertainty),
        },
        "reliability_curve": reliability,
        "n_samples": n,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Section 9 — Phase 5: Decision Optimization outputs
# ---------------------------------------------------------------------------

def generate_threshold_sweep(
    y_true: pd.Series,
    y_score: pd.Series,
    thresholds: list[float] | None = None,
) -> dict[str, Any]:
    """<threshold_sweep_report> — Section 9 required output."""
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.1, 1.0, 0.05)]

    y_true, y_score = y_true.align(y_score, join="inner")
    results = []
    for t in thresholds:
        preds = (y_score >= t).astype(int)
        tp = int(((preds == 1) & (y_true == 1)).sum())
        fp = int(((preds == 1) & (y_true == 0)).sum())
        fn = int(((preds == 0) & (y_true == 1)).sum())
        tn = int(((preds == 0) & (y_true == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        n_actions = int(preds.sum())
        results.append({
            "threshold": t,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "n_actions": n_actions,
            "pct_acted": round(n_actions / len(preds), 4) if len(preds) > 0 else 0.0,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })

    return {
        "report_type": "threshold_sweep_report",
        "thresholds_evaluated": len(results),
        "results": results,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def generate_abstention_report(
    y_true: pd.Series,
    y_score: pd.Series,
    confidence_levels: list[float] | None = None,
) -> dict[str, Any]:
    """<abstention_policy_report> — Section 9 required output.

    Evaluates performance when the system abstains on low-confidence predictions.
    """
    if confidence_levels is None:
        confidence_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    y_true, y_score = y_true.align(y_score, join="inner")
    results = []
    for min_conf in confidence_levels:
        # Confidence = distance from 0.5 for binary predictions
        confidence = (y_score - 0.5).abs()
        mask = confidence >= min_conf
        n_acted = int(mask.sum())
        n_abstained = int((~mask).sum())
        if n_acted > 0:
            acted_true = y_true[mask]
            acted_score = y_score[mask]
            acted_preds = (acted_score >= 0.5).astype(int)
            accuracy = float((acted_preds == acted_true).mean())
        else:
            accuracy = float("nan")

        results.append({
            "min_confidence": min_conf,
            "n_acted": n_acted,
            "n_abstained": n_abstained,
            "pct_abstained": round(n_abstained / len(y_true), 4) if len(y_true) > 0 else 0.0,
            "accuracy_on_acted": round(accuracy, 4) if np.isfinite(accuracy) else None,
        })

    return {
        "report_type": "abstention_policy_report",
        "levels_evaluated": len(results),
        "results": results,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Section 10 — Phase 6: Backtesting outputs
# ---------------------------------------------------------------------------

def generate_simulation_assumptions(
    transaction_cost_model: str = "fixed_plus_spread",
    spread_bps: float = 10.0,
    slippage_model: str = "square_root_impact",
    max_leverage: float = 1.0,
    max_position_pct: float = 0.1,
    rebalance_frequency: str = "daily",
    data_latency_hours: float = 0.0,
    additional: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """<simulation_assumptions> — Section 10 required output."""
    return {
        "report_type": "simulation_assumptions",
        "transaction_cost_model": transaction_cost_model,
        "spread_bps": spread_bps,
        "slippage_model": slippage_model,
        "max_leverage": max_leverage,
        "max_position_pct": max_position_pct,
        "rebalance_frequency": rebalance_frequency,
        "data_latency_hours": data_latency_hours,
        "additional": additional or {},
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def generate_risk_path_report(
    returns: pd.Series,
) -> dict[str, Any]:
    """<risk_path_report> — Section 10 required output."""
    from pipeline.eval.metrics import max_drawdown, drawdown_recovery_time, sharpe_sortino

    returns = returns.dropna()
    equity = (1 + returns).cumprod()
    peak = equity.cummax()
    dd_series = (equity - peak) / peak

    sharpe, sortino = sharpe_sortino(returns)

    # Losing streaks
    is_loss = returns < 0
    streaks = []
    current_streak = 0
    for loss in is_loss:
        if loss:
            current_streak += 1
        else:
            if current_streak > 0:
                streaks.append(current_streak)
            current_streak = 0
    if current_streak > 0:
        streaks.append(current_streak)

    return {
        "report_type": "risk_path_report",
        "max_drawdown": max_drawdown(returns),
        "recovery_time_days": drawdown_recovery_time(returns),
        "sharpe": sharpe,
        "sortino": sortino,
        "worst_day": float(returns.min()),
        "best_day": float(returns.max()),
        "max_losing_streak": max(streaks) if streaks else 0,
        "mean_losing_streak": float(np.mean(streaks)) if streaks else 0.0,
        "total_trading_days": len(returns),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Section 11 — Phase 7: Skeptical Audit outputs
# ---------------------------------------------------------------------------

def generate_robustness_report(
    returns: pd.Series,
    sharpe: float,
    n_obs: int,
    skew: float = 0.0,
    excess_kurtosis: float = 0.0,
    bootstrap_n: int = 500,
) -> dict[str, Any]:
    """<robustness_report> — Section 11 required output."""
    from pipeline.eval.robustness import bootstrap_ci, deflated_sharpe_ratio
    from pipeline.eval.metrics import sharpe_sortino

    sharpe_fn = lambda s: sharpe_sortino(s)[0]  # noqa: E731
    ci_lower, ci_upper = bootstrap_ci(returns, sharpe_fn, n_boot=bootstrap_n)
    dsr = deflated_sharpe_ratio(sharpe, n_obs, skew, excess_kurtosis)

    return {
        "report_type": "robustness_report",
        "sharpe": sharpe,
        "deflated_sharpe_probability": dsr,
        "bootstrap_ci_lower": ci_lower,
        "bootstrap_ci_upper": ci_upper,
        "bootstrap_n": bootstrap_n,
        "n_observations": n_obs,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def generate_reproducibility_report(
    experiments: list[dict[str, Any]],
) -> dict[str, Any]:
    """<reproducibility_report> — Section 11 required output."""
    hashes = [e.get("reproducibility_hash", "") for e in experiments]
    unique_hashes = set(h for h in hashes if h)
    duplicates = len(hashes) - len(unique_hashes) if unique_hashes else 0

    configs_captured = sum(
        1 for e in experiments
        if e.get("dataset_version") and e.get("hyperparameters")
    )

    return {
        "report_type": "reproducibility_report",
        "total_experiments": len(experiments),
        "unique_configs": len(unique_hashes),
        "duplicate_configs": duplicates,
        "configs_fully_captured": configs_captured,
        "pct_captured": round(configs_captured / max(len(experiments), 1), 4),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Section 12 — Phase 8: Codebase Review outputs
# ---------------------------------------------------------------------------

def generate_architecture_review(
    modules: list[dict[str, Any]],
    issues: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """<architecture_review_report> — Section 12 required output.

    modules: list of {name, type, dependencies, test_coverage}.
    issues: list of {location, severity, description, recommendation}.
    """
    return {
        "report_type": "architecture_review_report",
        "modules": modules,
        "total_modules": len(modules),
        "issues": issues or [],
        "total_issues": len(issues) if issues else 0,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def generate_refactoring_plan(
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    """<refactoring_plan> — Section 12 required output.

    items: list of {description, priority, impact, blast_radius,
    rollback_safety, pre_refactor_test}.
    """
    return {
        "report_type": "refactoring_plan",
        "items": sorted(items, key=lambda x: x.get("priority", 99)),
        "total_items": len(items),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
