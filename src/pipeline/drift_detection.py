"""Drift Detection Protocol (Agent Directive V7 — Section 18.3).

Monitors model drift along three independent axes:
1. **Concept drift** — relationship between features and target changes.
2. **Data drift (covariate shift)** — feature distributions shift.
3. **Label drift (prior shift)** — target base rate shifts.

Detection on any single axis triggers investigation; detection on
two or more triggers the retraining pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


class DriftAxis(str, Enum):
    CONCEPT = "concept_drift"
    DATA = "data_drift"
    LABEL = "label_drift"


class DriftSeverity(str, Enum):
    NONE = "none"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class DriftResult:
    """Result of a single drift check."""

    axis: DriftAxis
    severity: DriftSeverity
    metric_name: str
    metric_value: float
    threshold: float
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def triggered(self) -> bool:
        return self.severity != DriftSeverity.NONE


@dataclass
class DriftReport:
    """Aggregate drift report across all axes."""

    results: list[DriftResult] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def triggered_axes(self) -> list[DriftAxis]:
        return list({r.axis for r in self.results if r.triggered})

    @property
    def requires_retraining(self) -> bool:
        """Two or more axes firing triggers retraining (Section 18.3)."""
        return len(self.triggered_axes) >= 2

    @property
    def requires_investigation(self) -> bool:
        return len(self.triggered_axes) >= 1

    def summary(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "triggered_axes": [a.value for a in self.triggered_axes],
            "requires_retraining": self.requires_retraining,
            "requires_investigation": self.requires_investigation,
            "results": [
                {
                    "axis": r.axis.value,
                    "severity": r.severity.value,
                    "metric": r.metric_name,
                    "value": round(r.metric_value, 6),
                    "threshold": r.threshold,
                }
                for r in self.results
            ],
        }


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, n_bins: int = 10
) -> float:
    """Compute PSI between reference and current distributions.

    PSI > 0.1 indicates moderate shift; > 0.2 indicates significant shift.
    """
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if len(reference) < n_bins or len(current) < n_bins:
        return 0.0

    breakpoints = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    # Remove duplicate breakpoints
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 3:
        return 0.0

    ref_counts = np.histogram(reference, bins=breakpoints)[0].astype(float)
    cur_counts = np.histogram(current, bins=breakpoints)[0].astype(float)

    # Avoid division by zero
    ref_pct = np.clip(ref_counts / ref_counts.sum(), 1e-6, None)
    cur_pct = np.clip(cur_counts / cur_counts.sum(), 1e-6, None)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


class DriftDetector:
    """Detect drift along three axes per Section 18.3.

    Usage::

        detector = DriftDetector(
            reference_features=train_X,
            reference_target=train_y,
        )
        report = detector.run_all_checks(
            current_features=live_X,
            current_target=live_y,
            current_predictions=live_preds,
        )
        if report.requires_retraining:
            trigger_retraining_pipeline()
    """

    def __init__(
        self,
        reference_features: pd.DataFrame,
        reference_target: pd.Series,
        psi_threshold: float = 0.2,
        ks_pvalue_threshold: float = 0.01,
        ks_feature_pct_threshold: float = 0.15,
        label_shift_threshold: float = 0.20,
        conditional_error_ratio_threshold: float = 1.5,
    ):
        self.reference_features = reference_features
        self.reference_target = reference_target
        self.psi_threshold = psi_threshold
        self.ks_pvalue_threshold = ks_pvalue_threshold
        self.ks_feature_pct_threshold = ks_feature_pct_threshold
        self.label_shift_threshold = label_shift_threshold
        self.conditional_error_ratio_threshold = conditional_error_ratio_threshold

        # Pre-compute reference statistics
        self._ref_base_rate = float(reference_target.mean())
        self._ref_feature_means = reference_features.mean()
        self._ref_feature_stds = reference_features.std()

    def check_data_drift(self, current_features: pd.DataFrame) -> list[DriftResult]:
        """Detect covariate shift via PSI and KS tests (Section 18.3)."""
        results: list[DriftResult] = []
        common_cols = list(
            set(self.reference_features.columns) & set(current_features.columns)
        )

        psi_violations = []
        ks_violations = []

        for col in common_cols:
            ref_vals = self.reference_features[col].dropna().values
            cur_vals = current_features[col].dropna().values
            if len(ref_vals) < 10 or len(cur_vals) < 10:
                continue

            # PSI check
            psi = population_stability_index(ref_vals, cur_vals)
            if psi > self.psi_threshold:
                psi_violations.append({"feature": col, "psi": psi})

            # KS test
            ks_stat, ks_pval = sp_stats.ks_2samp(ref_vals, cur_vals)
            if ks_pval < self.ks_pvalue_threshold:
                ks_violations.append(
                    {"feature": col, "ks_stat": ks_stat, "ks_pvalue": ks_pval}
                )

        # PSI violations
        if psi_violations:
            worst = max(psi_violations, key=lambda x: x["psi"])
            results.append(
                DriftResult(
                    axis=DriftAxis.DATA,
                    severity=DriftSeverity.CRITICAL,
                    metric_name="psi",
                    metric_value=worst["psi"],
                    threshold=self.psi_threshold,
                    details={
                        "violations": psi_violations,
                        "n_features_checked": len(common_cols),
                    },
                )
            )

        # KS violations
        ks_pct = len(ks_violations) / max(len(common_cols), 1)
        if ks_pct > self.ks_feature_pct_threshold:
            results.append(
                DriftResult(
                    axis=DriftAxis.DATA,
                    severity=DriftSeverity.CRITICAL,
                    metric_name="ks_feature_pct",
                    metric_value=ks_pct,
                    threshold=self.ks_feature_pct_threshold,
                    details={
                        "n_ks_violations": len(ks_violations),
                        "n_features_checked": len(common_cols),
                        "violations": ks_violations[:10],
                    },
                )
            )

        if not results:
            results.append(
                DriftResult(
                    axis=DriftAxis.DATA,
                    severity=DriftSeverity.NONE,
                    metric_name="data_drift_check",
                    metric_value=0.0,
                    threshold=self.psi_threshold,
                )
            )

        return results

    def check_label_drift(self, current_target: pd.Series) -> list[DriftResult]:
        """Detect prior shift in target distribution (Section 18.3)."""
        current_base_rate = float(current_target.mean())
        if self._ref_base_rate == 0:
            relative_shift = abs(current_base_rate)
        else:
            relative_shift = abs(current_base_rate - self._ref_base_rate) / abs(
                self._ref_base_rate
            )

        severity = DriftSeverity.NONE
        if relative_shift > self.label_shift_threshold:
            severity = DriftSeverity.CRITICAL

        return [
            DriftResult(
                axis=DriftAxis.LABEL,
                severity=severity,
                metric_name="label_base_rate_shift",
                metric_value=relative_shift,
                threshold=self.label_shift_threshold,
                details={
                    "reference_base_rate": self._ref_base_rate,
                    "current_base_rate": current_base_rate,
                },
            )
        ]

    def check_concept_drift(
        self,
        current_features: pd.DataFrame,
        current_target: pd.Series,
        current_predictions: pd.Series | None = None,
    ) -> list[DriftResult]:
        """Detect concept drift via conditional prediction error (Section 18.3).

        Monitors the relationship between features and target by checking
        whether prediction errors vary significantly across feature segments.
        """
        results: list[DriftResult] = []

        if current_predictions is None:
            results.append(
                DriftResult(
                    axis=DriftAxis.CONCEPT,
                    severity=DriftSeverity.NONE,
                    metric_name="concept_drift_check",
                    metric_value=0.0,
                    threshold=self.conditional_error_ratio_threshold,
                    details={"skipped": "no predictions provided"},
                )
            )
            return results

        # Align all series
        idx = current_features.index.intersection(current_target.index).intersection(
            current_predictions.index
        )
        if len(idx) < 20:
            return results

        errors = (current_target.loc[idx] - current_predictions.loc[idx]).abs()
        overall_error = float(errors.mean())
        if overall_error == 0:
            return results

        # Check conditional error across top features
        common_cols = list(current_features.columns)[:20]
        concept_violations = []

        for col in common_cols:
            feat = current_features.loc[idx, col]
            if feat.nunique() < 3:
                continue
            try:
                median = feat.median()
                below = errors[feat <= median]
                above = errors[feat > median]
                if len(below) < 5 or len(above) < 5:
                    continue
                ratio = max(float(below.mean()), float(above.mean())) / max(
                    overall_error, 1e-10
                )
                if ratio > self.conditional_error_ratio_threshold:
                    concept_violations.append(
                        {"feature": col, "error_ratio": ratio}
                    )
            except (ValueError, ZeroDivisionError):
                continue

        if concept_violations:
            worst = max(concept_violations, key=lambda x: x["error_ratio"])
            results.append(
                DriftResult(
                    axis=DriftAxis.CONCEPT,
                    severity=DriftSeverity.CRITICAL,
                    metric_name="conditional_error_ratio",
                    metric_value=worst["error_ratio"],
                    threshold=self.conditional_error_ratio_threshold,
                    details={"violations": concept_violations},
                )
            )
        else:
            results.append(
                DriftResult(
                    axis=DriftAxis.CONCEPT,
                    severity=DriftSeverity.NONE,
                    metric_name="conditional_error_ratio",
                    metric_value=0.0,
                    threshold=self.conditional_error_ratio_threshold,
                )
            )

        return results

    def run_all_checks(
        self,
        current_features: pd.DataFrame,
        current_target: pd.Series,
        current_predictions: pd.Series | None = None,
    ) -> DriftReport:
        """Run all three drift detection axes and return a report."""
        all_results: list[DriftResult] = []
        all_results.extend(self.check_data_drift(current_features))
        all_results.extend(self.check_label_drift(current_target))
        all_results.extend(
            self.check_concept_drift(
                current_features, current_target, current_predictions
            )
        )

        report = DriftReport(results=all_results)

        if report.requires_retraining:
            logger.critical(
                "DRIFT: %d axes triggered — retraining required: %s",
                len(report.triggered_axes),
                [a.value for a in report.triggered_axes],
            )
        elif report.requires_investigation:
            logger.warning(
                "DRIFT: %d axis triggered — investigation required: %s",
                len(report.triggered_axes),
                [a.value for a in report.triggered_axes],
            )
        else:
            logger.info("DRIFT: no drift detected across all axes")

        return report
