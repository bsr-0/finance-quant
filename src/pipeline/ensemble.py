"""Ensemble Optimization and Calibration (Agent Directive V7 — Section 8).

Builds ensemble models from individual candidates using weighted averaging,
greedy forward selection, or stacking. Measures ensemble diversity and
compares raw vs calibrated vs ensemble-calibrated outputs.

Integrates with:
- ``walk_forward_validate()`` for temporal validation
- ``ExperimentRegistry`` for experiment tracking
- ``ComputeBudget`` for resource management
- ``EvaluationMatrix`` for standardized comparison
- ``Calibrator`` for probability calibration
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pipeline.backtesting.walk_forward import ValidationResult, walk_forward_validate
from pipeline.calibration import CalibrationMethod, CalibratedModelWrapper
from pipeline.compute_budget import ComputeBudget
from pipeline.experiment_registry import ExperimentRegistry
from pipeline.model_search import ModelSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class EnsembleComponent:
    """A single model in an ensemble."""

    component_id: str
    model_spec: ModelSpec | None = None
    weight: float = 1.0
    train_fn: Callable[[pd.DataFrame], Any] | None = None
    predict_fn: Callable[[Any, pd.DataFrame], pd.Series] | None = None


@dataclass
class DiversityMetrics:
    """Diversity measurements across ensemble components."""

    mean_pairwise_correlation: float
    mean_disagreement_rate: float
    component_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean_pairwise_correlation": self.mean_pairwise_correlation,
            "mean_disagreement_rate": self.mean_disagreement_rate,
            "component_count": self.component_count,
        }


@dataclass
class EnsembleResult:
    """Outcome of ensemble search."""

    method: str
    components: list[EnsembleComponent]
    diversity: DiversityMetrics | None = None
    validation_result: ValidationResult | None = None
    experiment_id: str = ""
    primary_metric_value: float = 0.0
    calibrated: bool = False


# ---------------------------------------------------------------------------
# Ensemble builder
# ---------------------------------------------------------------------------


class EnsembleBuilder:
    """Builds and evaluates ensemble models.

    Supports weighted averaging, greedy forward selection, and stacking.
    Measures diversity and compares calibrated vs uncalibrated variants.

    Usage::

        builder = EnsembleBuilder(registry, budget)
        result = builder.run_ensemble_search(
            df, target_col, search_results, eval_fn,
        )
    """

    def __init__(
        self,
        registry: ExperimentRegistry,
        budget: ComputeBudget | None = None,
        primary_metric: str = "sharpe",
        problem_id: str = "ensemble_search",
    ) -> None:
        self.registry = registry
        self.budget = budget
        self.primary_metric = primary_metric
        self.problem_id = problem_id

    def weighted_average(
        self,
        components: list[EnsembleComponent],
        weights: list[float] | None = None,
        target_col: str = "target",
    ) -> tuple[Callable[[pd.DataFrame], Any], Callable[[Any, pd.DataFrame], pd.Series]]:
        """Create train_fn/predict_fn that averages component predictions.

        Compatible with ``walk_forward_validate()``.
        """
        if weights is None:
            weights = [1.0 / len(components)] * len(components)
        else:
            total = sum(weights)
            if total == 0:
                weights = [1.0 / len(components)] * len(components)
            else:
                weights = [w / total for w in weights]

        component_fns = [
            (c.train_fn, c.predict_fn, w)
            for c, w in zip(components, weights)
            if c.train_fn is not None and c.predict_fn is not None
        ]

        def train_fn(train_df: pd.DataFrame) -> list[Any]:
            return [tfn(train_df) for tfn, _, _ in component_fns]

        def predict_fn(models: list[Any], test_df: pd.DataFrame) -> pd.Series:
            predictions = []
            for (_, pfn, w), model in zip(component_fns, models):
                pred = pfn(model, test_df)
                predictions.append(pred * w)

            combined = predictions[0].copy()
            for p in predictions[1:]:
                combined = combined + p
            return combined

        return train_fn, predict_fn

    def greedy_forward_select(
        self,
        components: list[EnsembleComponent],
        df: pd.DataFrame,
        target_col: str,
        eval_fn: Callable[[pd.Series, pd.Series], dict[str, float]],
        max_components: int = 5,
        train_size: int = 252,
        test_size: int = 63,
        embargo_size: int = 5,
    ) -> list[EnsembleComponent]:
        """Greedy forward selection of ensemble components.

        Starts with the best single model and adds models that improve
        the ensemble metric most.
        """
        if not components:
            return []

        # Score each component individually
        scored: list[tuple[float, int]] = []
        for i, comp in enumerate(components):
            if comp.train_fn is None or comp.predict_fn is None:
                continue
            try:
                result = walk_forward_validate(
                    df, comp.train_fn, comp.predict_fn, eval_fn,
                    target_col, train_size, test_size, embargo_size=embargo_size,
                )
                val = result.mean_metrics.get(self.primary_metric, float("-inf"))
                scored.append((val, i))
            except Exception as e:
                logger.warning("Component %s failed: %s", comp.component_id, e)

        if not scored:
            return []

        scored.sort(reverse=True)
        selected = [components[scored[0][1]]]
        best_score = scored[0][0]
        remaining_indices = [idx for _, idx in scored[1:]]

        while len(selected) < max_components and remaining_indices:
            best_candidate = None
            best_new_score = best_score

            for idx in remaining_indices:
                candidate = components[idx]
                trial = selected + [candidate]
                train_fn, predict_fn = self.weighted_average(trial, target_col=target_col)

                try:
                    result = walk_forward_validate(
                        df, train_fn, predict_fn, eval_fn,
                        target_col, train_size, test_size, embargo_size=embargo_size,
                    )
                    val = result.mean_metrics.get(self.primary_metric, float("-inf"))
                    if val > best_new_score:
                        best_new_score = val
                        best_candidate = idx
                except Exception:
                    continue

            if best_candidate is not None:
                selected.append(components[best_candidate])
                remaining_indices.remove(best_candidate)
                best_score = best_new_score
                logger.info(
                    "Greedy: added component %s (score: %.4f → %.4f)",
                    components[best_candidate].component_id,
                    scored[0][0],
                    best_score,
                )
            else:
                break

        return selected

    def stacking(
        self,
        components: list[EnsembleComponent],
        meta_family: str = "ridge",
        target_col: str = "target",
    ) -> tuple[Callable[[pd.DataFrame], Any], Callable[[Any, pd.DataFrame], pd.Series]]:
        """Two-level stacking ensemble.

        Level 0: Component models produce predictions.
        Level 1: Meta-learner trained on component predictions.

        Temporal safety: the meta-learner training uses component predictions
        on the same training data (not held-out), but the outer walk-forward
        validation ensures the final evaluation is always out-of-sample.
        For stricter temporal safety, use ``CalibratedModelWrapper`` which
        splits the training window.
        """
        component_fns = [
            (c.train_fn, c.predict_fn)
            for c in components
            if c.train_fn is not None and c.predict_fn is not None
        ]

        def train_fn(train_df: pd.DataFrame) -> dict[str, Any]:
            # Train all level-0 models
            level0_models = [tfn(train_df) for tfn, _ in component_fns]

            # Generate level-0 predictions on training data
            # (for meta-learner training)
            meta_features = []
            for (_, pfn), model in zip(component_fns, level0_models):
                pred = pfn(model, train_df)
                meta_features.append(pred.values)

            meta_X = np.column_stack(meta_features)
            meta_y = train_df[target_col].values

            # Train level-1 meta-learner
            if meta_family == "ridge":
                from sklearn.linear_model import Ridge

                meta_model = Ridge(alpha=1.0)
            else:
                from sklearn.linear_model import LinearRegression

                meta_model = LinearRegression()

            meta_model.fit(meta_X, meta_y)

            return {
                "level0_models": level0_models,
                "meta_model": meta_model,
            }

        def predict_fn(model_bundle: dict[str, Any], test_df: pd.DataFrame) -> pd.Series:
            level0_models = model_bundle["level0_models"]
            meta_model = model_bundle["meta_model"]

            meta_features = []
            for (_, pfn), model in zip(component_fns, level0_models):
                pred = pfn(model, test_df)
                meta_features.append(pred.values)

            meta_X = np.column_stack(meta_features)
            preds = meta_model.predict(meta_X)
            return pd.Series(preds, index=test_df.index, name="stacked_prediction")

        return train_fn, predict_fn

    def measure_diversity(
        self,
        predictions: dict[str, pd.Series],
    ) -> DiversityMetrics:
        """Measure diversity across component predictions.

        Computes pairwise correlation and disagreement rate.
        """
        if len(predictions) < 2:
            return DiversityMetrics(
                mean_pairwise_correlation=1.0,
                mean_disagreement_rate=0.0,
                component_count=len(predictions),
            )

        names = list(predictions.keys())
        correlations = []
        disagreements = []

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                p1 = predictions[names[i]]
                p2 = predictions[names[j]]
                aligned_p1, aligned_p2 = p1.align(p2, join="inner")

                # Pairwise correlation
                if len(aligned_p1) > 1:
                    corr = float(aligned_p1.corr(aligned_p2))
                    if np.isfinite(corr):
                        correlations.append(corr)

                # Disagreement rate (sign disagreement for regression)
                if len(aligned_p1) > 0:
                    disagree = float(
                        (np.sign(aligned_p1) != np.sign(aligned_p2)).mean()
                    )
                    disagreements.append(disagree)

        return DiversityMetrics(
            mean_pairwise_correlation=float(np.mean(correlations)) if correlations else 1.0,
            mean_disagreement_rate=float(np.mean(disagreements)) if disagreements else 0.0,
            component_count=len(predictions),
        )

    def run_ensemble_search(
        self,
        df: pd.DataFrame,
        target_col: str,
        components: list[EnsembleComponent],
        eval_fn: Callable[[pd.Series, pd.Series], dict[str, float]],
        methods: list[str] | None = None,
        calibration_method: CalibrationMethod = CalibrationMethod.NONE,
        train_size: int = 252,
        test_size: int = 63,
        embargo_size: int = 5,
    ) -> EnsembleResult:
        """Try multiple ensemble methods and return the best.

        Args:
            df: Feature DataFrame with DatetimeIndex.
            target_col: Target column name.
            components: List of EnsembleComponent with train_fn/predict_fn.
            eval_fn: Evaluation function.
            methods: Ensemble methods to try (default: weighted_average, greedy, stacking).
            calibration_method: Calibration to apply to ensemble output.
            train_size: Walk-forward training window.
            test_size: Walk-forward test window.
            embargo_size: Embargo between train and test.

        Returns:
            Best EnsembleResult.
        """
        if methods is None:
            methods = ["weighted_average", "greedy", "stacking"]

        best_result: EnsembleResult | None = None
        best_score = float("-inf")

        for method in methods:
            if self.budget and not self.budget.check_budget_available("ensemble_calibration"):
                logger.warning("Ensemble budget exhausted")
                break

            record = self.registry.create_experiment(
                problem_id=self.problem_id,
                model_family=f"ensemble_{method}",
                hyperparameters={"method": method, "n_components": len(components)},
                validation_scheme="walk_forward_expanding",
            )

            start_time = time.time()
            try:
                if method == "weighted_average":
                    train_fn, predict_fn = self.weighted_average(
                        components, target_col=target_col
                    )
                elif method == "greedy":
                    selected = self.greedy_forward_select(
                        components, df, target_col, eval_fn,
                        train_size=train_size, test_size=test_size,
                        embargo_size=embargo_size,
                    )
                    if not selected:
                        self.registry.fail_experiment(record.experiment_id, "No viable components")
                        continue
                    train_fn, predict_fn = self.weighted_average(
                        selected, target_col=target_col
                    )
                elif method == "stacking":
                    train_fn, predict_fn = self.stacking(
                        components, target_col=target_col
                    )
                else:
                    self.registry.fail_experiment(record.experiment_id, f"Unknown method: {method}")
                    continue

                # Optionally wrap with calibration
                if calibration_method != CalibrationMethod.NONE:
                    wrapper = CalibratedModelWrapper(
                        train_fn, predict_fn,
                        method=calibration_method,
                        target_col=target_col,
                    )
                    train_fn = wrapper.calibrated_train_fn
                    predict_fn = wrapper.calibrated_predict_fn

                if self.budget:
                    tracker = self.budget.track_experiment(
                        record.experiment_id, "ensemble_calibration"
                    )
                    tracker.__enter__()

                validation_result = walk_forward_validate(
                    df, train_fn, predict_fn, eval_fn,
                    target_col, train_size, test_size,
                    embargo_size=embargo_size,
                )

                elapsed = time.time() - start_time
                mean_metrics = validation_result.mean_metrics
                score = mean_metrics.get(self.primary_metric, float("-inf"))

                if self.budget:
                    tracker.primary_metric_value = score
                    tracker.__exit__(None, None, None)

                self.registry.complete_experiment(
                    experiment_id=record.experiment_id,
                    primary_metric=self.primary_metric,
                    primary_metric_value=score,
                    secondary_metrics=mean_metrics,
                    compute_cost_seconds=elapsed,
                )

                result = EnsembleResult(
                    method=method,
                    components=components,
                    validation_result=validation_result,
                    experiment_id=record.experiment_id,
                    primary_metric_value=score,
                    calibrated=calibration_method != CalibrationMethod.NONE,
                )

                if score > best_score:
                    best_score = score
                    best_result = result

                logger.info(
                    "Ensemble [%s]: %s=%.4f (%.1fs)",
                    method, self.primary_metric, score, elapsed,
                )

            except Exception as e:
                elapsed = time.time() - start_time
                if self.budget:
                    try:
                        tracker.__exit__(type(e), e, None)
                    except Exception:
                        pass
                self.registry.fail_experiment(record.experiment_id, str(e))
                logger.warning("Ensemble [%s] failed: %s", method, e)

        if best_result is None:
            return EnsembleResult(method="none", components=components)

        return best_result

    def compare_raw_vs_calibrated(
        self,
        y_true: pd.Series,
        raw_pred: pd.Series,
        calibrated_pred: pd.Series | None = None,
        ensemble_calibrated_pred: pd.Series | None = None,
    ) -> dict[str, Any]:
        """Compare raw vs calibrated vs ensemble-calibrated using EvaluationMatrix.

        Returns a dict with side-by-side metrics for each variant.
        """
        from pipeline.evaluation_matrix import EvaluationMatrix

        matrix = EvaluationMatrix()

        matrix.evaluate(
            candidate_id="raw",
            y_true=y_true,
            y_pred=raw_pred,
            y_prob=raw_pred if raw_pred.between(0, 1).all() else None,
        )

        if calibrated_pred is not None:
            matrix.evaluate(
                candidate_id="calibrated",
                y_true=y_true,
                y_pred=calibrated_pred,
                y_prob=calibrated_pred if calibrated_pred.between(0, 1).all() else None,
            )

        if ensemble_calibrated_pred is not None:
            matrix.evaluate(
                candidate_id="ensemble_calibrated",
                y_true=y_true,
                y_pred=ensemble_calibrated_pred,
                y_prob=(
                    ensemble_calibrated_pred
                    if ensemble_calibrated_pred.between(0, 1).all()
                    else None
                ),
            )

        return matrix.compare()
