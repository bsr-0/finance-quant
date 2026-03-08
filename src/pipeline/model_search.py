"""Model Search and Meta-Learning (Agent Directive V7 — Section 7).

Searches across diverse model families with temporal hyperparameter tuning.
Integrates with the existing walk-forward validation, experiment registry,
compute budget, and knowledge store infrastructure.

Supported model families:
- Core (scikit-learn): ridge, lasso, logistic, random_forest, gradient_boosting
- Optional: lightgbm, xgboost (install via ``pip install market-data-warehouse[ml]``)
"""

from __future__ import annotations

import itertools
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from pipeline.backtesting.walk_forward import ValidationResult, walk_forward_validate
from pipeline.compute_budget import ComputeBudget
from pipeline.experiment_registry import ExperimentRegistry, KnowledgeStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ModelSpec:
    """Immutable description of a model configuration."""

    model_family: str
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    feature_cols: list[str] | None = None
    objective: str = "mse"
    train_window: int | None = None


@dataclass
class SearchSpace:
    """Defines a hyperparameter search space for one model family."""

    model_family: str
    param_grid: dict[str, list[Any]] = field(default_factory=dict)
    train_windows: list[int] | None = None
    objectives: list[str] | None = None


@dataclass
class SearchResult:
    """Outcome of evaluating a single model configuration."""

    model_spec: ModelSpec
    experiment_id: str
    validation_result: ValidationResult
    primary_metric: str
    primary_metric_value: float
    secondary_metrics: dict[str, float] = field(default_factory=dict)
    compute_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

_SKLEARN_FAMILIES = {
    "ridge",
    "lasso",
    "logistic",
    "random_forest",
    "gradient_boosting",
}

_OPTIONAL_FAMILIES = {"lightgbm", "xgboost"}


def _create_estimator(spec: ModelSpec) -> Any:
    """Instantiate a scikit-learn-compatible estimator from a ModelSpec."""
    family = spec.model_family
    hp = dict(spec.hyperparameters)

    if family == "ridge":
        from sklearn.linear_model import Ridge

        return Ridge(**hp)

    if family == "lasso":
        from sklearn.linear_model import Lasso

        return Lasso(**hp)

    if family == "logistic":
        from sklearn.linear_model import LogisticRegression

        hp.setdefault("max_iter", 1000)
        hp.setdefault("solver", "lbfgs")
        return LogisticRegression(**hp)

    if family == "random_forest":
        hp.setdefault("n_jobs", -1)
        if spec.objective in ("logloss", "classification"):
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(**hp)
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(**hp)

    if family == "gradient_boosting":
        if spec.objective in ("logloss", "classification"):
            from sklearn.ensemble import GradientBoostingClassifier

            return GradientBoostingClassifier(**hp)
        from sklearn.ensemble import GradientBoostingRegressor

        return GradientBoostingRegressor(**hp)

    if family == "lightgbm":
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError(
                "LightGBM is not installed. Install it with: "
                "pip install market-data-warehouse[ml]"
            )
        hp.setdefault("verbosity", -1)
        hp.setdefault("n_jobs", -1)
        if spec.objective in ("logloss", "classification"):
            return lgb.LGBMClassifier(**hp)
        return lgb.LGBMRegressor(**hp)

    if family == "xgboost":
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError(
                "XGBoost is not installed. Install it with: "
                "pip install market-data-warehouse[ml]"
            )
        hp.setdefault("verbosity", 0)
        hp.setdefault("n_jobs", -1)
        if spec.objective in ("logloss", "classification"):
            return xgb.XGBClassifier(**hp)
        return xgb.XGBRegressor(**hp)

    raise ValueError(f"Unknown model family: {family}")


# ---------------------------------------------------------------------------
# Model searcher
# ---------------------------------------------------------------------------


class ModelSearcher:
    """Orchestrates model search across families and hyperparameter grids.

    Integrates with:
    - ``walk_forward_validate()`` for temporal validation
    - ``ExperimentRegistry`` for experiment tracking
    - ``ComputeBudget`` for resource management
    - ``KnowledgeStore`` for meta-learning

    Usage::

        searcher = ModelSearcher(registry, budget)
        results = searcher.run_search(
            df, target_col="fwd_return_1d",
            spaces=default_equity_search_spaces(),
            eval_fn=my_eval_fn,
        )
    """

    def __init__(
        self,
        registry: ExperimentRegistry,
        budget: ComputeBudget | None = None,
        knowledge_store: KnowledgeStore | None = None,
        problem_id: str = "equity_search",
        primary_metric: str = "sharpe",
        random_seed: int = 42,
    ) -> None:
        self.registry = registry
        self.budget = budget
        self.knowledge_store = knowledge_store
        self.problem_id = problem_id
        self.primary_metric = primary_metric
        self.rng = np.random.RandomState(random_seed)

    def build_model(
        self,
        spec: ModelSpec,
        target_col: str = "target",
    ) -> tuple[Callable[[pd.DataFrame], Any], Callable[[Any, pd.DataFrame], pd.Series]]:
        """Create train_fn and predict_fn callables for walk-forward validation.

        Returns:
            (train_fn, predict_fn) compatible with ``walk_forward_validate()``.
        """
        feature_cols = spec.feature_cols

        def train_fn(train_df: pd.DataFrame) -> Any:
            estimator = _create_estimator(spec)
            cols = feature_cols or [c for c in train_df.columns if c != target_col]
            X = train_df[cols].values
            y = train_df[target_col].values
            estimator.fit(X, y)
            return {"estimator": estimator, "feature_cols": cols}

        def predict_fn(model_bundle: Any, test_df: pd.DataFrame) -> pd.Series:
            estimator = model_bundle["estimator"]
            cols = model_bundle["feature_cols"]
            X = test_df[cols].values

            if hasattr(estimator, "predict_proba"):
                try:
                    preds = estimator.predict_proba(X)[:, 1]
                except (IndexError, AttributeError):
                    preds = estimator.predict(X)
            else:
                preds = estimator.predict(X)

            return pd.Series(preds, index=test_df.index, name="prediction")

        return train_fn, predict_fn

    def generate_candidates(
        self,
        spaces: list[SearchSpace],
        max_per_family: int = 20,
    ) -> list[ModelSpec]:
        """Enumerate model specs from search spaces.

        Large grids are randomly subsampled to ``max_per_family``.
        If the KnowledgeStore has negative findings for a family,
        those candidates are deprioritized (moved to end of list).
        """
        candidates: list[ModelSpec] = []
        deprioritized: list[ModelSpec] = []

        # Check knowledge store for families that don't work
        bad_families: set[str] = set()
        if self.knowledge_store:
            failures = self.knowledge_store.query_findings(works=False)
            bad_families = {f.model_family for f in failures if f.model_family}

        for space in spaces:
            family_specs = self._expand_grid(space)

            if len(family_specs) > max_per_family:
                indices = self.rng.choice(
                    len(family_specs), size=max_per_family, replace=False
                )
                family_specs = [family_specs[i] for i in sorted(indices)]

            if space.model_family in bad_families:
                deprioritized.extend(family_specs)
                logger.info(
                    "Deprioritized %d candidates for %s (known underperformer)",
                    len(family_specs),
                    space.model_family,
                )
            else:
                candidates.extend(family_specs)

        candidates.extend(deprioritized)
        return candidates

    def _expand_grid(self, space: SearchSpace) -> list[ModelSpec]:
        """Expand a SearchSpace into individual ModelSpec instances."""
        if not space.param_grid:
            windows = space.train_windows or [None]
            objectives = space.objectives or ["mse"]
            specs = []
            for w, obj in itertools.product(windows, objectives):
                specs.append(
                    ModelSpec(
                        model_family=space.model_family,
                        hyperparameters={},
                        train_window=w,
                        objective=obj,
                    )
                )
            return specs

        keys = list(space.param_grid.keys())
        values = list(space.param_grid.values())
        windows = space.train_windows or [None]
        objectives = space.objectives or ["mse"]

        specs = []
        for combo in itertools.product(*values):
            hp = dict(zip(keys, combo))
            for w, obj in itertools.product(windows, objectives):
                specs.append(
                    ModelSpec(
                        model_family=space.model_family,
                        hyperparameters=hp,
                        train_window=w,
                        objective=obj,
                    )
                )
        return specs

    def run_search(
        self,
        df: pd.DataFrame,
        target_col: str,
        spaces: list[SearchSpace],
        eval_fn: Callable[[pd.Series, pd.Series], dict[str, float]],
        feature_cols: list[str] | None = None,
        train_size: int = 252,
        test_size: int = 63,
        embargo_size: int = 5,
        expanding: bool = True,
        max_per_family: int = 20,
    ) -> list[SearchResult]:
        """Run model search across all candidate configurations.

        Args:
            df: Feature DataFrame with DatetimeIndex.
            target_col: Target column name.
            spaces: List of search spaces to explore.
            eval_fn: Evaluation function ``(y_true, y_pred) -> {metric: value}``.
            feature_cols: Feature columns (default: all except target_col).
            train_size: Walk-forward training window size.
            test_size: Walk-forward test window size.
            embargo_size: Embargo between train and test.
            expanding: Expanding or rolling window.
            max_per_family: Max candidates per model family.

        Returns:
            List of SearchResult sorted by primary metric (best first).
        """
        candidates = self.generate_candidates(spaces, max_per_family=max_per_family)
        results: list[SearchResult] = []

        logger.info("Starting model search: %d candidates", len(candidates))

        for i, spec in enumerate(candidates):
            # Budget check
            if self.budget and not self.budget.check_budget_available("model_search"):
                logger.warning("Model search budget exhausted after %d experiments", i)
                break

            # Search termination check
            if i > 0 and self.registry.check_search_termination(self.problem_id):
                logger.info("Search terminated: insufficient improvement after %d experiments", i)
                break

            if feature_cols:
                spec.feature_cols = feature_cols

            effective_train_size = spec.train_window or train_size

            # Register experiment
            record = self.registry.create_experiment(
                problem_id=self.problem_id,
                model_family=spec.model_family,
                hyperparameters=spec.hyperparameters,
                dataset_version="current",
                feature_set_id="default" if not spec.feature_cols else str(spec.feature_cols[:3]),
                validation_scheme=f"walk_forward_{'expanding' if expanding else 'rolling'}",
            )

            start_time = time.time()
            try:
                train_fn, predict_fn = self.build_model(spec, target_col=target_col)

                if self.budget:
                    tracker = self.budget.track_experiment(record.experiment_id, "model_search")
                    tracker.__enter__()

                validation_result = walk_forward_validate(
                    df=df,
                    train_fn=train_fn,
                    predict_fn=predict_fn,
                    eval_fn=eval_fn,
                    target_col=target_col,
                    train_size=effective_train_size,
                    test_size=test_size,
                    embargo_size=embargo_size,
                    expanding=expanding,
                )

                elapsed = time.time() - start_time
                mean_metrics = validation_result.mean_metrics
                primary_value = mean_metrics.get(self.primary_metric, float("-inf"))

                if self.budget:
                    tracker.primary_metric_value = primary_value
                    tracker.__exit__(None, None, None)

                self.registry.complete_experiment(
                    experiment_id=record.experiment_id,
                    primary_metric=self.primary_metric,
                    primary_metric_value=primary_value,
                    secondary_metrics=mean_metrics,
                    compute_cost_seconds=elapsed,
                )

                results.append(
                    SearchResult(
                        model_spec=spec,
                        experiment_id=record.experiment_id,
                        validation_result=validation_result,
                        primary_metric=self.primary_metric,
                        primary_metric_value=primary_value,
                        secondary_metrics=mean_metrics,
                        compute_seconds=elapsed,
                    )
                )

                logger.info(
                    "Candidate %d/%d [%s]: %s=%.4f (%.1fs)",
                    i + 1,
                    len(candidates),
                    spec.model_family,
                    self.primary_metric,
                    primary_value,
                    elapsed,
                )

            except Exception as e:
                elapsed = time.time() - start_time
                if self.budget:
                    try:
                        tracker.__exit__(type(e), e, None)
                    except Exception:
                        pass
                self.registry.fail_experiment(record.experiment_id, str(e))
                logger.warning("Candidate %d [%s] failed: %s", i + 1, spec.model_family, e)

        results.sort(key=lambda r: r.primary_metric_value, reverse=True)
        return results

    def update_meta_knowledge(
        self,
        results: list[SearchResult],
        domain: str = "finance",
        horizon: str = "daily",
    ) -> None:
        """Store findings in KnowledgeStore for future search cycles."""
        if not self.knowledge_store or not results:
            return

        # Group results by model family
        by_family: dict[str, list[SearchResult]] = {}
        for r in results:
            by_family.setdefault(r.model_spec.model_family, []).append(r)

        for family, family_results in by_family.items():
            best = max(family_results, key=lambda r: r.primary_metric_value)
            worst = min(family_results, key=lambda r: r.primary_metric_value)
            mean_val = np.mean([r.primary_metric_value for r in family_results])

            works = best.primary_metric_value > 0

            self.knowledge_store.store_finding(
                domain=domain,
                horizon=horizon,
                model_family=family,
                finding=(
                    f"{family}: best {self.primary_metric}={best.primary_metric_value:.4f}, "
                    f"mean={mean_val:.4f}, n_configs={len(family_results)}"
                ),
                evidence={
                    "best_metric": best.primary_metric_value,
                    "worst_metric": worst.primary_metric_value,
                    "mean_metric": float(mean_val),
                    "best_hyperparameters": best.model_spec.hyperparameters,
                    "n_configs_tested": len(family_results),
                },
                experiment_ids=[r.experiment_id for r in family_results],
                works=works,
            )


# ---------------------------------------------------------------------------
# Default search spaces
# ---------------------------------------------------------------------------


def default_equity_search_spaces(task_type: str = "regression") -> list[SearchSpace]:
    """Sensible default search spaces for equity swing trading.

    Args:
        task_type: "regression" for return prediction, "classification" for direction.

    Returns:
        List of SearchSpace covering Ridge, Random Forest, and Gradient Boosting.
    """
    objectives = ["mse"] if task_type == "regression" else ["classification"]

    spaces = [
        SearchSpace(
            model_family="ridge",
            param_grid={"alpha": [0.01, 0.1, 1.0, 10.0]},
            train_windows=[252, 504],
            objectives=objectives,
        ),
        SearchSpace(
            model_family="random_forest",
            param_grid={
                "n_estimators": [100, 300],
                "max_depth": [5, 10, None],
                "min_samples_leaf": [5, 20],
                "random_state": [42],
            },
            train_windows=[252, 504],
            objectives=objectives,
        ),
        SearchSpace(
            model_family="gradient_boosting",
            param_grid={
                "n_estimators": [100, 300],
                "learning_rate": [0.01, 0.1],
                "max_depth": [3, 5],
                "random_state": [42],
            },
            train_windows=[252, 504],
            objectives=objectives,
        ),
    ]

    return spaces
