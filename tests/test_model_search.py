"""Tests for model search module (Agent Directive V7 — Section 7)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.experiment_registry import ExperimentRegistry, ExperimentStatus, KnowledgeStore
from pipeline.model_search import (
    ModelSearcher,
    ModelSpec,
    SearchSpace,
    default_equity_search_spaces,
)


def _make_synthetic_data(n: int = 600, n_features: int = 5, seed: int = 42) -> pd.DataFrame:
    """Create synthetic feature data with a noisy linear target."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    features = {f"feat_{i}": rng.randn(n) for i in range(n_features)}
    # Target with weak signal
    X = np.column_stack(list(features.values()))
    beta = rng.randn(n_features) * 0.1
    target = X @ beta + rng.randn(n) * 0.5
    features["target"] = target
    return pd.DataFrame(features, index=dates)


def _simple_eval_fn(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Simple evaluation function for testing."""
    aligned_true, aligned_pred = y_true.align(y_pred, join="inner")
    rmse = float(np.sqrt(((aligned_true - aligned_pred) ** 2).mean()))
    # Simple directional accuracy
    hit = float((np.sign(aligned_true) == np.sign(aligned_pred)).mean())
    # Pseudo-Sharpe for testing
    errors = aligned_true - aligned_pred
    sharpe = float(errors.mean() / (errors.std() + 1e-10)) if len(errors) > 1 else 0.0
    return {"rmse": rmse, "hit_rate": hit, "sharpe": sharpe}


class TestModelSpec:
    """Tests for ModelSpec dataclass."""

    def test_construction(self) -> None:
        spec = ModelSpec(
            model_family="ridge",
            hyperparameters={"alpha": 1.0},
            objective="mse",
        )
        assert spec.model_family == "ridge"
        assert spec.hyperparameters == {"alpha": 1.0}
        assert spec.feature_cols is None
        assert spec.train_window is None

    def test_search_space_grid_expansion(self) -> None:
        space = SearchSpace(
            model_family="ridge",
            param_grid={"alpha": [0.1, 1.0, 10.0]},
            train_windows=[252, 504],
        )
        searcher = ModelSearcher(
            registry=ExperimentRegistry(storage_path="/tmp/test_unused.json"),
        )
        specs = searcher._expand_grid(space)
        # 3 alphas × 2 windows × 1 objective = 6
        assert len(specs) == 6
        families = {s.model_family for s in specs}
        assert families == {"ridge"}


class TestModelSearcher:
    """Tests for ModelSearcher class."""

    def test_build_model_ridge(self, tmp_path: object) -> None:
        registry = ExperimentRegistry(storage_path=f"{tmp_path}/registry.json")
        searcher = ModelSearcher(registry=registry)

        spec = ModelSpec(model_family="ridge", hyperparameters={"alpha": 1.0})
        train_fn, predict_fn = searcher.build_model(spec, target_col="target")

        df = _make_synthetic_data(n=200)
        train_df = df.iloc[:150]
        test_df = df.iloc[150:]

        model = train_fn(train_df)
        preds = predict_fn(model, test_df)

        assert len(preds) == 50
        assert isinstance(preds, pd.Series)
        assert not preds.isna().any()

    def test_build_model_random_forest(self, tmp_path: object) -> None:
        registry = ExperimentRegistry(storage_path=f"{tmp_path}/registry.json")
        searcher = ModelSearcher(registry=registry)

        spec = ModelSpec(
            model_family="random_forest",
            hyperparameters={"n_estimators": 10, "max_depth": 3, "random_state": 42},
        )
        train_fn, predict_fn = searcher.build_model(spec, target_col="target")

        df = _make_synthetic_data(n=200)
        model = train_fn(df.iloc[:150])
        preds = predict_fn(model, df.iloc[150:])

        assert len(preds) == 50
        assert not preds.isna().any()

    def test_build_model_gradient_boosting(self, tmp_path: object) -> None:
        registry = ExperimentRegistry(storage_path=f"{tmp_path}/registry.json")
        searcher = ModelSearcher(registry=registry)

        spec = ModelSpec(
            model_family="gradient_boosting",
            hyperparameters={"n_estimators": 10, "max_depth": 3, "random_state": 42},
        )
        train_fn, predict_fn = searcher.build_model(spec, target_col="target")

        df = _make_synthetic_data(n=200)
        model = train_fn(df.iloc[:150])
        preds = predict_fn(model, df.iloc[150:])

        assert len(preds) == 50

    def test_build_model_logistic(self, tmp_path: object) -> None:
        registry = ExperimentRegistry(storage_path=f"{tmp_path}/registry.json")
        searcher = ModelSearcher(registry=registry)

        spec = ModelSpec(
            model_family="logistic",
            hyperparameters={"C": 1.0},
            objective="classification",
        )
        train_fn, predict_fn = searcher.build_model(spec, target_col="target")

        df = _make_synthetic_data(n=200)
        # Make binary target for logistic
        df["target"] = (df["target"] > 0).astype(float)
        model = train_fn(df.iloc[:150])
        preds = predict_fn(model, df.iloc[150:])

        assert len(preds) == 50
        assert preds.min() >= 0.0
        assert preds.max() <= 1.0

    def test_generate_candidates(self, tmp_path: object) -> None:
        registry = ExperimentRegistry(storage_path=f"{tmp_path}/registry.json")
        searcher = ModelSearcher(registry=registry)

        spaces = [
            SearchSpace(
                model_family="ridge",
                param_grid={"alpha": [0.1, 1.0]},
            ),
            SearchSpace(
                model_family="random_forest",
                param_grid={"n_estimators": [50], "max_depth": [3]},
            ),
        ]

        candidates = searcher.generate_candidates(spaces, max_per_family=10)
        assert len(candidates) > 0
        families = {c.model_family for c in candidates}
        assert "ridge" in families
        assert "random_forest" in families

    def test_generate_candidates_subsampling(self, tmp_path: object) -> None:
        registry = ExperimentRegistry(storage_path=f"{tmp_path}/registry.json")
        searcher = ModelSearcher(registry=registry)

        # Large grid that would produce many candidates
        spaces = [
            SearchSpace(
                model_family="ridge",
                param_grid={"alpha": list(np.logspace(-3, 3, 50))},
                train_windows=[126, 252, 504],
            ),
        ]

        candidates = searcher.generate_candidates(spaces, max_per_family=5)
        assert len(candidates) == 5

    def test_generate_candidates_deprioritizes_bad_families(self, tmp_path: object) -> None:
        registry = ExperimentRegistry(storage_path=f"{tmp_path}/registry.json")
        ks = KnowledgeStore(storage_path=f"{tmp_path}/knowledge.json")

        # Mark ridge as not working
        ks.store_finding(
            domain="finance",
            model_family="ridge",
            finding="Ridge underperforms",
            works=False,
        )

        searcher = ModelSearcher(registry=registry, knowledge_store=ks)

        spaces = [
            SearchSpace(model_family="ridge", param_grid={"alpha": [1.0]}),
            SearchSpace(model_family="random_forest", param_grid={"n_estimators": [50]}),
        ]

        candidates = searcher.generate_candidates(spaces)
        # Random forest should come first (ridge deprioritized)
        assert candidates[0].model_family == "random_forest"
        # Ridge should still be present (just at end)
        assert any(c.model_family == "ridge" for c in candidates)

    def test_run_search_end_to_end(self, tmp_path: object) -> None:
        registry = ExperimentRegistry(storage_path=f"{tmp_path}/registry.json")
        searcher = ModelSearcher(
            registry=registry,
            problem_id="test_search",
            primary_metric="rmse",
        )

        df = _make_synthetic_data(n=600)

        spaces = [
            SearchSpace(
                model_family="ridge",
                param_grid={"alpha": [0.1, 1.0]},
            ),
        ]

        results = searcher.run_search(
            df=df,
            target_col="target",
            spaces=spaces,
            eval_fn=_simple_eval_fn,
            train_size=252,
            test_size=63,
        )

        assert len(results) > 0
        # Results should be sorted by primary metric (best first)
        for i in range(len(results) - 1):
            assert results[i].primary_metric_value >= results[i + 1].primary_metric_value

        # Experiments should be in registry
        all_exps = registry.list_experiments(problem_id="test_search")
        assert len(all_exps) == len(results)
        for exp in all_exps:
            assert exp.status == ExperimentStatus.COMPLETED

    def test_run_search_records_metrics(self, tmp_path: object) -> None:
        registry = ExperimentRegistry(storage_path=f"{tmp_path}/registry.json")
        searcher = ModelSearcher(
            registry=registry,
            problem_id="metrics_test",
            primary_metric="hit_rate",
        )

        df = _make_synthetic_data(n=500)
        spaces = [SearchSpace(model_family="ridge", param_grid={"alpha": [1.0]})]

        results = searcher.run_search(
            df=df, target_col="target", spaces=spaces, eval_fn=_simple_eval_fn
        )

        assert len(results) == 1
        result = results[0]
        assert result.primary_metric == "hit_rate"
        assert result.primary_metric_value is not None
        assert result.compute_seconds > 0
        assert "rmse" in result.secondary_metrics

    def test_meta_knowledge_stored(self, tmp_path: object) -> None:
        registry = ExperimentRegistry(storage_path=f"{tmp_path}/registry.json")
        ks = KnowledgeStore(storage_path=f"{tmp_path}/knowledge.json")
        searcher = ModelSearcher(
            registry=registry,
            knowledge_store=ks,
            problem_id="meta_test",
        )

        df = _make_synthetic_data(n=500)
        spaces = [
            SearchSpace(model_family="ridge", param_grid={"alpha": [0.1, 1.0]}),
        ]

        results = searcher.run_search(
            df=df, target_col="target", spaces=spaces, eval_fn=_simple_eval_fn
        )
        searcher.update_meta_knowledge(results)

        findings = ks.query_findings(model_family="ridge")
        assert len(findings) == 1
        assert "ridge" in findings[0].finding
        assert "best_metric" in findings[0].evidence

    def test_build_model_unknown_family(self, tmp_path: object) -> None:
        registry = ExperimentRegistry(storage_path=f"{tmp_path}/registry.json")
        searcher = ModelSearcher(registry=registry)

        spec = ModelSpec(model_family="nonexistent_model")
        train_fn, predict_fn = searcher.build_model(spec)

        df = _make_synthetic_data(n=100)
        with pytest.raises(ValueError, match="Unknown model family"):
            train_fn(df.iloc[:80])


class TestDefaultSearchSpaces:
    """Tests for default search space generation."""

    def test_regression_spaces(self) -> None:
        spaces = default_equity_search_spaces(task_type="regression")
        assert len(spaces) == 3
        families = {s.model_family for s in spaces}
        assert "ridge" in families
        assert "random_forest" in families
        assert "gradient_boosting" in families

    def test_classification_spaces(self) -> None:
        spaces = default_equity_search_spaces(task_type="classification")
        for space in spaces:
            assert "classification" in (space.objectives or [])
