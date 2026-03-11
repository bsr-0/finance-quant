"""Tests for ensemble module (Agent Directive V7 — Section 8)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.ensemble import EnsembleBuilder, EnsembleComponent
from pipeline.experiment_registry import ExperimentRegistry
from pipeline.model_search import ModelSearcher, ModelSpec


def _make_synthetic_data(n: int = 600, n_features: int = 5, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    features = {f"feat_{i}": rng.randn(n) for i in range(n_features)}
    X = np.column_stack(list(features.values()))
    beta = rng.randn(n_features) * 0.1
    target = X @ beta + rng.randn(n) * 0.5
    features["target"] = target
    return pd.DataFrame(features, index=dates)


def _simple_eval_fn(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    aligned_true, aligned_pred = y_true.align(y_pred, join="inner")
    rmse = float(np.sqrt(((aligned_true - aligned_pred) ** 2).mean()))
    hit = float((np.sign(aligned_true) == np.sign(aligned_pred)).mean())
    errors = aligned_true - aligned_pred
    sharpe = float(errors.mean() / (errors.std() + 1e-10)) if len(errors) > 1 else 0.0
    return {"rmse": rmse, "hit_rate": hit, "sharpe": sharpe}


def _make_components(tmp_path: object, df: pd.DataFrame) -> list[EnsembleComponent]:
    """Build a few trained components from model search."""
    registry = ExperimentRegistry(storage_path=f"{tmp_path}/comp_registry.json")
    searcher = ModelSearcher(registry=registry, problem_id="comp_build")

    components = []
    for family, hp in [
        ("ridge", {"alpha": 0.1}),
        ("ridge", {"alpha": 10.0}),
        ("gradient_boosting", {"n_estimators": 10, "max_depth": 3, "random_state": 42}),
    ]:
        spec = ModelSpec(model_family=family, hyperparameters=hp)
        train_fn, predict_fn = searcher.build_model(spec, target_col="target")
        components.append(
            EnsembleComponent(
                component_id=f"{family}_{hp}",
                model_spec=spec,
                train_fn=train_fn,
                predict_fn=predict_fn,
            )
        )
    return components


class TestEnsembleBuilder:
    """Tests for EnsembleBuilder class."""

    def test_weighted_average(self, tmp_path: object) -> None:
        df = _make_synthetic_data(n=500)
        components = _make_components(tmp_path, df)

        registry = ExperimentRegistry(storage_path=f"{tmp_path}/registry.json")
        builder = EnsembleBuilder(registry=registry, primary_metric="rmse")

        train_fn, predict_fn = builder.weighted_average(components, target_col="target")

        # Test that it works with walk-forward
        from pipeline.backtesting.walk_forward import walk_forward_validate

        result = walk_forward_validate(
            df, train_fn, predict_fn, _simple_eval_fn,
            target_col="target", train_size=252, test_size=63,
        )
        assert len(result.folds) > 0
        assert "rmse" in result.mean_metrics

    def test_weighted_average_with_custom_weights(self, tmp_path: object) -> None:
        df = _make_synthetic_data(n=400)
        components = _make_components(tmp_path, df)

        registry = ExperimentRegistry(storage_path=f"{tmp_path}/registry.json")
        builder = EnsembleBuilder(registry=registry)

        # Weight first component much higher
        train_fn, predict_fn = builder.weighted_average(
            components, weights=[10.0, 1.0, 1.0], target_col="target"
        )

        train_df = df.iloc[:300]
        test_df = df.iloc[300:]

        models = train_fn(train_df)
        preds = predict_fn(models, test_df)
        assert len(preds) == 100
        assert not preds.isna().any()

    def test_stacking(self, tmp_path: object) -> None:
        df = _make_synthetic_data(n=500)
        components = _make_components(tmp_path, df)

        registry = ExperimentRegistry(storage_path=f"{tmp_path}/registry.json")
        builder = EnsembleBuilder(registry=registry)

        train_fn, predict_fn = builder.stacking(components, target_col="target")

        from pipeline.backtesting.walk_forward import walk_forward_validate

        result = walk_forward_validate(
            df, train_fn, predict_fn, _simple_eval_fn,
            target_col="target", train_size=252, test_size=63,
        )
        assert len(result.folds) > 0
        for fold in result.folds:
            assert fold.predictions is not None
            assert not fold.predictions.isna().any()

    def test_stacking_uses_meta_learner(self, tmp_path: object) -> None:
        """Verify stacking trains a meta-learner on component predictions."""
        df = _make_synthetic_data(n=400)
        components = _make_components(tmp_path, df)

        registry = ExperimentRegistry(storage_path=f"{tmp_path}/registry.json")
        builder = EnsembleBuilder(registry=registry)

        train_fn, predict_fn = builder.stacking(
            components, meta_family="ridge", target_col="target"
        )

        model_bundle = train_fn(df.iloc[:300])

        assert "level0_models" in model_bundle
        assert "meta_model" in model_bundle
        assert len(model_bundle["level0_models"]) == len(components)

    def test_greedy_forward_select(self, tmp_path: object) -> None:
        df = _make_synthetic_data(n=500)
        components = _make_components(tmp_path, df)

        registry = ExperimentRegistry(storage_path=f"{tmp_path}/registry.json")
        builder = EnsembleBuilder(registry=registry, primary_metric="rmse")

        selected = builder.greedy_forward_select(
            components, df, "target", _simple_eval_fn,
            max_components=2, train_size=252, test_size=63,
        )

        assert len(selected) >= 1
        assert len(selected) <= 2

    def test_run_ensemble_search_end_to_end(self, tmp_path: object) -> None:
        df = _make_synthetic_data(n=500)
        components = _make_components(tmp_path, df)

        registry = ExperimentRegistry(storage_path=f"{tmp_path}/registry.json")
        builder = EnsembleBuilder(
            registry=registry, primary_metric="rmse", problem_id="ens_test"
        )

        result = builder.run_ensemble_search(
            df, "target", components, _simple_eval_fn,
            methods=["weighted_average", "stacking"],
            train_size=252, test_size=63,
        )

        assert result.method in ("weighted_average", "stacking")
        assert result.validation_result is not None
        assert result.experiment_id != ""

        # Check experiments were registered
        exps = registry.list_experiments(problem_id="ens_test")
        assert len(exps) >= 2

    def test_compare_raw_vs_calibrated(self, tmp_path: object) -> None:
        rng = np.random.RandomState(42)
        n = 200
        y_true = pd.Series(rng.randint(0, 2, size=n).astype(float))
        raw_pred = pd.Series(np.clip(rng.uniform(0, 1, size=n), 0.01, 0.99))
        cal_pred = pd.Series(np.clip(rng.uniform(0, 1, size=n), 0.01, 0.99))

        registry = ExperimentRegistry(storage_path=f"{tmp_path}/registry.json")
        builder = EnsembleBuilder(registry=registry)

        comparison = builder.compare_raw_vs_calibrated(
            y_true, raw_pred, calibrated_pred=cal_pred
        )

        assert comparison["report_type"] == "evaluation_matrix"
        assert comparison["total_candidates"] == 2
        ids = [c["candidate_id"] for c in comparison["candidates"]]
        assert "raw" in ids
        assert "calibrated" in ids


class TestDiversityMetrics:
    """Tests for diversity measurement."""

    def test_identical_predictions(self, tmp_path: object) -> None:
        preds = pd.Series([0.1, 0.5, 0.9, -0.3, 0.7])
        registry = ExperimentRegistry(storage_path=f"{tmp_path}/registry.json")
        builder = EnsembleBuilder(registry=registry)

        diversity = builder.measure_diversity({
            "model_a": preds,
            "model_b": preds.copy(),
        })

        assert diversity.mean_pairwise_correlation == pytest.approx(1.0, abs=0.01)
        assert diversity.mean_disagreement_rate == pytest.approx(0.0, abs=0.01)
        assert diversity.component_count == 2

    def test_diverse_predictions(self, tmp_path: object) -> None:
        rng = np.random.RandomState(42)
        n = 100
        preds_a = pd.Series(rng.randn(n))
        preds_b = pd.Series(rng.randn(n))  # Independent predictions

        registry = ExperimentRegistry(storage_path=f"{tmp_path}/registry.json")
        builder = EnsembleBuilder(registry=registry)

        diversity = builder.measure_diversity({"a": preds_a, "b": preds_b})

        assert diversity.mean_pairwise_correlation < 0.5
        assert diversity.component_count == 2

    def test_single_component(self, tmp_path: object) -> None:
        registry = ExperimentRegistry(storage_path=f"{tmp_path}/registry.json")
        builder = EnsembleBuilder(registry=registry)

        diversity = builder.measure_diversity({"a": pd.Series([1, 2, 3])})
        assert diversity.mean_pairwise_correlation == 1.0
        assert diversity.component_count == 1


# Need pytest for approx
import pytest  # noqa: E402
