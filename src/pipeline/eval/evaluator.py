"""Evaluator orchestration for institutional-grade model review."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pipeline.backtesting.transaction_costs import FixedPlusSpreadModel, SquareRootImpactModel, apply_transaction_costs
from pipeline.eval.factor_neutrality import compute_factor_exposures, factor_correlation_gate
from pipeline.eval.metrics import (
    brier_score,
    calibration_error,
    drawdown_recovery_time,
    hit_rate,
    information_ratio,
    log_loss,
    max_drawdown,
    sharpe_sortino,
    turnover,
)
from pipeline.eval.portfolio import ProbPortfolioConfig, SignalPortfolioConfig, generate_positions_from_probs, generate_positions_from_signals
from pipeline.eval.regime import classify_regimes, regime_performance
from pipeline.eval.stress import DEFAULT_SCENARIOS, evt_tail_risk, scenario_metrics
from pipeline.db import get_db_manager


@dataclass
class EvaluationResult:
    metrics: dict[str, float]
    regime_metrics: dict[str, dict[str, float]]
    factor_exposures: dict[str, Any]
    factor_corr: dict[str, float]
    stress_results: dict[str, dict[str, float]]
    passed_gates: dict[str, bool]


class DatabaseResultStore:
    """Persist evaluation artifacts to database tables."""

    def __init__(self):
        self.db = get_db_manager()

    def write_results(
        self,
        run_id: str,
        model_name: str,
        scope: str,
        dataset_id: str | None,
        config: dict,
        result: EvaluationResult,
    ) -> None:
        from sqlalchemy import text

        with self.db.engine.connect() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO meta_model_runs
                    (run_id, model_name, scope, dataset_id, config_json)
                    VALUES (:run_id, :model_name, :scope, :dataset_id, :config_json)
                """
                ),
                {
                    "run_id": run_id,
                    "model_name": model_name,
                    "scope": scope,
                    "dataset_id": dataset_id,
                    "config_json": json_dump(config),
                },
            )

            for metric_name, metric_value in result.metrics.items():
                conn.execute(
                    text(
                        """
                        INSERT INTO meta_model_metrics
                        (run_id, metric_name, metric_value)
                        VALUES (:run_id, :metric_name, :metric_value)
                    """
                    ),
                    {
                        "run_id": run_id,
                        "metric_name": metric_name,
                        "metric_value": metric_value,
                    },
                )

            for regime, metrics in result.regime_metrics.items():
                for metric_name, metric_value in metrics.items():
                    conn.execute(
                        text(
                            """
                            INSERT INTO meta_model_regime_metrics
                            (run_id, regime, metric_name, metric_value)
                            VALUES (:run_id, :regime, :metric_name, :metric_value)
                        """
                        ),
                        {
                            "run_id": run_id,
                            "regime": regime,
                            "metric_name": metric_name,
                            "metric_value": metric_value,
                        },
                    )

            betas = result.factor_exposures.get("betas", {})
            t_stats = result.factor_exposures.get("t_stats", {})
            p_vals = result.factor_exposures.get("p_values", {})
            r2 = result.factor_exposures.get("r2", np.nan)
            for factor, beta in betas.items():
                if factor == "intercept":
                    continue
                conn.execute(
                    text(
                        """
                        INSERT INTO meta_factor_exposures
                        (run_id, factor, beta, t_stat, p_value, r2)
                        VALUES (:run_id, :factor, :beta, :t_stat, :p_value, :r2)
                    """
                    ),
                    {
                        "run_id": run_id,
                        "factor": factor,
                        "beta": beta,
                        "t_stat": t_stats.get(factor),
                        "p_value": p_vals.get(factor),
                        "r2": r2,
                    },
                )

            for scenario, metrics in result.stress_results.items():
                conn.execute(
                    text(
                        """
                        INSERT INTO meta_stress_results
                        (run_id, scenario, var, es, max_dd, recovery_days)
                        VALUES (:run_id, :scenario, :var, :es, :max_dd, :recovery_days)
                    """
                    ),
                    {
                        "run_id": run_id,
                        "scenario": scenario,
                        "var": metrics.get("var"),
                        "es": metrics.get("es"),
                        "max_dd": metrics.get("max_dd"),
                        "recovery_days": metrics.get("recovery_days"),
                    },
                )

            conn.commit()


def json_dump(obj: Any) -> str:
    import json

    return json.dumps(obj, default=str)


class Evaluator:
    """Unified evaluator for equity and prediction market strategies."""

    def __init__(self, cost_bps: float = 20.0):
        self.cost_bps = cost_bps

    def _cost_model(self) -> FixedPlusSpreadModel:
        # FixedPlusSpreadModel uses half-spread internally, so double to match total bps.
        return FixedPlusSpreadModel(spread_bps=self.cost_bps * 2)

    def evaluate_equity(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        factor_returns: pd.DataFrame | None = None,
        benchmark_prices: pd.Series | None = None,
        config: SignalPortfolioConfig | None = None,
    ) -> EvaluationResult:
        positions = generate_positions_from_signals(signals, config)
        price_panel = prices.pivot(index="date", columns="symbol", values="price")

        adv_panel = None
        cost_model = self._cost_model()
        if "adv" in prices.columns:
            adv_panel = prices.pivot(index="date", columns="symbol", values="adv")
            cost_model = SquareRootImpactModel(spread_bps=self.cost_bps * 2)

        net = apply_transaction_costs(positions, price_panel, cost_model=cost_model, adv=adv_panel)
        returns = net["net_return"]

        metrics = self._compute_core_metrics(returns, positions)

        benchmark_returns = None
        if benchmark_prices is not None:
            benchmark_returns = benchmark_prices.pct_change().dropna()
        metrics["information_ratio"] = information_ratio(returns, benchmark_returns)

        metrics["hit_rate"] = self._signal_hit_rate(signals, prices)

        regimes = None
        regime_metrics = {}
        if benchmark_prices is not None:
            regimes = classify_regimes(benchmark_prices)
            regime_metrics = regime_performance(returns, regimes)

        stress = self._compute_stress(returns)

        factor_exposures = {}
        factor_corr = {}
        if factor_returns is not None:
            factor_exposures = compute_factor_exposures(returns, factor_returns)
            residuals = factor_exposures.get("residuals", pd.Series(dtype=float))
            passed_corr, corr = factor_correlation_gate(residuals, factor_returns)
            factor_corr = corr
            metrics["factor_corr_pass"] = 1.0 if passed_corr else 0.0

        gates = self._gate_metrics(metrics)

        return EvaluationResult(
            metrics=metrics,
            regime_metrics=regime_metrics,
            factor_exposures=factor_exposures,
            factor_corr=factor_corr,
            stress_results=stress,
            passed_gates=gates,
        )

    def evaluate_prediction_markets(
        self,
        probs: pd.DataFrame,
        prices: pd.DataFrame,
        outcomes: pd.DataFrame | None = None,
        factor_returns: pd.DataFrame | None = None,
        config: ProbPortfolioConfig | None = None,
    ) -> EvaluationResult:
        positions = generate_positions_from_probs(probs, config)
        price_panel = prices.pivot(index="date", columns="contract_id", values="market_price")

        net = apply_transaction_costs(positions, price_panel, cost_model=self._cost_model())
        returns = net["net_return"]

        metrics = self._compute_core_metrics(returns, positions)

        if outcomes is not None:
            merged = probs.merge(outcomes, on=["contract_id", "date"], how="inner")
            y_true = merged["outcome"].astype(float)
            y_prob = merged["model_prob"].astype(float)
            metrics["brier_score"] = brier_score(y_true, y_prob)
            metrics["log_loss"] = log_loss(y_true, y_prob)
            metrics["calibration_error"] = calibration_error(y_true, y_prob)

        stress = self._compute_stress(returns)

        factor_exposures = {}
        factor_corr = {}
        if factor_returns is not None:
            factor_exposures = compute_factor_exposures(returns, factor_returns)
            residuals = factor_exposures.get("residuals", pd.Series(dtype=float))
            passed_corr, corr = factor_correlation_gate(residuals, factor_returns)
            factor_corr = corr
            metrics["factor_corr_pass"] = 1.0 if passed_corr else 0.0

        gates = self._gate_metrics(metrics)

        return EvaluationResult(
            metrics=metrics,
            regime_metrics={},
            factor_exposures=factor_exposures,
            factor_corr=factor_corr,
            stress_results=stress,
            passed_gates=gates,
        )

    def _compute_core_metrics(self, returns: pd.Series, positions: pd.DataFrame) -> dict[str, float]:
        sharpe, sortino = sharpe_sortino(returns)
        turnover_series = turnover(positions)
        metrics = {
            "sharpe": sharpe,
            "sortino": sortino,
            "information_ratio": information_ratio(returns),
            "max_drawdown": max_drawdown(returns),
            "recovery_time": float(drawdown_recovery_time(returns)),
            "turnover_avg": float(turnover_series.mean()) if not turnover_series.empty else float("nan"),
        }
        return metrics

    def _compute_stress(self, returns: pd.Series) -> dict[str, dict[str, float]]:
        stress = {}
        for scenario in DEFAULT_SCENARIOS:
            stress[scenario.name] = scenario_metrics(returns, scenario)
        tail = evt_tail_risk(returns)
        stress["EVT_TAIL"] = {
            "var": tail["tail_var"],
            "es": tail["tail_es"],
            "max_dd": np.nan,
            "recovery_days": np.nan,
        }
        return stress

    def _gate_metrics(self, metrics: dict[str, float]) -> dict[str, bool]:
        gates = {
            "sharpe_gt_2": metrics.get("sharpe", np.nan) > 2.0,
            "sortino_gt_2": metrics.get("sortino", np.nan) > 2.0,
            "max_dd_gt_-0_10": metrics.get("max_drawdown", np.nan) > -0.10,
        }
        return gates

    def _signal_hit_rate(self, signals: pd.DataFrame, prices: pd.DataFrame) -> float:
        if signals.empty or prices.empty:
            return np.nan
        sig = signals.copy()
        px = prices.copy()
        sig["date"] = pd.to_datetime(sig["date"])
        px["date"] = pd.to_datetime(px["date"])

        px = px.sort_values(["symbol", "date"])
        px["fwd_ret"] = px.groupby("symbol")["price"].pct_change().shift(-1)

        merged = sig.merge(px[["date", "symbol", "fwd_ret"]], on=["date", "symbol"], how="inner")
        merged = merged.dropna(subset=["signal", "fwd_ret"])
        if merged.empty:
            return np.nan
        return hit_rate(merged["fwd_ret"], merged["signal"])
