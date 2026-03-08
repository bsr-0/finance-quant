"""Standardised Evaluation Matrix (Agent Directive V7 — Section 13).

Every serious candidate system must be reported through a common
evaluation matrix so results are comparable across candidates.

Required metric classes:
- Predictive accuracy (RMSE, MAE, log loss, Brier, AUC)
- Calibration (ECE, reliability, Brier decomposition)
- Decision utility (EV, profit, Sharpe, contest score)
- Risk (drawdown, volatility, worst-month, tail loss)
- Stability (performance by period, regime, segment)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class MatrixEntry:
    """One row in the evaluation matrix (one candidate system)."""

    candidate_id: str
    predictive_accuracy: dict[str, float] = field(default_factory=dict)
    calibration: dict[str, float] = field(default_factory=dict)
    decision_utility: dict[str, float] = field(default_factory=dict)
    risk: dict[str, float] = field(default_factory=dict)
    stability: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "predictive_accuracy": self.predictive_accuracy,
            "calibration": self.calibration,
            "decision_utility": self.decision_utility,
            "risk": self.risk,
            "stability": self.stability,
        }


class EvaluationMatrix:
    """Produce comparable evaluation output per Section 13.

    Usage::

        matrix = EvaluationMatrix()
        entry = matrix.evaluate(
            candidate_id="lgbm_v3",
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            returns=strategy_returns,
        )
        # Add more candidates ...
        comparison = matrix.compare()
    """

    def __init__(self) -> None:
        self._entries: list[MatrixEntry] = []

    def evaluate(
        self,
        candidate_id: str,
        y_true: pd.Series | None = None,
        y_pred: pd.Series | None = None,
        y_prob: pd.Series | None = None,
        returns: pd.Series | None = None,
    ) -> MatrixEntry:
        """Evaluate a candidate across all required metric classes."""
        from pipeline.eval.metrics import (
            brier_score,
            calibration_error,
            hit_rate,
            log_loss,
            max_drawdown,
            sharpe_sortino,
        )

        entry = MatrixEntry(candidate_id=candidate_id)

        # --- Predictive accuracy ---
        if y_true is not None and y_pred is not None:
            y_t, y_p = y_true.align(y_pred, join="inner")
            y_t = y_t.dropna()
            y_p = y_p.loc[y_t.index].dropna()
            y_t = y_t.loc[y_p.index]
            if not y_t.empty:
                entry.predictive_accuracy["rmse"] = float(
                    np.sqrt(((y_t - y_p) ** 2).mean())
                )
                entry.predictive_accuracy["mae"] = float((y_t - y_p).abs().mean())
                entry.predictive_accuracy["hit_rate"] = hit_rate(y_t, y_p)

        if y_true is not None and y_prob is not None:
            y_t, y_p = y_true.align(y_prob, join="inner")
            y_t = y_t.dropna()
            y_p = y_p.loc[y_t.index].dropna()
            y_t = y_t.loc[y_p.index]
            if not y_t.empty:
                entry.predictive_accuracy["brier"] = brier_score(y_t, y_p)
                entry.predictive_accuracy["log_loss"] = log_loss(y_t, y_p)

        # --- Calibration ---
        if y_true is not None and y_prob is not None:
            y_t, y_p = y_true.align(y_prob, join="inner")
            y_t = y_t.dropna()
            y_p = y_p.loc[y_t.index].dropna()
            y_t = y_t.loc[y_p.index]
            if not y_t.empty:
                entry.calibration["ece"] = calibration_error(y_t, y_p)
                # Over/under-confidence ratio
                df = pd.DataFrame({"y": y_t, "p": y_p})
                over_confident = ((df["p"] > 0.5) & (df["y"] == 0)).sum()
                under_confident = ((df["p"] < 0.5) & (df["y"] == 1)).sum()
                total = len(df)
                entry.calibration["over_confidence_rate"] = float(
                    over_confident / total
                ) if total > 0 else 0.0
                entry.calibration["under_confidence_rate"] = float(
                    under_confident / total
                ) if total > 0 else 0.0

        # --- Decision utility ---
        if returns is not None:
            returns = returns.dropna()
            if not returns.empty:
                sharpe, sortino = sharpe_sortino(returns)
                entry.decision_utility["total_return"] = float(
                    (1 + returns).prod() - 1
                )
                entry.decision_utility["sharpe"] = sharpe
                entry.decision_utility["sortino"] = sortino

        # --- Risk ---
        if returns is not None:
            returns = returns.dropna()
            if not returns.empty:
                entry.risk["max_drawdown"] = max_drawdown(returns)
                entry.risk["volatility"] = float(
                    returns.std() * np.sqrt(252)
                )
                # Worst month
                if hasattr(returns.index, "to_period"):
                    monthly = returns.groupby(
                        returns.index.to_period("M")
                    ).apply(lambda x: (1 + x).prod() - 1)
                    entry.risk["worst_month"] = float(monthly.min())
                entry.risk["var_95"] = float(returns.quantile(0.05))

        # --- Stability ---
        if returns is not None:
            returns = returns.dropna()
            if len(returns) >= 20:
                half = len(returns) // 2
                first_half = returns.iloc[:half]
                second_half = returns.iloc[half:]
                s1, _ = sharpe_sortino(first_half)
                s2, _ = sharpe_sortino(second_half)
                entry.stability["sharpe_first_half"] = s1
                entry.stability["sharpe_second_half"] = s2
                entry.stability["sharpe_stability"] = float(
                    min(s1, s2) / max(abs(s1), abs(s2), 1e-10)
                )

        self._entries.append(entry)
        return entry

    def compare(self) -> dict[str, Any]:
        """Side-by-side comparison of all evaluated candidates."""
        return {
            "report_type": "evaluation_matrix",
            "candidates": [e.to_dict() for e in self._entries],
            "total_candidates": len(self._entries),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def get_best(self, metric_path: str = "decision_utility.sharpe") -> str | None:
        """Return candidate_id of the best candidate by given metric.

        metric_path: e.g. "decision_utility.sharpe" or "risk.max_drawdown"
        """
        if not self._entries:
            return None
        category, metric = metric_path.split(".")
        values = []
        for e in self._entries:
            d = getattr(e, category, {})
            values.append(d.get(metric, float("-inf")))

        # For risk metrics like max_drawdown (negative), we want less negative
        if category == "risk":
            best_idx = int(np.argmax(values))
        else:
            best_idx = int(np.argmax(values))
        return self._entries[best_idx].candidate_id
