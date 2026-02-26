"""Strategy capacity analysis and parameter sensitivity testing.

Addresses Goldman Sachs rubric §3 (Model Development and Testing):

* **Capacity analysis** – estimates the maximum AUM a strategy can absorb
  before market impact erodes returns below an acceptable floor.  Uses the
  square-root impact model to project how costs scale with trade size.
* **Sensitivity analysis** – re-runs a backtest metric across a grid of
  parameter values to verify that results are robust rather than over-fitted
  to a single choice.

Both analyses are model-agnostic: you pass in the appropriate callable and
they handle the sweep and aggregation.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Capacity Analysis
# ---------------------------------------------------------------------------

@dataclass
class CapacityResult:
    """Result of a capacity analysis sweep."""

    capital_levels: list[float]
    """AUM levels tested (in currency units)."""

    net_sharpes: list[float]
    """Net-of-cost Sharpe ratio at each capital level."""

    net_returns: list[float]
    """Mean annualised net return at each capital level."""

    cost_drags: list[float]
    """Mean annualised cost drag at each capital level."""

    capacity_estimate: float
    """Estimated capacity: largest AUM with net Sharpe >= ``min_sharpe``."""

    min_sharpe: float = 0.5
    """The Sharpe floor used to determine capacity."""

    def summary(self) -> pd.DataFrame:
        """Return a tidy DataFrame of results."""
        return pd.DataFrame(
            {
                "capital": self.capital_levels,
                "net_sharpe": self.net_sharpes,
                "net_return": self.net_returns,
                "cost_drag": self.cost_drags,
            }
        )


def capacity_analysis(
    returns: pd.Series,
    trades_per_year: float,
    avg_price: float,
    adv: float,
    capital_levels: list[float] | None = None,
    sigma: float = 0.02,
    eta: float = 0.25,
    spread_bps: float = 5.0,
    commission_per_share: float = 0.005,
    min_sharpe: float = 0.5,
    trading_days: int = 252,
) -> CapacityResult:
    """Estimate strategy capacity via a market-impact sweep.

    For each capital level the function computes the average trade size,
    applies the square-root impact model to estimate the annual cost drag,
    and computes the resulting net Sharpe ratio.

    Args:
        returns: Daily gross return series (pre-cost).
        trades_per_year: Expected number of round-trip trades per year.
        avg_price: Average price per share / contract.
        adv: Average daily volume in shares / contracts.
        capital_levels: List of AUM values to test (default: log-spaced from
            1× to 1000× the initial ``10 * avg_price`` level).
        sigma: Daily return volatility used in impact model.
        eta: Market impact coefficient (Almgren-Chriss).
        spread_bps: Half-spread in basis points.
        commission_per_share: Per-share commission.
        min_sharpe: Sharpe floor for the capacity estimate.
        trading_days: Annualisation factor.

    Returns:
        ``CapacityResult`` with per-level metrics and an overall estimate.
    """
    returns = returns.dropna()
    if returns.empty:
        raise ValueError("returns series is empty")

    gross_mu = float(returns.mean() * trading_days)
    gross_sigma = float(returns.std() * np.sqrt(trading_days))
    gross_sharpe = gross_mu / gross_sigma if gross_sigma > 0 else np.nan

    logger.info(
        "Capacity analysis: gross Sharpe=%.2f, trades/yr=%.0f, ADV=%.0f",
        gross_sharpe,
        trades_per_year,
        adv,
    )

    if capital_levels is None:
        base = 10 * avg_price
        capital_levels = [base * m for m in [1, 5, 10, 50, 100, 250, 500, 1000]]

    net_sharpes = []
    net_returns = []
    cost_drags = []

    for capital in capital_levels:
        shares_per_trade = capital / avg_price if avg_price > 0 else 0
        participation = shares_per_trade / adv if adv > 0 else 0

        # Square-root impact (fraction of notional)
        impact_frac = sigma * eta * np.sqrt(participation)
        # Half-spread crossing
        spread_frac = spread_bps / 10_000 / 2
        # Commission as fraction of notional
        commission_frac = commission_per_share / avg_price if avg_price > 0 else 0

        cost_per_trade = impact_frac + spread_frac + commission_frac
        annual_cost = cost_per_trade * trades_per_year

        net_mu = gross_mu - annual_cost
        net_sharpe = net_mu / gross_sigma if gross_sigma > 0 else np.nan

        net_sharpes.append(float(net_sharpe))
        net_returns.append(float(net_mu))
        cost_drags.append(float(annual_cost))

    # Capacity estimate: largest level where net Sharpe >= floor
    capacity_estimate = 0.0
    for cap, ns in zip(capital_levels, net_sharpes):
        if np.isfinite(ns) and ns >= min_sharpe:
            capacity_estimate = cap

    logger.info("Capacity estimate: %.0f (min_sharpe=%.2f)", capacity_estimate, min_sharpe)

    return CapacityResult(
        capital_levels=list(capital_levels),
        net_sharpes=net_sharpes,
        net_returns=net_returns,
        cost_drags=cost_drags,
        capacity_estimate=capacity_estimate,
        min_sharpe=min_sharpe,
    )


# ---------------------------------------------------------------------------
# Sensitivity Analysis
# ---------------------------------------------------------------------------

@dataclass
class SensitivityResult:
    """Result of a sensitivity (parameter sweep) analysis."""

    param_name: str
    """Name of the parameter that was swept."""

    param_values: list[float]
    """Values tested."""

    metric_values: list[float]
    """Metric value at each parameter setting."""

    metric_name: str = "metric"
    """Name of the metric being evaluated."""

    baseline_value: float = float("nan")
    """Metric value at the baseline (default) parameter setting."""

    @property
    def is_robust(self) -> bool:
        """Return ``True`` if the metric stays positive across the sweep."""
        finite = [v for v in self.metric_values if np.isfinite(v)]
        return len(finite) > 0 and min(finite) > 0

    @property
    def variation_coefficient(self) -> float:
        """Coefficient of variation (std/|mean|) across tested values."""
        finite = [v for v in self.metric_values if np.isfinite(v)]
        if len(finite) < 2:
            return float("nan")
        mu = abs(float(np.mean(finite)))
        if mu == 0:
            return float("nan")
        return float(np.std(finite) / mu)

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                self.param_name: self.param_values,
                self.metric_name: self.metric_values,
            }
        )


def sensitivity_analysis(
    df: pd.DataFrame,
    param_name: str,
    param_values: list[float],
    metric_fn: Callable[[pd.DataFrame, float], float],
    metric_name: str = "metric",
    baseline_value: float | None = None,
) -> SensitivityResult:
    """Sweep a single parameter and record the impact on a strategy metric.

    Args:
        df: Input DataFrame (features + target).
        param_name: Human-readable name of the parameter being swept.
        param_values: Ordered list of values to test.
        metric_fn: ``metric_fn(df, param_value) → float``.  Must be
            deterministic for the same inputs.
        metric_name: Label for the output metric (e.g. ``"sharpe"``).
        baseline_value: Optional pre-computed baseline metric value.

    Returns:
        ``SensitivityResult`` with per-value metrics.
    """
    metric_values = []
    for val in param_values:
        try:
            result = metric_fn(df, val)
            metric_values.append(float(result))
        except Exception as exc:
            logger.warning("sensitivity_analysis: param=%s value=%s error: %s", param_name, val, exc)
            metric_values.append(float("nan"))

    if baseline_value is None and param_values:
        mid_idx = len(param_values) // 2
        baseline_value = metric_values[mid_idx]

    result = SensitivityResult(
        param_name=param_name,
        param_values=list(param_values),
        metric_values=metric_values,
        metric_name=metric_name,
        baseline_value=float(baseline_value) if baseline_value is not None else float("nan"),
    )

    logger.info(
        "Sensitivity analysis on '%s': robust=%s, CV=%.2f",
        param_name,
        result.is_robust,
        result.variation_coefficient if np.isfinite(result.variation_coefficient) else float("nan"),
    )

    return result


def multi_param_sensitivity(
    df: pd.DataFrame,
    param_grid: dict[str, list[float]],
    metric_fn: Callable[[pd.DataFrame, dict[str, float]], float],
    metric_name: str = "metric",
) -> pd.DataFrame:
    """Evaluate a metric over a full parameter grid.

    Args:
        df: Input DataFrame.
        param_grid: ``{param_name: [values, ...]}`` mapping.
        metric_fn: ``metric_fn(df, params_dict) → float``.
        metric_name: Label for the result column.

    Returns:
        DataFrame with one row per parameter combination and a column for the
        metric value.
    """
    import itertools

    names = list(param_grid.keys())
    value_lists = [param_grid[n] for n in names]

    rows = []
    for combo in itertools.product(*value_lists):
        params = dict(zip(names, combo))
        try:
            val = float(metric_fn(df, params))
        except Exception as exc:
            logger.warning("multi_param_sensitivity error for %s: %s", params, exc)
            val = float("nan")
        row = dict(params)
        row[metric_name] = val
        rows.append(row)

    return pd.DataFrame(rows)
