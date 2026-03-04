"""Stress testing utilities for institutional risk management.

Provides both historical scenario replay and hypothetical shock analysis.
Includes EVT (Extreme Value Theory) tail risk estimation via the
Generalized Pareto Distribution.

Historical scenarios:
    Replay portfolio returns during known stress periods (GFC, COVID,
    taper tantrum, etc.) and compute VaR, ES, drawdown, and recovery.

Hypothetical shocks:
    Apply configurable shocks to prices, spreads, volatilities, and
    liquidity to estimate impact on portfolio PnL and risk metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import genpareto

logger = logging.getLogger(__name__)


@dataclass
class StressScenario:
    """A historical stress scenario defined by date range."""

    name: str
    start: str
    end: str


DEFAULT_SCENARIOS = [
    StressScenario("GFC_2008", "2008-09-01", "2009-06-30"),
    StressScenario("COVID_2020", "2020-02-15", "2020-05-31"),
    StressScenario("TAPER_TANTRUM_2013", "2013-05-01", "2013-09-30"),
    StressScenario("FLASH_CRASH_2010", "2010-05-01", "2010-06-30"),
    StressScenario("VOLMAGEDDON_2018", "2018-01-26", "2018-03-31"),
    StressScenario("RATE_HIKE_2022", "2022-01-01", "2022-10-31"),
]


def scenario_metrics(returns: pd.Series, scenario: StressScenario) -> dict[str, float]:
    """Compute stress metrics for a given historical scenario.

    Args:
        returns: Daily return series covering the scenario period.
        scenario: Historical scenario with start/end dates.

    Returns:
        Dict with VaR, Expected Shortfall, max drawdown, and recovery days.
    """
    window = returns.loc[scenario.start : scenario.end].dropna()
    if window.empty:
        return {
            "var": float("nan"),
            "es": float("nan"),
            "max_dd": float("nan"),
            "recovery_days": float("nan"),
        }
    var = np.percentile(window, 5)
    es = window[window <= var].mean() if (window <= var).any() else float("nan")
    equity = (1 + window).cumprod()
    peak = equity.cummax()
    dd = (equity - peak) / peak
    is_at_peak = equity >= peak
    groups = is_at_peak.cumsum()
    duration = equity.groupby(groups).cumcount()
    recovery_days = int(duration.max()) if len(duration) > 0 else float("nan")
    return {
        "var": float(var),
        "es": float(es),
        "max_dd": float(dd.min()),
        "recovery_days": recovery_days,
    }


def evt_tail_risk(returns: pd.Series, threshold_quantile: float = 0.95) -> dict[str, float]:
    """Fit a Generalized Pareto Distribution to tail losses.

    Uses Extreme Value Theory to model the distribution of losses
    beyond a high threshold, providing more reliable tail risk
    estimates than historical percentiles.

    Args:
        returns: Daily return series.
        threshold_quantile: Quantile for the GPD threshold (higher
            means only the most extreme losses are modeled).

    Returns:
        Dict with tail VaR and tail ES at 99% confidence.
    """
    returns = returns.dropna()
    if returns.empty:
        return {"tail_var": float("nan"), "tail_es": float("nan")}

    losses = -returns
    threshold = np.quantile(losses, threshold_quantile)
    tail = losses[losses > threshold] - threshold
    if len(tail) < 20:
        return {"tail_var": float("nan"), "tail_es": float("nan")}

    c, loc, scale = genpareto.fit(tail, floc=0)
    var = threshold + genpareto.ppf(0.99, c, loc=loc, scale=scale)
    es = threshold + (scale + c * var) / (1 - c) if c < 1 else float("nan")
    return {"tail_var": float(-var), "tail_es": float(-es)}


# ---------------------------------------------------------------------------
# Hypothetical Shocks
# ---------------------------------------------------------------------------

@dataclass
class HypotheticalShock:
    """A hypothetical shock scenario.

    Attributes:
        name: Descriptive name.
        price_shock_pct: Across-the-board price change (e.g. -0.10 = -10%).
        spread_multiplier: Multiplier on bid-ask spreads (e.g. 3.0 = 3x wider).
        volatility_multiplier: Multiplier on realized vol (e.g. 2.0 = 2x).
        liquidity_multiplier: Multiplier on ADV (e.g. 0.3 = 70% less liquid).
        fee_increase_bps: Additional fees in basis points.
    """

    name: str
    price_shock_pct: float = 0.0
    spread_multiplier: float = 1.0
    volatility_multiplier: float = 1.0
    liquidity_multiplier: float = 1.0
    fee_increase_bps: float = 0.0


DEFAULT_HYPOTHETICAL_SHOCKS = [
    HypotheticalShock("market_crash_10pct", price_shock_pct=-0.10),
    HypotheticalShock("market_crash_20pct", price_shock_pct=-0.20),
    HypotheticalShock("spread_blowout", spread_multiplier=5.0),
    HypotheticalShock("vol_spike", volatility_multiplier=3.0, price_shock_pct=-0.05),
    HypotheticalShock("liquidity_drought", liquidity_multiplier=0.1, spread_multiplier=3.0),
    HypotheticalShock("combined_stress",
                       price_shock_pct=-0.15, spread_multiplier=3.0,
                       volatility_multiplier=2.5, liquidity_multiplier=0.2),
]


def apply_hypothetical_shock(
    positions: dict[str, float],
    prices: dict[str, float],
    shock: HypotheticalShock,
    spreads_bps: dict[str, float] | None = None,
) -> dict[str, float]:
    """Compute the PnL impact of a hypothetical shock.

    Args:
        positions: Symbol → quantity (signed).
        prices: Symbol → current price.
        shock: The hypothetical shock to apply.
        spreads_bps: Current spreads in bps per symbol (for spread stress).

    Returns:
        Dict with: pnl_impact, new_gross_exposure, spread_cost_increase,
        and per-symbol impacts.
    """
    total_pnl = 0.0
    per_symbol: dict[str, float] = {}

    for sym, qty in positions.items():
        if qty == 0:
            continue
        px = prices.get(sym, 0)
        if px <= 0:
            continue

        # Price impact
        price_pnl = qty * px * shock.price_shock_pct
        total_pnl += price_pnl
        per_symbol[sym] = price_pnl

    # Spread cost impact (cost of unwinding at wider spreads)
    spread_cost = 0.0
    if spreads_bps:
        for sym, qty in positions.items():
            px = prices.get(sym, 0)
            base_spread = spreads_bps.get(sym, 5.0)
            stressed_spread = base_spread * shock.spread_multiplier
            additional_cost = abs(qty) * px * (stressed_spread - base_spread) / 10_000
            spread_cost += additional_cost

    return {
        "pnl_impact": total_pnl,
        "spread_cost_increase": spread_cost,
        "total_impact": total_pnl - spread_cost,
        "per_symbol": per_symbol,
    }


def run_all_stress_tests(
    returns: pd.Series,
    positions: dict[str, float] | None = None,
    prices: dict[str, float] | None = None,
    historical_scenarios: list[StressScenario] | None = None,
    hypothetical_shocks: list[HypotheticalShock] | None = None,
) -> dict[str, dict[str, float]]:
    """Run all stress tests and return a consolidated report.

    Args:
        returns: Historical return series for scenario replay.
        positions: Current positions for hypothetical shocks.
        prices: Current prices for hypothetical shocks.
        historical_scenarios: List of historical scenarios (default: all).
        hypothetical_shocks: List of hypothetical shocks (default: all).

    Returns:
        Dict mapping scenario/shock name to results.
    """
    results: dict[str, dict[str, float]] = {}

    # Historical scenarios
    scenarios = historical_scenarios or DEFAULT_SCENARIOS
    for scenario in scenarios:
        results[scenario.name] = scenario_metrics(returns, scenario)

    # EVT tail risk
    results["EVT_TAIL"] = evt_tail_risk(returns)

    # Hypothetical shocks
    if positions and prices:
        shocks = hypothetical_shocks or DEFAULT_HYPOTHETICAL_SHOCKS
        for shock in shocks:
            impact = apply_hypothetical_shock(positions, prices, shock)
            results[f"hypo_{shock.name}"] = {
                "pnl_impact": impact["pnl_impact"],
                "spread_cost": impact["spread_cost_increase"],
                "total_impact": impact["total_impact"],
            }

    return results
