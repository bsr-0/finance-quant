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
from dataclasses import dataclass
from typing import Any

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
    window = returns.loc[scenario.start : scenario.end].dropna()  # type: ignore[misc]
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
    dd = (equity - peak) / peak.replace(0, np.nan)
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

    # Goodness-of-fit: Kolmogorov-Smirnov test against fitted GPD
    from scipy.stats import kstest
    ks_stat, ks_pval = kstest(tail, "genpareto", args=(c, loc, scale))
    if ks_pval < 0.05:
        logger.warning(
            "GPD fit failed KS test (p=%.4f, stat=%.4f). "
            "Tail risk estimates may be unreliable.",
            ks_pval, ks_stat,
        )

    var = threshold + genpareto.ppf(0.99, c, loc=loc, scale=scale)
    es = threshold + (scale + c * var) / (1 - c) if c < 1 else float("nan")
    return {
        "tail_var": float(-var),
        "tail_es": float(-es),
        "gpd_shape": float(c),
        "gpd_scale": float(scale),
        "ks_stat": float(ks_stat),
        "ks_pval": float(ks_pval),
    }


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
) -> dict[str, float | dict[str, float]]:
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
        "per_symbol": per_symbol,  # type: ignore[dict-item]
    }


def apply_correlated_shock(
    positions: dict[str, float],
    prices: dict[str, float],
    shock: HypotheticalShock,
    correlation_matrix: pd.DataFrame | None = None,
    stress_correlation: float = 0.85,
    n_simulations: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Compute PnL impact under correlated stress scenario.

    Unlike :func:`apply_hypothetical_shock`, this models the
    empirical observation that correlations spike during crises.
    When no correlation matrix is provided, assumes *stress_correlation*
    across all positions.

    Args:
        positions: Symbol -> quantity (signed).
        prices: Symbol -> current price.
        shock: The hypothetical shock to apply.
        correlation_matrix: Pre-stress correlation matrix (optional).
        stress_correlation: Correlation to assume during stress if
            no matrix is provided (default 0.85).
        n_simulations: Number of Monte Carlo draws.
        seed: Random seed.

    Returns:
        Dict with mean_pnl, var_95_pnl, cvar_95_pnl, worst_pnl.
    """
    rng = np.random.default_rng(seed)
    symbols = [s for s in positions if positions[s] != 0 and prices.get(s, 0) > 0]
    n_assets = len(symbols)
    if n_assets == 0:
        return {"mean_pnl": 0.0, "var_95_pnl": 0.0, "cvar_95_pnl": 0.0, "worst_pnl": 0.0}

    # Build stress correlation matrix
    if correlation_matrix is not None and all(s in correlation_matrix.index for s in symbols):
        corr = correlation_matrix.loc[symbols, symbols].values.copy()
        # Stress: shift correlations toward 1
        corr = corr + stress_correlation * (1 - corr)
        np.fill_diagonal(corr, 1.0)
    else:
        corr = np.full((n_assets, n_assets), stress_correlation)
        np.fill_diagonal(corr, 1.0)

    # Ensure positive semi-definite
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.maximum(eigvals, 1e-8)
    corr = eigvecs @ np.diag(eigvals) @ eigvecs.T

    # Simulate correlated shocks
    try:
        L = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        L = np.eye(n_assets)

    z = rng.standard_normal((n_simulations, n_assets))
    correlated_z = z @ L.T

    # Scale shocks: mean = price_shock_pct, vol = |price_shock_pct| * 0.5
    vol_scale = max(abs(shock.price_shock_pct) * 0.5, 0.01)
    asset_shocks = shock.price_shock_pct + vol_scale * correlated_z

    # Compute portfolio PnL for each simulation
    pos_vec = np.array([positions[s] for s in symbols])
    px_vec = np.array([prices[s] for s in symbols])
    notional = pos_vec * px_vec

    pnl_paths = asset_shocks @ notional
    mean_pnl = float(np.mean(pnl_paths))
    var_95 = float(np.percentile(pnl_paths, 5))
    tail = pnl_paths[pnl_paths <= var_95]
    cvar_95 = float(tail.mean()) if len(tail) > 0 else var_95

    return {
        "mean_pnl": mean_pnl,
        "var_95_pnl": var_95,
        "cvar_95_pnl": cvar_95,
        "worst_pnl": float(np.min(pnl_paths)),
    }


def run_all_stress_tests(
    returns: pd.Series,
    positions: dict[str, float] | None = None,
    prices: dict[str, float] | None = None,
    historical_scenarios: list[StressScenario] | None = None,
    hypothetical_shocks: list[HypotheticalShock] | None = None,
) -> dict[str, Any]:
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
    results: dict[str, Any] = {}

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
