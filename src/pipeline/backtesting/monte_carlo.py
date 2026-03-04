"""Monte Carlo simulation for PnL path analysis.

Estimates the distribution of outcomes, drawdowns, and probability of
ruin by resampling historical PnL paths under various assumptions.

Methods supported:
    1. **Block bootstrap**: Resample contiguous blocks of returns to
       preserve autocorrelation structure.
    2. **Execution randomization**: Stress slippage and cost assumptions
       within realistic ranges.
    3. **Parameter stress**: Perturb strategy parameters to assess
       sensitivity.

All simulations are seeded for reproducibility.

Assumptions:
    - Input returns/PnL are daily-frequency.
    - Block bootstrap preserves the serial dependence of returns.
    - Execution randomization assumes costs are proportional to notional.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation.

    Attributes:
        n_simulations: Number of simulated paths.
        block_size: Block size for block bootstrap (trading days).
        path_length: Length of each simulated path (trading days).
            If 0, uses the length of the input series.
        slippage_range_bps: (min, max) range for slippage stress.
        fee_range_bps: (min, max) range for fee stress.
        seed: Random seed for reproducibility.
    """

    n_simulations: int = 1000
    block_size: int = 21
    path_length: int = 0
    slippage_range_bps: tuple[float, float] = (0.0, 10.0)
    fee_range_bps: tuple[float, float] = (0.0, 5.0)
    seed: int = 42


@dataclass
class MonteCarloResult:
    """Result of a Monte Carlo simulation.

    Attributes:
        simulated_final_values: Array of terminal portfolio values.
        simulated_max_drawdowns: Array of max drawdown per path.
        simulated_sharpe_ratios: Array of Sharpe ratio per path.
        summary_stats: Dict of summary statistics.
        percentiles: Dict of percentile values for final value.
        probability_of_loss: Fraction of paths ending with a loss.
        probability_of_ruin: Fraction of paths hitting the ruin threshold.
    """

    simulated_final_values: np.ndarray
    simulated_max_drawdowns: np.ndarray
    simulated_sharpe_ratios: np.ndarray
    summary_stats: dict[str, float]
    percentiles: dict[str, float]
    probability_of_loss: float
    probability_of_ruin: float
    paths: np.ndarray | None = None


def block_bootstrap(
    returns: np.ndarray,
    n_paths: int,
    path_length: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate simulated return paths via block bootstrap.

    Args:
        returns: Historical return series (1D array).
        n_paths: Number of simulated paths.
        path_length: Length of each path.
        block_size: Size of contiguous blocks to resample.
        rng: Random number generator.

    Returns:
        2D array of shape (n_paths, path_length).
    """
    n = len(returns)
    if n < block_size:
        block_size = max(1, n)

    paths = np.zeros((n_paths, path_length))
    n_blocks = (path_length + block_size - 1) // block_size

    for i in range(n_paths):
        blocks = []
        for _ in range(n_blocks):
            start = rng.integers(0, n - block_size + 1)
            blocks.append(returns[start : start + block_size])
        path = np.concatenate(blocks)[:path_length]
        paths[i] = path

    return paths


def monte_carlo_simulation(
    returns: pd.Series,
    initial_capital: float = 1_000_000.0,
    config: MonteCarloConfig | None = None,
    ruin_threshold: float = 0.5,
) -> MonteCarloResult:
    """Run Monte Carlo simulation on a return series.

    Args:
        returns: Daily return series (as fractions, e.g. 0.01 = 1%).
        initial_capital: Starting portfolio value.
        config: Simulation configuration.
        ruin_threshold: Fraction of capital loss that constitutes ruin
            (e.g. 0.5 = 50% loss).

    Returns:
        ``MonteCarloResult`` with simulated outcomes and statistics.
    """
    config = config or MonteCarloConfig()
    rng = np.random.default_rng(config.seed)
    returns_arr = returns.dropna().values.astype(float)

    if len(returns_arr) < 10:
        raise ValueError("Need at least 10 return observations")

    path_length = config.path_length if config.path_length > 0 else len(returns_arr)

    # Generate simulated return paths via block bootstrap
    sim_returns = block_bootstrap(
        returns_arr, config.n_simulations, path_length,
        config.block_size, rng,
    )

    # Compute equity curves
    equity_paths = initial_capital * np.cumprod(1 + sim_returns, axis=1)

    # Terminal values
    final_values = equity_paths[:, -1]

    # Max drawdowns
    max_dds = np.zeros(config.n_simulations)
    for i in range(config.n_simulations):
        peaks = np.maximum.accumulate(equity_paths[i])
        dd = (equity_paths[i] - peaks) / peaks
        max_dds[i] = dd.min()

    # Sharpe ratios per path
    sharpe_ratios = np.zeros(config.n_simulations)
    for i in range(config.n_simulations):
        path_rets = sim_returns[i]
        if path_rets.std() > 0:
            sharpe_ratios[i] = path_rets.mean() / path_rets.std() * np.sqrt(252)
        else:
            sharpe_ratios[i] = 0.0

    # Summary statistics
    summary = {
        "mean_final_value": float(np.mean(final_values)),
        "median_final_value": float(np.median(final_values)),
        "std_final_value": float(np.std(final_values)),
        "mean_return": float(np.mean(final_values / initial_capital - 1)),
        "mean_max_drawdown": float(np.mean(max_dds)),
        "worst_max_drawdown": float(np.min(max_dds)),
        "mean_sharpe": float(np.mean(sharpe_ratios)),
        "median_sharpe": float(np.median(sharpe_ratios)),
        "return_volatility": float(np.std(final_values / initial_capital - 1)),
        "skewness": float(pd.Series(final_values).skew()),
        "kurtosis": float(pd.Series(final_values).kurtosis()),
        "downside_deviation": float(
            np.sqrt(np.mean(np.minimum(sim_returns, 0) ** 2)) * np.sqrt(252)
        ),
    }

    # Percentiles
    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    percentiles = {
        f"p{p}": float(np.percentile(final_values, p)) for p in pcts
    }

    # Probability of loss
    prob_loss = float(np.mean(final_values < initial_capital))

    # Probability of ruin
    ruin_level = initial_capital * (1 - ruin_threshold)
    min_values = equity_paths.min(axis=1)
    prob_ruin = float(np.mean(min_values < ruin_level))

    logger.info(
        "Monte Carlo: %d paths, mean_return=%.2f%%, P(loss)=%.1f%%, P(ruin)=%.1f%%",
        config.n_simulations,
        summary["mean_return"] * 100,
        prob_loss * 100,
        prob_ruin * 100,
    )

    return MonteCarloResult(
        simulated_final_values=final_values,
        simulated_max_drawdowns=max_dds,
        simulated_sharpe_ratios=sharpe_ratios,
        summary_stats=summary,
        percentiles=percentiles,
        probability_of_loss=prob_loss,
        probability_of_ruin=prob_ruin,
    )


def execution_stress_test(
    returns: pd.Series,
    initial_capital: float = 1_000_000.0,
    n_scenarios: int = 100,
    slippage_range_bps: tuple[float, float] = (0.0, 10.0),
    fee_range_bps: tuple[float, float] = (0.0, 5.0),
    trades_per_day: float = 10.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Stress-test execution assumptions.

    For each scenario, random slippage and fee levels are drawn and
    applied to the return series.

    Returns:
        DataFrame with columns: scenario, slippage_bps, fee_bps,
        final_value, sharpe, max_dd.
    """
    rng = np.random.default_rng(seed)
    returns_arr = returns.dropna().values.astype(float)

    records = []
    for i in range(n_scenarios):
        slip = rng.uniform(*slippage_range_bps)
        fee = rng.uniform(*fee_range_bps)
        daily_cost = (slip + fee) / 10_000 * trades_per_day
        adj_returns = returns_arr - daily_cost

        equity = initial_capital * np.cumprod(1 + adj_returns)
        final_val = equity[-1]
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak
        max_dd = dd.min()
        sharpe = (
            adj_returns.mean() / adj_returns.std() * np.sqrt(252)
            if adj_returns.std() > 0
            else 0.0
        )

        records.append({
            "scenario": i,
            "slippage_bps": slip,
            "fee_bps": fee,
            "final_value": float(final_val),
            "sharpe": float(sharpe),
            "max_dd": float(max_dd),
        })

    return pd.DataFrame(records)
