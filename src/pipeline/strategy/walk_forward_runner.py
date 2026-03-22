"""Walk-forward validation runner for strategies.

Connects the walk-forward validation infrastructure in
``pipeline.backtesting.walk_forward`` with the strategy definitions and
backtest harness to produce formal out-of-sample validation results.

Usage::

    from pipeline.strategy.walk_forward_runner import run_walk_forward_validation
    from pipeline.strategy.strategy_definition import cross_sectional_momentum_strategy

    strategy = cross_sectional_momentum_strategy()
    result = run_walk_forward_validation(strategy, price_data)
    print(result.summary())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pipeline.backtesting.walk_forward import (
    FoldResult,
    walk_forward_splits,
)
from pipeline.strategy.signal_library import SignalPipeline
from pipeline.strategy.strategy_definition import StrategyDefinition

logger = logging.getLogger(__name__)

_TRADING_DAYS = 252


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation of a strategy."""

    train_days: int = 504  # 2 years in-sample
    test_days: int = 63    # 3 months out-of-sample
    step_days: int | None = None  # Defaults to test_days (non-overlapping)
    expanding: bool = True
    embargo_days: int = 5
    cost_bps: float = 5.0  # Total round-trip cost estimate


@dataclass
class WalkForwardResult:
    """Aggregate walk-forward validation result for a strategy."""

    strategy_name: str
    config: WalkForwardConfig
    folds: list[FoldResult]
    in_sample_metrics: list[dict[str, float]]
    out_of_sample_metrics: list[dict[str, float]]

    @property
    def n_folds(self) -> int:
        return len(self.folds)

    @property
    def oos_mean_metrics(self) -> dict[str, float]:
        if not self.out_of_sample_metrics:
            return {}
        keys = self.out_of_sample_metrics[0].keys()
        return {
            k: float(np.nanmean([m.get(k, np.nan) for m in self.out_of_sample_metrics]))
            for k in keys
        }

    @property
    def oos_std_metrics(self) -> dict[str, float]:
        if not self.out_of_sample_metrics:
            return {}
        keys = self.out_of_sample_metrics[0].keys()
        return {
            k: float(np.nanstd([m.get(k, np.nan) for m in self.out_of_sample_metrics]))
            for k in keys
        }

    @property
    def is_viable(self) -> bool:
        """Quick check: positive OOS Sharpe and win rate above 50%."""
        mean = self.oos_mean_metrics
        return (
            mean.get("sharpe_ratio", 0) > 0
            and mean.get("win_rate", 0) > 0.50
            and mean.get("total_return", 0) > 0
        )

    def summary(self) -> pd.DataFrame:
        """Produce a summary DataFrame of per-fold OOS metrics."""
        if not self.out_of_sample_metrics:
            return pd.DataFrame()

        rows: list[dict[str, Any]] = []
        for i, (fold, oos) in enumerate(zip(self.folds, self.out_of_sample_metrics)):
            row: dict[str, Any] = {
                "fold": i,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "test_size": fold.test_size,
            }
            row.update(oos)
            rows.append(row)

        # Add mean row
        mean_row: dict[str, Any] = {"fold": "MEAN", "test_start": "", "test_end": "", "test_size": ""}
        mean_row.update(self.oos_mean_metrics)
        rows.append(mean_row)

        return pd.DataFrame(rows)


def _compute_fold_metrics(
    returns: pd.Series,
    cost_bps: float,
) -> dict[str, float]:
    """Compute standard performance metrics for a fold's return series."""
    if returns.empty or len(returns) < 2:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
        }

    # Apply transaction cost proxy
    cost_per_day = cost_bps / 10_000
    net_returns = returns - cost_per_day

    total_ret = (1 + net_returns).prod() - 1
    n_days = len(net_returns)
    ann_factor = _TRADING_DAYS / max(n_days, 1)
    ann_ret = (1 + total_ret) ** ann_factor - 1

    vol = net_returns.std() * np.sqrt(_TRADING_DAYS)
    sharpe = ann_ret / vol if vol > 0 else 0.0

    # Drawdown
    cum = (1 + net_returns).cumprod()
    running_max = cum.cummax()
    drawdowns = cum / running_max - 1
    max_dd = drawdowns.min()

    # Win rate
    wins = (net_returns > 0).sum()
    total = (net_returns != 0).sum()
    win_rate = wins / total if total > 0 else 0.0

    # Profit factor
    gross_wins = net_returns[net_returns > 0].sum()
    gross_losses = abs(net_returns[net_returns < 0].sum())
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    return {
        "total_return": float(total_ret),
        "annualized_return": float(ann_ret),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_dd),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
    }


def run_walk_forward_validation(
    strategy: StrategyDefinition,
    price_data: dict[str, pd.DataFrame],
    config: WalkForwardConfig | None = None,
) -> WalkForwardResult:
    """Run walk-forward validation for a strategy.

    Args:
        strategy: The strategy definition to validate.
        price_data: ``{ticker: DataFrame}`` with OHLCV data indexed by date.
        config: Walk-forward parameters (uses defaults if None).

    Returns:
        WalkForwardResult with per-fold in-sample and out-of-sample metrics.
    """
    cfg = config or WalkForwardConfig()

    # Build signal pipeline
    if not strategy.signal_definitions:
        raise ValueError("Strategy has no signal definitions")

    pipeline = SignalPipeline(strategy.signal_definitions[0])
    composite = pipeline.run(price_data)

    if composite.empty:
        logger.warning("Signal pipeline produced empty composite — no data to validate")
        return WalkForwardResult(
            strategy_name=strategy.strategy_name,
            config=cfg,
            folds=[],
            in_sample_metrics=[],
            out_of_sample_metrics=[],
        )

    # Use cross-sectional ranks for long-only top-decile approach
    ranked = pipeline.cross_sectional_rank(composite)

    # Compute daily returns for each ticker
    returns_dict: dict[str, pd.Series] = {}
    for ticker, df in price_data.items():
        if "close" in df.columns:
            returns_dict[ticker] = df["close"].pct_change()

    if not returns_dict:
        raise ValueError("No return data available")

    returns_panel = pd.DataFrame(returns_dict)

    # Align indices
    common_idx = composite.index.intersection(returns_panel.index)
    ranked = ranked.loc[common_idx]
    returns_panel = returns_panel.loc[common_idx]

    if len(common_idx) < cfg.train_days + cfg.test_days + cfg.embargo_days:
        logger.warning(
            f"Insufficient data for walk-forward: {len(common_idx)} obs < "
            f"{cfg.train_days + cfg.test_days + cfg.embargo_days} required"
        )
        return WalkForwardResult(
            strategy_name=strategy.strategy_name,
            config=cfg,
            folds=[],
            in_sample_metrics=[],
            out_of_sample_metrics=[],
        )

    # Compute equal-weight portfolio returns based on top-quintile signal
    def _portfolio_returns(ranked_slice: pd.DataFrame, ret_slice: pd.DataFrame) -> pd.Series:
        """Long top-quintile, equal-weight daily returns."""
        # For each day, go long tickers in the top 20% of signal rank
        long_mask = ranked_slice >= 0.8
        n_positions = long_mask.sum(axis=1).replace(0, np.nan)
        portfolio_ret = (ret_slice * long_mask).sum(axis=1) / n_positions
        return portfolio_ret.fillna(0.0)

    folds: list[FoldResult] = []
    is_metrics: list[dict[str, float]] = []
    oos_metrics: list[dict[str, float]] = []

    for fold_i, (train_idx, test_idx) in enumerate(
        walk_forward_splits(
            common_idx,
            train_size=cfg.train_days,
            test_size=cfg.test_days,
            step_size=cfg.step_days,
            expanding=cfg.expanding,
            embargo_size=cfg.embargo_days,
        )
    ):
        # In-sample
        is_ranked = ranked.iloc[train_idx]
        is_returns = returns_panel.iloc[train_idx]
        is_port_ret = _portfolio_returns(is_ranked, is_returns)
        is_met = _compute_fold_metrics(is_port_ret, cfg.cost_bps)

        # Out-of-sample
        oos_ranked = ranked.iloc[test_idx]
        oos_returns = returns_panel.iloc[test_idx]
        oos_port_ret = _portfolio_returns(oos_ranked, oos_returns)
        oos_met = _compute_fold_metrics(oos_port_ret, cfg.cost_bps)

        fold = FoldResult(
            fold_index=fold_i,
            train_start=common_idx[train_idx[0]],
            train_end=common_idx[train_idx[-1]],
            test_start=common_idx[test_idx[0]],
            test_end=common_idx[test_idx[-1]],
            train_size=len(train_idx),
            test_size=len(test_idx),
            metrics=oos_met,
        )

        folds.append(fold)
        is_metrics.append(is_met)
        oos_metrics.append(oos_met)

        logger.info(
            f"Fold {fold_i}: IS sharpe={is_met['sharpe_ratio']:.2f} "
            f"OOS sharpe={oos_met['sharpe_ratio']:.2f} "
            f"OOS return={oos_met['total_return']:.2%}"
        )

    result = WalkForwardResult(
        strategy_name=strategy.strategy_name,
        config=cfg,
        folds=folds,
        in_sample_metrics=is_metrics,
        out_of_sample_metrics=oos_metrics,
    )

    viable_str = "VIABLE" if result.is_viable else "NOT VIABLE"
    logger.info(
        f"Walk-forward complete: {result.n_folds} folds, "
        f"mean OOS Sharpe={result.oos_mean_metrics.get('sharpe_ratio', 0):.2f}, "
        f"verdict={viable_str}"
    )

    return result
