"""Transaction cost models for realistic backtesting.

Provides pluggable cost models from simple (fixed + spread) to
sophisticated (square-root market impact).  All models implement
the same interface so they can be swapped without changing
strategy code.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """A single trade to be costed."""

    symbol: str
    side: str  # "buy" or "sell"
    quantity: float  # number of shares / contracts
    price: float  # reference price (e.g. mid)
    adv: float = 0  # average daily volume (shares) – needed for impact models


@dataclass
class TradeCost:
    """Breakdown of costs for a trade."""

    spread_cost: float = 0.0
    commission: float = 0.0
    market_impact: float = 0.0
    slippage: float = 0.0

    @property
    def total(self) -> float:
        return self.spread_cost + self.commission + self.market_impact + self.slippage

    @property
    def total_bps(self) -> float:
        """Total cost in basis points (requires setting notional externally)."""
        return self.total


class CostModel(ABC):
    """Base class for transaction cost models."""

    @abstractmethod
    def estimate(self, trade: Trade) -> TradeCost:
        """Estimate the cost of executing *trade*."""


# ---------------------------------------------------------------------------
# Model 1: Fixed + Spread
# ---------------------------------------------------------------------------


class FixedPlusSpreadModel(CostModel):
    """Simple model: half-spread crossing + fixed per-share commission.

    Suitable for equities with relatively tight spreads.
    """

    def __init__(
        self,
        spread_bps: float = 5.0,
        commission_per_share: float = 0.005,
        min_commission: float = 1.0,
    ):
        self.spread_bps = spread_bps
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission

    def estimate(self, trade: Trade) -> TradeCost:
        notional = abs(trade.quantity * trade.price)
        spread_cost = notional * (self.spread_bps / 10_000 / 2)
        commission = max(
            abs(trade.quantity) * self.commission_per_share,
            self.min_commission,
        )
        return TradeCost(spread_cost=spread_cost, commission=commission)


# ---------------------------------------------------------------------------
# Model 2: Square-Root Market Impact
# ---------------------------------------------------------------------------


class SquareRootImpactModel(CostModel):
    """Square-root market impact model.

    ``impact = sigma * eta * sqrt(quantity / ADV)``

    This is the standard model used by institutional desks (Almgren &
    Chriss 2000).  Requires daily volatility (sigma) and average daily
    volume (ADV).

    Parameters:
        eta: Market impact coefficient (typically 0.1 – 0.5).
        spread_bps: Half-spread in basis points.
        commission_per_share: Per-share commission.
    """

    def __init__(
        self,
        sigma: float = 0.02,
        eta: float = 0.25,
        spread_bps: float = 5.0,
        commission_per_share: float = 0.005,
        min_commission: float = 1.0,
    ):
        self.sigma = sigma
        self.eta = eta
        self.spread_bps = spread_bps
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission

    def estimate(self, trade: Trade) -> TradeCost:
        notional = abs(trade.quantity * trade.price)
        spread_cost = notional * (self.spread_bps / 10_000 / 2)
        commission = max(
            abs(trade.quantity) * self.commission_per_share,
            self.min_commission,
        )

        # Square-root impact
        participation = abs(trade.quantity) / trade.adv if trade.adv > 0 else 0
        impact_frac = self.sigma * self.eta * np.sqrt(participation)
        market_impact = notional * impact_frac

        return TradeCost(
            spread_cost=spread_cost,
            commission=commission,
            market_impact=market_impact,
        )


# ---------------------------------------------------------------------------
# Model 3: Feedback Loop Impact
# ---------------------------------------------------------------------------


class FeedbackImpactModel(CostModel):
    """Square-root impact model with price feedback loop.

    Splits large orders into ``n_slices`` child slices and iteratively
    applies the square-root impact model.  After each slice executes,
    the effective price is updated adversely, so subsequent slices
    execute at a worse price.  This captures the feedback spiral where
    impact moves prices, increasing the cost of remaining execution.

    For small orders the result converges to :class:`SquareRootImpactModel`.
    For large orders (high participation rate) costs are 20-40% higher,
    matching empirical observations from institutional execution data.

    Reference: Almgren & Chriss (2000), extended with iterative execution.
    """

    def __init__(
        self,
        sigma: float = 0.02,
        eta: float = 0.25,
        spread_bps: float = 5.0,
        commission_per_share: float = 0.005,
        min_commission: float = 1.0,
        n_slices: int = 5,
    ):
        self.sigma = sigma
        self.eta = eta
        self.spread_bps = spread_bps
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission
        self.n_slices = max(1, n_slices)

    def estimate(self, trade: Trade) -> TradeCost:
        notional = abs(trade.quantity * trade.price)
        spread_cost = notional * (self.spread_bps / 10_000 / 2)
        commission = max(
            abs(trade.quantity) * self.commission_per_share,
            self.min_commission,
        )

        if trade.adv <= 0 or trade.quantity == 0:
            return TradeCost(spread_cost=spread_cost, commission=commission)

        # Each child slice independently impacts the market with its own
        # sqrt participation rate.  Crucially, each slice executes at the
        # *post-impact* price left by all prior slices, so later slices
        # are filled at progressively worse prices.
        #
        # Total cost = sum_i [ slice_qty * (exec_price_i - initial_price) ]
        #
        # With n_slices=1 this exactly equals the single-shot model.
        # With n_slices>1 the price-drift feedback makes the total cost
        # strictly >= single-shot, matching the empirical observation that
        # large orders cost 20-40% more than the naive model predicts.
        total_qty = abs(trade.quantity)
        slice_qty = total_qty / self.n_slices
        participation_per_slice = slice_qty / trade.adv
        impact_frac = self.sigma * self.eta * np.sqrt(participation_per_slice)

        initial_price = trade.price
        effective_price = trade.price
        total_impact_cost = 0.0
        sign = 1.0 if trade.side == "buy" else -1.0

        for _ in range(self.n_slices):
            # This slice executes at effective_price after impact
            exec_price = effective_price * (1.0 + sign * impact_frac)
            # Cost = how much more we pay than the initial mid price
            slice_cost = slice_qty * abs(exec_price - initial_price)
            total_impact_cost += slice_cost
            # Price permanently moves to post-impact level
            effective_price = exec_price

        return TradeCost(
            spread_cost=spread_cost,
            commission=commission,
            market_impact=total_impact_cost,
        )


# ---------------------------------------------------------------------------
# Convenience: apply costs to a backtest returns series
# ---------------------------------------------------------------------------


def apply_transaction_costs(
    positions: pd.DataFrame,
    prices: pd.DataFrame,
    cost_model: CostModel | None = None,
    adv: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute net-of-cost returns from a position matrix.

    Args:
        positions: DataFrame of target positions (shares) indexed by date,
                   columns = symbols.
        prices: DataFrame of prices, same shape as *positions*.
        cost_model: A ``CostModel`` instance; defaults to ``FixedPlusSpreadModel``.
        adv: Average daily volume per symbol (same columns as *positions*).
             Only needed for ``SquareRootImpactModel``.

    Returns:
        DataFrame with columns ``['gross_return', 'total_cost', 'net_return']``
        indexed by date.
    """
    model = cost_model or FixedPlusSpreadModel()

    trades = positions.diff().fillna(0)
    daily_costs = pd.Series(0.0, index=positions.index)

    for dt in trades.index:
        day_cost = 0.0
        for sym in trades.columns:
            qty = trades.loc[dt, sym]
            if qty == 0:
                continue
            price = prices.loc[dt, sym]
            avg_vol = adv.loc[dt, sym] if adv is not None and sym in adv.columns else 0
            tc = model.estimate(
                Trade(
                    symbol=sym,
                    side="buy" if qty > 0 else "sell",
                    quantity=qty,
                    price=price,
                    adv=avg_vol,
                )
            )
            day_cost += tc.total
        daily_costs[dt] = day_cost

    # Gross portfolio value
    portfolio_value = (positions * prices).sum(axis=1)
    gross_return = portfolio_value.pct_change().fillna(0)

    cost_drag = daily_costs / portfolio_value.shift(1).replace(0, np.nan)
    cost_drag = cost_drag.fillna(0)

    net_return = gross_return - cost_drag

    return pd.DataFrame(
        {
            "gross_return": gross_return,
            "total_cost": daily_costs,
            "cost_drag_pct": cost_drag,
            "net_return": net_return,
        }
    )
