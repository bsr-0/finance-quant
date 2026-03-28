"""Institutional-grade position sizing for systematic strategies.

Supports multiple sizing methods suitable for managing $10B+ in
global equity capital:
  - Fixed fraction of capital
  - Volatility-scaled (target risk per position)
  - Equal risk contribution
  - Signal-strength weighted
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SizingMethod(Enum):
    FIXED_FRACTION = "fixed_fraction"
    VOLATILITY_SCALED = "volatility_scaled"
    EQUAL_RISK = "equal_risk"
    SIGNAL_WEIGHTED = "signal_weighted"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class InstitutionalSizingConfig:
    """Configuration for institutional position sizing."""

    method: SizingMethod = SizingMethod.VOLATILITY_SCALED
    total_capital: float = 1e10  # $10B

    # Fixed fraction parameters
    equal_weight_fraction: float = 0.02  # 2% per name

    # Volatility-scaled parameters
    target_annual_vol: float = 0.10  # 10% annualized portfolio vol target
    target_position_risk: float = 0.005  # 50 bps per position at risk
    vol_lookback_days: int = 60
    vol_floor: float = 0.05  # Minimum annualized vol assumption
    vol_cap: float = 1.00  # Maximum annualized vol assumption

    # Position constraints
    max_position_weight: float = 0.05  # 5% max per name
    min_position_weight: float = 0.001  # 10 bps minimum
    max_gross_exposure: float = 2.0  # 200% gross (100% long + 100% short)
    max_net_exposure: float = 1.0  # 100% net long
    max_adv_participation: float = 0.10  # Max 10% of ADV per day
    min_trade_notional: float = 100_000  # $100K minimum trade

    # Conviction scaling
    conviction_scale_min: float = 0.5
    conviction_scale_max: float = 1.5


# ---------------------------------------------------------------------------
# Position sizing result
# ---------------------------------------------------------------------------

@dataclass
class PositionTarget:
    """Target position for a single instrument."""

    ticker: str
    target_weight: float  # Fraction of capital
    target_notional: float  # Dollar amount
    target_shares: int  # Share/contract count
    signal_value: float = 0.0
    volatility: float = 0.0
    risk_contribution: float = 0.0
    adv_participation: float = 0.0
    constrained: bool = False
    constraint_reason: str = ""


@dataclass
class PortfolioTargets:
    """Complete set of position targets for the portfolio."""

    positions: list[PositionTarget]
    total_capital: float
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    total_risk: float = 0.0
    date: pd.Timestamp | None = None

    @property
    def long_exposure(self) -> float:
        return sum(p.target_notional for p in self.positions if p.target_weight > 0)

    @property
    def short_exposure(self) -> float:
        return abs(sum(p.target_notional for p in self.positions if p.target_weight < 0))

    @property
    def position_count(self) -> int:
        return len(self.positions)

    def weight_series(self) -> pd.Series:
        return pd.Series(
            {p.ticker: p.target_weight for p in self.positions}
        )

    def notional_series(self) -> pd.Series:
        return pd.Series(
            {p.ticker: p.target_notional for p in self.positions}
        )

    def shares_series(self) -> pd.Series:
        return pd.Series(
            {p.ticker: p.target_shares for p in self.positions}
        )


# ---------------------------------------------------------------------------
# Sizing models
# ---------------------------------------------------------------------------

class PositionSizingModel(ABC):
    """Base class for position sizing models."""

    @abstractmethod
    def compute_targets(
        self,
        signals: pd.Series,
        prices: pd.Series,
        volatilities: pd.Series,
        adv: pd.Series | None = None,
        capital: float = 1e10,
    ) -> PortfolioTargets:
        """Compute target positions given signals and market data.

        Args:
            signals: Signal values per ticker (higher = more conviction).
            prices: Current prices per ticker.
            volatilities: Annualized volatility per ticker.
            adv: Average daily volume (shares) per ticker.
            capital: Total portfolio capital.
        """

    @property
    @abstractmethod
    def method_name(self) -> str:
        """Human-readable method name."""

    @property
    def formula(self) -> str:
        """LaTeX-ready formula for the sizing method."""
        return ""


class FixedFractionSizer(PositionSizingModel):
    r"""Equal-weight position sizing.

    .. math::
        w_i = \frac{1}{N} \cdot \text{sign}(s_i)
    """

    def __init__(self, config: InstitutionalSizingConfig | None = None) -> None:
        self.config = config or InstitutionalSizingConfig()

    @property
    def method_name(self) -> str:
        return "Fixed Fraction"

    @property
    def formula(self) -> str:
        return r"w_i = \frac{1}{N} \cdot \text{sign}(s_i)"

    def compute_targets(
        self,
        signals: pd.Series,
        prices: pd.Series,
        volatilities: pd.Series,
        adv: pd.Series | None = None,
        capital: float = 1e10,
    ) -> PortfolioTargets:
        cfg = self.config
        positions: list[PositionTarget] = []

        active = signals[signals != 0].dropna()
        if active.empty:
            return PortfolioTargets(positions=[], total_capital=capital)

        weight_per_name = cfg.equal_weight_fraction
        for ticker in active.index:
            sig = active[ticker]
            direction = np.sign(sig)
            raw_weight = direction * weight_per_name

            # Constrain
            constrained = False
            reason = ""
            if abs(raw_weight) > cfg.max_position_weight:
                raw_weight = np.sign(raw_weight) * cfg.max_position_weight
                constrained = True
                reason = "max_position_weight"

            notional = raw_weight * capital
            price = prices.get(ticker, 0.0)
            shares = int(notional / price) if price > 0 else 0

            if abs(notional) < cfg.min_trade_notional:
                continue

            adv_part = abs(shares) / adv.get(ticker, 1e12) if adv is not None else 0.0

            positions.append(PositionTarget(
                ticker=ticker,
                target_weight=raw_weight,
                target_notional=notional,
                target_shares=shares,
                signal_value=float(sig),
                volatility=volatilities.get(ticker, 0.0),
                adv_participation=adv_part,
                constrained=constrained,
                constraint_reason=reason,
            ))

        gross = sum(abs(p.target_notional) for p in positions)
        net = sum(p.target_notional for p in positions)
        return PortfolioTargets(
            positions=positions,
            total_capital=capital,
            gross_exposure=gross / capital if capital > 0 else 0,
            net_exposure=net / capital if capital > 0 else 0,
        )


class VolatilityScaledSizer(PositionSizingModel):
    r"""Volatility-scaled position sizing (inverse volatility weighting).

    .. math::
        w_i = \frac{\sigma_{\text{target}}}{\sigma_i \cdot \sqrt{N}}
              \cdot \text{sign}(s_i) \cdot c_i

    where :math:`c_i` is the conviction scalar derived from signal strength.
    """

    def __init__(self, config: InstitutionalSizingConfig | None = None) -> None:
        self.config = config or InstitutionalSizingConfig()

    @property
    def method_name(self) -> str:
        return "Volatility-Scaled"

    @property
    def formula(self) -> str:
        return (
            r"w_i = \frac{\sigma_{\text{target}}}{\sigma_i \cdot \sqrt{N}}"
            r" \cdot \text{sign}(s_i) \cdot c_i"
        )

    def compute_targets(
        self,
        signals: pd.Series,
        prices: pd.Series,
        volatilities: pd.Series,
        adv: pd.Series | None = None,
        capital: float = 1e10,
    ) -> PortfolioTargets:
        cfg = self.config
        positions: list[PositionTarget] = []

        active = signals[signals != 0].dropna()
        if active.empty:
            return PortfolioTargets(positions=[], total_capital=capital)

        n_positions = len(active)
        target_vol = cfg.target_annual_vol

        for ticker in active.index:
            sig = float(active[ticker])
            direction = np.sign(sig)
            vol = volatilities.get(ticker, cfg.vol_floor)
            vol = np.clip(vol, cfg.vol_floor, cfg.vol_cap)

            # Conviction scaling: map signal magnitude to [min, max] multiplier
            sig_abs = abs(sig)
            sig_max = float(active.abs().max()) if active.abs().max() > 0 else 1.0
            conviction = cfg.conviction_scale_min + (
                (cfg.conviction_scale_max - cfg.conviction_scale_min)
                * (sig_abs / sig_max)
            )

            # Base weight: target vol / (individual vol * sqrt(N))
            raw_weight = direction * (target_vol / (vol * math.sqrt(n_positions))) * conviction

            # Apply constraints
            constrained = False
            reason = ""
            if abs(raw_weight) > cfg.max_position_weight:
                raw_weight = np.sign(raw_weight) * cfg.max_position_weight
                constrained = True
                reason = "max_position_weight"

            if abs(raw_weight) < cfg.min_position_weight:
                continue

            notional = raw_weight * capital
            price = prices.get(ticker, 0.0)
            shares = int(notional / price) if price > 0 else 0

            if abs(notional) < cfg.min_trade_notional:
                continue

            # ADV check
            adv_part = 0.0
            if adv is not None:
                ticker_adv = adv.get(ticker, 0.0)
                if ticker_adv > 0:
                    adv_part = abs(shares) / ticker_adv
                    if adv_part > cfg.max_adv_participation:
                        max_shares = int(ticker_adv * cfg.max_adv_participation)
                        shares = max_shares * int(direction)
                        notional = shares * price
                        raw_weight = notional / capital if capital > 0 else 0
                        constrained = True
                        reason = "adv_participation"

            risk_contribution = abs(raw_weight) * vol

            positions.append(PositionTarget(
                ticker=ticker,
                target_weight=raw_weight,
                target_notional=notional,
                target_shares=shares,
                signal_value=sig,
                volatility=vol,
                risk_contribution=risk_contribution,
                adv_participation=adv_part,
                constrained=constrained,
                constraint_reason=reason,
            ))

        # Gross/net exposure check
        gross = sum(abs(p.target_weight) for p in positions)
        net = sum(p.target_weight for p in positions)

        if gross > cfg.max_gross_exposure:
            scale = cfg.max_gross_exposure / gross
            for p in positions:
                p.target_weight *= scale
                p.target_notional *= scale
                p.target_shares = int(p.target_shares * scale)

        total_risk = sum(p.risk_contribution for p in positions)

        return PortfolioTargets(
            positions=positions,
            total_capital=capital,
            gross_exposure=gross,
            net_exposure=net,
            total_risk=total_risk,
        )


class SignalWeightedSizer(PositionSizingModel):
    r"""Position sizing proportional to signal strength, volatility-adjusted.

    .. math::
        w_i = \frac{s_i / \sigma_i}{\sum_j |s_j / \sigma_j|} \cdot L

    where :math:`L` is the target leverage (gross exposure).
    """

    def __init__(self, config: InstitutionalSizingConfig | None = None) -> None:
        self.config = config or InstitutionalSizingConfig()

    @property
    def method_name(self) -> str:
        return "Signal-Weighted (Vol-Adjusted)"

    @property
    def formula(self) -> str:
        return (
            r"w_i = \frac{s_i / \sigma_i}{\sum_j |s_j / \sigma_j|} \cdot L"
        )

    def compute_targets(
        self,
        signals: pd.Series,
        prices: pd.Series,
        volatilities: pd.Series,
        adv: pd.Series | None = None,
        capital: float = 1e10,
    ) -> PortfolioTargets:
        cfg = self.config
        positions: list[PositionTarget] = []

        active = signals[signals != 0].dropna()
        if active.empty:
            return PortfolioTargets(positions=[], total_capital=capital)

        # Compute vol-adjusted signals
        vol_adj: dict[str, float] = {}
        for ticker in active.index:
            vol = volatilities.get(ticker, cfg.vol_floor)
            vol = max(vol, cfg.vol_floor)
            vol_adj[ticker] = float(active[ticker]) / vol

        total_abs = sum(abs(v) for v in vol_adj.values())
        if total_abs == 0:
            return PortfolioTargets(positions=[], total_capital=capital)

        target_leverage = min(cfg.max_gross_exposure, 1.0)

        for ticker, va_signal in vol_adj.items():
            raw_weight = (va_signal / total_abs) * target_leverage

            constrained = False
            reason = ""
            if abs(raw_weight) > cfg.max_position_weight:
                raw_weight = np.sign(raw_weight) * cfg.max_position_weight
                constrained = True
                reason = "max_position_weight"

            if abs(raw_weight) < cfg.min_position_weight:
                continue

            notional = raw_weight * capital
            price = prices.get(ticker, 0.0)
            shares = int(notional / price) if price > 0 else 0

            if abs(notional) < cfg.min_trade_notional:
                continue

            vol = volatilities.get(ticker, cfg.vol_floor)
            adv_part = abs(shares) / adv.get(ticker, 1e12) if adv is not None else 0.0

            positions.append(PositionTarget(
                ticker=ticker,
                target_weight=raw_weight,
                target_notional=notional,
                target_shares=shares,
                signal_value=float(active[ticker]),
                volatility=vol,
                risk_contribution=abs(raw_weight) * vol,
                adv_participation=adv_part,
                constrained=constrained,
                constraint_reason=reason,
            ))

        gross = sum(abs(p.target_weight) for p in positions)
        net = sum(p.target_weight for p in positions)
        total_risk = sum(p.risk_contribution for p in positions)

        return PortfolioTargets(
            positions=positions,
            total_capital=capital,
            gross_exposure=gross,
            net_exposure=net,
            total_risk=total_risk,
        )


def create_sizer(config: InstitutionalSizingConfig) -> PositionSizingModel:
    """Factory to create the appropriate sizing model from config."""
    if config.method == SizingMethod.FIXED_FRACTION:
        return FixedFractionSizer(config)
    if config.method == SizingMethod.VOLATILITY_SCALED:
        return VolatilityScaledSizer(config)
    if config.method == SizingMethod.SIGNAL_WEIGHTED:
        return SignalWeightedSizer(config)
    return VolatilityScaledSizer(config)
