"""Position sizing for micro-capital swing accounts.

Implements risk-based position sizing: the number of shares is determined by
how much capital the trader is willing to lose if the stop-loss is hit, not
by a fixed dollar amount.

The risk fraction scales with account size because very small accounts face
practical constraints (minimum 1 share, discrete lot sizes).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SizingConfig:
    """Configuration for the position sizer."""

    # Risk fraction schedule (by account equity bracket)
    risk_fraction_small: float = 0.015   # $100-$500: 1.5%
    risk_fraction_large: float = 0.010   # $500+: 1.0%
    risk_bracket_threshold: float = 500.0

    # Maximum position as fraction of equity (by account bracket)
    max_position_pct_tiny: float = 1.00   # $100-$250: 100%
    max_position_pct_small: float = 0.60  # $250-$500: 60%
    max_position_pct_medium: float = 0.40 # $500-$1000: 40%
    max_position_pct_large: float = 0.30  # $1000+: 30%

    # Max simultaneous positions (by bracket)
    max_positions_tiny: int = 1    # $100-$250
    max_positions_small: int = 2   # $250-$500
    max_positions_medium: int = 3  # $500-$1000
    max_positions_large: int = 4   # $1000+

    # Portfolio-level risk cap
    max_portfolio_risk_pct: float = 0.03  # 3% total portfolio risk

    # ATR multiplier for stop-loss distance
    stop_atr_multiple: float = 1.5

    # Minimum share price filter
    min_share_price: float = 5.0


@dataclass
class SizeResult:
    """Output of the position sizer."""

    shares: int
    position_value: float
    risk_per_share: float
    total_risk: float
    risk_pct_of_equity: float
    stop_price: float
    rejected: bool = False
    reject_reason: str = ""


class PositionSizer:
    """Compute position sizes based on risk budget and account constraints."""

    def __init__(self, config: SizingConfig | None = None) -> None:
        self.config = config or SizingConfig()

    def _risk_fraction(self, equity: float) -> float:
        if equity < self.config.risk_bracket_threshold:
            return self.config.risk_fraction_small
        return self.config.risk_fraction_large

    def _max_position_pct(self, equity: float) -> float:
        if equity < 250:
            return self.config.max_position_pct_tiny
        if equity < 500:
            return self.config.max_position_pct_small
        if equity < 1000:
            return self.config.max_position_pct_medium
        return self.config.max_position_pct_large

    def max_positions(self, equity: float) -> int:
        if equity < 250:
            return self.config.max_positions_tiny
        if equity < 500:
            return self.config.max_positions_small
        if equity < 1000:
            return self.config.max_positions_medium
        return self.config.max_positions_large

    def compute(
        self,
        equity: float,
        entry_price: float,
        atr: float,
        signal_score: int,
        regime: str,
        current_positions: int = 0,
        current_portfolio_risk_pct: float = 0.0,
    ) -> SizeResult:
        """Compute the position size for a proposed trade.

        Args:
            equity: Current account equity (cash + positions).
            entry_price: Expected entry price.
            atr: Current ATR(14) for the symbol.
            signal_score: Composite signal score (0-100).
            regime: Market regime (``BULL``, ``NEUTRAL``, ``BEAR``).
            current_positions: Number of currently open positions.
            current_portfolio_risk_pct: Sum of risk from open positions as
                fraction of equity.

        Returns:
            ``SizeResult`` with the computed shares, stop price, and risk.
        """
        cfg = self.config

        # --- Pre-checks ---
        if regime == "BEAR":
            return SizeResult(
                shares=0, position_value=0, risk_per_share=0,
                total_risk=0, risk_pct_of_equity=0, stop_price=0,
                rejected=True, reject_reason="BEAR regime — no new positions",
            )

        if entry_price < cfg.min_share_price:
            return SizeResult(
                shares=0, position_value=0, risk_per_share=0,
                total_risk=0, risk_pct_of_equity=0, stop_price=0,
                rejected=True, reject_reason=f"Price ${entry_price:.2f} below minimum ${cfg.min_share_price:.2f}",
            )

        max_pos = self.max_positions(equity)
        if current_positions >= max_pos:
            return SizeResult(
                shares=0, position_value=0, risk_per_share=0,
                total_risk=0, risk_pct_of_equity=0, stop_price=0,
                rejected=True, reject_reason=f"Max positions ({max_pos}) reached",
            )

        # --- Stop price ---
        stop_distance = atr * cfg.stop_atr_multiple
        stop_price = entry_price - stop_distance
        risk_per_share = stop_distance

        if risk_per_share <= 0:
            return SizeResult(
                shares=0, position_value=0, risk_per_share=0,
                total_risk=0, risk_pct_of_equity=0, stop_price=0,
                rejected=True, reject_reason="ATR is zero or negative",
            )

        # --- Risk budget ---
        risk_frac = self._risk_fraction(equity)
        risk_budget = equity * risk_frac

        # Conviction scaling
        if signal_score >= 80:
            conviction = 1.0
        elif signal_score >= 70:
            conviction = 0.75
        else:
            conviction = 0.50
        risk_budget *= conviction

        # Regime scaling
        regime_mult = 1.0 if regime == "BULL" else 0.5
        risk_budget *= regime_mult

        # Portfolio risk headroom
        remaining_risk = cfg.max_portfolio_risk_pct - current_portfolio_risk_pct
        if remaining_risk <= 0:
            return SizeResult(
                shares=0, position_value=0, risk_per_share=0,
                total_risk=0, risk_pct_of_equity=0, stop_price=stop_price,
                rejected=True, reject_reason="Portfolio risk budget exhausted",
            )
        max_risk_for_this_trade = equity * remaining_risk
        risk_budget = min(risk_budget, max_risk_for_this_trade)

        # --- Shares ---
        raw_shares = risk_budget / risk_per_share
        shares = max(math.floor(raw_shares), 0)

        # Enforce max position size
        max_pos_value = equity * self._max_position_pct(equity)
        max_shares_by_value = math.floor(max_pos_value / entry_price)
        shares = min(shares, max_shares_by_value)

        # Enforce minimum 1 share if we have any budget
        if shares == 0 and raw_shares > 0:
            one_share_value = entry_price
            one_share_risk = risk_per_share / equity
            if one_share_value <= max_pos_value and one_share_risk <= remaining_risk:
                shares = 1

        if shares <= 0:
            return SizeResult(
                shares=0, position_value=0, risk_per_share=risk_per_share,
                total_risk=0, risk_pct_of_equity=0, stop_price=stop_price,
                rejected=True, reject_reason="Insufficient equity for minimum position",
            )

        position_value = shares * entry_price
        total_risk = shares * risk_per_share
        risk_pct = total_risk / equity if equity > 0 else 0

        logger.info(
            "Sized %d shares @ $%.2f = $%.2f (risk $%.2f = %.2f%% of equity, "
            "conviction=%.0f%%, regime=%s)",
            shares,
            entry_price,
            position_value,
            total_risk,
            risk_pct * 100,
            conviction * 100,
            regime,
        )

        return SizeResult(
            shares=shares,
            position_value=position_value,
            risk_per_share=risk_per_share,
            total_risk=total_risk,
            risk_pct_of_equity=risk_pct,
            stop_price=stop_price,
        )
