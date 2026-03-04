"""Portfolio-level risk management for the swing strategy.

Implements drawdown circuit breakers, correlation checks, and portfolio-level
risk budgeting tailored to micro-capital accounts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DrawdownLevel(IntEnum):
    """Drawdown severity levels for circuit breakers."""

    GREEN = 0   # Normal operation
    YELLOW = 1  # 5% drawdown — reduce size
    ORANGE = 2  # 10% drawdown — no new entries
    RED = 3     # 15% drawdown — close everything, cooldown


@dataclass
class RiskState:
    """Current risk state of the portfolio."""

    equity: float
    peak_equity: float
    drawdown_pct: float
    drawdown_level: DrawdownLevel
    total_risk_pct: float
    open_positions: int
    consecutive_losses: int
    can_open_new: bool
    cooldown_remaining_days: int = 0


class SwingRiskManager:
    """Portfolio-level risk management and drawdown circuit breakers."""

    def __init__(
        self,
        yellow_threshold: float = 0.05,
        orange_threshold: float = 0.10,
        red_threshold: float = 0.15,
        max_consecutive_losses: int = 4,
        cooldown_days: int = 30,
        max_correlation: float = 0.70,
        max_daily_loss_pct: float = 0.02,
        max_portfolio_risk_pct: float = 0.03,
    ) -> None:
        self.yellow_threshold = yellow_threshold
        self.orange_threshold = orange_threshold
        self.red_threshold = red_threshold
        self.max_consecutive_losses = max_consecutive_losses
        self.cooldown_days = cooldown_days
        self.max_correlation = max_correlation
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_portfolio_risk_pct = max_portfolio_risk_pct

        # Mutable state
        self._peak_equity: float = 0.0
        self._consecutive_losses: int = 0
        self._cooldown_remaining: int = 0
        self._trade_history: list[float] = []

    def initialize(self, starting_equity: float) -> None:
        """Set the starting equity and reset state."""
        self._peak_equity = starting_equity
        self._consecutive_losses = 0
        self._cooldown_remaining = 0
        self._trade_history = []

    def update_equity(self, current_equity: float) -> None:
        """Update the peak equity watermark."""
        self._peak_equity = max(self._peak_equity, current_equity)

    def record_trade_result(self, pnl: float) -> None:
        """Record a completed trade P&L for consecutive-loss tracking."""
        self._trade_history.append(pnl)
        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

    def tick_cooldown(self) -> None:
        """Advance the cooldown timer by one day."""
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

    def get_drawdown_level(self, current_equity: float) -> DrawdownLevel:
        """Compute the current drawdown level."""
        if self._peak_equity <= 0:
            return DrawdownLevel.GREEN

        dd = (current_equity - self._peak_equity) / self._peak_equity

        if dd <= -self.red_threshold:
            return DrawdownLevel.RED
        if dd <= -self.orange_threshold:
            return DrawdownLevel.ORANGE
        if dd <= -self.yellow_threshold:
            return DrawdownLevel.YELLOW
        return DrawdownLevel.GREEN

    def get_risk_state(
        self,
        current_equity: float,
        open_positions: int,
        total_risk_pct: float,
        daily_return: float = 0.0,
    ) -> RiskState:
        """Compute the full risk state of the portfolio.

        Args:
            current_equity: Current total equity (cash + positions).
            open_positions: Number of open positions.
            total_risk_pct: Sum of risk from open positions as fraction of equity.
            daily_return: Today's return so far (for daily loss limit check).
        """
        self.update_equity(current_equity)
        dd_pct = (current_equity - self._peak_equity) / self._peak_equity if self._peak_equity > 0 else 0
        dd_level = self.get_drawdown_level(current_equity)

        # Determine if new entries are allowed
        can_open = True
        if dd_level >= DrawdownLevel.ORANGE:
            can_open = False
        if self._cooldown_remaining > 0:
            can_open = False
        if self._consecutive_losses >= self.max_consecutive_losses:
            can_open = False
        if total_risk_pct >= self.max_portfolio_risk_pct:
            can_open = False
        # Daily loss limit: block new entries if today's loss exceeds threshold
        if daily_return < -self.max_daily_loss_pct:
            can_open = False
            logger.warning(
                "Daily loss limit hit: %.2f%% exceeds max %.2f%%. Blocking new entries.",
                abs(daily_return) * 100,
                self.max_daily_loss_pct * 100,
            )

        # Trigger cooldown on RED
        if dd_level == DrawdownLevel.RED and self._cooldown_remaining == 0:
            self._cooldown_remaining = self.cooldown_days
            logger.critical(
                "RED ALERT: %.1f%% drawdown. Entering %d-day cooldown.",
                abs(dd_pct) * 100, self.cooldown_days,
            )

        return RiskState(
            equity=current_equity,
            peak_equity=self._peak_equity,
            drawdown_pct=dd_pct,
            drawdown_level=dd_level,
            total_risk_pct=total_risk_pct,
            open_positions=open_positions,
            consecutive_losses=self._consecutive_losses,
            can_open_new=can_open,
            cooldown_remaining_days=self._cooldown_remaining,
        )

    def check_correlation(
        self,
        candidate_returns: pd.Series,
        existing_returns: dict[str, pd.Series],
        window: int = 60,
    ) -> tuple[bool, str]:
        """Check if a candidate position is too correlated with existing ones.

        Returns:
            (passes, reason): ``True`` if the candidate passes the check.
        """
        for symbol, ret_series in existing_returns.items():
            aligned_cand, aligned_exist = candidate_returns.align(ret_series, join="inner")
            if len(aligned_cand) < 20:
                continue
            # Use the most recent *window* observations
            recent_cand = aligned_cand.iloc[-window:]
            recent_exist = aligned_exist.iloc[-window:]
            corr = recent_cand.corr(recent_exist)
            if not np.isnan(corr) and abs(corr) > self.max_correlation:
                reason = f"Correlation with {symbol} = {corr:.2f} exceeds {self.max_correlation}"
                logger.warning("Correlation check FAILED: %s", reason)
                return False, reason

        return True, "OK"

    def entry_score_threshold(self, dd_level: DrawdownLevel) -> int:
        """Return the minimum signal score required at the current drawdown level."""
        if dd_level == DrawdownLevel.YELLOW:
            return 75
        if dd_level >= DrawdownLevel.ORANGE:
            return 999  # Effectively blocks entries
        return 60

    def position_size_multiplier(self, dd_level: DrawdownLevel) -> float:
        """Return the position size multiplier for the current drawdown level."""
        if dd_level == DrawdownLevel.YELLOW:
            return 0.5
        if dd_level >= DrawdownLevel.ORANGE:
            return 0.0
        return 1.0
