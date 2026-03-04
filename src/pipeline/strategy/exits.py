"""Exit logic for the Trend-Aligned Pullback Reversion strategy.

Implements a layered exit framework with five independent triggers:
  1. Stop-loss (capital preservation)
  2. Regime change to BEAR (systemic risk)
  3. Signal reversal (thesis invalidated)
  4. Profit target (take profits)
  5. Time-based (avoid dead money)

The first trigger hit closes the position.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ExitReason(Enum):
    NONE = "none"
    STOP_LOSS = "stop_loss"
    TRAILING_STOP = "trailing_stop"
    REGIME_BEAR = "regime_bear"
    TREND_REVERSAL = "trend_reversal"
    RSI_OVERBOUGHT = "rsi_overbought"
    PROFIT_TARGET = "profit_target"
    TIME_EXIT = "time_exit"


@dataclass
class ExitSignal:
    """Result of exit evaluation for a single position."""

    should_exit: bool
    reason: ExitReason
    exit_price: float = 0.0
    pnl_dollars: float = 0.0
    pnl_pct: float = 0.0
    days_held: int = 0


@dataclass
class PositionState:
    """Tracked state for an open position."""

    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: int
    stop_price: float
    atr_at_entry: float
    trailing_stop: float = 0.0
    trailing_activated: bool = False
    highest_price: float = 0.0
    target_1: float = 0.0
    target_2: float = 0.0

    def __post_init__(self) -> None:
        self.highest_price = self.entry_price
        self.trailing_stop = self.stop_price
        self.target_1 = self.entry_price + self.atr_at_entry * 2.0
        self.target_2 = self.entry_price + self.atr_at_entry * 3.0


class ExitEngine:
    """Evaluate exit conditions for open positions."""

    def __init__(
        self,
        max_holding_days: int = 15,
        stop_atr_multiple: float = 1.5,
        trailing_atr_multiple: float = 2.0,
        trailing_activation_atr: float = 1.0,
        target_atr_multiple: float = 2.0,
        rsi_overbought: float = 70.0,
    ) -> None:
        self.max_holding_days = max_holding_days
        self.stop_atr_multiple = stop_atr_multiple
        self.trailing_atr_multiple = trailing_atr_multiple
        self.trailing_activation_atr = trailing_activation_atr
        self.target_atr_multiple = target_atr_multiple
        self.rsi_overbought = rsi_overbought

    def check_exit(
        self,
        position: PositionState,
        current_date: pd.Timestamp,
        current_close: float,
        current_high: float,
        current_atr: float,
        current_rsi: float,
        current_sma_50: float,
        regime: str,
    ) -> ExitSignal:
        """Evaluate all exit conditions for a position.

        Checks triggers in priority order and returns the first one that fires.
        """
        days_held = (current_date - position.entry_date).days
        pnl_dollars = (current_close - position.entry_price) * position.shares
        pnl_pct = (current_close - position.entry_price) / position.entry_price

        # Update trailing state
        position.highest_price = max(position.highest_price, current_high)
        unrealized_atr = (position.highest_price - position.entry_price) / position.atr_at_entry

        if unrealized_atr >= self.trailing_activation_atr and not position.trailing_activated:
            position.trailing_activated = True
            position.trailing_stop = position.highest_price - current_atr * self.trailing_atr_multiple
            logger.debug(
                "%s: trailing stop activated at $%.2f",
                position.symbol, position.trailing_stop,
            )

        if position.trailing_activated:
            new_trail = position.highest_price - current_atr * self.trailing_atr_multiple
            position.trailing_stop = max(position.trailing_stop, new_trail)

        def _signal(reason: ExitReason) -> ExitSignal:
            return ExitSignal(
                should_exit=True,
                reason=reason,
                exit_price=current_close,
                pnl_dollars=pnl_dollars,
                pnl_pct=pnl_pct,
                days_held=days_held,
            )

        # --- Priority 1: Hard stop-loss ---
        if current_close < position.stop_price:
            logger.info(
                "EXIT %s: STOP LOSS hit at $%.2f (stop=$%.2f, loss=$%.2f)",
                position.symbol, current_close, position.stop_price, pnl_dollars,
            )
            return _signal(ExitReason.STOP_LOSS)

        # --- Priority 1b: Trailing stop ---
        if position.trailing_activated and current_close < position.trailing_stop:
            logger.info(
                "EXIT %s: TRAILING STOP at $%.2f (trail=$%.2f, P&L=$%.2f)",
                position.symbol, current_close, position.trailing_stop, pnl_dollars,
            )
            return _signal(ExitReason.TRAILING_STOP)

        # --- Priority 2: Regime change to BEAR ---
        if regime == "BEAR":
            logger.info(
                "EXIT %s: REGIME BEAR detected, closing at $%.2f, P&L=$%.2f",
                position.symbol, current_close, pnl_dollars,
            )
            return _signal(ExitReason.REGIME_BEAR)

        # --- Priority 3: Signal reversal (trend break) ---
        if not np.isnan(current_sma_50) and current_close < current_sma_50:
            logger.info(
                "EXIT %s: TREND REVERSAL, close $%.2f < SMA50 $%.2f, P&L=$%.2f",
                position.symbol, current_close, current_sma_50, pnl_dollars,
            )
            return _signal(ExitReason.TREND_REVERSAL)

        # --- Priority 3b: RSI overbought (take profit) ---
        if current_rsi > self.rsi_overbought and pnl_pct > 0:
            logger.info(
                "EXIT %s: RSI OVERBOUGHT %.1f with profit, P&L=$%.2f",
                position.symbol, current_rsi, pnl_dollars,
            )
            return _signal(ExitReason.RSI_OVERBOUGHT)

        # --- Priority 4: Profit target ---
        if current_close >= position.target_1:
            logger.info(
                "EXIT %s: PROFIT TARGET at $%.2f (target=$%.2f, P&L=$%.2f)",
                position.symbol, current_close, position.target_1, pnl_dollars,
            )
            return _signal(ExitReason.PROFIT_TARGET)

        # --- Priority 5: Time-based exit ---
        if days_held >= self.max_holding_days:
            logger.info(
                "EXIT %s: TIME EXIT after %d days, P&L=$%.2f",
                position.symbol, days_held, pnl_dollars,
            )
            return _signal(ExitReason.TIME_EXIT)

        # No exit triggered
        return ExitSignal(
            should_exit=False,
            reason=ExitReason.NONE,
            pnl_dollars=pnl_dollars,
            pnl_pct=pnl_pct,
            days_held=days_held,
        )
