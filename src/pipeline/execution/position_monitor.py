"""Live position monitor: checks exit conditions on open positions.

Polls the broker for current prices and evaluates the same exit triggers
used in backtesting (stop-loss, trailing stop, profit target, time exit,
regime change) against live positions.  When an exit triggers, it submits
a market sell order through the broker.

Also handles RED circuit breaker conditions — if drawdown hits 15%,
all positions are immediately closed.

Usage::

    monitor = PositionMonitor(broker=alpaca_broker, guard_config=config)
    monitor.initialize()

    # Run once (e.g., called by scheduler every 5 minutes during market hours)
    exits = monitor.check_and_exit()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from pipeline.execution.broker import (
    BaseBroker,
    BrokerError,
)
from pipeline.execution.capital_guard import CapitalGuardConfig, CapitalGuard
from pipeline.execution.realtime_feed import RealtimePriceFeed
from pipeline.infrastructure.notifier import AlertSeverity, notify
from pipeline.strategy.exits import ExitEngine, ExitReason, PositionState
from pipeline.strategy.risk import DrawdownLevel, SwingRiskManager

logger = logging.getLogger(__name__)


@dataclass
class ExitAction:
    """Record of an exit action taken by the monitor."""

    symbol: str
    reason: ExitReason
    shares: float
    exit_price: float
    pnl_estimate: float
    order_id: str = ""
    success: bool = False
    error: str = ""


@dataclass
class MonitorResult:
    """Result of a single monitoring cycle."""

    timestamp: datetime
    positions_checked: int
    exits_triggered: int
    exits_executed: int
    exits_failed: int
    circuit_breaker_level: str
    actions: list[ExitAction] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"Monitor {self.timestamp.strftime('%H:%M:%S')}: "
            f"{self.positions_checked} positions checked, "
            f"{self.exits_triggered} exits triggered, "
            f"{self.exits_executed} executed, "
            f"{self.exits_failed} failed, "
            f"circuit_breaker={self.circuit_breaker_level}"
        )


@dataclass
class TrackedPosition:
    """Extended position tracking with entry metadata for exit logic."""

    symbol: str
    entry_date: datetime
    entry_price: float
    shares: float
    stop_price: float
    atr_at_entry: float
    trailing_stop: float = 0.0
    trailing_activated: bool = False
    highest_price: float = 0.0
    target_1: float = 0.0
    target_2: float = 0.0
    signal_score: int = 0

    def to_position_state(self) -> PositionState:
        """Convert to the backtest engine's PositionState for exit checking."""
        import pandas as pd
        ps = PositionState(
            symbol=self.symbol,
            entry_date=pd.Timestamp(self.entry_date),
            entry_price=self.entry_price,
            shares=int(self.shares),
            stop_price=self.stop_price,
            atr_at_entry=self.atr_at_entry,
        )
        ps.trailing_stop = self.trailing_stop
        ps.trailing_activated = self.trailing_activated
        ps.highest_price = self.highest_price or self.entry_price
        # Only override targets if explicitly set (non-zero)
        if self.target_1 > 0:
            ps.target_1 = self.target_1
        if self.target_2 > 0:
            ps.target_2 = self.target_2
        return ps


class PositionMonitor:
    """Monitor open positions and execute exits based on strategy rules.

    Reuses the same ExitEngine from the backtest to ensure consistency
    between backtested and live exit behavior.
    """

    def __init__(
        self,
        broker: BaseBroker,
        guard_config: CapitalGuardConfig,
        exit_engine: ExitEngine | None = None,
        risk_manager: SwingRiskManager | None = None,
        regime: str = "BULL",
        realtime_feed: RealtimePriceFeed | None = None,
    ) -> None:
        self.broker = broker
        self.guard = CapitalGuard(config=guard_config, account_provider=broker)
        self.exit_engine = exit_engine or ExitEngine()
        self.risk_mgr = risk_manager or SwingRiskManager()
        self.regime = regime
        self.realtime_feed = realtime_feed

        # Tracked positions with entry metadata
        self._tracked: dict[str, TrackedPosition] = {}
        self._initialized = False

    def initialize(self) -> None:
        """Sync with broker and initialize risk state.

        Only initializes the risk manager if it hasn't been initialized
        already (preserves peak equity from prior sessions).
        """
        account = self.broker.get_account_snapshot()
        if self.risk_mgr._peak_equity <= 0:
            self.risk_mgr.initialize(account.equity)
        self._initialized = True
        logger.info("Position monitor initialized: equity=$%.2f", account.equity)

    def register_position(self, tracked: TrackedPosition) -> None:
        """Register a position with entry metadata for exit tracking.

        Call this after each successful entry order fill.
        """
        self._tracked[tracked.symbol] = tracked
        logger.info(
            "Registered position: %s %.2f shares @ $%.2f, "
            "stop=$%.2f, target=$%.2f",
            tracked.symbol, tracked.shares, tracked.entry_price,
            tracked.stop_price, tracked.target_1,
        )

    def set_regime(self, regime: str) -> None:
        """Update the current market regime (BULL/NEUTRAL/BEAR)."""
        old = self.regime
        self.regime = regime.upper()
        if old != self.regime:
            logger.info("Regime changed: %s → %s", old, self.regime)

    def check_and_exit(self) -> MonitorResult:
        """Check all positions for exit conditions and execute exits.

        Returns:
            MonitorResult with details of any exits taken.
        """
        import numpy as np
        import pandas as pd

        if not self._initialized:
            self.initialize()

        now = datetime.now(timezone.utc)
        result = MonitorResult(
            timestamp=now,
            positions_checked=0,
            exits_triggered=0,
            exits_executed=0,
            exits_failed=0,
            circuit_breaker_level="GREEN",
        )

        # Get current account state
        account = self.broker.get_account_snapshot()
        positions = self.broker.get_positions()
        position_map = {p.symbol: p for p in positions}

        result.positions_checked = len(positions)

        # Check drawdown / circuit breaker
        risk_state = self.risk_mgr.get_risk_state(
            current_equity=account.equity,
            open_positions=len(positions),
            total_risk_pct=0.0,
        )
        dd_level = DrawdownLevel(risk_state.drawdown_level)
        result.circuit_breaker_level = dd_level.name

        # RED circuit breaker: close everything immediately
        if dd_level >= DrawdownLevel.RED:
            logger.critical(
                "RED CIRCUIT BREAKER: drawdown=%.2f%%, closing all positions",
                abs(risk_state.drawdown_pct) * 100,
            )
            notify(
                AlertSeverity.CRITICAL,
                "RED Circuit Breaker — Closing All Positions",
                f"Drawdown {abs(risk_state.drawdown_pct) * 100:.1f}%. "
                f"Emergency close of {len(positions)} position(s).",
                {
                    "equity": round(account.equity, 2),
                    "drawdown_pct": round(abs(risk_state.drawdown_pct) * 100, 1),
                    "positions": [p.symbol for p in positions],
                },
            )
            try:
                close_orders = self.broker.close_all_positions()
                for order in close_orders:
                    result.actions.append(ExitAction(
                        symbol=order.symbol,
                        reason=ExitReason.STOP_LOSS,
                        shares=order.qty,
                        exit_price=0.0,  # market order
                        pnl_estimate=0.0,
                        order_id=order.order_id,
                        success=True,
                    ))
                result.exits_triggered = len(close_orders)
                result.exits_executed = len(close_orders)
                self._tracked.clear()
            except BrokerError as e:
                logger.critical("Failed to close all positions: %s", e)
                notify(
                    AlertSeverity.CRITICAL,
                    "FAILED to Close Positions",
                    f"RED circuit breaker fired but broker close failed: {e}",
                    {"positions": [p.symbol for p in positions]},
                )
                result.exits_failed = len(positions)

            return result

        # Check each tracked position for exit conditions
        for symbol, tracked in list(self._tracked.items()):
            broker_pos = position_map.get(symbol)
            if broker_pos is None:
                # Position closed externally; remove tracking
                logger.warning(
                    "Position %s not found in broker; removing from tracking",
                    symbol,
                )
                del self._tracked[symbol]
                continue

            # Build state for exit engine — prefer real-time prices when available
            pos_state = tracked.to_position_state()
            rt_quote = (
                self.realtime_feed.get_latest(symbol)
                if self.realtime_feed and self.realtime_feed.is_running
                else None
            )

            if rt_quote and not self.realtime_feed.is_stale(symbol):
                current_close = rt_quote.price
                current_high = rt_quote.high if rt_quote.high > 0 else rt_quote.price
                logger.debug(
                    "%s: using realtime price $%.2f (high=$%.2f, age=%.0fs)",
                    symbol, current_close, current_high, rt_quote.age_seconds,
                )
            else:
                current_close = broker_pos.current_price
                current_high = broker_pos.current_price
                if rt_quote:
                    logger.warning(
                        "%s: realtime quote stale (%.0fs), using broker price $%.2f",
                        symbol, rt_quote.age_seconds, current_close,
                    )

            exit_signal = self.exit_engine.check_exit(
                position=pos_state,
                current_date=pd.Timestamp(now),
                current_close=current_close,
                current_high=current_high,
                current_atr=tracked.atr_at_entry,  # Use entry ATR as approximation
                current_rsi=50.0,  # Would need real-time RSI; use neutral
                current_sma_50=np.nan,  # Would need real-time SMA; skip
                regime=self.regime,
            )

            # Sync trailing stop state back
            tracked.trailing_stop = pos_state.trailing_stop
            tracked.trailing_activated = pos_state.trailing_activated
            tracked.highest_price = pos_state.highest_price

            if exit_signal.should_exit:
                result.exits_triggered += 1
                pnl = (current_close - tracked.entry_price) * tracked.shares

                logger.info(
                    "EXIT TRIGGERED %s: reason=%s, price=$%.2f, "
                    "entry=$%.2f, P&L=$%.2f",
                    symbol, exit_signal.reason.value, current_close,
                    tracked.entry_price, pnl,
                )

                sev = AlertSeverity.WARNING if exit_signal.reason in (
                    ExitReason.STOP_LOSS, ExitReason.REGIME_BEAR,
                ) else AlertSeverity.INFO
                notify(
                    sev,
                    f"Exit Triggered — {symbol}",
                    f"Reason: {exit_signal.reason.value}, "
                    f"price=${current_close:.2f}, P&L=${pnl:+.2f}",
                    {
                        "symbol": symbol,
                        "reason": exit_signal.reason.value,
                        "entry_price": round(tracked.entry_price, 2),
                        "exit_price": round(current_close, 2),
                        "pnl": round(pnl, 2),
                        "shares": tracked.shares,
                    },
                )

                try:
                    order = self.broker.close_position(symbol)
                    result.exits_executed += 1
                    self.risk_mgr.record_trade_result(pnl)
                    del self._tracked[symbol]

                    result.actions.append(ExitAction(
                        symbol=symbol,
                        reason=exit_signal.reason,
                        shares=tracked.shares,
                        exit_price=current_close,
                        pnl_estimate=pnl,
                        order_id=order.order_id,
                        success=True,
                    ))

                except BrokerError as e:
                    result.exits_failed += 1
                    result.actions.append(ExitAction(
                        symbol=symbol,
                        reason=exit_signal.reason,
                        shares=tracked.shares,
                        exit_price=current_close,
                        pnl_estimate=pnl,
                        success=False,
                        error=str(e),
                    ))
                    logger.error("Failed to close %s: %s", symbol, e)

        logger.info(result.summary())
        return result

    def start_realtime_feed(self) -> None:
        """Start the real-time feed for currently tracked symbols.

        Creates a feed from environment variables if one wasn't provided
        at construction time.  No-op if the feed is already running.
        """
        symbols = list(self._tracked.keys())
        if not symbols:
            logger.info("No tracked positions — skipping realtime feed start")
            return

        if self.realtime_feed is None:
            self.realtime_feed = RealtimePriceFeed.create_for_positions(symbols)

        if not self.realtime_feed.is_running:
            # Ensure all tracked symbols are subscribed
            self.realtime_feed.add_symbols(symbols)
            self.realtime_feed.start()

    def stop_realtime_feed(self) -> None:
        """Stop the real-time feed if running."""
        if self.realtime_feed and self.realtime_feed.is_running:
            self.realtime_feed.stop()

    @property
    def tracked_positions(self) -> dict[str, TrackedPosition]:
        """Current tracked positions (read-only view)."""
        return dict(self._tracked)
