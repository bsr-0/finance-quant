"""Pre-trade risk controls, intraday PnL monitoring, and kill-switch.

Implements the hard controls required for production trading (Goldman Sachs
rubric §5 – Risk, Controls, and Compliance):

* **Pre-trade checks** – hard limits on gross/net exposure, leverage,
  single-name concentration, and individual order size.
* **Intraday monitor** – tracks realised PnL and running drawdown; triggers
  automatic throttling or a full shutdown when configurable thresholds are
  breached.
* **Kill switch** – a global, thread-safe flag that immediately halts all
  order submission.
* **Anomaly detection** – flags unusual spikes in position/PnL that could
  indicate data errors or model misbehaviour.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RiskLimits:
    """Hard pre-trade and intraday risk limits.

    All monetary values are in the same currency as the portfolio.
    All fraction values are between 0 and 1 (e.g. 0.05 = 5 %).
    """

    # --- Pre-trade (order-level) ---
    max_order_notional: float = 1_000_000.0
    """Maximum notional value of a single order."""

    max_gross_exposure: float = 10_000_000.0
    """Maximum total long + short notional exposure."""

    max_net_exposure: float = 5_000_000.0
    """Maximum |long - short| notional exposure."""

    max_leverage: float = 4.0
    """Maximum gross exposure / net asset value."""

    max_concentration: float = 0.20
    """Maximum fraction of gross exposure in a single name."""

    # --- Intraday (session-level) ---
    max_daily_loss: float = 100_000.0
    """Maximum cumulative loss within a trading session before throttling."""

    max_drawdown_pct: float = 0.05
    """Maximum intraday drawdown (fraction of session-start NAV) before
    throttling.  A full shutdown is triggered at 2× this threshold."""

    # --- Anomaly detection ---
    pnl_spike_z_threshold: float = 5.0
    """Z-score threshold for flagging an anomalous single-period PnL move."""


# ---------------------------------------------------------------------------
# Pre-trade check result
# ---------------------------------------------------------------------------

class CheckStatus(Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    THROTTLED = "throttled"


@dataclass
class OrderCheckResult:
    """Result of a pre-trade risk check."""

    status: CheckStatus
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def approved(self) -> bool:
        return self.status == CheckStatus.APPROVED


# ---------------------------------------------------------------------------
# Kill switch
# ---------------------------------------------------------------------------

class KillSwitch:
    """Global, thread-safe trading halt mechanism.

    When *engaged* all pre-trade checks will reject orders regardless of
    limits.  The switch can only be reset by an explicit ``reset()`` call,
    ensuring that an operator must consciously restart trading.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._engaged: bool = False
        self._engaged_at: datetime | None = None
        self._reason: str = ""

    @property
    def engaged(self) -> bool:
        """Return ``True`` if the kill switch is currently active."""
        with self._lock:
            return self._engaged

    def engage(self, reason: str = "") -> None:
        """Engage the kill switch and halt all trading."""
        with self._lock:
            if not self._engaged:
                self._engaged = True
                self._engaged_at = datetime.now(UTC)
                self._reason = reason
                logger.critical(
                    "KILL SWITCH ENGAGED at %s – reason: %s",
                    self._engaged_at.isoformat(),
                    reason or "(no reason given)",
                )

    def reset(self, reason: str = "") -> None:
        """Reset the kill switch and allow trading to resume."""
        with self._lock:
            self._engaged = False
            self._engaged_at = None
            logger.warning("Kill switch RESET – reason: %s", reason or "(no reason given)")

    @property
    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "engaged": self._engaged,
                "engaged_at": self._engaged_at.isoformat() if self._engaged_at else None,
                "reason": self._reason,
            }


# ---------------------------------------------------------------------------
# Intraday risk monitor
# ---------------------------------------------------------------------------

@dataclass
class SessionState:
    """Mutable state tracked within a single trading session."""

    start_nav: float = 0.0
    peak_nav: float = 0.0
    realised_pnl: float = 0.0
    pnl_history: list[float] = field(default_factory=list)
    throttled: bool = False
    shutdown: bool = False


class IntradayRiskMonitor:
    """Track intraday PnL and drawdown; throttle or shutdown on breaches.

    Usage::

        monitor = IntradayRiskMonitor(limits, kill_switch)
        monitor.start_session(nav=1_000_000)

        for each_trade_pnl in trade_pnl_stream:
            action = monitor.record_pnl(each_trade_pnl)
            if action == "shutdown":
                break
    """

    def __init__(
        self,
        limits: RiskLimits,
        kill_switch: KillSwitch | None = None,
    ) -> None:
        self.limits = limits
        self.kill_switch = kill_switch or KillSwitch()
        self._session = SessionState()

    def start_session(self, nav: float) -> None:
        """Initialise a new trading session with the given starting NAV."""
        self._session = SessionState(start_nav=nav, peak_nav=nav)
        logger.info("Intraday monitor session started with NAV=%.2f", nav)

    def record_pnl(self, pnl_delta: float) -> str:
        """Record a PnL increment and check limits.

        Returns:
            ``"ok"`` – within limits.
            ``"throttle"`` – soft limit breached; slow down order flow.
            ``"shutdown"`` – hard limit breached; trading halted.
        """
        s = self._session
        s.realised_pnl += pnl_delta
        current_nav = s.start_nav + s.realised_pnl
        s.peak_nav = max(s.peak_nav, current_nav)
        s.pnl_history.append(pnl_delta)

        # Anomaly detection
        self._check_pnl_anomaly(pnl_delta)

        # Drawdown from intraday peak
        drawdown_pct = (current_nav - s.peak_nav) / s.peak_nav if s.peak_nav > 0 else 0.0
        hard_drawdown = 2 * self.limits.max_drawdown_pct

        if drawdown_pct <= -hard_drawdown or s.realised_pnl <= -self.limits.max_daily_loss * 2:
            s.shutdown = True
            reason = (
                f"Hard limit breached: drawdown={drawdown_pct:.2%}, "
                f"daily_loss={s.realised_pnl:.2f}"
            )
            self.kill_switch.engage(reason)
            logger.critical("SHUTDOWN triggered – %s", reason)
            return "shutdown"

        if (
            drawdown_pct <= -self.limits.max_drawdown_pct
            or s.realised_pnl <= -self.limits.max_daily_loss
        ):
            if not s.throttled:
                s.throttled = True
                logger.warning(
                    "THROTTLE engaged – drawdown=%.2f%%, daily_loss=%.2f",
                    abs(drawdown_pct) * 100,
                    abs(s.realised_pnl),
                )
            return "throttle"

        return "ok"

    def _check_pnl_anomaly(self, pnl_delta: float) -> None:
        """Emit a warning if the latest PnL move is an outlier."""
        history = self._session.pnl_history
        if len(history) < 10:
            return
        import numpy as np

        mu = float(np.mean(history[:-1]))
        sigma = float(np.std(history[:-1]))
        if sigma == 0:
            return
        z = abs((pnl_delta - mu) / sigma)
        if z > self.limits.pnl_spike_z_threshold:
            logger.warning(
                "PnL anomaly detected: delta=%.2f z-score=%.1f (threshold=%.1f)",
                pnl_delta,
                z,
                self.limits.pnl_spike_z_threshold,
            )

    @property
    def session_summary(self) -> dict[str, Any]:
        """Return a summary of the current session state."""
        s = self._session
        current_nav = s.start_nav + s.realised_pnl
        drawdown_pct = (current_nav - s.peak_nav) / s.peak_nav if s.peak_nav > 0 else 0.0
        return {
            "start_nav": s.start_nav,
            "current_nav": current_nav,
            "realised_pnl": s.realised_pnl,
            "peak_nav": s.peak_nav,
            "drawdown_pct": drawdown_pct,
            "throttled": s.throttled,
            "shutdown": s.shutdown,
        }


# ---------------------------------------------------------------------------
# Pre-trade checker
# ---------------------------------------------------------------------------

@dataclass
class PortfolioState:
    """Snapshot of the current portfolio for pre-trade checks."""

    net_asset_value: float
    """Current NAV (capital base)."""

    positions: dict[str, float] = field(default_factory=dict)
    """symbol → current notional (positive = long, negative = short)."""

    @property
    def gross_exposure(self) -> float:
        return sum(abs(v) for v in self.positions.values())

    @property
    def net_exposure(self) -> float:
        return abs(sum(self.positions.values()))

    @property
    def leverage(self) -> float:
        return self.gross_exposure / self.net_asset_value if self.net_asset_value > 0 else 0.0

    def concentration(self, symbol: str) -> float:
        """Single-name concentration after a proposed trade."""
        if self.gross_exposure == 0:
            return 0.0
        return abs(self.positions.get(symbol, 0.0)) / self.gross_exposure


class PreTradeChecker:
    """Validate proposed orders against hard risk limits.

    All checks are stateless – the caller provides the current
    ``PortfolioState`` and the proposed order.  The checker does *not*
    mutate state; it only approves or rejects.
    """

    def __init__(
        self,
        limits: RiskLimits,
        kill_switch: KillSwitch | None = None,
        intraday_monitor: IntradayRiskMonitor | None = None,
    ) -> None:
        self.limits = limits
        self.kill_switch = kill_switch or KillSwitch()
        self.intraday_monitor = intraday_monitor

    def check(
        self,
        symbol: str,
        side: str,
        notional: float,
        portfolio: PortfolioState,
    ) -> OrderCheckResult:
        """Run all pre-trade checks for a proposed order.

        Args:
            symbol: Instrument identifier.
            side: ``"buy"`` or ``"sell"``.
            notional: Unsigned notional value of the proposed order.
            portfolio: Current portfolio state.

        Returns:
            ``OrderCheckResult`` indicating approval/rejection and reason.
        """
        # Kill-switch overrides everything
        if self.kill_switch.engaged:
            return OrderCheckResult(
                status=CheckStatus.REJECTED,
                reason="Kill switch is engaged – no orders permitted",
                details=self.kill_switch.status,
            )

        # Throttle check
        if self.intraday_monitor and self.intraday_monitor._session.throttled:
            return OrderCheckResult(
                status=CheckStatus.THROTTLED,
                reason="Intraday throttle active – order flow restricted",
            )

        lim = self.limits

        # 1. Single-order notional limit
        if notional > lim.max_order_notional:
            return OrderCheckResult(
                status=CheckStatus.REJECTED,
                reason=(
                    f"Order notional {notional:,.0f} exceeds limit {lim.max_order_notional:,.0f}"
                ),
                details={"notional": notional, "limit": lim.max_order_notional},
            )

        # Project the portfolio state after this order
        delta = notional if side.lower() == "buy" else -notional
        new_positions = dict(portfolio.positions)
        new_positions[symbol] = new_positions.get(symbol, 0.0) + delta
        projected = PortfolioState(
            net_asset_value=portfolio.net_asset_value,
            positions=new_positions,
        )

        # 2. Gross exposure limit
        if projected.gross_exposure > lim.max_gross_exposure:
            return OrderCheckResult(
                status=CheckStatus.REJECTED,
                reason=(
                    f"Gross exposure {projected.gross_exposure:,.0f} would exceed "
                    f"limit {lim.max_gross_exposure:,.0f}"
                ),
                details={
                    "projected_gross": projected.gross_exposure,
                    "limit": lim.max_gross_exposure,
                },
            )

        # 3. Net exposure limit
        if projected.net_exposure > lim.max_net_exposure:
            return OrderCheckResult(
                status=CheckStatus.REJECTED,
                reason=(
                    f"Net exposure {projected.net_exposure:,.0f} would exceed "
                    f"limit {lim.max_net_exposure:,.0f}"
                ),
                details={
                    "projected_net": projected.net_exposure,
                    "limit": lim.max_net_exposure,
                },
            )

        # 4. Leverage limit
        if projected.leverage > lim.max_leverage:
            return OrderCheckResult(
                status=CheckStatus.REJECTED,
                reason=(
                    f"Leverage {projected.leverage:.2f}x would exceed "
                    f"limit {lim.max_leverage:.2f}x"
                ),
                details={
                    "projected_leverage": projected.leverage,
                    "limit": lim.max_leverage,
                },
            )

        # 5. Single-name concentration limit
        symbol_notional = abs(new_positions.get(symbol, 0.0))
        concentration = (
            symbol_notional / projected.gross_exposure
            if projected.gross_exposure > 0
            else 0.0
        )
        if concentration > lim.max_concentration:
            return OrderCheckResult(
                status=CheckStatus.REJECTED,
                reason=(
                    f"Concentration in {symbol} would be {concentration:.1%}, "
                    f"exceeding limit {lim.max_concentration:.1%}"
                ),
                details={
                    "symbol": symbol,
                    "projected_concentration": concentration,
                    "limit": lim.max_concentration,
                },
            )

        return OrderCheckResult(status=CheckStatus.APPROVED, reason="All checks passed")
