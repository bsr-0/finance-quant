"""Capital guard: hard limits on total deployed capital.

This module provides the outermost QAQC layer that ensures you NEVER invest
more than a configured maximum, regardless of what the strategy engine or
position sizer calculates.  It queries real broker account state (buying power,
equity, positions) and enforces hard caps before any order reaches the broker.

Three independent layers prevent over-investment:

    Layer 1 (this module) — Pre-order capital guard
        Hard cap on total capital deployed.  Rejects orders that would exceed
        the user-configured maximum investment.

    Layer 2 — Broker-side enforcement
        Alpaca cash accounts physically reject orders exceeding settled cash.
        Margin accounts are explicitly blocked by this guard.

    Layer 3 — Post-trade reconciliation
        Daily comparison of system state vs broker state detects any drift.

Design principle: every check is *independent*.  Even if one layer has a bug,
the other two prevent overdraft.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Account data protocol (broker-agnostic)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AccountSnapshot:
    """Immutable snapshot of broker account state.

    All values come directly from the broker API — never from internal
    tracking — so they reflect reality even if the system has a bug.
    """

    equity: float
    """Total account value (cash + long market value - short market value)."""

    cash: float
    """Settled cash balance."""

    buying_power: float
    """Broker-reported buying power (for cash accounts, equals settled cash)."""

    positions_market_value: float
    """Sum of all open position market values."""

    position_count: int
    """Number of distinct open positions."""

    is_margin_account: bool
    """True if the account has margin enabled."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    """When this snapshot was taken."""


class AccountProvider(Protocol):
    """Protocol for fetching live account data from a broker."""

    def get_account_snapshot(self) -> AccountSnapshot: ...


# ---------------------------------------------------------------------------
# Guard configuration
# ---------------------------------------------------------------------------


@dataclass
class CapitalGuardConfig:
    """Configuration for the capital guard.

    These are HARD limits — they override everything the strategy engine
    computes.  Set them conservatively.
    """

    max_capital: float
    """Absolute maximum dollars you are willing to have deployed at any time.
    This is the single most important safety parameter.  Example: if you
    deposit $500 but only want to risk $300, set this to 300."""

    max_single_order_dollars: float = 0.0
    """Maximum notional value for any single order.  Set to 0 to auto-derive
    from max_capital (defaults to max_capital * max_single_order_pct)."""

    max_single_order_pct: float = 0.60
    """Maximum single order as fraction of max_capital (used when
    max_single_order_dollars is 0)."""

    max_positions: int = 2
    """Maximum number of simultaneous positions."""

    require_cash_account: bool = True
    """If True, reject ALL orders if the broker account has margin enabled.
    Cash accounts make overdraft physically impossible at the broker level."""

    min_cash_buffer_pct: float = 0.05
    """Reserve this fraction of max_capital as a cash buffer.  Orders that
    would leave less than this in cash are rejected.  Protects against
    rounding, fees, and settlement timing."""

    max_daily_orders: int = 10
    """Maximum number of orders per calendar day.  Prevents runaway loops."""

    max_portfolio_utilization: float = 0.95
    """Maximum fraction of equity that can be in positions.  At 0.95, you
    always keep at least 5% in cash."""

    def __post_init__(self) -> None:
        if self.max_capital <= 0:
            raise ValueError(f"max_capital must be positive, got {self.max_capital}")
        if self.max_single_order_dollars == 0:
            self.max_single_order_dollars = self.max_capital * self.max_single_order_pct


# ---------------------------------------------------------------------------
# Check result
# ---------------------------------------------------------------------------


class GuardVerdict(Enum):
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass
class GuardCheckResult:
    """Result of a capital guard check."""

    verdict: GuardVerdict
    checks_run: list[str] = field(default_factory=list)
    checks_passed: list[str] = field(default_factory=list)
    checks_failed: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def approved(self) -> bool:
        return self.verdict == GuardVerdict.APPROVED

    def summary(self) -> str:
        status = "APPROVED" if self.approved else "REJECTED"
        failed = ", ".join(self.checks_failed) if self.checks_failed else "none"
        return (
            f"{status} ({len(self.checks_passed)}/"
            f"{len(self.checks_run)} passed, failed: {failed})"
        )


# ---------------------------------------------------------------------------
# Capital guard
# ---------------------------------------------------------------------------


class CapitalGuard:
    """Pre-order capital guard with hard investment limits.

    Usage::

        guard = CapitalGuard(
            config=CapitalGuardConfig(max_capital=300.0),
            account_provider=alpaca_broker,
        )

        result = guard.check_order("AAPL", "buy", shares=2, limit_price=150.0)
        if not result.approved:
            print(f"BLOCKED: {result.summary()}")
    """

    def __init__(
        self,
        config: CapitalGuardConfig,
        account_provider: AccountProvider,
    ) -> None:
        self.config = config
        self._account_provider = account_provider
        self._daily_order_count: int = 0
        self._daily_order_date: str = ""

    def check_order(
        self,
        symbol: str,
        side: str,
        shares: float,
        limit_price: float,
    ) -> GuardCheckResult:
        """Run all QAQC checks before allowing an order.

        Args:
            symbol: Ticker symbol.
            side: "buy" or "sell".
            shares: Number of shares (can be fractional).
            limit_price: Limit price (use last price for market orders).

        Returns:
            GuardCheckResult with pass/fail for every check.
        """
        order_notional = abs(shares * limit_price)
        result = GuardCheckResult(verdict=GuardVerdict.APPROVED)

        # Sell orders only need basic checks
        if side.lower() == "sell":
            result.checks_run.append("sell_passthrough")
            result.checks_passed.append("sell_passthrough")
            logger.info("Capital guard: APPROVED sell order for %s (sells always allowed)", symbol)
            return result

        # Fetch live account state from broker
        try:
            account = self._account_provider.get_account_snapshot()
        except Exception as e:
            result.verdict = GuardVerdict.REJECTED
            result.checks_run.append("account_fetch")
            result.checks_failed.append("account_fetch")
            result.details["error"] = str(e)
            logger.error("Capital guard: REJECTED — cannot fetch account state: %s", e)
            return result

        result.details["account_equity"] = account.equity
        result.details["account_cash"] = account.cash
        result.details["account_buying_power"] = account.buying_power
        result.details["positions_market_value"] = account.positions_market_value
        result.details["order_notional"] = order_notional

        checks = [
            ("margin_check", self._check_margin_account, account, order_notional),
            ("max_capital_check", self._check_max_capital, account, order_notional),
            ("buying_power_check", self._check_buying_power, account, order_notional),
            ("single_order_check", self._check_single_order_size, account, order_notional),
            ("position_count_check", self._check_position_count, account, order_notional),
            ("cash_buffer_check", self._check_cash_buffer, account, order_notional),
            ("utilization_check", self._check_utilization, account, order_notional),
            ("daily_order_limit", self._check_daily_order_count, account, order_notional),
        ]

        for name, check_fn, acct, notional in checks:
            result.checks_run.append(name)
            passed, reason = check_fn(acct, notional)
            if passed:
                result.checks_passed.append(name)
            else:
                result.checks_failed.append(name)
                result.details[name] = reason
                result.verdict = GuardVerdict.REJECTED
                logger.warning("Capital guard check FAILED [%s]: %s", name, reason)

        if result.approved:
            self._increment_daily_orders()
            logger.info(
                "Capital guard: APPROVED buy %s %.4f shares @ $%.2f ($%.2f notional). "
                "Account equity=$%.2f, deployed=$%.2f, remaining=$%.2f",
                symbol,
                shares,
                limit_price,
                order_notional,
                account.equity,
                account.positions_market_value,
                account.buying_power,
            )
        else:
            logger.warning(
                "Capital guard: REJECTED buy %s — %s",
                symbol,
                result.summary(),
            )

        return result

    # --- Individual checks ---

    def _check_margin_account(
        self,
        account: AccountSnapshot,
        order_notional: float,
    ) -> tuple[bool, str]:
        """QAQC-1: Reject if margin account and require_cash_account is set."""
        if self.config.require_cash_account and account.is_margin_account:
            return False, (
                "Account has margin enabled. Cash account required to prevent "
                "overdraft. Disable margin in your broker settings."
            )
        return True, ""

    def _check_max_capital(
        self,
        account: AccountSnapshot,
        order_notional: float,
    ) -> tuple[bool, str]:
        """QAQC-2: Total deployed + this order must not exceed max_capital."""
        total_after = account.positions_market_value + order_notional
        if total_after > self.config.max_capital:
            return False, (
                f"Would deploy ${total_after:.2f} total "
                f"(current ${account.positions_market_value:.2f} + "
                f"order ${order_notional:.2f}), "
                f"exceeding max_capital ${self.config.max_capital:.2f}"
            )
        return True, ""

    def _check_buying_power(
        self,
        account: AccountSnapshot,
        order_notional: float,
    ) -> tuple[bool, str]:
        """QAQC-3: Order must not exceed broker-reported buying power."""
        if order_notional > account.buying_power:
            return False, (
                f"Order ${order_notional:.2f} exceeds broker buying power "
                f"${account.buying_power:.2f}"
            )
        return True, ""

    def _check_single_order_size(
        self,
        account: AccountSnapshot,
        order_notional: float,
    ) -> tuple[bool, str]:
        """QAQC-4: Single order must not exceed per-order dollar limit."""
        if order_notional > self.config.max_single_order_dollars:
            return False, (
                f"Order ${order_notional:.2f} exceeds single-order limit "
                f"${self.config.max_single_order_dollars:.2f}"
            )
        return True, ""

    def _check_position_count(
        self,
        account: AccountSnapshot,
        order_notional: float,
    ) -> tuple[bool, str]:
        """QAQC-5: Must not exceed max simultaneous positions."""
        if account.position_count >= self.config.max_positions:
            return False, (
                f"Already at {account.position_count} positions, "
                f"max is {self.config.max_positions}"
            )
        return True, ""

    def _check_cash_buffer(
        self,
        account: AccountSnapshot,
        order_notional: float,
    ) -> tuple[bool, str]:
        """QAQC-6: Must keep a minimum cash buffer after the order."""
        min_buffer = self.config.max_capital * self.config.min_cash_buffer_pct
        cash_after = account.cash - order_notional
        if cash_after < min_buffer:
            return False, (
                f"Cash after order would be ${cash_after:.2f}, "
                f"below required buffer ${min_buffer:.2f} "
                f"({self.config.min_cash_buffer_pct:.0%} of max_capital)"
            )
        return True, ""

    def _check_utilization(
        self,
        account: AccountSnapshot,
        order_notional: float,
    ) -> tuple[bool, str]:
        """QAQC-7: Portfolio utilization must stay below max."""
        if account.equity <= 0:
            return False, "Account equity is zero or negative"
        total_positions = account.positions_market_value + order_notional
        utilization = total_positions / account.equity
        if utilization > self.config.max_portfolio_utilization:
            return False, (
                f"Portfolio utilization would be {utilization:.1%}, "
                f"exceeding max {self.config.max_portfolio_utilization:.1%}"
            )
        return True, ""

    def _check_daily_order_count(
        self,
        account: AccountSnapshot,
        order_notional: float,
    ) -> tuple[bool, str]:
        """QAQC-8: Must not exceed daily order count limit."""
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        if self._daily_order_date != today:
            self._daily_order_count = 0
            self._daily_order_date = today

        if self._daily_order_count >= self.config.max_daily_orders:
            return False, (
                f"Already placed {self._daily_order_count} orders today, "
                f"max is {self.config.max_daily_orders}"
            )
        return True, ""

    def _increment_daily_orders(self) -> None:
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        if self._daily_order_date != today:
            self._daily_order_count = 0
            self._daily_order_date = today
        self._daily_order_count += 1
