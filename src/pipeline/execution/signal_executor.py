"""Signal executor: reads signal CSV and submits orders to Alpaca.

This module bridges the gap between signal generation (CSV output) and
broker execution.  It consumes the standard signal CSV format produced by
``signal_output.py`` and translates each row into a validated, risk-checked
order submitted to the broker.

Flow::

    signal CSV → parse → capital guard check → position size → submit order
                                                           → set stop-loss
                                                           → log trade

Every order passes through the capital guard (QAQC Layer 1) before reaching
the broker (Layer 2).  Post-execution, the reconciler (Layer 3) verifies
that broker state matches system state.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from pipeline.execution.broker import (
    BaseBroker,
    BrokerError,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
)
from pipeline.execution.capital_guard import CapitalGuard, CapitalGuardConfig
from pipeline.execution.reconciler import PositionReconciler, SystemPosition
from pipeline.infrastructure.notifier import AlertSeverity, notify
from pipeline.strategy.risk import DrawdownLevel, SwingRiskManager
from pipeline.strategy.sizing import PositionSizer, SizingConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal row (parsed from CSV)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParsedSignal:
    """A signal parsed from the CSV output."""

    date: pd.Timestamp
    ticker: str
    direction: str
    score: int
    entry_price: float
    stop_price: float
    target_1: float
    target_2: float
    atr: float
    regime: str
    confidence: str


# ---------------------------------------------------------------------------
# Execution result
# ---------------------------------------------------------------------------


@dataclass
class ExecutionResult:
    """Result of executing a batch of signals."""

    date: datetime
    signals_parsed: int
    signals_eligible: int
    orders_submitted: int
    orders_filled: int
    orders_rejected: int
    guard_rejections: int
    details: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"Execution {self.date.strftime('%Y-%m-%d')}: "
            f"{self.signals_parsed} signals → {self.signals_eligible} eligible → "
            f"{self.orders_submitted} submitted → {self.orders_filled} filled, "
            f"{self.orders_rejected} rejected, {self.guard_rejections} guard-blocked"
        )


# ---------------------------------------------------------------------------
# Signal executor
# ---------------------------------------------------------------------------


class SignalExecutor:
    """Execute trading signals through the broker with full QAQC.

    This is the main integration point between the strategy's signal
    generation and the broker's order submission.  It enforces:

    1. Capital guard checks (hard dollar caps, buying power, etc.)
    2. Position sizing (risk-based, micro-capital brackets)
    3. Risk management (drawdown circuit breakers, consecutive losses)
    4. Post-execution reconciliation

    Usage::

        executor = SignalExecutor(
            broker=alpaca_broker,
            guard_config=CapitalGuardConfig(max_capital=300),
        )

        # From signal CSV:
        result = executor.execute_signals("data/signals/signals_20250306.csv")

        # Or from DataFrame:
        result = executor.execute_signal_df(signals_df)
    """

    def __init__(
        self,
        broker: BaseBroker,
        guard_config: CapitalGuardConfig,
        sizing_config: SizingConfig | None = None,
        risk_manager: SwingRiskManager | None = None,
        reconciler: PositionReconciler | None = None,
        use_fractional_shares: bool = True,
        dry_run: bool = False,
        fill_poll_interval: int = 5,
        fill_poll_timeout: int = 120,
    ) -> None:
        """
        Args:
            broker: Broker implementation (Alpaca, etc.)
            guard_config: Capital guard configuration with hard limits.
            sizing_config: Position sizing config. Uses defaults if None.
            risk_manager: Risk manager instance. Creates new one if None.
            reconciler: Position reconciler. Creates new one if None.
            use_fractional_shares: Allow fractional shares (Alpaca supports).
            dry_run: If True, do everything except submit orders. Useful for
                validating the pipeline without risking capital.
            fill_poll_interval: Seconds between fill status checks for
                limit orders.
            fill_poll_timeout: Maximum seconds to wait for a limit order
                fill before cancelling.
        """
        self.broker = broker
        self.guard = CapitalGuard(config=guard_config, account_provider=broker)
        self.sizer = PositionSizer(sizing_config or SizingConfig())
        self.risk_mgr = risk_manager or SwingRiskManager()
        self.reconciler = reconciler or PositionReconciler(broker=broker)
        self.use_fractional_shares = use_fractional_shares
        self.dry_run = dry_run
        self.fill_poll_interval = fill_poll_interval
        self.fill_poll_timeout = fill_poll_timeout

        # Internal tracking for reconciliation
        self._system_positions: dict[str, SystemPosition] = {}
        self._initialized = False

    def initialize(self) -> None:
        """Initialize from current broker state.

        Must be called once before executing signals.  Syncs the risk
        manager's equity watermark and loads existing positions.
        """
        account = self.broker.get_account_snapshot()
        self.risk_mgr.initialize(account.equity)

        # Load existing positions into system tracking
        self._system_positions.clear()
        for pos in self.broker.get_positions():
            self._system_positions[pos.symbol] = SystemPosition(
                symbol=pos.symbol,
                qty=pos.qty,
                avg_entry_price=pos.avg_entry_price,
                side=pos.side,
            )

        self._initialized = True
        logger.info(
            "Executor initialized: equity=$%.2f, %d existing positions",
            account.equity,
            len(self._system_positions),
        )

    def execute_signals(self, csv_path: str | Path) -> ExecutionResult:
        """Execute signals from a CSV file.

        Args:
            csv_path: Path to the signal CSV (from ``write_signal_csv``).

        Returns:
            ExecutionResult with counts of what happened.
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Signal CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        return self.execute_signal_df(df)

    def execute_signal_df(self, signals_df: pd.DataFrame) -> ExecutionResult:
        """Execute signals from a DataFrame.

        Args:
            signals_df: DataFrame in the standard signal output format.

        Returns:
            ExecutionResult with counts of what happened.
        """
        if not self._initialized:
            self.initialize()

        result = ExecutionResult(
            date=datetime.now(UTC),
            signals_parsed=len(signals_df),
            signals_eligible=0,
            orders_submitted=0,
            orders_filled=0,
            orders_rejected=0,
            guard_rejections=0,
        )

        if signals_df.empty:
            logger.info("No signals to execute")
            return result

        # Parse signals
        signals = self._parse_signals(signals_df)
        result.signals_parsed = len(signals)

        # Get current account state for sizing
        account = self.broker.get_account_snapshot()
        equity = account.equity

        # Check risk state
        existing_positions = self.broker.get_positions()
        held_symbols = {p.symbol for p in existing_positions}
        total_risk_pct = 0.0  # Would compute from positions + ATR
        risk_state = self.risk_mgr.get_risk_state(
            current_equity=equity,
            open_positions=len(existing_positions),
            total_risk_pct=total_risk_pct,
        )

        if not risk_state.can_open_new:
            dd_name = DrawdownLevel(risk_state.drawdown_level).name
            logger.warning(
                "Risk state blocks new entries: level=%s, consecutive_losses=%d, "
                "cooldown=%d days",
                dd_name,
                risk_state.consecutive_losses,
                risk_state.cooldown_remaining_days,
            )
            notify(
                AlertSeverity.WARNING,
                "Entries Blocked by Risk State",
                f"Level={dd_name}, consecutive_losses={risk_state.consecutive_losses}, "
                f"cooldown={risk_state.cooldown_remaining_days}d. "
                f"{len(signals_df)} signal(s) will not be executed.",
                {
                    "drawdown_level": dd_name,
                    "consecutive_losses": risk_state.consecutive_losses,
                    "cooldown_days": risk_state.cooldown_remaining_days,
                },
            )
            result.details.append(
                {
                    "action": "BLOCKED_BY_RISK",
                    "drawdown_level": dd_name,
                }
            )
            return result

        # Track pending limit orders for fill polling
        pending_order_ids: list[str] = []

        # Execute each signal
        for signal in signals:
            if signal.ticker in held_symbols:
                logger.info("Skipping %s: already held", signal.ticker)
                result.details.append(
                    {
                        "ticker": signal.ticker,
                        "action": "SKIP_ALREADY_HELD",
                    }
                )
                continue

            if signal.regime == "BEAR":
                logger.info("Skipping %s: BEAR regime", signal.ticker)
                result.details.append(
                    {
                        "ticker": signal.ticker,
                        "action": "SKIP_BEAR_REGIME",
                    }
                )
                continue

            result.signals_eligible += 1

            # Compute position size
            dd_level = DrawdownLevel(risk_state.drawdown_level)
            min_score = self.risk_mgr.entry_score_threshold(dd_level)
            if signal.score < min_score:
                logger.info(
                    "Skipping %s: score %d < threshold %d (drawdown=%s)",
                    signal.ticker,
                    signal.score,
                    min_score,
                    dd_level.name,
                )
                result.details.append(
                    {
                        "ticker": signal.ticker,
                        "action": "SKIP_SCORE_TOO_LOW",
                        "score": signal.score,
                        "threshold": min_score,
                    }
                )
                continue

            # Refresh account for accurate sizing
            account = self.broker.get_account_snapshot()
            equity = account.equity

            size_result = self.sizer.compute(
                equity=equity,
                entry_price=signal.entry_price,
                atr=signal.atr,
                signal_score=signal.score,
                regime=signal.regime,
                current_positions=account.position_count,
                current_portfolio_risk_pct=total_risk_pct,
            )

            if size_result.rejected:
                logger.info(
                    "Sizer rejected %s: %s",
                    signal.ticker,
                    size_result.reject_reason,
                )
                result.details.append(
                    {
                        "ticker": signal.ticker,
                        "action": "SIZER_REJECTED",
                        "reason": size_result.reject_reason,
                    }
                )
                continue

            shares: int | float = size_result.shares
            if self.use_fractional_shares and shares == 0:
                # Try fractional for micro-capital
                shares = round(size_result.position_value / signal.entry_price, 4)
                if shares < 0.01:
                    continue

            # Capital guard check (QAQC Layer 1)
            guard_result = self.guard.check_order(
                symbol=signal.ticker,
                side="buy",
                shares=shares,
                limit_price=signal.entry_price,
            )

            if not guard_result.approved:
                logger.warning(
                    "Guard REJECTED %s: %s",
                    signal.ticker,
                    guard_result.summary(),
                )
                result.guard_rejections += 1
                result.details.append(
                    {
                        "ticker": signal.ticker,
                        "action": "GUARD_REJECTED",
                        "summary": guard_result.summary(),
                        "failed_checks": guard_result.checks_failed,
                    }
                )
                continue

            # Submit order
            if self.dry_run:
                logger.info(
                    "DRY RUN: would buy %s %.4f shares @ ~$%.2f",
                    signal.ticker,
                    shares,
                    signal.entry_price,
                )
                result.orders_submitted += 1
                result.details.append(
                    {
                        "ticker": signal.ticker,
                        "action": "DRY_RUN_APPROVED",
                        "shares": shares,
                        "price": signal.entry_price,
                        "stop": signal.stop_price,
                        "target": signal.target_1,
                    }
                )
                continue

            try:
                order = Order(
                    symbol=signal.ticker,
                    side=OrderSide.BUY,
                    order_type=OrderType.LIMIT,
                    qty=shares,
                    limit_price=signal.entry_price,
                )
                submitted = self.broker.submit_order(order)
                result.orders_submitted += 1

                if submitted.status == OrderStatus.FILLED:
                    result.orders_filled += 1
                    held_symbols.add(signal.ticker)
                    self._system_positions[signal.ticker] = SystemPosition(
                        symbol=signal.ticker,
                        qty=submitted.filled_qty or shares,
                        avg_entry_price=submitted.filled_avg_price or signal.entry_price,
                        side="long",
                    )
                elif submitted.status in (
                    OrderStatus.SUBMITTED,
                    OrderStatus.PENDING,
                    OrderStatus.PARTIAL,
                ):
                    pending_order_ids.append(submitted.order_id)

                result.details.append(
                    {
                        "ticker": signal.ticker,
                        "action": "ORDER_SUBMITTED",
                        "order_id": submitted.order_id,
                        "status": submitted.status.value,
                        "shares": shares,
                        "limit_price": signal.entry_price,
                        "stop_price": signal.stop_price,
                        "target_1": signal.target_1,
                    }
                )

                logger.info(
                    "ORDER %s: %s %.4f shares @ $%.2f limit, " "stop=$%.2f, target=$%.2f → %s",
                    submitted.order_id,
                    signal.ticker,
                    shares,
                    signal.entry_price,
                    signal.stop_price,
                    signal.target_1,
                    submitted.status.value,
                )
                notify(
                    AlertSeverity.INFO,
                    f"Order Submitted — {signal.ticker}",
                    f"{shares:.4f} shares @ ${signal.entry_price:.2f}"
                    f" limit → {submitted.status.value}",
                    {
                        "order_id": submitted.order_id,
                        "ticker": signal.ticker,
                        "shares": shares,
                        "limit_price": signal.entry_price,
                        "stop": signal.stop_price,
                        "target": signal.target_1,
                        "status": submitted.status.value,
                    },
                )

            except BrokerError as e:
                result.orders_rejected += 1
                result.details.append(
                    {
                        "ticker": signal.ticker,
                        "action": "BROKER_REJECTED",
                        "error": str(e),
                    }
                )
                logger.error("Broker rejected %s: %s", signal.ticker, e)

        # Poll for fills on submitted limit orders
        if not self.dry_run and pending_order_ids:
            logger.info(
                "Polling %d pending order(s) for fills (timeout=%ds)...",
                len(pending_order_ids),
                self.fill_poll_timeout,
            )
            fill_results = self.poll_pending_orders(pending_order_ids)
            for oid, order in fill_results.items():
                if order.status == OrderStatus.FILLED:
                    result.orders_filled += 1
                    held_symbols.add(order.symbol)
                    self._system_positions[order.symbol] = SystemPosition(
                        symbol=order.symbol,
                        qty=order.filled_qty,
                        avg_entry_price=order.filled_avg_price,
                        side="long",
                    )
                    notify(
                        AlertSeverity.INFO,
                        f"Order Filled — {order.symbol}",
                        f"{order.filled_qty:.4f} shares @ ${order.filled_avg_price:.2f}",
                        {"order_id": oid, "symbol": order.symbol},
                    )
                elif order.status in (
                    OrderStatus.CANCELLED,
                    OrderStatus.EXPIRED,
                ):
                    logger.info("Order %s (%s) → %s", oid, order.symbol, order.status.value)

        # Post-execution reconciliation (QAQC Layer 3)
        if not self.dry_run and result.orders_submitted > 0:
            recon = self.reconciler.reconcile(self._system_positions)
            if not recon.is_clean:
                logger.warning("Post-execution reconciliation: %s", recon.summary())
                result.details.append(
                    {
                        "action": "RECONCILIATION_DIRTY",
                        "summary": recon.summary(),
                    }
                )
            else:
                logger.info("Post-execution reconciliation: CLEAN")

        logger.info(result.summary())
        return result

    def poll_pending_orders(self, order_ids: list[str]) -> dict[str, Order]:
        """Poll broker for fill status on submitted limit orders.

        Waits up to ``fill_poll_timeout`` seconds, checking every
        ``fill_poll_interval`` seconds.  Orders that are still open after
        the timeout are cancelled.

        Args:
            order_ids: Broker order IDs to track.

        Returns:
            Mapping of order_id → final Order state.
        """
        if not order_ids:
            return {}

        pending = set(order_ids)
        results: dict[str, Order] = {}
        terminal = {
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        }
        deadline = time.monotonic() + self.fill_poll_timeout

        while pending and time.monotonic() < deadline:
            time.sleep(self.fill_poll_interval)
            for oid in list(pending):
                try:
                    order = self.broker.get_order_status(oid)
                    results[oid] = order
                    if order.status in terminal:
                        pending.discard(oid)
                        logger.info(
                            "Order %s → %s (filled=%.4f @ $%.2f)",
                            oid,
                            order.status.value,
                            order.filled_qty,
                            order.filled_avg_price,
                        )
                except (BrokerError, NotImplementedError) as e:
                    logger.warning("Failed to poll order %s: %s", oid, e)
                    pending.discard(oid)  # Remove from poll set on unrecoverable error

        # Cancel orders still open after timeout
        for oid in pending:
            logger.warning(
                "Order %s still open after %ds — cancelling",
                oid,
                self.fill_poll_timeout,
            )
            try:
                self.broker.cancel_order(oid)
                try:
                    final = self.broker.get_order_status(oid)
                    results[oid] = final
                except (BrokerError, NotImplementedError):
                    pass
            except (BrokerError, NotImplementedError) as e:
                logger.error("Failed to cancel timed-out order %s: %s", oid, e)

        return results

    def _parse_signals(self, df: pd.DataFrame) -> list[ParsedSignal]:
        """Parse a signal DataFrame into typed signal objects."""
        signals = []
        for _, row in df.iterrows():
            try:
                signals.append(
                    ParsedSignal(
                        date=pd.Timestamp(row["date"]),
                        ticker=str(row["ticker"]).upper(),
                        direction=str(row.get("direction", "LONG")),
                        score=int(row["score"]),
                        entry_price=float(row["entry_price"]),
                        stop_price=float(row["stop_price"]),
                        target_1=float(row["target_1"]),
                        target_2=float(row["target_2"]),
                        atr=float(row["atr"]),
                        regime=str(row.get("regime", "BULL")),
                        confidence=str(row.get("confidence", "LOW")),
                    )
                )
            except (KeyError, ValueError) as e:
                logger.warning("Failed to parse signal row: %s", e)
                continue

        # Sort by score descending (best signals first)
        signals.sort(key=lambda s: s.score, reverse=True)
        return signals
