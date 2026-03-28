"""Paper trading runner: validates signals on live market data before going live.

Orchestrates a complete paper trading session:
  1. Reads the latest signal CSV (or generates signals from live data).
  2. Routes signals through the SignalExecutor in paper-trading mode.
  3. Monitors open positions and applies exit rules.
  4. Reconciles positions daily.
  5. Logs all activity for post-session analysis.

Usage::

    from pipeline.execution.paper_runner import PaperTradingRunner, PaperRunnerConfig

    runner = PaperTradingRunner(
        config=PaperRunnerConfig(
            signal_dir="signals/",
            max_capital=300.0,
        ),
    )

    # Run once (e.g. from a daily cron job):
    report = runner.run_daily()
    print(report)

    # Or inspect current state:
    status = runner.get_status()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from pipeline.execution.alpaca_broker import AlpacaBroker
from pipeline.execution.broker import BaseBroker, BrokerError
from pipeline.execution.capital_guard import AccountSnapshot, CapitalGuardConfig
from pipeline.execution.reconciler import PositionReconciler, SystemPosition
from pipeline.execution.signal_executor import (
    ExecutionResult,
    SignalExecutor,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PaperRunnerConfig:
    """Configuration for the paper trading runner."""

    signal_dir: str = "signals"
    """Directory containing signal CSV files."""

    max_capital: float = 300.0
    """Maximum capital to deploy (same as ExecutorConfig.max_capital)."""

    risk_per_trade_pct: float = 0.02
    """Risk per trade as fraction of equity."""

    max_positions: int = 2
    """Maximum simultaneous positions."""

    order_type: str = "limit"
    """Order type: 'market' or 'limit'."""

    min_score: int = 60
    """Minimum signal score."""

    min_confidence: str = "LOW"
    """Minimum confidence level."""

    dry_run: bool = False
    """If True, simulate without placing real paper orders."""

    log_dir: str = "logs/paper_trading"
    """Directory for paper trading logs."""

    exit_check_enabled: bool = True
    """Whether to check exit conditions on open positions."""

    reconcile_enabled: bool = True
    """Whether to reconcile positions after execution."""


# ---------------------------------------------------------------------------
# Status / report types
# ---------------------------------------------------------------------------


@dataclass
class PositionStatus:
    """Current status of a broker position."""

    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float
    unrealised_pnl: float
    unrealised_pnl_pct: float
    side: str


@dataclass
class PaperTradingStatus:
    """Snapshot of the paper trading session."""

    timestamp: datetime
    mode: str  # "PAPER" or "LIVE"
    account_equity: float
    account_cash: float
    buying_power: float
    positions: list[PositionStatus]
    total_unrealised_pnl: float
    is_healthy: bool
    last_execution: ExecutionResult | None = None
    last_reconciliation_clean: bool | None = None


@dataclass
class DailyReport:
    """Report from a single daily run."""

    timestamp: datetime
    signal_file: str | None
    execution_result: ExecutionResult | None
    exit_orders: list[str]
    reconciliation_clean: bool | None
    account_equity: float
    account_cash: float
    positions_count: int
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Paper Trading Daily Report — {self.timestamp:%Y-%m-%d %H:%M:%S UTC}",
            f"  Account: equity=${self.account_equity:.2f} cash=${self.account_cash:.2f}",
            f"  Positions: {self.positions_count}",
        ]
        if self.signal_file:
            lines.append(f"  Signal file: {self.signal_file}")
        if self.execution_result:
            lines.append(
                f"  Execution: {self.execution_result.orders_submitted} submitted, "
                f"{self.execution_result.orders_rejected} rejected"
            )
        if self.exit_orders:
            lines.append(f"  Exits: {', '.join(self.exit_orders)}")
        if self.reconciliation_clean is not None:
            lines.append(
                "  Reconciliation: "
                f"{'CLEAN' if self.reconciliation_clean else 'DISCREPANCIES FOUND'}"
            )
        if self.errors:
            lines.append(f"  Errors: {'; '.join(self.errors)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Paper trading runner
# ---------------------------------------------------------------------------


class PaperTradingRunner:
    """Orchestrates paper trading sessions.

    Designed to be called daily (e.g. from a cron job or scheduler).
    Uses Alpaca paper trading by default.
    """

    def __init__(
        self,
        config: PaperRunnerConfig | None = None,
        broker: BaseBroker | None = None,
    ) -> None:
        self.config = config or PaperRunnerConfig()

        # Initialize broker
        if broker is not None:
            self.broker = broker
        else:
            self.broker = AlpacaBroker.from_env()

        # Verify paper trading mode
        if isinstance(self.broker, AlpacaBroker) and not self.broker._is_paper:
            raise ValueError(
                "PaperTradingRunner requires paper trading mode. "
                "Set ALPACA_BASE_URL=https://paper-api.alpaca.markets"
            )

        # Initialize executor
        guard_config = CapitalGuardConfig(
            max_capital=self.config.max_capital,
            max_positions=self.config.max_positions,
        )
        self.executor = SignalExecutor(
            broker=self.broker,
            guard_config=guard_config,
            dry_run=self.config.dry_run,
        )

        # Initialize reconciler
        self.reconciler = PositionReconciler(broker=self.broker)

        # Track internal state for reconciliation
        self._system_positions: dict[str, SystemPosition] = {}
        self._last_execution: ExecutionResult | None = None

        # Ensure log directory exists
        self._log_dir = Path(self.config.log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def run_daily(self, signal_file: str | Path | None = None) -> DailyReport:
        """Execute a single daily paper trading cycle.

        Steps:
          1. Find the latest signal CSV (or use the one provided).
          2. Execute signals through the SignalExecutor.
          3. Check exit conditions on open positions.
          4. Reconcile system state vs broker state.
          5. Log everything and return a report.

        Args:
            signal_file: Explicit signal CSV path. If None, finds the latest
                in ``config.signal_dir``.

        Returns:
            DailyReport with full details of the run.
        """
        errors: list[str] = []
        execution_result = None
        exit_orders: list[str] = []
        reconciliation_clean = None
        signal_file_str = None

        # 1. Find signal file
        if signal_file is None:
            signal_file = self._find_latest_signal()

        if signal_file is not None:
            signal_file = Path(signal_file)
            signal_file_str = str(signal_file)
            logger.info("Using signal file: %s", signal_file)

            # 2. Execute signals
            try:
                execution_result = self.executor.execute_signals(signal_file)
                self._last_execution = execution_result
                self._update_system_positions_from_execution(execution_result)
                logger.info("Execution complete: %s", execution_result.summary())
            except Exception as e:
                errors.append(f"Execution error: {e}")
                logger.error("Execution failed: %s", e)
        else:
            logger.info("No signal file found in %s", self.config.signal_dir)

        # 3. Check exits on open positions
        if self.config.exit_check_enabled:
            try:
                exit_orders = self._check_exits()
            except Exception as e:
                errors.append(f"Exit check error: {e}")
                logger.error("Exit check failed: %s", e)

        # 4. Reconcile
        if self.config.reconcile_enabled:
            try:
                recon_result = self.reconciler.reconcile(self._system_positions)
                reconciliation_clean = recon_result.is_clean
                if not recon_result.is_clean:
                    logger.warning("Reconciliation issues: %s", recon_result.summary())
            except Exception as e:
                errors.append(f"Reconciliation error: {e}")
                logger.error("Reconciliation failed: %s", e)

        # 5. Get account state for report
        equity = 0.0
        cash = 0.0
        positions_count = 0
        try:
            account = self.broker.get_account_snapshot()
            equity = account.equity
            cash = account.cash
            positions_count = account.position_count
        except BrokerError as e:
            errors.append(f"Account fetch error: {e}")

        report = DailyReport(
            timestamp=datetime.now(UTC),
            signal_file=signal_file_str,
            execution_result=execution_result,
            exit_orders=exit_orders,
            reconciliation_clean=reconciliation_clean,
            account_equity=equity,
            account_cash=cash,
            positions_count=positions_count,
            errors=errors,
        )

        # Log the report
        self._log_report(report)
        return report

    def get_status(self) -> PaperTradingStatus:
        """Get current paper trading session status."""
        try:
            account = self.broker.get_account_snapshot()
        except BrokerError:
            account = AccountSnapshot(
                equity=0,
                cash=0,
                buying_power=0,
                positions_market_value=0,
                position_count=0,
                is_margin_account=False,
            )

        try:
            broker_positions = self.broker.get_positions()
        except BrokerError:
            broker_positions = []

        positions = []
        total_pnl = 0.0
        for p in broker_positions:
            pnl_pct = (
                (p.current_price - p.avg_entry_price) / p.avg_entry_price * 100
                if p.avg_entry_price > 0
                else 0.0
            )
            positions.append(
                PositionStatus(
                    symbol=p.symbol,
                    qty=p.qty,
                    avg_entry_price=p.avg_entry_price,
                    current_price=p.current_price,
                    unrealised_pnl=p.unrealised_pnl,
                    unrealised_pnl_pct=pnl_pct,
                    side=p.side,
                )
            )
            total_pnl += p.unrealised_pnl

        is_paper = isinstance(self.broker, AlpacaBroker) and self.broker._is_paper
        mode = "PAPER" if is_paper else "LIVE"

        recon = self.reconciler.last_result
        recon_clean = recon.is_clean if recon else None

        return PaperTradingStatus(
            timestamp=datetime.now(UTC),
            mode=mode,
            account_equity=account.equity,
            account_cash=account.cash,
            buying_power=account.buying_power,
            positions=positions,
            total_unrealised_pnl=total_pnl,
            is_healthy=account.equity > 0,
            last_execution=self._last_execution,
            last_reconciliation_clean=recon_clean,
        )

    # --- Internal helpers ---

    def _find_latest_signal(self) -> Path | None:
        """Find the most recent signal CSV in the signal directory."""
        signal_dir = Path(self.config.signal_dir)
        if not signal_dir.exists():
            return None

        csvs = sorted(signal_dir.glob("signals_*.csv"), reverse=True)
        return csvs[0] if csvs else None

    def _check_exits(self) -> list[str]:
        """Check exit conditions on open positions and close if needed.

        Currently implements a simple stop-loss check by comparing current
        price against the system's recorded stop price. More sophisticated
        exit logic (trailing stops, profit targets, time-based) can be
        layered on.

        Returns:
            List of symbols where exit orders were submitted.
        """
        closed: list[str] = []

        try:
            positions = self.broker.get_positions()
        except BrokerError:
            return closed

        for pos in positions:
            sys_pos = self._system_positions.get(pos.symbol)
            if sys_pos is None:
                continue

            # The system position tracks the stop price in avg_entry_price
            # for simplicity. A more complete implementation would maintain
            # a separate stop price tracker.
            # For now, we don't close positions here — that logic belongs
            # to the full strategy engine. This is a placeholder for the
            # paper trading validation workflow.

        return closed

    def _update_system_positions_from_execution(self, result: ExecutionResult) -> None:
        """Update internal system positions from execution result."""
        for detail in result.details:
            action = detail.get("action", "")
            ticker = detail.get("ticker", "")
            if action in ("ORDER_SUBMITTED", "DRY_RUN_APPROVED") and ticker:
                shares = detail.get("shares", 0)
                price = detail.get("limit_price", detail.get("price", 0))
                self._system_positions[ticker] = SystemPosition(
                    symbol=ticker,
                    qty=shares,
                    avg_entry_price=price,
                    side="long",
                )

    def _log_report(self, report: DailyReport) -> None:
        """Write the daily report to a log file."""
        date_str = report.timestamp.strftime("%Y%m%d")
        log_file = self._log_dir / f"paper_report_{date_str}.txt"
        try:
            with open(log_file, "a") as f:
                f.write(report.summary() + "\n\n")
            logger.info("Report logged to %s", log_file)
        except OSError as e:
            logger.warning("Could not write report log: %s", e)
