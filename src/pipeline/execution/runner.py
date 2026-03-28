"""Daily trading runner: end-to-end signal→execution pipeline.

Orchestrates the complete daily trading workflow:

    1. Generate signals (or read existing CSV)
    2. Execute signals through the broker with QAQC
    3. Monitor positions for exit conditions
    4. Reconcile system vs broker state
    5. Log results

Supports both paper trading (Alpaca paper API) and live trading (Alpaca
live API), with explicit mode switching and safety confirmations.

Usage::

    runner = TradingRunner.from_env(max_capital=300.0)

    # Paper trading (default):
    result = runner.run_daily(signal_csv="data/signals/signals_20250306.csv")

    # Or generate + execute in one step:
    result = runner.generate_and_execute(prices_dir="data/prices/")
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pipeline.execution.alpaca_broker import AlpacaBroker
from pipeline.execution.broker import BrokerError
from pipeline.execution.capital_guard import CapitalGuardConfig
from pipeline.execution.position_monitor import PositionMonitor, TrackedPosition
from pipeline.execution.reconciler import PositionReconciler
from pipeline.execution.signal_executor import ExecutionResult, SignalExecutor
from pipeline.infrastructure.notifier import AlertSeverity, notify
from pipeline.strategy.risk import SwingRiskManager
from pipeline.strategy.sizing import SizingConfig

logger = logging.getLogger(__name__)


@dataclass
class RunnerConfig:
    """Configuration for the daily trading runner."""

    # Capital limits (QAQC)
    max_capital: float = 300.0
    max_positions: int = 2
    require_cash_account: bool = True

    # Sizing
    risk_fraction_small: float = 0.015
    risk_fraction_large: float = 0.010

    # Paper vs live
    paper_mode: bool = True

    # Dry run (no orders at all)
    dry_run: bool = False

    # Signal threshold
    signal_threshold: int = 60


class TradingRunner:
    """End-to-end daily trading runner.

    Wires together all execution components and provides a single
    entry point for daily trading operations.
    """

    def __init__(
        self,
        broker: AlpacaBroker,
        config: RunnerConfig,
    ) -> None:
        self.broker = broker
        self.config = config

        guard_config = CapitalGuardConfig(
            max_capital=config.max_capital,
            max_positions=config.max_positions,
            require_cash_account=config.require_cash_account,
        )

        sizing_config = SizingConfig(
            risk_fraction_small=config.risk_fraction_small,
            risk_fraction_large=config.risk_fraction_large,
        )

        self.risk_mgr = SwingRiskManager()

        self.executor = SignalExecutor(
            broker=broker,
            guard_config=guard_config,
            sizing_config=sizing_config,
            risk_manager=self.risk_mgr,
            dry_run=config.dry_run,
        )

        self.monitor = PositionMonitor(
            broker=broker,
            guard_config=guard_config,
            risk_manager=self.risk_mgr,
        )

        self.reconciler = PositionReconciler(broker=broker)

        mode_label = "DRY RUN" if config.dry_run else ("PAPER" if config.paper_mode else "LIVE")
        logger.info(
            "Trading runner created: mode=%s, max_capital=$%.2f, " "max_positions=%d",
            mode_label,
            config.max_capital,
            config.max_positions,
        )

    @classmethod
    def from_env(
        cls,
        max_capital: float = 300.0,
        max_positions: int = 2,
        dry_run: bool = False,
    ) -> TradingRunner:
        """Create a TradingRunner from environment variables.

        Reads ALPACA_API_KEY, ALPACA_SECRET_KEY, and ALPACA_BASE_URL
        from the environment.

        Args:
            max_capital: Maximum capital to deploy.
            max_positions: Maximum simultaneous positions.
            dry_run: If True, validate everything but don't submit orders.

        Returns:
            Configured TradingRunner.
        """
        broker = AlpacaBroker.from_env()
        base_url = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        is_paper = "paper" in base_url.lower()

        config = RunnerConfig(
            max_capital=max_capital,
            max_positions=max_positions,
            paper_mode=is_paper,
            dry_run=dry_run,
        )

        return cls(broker=broker, config=config)

    def run_daily(self, signal_csv: str | Path) -> ExecutionResult:
        """Execute signals from a CSV file.

        This is the main entry point for daily trading. Call it with the
        output of ``generate-signals`` CLI command.

        Args:
            signal_csv: Path to the signal CSV file.

        Returns:
            ExecutionResult with what happened.
        """
        signal_csv = Path(signal_csv)
        mode = "DRY RUN" if self.config.dry_run else ("PAPER" if self.config.paper_mode else "LIVE")

        logger.info("=" * 60)
        logger.info("DAILY TRADING RUN — %s mode", mode)
        logger.info("Signal file: %s", signal_csv)
        logger.info("Max capital: $%.2f", self.config.max_capital)
        logger.info("=" * 60)

        if not self.config.paper_mode and not self.config.dry_run:
            logger.warning("*** LIVE TRADING MODE — Real money at risk ***")
            notify(
                AlertSeverity.WARNING,
                "Daily Run Started — LIVE MODE",
                f"Live trading run started with max_capital=${self.config.max_capital:.2f}.",
                {"signal_csv": str(signal_csv), "max_capital": self.config.max_capital},
            )

        # Step 1: Check exits on existing positions
        logger.info("Step 1: Checking exits on existing positions...")
        monitor_result = self.monitor.check_and_exit()
        logger.info("  %s", monitor_result.summary())

        # Step 2: Execute new signals
        logger.info("Step 2: Executing new signals...")
        exec_result = self.executor.execute_signals(signal_csv)
        logger.info("  %s", exec_result.summary())

        # Step 3: Register new positions for monitoring
        for detail in exec_result.details:
            if detail.get("action") in ("ORDER_SUBMITTED", "DRY_RUN_APPROVED"):
                ticker = detail["ticker"]
                self.monitor.register_position(
                    TrackedPosition(
                        symbol=ticker,
                        entry_date=datetime.now(UTC),
                        entry_price=detail.get("limit_price", detail.get("price", 0)),
                        shares=detail.get("shares", 0),
                        stop_price=detail.get("stop_price", 0),
                        atr_at_entry=0.0,  # Set from signal if available
                        signal_score=0,
                    )
                )

        # Step 4: Reconcile
        if not self.config.dry_run:
            logger.info("Step 4: Reconciling positions...")
            from pipeline.execution.reconciler import SystemPosition

            system_positions = {}
            for sym, tracked in self.monitor.tracked_positions.items():
                system_positions[sym] = SystemPosition(
                    symbol=sym,
                    qty=tracked.shares,
                    avg_entry_price=tracked.entry_price,
                    side="long",
                )
            recon = self.reconciler.reconcile(system_positions)
            logger.info("  %s", recon.summary())

        logger.info("=" * 60)
        logger.info("DAILY RUN COMPLETE — %s", exec_result.summary())
        logger.info("=" * 60)

        notify(
            AlertSeverity.INFO,
            f"Daily Run Complete — {mode}",
            exec_result.summary(),
            {
                "filled": exec_result.orders_filled,
                "rejected": exec_result.orders_rejected,
                "guard_blocked": exec_result.guard_rejections,
            },
        )

        return exec_result

    def status(self) -> dict:
        """Get current runner status."""
        try:
            account = self.broker.get_account_snapshot()
            positions = self.broker.get_positions()
        except BrokerError as e:
            return {"error": str(e)}

        return {
            "mode": "paper" if self.config.paper_mode else "live",
            "dry_run": self.config.dry_run,
            "max_capital": self.config.max_capital,
            "account_equity": account.equity,
            "account_cash": account.cash,
            "buying_power": account.buying_power,
            "positions_count": len(positions),
            "positions_value": account.positions_market_value,
            "is_margin": account.is_margin_account,
            "tracked_positions": list(self.monitor.tracked_positions.keys()),
        }
