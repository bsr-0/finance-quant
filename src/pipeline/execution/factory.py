"""Factory functions for creating execution components from settings.

Provides a single entry point to create a fully wired execution stack
(broker, executor, runner, journal) from the pipeline settings, without
callers needing to know the wiring details.

Usage::

    from pipeline.execution.factory import create_trading_runner, create_paper_runner

    # Full daily runner from config.yaml + env vars:
    runner = create_trading_runner()
    result = runner.run_daily("signals/signals_20250306.csv")

    # Paper trading runner:
    paper = create_paper_runner()
    report = paper.run_daily()

    # Just the broker:
    broker = create_broker()
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from pipeline.settings import ExecutionSettings, get_settings

logger = logging.getLogger(__name__)


def create_broker(settings: ExecutionSettings | None = None):
    """Create a broker instance from settings.

    Currently only Alpaca is supported.  The broker reads API keys from
    environment variables (``ALPACA_API_KEY``, ``ALPACA_SECRET_KEY``).

    Args:
        settings: Execution settings.  Uses global config if None.

    Returns:
        An AlpacaBroker instance.
    """
    from pipeline.execution.alpaca_broker import AlpacaBroker

    if settings is None:
        settings = get_settings().execution

    # Ensure ALPACA_BASE_URL reflects the mode setting
    if settings.is_paper:
        os.environ.setdefault("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    else:
        os.environ.setdefault("ALPACA_BASE_URL", settings.base_url)

    return AlpacaBroker.from_env()


def create_signal_executor(settings: ExecutionSettings | None = None, broker=None):
    """Create a SignalExecutor from settings.

    Wires together the broker, capital guard, position sizer, risk manager,
    reconciler, and trade journal.

    Args:
        settings: Execution settings.  Uses global config if None.
        broker: Broker instance.  Creates one from settings if None.

    Returns:
        A fully configured SignalExecutor.
    """
    from pipeline.execution.capital_guard import CapitalGuardConfig
    from pipeline.execution.signal_executor import SignalExecutor
    from pipeline.execution.trade_journal import TradeJournal
    from pipeline.strategy.sizing import SizingConfig

    if settings is None:
        settings = get_settings().execution
    if broker is None:
        broker = create_broker(settings)

    guard_config = CapitalGuardConfig(
        max_capital=settings.max_capital,
        max_positions=settings.max_positions,
        require_cash_account=settings.require_cash_account,
        max_daily_orders=settings.max_daily_orders,
    )

    sizing_config = SizingConfig(
        risk_fraction_small=settings.risk_per_trade_pct,
    )

    return SignalExecutor(
        broker=broker,
        guard_config=guard_config,
        sizing_config=sizing_config,
        dry_run=False,
        fill_poll_interval=settings.fill_poll_interval_seconds,
        fill_poll_timeout=settings.fill_poll_timeout_seconds,
    )


def create_trading_runner(
    settings: ExecutionSettings | None = None,
    dry_run: bool = False,
):
    """Create a TradingRunner from settings.

    Args:
        settings: Execution settings.  Uses global config if None.
        dry_run: Override dry_run regardless of settings.

    Returns:
        A configured TradingRunner.
    """
    from pipeline.execution.alpaca_broker import AlpacaBroker
    from pipeline.execution.runner import RunnerConfig, TradingRunner

    if settings is None:
        settings = get_settings().execution

    broker = create_broker(settings)

    config = RunnerConfig(
        max_capital=settings.max_capital,
        max_positions=settings.max_positions,
        require_cash_account=settings.require_cash_account,
        paper_mode=settings.is_paper,
        dry_run=dry_run,
    )

    return TradingRunner(broker=broker, config=config)


def create_paper_runner(settings: ExecutionSettings | None = None):
    """Create a PaperTradingRunner from settings.

    The broker is forced into paper mode regardless of settings.

    Args:
        settings: Execution settings.  Uses global config if None.

    Returns:
        A configured PaperTradingRunner.
    """
    from pipeline.execution.paper_runner import PaperRunnerConfig, PaperTradingRunner

    if settings is None:
        settings = get_settings().execution

    # Force paper mode for this factory
    os.environ.setdefault("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    broker = create_broker(settings)

    config = PaperRunnerConfig(
        signal_dir=settings.signal_dir,
        max_capital=settings.max_capital,
        max_positions=settings.max_positions,
        order_type=settings.order_type,
        min_score=settings.min_score,
        min_confidence=settings.min_confidence,
        log_dir=settings.log_dir,
    )

    return PaperTradingRunner(config=config, broker=broker)


def create_trade_journal(settings: ExecutionSettings | None = None):
    """Create a TradeJournal from settings.

    Args:
        settings: Execution settings.  Uses global config if None.

    Returns:
        A configured TradeJournal.
    """
    from pipeline.execution.trade_journal import TradeJournal

    if settings is None:
        settings = get_settings().execution

    return TradeJournal(journal_dir=settings.journal_dir)
