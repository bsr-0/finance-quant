"""Execution layer: broker integration, capital guards, and reconciliation.

Public API::

    from pipeline.execution import (
        # Broker interface & types
        BaseBroker, BrokerError, Order, OrderSide, OrderStatus, OrderType, Position,
        # Alpaca implementation
        AlpacaBroker,
        # Capital guard
        CapitalGuard, CapitalGuardConfig, AccountSnapshot,
        # Signal executor
        SignalExecutor, ExecutionResult, ParsedSignal,
        # Reconciler
        PositionReconciler, ReconciliationResult,
        # Position monitor
        PositionMonitor, TrackedPosition,
        # Real-time feed
        RealtimePriceFeed, PriceQuote,
        # Runners
        TradingRunner, PaperTradingRunner,
        # Trade journal
        TradeJournal,
        # Factory functions (create from settings)
        create_broker, create_signal_executor, create_trading_runner,
        create_paper_runner, create_trade_journal,
    )
"""

# Broker interface & types
from pipeline.execution.broker import (
    BaseBroker,
    BrokerError,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)

# Alpaca implementation
from pipeline.execution.alpaca_broker import AlpacaBroker

# Capital guard
from pipeline.execution.capital_guard import (
    AccountSnapshot,
    CapitalGuard,
    CapitalGuardConfig,
)

# Signal executor
from pipeline.execution.signal_executor import (
    ExecutionResult,
    ParsedSignal,
    SignalExecutor,
)

# Reconciler
from pipeline.execution.reconciler import (
    PositionReconciler,
    ReconciliationResult,
)

# Position monitor
from pipeline.execution.position_monitor import (
    PositionMonitor,
    TrackedPosition,
)

# Real-time feed
from pipeline.execution.realtime_feed import (
    PriceQuote,
    RealtimePriceFeed,
)

# Runners
from pipeline.execution.runner import TradingRunner
from pipeline.execution.paper_runner import PaperTradingRunner

# Trade journal
from pipeline.execution.trade_journal import TradeJournal

# Factory functions
from pipeline.execution.factory import (
    create_broker,
    create_paper_runner,
    create_signal_executor,
    create_trade_journal,
    create_trading_runner,
)

__all__ = [
    # Broker
    "BaseBroker", "BrokerError", "Order", "OrderSide", "OrderStatus",
    "OrderType", "Position", "AlpacaBroker",
    # Guard
    "AccountSnapshot", "CapitalGuard", "CapitalGuardConfig",
    # Executor
    "ExecutionResult", "ParsedSignal", "SignalExecutor",
    # Reconciler
    "PositionReconciler", "ReconciliationResult",
    # Monitor
    "PositionMonitor", "TrackedPosition",
    # Feed
    "PriceQuote", "RealtimePriceFeed",
    # Runners
    "TradingRunner", "PaperTradingRunner",
    # Journal
    "TradeJournal",
    # Factory
    "create_broker", "create_signal_executor", "create_trading_runner",
    "create_paper_runner", "create_trade_journal",
]
