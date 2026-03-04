"""Quantitative Strategy Engine.

Provides both the original QSG-MICRO-SWING-001 pullback reversion strategy
and a new institutional-grade strategy architecture engine supporting
configurable universe selection, signal generation, entry/exit rules,
position sizing, risk constraints, backtesting, benchmarking, edge decay
monitoring, and auto-generated Goldman Sachs-style strategy memos.
"""

# Original swing strategy components
from pipeline.strategy.signals import SignalEngine, SignalScore
from pipeline.strategy.sizing import PositionSizer, SizingConfig
from pipeline.strategy.exits import ExitEngine, ExitSignal
from pipeline.strategy.risk import SwingRiskManager, DrawdownLevel
from pipeline.strategy.engine import SwingStrategyEngine, StrategyConfig
from pipeline.strategy.edge_decay import EdgeDecayMonitor, AlertLevel

# New institutional strategy architecture
from pipeline.strategy.universe import (
    Universe,
    UniverseBuilder,
    UniverseFilter,
    InstrumentMetadata,
    AssetClass,
    Region,
    Exchange,
)
from pipeline.strategy.signal_library import (
    SignalDefinition,
    SignalPipeline,
    SignalFamily,
    SignalConfig,
    NormalizationMethod,
    RawIndicator,
    MomentumReturn,
    MovingAverageCrossover,
    RSIMeanReversion,
    VolatilitySignal,
    momentum_signal,
    mean_reversion_signal,
)
from pipeline.strategy.entry_rules import (
    EntryRuleSet,
    EntryCondition,
    EntryContext,
    EntryDecision,
    SignalThresholdCondition,
    RegimeCondition,
    MaxPositionsCondition,
    NoDuplicatePositionCondition,
    RiskBudgetCondition,
    SectorExposureCondition,
    institutional_entry_rules,
)
from pipeline.strategy.position_sizing import (
    InstitutionalSizingConfig,
    PositionSizingModel,
    VolatilityScaledSizer,
    FixedFractionSizer,
    SignalWeightedSizer,
    PortfolioTargets,
    PositionTarget,
    SizingMethod,
    create_sizer,
)
from pipeline.strategy.risk_constraints import (
    RiskConstraintSet,
    RiskConstraint,
    ConstraintType,
    ConstraintSeverity,
    ConstraintCheckResult,
    institutional_constraints,
)
from pipeline.strategy.benchmark import (
    BenchmarkConfig,
    BenchmarkSuite,
    BenchmarkAnalysis,
    compute_benchmark_analysis,
    compute_all_benchmarks,
    US_EQUITY_BENCHMARKS,
)
from pipeline.strategy.backtest_harness import (
    BacktestHarness,
    BacktestConfig,
    BacktestMetrics,
    HarnessBacktestResult,
)
from pipeline.strategy.strategy_definition import (
    StrategyDefinition,
    StrategyThesis,
    cross_sectional_momentum_strategy,
)
from pipeline.strategy.memo_generator import (
    MemoGenerator,
    generate_memo,
)

__all__ = [
    # Original swing strategy
    "SignalEngine",
    "SignalScore",
    "PositionSizer",
    "SizingConfig",
    "ExitEngine",
    "ExitSignal",
    "SwingRiskManager",
    "DrawdownLevel",
    "SwingStrategyEngine",
    "StrategyConfig",
    "EdgeDecayMonitor",
    "AlertLevel",
    # Universe
    "Universe",
    "UniverseBuilder",
    "UniverseFilter",
    "InstrumentMetadata",
    "AssetClass",
    "Region",
    "Exchange",
    # Signals
    "SignalDefinition",
    "SignalPipeline",
    "SignalFamily",
    "SignalConfig",
    "NormalizationMethod",
    "RawIndicator",
    "MomentumReturn",
    "MovingAverageCrossover",
    "RSIMeanReversion",
    "VolatilitySignal",
    "momentum_signal",
    "mean_reversion_signal",
    # Entry rules
    "EntryRuleSet",
    "EntryCondition",
    "EntryContext",
    "EntryDecision",
    "SignalThresholdCondition",
    "RegimeCondition",
    "MaxPositionsCondition",
    "NoDuplicatePositionCondition",
    "RiskBudgetCondition",
    "SectorExposureCondition",
    "institutional_entry_rules",
    # Position sizing
    "InstitutionalSizingConfig",
    "PositionSizingModel",
    "VolatilityScaledSizer",
    "FixedFractionSizer",
    "SignalWeightedSizer",
    "PortfolioTargets",
    "PositionTarget",
    "SizingMethod",
    "create_sizer",
    # Risk constraints
    "RiskConstraintSet",
    "RiskConstraint",
    "ConstraintType",
    "ConstraintSeverity",
    "ConstraintCheckResult",
    "institutional_constraints",
    # Benchmark
    "BenchmarkConfig",
    "BenchmarkSuite",
    "BenchmarkAnalysis",
    "compute_benchmark_analysis",
    "compute_all_benchmarks",
    "US_EQUITY_BENCHMARKS",
    # Backtest harness
    "BacktestHarness",
    "BacktestConfig",
    "BacktestMetrics",
    "HarnessBacktestResult",
    # Strategy definition
    "StrategyDefinition",
    "StrategyThesis",
    "cross_sectional_momentum_strategy",
    # Memo generation
    "MemoGenerator",
    "generate_memo",
]
