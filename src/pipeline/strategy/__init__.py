"""Quantitative Strategy Engine.

Provides both the original QSG-MICRO-SWING-001 pullback reversion strategy
and a new institutional-grade strategy architecture engine supporting
configurable universe selection, signal generation, entry/exit rules,
position sizing, risk constraints, backtesting, benchmarking, edge decay
monitoring, and auto-generated Goldman Sachs-style strategy memos.
"""

# Original swing strategy components
from pipeline.strategy.backtest_harness import (
    BacktestConfig,
    BacktestHarness,
    BacktestMetrics,
    HarnessBacktestResult,
)
from pipeline.strategy.benchmark import (
    US_EQUITY_BENCHMARKS,
    BenchmarkAnalysis,
    BenchmarkConfig,
    BenchmarkSuite,
    compute_all_benchmarks,
    compute_benchmark_analysis,
)
from pipeline.strategy.edge_decay import AlertLevel, EdgeDecayMonitor
from pipeline.strategy.engine import StrategyConfig, SwingStrategyEngine
from pipeline.strategy.entry_rules import (
    EntryCondition,
    EntryContext,
    EntryDecision,
    EntryRuleSet,
    MaxPositionsCondition,
    NoDuplicatePositionCondition,
    RegimeCondition,
    RiskBudgetCondition,
    SectorExposureCondition,
    SignalThresholdCondition,
    institutional_entry_rules,
)
from pipeline.strategy.exits import ExitEngine, ExitSignal
from pipeline.strategy.memo_generator import (
    MemoGenerator,
    generate_memo,
)
from pipeline.strategy.position_sizing import (
    FixedFractionSizer,
    InstitutionalSizingConfig,
    PortfolioTargets,
    PositionSizingModel,
    PositionTarget,
    SignalWeightedSizer,
    SizingMethod,
    VolatilityScaledSizer,
    create_sizer,
)
from pipeline.strategy.risk import DrawdownLevel, SwingRiskManager
from pipeline.strategy.risk_constraints import (
    ConstraintCheckResult,
    ConstraintSeverity,
    ConstraintType,
    RiskConstraint,
    RiskConstraintSet,
    institutional_constraints,
)
from pipeline.strategy.signal_library import (
    MomentumReturn,
    MovingAverageCrossover,
    NormalizationMethod,
    RawIndicator,
    RSIMeanReversion,
    SignalConfig,
    SignalDefinition,
    SignalFamily,
    SignalPipeline,
    VolatilitySignal,
    mean_reversion_signal,
    momentum_signal,
)
from pipeline.strategy.signals import SignalEngine, SignalScore
from pipeline.strategy.sizing import PositionSizer, SizingConfig
from pipeline.strategy.strategy_definition import (
    StrategyDefinition,
    StrategyThesis,
    cross_sectional_momentum_strategy,
)

# New institutional strategy architecture
from pipeline.strategy.universe import (
    AssetClass,
    Exchange,
    InstrumentMetadata,
    Region,
    Universe,
    UniverseBuilder,
    UniverseFilter,
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
