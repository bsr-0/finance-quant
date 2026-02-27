"""QSG-MICRO-SWING-001: Trend-Aligned Pullback Reversion strategy.

A rules-based swing trading strategy for micro-capital accounts ($100-$1,000)
that buys pullbacks in established uptrends and sells into strength.
"""

from pipeline.strategy.signals import SignalEngine, SignalScore
from pipeline.strategy.sizing import PositionSizer, SizingConfig
from pipeline.strategy.exits import ExitEngine, ExitSignal
from pipeline.strategy.risk import SwingRiskManager, DrawdownLevel
from pipeline.strategy.engine import SwingStrategyEngine, StrategyConfig
from pipeline.strategy.edge_decay import EdgeDecayMonitor, AlertLevel

__all__ = [
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
]
