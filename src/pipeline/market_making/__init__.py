"""Market-making algorithm framework.

Production-grade market-making components for profiting from the bid-ask
spread while managing inventory risk and adverse selection.

Modules:
    spread       - Dynamic spread calculation with volatility and imbalance scaling.
    inventory    - Inventory tracking, risk penalties, and automatic de-risking.
    quoting      - Event-driven quote generation and throttling.
    adverse      - Adverse selection / toxic flow detection.
    hedging      - Risk-driven hedging with cost-aware sizing.
    microstructure - Order book analytics and fill diagnostics.
    engine       - Top-level market-making engine orchestrating all components.
"""

from pipeline.market_making.adverse import AdverseConfig, AdverseSelectionDetector
from pipeline.market_making.engine import MarketMakingConfig, MarketMakingEngine
from pipeline.market_making.hedging import HedgeConfig, HedgeManager
from pipeline.market_making.inventory import InventoryConfig, InventoryManager
from pipeline.market_making.microstructure import MicrostructureAnalyzer
from pipeline.market_making.quoting import QuoteConfig, QuoteEngine
from pipeline.market_making.spread import SpreadCalculator, SpreadConfig

__all__ = [
    "SpreadCalculator",
    "SpreadConfig",
    "InventoryManager",
    "InventoryConfig",
    "QuoteEngine",
    "QuoteConfig",
    "AdverseSelectionDetector",
    "AdverseConfig",
    "HedgeManager",
    "HedgeConfig",
    "MicrostructureAnalyzer",
    "MarketMakingEngine",
    "MarketMakingConfig",
]
