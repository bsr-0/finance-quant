"""Strategy definition schema — top-level configuration for a complete strategy.

A ``StrategyDefinition`` is the single source of truth for:
  - Strategy thesis and description
  - Universe selection criteria
  - Signal definitions with formulas
  - Entry/exit rules
  - Position sizing method
  - Risk constraints
  - Benchmark configuration
  - Backtest parameters
  - Edge decay monitoring settings

The memo generator reads this definition to produce a consistent,
Goldman Sachs-style strategy memo with no free-form drift from the
implemented configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pipeline.strategy.backtest_harness import BacktestConfig
from pipeline.strategy.benchmark import US_EQUITY_BENCHMARKS, BenchmarkSuite
from pipeline.strategy.entry_rules import EntryRuleSet, institutional_entry_rules
from pipeline.strategy.exits import ExitEngine
from pipeline.strategy.position_sizing import (
    InstitutionalSizingConfig,
    SizingMethod,
)
from pipeline.strategy.risk_constraints import (
    RiskConstraintSet,
    institutional_constraints,
)
from pipeline.strategy.signal_library import SignalDefinition
from pipeline.strategy.universe import US_LARGE_CAP_EQUITY, UniverseFilter

# ---------------------------------------------------------------------------
# Strategy thesis
# ---------------------------------------------------------------------------

@dataclass
class StrategyThesis:
    """Describes the market inefficiency and alpha source."""

    name: str = ""
    classification: str = ""  # e.g., "Systematic Equity | Long-Only | Momentum"
    inefficiency: str = ""  # What market inefficiency is exploited
    drivers: list[str] = field(default_factory=list)  # Behavioral, structural, etc.
    holding_period: str = ""  # e.g., "1-3 months (medium frequency)"
    turnover_regime: str = ""  # e.g., "Monthly rebalance, ~200% annual turnover"
    alpha_source: str = ""  # e.g., "Cross-sectional momentum premium"
    risk_premium_exposure: str = ""  # e.g., "Market beta, momentum factor"
    when_edge_disappears: list[str] = field(default_factory=list)
    target_aum: str = ""  # e.g., "$10B+"
    status: str = "RESEARCH"


# ---------------------------------------------------------------------------
# Full strategy definition
# ---------------------------------------------------------------------------

@dataclass
class StrategyDefinition:
    """Complete strategy definition — the single source of truth.

    All other components (backtest, memo, risk checks) derive from this.
    """

    thesis: StrategyThesis
    universe_filter: UniverseFilter
    signal_definitions: list[SignalDefinition]
    entry_rules: EntryRuleSet
    exit_engine: ExitEngine
    sizing_config: InstitutionalSizingConfig
    risk_constraints: RiskConstraintSet
    benchmark_suite: BenchmarkSuite
    backtest_config: BacktestConfig
    edge_decay_config: dict = field(default_factory=dict)

    @property
    def strategy_name(self) -> str:
        return self.thesis.name

    @property
    def primary_signal(self) -> SignalDefinition | None:
        return self.signal_definitions[0] if self.signal_definitions else None


# ---------------------------------------------------------------------------
# Example strategy: Cross-sectional momentum on US large-cap equities
# ---------------------------------------------------------------------------

def cross_sectional_momentum_strategy() -> StrategyDefinition:
    """Pre-built: multi-timeframe cross-sectional momentum with crash protection.

    Improvements over the original 12-1 month single-window approach:

    1. **Faster lookback** — primary window 6-1 month (126-21 days) plus a
       3-1 month fast component, adapting quicker to regime changes.
    2. **Crash protection** — monitors abnormal return dispersion and dampens
       the composite signal when momentum-crash risk is elevated.
    3. **Higher signal threshold** (0.3) — filters out low-conviction entries,
       reducing trade count and cost drag.
    4. **Tighter exits** — 42-day max hold, 1.5x ATR stop, 2.0x ATR trailing,
       3.0x ATR profit target for faster realisation.
    5. **Faster trend confirmation** — 20/50 MA crossover instead of 50/200.

    This is a complete, production-ready strategy definition that can
    be backtested and used to generate a strategy memo.
    """
    from pipeline.strategy.signal_library import momentum_signal

    thesis = StrategyThesis(
        name="QSG-SYSTEMATIC-MOM-001",
        classification="Systematic Equity | Long-Only | Cross-Sectional Momentum",
        inefficiency=(
            "Cross-sectional momentum exploits the tendency for recent winners "
            "to continue outperforming recent losers over 3-6 month horizons, "
            "driven by investor underreaction to fundamental news and herding "
            "behavior.  The multi-timeframe signal blends 6-1 and 3-1 month "
            "windows to capture both intermediate and short-term continuation, "
            "with a dispersion-based crash-protection overlay that de-levers "
            "when regime-reversal risk is elevated."
        ),
        drivers=[
            "Behavioral: Investor underreaction to earnings surprises and "
            "fundamental news (Hong & Stein 1999)",
            "Behavioral: Herding and trend-following by institutional investors",
            "Structural: Slow-moving capital and rebalancing constraints "
            "of passive funds",
            "Liquidity: Gradual diffusion of information across heterogeneous "
            "investor populations",
            "Crash protection: Dispersion-based de-leveraging during "
            "momentum-reversal regimes (Daniel & Moskowitz 2016)",
        ],
        holding_period="1-2 months (medium frequency, monthly rebalance)",
        turnover_regime="Monthly rebalance, ~150% annualized turnover",
        alpha_source=(
            "Cross-sectional momentum premium, distinct from market beta. "
            "Expected alpha of 3-5% annualized above the market after costs, "
            "with crash protection reducing tail risk during reversals."
        ),
        risk_premium_exposure=(
            "Primary: Momentum factor (UMD). Secondary: Market beta (reduced "
            "during drawdown via regime filter and crash overlay). Minimal "
            "exposure to value, size, or quality factors by construction."
        ),
        when_edge_disappears=[
            "Sustained momentum crashes (rapid regime reversals as in 2009 Q1)",
            "Prolonged mean-reverting markets with no persistent trends",
            "High-correlation panic regimes (VIX > 35, correlations approach 1)",
            "Significant crowding in momentum strategies reducing the premium",
        ],
        target_aum="$100M - $10B (capacity constrained by liquidity filters)",
        status="RESEARCH — Paper Validation",
    )

    universe = UniverseFilter(
        asset_classes=US_LARGE_CAP_EQUITY.asset_classes,
        regions=US_LARGE_CAP_EQUITY.regions,
        exchanges=US_LARGE_CAP_EQUITY.exchanges,
        min_adv_dollars=5e8,
        min_price=10.0,
        min_market_cap=10e9,
        max_spread_bps=5.0,
    )

    signal = momentum_signal(
        lookback=126,           # 6-1 month primary (was 252)
        skip=21,
        fast_lookback=63,       # 3-1 month fast component
        vol_window=60,
        crash_protection=True,  # Dispersion-based de-leveraging
    )

    entry_rules = institutional_entry_rules(
        signal_threshold=0.3,      # Higher conviction only (was 0.0)
        blocked_regimes=["BEAR"],
        max_sector_exposure=0.30,
    )

    exit_engine = ExitEngine(
        max_holding_days=42,        # ~2 months (was 63)
        stop_atr_multiple=1.5,      # Tighter stop (was 2.0)
        trailing_atr_multiple=2.0,  # Tighter trail (was 2.5)
        trailing_activation_atr=1.0,  # Activate sooner (was 1.5)
        target_atr_multiple=3.0,    # Take profits sooner (was 4.0)
        rsi_overbought=75.0,        # Slightly more aggressive (was 80)
    )

    sizing = InstitutionalSizingConfig(
        method=SizingMethod.VOLATILITY_SCALED,
        total_capital=1e8,
        target_annual_vol=0.10,
        target_position_risk=0.005,
        vol_lookback_days=60,
        max_position_weight=0.05,
        min_position_weight=0.005,
        max_gross_exposure=1.0,
        max_net_exposure=1.0,
        max_adv_participation=0.05,
        min_trade_notional=50_000,
    )

    constraints = institutional_constraints(
        max_position_weight=0.05,
        max_sector_exposure=0.30,
        max_country_exposure=0.40,
        max_gross_exposure=1.0,
        max_net_exposure=1.0,
        max_drawdown=0.15,
        max_adv_participation=0.05,
        max_turnover=0.20,
        sectors=[
            "Technology", "Healthcare", "Financials", "Consumer Discretionary",
            "Industrials", "Energy", "Utilities", "Materials",
            "Consumer Staples", "Real Estate", "Communication Services",
        ],
    )

    benchmarks = US_EQUITY_BENCHMARKS

    backtest = BacktestConfig(
        initial_capital=1e8,
        start_date="2019-01-01",
        end_date="2024-12-31",
        spread_bps=3.0,
        commission_per_share=0.005,
        slippage_bps=2.0,
        signal_lag_days=1,
    )

    return StrategyDefinition(
        thesis=thesis,
        universe_filter=universe,
        signal_definitions=[signal],
        entry_rules=entry_rules,
        exit_engine=exit_engine,
        sizing_config=sizing,
        risk_constraints=constraints,
        benchmark_suite=benchmarks,
        backtest_config=backtest,
        edge_decay_config={
            "window": 60,
            "min_trades": 10,
            "win_rate_floor": 0.48,
            "profit_factor_floor": 1.1,
            "sharpe_floor": 0.3,
        },
    )
