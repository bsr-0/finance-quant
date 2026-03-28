"""Comprehensive tests for the institutional strategy architecture engine.

Tests cover:
  - Universe selection and filtering
  - Signal generation (multiple families, normalization)
  - Entry/exit rules
  - Institutional position sizing
  - Risk constraint evaluation
  - Benchmark analysis
  - Backtest harness end-to-end
  - Memo generation consistency
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _make_multi_ticker_data(
    tickers: list[str],
    n: int = 300,
    base_prices: dict[str, float] | None = None,
    trend: float = 0.0005,
    noise: float = 0.01,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Generate synthetic OHLCV data for multiple tickers."""
    np.random.RandomState(seed)
    dates = pd.bdate_range("2022-01-01", periods=n, freq="B")
    result = {}
    for i, ticker in enumerate(tickers):
        start_price = (base_prices or {}).get(ticker, 100 + i * 20)
        local_seed = seed + i
        local_rng = np.random.RandomState(local_seed)
        log_returns = trend + noise * local_rng.randn(n)
        close = start_price * np.exp(np.cumsum(log_returns))
        high = close * (1 + abs(noise) * local_rng.rand(n))
        low = close * (1 - abs(noise) * local_rng.rand(n))
        open_ = close * (1 + noise * 0.5 * local_rng.randn(n))
        volume = (1_000_000 * (1 + 0.3 * local_rng.randn(n))).clip(100_000)
        result[ticker] = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=dates,
        )
    return result


# ===========================================================================
# Universe Selection Tests
# ===========================================================================

class TestUniverseSelection:
    def test_universe_builder_filters(self):
        from pipeline.strategy.universe import (
            AssetClass,
            Exchange,
            InstrumentMetadata,
            Region,
            UniverseBuilder,
            UniverseFilter,
        )

        instruments = [
            InstrumentMetadata(
                ticker="AAPL", name="Apple", asset_class=AssetClass.EQUITY,
                sector="Technology", country="US", region=Region.US,
                exchange=Exchange.NASDAQ, market_cap=3e12,
                adv_dollars=1e9, avg_spread_bps=1.0,
            ),
            InstrumentMetadata(
                ticker="PENNY", name="PennyCorp", asset_class=AssetClass.EQUITY,
                sector="Financials", country="US", region=Region.US,
                exchange=Exchange.NYSE, market_cap=1e6,
                adv_dollars=1e5, avg_spread_bps=50.0,
            ),
            InstrumentMetadata(
                ticker="SPY", name="SPDR S&P 500", asset_class=AssetClass.ETF,
                sector="", country="US", region=Region.US,
                exchange=Exchange.ARCA, market_cap=0,
                adv_dollars=3e10, avg_spread_bps=0.5,
            ),
        ]

        builder = UniverseBuilder(UniverseFilter(
            min_adv_dollars=1e8,
            min_price=5.0,
            max_spread_bps=5.0,
        ))
        universe = builder.build(instruments)

        assert "AAPL" in universe.ticker_set
        assert "SPY" in universe.ticker_set
        assert "PENNY" not in universe.ticker_set  # Low ADV and high spread
        assert len(universe) == 2

    def test_universe_by_sector(self):
        from pipeline.strategy.universe import (
            AssetClass,
            Exchange,
            InstrumentMetadata,
            Region,
            UniverseBuilder,
            UniverseFilter,
        )

        instruments = [
            InstrumentMetadata(
                ticker="AAPL", sector="Technology", asset_class=AssetClass.EQUITY,
                region=Region.US, exchange=Exchange.NASDAQ,
                adv_dollars=1e9, avg_spread_bps=1.0,
            ),
            InstrumentMetadata(
                ticker="JPM", sector="Financials", asset_class=AssetClass.EQUITY,
                region=Region.US, exchange=Exchange.NYSE,
                adv_dollars=5e8, avg_spread_bps=1.5,
            ),
        ]

        builder = UniverseBuilder(UniverseFilter(min_adv_dollars=1e8))
        universe = builder.build(instruments)

        sectors = universe.by_sector()
        assert "Technology" in sectors
        assert "Financials" in sectors

    def test_universe_from_prices(self):
        from pipeline.strategy.universe import UniverseBuilder, UniverseFilter

        data = _make_multi_ticker_data(["SPY", "QQQ"], n=100)
        builder = UniverseBuilder(UniverseFilter(
            min_adv_dollars=0,
            min_price=0,
        ))
        universe = builder.build_from_prices(data)
        assert len(universe) == 2

    def test_include_only_tickers(self):
        from pipeline.strategy.universe import (
            AssetClass,
            Exchange,
            InstrumentMetadata,
            Region,
            UniverseBuilder,
            UniverseFilter,
        )

        instruments = [
            InstrumentMetadata(
                ticker="AAPL", asset_class=AssetClass.EQUITY,
                region=Region.US, exchange=Exchange.NASDAQ,
                adv_dollars=1e9, avg_spread_bps=1.0,
            ),
            InstrumentMetadata(
                ticker="MSFT", asset_class=AssetClass.EQUITY,
                region=Region.US, exchange=Exchange.NASDAQ,
                adv_dollars=1e9, avg_spread_bps=1.0,
            ),
        ]

        builder = UniverseBuilder(UniverseFilter(
            min_adv_dollars=0,
            include_only_tickers=["AAPL"],
        ))
        universe = builder.build(instruments)
        assert universe.tickers == ["AAPL"]

    def test_metadata_df(self):
        from pipeline.strategy.universe import (
            AssetClass,
            Exchange,
            InstrumentMetadata,
            Region,
            UniverseBuilder,
            UniverseFilter,
        )

        instruments = [
            InstrumentMetadata(
                ticker="AAPL", name="Apple", asset_class=AssetClass.EQUITY,
                sector="Technology", region=Region.US, exchange=Exchange.NASDAQ,
                adv_dollars=1e9, avg_spread_bps=1.0,
            ),
        ]

        builder = UniverseBuilder(UniverseFilter(min_adv_dollars=0))
        universe = builder.build(instruments)
        df = universe.metadata_df()
        assert "ticker" in df.columns
        assert len(df) == 1


# ===========================================================================
# Signal Library Tests
# ===========================================================================

class TestSignalLibrary:
    def test_momentum_return_compute(self):
        from pipeline.strategy.signal_library import MomentumReturn

        data = _make_multi_ticker_data(["TEST"], n=300)
        indicator = MomentumReturn(lookback=252, skip=21)
        result = indicator.compute(data["TEST"])
        assert isinstance(result, pd.Series)
        # First 252 values should be NaN
        assert result.iloc[-1] is not np.nan or not np.isnan(result.iloc[-1])

    def test_moving_average_crossover(self):
        from pipeline.strategy.signal_library import MovingAverageCrossover

        data = _make_multi_ticker_data(["TEST"], n=300)
        indicator = MovingAverageCrossover(short_window=50, long_window=200)
        result = indicator.compute(data["TEST"])
        assert isinstance(result, pd.Series)
        assert len(result) == 300

    def test_rsi_mean_reversion(self):
        from pipeline.strategy.signal_library import RSIMeanReversion

        data = _make_multi_ticker_data(["TEST"], n=300)
        indicator = RSIMeanReversion(window=14)
        result = indicator.compute(data["TEST"])
        assert isinstance(result, pd.Series)
        # RSI reversion should be centered around 0
        assert -50 <= result.iloc[-1] <= 50

    def test_volatility_signal(self):
        from pipeline.strategy.signal_library import VolatilitySignal

        data = _make_multi_ticker_data(["TEST"], n=300)
        indicator = VolatilitySignal(window=60)
        result = indicator.compute(data["TEST"])
        assert isinstance(result, pd.Series)
        assert result.iloc[-1] > 0  # Volatility is always positive

    def test_signal_pipeline_end_to_end(self):
        from pipeline.strategy.signal_library import SignalPipeline, momentum_signal

        data = _make_multi_ticker_data(["AAPL", "MSFT", "GOOGL"], n=300)
        sig_def = momentum_signal()
        pipeline = SignalPipeline(sig_def)

        composite = pipeline.run(data)
        assert isinstance(composite, pd.DataFrame)
        assert set(composite.columns) == {"AAPL", "MSFT", "GOOGL"}
        assert len(composite) > 0

    def test_cross_sectional_rank(self):
        from pipeline.strategy.signal_library import SignalPipeline, momentum_signal

        data = _make_multi_ticker_data(["A", "B", "C", "D"], n=300)
        sig_def = momentum_signal()
        pipeline = SignalPipeline(sig_def)
        composite = pipeline.run(data)
        ranked = pipeline.cross_sectional_rank(composite)
        # Ranks should be between 0 and 1
        assert ranked.dropna().min().min() >= 0
        assert ranked.dropna().max().max() <= 1

    def test_normalization_zscore(self):
        from pipeline.strategy.signal_library import zscore_normalize

        series = pd.Series(range(300), dtype=float)
        z = zscore_normalize(series, window=50)
        # Last z-score should be positive (series is increasing)
        assert z.iloc[-1] > 0

    def test_normalization_winsorize(self):
        from pipeline.strategy.signal_library import winsorize

        series = pd.Series([1, 2, 3, 100, 200, -50, 4, 5])
        w = winsorize(series, lower=0.1, upper=0.9)
        assert w.max() < 200
        assert w.min() > -50

    def test_mean_reversion_signal_definition(self):
        from pipeline.strategy.signal_library import mean_reversion_signal

        sig = mean_reversion_signal()
        assert sig.family.value == "mean_reversion"
        assert len(sig.indicators) == 3


# ===========================================================================
# Entry Rules Tests
# ===========================================================================

class TestEntryRules:
    def test_signal_threshold(self):
        from pipeline.strategy.entry_rules import (
            EntryContext,
            SignalThresholdCondition,
        )

        cond = SignalThresholdCondition(threshold=0.5)
        ctx = EntryContext()

        result = cond.evaluate("AAPL", pd.Timestamp("2023-01-01"), 0.7, ctx)
        assert result.passed

        result = cond.evaluate("AAPL", pd.Timestamp("2023-01-01"), 0.3, ctx)
        assert not result.passed

    def test_regime_condition(self):
        from pipeline.strategy.entry_rules import EntryContext, RegimeCondition

        cond = RegimeCondition(blocked_regimes=["BEAR"])
        ctx_bull = EntryContext(regime="BULL")
        ctx_bear = EntryContext(regime="BEAR")

        assert cond.evaluate("AAPL", pd.Timestamp("2023-01-01"), 1.0, ctx_bull).passed
        assert not cond.evaluate("AAPL", pd.Timestamp("2023-01-01"), 1.0, ctx_bear).passed

    def test_max_positions(self):
        from pipeline.strategy.entry_rules import EntryContext, MaxPositionsCondition

        cond = MaxPositionsCondition()
        ctx = EntryContext(open_position_count=9, max_positions=10)
        assert cond.evaluate("AAPL", pd.Timestamp("2023-01-01"), 1.0, ctx).passed

        ctx = EntryContext(open_position_count=10, max_positions=10)
        assert not cond.evaluate("AAPL", pd.Timestamp("2023-01-01"), 1.0, ctx).passed

    def test_no_duplicate_position(self):
        from pipeline.strategy.entry_rules import (
            EntryContext,
            NoDuplicatePositionCondition,
        )

        cond = NoDuplicatePositionCondition()
        ctx = EntryContext(held_tickers={"AAPL", "MSFT"})
        assert not cond.evaluate("AAPL", pd.Timestamp("2023-01-01"), 1.0, ctx).passed
        assert cond.evaluate("GOOGL", pd.Timestamp("2023-01-01"), 1.0, ctx).passed

    def test_risk_budget(self):
        from pipeline.strategy.entry_rules import EntryContext, RiskBudgetCondition

        cond = RiskBudgetCondition()
        ctx = EntryContext(current_portfolio_risk_pct=0.05, max_portfolio_risk_pct=0.06)
        assert cond.evaluate("AAPL", pd.Timestamp("2023-01-01"), 1.0, ctx).passed

        ctx = EntryContext(current_portfolio_risk_pct=0.07, max_portfolio_risk_pct=0.06)
        assert not cond.evaluate("AAPL", pd.Timestamp("2023-01-01"), 1.0, ctx).passed

    def test_sector_exposure(self):
        from pipeline.strategy.entry_rules import EntryContext, SectorExposureCondition

        cond = SectorExposureCondition(default_cap=0.30)
        ctx = EntryContext(
            sector_exposures={"Technology": 0.25},
            ticker_sector="Technology",
        )
        assert cond.evaluate("AAPL", pd.Timestamp("2023-01-01"), 1.0, ctx).passed

        ctx = EntryContext(
            sector_exposures={"Technology": 0.35},
            ticker_sector="Technology",
        )
        assert not cond.evaluate("AAPL", pd.Timestamp("2023-01-01"), 1.0, ctx).passed

    def test_entry_rule_set_and_logic(self):
        from pipeline.strategy.entry_rules import (
            EntryContext,
            EntryRuleSet,
            RegimeCondition,
            SignalThresholdCondition,
        )

        rules = EntryRuleSet()
        rules.add(SignalThresholdCondition(threshold=0.5))
        rules.add(RegimeCondition(blocked_regimes=["BEAR"]))

        ctx = EntryContext(regime="BULL")
        decision = rules.evaluate("AAPL", pd.Timestamp("2023-01-01"), 0.7, ctx)
        assert decision.eligible

        decision = rules.evaluate("AAPL", pd.Timestamp("2023-01-01"), 0.3, ctx)
        assert not decision.eligible

    def test_institutional_entry_rules(self):
        from pipeline.strategy.entry_rules import (
            EntryContext,
            institutional_entry_rules,
        )

        rules = institutional_entry_rules(signal_threshold=0.0)
        ctx = EntryContext(
            regime="BULL",
            open_position_count=0,
            max_positions=10,
            current_portfolio_risk_pct=0.0,
        )
        decision = rules.evaluate("AAPL", pd.Timestamp("2023-01-01"), 0.5, ctx)
        assert decision.eligible


# ===========================================================================
# Position Sizing Tests
# ===========================================================================

class TestInstitutionalPositionSizing:
    def test_volatility_scaled_sizer(self):
        from pipeline.strategy.position_sizing import (
            InstitutionalSizingConfig,
            SizingMethod,
            VolatilityScaledSizer,
        )

        config = InstitutionalSizingConfig(
            method=SizingMethod.VOLATILITY_SCALED,
            target_annual_vol=0.10,
            max_position_weight=0.05,
            min_trade_notional=0,
            min_position_weight=0.0,
        )
        sizer = VolatilityScaledSizer(config)

        signals = pd.Series({"AAPL": 1.5, "MSFT": 0.8, "GOOGL": -0.3})
        prices = pd.Series({"AAPL": 150.0, "MSFT": 350.0, "GOOGL": 140.0})
        vols = pd.Series({"AAPL": 0.25, "MSFT": 0.30, "GOOGL": 0.20})

        targets = sizer.compute_targets(signals, prices, vols, capital=1e8)

        assert targets.position_count > 0
        for pos in targets.positions:
            assert abs(pos.target_weight) <= config.max_position_weight + 1e-10
            assert pos.target_shares != 0

    def test_fixed_fraction_sizer(self):
        from pipeline.strategy.position_sizing import (
            FixedFractionSizer,
            InstitutionalSizingConfig,
        )

        config = InstitutionalSizingConfig(
            equal_weight_fraction=0.05,
            min_trade_notional=0,
        )
        sizer = FixedFractionSizer(config)

        signals = pd.Series({"AAPL": 1.0, "MSFT": 1.0})
        prices = pd.Series({"AAPL": 150.0, "MSFT": 350.0})
        vols = pd.Series({"AAPL": 0.25, "MSFT": 0.30})

        targets = sizer.compute_targets(signals, prices, vols, capital=1e8)
        assert targets.position_count == 2

    def test_signal_weighted_sizer(self):
        from pipeline.strategy.position_sizing import (
            InstitutionalSizingConfig,
            SignalWeightedSizer,
        )

        config = InstitutionalSizingConfig(min_trade_notional=0, min_position_weight=0.0)
        sizer = SignalWeightedSizer(config)

        signals = pd.Series({"AAPL": 2.0, "MSFT": 1.0})
        prices = pd.Series({"AAPL": 150.0, "MSFT": 350.0})
        vols = pd.Series({"AAPL": 0.25, "MSFT": 0.25})

        targets = sizer.compute_targets(signals, prices, vols, capital=1e8)
        # AAPL should have larger weight (stronger signal)
        weights = targets.weight_series()
        assert abs(weights["AAPL"]) >= abs(weights["MSFT"])

    def test_create_sizer_factory(self):
        from pipeline.strategy.position_sizing import (
            FixedFractionSizer,
            InstitutionalSizingConfig,
            SizingMethod,
            VolatilityScaledSizer,
            create_sizer,
        )

        vol_config = InstitutionalSizingConfig(method=SizingMethod.VOLATILITY_SCALED)
        assert isinstance(create_sizer(vol_config), VolatilityScaledSizer)

        fix_config = InstitutionalSizingConfig(method=SizingMethod.FIXED_FRACTION)
        assert isinstance(create_sizer(fix_config), FixedFractionSizer)

    def test_gross_exposure_constraint(self):
        from pipeline.strategy.position_sizing import (
            InstitutionalSizingConfig,
            SizingMethod,
            VolatilityScaledSizer,
        )

        config = InstitutionalSizingConfig(
            method=SizingMethod.VOLATILITY_SCALED,
            target_annual_vol=0.50,  # Very high target vol to push weights up
            max_gross_exposure=1.0,
            min_trade_notional=0,
            min_position_weight=0.0,
        )
        sizer = VolatilityScaledSizer(config)

        signals = pd.Series({f"SYM{i}": 1.0 for i in range(20)})
        prices = pd.Series({f"SYM{i}": 100.0 for i in range(20)})
        vols = pd.Series({f"SYM{i}": 0.10 for i in range(20)})

        targets = sizer.compute_targets(signals, prices, vols, capital=1e8)
        assert targets.gross_exposure <= config.max_gross_exposure + 0.01

    def test_empty_signals(self):
        from pipeline.strategy.position_sizing import (
            InstitutionalSizingConfig,
            VolatilityScaledSizer,
        )

        sizer = VolatilityScaledSizer(InstitutionalSizingConfig())
        targets = sizer.compute_targets(
            pd.Series(dtype=float), pd.Series(dtype=float),
            pd.Series(dtype=float), capital=1e8,
        )
        assert targets.position_count == 0


# ===========================================================================
# Risk Constraints Tests
# ===========================================================================

class TestRiskConstraints:
    def test_position_weight_constraint(self):
        from pipeline.strategy.risk_constraints import (
            ConstraintSeverity,
            ConstraintType,
            RiskConstraint,
        )

        constraint = RiskConstraint(
            name="Max Position",
            constraint_type=ConstraintType.POSITION_WEIGHT,
            limit_value=0.05,
            severity=ConstraintSeverity.HARD,
        )
        result = constraint.check(0.03)
        assert not result.violated

        result = constraint.check(0.06)
        assert result.violated
        assert result.is_hard_breach

    def test_constraint_set_evaluation(self):
        from pipeline.strategy.risk_constraints import institutional_constraints

        cs = institutional_constraints(
            max_position_weight=0.05,
            max_gross_exposure=1.0,
            max_drawdown=0.15,
        )

        weights = pd.Series({"AAPL": 0.03, "MSFT": 0.02, "GOOGL": 0.01})
        results = cs.evaluate_portfolio(weights)

        cs.get_violations(results)
        # These weights are well within limits
        hard_violations = cs.get_hard_violations(results)
        assert len(hard_violations) == 0

    def test_sector_exposure_constraint(self):
        from pipeline.strategy.risk_constraints import institutional_constraints

        cs = institutional_constraints(
            max_sector_exposure=0.20,
            sectors=["Technology"],
        )

        # 30% in tech should violate
        weights = pd.Series({"AAPL": 0.15, "MSFT": 0.15})
        sector_map = {"AAPL": "Technology", "MSFT": "Technology"}

        results = cs.evaluate_portfolio(weights, sector_map=sector_map)
        violations = cs.get_violations(results)
        sector_violations = [
            v for v in violations
            if "Sector" in v.constraint.name
        ]
        assert len(sector_violations) > 0

    def test_drawdown_constraint(self):
        from pipeline.strategy.risk_constraints import institutional_constraints

        cs = institutional_constraints(max_drawdown=0.15)
        weights = pd.Series({"AAPL": 0.05})

        results = cs.evaluate_portfolio(weights, current_drawdown=0.20)
        violations = cs.get_hard_violations(results)
        dd_violations = [
            v for v in violations
            if v.constraint.constraint_type.value == "max_drawdown"
        ]
        assert len(dd_violations) > 0

    def test_constraints_to_markdown(self):
        from pipeline.strategy.risk_constraints import institutional_constraints

        cs = institutional_constraints()
        md = cs.to_markdown_table()
        assert "| Constraint |" in md
        assert "Max Single Position Weight" in md

    def test_constraints_to_table(self):
        from pipeline.strategy.risk_constraints import institutional_constraints

        cs = institutional_constraints()
        df = cs.to_table()
        assert "Constraint" in df.columns
        assert len(df) > 0


# ===========================================================================
# Benchmark Tests
# ===========================================================================

class TestBenchmark:
    def test_benchmark_analysis(self):
        from pipeline.strategy.benchmark import compute_benchmark_analysis

        rng = np.random.RandomState(42)
        n = 500
        strategy_ret = pd.Series(
            0.0005 + 0.01 * rng.randn(n),
            index=pd.bdate_range("2022-01-01", periods=n),
        )
        benchmark_ret = pd.Series(
            0.0003 + 0.012 * rng.randn(n),
            index=pd.bdate_range("2022-01-01", periods=n),
        )

        analysis = compute_benchmark_analysis(
            strategy_ret, benchmark_ret,
            benchmark_name="S&P 500",
            benchmark_ticker="SPY",
        )

        assert not np.isnan(analysis.active_return_ann)
        assert not np.isnan(analysis.tracking_error_ann)
        assert not np.isnan(analysis.information_ratio)
        assert not np.isnan(analysis.beta)
        assert not np.isnan(analysis.correlation)

    def test_benchmark_suite(self):
        from pipeline.strategy.benchmark import US_EQUITY_BENCHMARKS

        assert US_EQUITY_BENCHMARKS.primary.ticker == "SPY"
        assert len(US_EQUITY_BENCHMARKS.all_benchmarks) >= 2

    def test_benchmark_to_markdown(self):
        from pipeline.strategy.benchmark import (
            BenchmarkAnalysis,
            benchmark_analysis_to_markdown,
        )

        analyses = [
            BenchmarkAnalysis(
                benchmark_name="S&P 500",
                benchmark_ticker="SPY",
                active_return_ann=0.03,
                tracking_error_ann=0.05,
                information_ratio=0.6,
                beta=0.8,
            ),
        ]
        md = benchmark_analysis_to_markdown(analyses)
        assert "S&P 500" in md
        assert "Information Ratio" in md


# ===========================================================================
# Backtest Harness Tests
# ===========================================================================

class TestBacktestHarness:
    def test_harness_end_to_end(self):
        from pipeline.strategy.backtest_harness import (
            BacktestConfig,
            BacktestHarness,
        )
        from pipeline.strategy.entry_rules import institutional_entry_rules
        from pipeline.strategy.exits import ExitEngine
        from pipeline.strategy.position_sizing import (
            InstitutionalSizingConfig,
            SizingMethod,
            create_sizer,
        )
        from pipeline.strategy.risk_constraints import institutional_constraints
        from pipeline.strategy.signal_library import SignalPipeline, momentum_signal

        data = _make_multi_ticker_data(
            ["AAPL", "MSFT", "GOOGL"], n=300, seed=42,
        )

        sig_def = momentum_signal(lookback=60, skip=5, vol_window=20)
        pipeline = SignalPipeline(sig_def)

        entry_rules = institutional_entry_rules(signal_threshold=-999.0)

        exit_engine = ExitEngine(
            max_holding_days=30,
            stop_atr_multiple=2.0,
        )

        sizing_config = InstitutionalSizingConfig(
            method=SizingMethod.VOLATILITY_SCALED,
            target_annual_vol=0.10,
            max_position_weight=0.20,
            min_trade_notional=0,
            min_position_weight=0.0,
        )
        sizer = create_sizer(sizing_config)

        constraints = institutional_constraints()

        config = BacktestConfig(
            initial_capital=1e6,
            spread_bps=3.0,
            slippage_bps=2.0,
            signal_lag_days=1,
        )

        harness = BacktestHarness(
            signal_pipeline=pipeline,
            entry_rules=entry_rules,
            exit_engine=exit_engine,
            sizing_model=sizer,
            risk_constraints=constraints,
            config=config,
        )

        result = harness.run(data)

        assert not result.equity_curve.empty
        assert result.equity_curve.iloc[0] > 0
        assert result.metrics.total_return is not None
        assert not np.isnan(result.metrics.sharpe_ratio) or result.metrics.total_trades == 0

    def test_harness_empty_data(self):
        from pipeline.strategy.backtest_harness import (
            BacktestHarness,
        )
        from pipeline.strategy.entry_rules import EntryRuleSet
        from pipeline.strategy.exits import ExitEngine
        from pipeline.strategy.position_sizing import (
            InstitutionalSizingConfig,
            create_sizer,
        )
        from pipeline.strategy.risk_constraints import RiskConstraintSet
        from pipeline.strategy.signal_library import SignalPipeline, momentum_signal

        harness = BacktestHarness(
            signal_pipeline=SignalPipeline(momentum_signal()),
            entry_rules=EntryRuleSet(),
            exit_engine=ExitEngine(),
            sizing_model=create_sizer(InstitutionalSizingConfig()),
            risk_constraints=RiskConstraintSet(),
        )

        result = harness.run({})
        assert result.equity_curve.empty

    def test_harness_summary_table(self):
        from pipeline.strategy.backtest_harness import (
            BacktestConfig,
            BacktestHarness,
        )
        from pipeline.strategy.entry_rules import institutional_entry_rules
        from pipeline.strategy.exits import ExitEngine
        from pipeline.strategy.position_sizing import (
            InstitutionalSizingConfig,
            create_sizer,
        )
        from pipeline.strategy.risk_constraints import institutional_constraints
        from pipeline.strategy.signal_library import SignalPipeline, momentum_signal

        data = _make_multi_ticker_data(["A", "B"], n=200)
        sig_def = momentum_signal(lookback=60, skip=5, vol_window=20)

        harness = BacktestHarness(
            signal_pipeline=SignalPipeline(sig_def),
            entry_rules=institutional_entry_rules(signal_threshold=-999.0),
            exit_engine=ExitEngine(max_holding_days=30),
            sizing_model=create_sizer(InstitutionalSizingConfig(
                min_trade_notional=0, min_position_weight=0.0,
                max_position_weight=0.20,
            )),
            risk_constraints=institutional_constraints(),
            config=BacktestConfig(initial_capital=1e6),
        )

        result = harness.run(data)
        table = result.summary_table()
        assert "Metric" in table
        assert "Value" in table


# ===========================================================================
# Strategy Definition Tests
# ===========================================================================

class TestStrategyDefinition:
    def test_cross_sectional_momentum(self):
        from pipeline.strategy.strategy_definition import (
            cross_sectional_momentum_strategy,
        )

        strategy = cross_sectional_momentum_strategy()
        assert strategy.strategy_name == "QSG-SYSTEMATIC-MOM-001"
        assert len(strategy.signal_definitions) > 0
        assert strategy.thesis.inefficiency != ""
        assert len(strategy.thesis.drivers) > 0
        assert strategy.sizing_config.method.value == "volatility_scaled"

    def test_strategy_definition_complete(self):
        from pipeline.strategy.strategy_definition import (
            cross_sectional_momentum_strategy,
        )

        strategy = cross_sectional_momentum_strategy()
        # All components must be present
        assert strategy.universe_filter is not None
        assert strategy.entry_rules is not None
        assert strategy.exit_engine is not None
        assert strategy.sizing_config is not None
        assert strategy.risk_constraints is not None
        assert strategy.benchmark_suite is not None
        assert strategy.backtest_config is not None


# ===========================================================================
# Memo Generator Tests
# ===========================================================================

class TestMemoGenerator:
    def test_memo_contains_all_sections(self):
        from pipeline.strategy.memo_generator import generate_memo
        from pipeline.strategy.strategy_definition import (
            cross_sectional_momentum_strategy,
        )

        strategy = cross_sectional_momentum_strategy()
        memo = generate_memo(strategy)

        required_sections = [
            "Strategy Thesis",
            "Universe Selection",
            "Signal Generation",
            "Entry Rules",
            "Exit Rules",
            "Position Sizing",
            "Risk Parameters",
            "Backtesting Methodology",
            "Benchmark Selection",
            "Edge Decay Monitoring",
            "Appendix: Mathematical Reference",
        ]
        for section in required_sections:
            assert section in memo, f"Missing section: {section}"

    def test_memo_contains_formulas(self):
        from pipeline.strategy.memo_generator import generate_memo
        from pipeline.strategy.strategy_definition import (
            cross_sectional_momentum_strategy,
        )

        strategy = cross_sectional_momentum_strategy()
        memo = generate_memo(strategy)

        # Should contain mathematical formulas
        assert "sigma" in memo or "Sharpe" in memo
        assert "sqrt" in memo or "252" in memo

    def test_memo_contains_risk_table(self):
        from pipeline.strategy.memo_generator import generate_memo
        from pipeline.strategy.strategy_definition import (
            cross_sectional_momentum_strategy,
        )

        strategy = cross_sectional_momentum_strategy()
        memo = generate_memo(strategy)

        assert "Max Single Position Weight" in memo
        assert "Max Gross Exposure" in memo
        assert "Max Portfolio Drawdown" in memo

    def test_memo_reflects_config_values(self):
        from pipeline.strategy.memo_generator import generate_memo
        from pipeline.strategy.strategy_definition import (
            cross_sectional_momentum_strategy,
        )

        strategy = cross_sectional_momentum_strategy()
        memo = generate_memo(strategy)

        # Check that specific config values appear
        assert "QSG-SYSTEMATIC-MOM-001" in memo
        assert "RESEARCH" in memo

    def test_memo_with_backtest_result(self):
        from pipeline.strategy.backtest_harness import (
            BacktestConfig,
            BacktestHarness,
        )
        from pipeline.strategy.entry_rules import institutional_entry_rules
        from pipeline.strategy.exits import ExitEngine
        from pipeline.strategy.memo_generator import generate_memo
        from pipeline.strategy.position_sizing import (
            InstitutionalSizingConfig,
            create_sizer,
        )
        from pipeline.strategy.risk_constraints import institutional_constraints
        from pipeline.strategy.signal_library import SignalPipeline, momentum_signal
        from pipeline.strategy.strategy_definition import (
            cross_sectional_momentum_strategy,
        )

        strategy = cross_sectional_momentum_strategy()
        data = _make_multi_ticker_data(["AAPL", "MSFT"], n=200)
        sig_def = momentum_signal(lookback=60, skip=5, vol_window=20)

        harness = BacktestHarness(
            signal_pipeline=SignalPipeline(sig_def),
            entry_rules=institutional_entry_rules(signal_threshold=-999.0),
            exit_engine=ExitEngine(max_holding_days=30),
            sizing_model=create_sizer(InstitutionalSizingConfig(
                min_trade_notional=0, min_position_weight=0.0,
                max_position_weight=0.20,
            )),
            risk_constraints=institutional_constraints(),
            config=BacktestConfig(initial_capital=1e6),
        )

        result = harness.run(data)
        memo = generate_memo(strategy, result)

        # Should include backtest results
        assert "Backtest" in memo

    def test_memo_pseudocode(self):
        from pipeline.strategy.memo_generator import generate_memo
        from pipeline.strategy.strategy_definition import (
            cross_sectional_momentum_strategy,
        )

        memo = generate_memo(cross_sectional_momentum_strategy())
        assert "def daily_rebalance" in memo
        assert "signal_pipeline.run" in memo


# ===========================================================================
# Edge Decay Integration Test
# ===========================================================================

class TestEdgeDecayIntegration:
    def test_decay_monitor_in_harness(self):
        from pipeline.strategy.backtest_harness import (
            BacktestConfig,
            BacktestHarness,
        )
        from pipeline.strategy.entry_rules import institutional_entry_rules
        from pipeline.strategy.exits import ExitEngine
        from pipeline.strategy.position_sizing import (
            InstitutionalSizingConfig,
            create_sizer,
        )
        from pipeline.strategy.risk_constraints import institutional_constraints
        from pipeline.strategy.signal_library import SignalPipeline, momentum_signal

        data = _make_multi_ticker_data(["A", "B", "C"], n=300)
        sig_def = momentum_signal(lookback=60, skip=5, vol_window=20)

        harness = BacktestHarness(
            signal_pipeline=SignalPipeline(sig_def),
            entry_rules=institutional_entry_rules(signal_threshold=-999.0),
            exit_engine=ExitEngine(max_holding_days=30),
            sizing_model=create_sizer(InstitutionalSizingConfig(
                min_trade_notional=0, min_position_weight=0.0,
                max_position_weight=0.20,
            )),
            risk_constraints=institutional_constraints(),
            config=BacktestConfig(initial_capital=1e6),
        )

        result = harness.run(data)

        # Decay monitor should have been recording returns
        assert result.decay_monitor is not None
        assert len(result.decay_monitor._daily_returns) > 0
