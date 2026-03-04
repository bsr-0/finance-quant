"""Tests for extended backtesting modules: event engine, Monte Carlo,
survivorship bias, and look-ahead bias checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.backtesting.event_engine import (
    EventDrivenBacktester,
    EventEngineConfig,
    Order,
)
from pipeline.backtesting.monte_carlo import (
    MonteCarloConfig,
    block_bootstrap,
    execution_stress_test,
    monte_carlo_simulation,
)
from pipeline.backtesting.survivorship import (
    CorporateAction,
    CorporateActionMapper,
    SymbolInfo,
    SymbolUniverse,
    filter_universe_at_date,
)
from pipeline.backtesting.bias_checks import (
    check_no_future_data,
    data_shift_test,
    enforce_timestamp_ordering,
    random_shuffle_test,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_returns():
    """Daily returns with slight positive drift."""
    np.random.seed(42)
    n = 500
    idx = pd.bdate_range("2022-01-01", periods=n)
    returns = np.random.normal(0.0003, 0.01, n)
    return pd.Series(returns, index=idx)


@pytest.fixture
def sample_price_df():
    """Tick-level price data for event engine testing."""
    np.random.seed(42)
    n = 200
    idx = pd.bdate_range("2024-01-01", periods=n)
    prices = 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.01, n)))
    return pd.DataFrame({
        "close": prices,
        "open": prices * (1 + np.random.randn(n) * 0.002),
        "high": prices * (1 + abs(np.random.randn(n) * 0.005)),
        "low": prices * (1 - abs(np.random.randn(n) * 0.005)),
        "volume": np.random.randint(100_000, 1_000_000, n).astype(float),
    }, index=idx)


# ---------------------------------------------------------------------------
# Event-Driven Backtester Tests
# ---------------------------------------------------------------------------


class TestEventDrivenBacktester:
    def test_basic_run(self, sample_price_df):
        config = EventEngineConfig(initial_capital=100_000)
        engine = EventDrivenBacktester(config)
        engine.load_market_data(sample_price_df, symbol="TEST")
        result = engine.run()
        assert not result.equity_curve.empty
        assert result.equity_curve.iloc[0] == pytest.approx(100_000)

    def test_market_order_execution(self, sample_price_df):
        config = EventEngineConfig(initial_capital=100_000, latency_ms=0)
        engine = EventDrivenBacktester(config)
        engine.load_market_data(sample_price_df, symbol="TEST")

        buy_triggered = [False]

        def on_data(data):
            if not buy_triggered[0] and data.get("price", 0) > 0:
                buy_triggered[0] = True
                return [Order(
                    symbol="TEST", side="buy",
                    quantity=100, order_type="market",
                )]
            return None

        engine.on_market_data = on_data
        result = engine.run()
        assert len(result.fills) >= 1
        assert result.fills[0].fill_price > 0

    def test_fees_deducted(self, sample_price_df):
        config = EventEngineConfig(
            initial_capital=100_000,
            taker_fee_bps=5.0,
            latency_ms=0,
        )
        engine = EventDrivenBacktester(config)
        engine.load_market_data(sample_price_df, symbol="TEST")

        def on_data(data):
            return [Order(symbol="TEST", side="buy", quantity=10, order_type="market")]

        engine.on_market_data = on_data
        result = engine.run()
        assert result.total_fees > 0

    def test_summary_keys(self, sample_price_df):
        engine = EventDrivenBacktester()
        engine.load_market_data(sample_price_df, symbol="TEST")
        result = engine.run()
        summary = result.summary()
        assert "initial_nav" in summary
        assert "total_return" in summary
        assert "sharpe_ratio" in summary
        assert "max_drawdown" in summary

    def test_deterministic_with_seed(self, sample_price_df):
        results = []
        for _ in range(2):
            engine = EventDrivenBacktester(EventEngineConfig(seed=42))
            engine.load_market_data(sample_price_df, symbol="TEST")
            result = engine.run()
            results.append(result.equity_curve.iloc[-1])
        assert results[0] == pytest.approx(results[1])


# ---------------------------------------------------------------------------
# Monte Carlo Tests
# ---------------------------------------------------------------------------


class TestMonteCarlo:
    def test_block_bootstrap_shape(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, 252)
        paths = block_bootstrap(returns, n_paths=100, path_length=252, block_size=21, rng=rng)
        assert paths.shape == (100, 252)

    def test_monte_carlo_simulation(self, sample_returns):
        cfg = MonteCarloConfig(n_simulations=100, seed=42)
        result = monte_carlo_simulation(sample_returns, initial_capital=1_000_000, config=cfg)
        assert len(result.simulated_final_values) == 100
        assert len(result.simulated_max_drawdowns) == 100
        assert result.probability_of_loss >= 0
        assert result.probability_of_ruin >= 0

    def test_summary_stats(self, sample_returns):
        result = monte_carlo_simulation(
            sample_returns, config=MonteCarloConfig(n_simulations=50, seed=42)
        )
        stats = result.summary_stats
        assert "mean_final_value" in stats
        assert "mean_max_drawdown" in stats
        assert "skewness" in stats
        assert "kurtosis" in stats

    def test_percentiles(self, sample_returns):
        result = monte_carlo_simulation(
            sample_returns, config=MonteCarloConfig(n_simulations=100, seed=42)
        )
        assert "p5" in result.percentiles
        assert "p95" in result.percentiles
        assert result.percentiles["p5"] <= result.percentiles["p95"]

    def test_execution_stress(self, sample_returns):
        result = execution_stress_test(
            sample_returns, n_scenarios=20, seed=42
        )
        assert len(result) == 20
        assert "slippage_bps" in result.columns
        assert "sharpe" in result.columns

    def test_too_few_returns_raises(self):
        short = pd.Series([0.01, 0.02, 0.03])
        with pytest.raises(ValueError, match="at least 10"):
            monte_carlo_simulation(short)


# ---------------------------------------------------------------------------
# Survivorship Bias Tests
# ---------------------------------------------------------------------------


class TestSymbolUniverse:
    def test_active_symbols(self):
        universe = SymbolUniverse()
        universe.add_symbol(SymbolInfo(
            "AAPL", listing_date=pd.Timestamp("1980-12-12")
        ))
        universe.add_symbol(SymbolInfo(
            "ENRN",
            listing_date=pd.Timestamp("1985-04-01"),
            delisting_date=pd.Timestamp("2001-12-02"),
        ))

        # In 2000, both active
        active_2000 = universe.get_active_symbols(pd.Timestamp("2000-01-01"))
        assert "AAPL" in active_2000
        assert "ENRN" in active_2000

        # In 2002, ENRN delisted
        active_2002 = universe.get_active_symbols(pd.Timestamp("2002-01-01"))
        assert "AAPL" in active_2002
        assert "ENRN" not in active_2002

    def test_listing_date_filter(self):
        universe = SymbolUniverse()
        universe.add_symbol(SymbolInfo(
            "NEW", listing_date=pd.Timestamp("2020-01-01")
        ))
        # Before listing
        assert "NEW" not in universe.get_active_symbols(pd.Timestamp("2019-01-01"))
        # After listing
        assert "NEW" in universe.get_active_symbols(pd.Timestamp("2020-06-01"))

    def test_summary(self):
        universe = SymbolUniverse()
        universe.add_symbol(SymbolInfo("A"))
        universe.add_symbol(SymbolInfo("B", delisting_date=pd.Timestamp("2020-01-01")))
        summary = universe.summary()
        assert summary["total"] == 2
        assert summary["active"] == 1
        assert summary["delisted"] == 1

    def test_bulk_load(self):
        df = pd.DataFrame({
            "ticker": ["X", "Y"],
            "listing_date": ["2010-01-01", "2015-01-01"],
            "delisting_date": [None, "2020-01-01"],
        })
        universe = SymbolUniverse()
        universe.add_symbols_from_df(df)
        assert len(universe.all_symbols) == 2


class TestCorporateActionMapper:
    def test_rename_resolution(self):
        mapper = CorporateActionMapper()
        mapper.add_action(CorporateAction(
            date=pd.Timestamp("2020-01-01"),
            old_ticker="FB",
            new_ticker="META",
        ))
        assert mapper.resolve_current("FB") == "META"
        assert "FB" in mapper.resolve_historical("META")

    def test_chain_resolution(self):
        mapper = CorporateActionMapper()
        mapper.add_action(CorporateAction(
            date=pd.Timestamp("2010-01-01"),
            old_ticker="A", new_ticker="B",
        ))
        mapper.add_action(CorporateAction(
            date=pd.Timestamp("2020-01-01"),
            old_ticker="B", new_ticker="C",
        ))
        assert mapper.resolve_current("A") == "C"
        historical = mapper.resolve_historical("C")
        assert "A" in historical
        assert "B" in historical

    def test_price_adjustment(self):
        mapper = CorporateActionMapper()
        mapper.add_action(CorporateAction(
            date=pd.Timestamp("2020-06-01"),
            old_ticker="AAPL",
            new_ticker="AAPL",
            action_type="split",
            adjustment_factor=4.0,
        ))
        assert mapper.get_price_adjustment("AAPL") == pytest.approx(4.0)


class TestFilterUniverseAtDate:
    def test_filters_correctly(self):
        universe = SymbolUniverse()
        universe.add_symbol(SymbolInfo(
            "A", listing_date=pd.Timestamp("2020-01-01")
        ))
        universe.add_symbol(SymbolInfo(
            "B", listing_date=pd.Timestamp("2020-01-01"),
            delisting_date=pd.Timestamp("2021-06-01"),
        ))

        dates = pd.bdate_range("2020-01-01", periods=500)
        price_data = {
            "A": pd.DataFrame({"close": range(500)}, index=dates),
            "B": pd.DataFrame({"close": range(500)}, index=dates),
        }

        # In 2021, both active
        filtered = filter_universe_at_date(
            price_data, universe, pd.Timestamp("2021-01-01")
        )
        assert "A" in filtered
        assert "B" in filtered
        # Data truncated to date
        assert filtered["A"].index[-1] <= pd.Timestamp("2021-01-01")

        # In 2022, B delisted
        filtered = filter_universe_at_date(
            price_data, universe, pd.Timestamp("2022-01-01")
        )
        assert "A" in filtered
        assert "B" not in filtered


# ---------------------------------------------------------------------------
# Bias Check Tests
# ---------------------------------------------------------------------------


class TestBiasChecks:
    def test_check_no_future_data_sorted(self):
        idx = pd.bdate_range("2020-01-01", periods=100)
        features = pd.DataFrame({"a": range(100)}, index=idx)
        signals = pd.DataFrame({"s": range(100)}, index=idx)
        passed, violations = check_no_future_data(features, signals)
        assert passed
        assert len(violations) == 0

    def test_check_no_future_data_unsorted(self):
        idx = pd.bdate_range("2020-01-01", periods=100)
        features = pd.DataFrame({"a": range(100)}, index=idx[::-1])
        signals = pd.DataFrame({"s": range(100)}, index=idx)
        passed, violations = check_no_future_data(features, signals)
        assert not passed
        assert any("not sorted" in v.lower() for v in violations)

    def test_enforce_timestamp_ordering(self):
        idx = pd.bdate_range("2020-01-01", periods=100)
        shuffled = pd.DataFrame({"a": range(100)}, index=idx[::-1])
        sorted_df = enforce_timestamp_ordering(shuffled)
        assert sorted_df.index.is_monotonic_increasing

    def test_random_shuffle_test(self, sample_returns):
        def sharpe_fn(rets):
            if rets.std() == 0:
                return 0.0
            return float(rets.mean() / rets.std() * np.sqrt(252))

        suspicious, stats = random_shuffle_test(
            sample_returns, sharpe_fn, n_shuffles=50, seed=42
        )
        assert "original_metric" in stats
        assert "z_score" in stats

    def test_data_shift_test(self):
        idx = pd.bdate_range("2020-01-01", periods=200)
        np.random.seed(42)
        features = pd.DataFrame({
            "feature": np.random.randn(200),
            "target": np.random.randn(200) * 0.01,
        }, index=idx)

        def metric_fn(df):
            return float(df["target"].mean() * 252)

        suspicious, stats = data_shift_test(features, "target", metric_fn)
        assert "original_metric" in stats
        assert "shifted_metric" in stats
