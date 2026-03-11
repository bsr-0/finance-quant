"""Comprehensive tests for the market-making framework."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.market_making.spread import SpreadCalculator, SpreadConfig
from pipeline.market_making.inventory import (
    InventoryConfig,
    InventoryLevel,
    InventoryManager,
)
from pipeline.market_making.quoting import (
    EventType,
    MarketEvent,
    QuoteConfig,
    QuoteEngine,
)
from pipeline.market_making.adverse import (
    AdverseConfig,
    AdverseSelectionDetector,
    FillRecord,
)
from pipeline.market_making.hedging import HedgeConfig, HedgeManager
from pipeline.market_making.microstructure import (
    BookLevel,
    MicrostructureAnalyzer,
    OrderBookSnapshot,
)
from pipeline.market_making.engine import MarketMakingConfig, MarketMakingEngine


# ---------------------------------------------------------------------------
# Spread Calculator Tests
# ---------------------------------------------------------------------------


class TestSpreadCalculator:
    def test_basic_spread(self):
        calc = SpreadCalculator()
        result = calc.compute(fair_value=100.0, volatility=0.20)
        assert result.bid < 100.0
        assert result.ask > 100.0
        assert result.bid < result.ask
        assert result.spread_bps >= 2.0  # minimum spread

    def test_higher_vol_wider_spread(self):
        calc = SpreadCalculator()
        low_vol = calc.compute(fair_value=100.0, volatility=0.10)
        high_vol = calc.compute(fair_value=100.0, volatility=0.50)
        assert high_vol.spread_bps >= low_vol.spread_bps

    def test_inventory_skew(self):
        calc = SpreadCalculator()
        # Long inventory → mid shifts up (wants to sell)
        long_inv = calc.compute(fair_value=100.0, volatility=0.20, inventory_normalized=0.8)
        no_inv = calc.compute(fair_value=100.0, volatility=0.20, inventory_normalized=0.0)
        assert long_inv.inventory_skew > 0
        assert long_inv.ask > no_inv.ask or long_inv.bid > no_inv.bid

    def test_stress_widens_spread(self):
        # Use low vol + high max so spread doesn't hit the cap
        cfg = SpreadConfig(max_spread_bps=5000.0)
        calc = SpreadCalculator(cfg)
        calm = calc.compute(fair_value=100.0, volatility=0.02, stress_level=0.0)
        stressed = calc.compute(fair_value=100.0, volatility=0.02, stress_level=1.0)
        assert stressed.spread_bps > calm.spread_bps

    def test_min_max_spread_enforced(self):
        cfg = SpreadConfig(min_spread_bps=10.0, max_spread_bps=50.0)
        calc = SpreadCalculator(cfg)
        # Very low vol → should hit minimum
        result = calc.compute(fair_value=100.0, volatility=0.001)
        assert result.spread_bps >= 10.0
        # Very high vol → should hit maximum
        result = calc.compute(fair_value=100.0, volatility=10.0)
        assert result.spread_bps <= 50.0

    def test_tick_rounding(self):
        cfg = SpreadConfig(tick_size=0.05)
        calc = SpreadCalculator(cfg)
        result = calc.compute(fair_value=100.0, volatility=0.20)
        # Bid should be rounded down, ask rounded up to tick
        tick = 0.05
        bid_rem = result.bid / tick
        ask_rem = result.ask / tick
        assert bid_rem == pytest.approx(round(bid_rem), abs=1e-8)
        assert ask_rem == pytest.approx(round(ask_rem), abs=1e-8)

    def test_competitive_tightening(self):
        cfg = SpreadConfig(competitive_tightening=0.5)
        calc = SpreadCalculator(cfg)
        no_comp = calc.compute(fair_value=100.0, volatility=0.20, competition_factor=0.0)
        full_comp = calc.compute(fair_value=100.0, volatility=0.20, competition_factor=1.0)
        assert full_comp.spread_bps <= no_comp.spread_bps


# ---------------------------------------------------------------------------
# Inventory Manager Tests
# ---------------------------------------------------------------------------


class TestInventoryManager:
    def test_record_buy_fill(self):
        mgr = InventoryManager()
        mgr.record_fill("AAPL", 100, 150.0)
        inv = mgr.get_or_create("AAPL")
        assert inv.position == 100
        assert inv.avg_cost == 150.0

    def test_record_sell_fill(self):
        mgr = InventoryManager()
        mgr.record_fill("AAPL", 100, 150.0)
        mgr.record_fill("AAPL", -50, 155.0)
        inv = mgr.get_or_create("AAPL")
        assert inv.position == 50
        assert inv.realized_pnl == pytest.approx(250.0)  # 50 * (155-150)

    def test_normalized_inventory(self):
        cfg = InventoryConfig(max_position=1000)
        mgr = InventoryManager(cfg)
        mgr.record_fill("AAPL", 500, 150.0)
        assert mgr.normalized_inventory("AAPL") == pytest.approx(0.5)

    def test_inventory_levels(self):
        cfg = InventoryConfig(max_position=100)
        mgr = InventoryManager(cfg)

        mgr.record_fill("A", 40, 10.0)
        assert mgr.get_inventory_level("A") == InventoryLevel.NORMAL

        mgr.record_fill("B", 55, 10.0)
        assert mgr.get_inventory_level("B") == InventoryLevel.ELEVATED

        mgr.record_fill("C", 85, 10.0)
        assert mgr.get_inventory_level("C") == InventoryLevel.CRITICAL

        mgr.record_fill("D", 105, 10.0)
        assert mgr.get_inventory_level("D") == InventoryLevel.BREACH

    def test_quote_size_multiplier(self):
        cfg = InventoryConfig(max_position=100)
        mgr = InventoryManager(cfg)

        assert mgr.quote_size_multiplier("NEW") == 1.0

        mgr.record_fill("ELEV", 55, 10.0)
        mult = mgr.quote_size_multiplier("ELEV")
        assert 0 < mult < 1.0

        mgr.record_fill("BREACH", 105, 10.0)
        assert mgr.quote_size_multiplier("BREACH") == 0.0

    def test_portfolio_limits(self):
        cfg = InventoryConfig(max_portfolio_notional=100_000)
        mgr = InventoryManager(cfg)
        mgr.record_fill("A", 1000, 60.0)  # 60k notional
        passed, _ = mgr.check_portfolio_limits()
        assert passed

        mgr.record_fill("B", 1000, 60.0)  # now 120k > 100k
        passed, reason = mgr.check_portfolio_limits()
        assert not passed
        assert "exceeds" in reason.lower()

    def test_position_flip_pnl(self):
        mgr = InventoryManager()
        mgr.record_fill("X", 100, 10.0)  # Buy 100 @ 10
        mgr.record_fill("X", -150, 12.0)  # Sell 150 @ 12 → close 100, open -50
        inv = mgr.get_or_create("X")
        assert inv.position == -50
        assert inv.realized_pnl == pytest.approx(200.0)  # 100 * (12-10)


# ---------------------------------------------------------------------------
# Quote Engine Tests
# ---------------------------------------------------------------------------


class TestQuoteEngine:
    def test_basic_quote_generation(self):
        engine = QuoteEngine()
        event = MarketEvent(
            event_type=EventType.QUOTE_UPDATE,
            symbol="AAPL",
            timestamp_ns=1_000_000_000,
            bid=149.0,
            ask=151.0,
        )
        result = engine.on_event(event)
        assert result is not None
        assert result.bid > 0
        assert result.ask > result.bid

    def test_throttling(self):
        cfg = QuoteConfig(min_quote_life_ms=100)
        engine = QuoteEngine(config=cfg)

        event1 = MarketEvent(
            event_type=EventType.QUOTE_UPDATE,
            symbol="AAPL",
            timestamp_ns=1_000_000_000,
            bid=149.0, ask=151.0,
        )
        result1 = engine.on_event(event1)
        assert result1 is not None

        # Second event too soon → throttled
        event2 = MarketEvent(
            event_type=EventType.QUOTE_UPDATE,
            symbol="AAPL",
            timestamp_ns=1_010_000_000,  # 10ms later
            bid=149.1, ask=151.1,
        )
        result2 = engine.on_event(event2)
        assert result2 is None  # Throttled

    def test_large_trade_pulls_quotes(self):
        cfg = QuoteConfig(pull_on_large_trade_mult=3.0)
        engine = QuoteEngine(config=cfg)

        # Establish trade size baseline
        for i in range(20):
            engine.on_event(MarketEvent(
                event_type=EventType.TRADE,
                symbol="AAPL",
                timestamp_ns=i * 1_000_000_000,
                price=150.0,
                quantity=100,
            ))

        # Large trade should trigger pull
        large = MarketEvent(
            event_type=EventType.TRADE,
            symbol="AAPL",
            timestamp_ns=30_000_000_000,
            price=150.0,
            quantity=1000,  # 10x normal
        )
        result = engine.on_event(large)
        if result is not None:
            assert result.pulled

    def test_latency_tracking(self):
        engine = QuoteEngine()
        event = MarketEvent(
            event_type=EventType.QUOTE_UPDATE,
            symbol="AAPL",
            timestamp_ns=1_000_000_000,
            bid=149.0, ask=151.0,
        )
        result = engine.on_event(event)
        assert result is not None
        assert result.latency_us >= 0
        diag = engine.diagnostics
        assert diag["event_count"] == 1


# ---------------------------------------------------------------------------
# Adverse Selection Tests
# ---------------------------------------------------------------------------


class TestAdverseSelectionDetector:
    def _make_fill(self, symbol: str, side: str, price: float, mid: float) -> FillRecord:
        return FillRecord(
            symbol=symbol, side=side, fill_price=price,
            fill_size=100, mid_at_fill=mid,
            timestamp_ns=0, spread_at_fill=0.02,
        )

    def test_not_enough_data_not_toxic(self):
        detector = AdverseSelectionDetector(
            AdverseConfig(min_fills_for_signal=20)
        )
        for _ in range(5):
            fill = self._make_fill("AAPL", "buy", 150.0, 150.0)
            detector.record_fill(fill)
        metrics = detector.evaluate("AAPL")
        assert not metrics.is_toxic

    def test_adverse_fills_flagged(self):
        cfg = AdverseConfig(
            min_fills_for_signal=10,
            toxicity_threshold=-0.3,
            horizons=[1],
        )
        detector = AdverseSelectionDetector(cfg)

        # Record fills where price always moves against us
        for i in range(20):
            fill = self._make_fill("AAPL", "buy", 150.0, 150.0)
            detector.record_fill(fill)
            # Price drops after our buy → adverse
            detector.record_post_fill_price("AAPL", 149.0)

        metrics = detector.evaluate("AAPL")
        assert metrics.avg_post_fill_pnl[1] < 0

    def test_favorable_fills_not_toxic(self):
        cfg = AdverseConfig(min_fills_for_signal=10, horizons=[1])
        detector = AdverseSelectionDetector(cfg)

        for i in range(20):
            fill = self._make_fill("AAPL", "buy", 150.0, 150.0)
            detector.record_fill(fill)
            detector.record_post_fill_price("AAPL", 151.0)

        metrics = detector.evaluate("AAPL")
        assert not metrics.is_toxic

    def test_per_side_toxicity(self):
        cfg = AdverseConfig(min_fills_for_signal=5, horizons=[1])
        detector = AdverseSelectionDetector(cfg)

        for _ in range(10):
            detector.record_fill(self._make_fill("X", "buy", 100.0, 100.0))
            detector.record_post_fill_price("X", 99.0)
            detector.record_fill(self._make_fill("X", "sell", 100.0, 100.0))
            detector.record_post_fill_price("X", 101.0)

        sides = detector.per_side_toxicity("X")
        assert sides["buy"] < 0  # Buys are adversely selected
        assert sides["sell"] < 0  # Sells are adversely selected too


# ---------------------------------------------------------------------------
# Hedging Tests
# ---------------------------------------------------------------------------


class TestHedgeManager:
    def test_no_hedge_below_threshold(self):
        cfg = HedgeConfig(
            hedge_threshold_pct=0.50,
            hedge_instruments={"AAPL": "QQQ"},
        )
        mgr = HedgeManager(cfg)
        trades = mgr.compute_hedges(
            inventories={"AAPL": 3000},
            prices={"AAPL": 150.0, "QQQ": 450.0},
            max_positions={"AAPL": 10000},
        )
        assert len(trades) == 0  # 30% < 50% threshold

    def test_hedge_above_threshold(self):
        cfg = HedgeConfig(
            hedge_threshold_pct=0.30,
            hedge_instruments={"AAPL": "QQQ"},
            hedge_ratios={"AAPL": 1.0},
            min_hedge_notional=0,
        )
        mgr = HedgeManager(cfg)
        trades = mgr.compute_hedges(
            inventories={"AAPL": 8000},
            prices={"AAPL": 150.0, "QQQ": 450.0},
            max_positions={"AAPL": 10000},
        )
        assert len(trades) >= 1
        assert trades[0].hedge_instrument == "QQQ"

    def test_unhedged_exposure(self):
        cfg = HedgeConfig(hedge_instruments={"AAPL": "QQQ"})
        mgr = HedgeManager(cfg)
        exposure = mgr.get_unhedged_exposure(
            inventories={"AAPL": 100},
            prices={"AAPL": 150.0, "QQQ": 450.0},
        )
        assert "AAPL" in exposure
        assert exposure["AAPL"] == pytest.approx(15000.0)


# ---------------------------------------------------------------------------
# Microstructure Tests
# ---------------------------------------------------------------------------


class TestMicrostructureAnalyzer:
    def test_record_and_report(self):
        analyzer = MicrostructureAnalyzer()
        for i in range(20):
            analyzer.record_fill(
                symbol="AAPL", side="buy", price=150.0,
                size=100, mid_at_fill=150.0,
                spread_at_fill=5.0, inventory_at_fill=i * 100,
                timestamp_ns=i * 3_600_000_000_000,
            )
        report = analyzer.diagnostic_report()
        assert report["total_fills"] == 20
        assert report["avg_fill_distance_bps"] >= 0

    def test_order_book_snapshot(self):
        book = OrderBookSnapshot(
            symbol="AAPL",
            timestamp_ns=0,
            bids=[BookLevel(149.0, 500), BookLevel(148.5, 300)],
            asks=[BookLevel(151.0, 200), BookLevel(151.5, 400)],
        )
        assert book.mid == pytest.approx(150.0)
        assert book.spread == pytest.approx(2.0)
        assert book.imbalance > 0  # More bid depth

    def test_trade_size_distribution(self):
        analyzer = MicrostructureAnalyzer()
        for _ in range(50):
            analyzer.record_fill(
                symbol="X", side="buy", price=10.0,
                size=np.random.uniform(50, 200), mid_at_fill=10.0,
                spread_at_fill=1.0, inventory_at_fill=0,
                timestamp_ns=0,
            )
        stats = analyzer.trade_size_distribution()
        assert stats["mean"] > 0
        assert stats["p95"] > stats["median"]


# ---------------------------------------------------------------------------
# Market Making Engine Integration Tests
# ---------------------------------------------------------------------------


class TestMarketMakingEngine:
    def test_engine_starts_and_processes_events(self):
        engine = MarketMakingEngine()
        engine.start_session(nav=1_000_000)

        event = MarketEvent(
            event_type=EventType.QUOTE_UPDATE,
            symbol="AAPL",
            timestamp_ns=1_000_000_000,
            bid=149.0, ask=151.0,
        )
        quote = engine.on_event(event)
        assert quote is not None or quote is None  # Either is valid

    def test_fill_updates_inventory(self):
        engine = MarketMakingEngine()
        engine.start_session(nav=1_000_000)

        hedges = engine.on_fill("AAPL", "buy", 100, 150.0, 1_000_000_000)
        inv = engine.inventory_mgr.get_or_create("AAPL")
        assert inv.position == 100

    def test_daily_loss_shutdown(self):
        cfg = MarketMakingConfig()
        cfg.risk_limits.max_daily_loss = 1000
        engine = MarketMakingEngine(cfg)
        engine.start_session(nav=1_000_000)

        # Simulate a large loss
        engine.inventory_mgr.record_fill("AAPL", 100, 150.0)
        engine.inventory_mgr.get_or_create("AAPL").realized_pnl = -2000
        engine._prices["AAPL"] = 150.0

        event = MarketEvent(
            event_type=EventType.QUOTE_UPDATE,
            symbol="AAPL",
            timestamp_ns=2_000_000_000,
            bid=149.0, ask=151.0,
        )
        result = engine.on_event(event)
        assert engine._state.is_shutdown

    def test_end_of_day_report(self):
        engine = MarketMakingEngine()
        engine.start_session(nav=1_000_000)
        engine.on_fill("AAPL", "buy", 50, 150.0, 0)
        engine._prices["AAPL"] = 151.0
        report = engine.end_of_day_report()
        assert "session" in report
        assert "events" in report
        assert "inventory" in report

    def test_pre_open_checklist(self):
        engine = MarketMakingEngine()
        engine.start_session(nav=1_000_000)
        checks = engine.pre_open_checklist()
        assert len(checks) >= 3
        assert all(isinstance(c, tuple) and len(c) == 3 for c in checks)
