"""Tests for the QSG-MICRO-SWING-001 strategy modules."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.strategy.signals import SignalEngine, compute_indicators
from pipeline.strategy.sizing import PositionSizer
from pipeline.strategy.exits import ExitEngine, ExitReason, PositionState
from pipeline.strategy.risk import SwingRiskManager, DrawdownLevel
from pipeline.strategy.edge_decay import EdgeDecayMonitor, AlertLevel
from pipeline.strategy.engine import SwingStrategyEngine, StrategyConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_price_df(
    n: int = 200,
    start_price: float = 100.0,
    trend: float = 0.0005,
    noise: float = 0.01,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2023-01-01", periods=n, freq="B")
    log_returns = trend + noise * rng.randn(n)
    close = start_price * np.exp(np.cumsum(log_returns))
    high = close * (1 + abs(noise) * rng.rand(n))
    low = close * (1 - abs(noise) * rng.rand(n))
    open_ = close * (1 + noise * 0.5 * rng.randn(n))
    volume = (1_000_000 * (1 + 0.3 * rng.randn(n))).clip(100_000)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def _make_pullback_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate data with a clear uptrend followed by a pullback."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2023-01-01", periods=n, freq="B")

    # Strong uptrend for 150 days, then pullback for 30 days, then recovery
    close = np.zeros(n)
    close[0] = 100.0
    for i in range(1, 150):
        close[i] = close[i - 1] * (1 + 0.003 + 0.005 * rng.randn())
    for i in range(150, 180):
        close[i] = close[i - 1] * (1 - 0.005 + 0.003 * rng.randn())
    for i in range(180, n):
        close[i] = close[i - 1] * (1 + 0.004 + 0.005 * rng.randn())

    high = close * (1 + 0.005 * abs(rng.randn(n)))
    low = close * (1 - 0.005 * abs(rng.randn(n)))
    open_ = close * (1 + 0.002 * rng.randn(n))
    volume = (500_000 * (1 + 0.2 * rng.randn(n))).clip(100_000)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


# ---------------------------------------------------------------------------
# Signal Engine Tests
# ---------------------------------------------------------------------------

class TestSignalEngine:
    def test_compute_indicators_adds_columns(self):
        df = _make_price_df(100)
        result = compute_indicators(df)
        expected_cols = [
            "sma_20", "sma_50", "sma_200", "rsi_14", "bb_upper", "bb_lower",
            "stoch_k", "atr_14", "atr_pct", "volume_sma_20", "obv",
            "macd_hist", "williams_r",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing indicator column: {col}"

    def test_compute_indicators_no_nans_at_end(self):
        df = _make_price_df(200)
        result = compute_indicators(df)
        last_row = result.iloc[-1]
        for col in ["sma_20", "rsi_14", "atr_14", "stoch_k"]:
            assert not np.isnan(last_row[col]), f"NaN in {col} at last row"

    def test_signal_score_range(self):
        engine = SignalEngine()
        df = _make_price_df(200)
        indicator_df = compute_indicators(df)
        for idx in indicator_df.index[-20:]:
            row = indicator_df.loc[idx]
            total, trend, pb, vol, volat = engine._score_row(row)
            assert 0 <= total <= 100
            assert 0 <= trend <= 40
            assert 0 <= pb <= 30
            assert 0 <= vol <= 15
            assert 0 <= volat <= 15

    def test_scan_history_returns_dataframe(self):
        engine = SignalEngine()
        df = _make_price_df(200)
        result = engine.scan_history(df, "TEST")
        assert isinstance(result, pd.DataFrame)
        assert "score" in result.columns
        assert "entry_eligible" in result.columns
        assert len(result) == len(df)

    def test_score_universe(self):
        engine = SignalEngine()
        df = _make_price_df(200)
        indicator_df = compute_indicators(df)
        scores = engine.score_universe({"TEST": indicator_df})
        assert isinstance(scores, list)
        assert len(scores) == 1
        assert scores[0].symbol == "TEST"

    def test_bear_regime_blocks_entry(self):
        engine = SignalEngine()
        df = _make_price_df(200)
        indicator_df = compute_indicators(df)
        # Force a bear regime via spy_prices that are declining
        spy = pd.Series(
            np.linspace(200, 100, 200),
            index=df.index,
        )
        scores = engine.score_universe({"TEST": indicator_df}, spy_prices=spy)
        for s in scores:
            assert not s.entry_eligible or s.regime != "BEAR"


# ---------------------------------------------------------------------------
# Position Sizing Tests
# ---------------------------------------------------------------------------

class TestPositionSizer:
    def test_basic_sizing(self):
        sizer = PositionSizer()
        result = sizer.compute(
            equity=500,
            entry_price=50,
            atr=2.0,
            signal_score=75,
            regime="BULL",
        )
        assert not result.rejected
        assert result.shares >= 1
        assert result.stop_price == 50 - 2.0 * 1.5

    def test_bear_regime_rejected(self):
        sizer = PositionSizer()
        result = sizer.compute(
            equity=500, entry_price=50, atr=2.0,
            signal_score=75, regime="BEAR",
        )
        assert result.rejected
        assert "BEAR" in result.reject_reason

    def test_low_price_rejected(self):
        sizer = PositionSizer()
        result = sizer.compute(
            equity=500, entry_price=3.0, atr=0.5,
            signal_score=75, regime="BULL",
        )
        assert result.rejected
        assert "minimum" in result.reject_reason.lower()

    def test_max_positions_enforced(self):
        sizer = PositionSizer()
        result = sizer.compute(
            equity=500, entry_price=50, atr=2.0,
            signal_score=75, regime="BULL",
            current_positions=3,  # $500-$1000 bracket: max 3
        )
        assert result.rejected

    def test_conviction_scaling(self):
        sizer = PositionSizer()
        high = sizer.compute(equity=1000, entry_price=50, atr=2.0, signal_score=85, regime="BULL")
        low = sizer.compute(equity=1000, entry_price=50, atr=2.0, signal_score=60, regime="BULL")
        # High conviction should get equal or more shares
        assert high.shares >= low.shares

    def test_risk_budget_respects_portfolio_cap(self):
        sizer = PositionSizer()
        result = sizer.compute(
            equity=500, entry_price=50, atr=2.0,
            signal_score=75, regime="BULL",
            current_portfolio_risk_pct=0.03,  # Already at cap
        )
        assert result.rejected
        assert "exhausted" in result.reject_reason.lower()

    def test_max_positions_by_equity(self):
        sizer = PositionSizer()
        assert sizer.max_positions(100) == 1
        assert sizer.max_positions(300) == 2
        assert sizer.max_positions(700) == 3
        assert sizer.max_positions(1500) == 4


# ---------------------------------------------------------------------------
# Exit Engine Tests
# ---------------------------------------------------------------------------

class TestExitEngine:
    def _make_position(self, entry_price: float = 100.0) -> PositionState:
        return PositionState(
            symbol="TEST",
            entry_date=pd.Timestamp("2023-06-01"),
            entry_price=entry_price,
            shares=10,
            stop_price=entry_price - 3.0,
            atr_at_entry=2.0,
        )

    def test_stop_loss_triggers(self):
        engine = ExitEngine()
        pos = self._make_position(100.0)
        sig = engine.check_exit(
            pos, pd.Timestamp("2023-06-05"),
            current_close=96.0, current_high=97.0, current_atr=2.0,
            current_rsi=40, current_sma_50=99.0, regime="BULL",
        )
        assert sig.should_exit
        assert sig.reason == ExitReason.STOP_LOSS

    def test_no_exit_within_range(self):
        engine = ExitEngine()
        pos = self._make_position(100.0)
        sig = engine.check_exit(
            pos, pd.Timestamp("2023-06-05"),
            current_close=101.0, current_high=101.5, current_atr=2.0,
            current_rsi=55, current_sma_50=99.0, regime="BULL",
        )
        assert not sig.should_exit
        assert sig.reason == ExitReason.NONE

    def test_profit_target_triggers(self):
        engine = ExitEngine()
        pos = self._make_position(100.0)
        # target_1 = 100 + 2*2.0 = 104
        sig = engine.check_exit(
            pos, pd.Timestamp("2023-06-10"),
            current_close=105.0, current_high=105.5, current_atr=2.0,
            current_rsi=60, current_sma_50=101.0, regime="BULL",
        )
        assert sig.should_exit
        assert sig.reason == ExitReason.PROFIT_TARGET

    def test_time_exit_triggers(self):
        engine = ExitEngine(max_holding_days=15)
        pos = self._make_position(100.0)
        sig = engine.check_exit(
            pos, pd.Timestamp("2023-06-20"),
            current_close=101.0, current_high=101.5, current_atr=2.0,
            current_rsi=55, current_sma_50=101.0, regime="BULL",
        )
        assert sig.should_exit
        assert sig.reason == ExitReason.TIME_EXIT

    def test_regime_bear_exit(self):
        engine = ExitEngine()
        pos = self._make_position(100.0)
        sig = engine.check_exit(
            pos, pd.Timestamp("2023-06-05"),
            current_close=101.0, current_high=101.5, current_atr=2.0,
            current_rsi=55, current_sma_50=102.0, regime="BEAR",
        )
        assert sig.should_exit
        assert sig.reason == ExitReason.REGIME_BEAR

    def test_rsi_overbought_exit(self):
        engine = ExitEngine()
        pos = self._make_position(100.0)
        sig = engine.check_exit(
            pos, pd.Timestamp("2023-06-05"),
            current_close=103.0, current_high=103.5, current_atr=2.0,
            current_rsi=75, current_sma_50=101.0, regime="BULL",
        )
        assert sig.should_exit
        assert sig.reason == ExitReason.RSI_OVERBOUGHT

    def test_trailing_stop_activates(self):
        engine = ExitEngine()
        pos = self._make_position(100.0)
        # Move price up enough to activate trailing (1 ATR = $2 profit)
        engine.check_exit(
            pos, pd.Timestamp("2023-06-03"),
            current_close=102.5, current_high=103.0, current_atr=2.0,
            current_rsi=55, current_sma_50=101.0, regime="BULL",
        )
        assert pos.trailing_activated
        assert pos.trailing_stop > pos.stop_price


# ---------------------------------------------------------------------------
# Risk Manager Tests
# ---------------------------------------------------------------------------

class TestSwingRiskManager:
    def test_green_drawdown(self):
        mgr = SwingRiskManager()
        mgr.initialize(1000)
        level = mgr.get_drawdown_level(990)
        assert level == DrawdownLevel.GREEN

    def test_yellow_drawdown(self):
        mgr = SwingRiskManager()
        mgr.initialize(1000)
        level = mgr.get_drawdown_level(940)
        assert level == DrawdownLevel.YELLOW

    def test_orange_drawdown(self):
        mgr = SwingRiskManager()
        mgr.initialize(1000)
        level = mgr.get_drawdown_level(890)
        assert level == DrawdownLevel.ORANGE

    def test_red_drawdown(self):
        mgr = SwingRiskManager()
        mgr.initialize(1000)
        level = mgr.get_drawdown_level(840)
        assert level == DrawdownLevel.RED

    def test_consecutive_losses_block_entry(self):
        mgr = SwingRiskManager(max_consecutive_losses=3)
        mgr.initialize(1000)
        mgr.record_trade_result(-10)
        mgr.record_trade_result(-10)
        mgr.record_trade_result(-10)
        state = mgr.get_risk_state(980, 0, 0)
        assert not state.can_open_new

    def test_win_resets_consecutive_losses(self):
        mgr = SwingRiskManager(max_consecutive_losses=3)
        mgr.initialize(1000)
        mgr.record_trade_result(-10)
        mgr.record_trade_result(-10)
        mgr.record_trade_result(20)  # Reset
        state = mgr.get_risk_state(1000, 0, 0)
        assert state.can_open_new

    def test_cooldown_on_red(self):
        mgr = SwingRiskManager(cooldown_days=30)
        mgr.initialize(1000)
        state = mgr.get_risk_state(840, 0, 0)
        assert state.drawdown_level == DrawdownLevel.RED
        assert state.cooldown_remaining_days == 30
        assert not state.can_open_new

    def test_entry_score_threshold(self):
        mgr = SwingRiskManager()
        assert mgr.entry_score_threshold(DrawdownLevel.GREEN) == 60
        assert mgr.entry_score_threshold(DrawdownLevel.YELLOW) == 75
        assert mgr.entry_score_threshold(DrawdownLevel.ORANGE) == 999


# ---------------------------------------------------------------------------
# Edge Decay Monitor Tests
# ---------------------------------------------------------------------------

class TestEdgeDecayMonitor:
    def test_not_enough_trades_returns_green(self):
        mon = EdgeDecayMonitor(min_trades=10)
        for _ in range(5):
            mon.record_trade(10, True)
        m = mon.evaluate()
        assert m.alert_level == AlertLevel.GREEN

    def test_good_trades_green(self):
        mon = EdgeDecayMonitor(min_trades=10)
        for _ in range(20):
            mon.record_trade(10, True)
            mon.record_daily_return(0.002, 1000)
        m = mon.evaluate()
        assert m.alert_level == AlertLevel.GREEN

    def test_poor_win_rate_triggers_yellow(self):
        mon = EdgeDecayMonitor(min_trades=10, win_rate_floor=0.50)
        # 4 wins, 16 losses → 20% win rate
        for _ in range(4):
            mon.record_trade(10, True)
        for _ in range(16):
            mon.record_trade(-5, False)
        for _ in range(20):
            mon.record_daily_return(-0.001, 1000)
        m = mon.evaluate()
        assert m.alert_level >= AlertLevel.YELLOW

    def test_multiple_breaches_orange(self):
        mon = EdgeDecayMonitor(min_trades=10)
        # All losing trades
        for _ in range(20):
            mon.record_trade(-10, False)
            mon.record_daily_return(-0.005, 1000)
        m = mon.evaluate()
        assert m.alert_level >= AlertLevel.ORANGE


# ---------------------------------------------------------------------------
# Strategy Engine Integration Tests
# ---------------------------------------------------------------------------

class TestSwingStrategyEngine:
    def test_engine_runs_without_error(self):
        config = StrategyConfig(initial_capital=1000)
        engine = SwingStrategyEngine(config)
        df = _make_price_df(200)
        result = engine.run({"TEST": df}, spy_prices=df["close"])
        assert len(result.snapshots) > 0
        assert result.equity_curve.iloc[-1] > 0

    def test_engine_summary_keys(self):
        config = StrategyConfig(initial_capital=500)
        engine = SwingStrategyEngine(config)
        df = _make_price_df(200)
        result = engine.run({"TEST": df}, spy_prices=df["close"])
        summary = result.summary()
        expected_keys = [
            "initial_capital", "final_equity", "total_return", "cagr",
            "sharpe_ratio", "max_drawdown", "total_trades", "win_rate",
        ]
        for key in expected_keys:
            assert key in summary, f"Missing summary key: {key}"

    def test_engine_preserves_capital_in_downtrend(self):
        """In a downtrend the engine should stay mostly in cash."""
        config = StrategyConfig(initial_capital=500)
        engine = SwingStrategyEngine(config)
        # Downtrending data
        df = _make_price_df(200, trend=-0.002, seed=99)
        result = engine.run({"TEST": df}, spy_prices=df["close"])
        # Should have very few trades (regime filter should block most)
        assert result.summary()["total_trades"] <= 5

    def test_engine_with_multiple_symbols(self):
        config = StrategyConfig(initial_capital=1000)
        engine = SwingStrategyEngine(config)
        df1 = _make_price_df(200, seed=1)
        df2 = _make_price_df(200, start_price=50, seed=2)
        spy = _make_price_df(200, seed=3)
        result = engine.run(
            {"SYM1": df1, "SYM2": df2},
            spy_prices=spy["close"],
        )
        assert len(result.snapshots) > 0

    def test_trade_record_fields(self):
        config = StrategyConfig(initial_capital=1000)
        engine = SwingStrategyEngine(config)
        df = _make_pullback_df(200)
        result = engine.run({"TEST": df}, spy_prices=df["close"])
        tdf = result.trade_df
        if not tdf.empty:
            assert "symbol" in tdf.columns
            assert "entry_price" in tdf.columns
            assert "exit_reason" in tdf.columns
            assert "pnl_dollars" in tdf.columns
