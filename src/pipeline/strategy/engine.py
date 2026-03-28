"""Main strategy engine: orchestrates signals, sizing, exits, and risk.

This module brings together all strategy components into a single backtest-
capable engine that processes daily OHLCV data and produces a full equity
curve with trade-level detail.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pipeline.eval.regime import classify_regimes
from pipeline.strategy.edge_decay import EdgeDecayMonitor
from pipeline.strategy.exits import ExitEngine, ExitReason, PositionState
from pipeline.strategy.risk import DrawdownLevel, SwingRiskManager
from pipeline.strategy.signals import SignalEngine, compute_indicators
from pipeline.strategy.sizing import PositionSizer, SizeResult, SizingConfig

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Complete record of a single round-trip trade."""

    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: int
    exit_date: pd.Timestamp | None = None
    exit_price: float = 0.0
    exit_reason: str = ""
    pnl_dollars: float = 0.0
    pnl_pct: float = 0.0
    days_held: int = 0
    signal_score: int = 0
    hit_target: bool = False


@dataclass
class DailySnapshot:
    """Portfolio state at the end of each trading day."""

    date: pd.Timestamp
    equity: float
    cash: float
    positions_value: float
    num_positions: int
    daily_return: float
    drawdown_pct: float
    drawdown_level: int
    regime: str


@dataclass
class StrategyConfig:
    """Top-level configuration for the swing strategy engine."""

    initial_capital: float = 500.0

    # Signal thresholds
    entry_threshold: int = 60
    neutral_threshold: int = 70

    # ATR / volatility filters
    atr_pct_min: float = 0.5
    atr_pct_max: float = 4.0

    # Sizing
    sizing: SizingConfig = field(default_factory=SizingConfig)

    # Exit parameters
    max_holding_days: int = 15
    stop_atr_multiple: float = 1.5
    trailing_atr_multiple: float = 2.0
    target_atr_multiple: float = 2.0
    rsi_overbought: float = 70.0

    # Risk management
    yellow_dd: float = 0.05
    orange_dd: float = 0.10
    red_dd: float = 0.15
    max_consecutive_losses: int = 4
    cooldown_days: int = 30
    max_correlation: float = 0.70

    # Commission-free broker, but model spread
    spread_bps: float = 3.0


class SwingStrategyEngine:
    """End-to-end backtest engine for the pullback-reversion strategy.

    Usage::

        engine = SwingStrategyEngine(StrategyConfig(initial_capital=500))
        result = engine.run(
            price_data={"SPY": spy_df, "QQQ": qqq_df},
            spy_prices=spy_df["close"],
        )
        print(result.summary())
    """

    def __init__(self, config: StrategyConfig | None = None) -> None:
        self.config = config or StrategyConfig()
        c = self.config

        self.signal_engine = SignalEngine(
            entry_threshold=c.entry_threshold,
            neutral_threshold=c.neutral_threshold,
            atr_pct_min=c.atr_pct_min,
            atr_pct_max=c.atr_pct_max,
        )
        self.sizer = PositionSizer(c.sizing)
        self.exit_engine = ExitEngine(
            max_holding_days=c.max_holding_days,
            stop_atr_multiple=c.stop_atr_multiple,
            trailing_atr_multiple=c.trailing_atr_multiple,
            target_atr_multiple=c.target_atr_multiple,
            rsi_overbought=c.rsi_overbought,
        )
        self.risk_mgr = SwingRiskManager(
            yellow_threshold=c.yellow_dd,
            orange_threshold=c.orange_dd,
            red_threshold=c.red_dd,
            max_consecutive_losses=c.max_consecutive_losses,
            cooldown_days=c.cooldown_days,
            max_correlation=c.max_correlation,
        )
        self.decay_monitor = EdgeDecayMonitor()

    def run(
        self,
        price_data: dict[str, pd.DataFrame],
        spy_prices: pd.Series | None = None,
    ) -> BacktestResult:
        """Run the strategy over historical data.

        Args:
            price_data: ``{symbol: DataFrame}`` with columns
                ``[open, high, low, close, volume]`` indexed by date.
            spy_prices: SPY close prices for regime classification.
                If None, uses ``price_data["SPY"]["close"]`` if available.

        Returns:
            ``BacktestResult`` with trades, equity curve, and performance.
        """
        cfg = self.config

        # Resolve SPY prices for regime
        if spy_prices is None and "SPY" in price_data:
            spy_prices = price_data["SPY"]["close"]

        # Pre-compute indicators for all symbols
        indicator_data: dict[str, pd.DataFrame] = {}
        for sym, df in price_data.items():
            if df.empty:
                continue
            indicator_data[sym] = compute_indicators(df)

        # Build a unified date index
        all_dates: set[pd.Timestamp] = set()
        for df in indicator_data.values():
            all_dates.update(df.index)
        dates = sorted(all_dates)

        # Regime series
        if spy_prices is not None and len(spy_prices) >= 50:
            regimes = classify_regimes(spy_prices)
        else:
            regimes = pd.Series("bull", index=pd.DatetimeIndex(dates))

        # State
        cash = cfg.initial_capital
        self.risk_mgr.initialize(cfg.initial_capital)
        open_positions: list[PositionState] = []
        trades: list[TradeRecord] = []
        snapshots: list[DailySnapshot] = []
        active_trades: dict[str, TradeRecord] = {}  # symbol → open TradeRecord

        prev_equity = cfg.initial_capital

        for date in dates:
            # --- Compute current equity ---
            positions_value = 0.0
            for pos in open_positions:
                sym_df = indicator_data.get(pos.symbol)
                if sym_df is not None and date in sym_df.index:
                    positions_value += sym_df.loc[date, "close"] * pos.shares

            equity = cash + positions_value
            daily_ret = (equity - prev_equity) / prev_equity if prev_equity > 0 else 0
            self.decay_monitor.record_daily_return(daily_ret, equity)

            # --- Risk state ---
            total_risk_pct = sum(
                (pos.entry_price - pos.stop_price) * pos.shares / equity
                for pos in open_positions
            ) if equity > 0 else 0
            risk_state = self.risk_mgr.get_risk_state(
                equity, len(open_positions), total_risk_pct,
                daily_return=daily_ret,
            )
            self.risk_mgr.tick_cooldown()

            # Regime
            regime_val = regimes.get(date, "bull") if date in regimes.index else "bull"
            regime = regime_val.upper() if isinstance(regime_val, str) else "BULL"
            if regime == "FLAT":
                regime = "NEUTRAL"

            # --- Check exits on existing positions ---
            if risk_state.drawdown_level >= DrawdownLevel.RED:
                # Close everything
                for pos in list(open_positions):
                    sym_df = indicator_data.get(pos.symbol)
                    if sym_df is None or date not in sym_df.index:
                        continue
                    close_px = sym_df.loc[date, "close"]
                    pnl = (close_px - pos.entry_price) * pos.shares
                    cash += close_px * pos.shares
                    cost = abs(close_px * pos.shares) * (cfg.spread_bps / 10_000 / 2)
                    cash -= cost

                    tr = active_trades.pop(pos.symbol, None)
                    if tr:
                        tr.exit_date = date
                        tr.exit_price = close_px
                        tr.exit_reason = "RED_CIRCUIT_BREAKER"
                        tr.pnl_dollars = pnl - cost
                        tr.pnl_pct = (close_px - pos.entry_price) / pos.entry_price
                        tr.days_held = (date - pos.entry_date).days
                        trades.append(tr)
                    self.risk_mgr.record_trade_result(pnl - cost)
                    self.decay_monitor.record_trade(pnl - cost, False)

                open_positions.clear()
            else:
                closed_symbols: list[str] = []
                for pos in open_positions:
                    sym_df = indicator_data.get(pos.symbol)
                    if sym_df is None or date not in sym_df.index:
                        continue
                    row = sym_df.loc[date]

                    exit_sig = self.exit_engine.check_exit(
                        position=pos,
                        current_date=date,
                        current_close=row["close"],
                        current_high=row["high"],
                        current_atr=row.get("atr_14", pos.atr_at_entry),
                        current_rsi=row.get("rsi_14", 50),
                        current_sma_50=row.get("sma_50", np.nan),
                        regime=regime,
                    )

                    if exit_sig.should_exit:
                        close_px = row["close"]
                        pnl = (close_px - pos.entry_price) * pos.shares
                        cash += close_px * pos.shares
                        cost = abs(close_px * pos.shares) * (cfg.spread_bps / 10_000 / 2)
                        cash -= cost

                        tr = active_trades.pop(pos.symbol, None)
                        if tr:
                            tr.exit_date = date
                            tr.exit_price = close_px
                            tr.exit_reason = exit_sig.reason.value
                            tr.pnl_dollars = pnl - cost
                            tr.pnl_pct = (close_px - pos.entry_price) / pos.entry_price
                            tr.days_held = exit_sig.days_held
                            tr.hit_target = exit_sig.reason == ExitReason.PROFIT_TARGET
                            trades.append(tr)

                        self.risk_mgr.record_trade_result(pnl - cost)
                        self.decay_monitor.record_trade(pnl - cost, tr.hit_target if tr else False)
                        closed_symbols.append(pos.symbol)

                open_positions = [p for p in open_positions if p.symbol not in closed_symbols]

            # --- Check for new entries ---
            if risk_state.can_open_new and regime != "BEAR":
                dd_level = risk_state.drawdown_level
                min_score = self.risk_mgr.entry_score_threshold(dd_level)
                size_mult = self.risk_mgr.position_size_multiplier(dd_level)
                held_symbols = {p.symbol for p in open_positions}

                for sym, sym_df in indicator_data.items():
                    if sym in held_symbols:
                        continue
                    if date not in sym_df.index:
                        continue

                    row = sym_df.loc[date]
                    score_total, trend, pb, vol_pts, volat_pts = self.signal_engine._score_row(row)

                    # Check entry conditions
                    eligible = (
                        trend >= 25
                        and pb > 0
                        and score_total >= min_score
                    )
                    if not eligible:
                        continue

                    entry_price = row["close"]
                    atr = row.get("atr_14", 0)
                    if atr <= 0:
                        continue

                    # Recompute risk budget
                    current_risk = sum(
                        (p.entry_price - p.stop_price) * p.shares / equity
                        for p in open_positions
                    ) if equity > 0 else 0

                    size_result = self.sizer.compute(
                        equity=equity,
                        entry_price=entry_price,
                        atr=atr,
                        signal_score=score_total,
                        regime=regime,
                        current_positions=len(open_positions),
                        current_portfolio_risk_pct=current_risk,
                    )

                    if size_result.rejected:
                        continue

                    # Apply drawdown multiplier
                    if size_mult < 1.0:
                        adjusted_shares = max(1, int(size_result.shares * size_mult))
                        size_result = SizeResult(
                            shares=adjusted_shares,
                            position_value=adjusted_shares * entry_price,
                            risk_per_share=size_result.risk_per_share,
                            total_risk=adjusted_shares * size_result.risk_per_share,
                            risk_pct_of_equity=(
                                adjusted_shares * size_result.risk_per_share / equity
                            ),
                            stop_price=size_result.stop_price,
                        )

                    # Check we have enough cash
                    purchase_cost = size_result.position_value
                    spread_cost = purchase_cost * (cfg.spread_bps / 10_000 / 2)
                    if cash < purchase_cost + spread_cost:
                        continue

                    # Open position
                    cash -= purchase_cost + spread_cost
                    pos = PositionState(
                        symbol=sym,
                        entry_date=date,
                        entry_price=entry_price,
                        shares=size_result.shares,
                        stop_price=size_result.stop_price,
                        atr_at_entry=atr,
                    )
                    open_positions.append(pos)
                    active_trades[sym] = TradeRecord(
                        symbol=sym,
                        entry_date=date,
                        entry_price=entry_price,
                        shares=size_result.shares,
                        signal_score=score_total,
                    )
                    logger.info(
                        "ENTRY %s: %d shares @ $%.2f, score=%d, stop=$%.2f",
                        sym, size_result.shares, entry_price, score_total,
                        size_result.stop_price,
                    )

            # --- Daily snapshot ---
            positions_value = sum(
                indicator_data[p.symbol].loc[date, "close"] * p.shares
                for p in open_positions
                if p.symbol in indicator_data and date in indicator_data[p.symbol].index
            )
            equity = cash + positions_value
            daily_ret = (equity - prev_equity) / prev_equity if prev_equity > 0 else 0
            if self.risk_mgr._peak_equity > 0:
                dd_pct = (equity - self.risk_mgr._peak_equity) / self.risk_mgr._peak_equity
            else:
                dd_pct = 0

            snapshots.append(DailySnapshot(
                date=date,
                equity=equity,
                cash=cash,
                positions_value=positions_value,
                num_positions=len(open_positions),
                daily_return=daily_ret,
                drawdown_pct=dd_pct,
                drawdown_level=risk_state.drawdown_level,
                regime=regime,
            ))

            prev_equity = equity

        return BacktestResult(
            config=cfg,
            trades=trades,
            snapshots=snapshots,
            decay_monitor=self.decay_monitor,
        )


@dataclass
class BacktestResult:
    """Complete result of a strategy backtest."""

    config: StrategyConfig
    trades: list[TradeRecord]
    snapshots: list[DailySnapshot]
    decay_monitor: EdgeDecayMonitor

    @property
    def equity_curve(self) -> pd.Series:
        if not self.snapshots:
            return pd.Series(dtype=float)
        return pd.Series(
            {s.date: s.equity for s in self.snapshots}
        ).sort_index()

    @property
    def daily_returns(self) -> pd.Series:
        if not self.snapshots:
            return pd.Series(dtype=float)
        return pd.Series(
            {s.date: s.daily_return for s in self.snapshots}
        ).sort_index()

    @property
    def trade_df(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        records = []
        for t in self.trades:
            records.append({
                "symbol": t.symbol,
                "entry_date": t.entry_date,
                "entry_price": t.entry_price,
                "shares": t.shares,
                "exit_date": t.exit_date,
                "exit_price": t.exit_price,
                "exit_reason": t.exit_reason,
                "pnl_dollars": t.pnl_dollars,
                "pnl_pct": t.pnl_pct,
                "days_held": t.days_held,
                "signal_score": t.signal_score,
                "hit_target": t.hit_target,
            })
        return pd.DataFrame(records)

    def summary(self) -> dict:
        """Compute summary performance statistics."""
        eq = self.equity_curve
        if eq.empty:
            return {}

        rets = self.daily_returns
        total_return = (eq.iloc[-1] - self.config.initial_capital) / self.config.initial_capital

        # Sharpe
        if len(rets) > 20 and rets.std() > 0:
            sharpe = rets.mean() / rets.std() * np.sqrt(252)
        else:
            sharpe = np.nan

        # Sortino
        downside = rets[rets < 0]
        if len(downside) > 0:
            ds_std = np.sqrt((downside ** 2).mean())
            sortino = rets.mean() / ds_std * np.sqrt(252) if ds_std > 0 else np.nan
        else:
            sortino = np.nan

        # Max drawdown
        peak = eq.cummax()
        dd = (eq - peak) / peak.replace(0, np.nan)
        max_dd = dd.min()

        # Trade stats
        tdf = self.trade_df
        if not tdf.empty:
            winners = tdf[tdf["pnl_dollars"] > 0]
            losers = tdf[tdf["pnl_dollars"] <= 0]
            win_rate = len(winners) / len(tdf)
            avg_win = winners["pnl_dollars"].mean() if len(winners) > 0 else 0
            avg_loss = losers["pnl_dollars"].mean() if len(losers) > 0 else 0
            profit_factor = (
                winners["pnl_dollars"].sum() / abs(losers["pnl_dollars"].sum())
                if len(losers) > 0 and losers["pnl_dollars"].sum() != 0
                else float("inf")
            )
            avg_hold = tdf["days_held"].mean()
        else:
            win_rate = avg_win = avg_loss = profit_factor = avg_hold = np.nan

        # Time in market
        if self.snapshots:
            in_market = sum(1 for s in self.snapshots if s.num_positions > 0)
            time_in_market = in_market / len(self.snapshots)
        else:
            time_in_market = 0

        # CAGR
        if len(eq) > 1:
            years = (eq.index[-1] - eq.index[0]).days / 365.25
            if years > 0 and eq.iloc[-1] > 0:
                cagr = (eq.iloc[-1] / self.config.initial_capital) ** (1 / years) - 1
            else:
                cagr = np.nan
        else:
            cagr = np.nan

        return {
            "initial_capital": self.config.initial_capital,
            "final_equity": eq.iloc[-1],
            "total_return": total_return,
            "cagr": cagr,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "total_trades": len(tdf),
            "win_rate": win_rate,
            "avg_winner": avg_win,
            "avg_loser": avg_loss,
            "profit_factor": profit_factor,
            "avg_holding_days": avg_hold,
            "time_in_market": time_in_market,
        }

    def print_summary(self) -> None:
        """Print a formatted performance summary."""
        s = self.summary()
        if not s:
            print("No data to summarize.")
            return

        print("=" * 60)
        print("  QSG-MICRO-SWING-001 — Backtest Summary")
        print("=" * 60)
        print(f"  Initial Capital:    ${s['initial_capital']:>10,.2f}")
        print(f"  Final Equity:       ${s['final_equity']:>10,.2f}")
        print(f"  Total Return:        {s['total_return']:>10.2%}")
        print(f"  CAGR:                {s['cagr']:>10.2%}")
        print("-" * 60)
        print(f"  Sharpe Ratio:        {s['sharpe_ratio']:>10.2f}")
        print(f"  Sortino Ratio:       {s['sortino_ratio']:>10.2f}")
        print(f"  Max Drawdown:        {s['max_drawdown']:>10.2%}")
        print("-" * 60)
        print(f"  Total Trades:        {s['total_trades']:>10d}")
        print(f"  Win Rate:            {s['win_rate']:>10.2%}")
        print(f"  Avg Winner:          ${s['avg_winner']:>10.2f}")
        print(f"  Avg Loser:           ${s['avg_loser']:>10.2f}")
        print(f"  Profit Factor:       {s['profit_factor']:>10.2f}")
        print(f"  Avg Holding (days):  {s['avg_holding_days']:>10.1f}")
        print(f"  Time in Market:      {s['time_in_market']:>10.2%}")
        print("=" * 60)
