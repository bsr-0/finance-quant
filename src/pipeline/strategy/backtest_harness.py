"""Modular backtest harness for systematic strategies.

Ingests universe definition, signals, entry/exit rules, sizing model,
and risk parameters to run historical simulations. Computes all standard
performance metrics and supports transaction cost modeling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pipeline.strategy.benchmark import (
    BenchmarkAnalysis,
    BenchmarkSuite,
    compute_all_benchmarks,
)
from pipeline.strategy.edge_decay import EdgeDecayMonitor
from pipeline.strategy.entry_rules import EntryContext, EntryRuleSet
from pipeline.strategy.exits import ExitEngine, ExitReason, PositionState
from pipeline.strategy.position_sizing import (
    PositionSizingModel,
)
from pipeline.strategy.risk_constraints import (
    ConstraintCheckResult,
    RiskConstraintSet,
)
from pipeline.strategy.signal_library import SignalPipeline
from pipeline.strategy.universe import Universe

logger = logging.getLogger(__name__)

_TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Backtest configuration
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    """Configuration for the backtest harness."""

    initial_capital: float = 1e7  # $10M default
    start_date: str = ""
    end_date: str = ""

    # Transaction costs
    spread_bps: float = 3.0
    commission_per_share: float = 0.005
    slippage_bps: float = 2.0

    # Rebalance frequency
    rebalance_frequency: str = "daily"  # daily, weekly, monthly

    # Signal lag (avoid look-ahead bias)
    signal_lag_days: int = 1


# ---------------------------------------------------------------------------
# Backtest result
# ---------------------------------------------------------------------------

@dataclass
class BacktestMetrics:
    """Comprehensive backtest performance metrics."""

    total_return: float = np.nan
    cagr: float = np.nan
    sharpe_ratio: float = np.nan
    sortino_ratio: float = np.nan
    max_drawdown: float = np.nan
    avg_drawdown: float = np.nan
    hit_rate: float = np.nan
    avg_win: float = np.nan
    avg_loss: float = np.nan
    profit_factor: float = np.nan
    total_trades: int = 0
    avg_holding_days: float = np.nan
    annualized_turnover: float = np.nan
    avg_gross_exposure: float = np.nan
    avg_net_exposure: float = np.nan
    total_transaction_costs: float = 0.0
    time_in_market: float = np.nan
    calmar_ratio: float = np.nan
    skewness: float = np.nan
    kurtosis: float = np.nan

    def to_dict(self) -> dict:
        return {
            "Total Return": self.total_return,
            "CAGR": self.cagr,
            "Sharpe Ratio": self.sharpe_ratio,
            "Sortino Ratio": self.sortino_ratio,
            "Max Drawdown": self.max_drawdown,
            "Calmar Ratio": self.calmar_ratio,
            "Hit Rate": self.hit_rate,
            "Avg Winner": self.avg_win,
            "Avg Loser": self.avg_loss,
            "Profit Factor": self.profit_factor,
            "Total Trades": self.total_trades,
            "Avg Holding Days": self.avg_holding_days,
            "Annualized Turnover": self.annualized_turnover,
            "Avg Gross Exposure": self.avg_gross_exposure,
            "Total Costs": self.total_transaction_costs,
            "Time in Market": self.time_in_market,
        }


@dataclass
class HarnessBacktestResult:
    """Complete result from the backtest harness."""

    config: BacktestConfig
    metrics: BacktestMetrics
    equity_curve: pd.Series
    daily_returns: pd.Series
    positions_history: pd.DataFrame
    trade_log: pd.DataFrame
    benchmark_analyses: list[BenchmarkAnalysis] = field(default_factory=list)
    decay_monitor: EdgeDecayMonitor | None = None
    constraint_violations: list[ConstraintCheckResult] = field(default_factory=list)

    def summary_table(self) -> str:
        """Format metrics as a Markdown table."""
        d = self.metrics.to_dict()
        lines = ["| Metric | Value |", "|---|---|"]
        for k, v in d.items():
            if isinstance(v, float):
                if np.isnan(v):
                    val_str = "N/A"
                elif abs(v) < 10 and "Ratio" in k or "Factor" in k or "Exposure" in k:
                    val_str = f"{v:.3f}"
                elif any(
                    s in k for s in ("Return", "Drawdown", "Rate", "Turnover", "Market")
                ):
                    val_str = f"{v:.2%}"
                else:
                    val_str = f"{v:,.2f}"
            elif isinstance(v, int):
                val_str = f"{v:,d}"
            else:
                val_str = str(v)
            lines.append(f"| {k} | {val_str} |")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Backtest harness
# ---------------------------------------------------------------------------

class BacktestHarness:
    """Modular backtest harness for systematic strategies.

    Coordinates the signal pipeline, entry/exit rules, position sizing,
    and risk constraints to run a historical simulation.
    """

    def __init__(
        self,
        signal_pipeline: SignalPipeline,
        entry_rules: EntryRuleSet,
        exit_engine: ExitEngine,
        sizing_model: PositionSizingModel,
        risk_constraints: RiskConstraintSet,
        config: BacktestConfig | None = None,
        benchmark_suite: BenchmarkSuite | None = None,
    ) -> None:
        self.signal_pipeline = signal_pipeline
        self.entry_rules = entry_rules
        self.exit_engine = exit_engine
        self.sizing_model = sizing_model
        self.risk_constraints = risk_constraints
        self.config = config or BacktestConfig()
        self.benchmark_suite = benchmark_suite
        self.decay_monitor = EdgeDecayMonitor()

    def run(
        self,
        price_data: dict[str, pd.DataFrame],
        universe: Universe | None = None,
        benchmark_returns: dict[str, pd.Series] | None = None,
    ) -> HarnessBacktestResult:
        """Run backtest over historical data.

        Args:
            price_data: ``{ticker: DataFrame}`` with OHLCV columns.
            universe: Optional universe filter (if None, uses all tickers).
            benchmark_returns: ``{ticker: daily_returns}`` for benchmarks.

        Returns:
            ``HarnessBacktestResult`` with full metrics and history.
        """
        cfg = self.config

        # Filter universe
        tickers = list(price_data.keys())
        if universe is not None:
            tickers = [t for t in tickers if t in universe.ticker_set]

        if not tickers:
            return self._empty_result()

        # Compute signals
        filtered_data = {t: price_data[t] for t in tickers if not price_data[t].empty}
        composite_signals = self.signal_pipeline.run(filtered_data)

        if composite_signals.empty:
            return self._empty_result()

        # Lag signals to avoid look-ahead bias
        if cfg.signal_lag_days > 0:
            composite_signals = composite_signals.shift(cfg.signal_lag_days)

        # Build date index
        all_dates = composite_signals.index.dropna().sort_values()

        # State
        capital = cfg.initial_capital
        cash = capital
        positions: dict[str, _LivePosition] = {}
        equity_history: list[tuple[pd.Timestamp, float]] = []
        trade_log: list[dict] = []
        daily_returns_list: list[tuple[pd.Timestamp, float]] = []
        positions_history: list[dict] = []
        total_costs = 0.0
        prev_equity = capital
        peak_equity = capital
        gross_exposures: list[float] = []

        for date in all_dates:
            # Current prices
            current_prices: dict[str, float] = {}
            current_vols: dict[str, float] = {}
            for ticker in tickers:
                df = price_data.get(ticker)
                if df is None or df.empty or date not in df.index:
                    continue
                current_prices[ticker] = float(df.loc[date, "close"])
                # Estimate vol from recent returns
                if len(df.loc[:date]) >= 21:
                    recent = df.loc[:date, "close"].pct_change().dropna().tail(60)
                    if len(recent) >= 20:
                        current_vols[ticker] = float(recent.std() * np.sqrt(_TRADING_DAYS))

            # Compute equity
            pos_value = sum(
                current_prices.get(t, p.entry_price) * p.shares
                for t, p in positions.items()
            )
            equity = cash + pos_value

            # Update peak and drawdown
            peak_equity = max(peak_equity, equity)
            drawdown = (equity - peak_equity) / peak_equity if peak_equity > 0 else 0

            # Daily return
            daily_ret = (equity - prev_equity) / prev_equity if prev_equity > 0 else 0
            daily_returns_list.append((date, daily_ret))
            self.decay_monitor.record_daily_return(daily_ret, equity)

            # --- Check exits ---
            closed_tickers: list[str] = []
            for ticker, pos in list(positions.items()):
                df = price_data.get(ticker)
                if df is None or date not in df.index:
                    continue
                row = df.loc[date]
                pos_state = PositionState(
                    symbol=ticker,
                    entry_date=pos.entry_date,
                    entry_price=pos.entry_price,
                    shares=pos.shares,
                    stop_price=pos.stop_price,
                    atr_at_entry=pos.atr_at_entry,
                    trailing_stop=pos.trailing_stop,
                    trailing_activated=pos.trailing_activated,
                    highest_price=pos.highest_price,
                )
                exit_sig = self.exit_engine.check_exit(
                    position=pos_state,
                    current_date=date,
                    current_close=float(row["close"]),
                    current_high=float(row["high"]),
                    current_atr=(
                        current_vols.get(ticker, pos.atr_at_entry)
                        / np.sqrt(_TRADING_DAYS) * float(row["close"])
                        if current_vols.get(ticker)
                        else pos.atr_at_entry
                    ),
                    current_rsi=50.0,
                    current_sma_50=float(row["close"]),
                    regime="BULL",
                )
                # Sync trailing stop state back
                pos.trailing_stop = pos_state.trailing_stop
                pos.trailing_activated = pos_state.trailing_activated
                pos.highest_price = pos_state.highest_price

                if exit_sig.should_exit:
                    close_px = float(row["close"])
                    notional = close_px * pos.shares
                    cost = notional * (cfg.spread_bps / 10_000 / 2 + cfg.slippage_bps / 10_000)
                    pnl = (close_px - pos.entry_price) * pos.shares - cost
                    cash += notional - cost
                    total_costs += cost

                    trade_log.append({
                        "ticker": ticker,
                        "entry_date": pos.entry_date,
                        "entry_price": pos.entry_price,
                        "shares": pos.shares,
                        "exit_date": date,
                        "exit_price": close_px,
                        "exit_reason": exit_sig.reason.value,
                        "pnl": pnl,
                        "pnl_pct": (close_px - pos.entry_price) / pos.entry_price,
                        "days_held": (date - pos.entry_date).days,
                        "cost": cost,
                    })
                    self.decay_monitor.record_trade(
                        pnl, exit_sig.reason == ExitReason.PROFIT_TARGET,
                    )
                    closed_tickers.append(ticker)

            for t in closed_tickers:
                del positions[t]

            # --- Check entries ---
            if date in composite_signals.index:
                signals_today = composite_signals.loc[date].dropna()

                # Sort by signal strength (descending)
                signals_today = signals_today.sort_values(ascending=False)

                for ticker in signals_today.index:
                    if ticker in positions:
                        continue
                    sig_val = float(signals_today[ticker])

                    price = current_prices.get(ticker, 0.0)
                    if price <= 0:
                        continue

                    vol = current_vols.get(ticker, 0.20)
                    atr_est = vol / np.sqrt(_TRADING_DAYS) * price

                    # Build entry context
                    ctx = EntryContext(
                        portfolio_equity=equity,
                        available_cash=cash,
                        open_position_count=len(positions),
                        max_positions=50,
                        held_tickers=set(positions.keys()),
                        regime="BULL",
                    )

                    decision = self.entry_rules.evaluate(
                        ticker, date, sig_val, ctx,
                    )
                    if not decision.eligible:
                        continue

                    # Position sizing
                    sig_series = pd.Series({ticker: sig_val})
                    price_series = pd.Series(current_prices)
                    vol_series = pd.Series(current_vols)

                    targets = self.sizing_model.compute_targets(
                        signals=sig_series,
                        prices=price_series,
                        volatilities=vol_series,
                        capital=equity,
                    )

                    if not targets.positions:
                        continue

                    target = targets.positions[0]
                    if target.target_shares <= 0:
                        continue

                    # Cost check
                    entry_cost = abs(target.target_notional) * (
                        cfg.spread_bps / 10_000 / 2 + cfg.slippage_bps / 10_000
                    )
                    if cash < abs(target.target_notional) + entry_cost:
                        continue

                    # Open position
                    stop_price = price - atr_est * 1.5
                    cash -= abs(target.target_notional) + entry_cost
                    total_costs += entry_cost

                    positions[ticker] = _LivePosition(
                        entry_date=date,
                        entry_price=price,
                        shares=abs(target.target_shares),
                        stop_price=stop_price,
                        atr_at_entry=atr_est,
                    )

            # Record snapshot
            pos_value = sum(
                current_prices.get(t, p.entry_price) * p.shares
                for t, p in positions.items()
            )
            equity = cash + pos_value
            equity_history.append((date, equity))

            gross_exp = pos_value / equity if equity > 0 else 0
            gross_exposures.append(gross_exp)

            positions_history.append({
                "date": date,
                "equity": equity,
                "cash": cash,
                "positions_value": pos_value,
                "num_positions": len(positions),
                "gross_exposure": gross_exp,
                "drawdown": drawdown,
            })

            prev_equity = equity

        # Build result DataFrames
        eq_series = pd.Series(
            dict(equity_history), dtype=float,
        ).sort_index() if equity_history else pd.Series(dtype=float)

        ret_series = pd.Series(
            dict(daily_returns_list), dtype=float,
        ).sort_index() if daily_returns_list else pd.Series(dtype=float)

        trade_df = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()
        pos_df = pd.DataFrame(positions_history) if positions_history else pd.DataFrame()

        # Compute metrics
        metrics = self._compute_metrics(
            eq_series, ret_series, trade_df, cfg.initial_capital,
            total_costs, gross_exposures,
        )

        # Benchmark analysis
        bm_analyses: list[BenchmarkAnalysis] = []
        if self.benchmark_suite and benchmark_returns and not ret_series.empty:
            bm_analyses = compute_all_benchmarks(
                ret_series, benchmark_returns, self.benchmark_suite,
            )

        return HarnessBacktestResult(
            config=cfg,
            metrics=metrics,
            equity_curve=eq_series,
            daily_returns=ret_series,
            positions_history=pos_df,
            trade_log=trade_df,
            benchmark_analyses=bm_analyses,
            decay_monitor=self.decay_monitor,
        )

    def _compute_metrics(
        self,
        equity: pd.Series,
        returns: pd.Series,
        trades: pd.DataFrame,
        initial_capital: float,
        total_costs: float,
        gross_exposures: list[float],
    ) -> BacktestMetrics:
        if equity.empty:
            return BacktestMetrics()

        total_return = (equity.iloc[-1] - initial_capital) / initial_capital

        # CAGR
        years = (equity.index[-1] - equity.index[0]).days / 365.25 if len(equity) > 1 else 0
        if years > 0 and equity.iloc[-1] > 0:
            cagr = (equity.iloc[-1] / initial_capital) ** (1 / years) - 1
        else:
            cagr = np.nan

        # Sharpe
        if len(returns) > 20 and returns.std() > 0:
            sharpe = float(returns.mean() / returns.std() * np.sqrt(_TRADING_DAYS))
        else:
            sharpe = np.nan

        # Sortino
        downside = returns[returns < 0]
        if len(downside) > 0:
            ds_std = float(np.sqrt((downside ** 2).mean()))
            sortino = (
                float(returns.mean() / ds_std * np.sqrt(_TRADING_DAYS))
                if ds_std > 0 else np.nan
            )
        else:
            sortino = np.nan

        # Drawdown
        peak = equity.cummax()
        dd = (equity - peak) / peak.replace(0, np.nan)
        max_dd = float(dd.min())
        avg_dd = float(dd.mean())

        # Calmar
        calmar = cagr / abs(max_dd) if not np.isnan(cagr) and max_dd < 0 else np.nan

        # Trade stats
        if not trades.empty and "pnl" in trades.columns:
            winners = trades[trades["pnl"] > 0]
            losers = trades[trades["pnl"] <= 0]
            hit_rate = len(winners) / len(trades) if len(trades) > 0 else np.nan
            avg_win = float(winners["pnl"].mean()) if len(winners) > 0 else 0.0
            avg_loss = float(losers["pnl"].mean()) if len(losers) > 0 else 0.0
            pf_num = winners["pnl"].sum() if len(winners) > 0 else 0
            pf_den = abs(losers["pnl"].sum()) if len(losers) > 0 else 0
            profit_factor = pf_num / pf_den if pf_den > 0 else float("inf")
            avg_hold = (
                float(trades["days_held"].mean())
                if "days_held" in trades.columns else np.nan
            )
        else:
            hit_rate = avg_win = avg_loss = avg_hold = np.nan
            profit_factor = np.nan

        # Exposure
        avg_gross = float(np.mean(gross_exposures)) if gross_exposures else 0

        # Higher moments
        skew = float(returns.skew()) if len(returns) > 20 else np.nan
        kurt = float(returns.kurtosis()) if len(returns) > 20 else np.nan

        return BacktestMetrics(
            total_return=total_return,
            cagr=cagr,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            avg_drawdown=avg_dd,
            hit_rate=hit_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            total_trades=len(trades),
            avg_holding_days=avg_hold,
            avg_gross_exposure=avg_gross,
            total_transaction_costs=total_costs,
            time_in_market=avg_gross,
            calmar_ratio=calmar,
            skewness=skew,
            kurtosis=kurt,
        )

    def _empty_result(self) -> HarnessBacktestResult:
        return HarnessBacktestResult(
            config=self.config,
            metrics=BacktestMetrics(),
            equity_curve=pd.Series(dtype=float),
            daily_returns=pd.Series(dtype=float),
            positions_history=pd.DataFrame(),
            trade_log=pd.DataFrame(),
        )


@dataclass
class _LivePosition:
    """Internal mutable position state during backtest."""

    entry_date: pd.Timestamp
    entry_price: float
    shares: int
    stop_price: float
    atr_at_entry: float
    trailing_stop: float = 0.0
    trailing_activated: bool = False
    highest_price: float = 0.0

    def __post_init__(self) -> None:
        self.highest_price = self.entry_price
        self.trailing_stop = self.stop_price
