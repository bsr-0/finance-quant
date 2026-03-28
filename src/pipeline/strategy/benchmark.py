"""Benchmark configuration and relative performance analysis.

Provides benchmark selection, computation of relative metrics
(active return, tracking error, information ratio), and rendering
for the strategy memo.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for a single benchmark."""

    name: str
    ticker: str
    description: str = ""
    justification: str = ""


@dataclass
class BenchmarkSuite:
    """Complete benchmark configuration for a strategy."""

    primary: BenchmarkConfig
    secondary: list[BenchmarkConfig] = field(default_factory=list)

    @property
    def all_benchmarks(self) -> list[BenchmarkConfig]:
        return [self.primary] + self.secondary

    @property
    def all_tickers(self) -> list[str]:
        return [b.ticker for b in self.all_benchmarks]


# ---------------------------------------------------------------------------
# Relative performance metrics
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkAnalysis:
    """Complete relative performance analysis vs a benchmark."""

    benchmark_name: str
    benchmark_ticker: str
    active_return_ann: float = np.nan
    tracking_error_ann: float = np.nan
    information_ratio: float = np.nan
    beta: float = np.nan
    alpha_ann: float = np.nan
    up_capture: float = np.nan
    down_capture: float = np.nan
    correlation: float = np.nan
    strategy_sharpe: float = np.nan
    benchmark_sharpe: float = np.nan
    strategy_total_return: float = np.nan
    benchmark_total_return: float = np.nan
    strategy_max_dd: float = np.nan
    benchmark_max_dd: float = np.nan


def compute_benchmark_analysis(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    benchmark_name: str = "",
    benchmark_ticker: str = "",
    risk_free_rate: float = 0.0,
) -> BenchmarkAnalysis:
    """Compute comprehensive relative performance metrics.

    Args:
        strategy_returns: Daily strategy returns.
        benchmark_returns: Daily benchmark returns.
        benchmark_name: Human-readable benchmark name.
        benchmark_ticker: Benchmark ticker symbol.
        risk_free_rate: Annualized risk-free rate.
    """
    s_ret, b_ret = strategy_returns.align(benchmark_returns, join="inner")
    s_ret = s_ret.dropna()
    b_ret = b_ret.loc[s_ret.index].dropna()
    s_ret = s_ret.loc[b_ret.index]

    if len(s_ret) < 20:
        return BenchmarkAnalysis(
            benchmark_name=benchmark_name,
            benchmark_ticker=benchmark_ticker,
        )

    # Active return
    active = s_ret - b_ret
    active_ann = float(active.mean() * _TRADING_DAYS)
    te_ann = float(active.std() * np.sqrt(_TRADING_DAYS))
    ir = active_ann / te_ann if te_ann > 0 else np.nan

    # Beta and alpha
    cov = np.cov(s_ret.values, b_ret.values)
    beta = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] > 0 else np.nan
    alpha = (
        float(s_ret.mean() - beta * b_ret.mean()) * _TRADING_DAYS if not np.isnan(beta) else np.nan
    )

    # Capture ratios
    up_months = b_ret > 0
    down_months = b_ret < 0
    up_capture = np.nan
    down_capture = np.nan
    if up_months.sum() > 0:
        up_capture = float(s_ret[up_months].mean() / b_ret[up_months].mean())
    if down_months.sum() > 0:
        bm_down_mean = b_ret[down_months].mean()
        if bm_down_mean != 0:
            down_capture = float(s_ret[down_months].mean() / bm_down_mean)

    # Correlation
    corr = float(s_ret.corr(b_ret))

    # Sharpe ratios
    rf_daily = risk_free_rate / _TRADING_DAYS
    s_excess = s_ret - rf_daily
    b_excess = b_ret - rf_daily
    s_sharpe = (
        float(s_excess.mean() / s_excess.std() * np.sqrt(_TRADING_DAYS))
        if s_excess.std() > 0
        else np.nan
    )
    b_sharpe = (
        float(b_excess.mean() / b_excess.std() * np.sqrt(_TRADING_DAYS))
        if b_excess.std() > 0
        else np.nan
    )

    # Total returns
    s_cum = (1 + s_ret).cumprod()
    b_cum = (1 + b_ret).cumprod()
    s_total = float(s_cum.iloc[-1] - 1)
    b_total = float(b_cum.iloc[-1] - 1)

    # Max drawdowns
    s_peak = s_cum.cummax()
    b_peak = b_cum.cummax()
    s_dd = float(((s_cum - s_peak) / s_peak).min())
    b_dd = float(((b_cum - b_peak) / b_peak).min())

    return BenchmarkAnalysis(
        benchmark_name=benchmark_name,
        benchmark_ticker=benchmark_ticker,
        active_return_ann=active_ann,
        tracking_error_ann=te_ann,
        information_ratio=ir,
        beta=beta,
        alpha_ann=alpha,
        up_capture=up_capture,
        down_capture=down_capture,
        correlation=corr,
        strategy_sharpe=s_sharpe,
        benchmark_sharpe=b_sharpe,
        strategy_total_return=s_total,
        benchmark_total_return=b_total,
        strategy_max_dd=s_dd,
        benchmark_max_dd=b_dd,
    )


def compute_all_benchmarks(
    strategy_returns: pd.Series,
    benchmark_data: dict[str, pd.Series],
    suite: BenchmarkSuite,
    risk_free_rate: float = 0.0,
) -> list[BenchmarkAnalysis]:
    """Compute relative metrics against all benchmarks in the suite."""
    results: list[BenchmarkAnalysis] = []
    for bm in suite.all_benchmarks:
        if bm.ticker not in benchmark_data:
            logger.warning("Benchmark %s (%s) not in data", bm.name, bm.ticker)
            continue
        analysis = compute_benchmark_analysis(
            strategy_returns,
            benchmark_data[bm.ticker],
            benchmark_name=bm.name,
            benchmark_ticker=bm.ticker,
            risk_free_rate=risk_free_rate,
        )
        results.append(analysis)
    return results


def benchmark_analysis_to_markdown(analyses: list[BenchmarkAnalysis]) -> str:
    """Render benchmark analyses as a Markdown table."""
    if not analyses:
        return "No benchmark data available."

    lines = [
        "| Metric | " + " | ".join(a.benchmark_name for a in analyses) + " |",
        "|---" + "|---" * len(analyses) + "|",
    ]

    def fmt(val: float, pct: bool = False) -> str:
        if np.isnan(val):
            return "N/A"
        return f"{val:.2%}" if pct else f"{val:.3f}"

    rows = [
        ("Active Return (ann.)", [fmt(a.active_return_ann, True) for a in analyses]),
        ("Tracking Error (ann.)", [fmt(a.tracking_error_ann, True) for a in analyses]),
        ("Information Ratio", [fmt(a.information_ratio) for a in analyses]),
        ("Beta", [fmt(a.beta) for a in analyses]),
        ("Alpha (ann.)", [fmt(a.alpha_ann, True) for a in analyses]),
        ("Up Capture", [fmt(a.up_capture, True) for a in analyses]),
        ("Down Capture", [fmt(a.down_capture, True) for a in analyses]),
        ("Correlation", [fmt(a.correlation) for a in analyses]),
        ("Strategy Sharpe", [fmt(a.strategy_sharpe) for a in analyses]),
        ("Benchmark Sharpe", [fmt(a.benchmark_sharpe) for a in analyses]),
        ("Strategy Total Return", [fmt(a.strategy_total_return, True) for a in analyses]),
        ("Benchmark Total Return", [fmt(a.benchmark_total_return, True) for a in analyses]),
        ("Strategy Max DD", [fmt(a.strategy_max_dd, True) for a in analyses]),
        ("Benchmark Max DD", [fmt(a.benchmark_max_dd, True) for a in analyses]),
    ]
    for label, vals in rows:
        lines.append("| " + label + " | " + " | ".join(vals) + " |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pre-built benchmark suites
# ---------------------------------------------------------------------------

US_EQUITY_BENCHMARKS = BenchmarkSuite(
    primary=BenchmarkConfig(
        name="S&P 500",
        ticker="SPY",
        description="S&P 500 ETF Trust (Total Return)",
        justification=(
            "SPY represents the opportunity cost of passive US large-cap equity "
            "exposure. A systematic strategy must justify its complexity by "
            "outperforming this alternative on a risk-adjusted basis."
        ),
    ),
    secondary=[
        BenchmarkConfig(
            name="Risk-Free Rate",
            ticker="TBILL",
            description="3-Month T-Bill Rate",
            justification="Does the strategy beat holding cash?",
        ),
        BenchmarkConfig(
            name="MSCI World",
            ticker="URTH",
            description="iShares MSCI World ETF",
            justification="Global equity benchmark for strategies with international exposure.",
        ),
    ],
)
