"""Example: Run the cross-sectional momentum strategy end-to-end.

Demonstrates:
  1. Define strategy via ``StrategyDefinition``
  2. Generate synthetic data (replace with real data in production)
  3. Build universe
  4. Run backtest via ``BacktestHarness``
  5. Auto-generate Goldman Sachs-style strategy memo

Usage::

    python -m pipeline.strategy.example_momentum
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.strategy.backtest_harness import BacktestConfig, BacktestHarness
from pipeline.strategy.memo_generator import generate_memo
from pipeline.strategy.position_sizing import create_sizer
from pipeline.strategy.signal_library import SignalPipeline
from pipeline.strategy.strategy_definition import cross_sectional_momentum_strategy
from pipeline.strategy.universe import UniverseBuilder


def _generate_synthetic_universe(
    n_stocks: int = 20,
    n_days: int = 500,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Generate synthetic OHLCV data for a universe of stocks.

    In production, this would be replaced by a database query or
    API call to a market data provider.
    """
    rng = np.random.RandomState(seed)
    tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META",
        "NVDA", "TSLA", "JPM", "V", "UNH",
        "HD", "PG", "MA", "DIS", "NFLX",
        "ADBE", "CRM", "PYPL", "INTC", "CSCO",
    ][:n_stocks]

    dates = pd.bdate_range("2021-01-01", periods=n_days, freq="B")
    data: dict[str, pd.DataFrame] = {}

    for i, ticker in enumerate(tickers):
        local_rng = np.random.RandomState(seed + i * 7)
        trend = 0.0003 + 0.0002 * local_rng.randn()
        noise = 0.015 + 0.005 * abs(local_rng.randn())
        start_price = 80 + 200 * local_rng.rand()

        log_returns = trend + noise * local_rng.randn(n_days)
        close = start_price * np.exp(np.cumsum(log_returns))
        high = close * (1 + abs(noise) * 0.5 * local_rng.rand(n_days))
        low = close * (1 - abs(noise) * 0.5 * local_rng.rand(n_days))
        open_ = close * (1 + noise * 0.3 * local_rng.randn(n_days))
        volume = (2_000_000 * (1 + 0.3 * local_rng.randn(n_days))).clip(200_000)

        data[ticker] = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=dates,
        )

    # Also create SPY as benchmark
    spy_rng = np.random.RandomState(seed + 999)
    spy_close = 400 * np.exp(np.cumsum(0.0004 + 0.012 * spy_rng.randn(n_days)))
    data["SPY"] = pd.DataFrame(
        {
            "open": spy_close * (1 + 0.003 * spy_rng.randn(n_days)),
            "high": spy_close * (1 + 0.005 * abs(spy_rng.randn(n_days))),
            "low": spy_close * (1 - 0.005 * abs(spy_rng.randn(n_days))),
            "close": spy_close,
            "volume": 50_000_000 + 10_000_000 * spy_rng.randn(n_days),
        },
        index=dates,
    )

    return data


def run_example() -> str:
    """Run the full example and return the generated memo."""

    # 1. Load the strategy definition
    strategy = cross_sectional_momentum_strategy()
    print(f"Strategy: {strategy.strategy_name}")
    print(f"Classification: {strategy.thesis.classification}")

    # 2. Generate synthetic data
    print("\nGenerating synthetic universe data...")
    price_data = _generate_synthetic_universe(n_stocks=15, n_days=400)
    print(f"  Tickers: {list(price_data.keys())}")
    print(f"  Date range: {list(price_data.values())[0].index[0]} to {list(price_data.values())[0].index[-1]}")

    # 3. Build universe from price data
    # Use relaxed filters for synthetic data (no real ADV/market cap metadata)
    from pipeline.strategy.universe import UniverseFilter
    synthetic_filter = UniverseFilter(
        min_adv_dollars=0,
        min_price=5.0,
        min_market_cap=0,
        max_spread_bps=100.0,
    )
    builder = UniverseBuilder(synthetic_filter)
    universe = builder.build_from_prices(price_data)
    print(f"  Universe: {len(universe)} instruments after filtering")

    # 4. Set up the signal pipeline
    sig_def = strategy.signal_definitions[0]
    pipeline = SignalPipeline(sig_def)
    print(f"\nSignal: {sig_def.name} ({sig_def.family.value})")
    print(f"  Indicators: {sig_def.indicator_names}")

    # 5. Set up the backtest
    sizer = create_sizer(strategy.sizing_config)
    harness = BacktestHarness(
        signal_pipeline=pipeline,
        entry_rules=strategy.entry_rules,
        exit_engine=strategy.exit_engine,
        sizing_model=sizer,
        risk_constraints=strategy.risk_constraints,
        config=BacktestConfig(
            initial_capital=1e7,
            spread_bps=3.0,
            slippage_bps=2.0,
            signal_lag_days=1,
        ),
        benchmark_suite=strategy.benchmark_suite,
    )

    # 6. Run backtest
    print("\nRunning backtest...")
    benchmark_returns = {}
    if "SPY" in price_data:
        benchmark_returns["SPY"] = price_data["SPY"]["close"].pct_change().dropna()

    result = harness.run(
        price_data=price_data,
        universe=universe,
        benchmark_returns=benchmark_returns,
    )

    # 7. Print summary
    m = result.metrics
    print(f"\n{'=' * 60}")
    print(f"  Backtest Results: {strategy.strategy_name}")
    print(f"{'=' * 60}")
    print(f"  Total Return:    {m.total_return:.2%}")
    print(f"  CAGR:            {m.cagr:.2%}")
    print(f"  Sharpe Ratio:    {m.sharpe_ratio:.2f}")
    print(f"  Sortino Ratio:   {m.sortino_ratio:.2f}")
    print(f"  Max Drawdown:    {m.max_drawdown:.2%}")
    print(f"  Hit Rate:        {m.hit_rate:.2%}")
    print(f"  Profit Factor:   {m.profit_factor:.2f}")
    print(f"  Total Trades:    {m.total_trades:,d}")
    print(f"  Avg Hold (days): {m.avg_holding_days:.1f}")
    print(f"  Total Costs:     ${m.total_transaction_costs:,.2f}")
    print(f"{'=' * 60}")

    # 8. Generate memo
    print("\nGenerating strategy memo...")
    memo = generate_memo(strategy, result)
    print(f"  Memo length: {len(memo):,d} characters")
    print(f"  Memo lines:  {memo.count(chr(10)):,d}")

    return memo


if __name__ == "__main__":
    memo = run_example()
    # Write memo to file
    with open("STRATEGY_MEMO_GENERATED.md", "w") as f:
        f.write(memo)
    print("\nMemo written to STRATEGY_MEMO_GENERATED.md")
